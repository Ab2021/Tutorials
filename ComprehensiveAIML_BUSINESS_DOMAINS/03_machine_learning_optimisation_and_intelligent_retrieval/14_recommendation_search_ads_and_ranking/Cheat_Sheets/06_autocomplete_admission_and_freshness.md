---
title: "Autocomplete admission and freshness"
topic: "deciding which query completions may be surfaced, and how fast their popularity state must decay"
collection: "Part 03 — Recommendation, Search, Ads and Ranking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Autocomplete admission and freshness

Every count, rate and threshold in this sheet is a constructed teaching value marked `ILLUSTRATIVE`.
The formulas are derived from stated assumptions, and every arithmetic result printed below was
recomputed in two independent ways (a block-sum pass and an equivalent recursive pass), with the two
results quoted side by side wherever they should agree. No search vendor, retailer or platform's
measured latency, admission rate or click-through is claimed here: where a real system's behaviour
matters, the statement is a property of the algorithm being described, not a benchmark.

The three candidate series used throughout — candidates `A`, `B` and `C` — are synthetic daily event
counts chosen to expose three distinct pathologies: a decayed-but-formerly-popular string, a flat
mid-volume string, and a ramping string. They are not sampled from a query log.

## 1. File Header

**Purpose.** Decide, for one concrete autocomplete surface, which candidate strings may be shown as
completions of which prefixes, and how quickly the popularity evidence behind that decision must be
forgotten. Both halves are decided in advance, in writing, with numbers — not inferred from whatever
the ranking code happens to do at request time.

**Audience.** Search and ranking engineers who own a typeahead surface; marketplace and content
catalog owners who supply the strings; trust-and-safety reviewers who must sign off on what the
surface can be made to say; and interview candidates who need to defend a threshold rather than quote
a library default.

**Prerequisites assumed, not re-taught.** The difference between a count and a rate; the meaning of a
confidence interval; the fact that a half-life is a unit of time and not a smoothing dial; log
aggregation over event time; and the habit of attaching units to every quantity that appears in a
sentence. A reader who cannot state whether a threshold is "30 events per day" or "30 events" will
misread half of this sheet.

**Outcome after working through the sheet.** You can (a) size the number of candidate strings a
funnel actually admits and explain each gate's marginal yield, (b) compute a decayed popularity score
and its effective sample size in closed form, (c) state the precision you are entitled to claim at a
given event rate, (d) choose a half-life and prove that the choice does not silently re-rank two
candidates, (e) design an entry/exit threshold pair and predict the resulting dwell time, and (f)
diagnose four or more distinct mechanisms that produce the same symptom, "the suggestion list went
stale" or "the suggestion list is thrashing".

**Exclusions.** This sheet does not cover the language model that ranks completions, the embedding
index behind semantic typeahead, personalised completion, or the bidding rules of ads that may appear
above completions. It does not claim any organisation's numbers. It does not authorise a deployment:
the arithmetic sizes a plan and predicts a failure, and the owner of the surface still decides.

**Notation convention, used consistently below.** `H` is a half-life in days; `λ = 2^(-1/H)` is the
per-day decay factor, dimensionless, in `(0,1)`; `W` is the retention window in days (the number of
daily buckets kept); `y_t` is the event count on day `t`; `S` is the decayed score in "events"
(weighted, so not an integer); `r̂ = S / Σw` is the estimated events-per-day rate; `Neff` is the
effective number of independent days. Arithmetic is decimal and rounded at the point of display; any
intermediate value shown is unrounded unless it is labelled "rounded".

**Evidence boundary.** The gate retention rates in section 5 are constructed to be internally
consistent (they multiply to the stated admitted count) and are labelled `ILLUSTRATIVE`; they are not
measured on any real query log. What is *derived* is everything relational: given those retentions,
the admitted count, the state size, the score, the precision and the expiry date follow. A reader who
needs a production number must measure the retention of each gate on their own log, because the sheet
shows that two gates carry 72% of the total loss and the others are nearly free.

**How to use the sheet.** Sections 2 and 3 give the vocabulary and the separation of concerns. Section
4 gives the derivations with explicit validity conditions. Section 5 works one full example end to end
with an independent arithmetic re-check. Section 7 is the table to open at a design review. Section 8
is the debugging checklist, organised by *competing cause*. Section 12 is the card to keep beside a
terminal.

## 2. Concept Map / Mental Model

The mental model that prevents most typeahead mistakes is a **funnel with two clocks and one
irreversible gate**. Strings enter the funnel from a query log; a chain of gates decides which of them
may ever be surfaced; a decayed score decides the order in which the survivors appear; and a separate
expiry rule decides when a survivor stops being eligible. The score never grants eligibility on its
own, and eligibility is never re-derived from the score at request time.

```text
                        event time t_event            processing time t_proc
query log  ──►  normalise  ──►  hygiene  ──►  volume gate  ──►  user gate
                                                                   │
      ┌────────────────────────────────────────────────────────────┘
      ▼
  safety + legal gate ──► near-duplicate merge ──► seasonality gate ──► locale gate
                                                                   │
                                                                   ▼
                                                        ADMITTED SET (boolean, audited)
                                                                   │
                                                     decayed score S (continuous)
                                                                   │
                                    entry / exit thresholds + family cap + catalog liveness
                                                                   │
                                                                   ▼
                                                        SERVED COMPLETIONS
```

Three structural claims follow from that picture, and each one is a decision a team must make
explicitly rather than discover in production.

**Claim one: admission is boolean, freshness is continuous, and they fail differently.** An admission
bug shows up as a string that should never appear appearing (a slur completed from a popular prefix, a
competitor's brand, a discontinued product). A freshness bug shows up as a string that should appear
not appearing, or appearing in the wrong order. The first is a reviewable policy question answerable
by reading a list; the second is a statistics question answerable by recomputing a score. Mixing them
into one number makes both unauditable: you cannot read a float and tell whether it is 0.4 because the
string is marginal or because it is out of season.

**Claim two: the same statistic cannot serve both roles without a second threshold.** If a single
cut-off on `S` both admits and ranks, then changing the half-life changes the admitted set. Section 5
shows two candidates whose order flips at `H ≈ 14.88` days; at `H = 14` their scores differ by 4.43
events on a standard error of 19.50, so the difference is noise. A half-life that can re-order two
statistically indistinguishable candidates is harmless; a half-life that can move one of them across
an admission threshold is not, because the admitted set is what safety and legal review signed.

**Claim three: every state row has an age, and the age is not uniform.** A prefix whose traffic peaked
in week one and a prefix that is ramping today both have rows in the same table, and the same query
counts them identically. The mean age of the weight distribution for `W = 28`, `H = 14` is 10.37 days:
on average, a row's score is describing a world about ten days old. That is the honest interpretation
of a "fresh" score, and it is the number to quote when someone asks how current the surface is.

**The fan-out problem.** A completion is not attached to one key; it is attached to every prefix of its
own normalised form, and the eligibility decision is per (prefix, string) pair. A three-token string
with a four-token index limit and a two-character minimum generates roughly a dozen nodes. At the
constructed admitted count of 756,612 strings and 14 nodes per string, the serving index holds
10.59 million (prefix, string) rows — an order of magnitude more rows than candidates. This is why the
serving index stores only the top `K` ids per prefix (`K = 10` here) while the candidate table stores
the full state: the index is a cache of a decision, and the candidate table is the decision's evidence.

**The three clocks.** Event time is when the query was typed. Processing time is when the row was
appended to the log. Serving time is when a user typed the prefix. A score computed at serving time
from event-time buckets is correct; a score computed from processing-time buckets is biased by
ingestion lag, and section 5 quantifies that bias at 6.44% per day of lag. A team that never separates
these clocks cannot tell a real demand drop from a broken pipeline, because both look like "the counts
went down".

**The feedback loop.** Completions are not passive: a string that is surfaced gets typed, and being
typed raises its count. Popularity-based admission is therefore a self-reinforcing sampler. The
practical consequence is that the decayed score measures *demand given exposure*, not demand. Section
8 treats this as a distinct failure mechanism with its own discriminating check, because it produces
exactly the same symptom as a genuine hit: a runaway suggestion.

## 3. Foundational Definitions

Each definition states meaning, unit and valid range, because every downstream formula depends on the
distinction being made at the definition rather than discovered at the formula.

**Suggestion (`s`).** A normalised string that may be offered as a completion, stored without leading
or trailing whitespace, lower-cased per locale, with digits and punctuation preserved when they carry
meaning ("d3 2000 iu"). Unit: string. Valid range: non-empty, length 1–80 characters after
normalisation.

**Prefix node (`p`).** A normalised character prefix of one or more suggestions, used as the serving
key. Unit: string. Valid range: 2 characters to the full length of the longest suggestion it indexes.
A prefix node with no admitted child is a legitimate state (it returns nothing) and must not be
confused with a missing key.

**Candidate.** The pair `(s)` in the candidate table, i.e. a string with state. Unit: row. Valid range:
one row per normalised string; duplicates are removed before counting, which is the single most common
source of double-counted popularity.

**Event (`y`).** One logged occurrence of a user completing a query that normalises to `s`, within one
day bucket. Unit: events/day when bucketed. Valid range: non-negative integer per bucket; the sum over
a window is a count, the quotient by window length is a rate.

**Half-life (`H`).** The elapsed time after which a day's evidence counts half as much. Unit: days.
Valid range: `H > 0`; practically `3 ≤ H ≤ 60` for typeahead, because `H < 3` gives too few effective
days to estimate a rate and `H > 60` makes the surface unable to forget a dead fad within a quarter.

**Decay factor (`λ`).** `λ = 2^(-1/H)`, the multiplicative weight applied per elapsed day. Unit:
dimensionless. Valid range: `0 < λ < 1`; `λ = 0.951695` at `H = 14`. Note `λ ≈ 1 - ln2/H` for large
`H`, so the per-day "forgetting fraction" is about `0.693/H`.

**Score (`S`).** The weight-sum of events in the retention window, `S = Σ y_t λ^(W-t)`. Unit: events
(weighted). Valid range: `S ≥ 0`, unbounded above. It is not a count and must never be printed next to
a count without its weights.

**Estimated rate (`r̂`).** `r̂ = S / Σw`, the score divided by the sum of the weights actually used.
Unit: events/day. Valid range: `r̂ ≥ 0`. This is the quantity thresholds are expressed in, because `S`
changes meaning when `W` changes and `r̂` does not.

**Effective sample size (`Neff`).** `Neff = (Σw)² / Σw²`, the number of equally weighted days that
would give the same estimator variance. Unit: days. Valid range: `1 ≤ Neff ≤ W`; `Neff = 24.242` at
`W = 28`, `H = 14`, so a 28-day decayed window carries about 24.2 days of information.

**Precision floor (`r_min`).** The event rate at which the relative standard error of `r̂` reaches an
agreed tolerance. Unit: events/day. Valid range: depends on tolerance and on the z convention;
`r_min = (z/tol)²/Neff` gives 1.031 events/day for `z = 1`, `tol = 0.20` here, and 3.96 events/day for a
two-sided 95% claim at the same tolerance.

**Admission threshold (`τ_entry`).** The rate at which a non-admitted candidate becomes eligible. Unit:
events/day. Valid range: must exceed `τ_exit`; 30/day in the worked example.

**Exit threshold (`τ_exit`).** The rate below which an admitted candidate loses eligibility. Unit:
events/day. Valid range: strictly below `τ_entry`; 15/day in the worked example.

**Hysteresis band.** The interval `[τ_exit, τ_entry)` inside which the current state is preserved.
Unit: events/day. Valid range: width at least 4 standard errors of `r̂` at the band's own rate, or the
band degenerates into a coin flip.

**Dwell time.** The time a candidate spends between crossing one threshold and crossing the other.
Unit: days. Valid range: `≥ 0`; equals `H · log2(τ_entry/τ_exit)` for a candidate whose events stop,
which is exactly 14 days for 30 against 15 at `H = 14`.

**Stale-window bias.** The relative understatement of the current rate caused by scoring events that
arrived late or not at all. Unit: fraction. Valid range: `0 ≤ bias < 1`; 6.44% per day of lag at
`H = 14`.

**Canary.** A small, fixed set of prefixes whose suggestion lists are compared element by element
between the incumbent and candidate policy. Unit: prefix count. Valid range: 200–2,000 prefixes chosen
to cover high, mid and low traffic plus every safety category.

**Near-duplicate family.** A set of strings whose shingle-level Jaccard similarity with each other
exceeds a threshold, treated as one intent for slot-allocation purposes. Unit: set of strings. Valid
range: threshold 0.8–0.9; at 0.8 the family merges spelling variants, at 0.34 it would merge distinct
SKUs, which is the counterexample in section 4.4.

**Bot share.** The fraction of a prefix's events attributable to automated or scripted traffic. Unit:
fraction. Valid range: `0 ≤ share ≤ 1`; gate at 0.35 in the constructed funnel.

**Distinct-user ratio.** Distinct user identifiers per event for a candidate, over the window. Unit:
dimensionless, `(0,1]`. Valid range: 1.0 means every event came from a different user; values far below
the population norm indicate repetition by few users.

**Impression.** One rendering of a completion in a slot for one query session. Unit: count. Valid range:
`≥` the number of completions rendered; needed to convert clicks into a rate, and absent from pure
query-log admission, which is the blind spot section 8 names.

**Score floor.** A hard minimum `S` below which a row is pruned from the table rather than kept at a
tiny value. Unit: events (weighted). Valid range: chosen so that the row would need more than `1/λ^W`
days to recover, i.e. it is effectively dead.

**Shadow mode.** A policy run that computes decisions but does not serve them, used to compare
admitted sets before cutover. Unit: boolean per policy. Valid range: must run for at least two
half-lives, because a one-half-life shadow cannot see the decay behaviour it is meant to test.

**Family cap.** The maximum number of slots one near-duplicate family may occupy in a single
completion list. Unit: slots. Valid range: 1–3 for a 10-slot list; 2 in the worked example.

**Sequence prior.** A language-model probability of the completion given the prefix, used as a
tie-breaker or as a floor for strings with too little traffic to pass a count gate. Unit: probability
per candidate. Valid range: `(0,1]`; must not be used as an admission gate without a calibration
check, because a model can be confidently wrong about strings nobody has ever typed.

## 4. Core Formulas with Full Derivation

Each result below uses the same five-part anatomy, and each subsection ends with the condition under
which the result stops being valid. The conditions are the deliverable; the formulas are the easy part.

### 4.1 Base case

**R4.1 — Decayed popularity score and its normalised rate.**

```text
FORMULA
  S(T)  = Σ_{t=T-W+1}^{T}  y_t · λ^(T-t)          with λ = 2^(-1/H)
  Σw    = Σ_{j=0}^{W-1} λ^j = (1 - λ^W) / (1 - λ)
  r̂     = S / Σw
WHERE
  T        integer day index (event time, not ingestion time)
  y_t      events on day t, counted after de-duplication of identical normalised queries
           within one session (a session that types "run" four times contributes one event)
  W        retention window, days; rows older than T-W+1 are dropped, not merely down-weighted
  H        half-life, days; λ per-day decay factor
CONSTRAINT
  The recursion S_T = λ·S_{T-1} + y_T must reproduce the closed form exactly; if it does not, the
  window edge is being handled inconsistently (one code path drops the oldest term, another does not).
NUMERICAL EXAMPLE  (H = 14, so λ = 0.951695, W = 28)
  Σw = (1 - 0.951695^28) / (1 - 0.951695)
     = (1 - 0.250000) / 0.048305
     = 15.5264                     [weights summed, unit: days]
  oldest weight  λ^27 = 0.262689     newest weight  λ^0 = 1.000000
  candidate C: S = 682.054, so r̂ = 682.054 / 15.5264 = 43.929 events/day
INTERPRETATION
  682 "events" is not 682 queries. The window held 884 raw events; the decayed score is 77.2% of that
  because the mass sits late in the window, and the rate 43.93/day is the quantity a threshold tests.
```

Two derived quantities are worth carrying forward, because both are mistaken for diagnostics and neither
is one. First, `S / r̂ = Σw` identically, by the definition of `r̂`; a "score-to-rate ratio" therefore
carries no information about a candidate, and any dashboard plotting it is plotting a constant 15.5264.
Second, the *untruncated* fixed point is `S∞ = r/(1−λ) = 20.702 r` at `H = 14`, while the 28-day window
delivers only `Σw = 15.5264 = 0.75 × 20.702` — the remaining quarter of the geometric mass sits beyond
the window edge and is not observable. So `20.702 × 0.75 = 15.5264` is an identity, not a finding, and it
holds for every candidate regardless of trend.

The real diagnostic is a comparison across two estimators of the same rate. Compare `r̂` with the mean of
the last seven raw days:

```text
candidate   r̂       mean of days 22–28    last7 / r̂     reading
A           21.114   9.000                0.426          declining: the recent week is well below
B           21.399   21.429               1.001          stationary: the two agree to 0.1%
C           43.929   80.857               1.841          ramping: the recent week is nearly double
```

The interpretation follows from the estimator's mean age of 10.37 days: for a declining candidate the
recent week must be *below* `r̂`, for a ramping one *above*, and for a stationary one equal. The check is
falsifiable, costs one extra column, and is the quantity that belongs on the dashboard in place of the
constant. It is also why `W ≥ 2H` is a design rule rather than a preference: at `W = 28, H = 14` the
window holds exactly two half-lives, so `λ^W = 0.25` and the newest week carries 73.0% of the weight of
candidate `C`'s score — enough for the comparison above to be sensitive, and not so much that the older
three weeks are decorative.

**Validity condition.** R4.1 is valid when (i) events are bucketed by event time, (ii) each day bucket
is complete before it is scored, and (iii) the retention window is at least two half-lives. Break
condition (i) and the score measures pipeline health; break (ii) and the newest bucket is
systematically light, which biases `r̂` downward by between 4.8% and 29.3% depending on how late the
last events arrive; break (iii) and `S/Σw` is not comparable across candidates with different history.

### 4.2 General case

**R4.2 — Effective sample size, relative error and the precision floor.**

```text
FORMULA
  Neff      = (Σ w_j)² / Σ w_j²
  Var(S)    = r · Σ w_j²                  (Poisson events, independent days)
  relSE(r̂)  = sqrt(r · Σ w_j²) / (r · Σ w_j) = 1 / sqrt(Neff · r)
  r_min     = (z/tol)² / Neff            with tol the relative tolerance and z the quantile multiple
WHERE
  w_j      the weight of the j-th retained day, w_j = λ^(W-1-j)
  r        the true rate, events/day
  tol      relative tolerance, e.g. 0.20 for "within 20%"
  z        the number of standard errors required: z = 1 for a 1σ "error bar" claim, 1.645 for
           one-sided 95%, 1.96 for two-sided 95%
CONSTRAINT
  Σ w_j² is not (Σ w_j)² / W. Using W as the effective sample size understates the variance by
  (Neff/W)², which at 24.242 against 28 is a 13.4% understatement of the standard error.
  The z convention must be stated with the number: 25/Neff is the z = 1 floor, and quoting it beside a
  95% claim understates the required rate by 3.84× (96.04/25).
NUMERICAL EXAMPLE  (W = 28, H = 14)
  Σw² = 9.9442  (sum of squared weights)
  Neff = 15.5264² / 9.9442 = 241.07 / 9.9442 = 24.242 days
  Neff / W = 24.242 / 28 = 86.58% of a rectangular window's information
  at r = 43.929: relSE = 1 / sqrt(24.242 × 43.929) = 1 / 32.63 = 3.06%
  observed route for candidate C: Var = Σ y_t w² = 552.850; SE = 23.513; 23.513/682.054 = 3.45%
  ratio of observed to Poisson variance = 552.850 / 436.84 = 1.2656
NUMERICAL EXAMPLE  (tolerance 20%)
  r_min = 25 / 24.242 = 1.0313 events/day, so at 1 event/day the claim "popular" is a 20% claim
  check: S = 1.0313 × 15.5264 = 16.012; Var = 1.0313 × 9.9442 = 10.255; SE = 3.202;
         3.202 / 16.012 = 20.00%  ✓ (independent re-check by the direct route)
INTERPRETATION
  The precision floor converts a vague instruction ("only suggest things that are actually popular")
  into a testable statement: below 1.03 events/day this sheet's window cannot support a 20% rate claim,
  so a candidate at that level needs either a longer window or an external prior, not a smaller
  threshold.
```

The two percentages now differ for a reason worth naming precisely. The closed form assumes Poisson
events, so `Var(S) = r·Σw² = 43.929 × 9.9442 = 436.84`; the observed route uses the actual counts, and
for candidate `C` the observed variance is 552.850 — a ratio of 1.2656. The excess is over-dispersion,
and `C` is exactly the candidate expected to show it: a ramping series violates the constant-rate
assumption, so the weighted sum of squares of a trend is larger than the Poisson prediction. For
candidate `B`, whose series is flat, the ratio is 1.0001, and for declining `A` it is 0.7966 — below one,
because a monotone decline is *less* variable than Poisson. The practical rules are therefore: quote the
closed form when planning (it needs only `r`), quote the observed route when the data are in hand, and
treat the ratio itself as a trend test, since `> 1.1` indicates ramping and `< 0.9` declining. Mixing the
two across dashboards produces a 12.6% disagreement that looks like a bug and is not.

A related convention matters when comparing across candidate populations: `relSE` depends on `Neff` and
on `r`, so a low-volume candidate with a *long* window can out-report a high-volume candidate with a
short one. At `W = 90, H = 30` (`Neff = 67.329`), a candidate at 1 event/day has
`relSE = 1/sqrt(67.329) = 12.2%`, better than the same rate under `W = 28, H = 14` (20.3%). Any
precision claim is meaningless without the `(W, H)` pair attached, which is why the parameter version
belongs on the dashboard and in the audit row.

**Validity condition.** R4.2 assumes independent day buckets and a rate that is constant over the
window. Both are false in the interesting cases. When the rate trends, the estimator is biased toward
the *older* part of the window (because the newest day has weight 1 and the oldest 0.26, so the estimate
is not a centring of the current rate but a lagged average); the lag is quantified in section 4.3. When
days are correlated through weekday effects, `Var(S)` is understated and every z-score in this sheet
should be divided by `sqrt(1 + 2ρ)` where `ρ` is the one-day autocorrelation.

**R4.3 — Variance inflation factor for a weighted sum.** For a stationary Poisson process the weighted
sum's variance-to-mean ratio is `(1 + λ^W) / (1 + λ)`. At `W = 28`, `H = 14` that is
`1.25 / 1.951695 = 0.6405`: the weighted sum is *less* variable than a count of the same expectation,
because the weights deliberately trade sample size for recency. That number is also the answer to
"why is the decayed score not just a count divided by something" — a count with the same mean carries
1.56× the variance, and the price paid for that reduction is the recency bias quantified next.

### 4.3 Variants

**V4.3.1 — Rectangular window count.** Replace the weights with 1 and `S` becomes a 28-day count:
`Neff = W = 28`, `relSE = 1/sqrt(28 r)`, and the estimator is unbiased for a stationary rate but has a
mean age of 14.0 days instead of 10.37. The failure is ordering, not precision: candidate `A` has 734 raw
events against `B`'s 599, so a count ranks `A` above `B` while the decayed score ranks `B` above `A`
(332.256 against 327.830). Both rankings are defensible; only one can be the admission rule, and a
surface that switches between them silently will appear to "randomly" drop a suggestion. Validity: use
the count when the decision horizon is a month and the demand is stationary; do not use it when the
product promise is "what people are searching for *now*".

**V4.3.2 — Lazy (carry-forward) update.** Rather than decaying every row nightly, store `(S, t_last)`
and compute `S_now = λ^(Δ) · S_stored` on read, where `Δ = T_now − t_last` in days. Then fold new events
with `S_now + y`. The invariant is `S_after = λ^Δ S_before + y`, and it is exact for a single row
without a window edge; the window edge must still be enforced, which is why lazy schemes keep the
per-day counters for the oldest buckets or accept a bounded over-count. At 30 events per admitted
candidate per day and 756,612 candidates the event rate is 22,698,360/day (262.7/s), so the lazy form
saves 756,612 nightly writes at the cost of one exponentiation per read; at 480,000 reads/s that cost
is the dominant one, so the practical answer is a hybrid: nightly rollup for all rows, lazy catch-up
only for rows that received events since the rollup.

**V4.3.3 — Two-threshold admission with dwell.** Admit at `τ_entry`, retire at `τ_exit`, and preserve
the current state inside the band. For a candidate whose events stop, the score halves every `H` days,
so the time to fall from `τ_entry` to `τ_exit` is `H · log2(τ_entry/τ_exit)`; with 30 and 15 that is
exactly `14 · 1 = 14` days, and the reverse direction (recovery) takes the same 14 days but at a rate
that must exceed `τ_entry` — asymmetric, which is the point. Worked: a candidate at 25/day sits 4.51
days of decay above 20/day and 10.32 days above 15/day, so a retirement at 15 buys it ten days of
grace after it has visibly stopped growing.

**V4.3.4 — Wilson lower-bound gate on an engagement rate.** Where impressions are logged, gate on the
lower bound of clicks per impression rather than on a count: with `k` clicks in `n` impressions, the
95% Wilson interval is `(p̂ + z²/2n ± z·sqrt(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`. At `12/240` (5.0%) the
interval is `[2.88%, 8.54%]` — wide enough that the gate "at least 5%" is not supported by the data;
at `118/3000` (3.93%) it is `[3.30%, 4.69%]` and the same conclusion holds at a 3.5% bar. The gate must
use the lower bound, not the point estimate, or the policy admits whatever got lucky.

**V4.3.5 — Poisson upper-tail gate against a background rate.** For a candidate with expected `μ0`
events under the null (its prefix's background rate scaled by exposure), require the observed count to
exceed a quantile: `P(X ≥ k | μ0) ≤ α`. The tail is what protects against admitting a burst that is
merely the prefix's normal traffic. Values: `P(X ≥ 200 | 90) = 9.78e-11`, `P(X ≥ 30 | 12) = 8.87e-6`,
`P(X ≥ 60 | 36) = 1.58e-4`, `P(X ≥ 5 | 1.2) = 7.75e-3`. The last is the instructive one: a rule of
"at least 5 events" is not a rule at all when the background is 1.2, because 0.8% of prefixes will pass
it by chance on any given day.

**V4.3.6 — Time-to-live from the score.** An admitted candidate with score `S` and no new events expires
after `t = H · log2(S / (τ_exit · Σw))` days. With `S = 682.054` and `τ_exit · Σw = 232.896`, that is
`14 · log2(2.9286) = 21.70` days — the number to quote when someone asks how long a suggestion lingers
after the product is discontinued. The same formula run backwards gives a re-admission time, which is
why retirement and revival must be logged as separate events: a row that expires in 21.7 days and
re-enters the next day has a bug, not a revival.

**V4.3.7 — Family-capped slot allocation.** Sort by `S`, then walk the sorted list admitting at most
`cap` strings per near-duplicate family. This is not a ranking formula but it is a variant of the
decision, and it changes what the surface says: on the constructed 12-candidate list the uncapped top
ten scores sum to 177.84 and the capped list sums to 157.64, an 11.36% loss of score in exchange for
family diversity — and the capped list fills only nine slots, because the tenth would have been the
third member of an already-capped family. The decision left to the implementer is whether to serve a
tenth item from a lower-ranked different intent or to serve nine; slot ten's position-dependent click
rate, about 0.158 of slot one under a `p^-0.8` curve, prices that choice at roughly one sixth of a
top-slot click per impression.

**V4.3.8 — Near-duplicate candidate generation with banded MinHash.** The collision probability for two
strings with Jaccard similarity `s`, using `b` bands of `r` rows, is `1 − (1 − s^r)^b`. At `b = 16`,
`r = 8` (128 bits) the curve gives 1.6e-7 at `s = 0.1`, 0.0607 at `s = 0.5`, 0.6133 at `s = 0.7`,
0.9966 at `s = 0.86` and essentially 1.0 at `s = 0.95`. The same curve at `b = 8`, `r = 8` gives 0.9418
at `s = 0.86` and 0.3781 at `s = 0.7`: fewer bits, less recall, half the memory. Validity: the formula
assumes independent hash permutations and exact Jaccard on the shingle sets; a shingle set built after
lower-casing loses the case distinction that separates "iPhone" from "iphone", which is a real product
difference in some catalogs.

### 4.4 Limits and counterexamples

**L4.4.1 — The half-life that flips an order.** For candidates `A` (falling 48 → 4) and `B` (flat 21–23)
the decayed scores are 327.830 and 332.256 at `H = 14`. Sweeping `H` gives 50.09/103.55 at `H = 3`,
167.25/212.81 at 7, 327.83/332.26 at 14, 422.76/397.50 at 21 and 601.30/514.79 at 60: the ordering
crosses at `H = 14.88` days. The counterexample is not that one half-life is wrong; it is that `H` is a
*ranking parameter* whenever the series have different trends, and therefore cannot be set by "what
feels fresh". Any team that changes `H` must re-run the admitted set, because the same change that
reorders two candidates can move one of them across a threshold.

**L4.4.2 — The similarity threshold that merges distinct intents.** Shingle (4-gram) Jaccard for
"noise cancelling" against "noise canceling" is 0.7059, and for "cheap flights to lisbon" against
"cheap flights to porto" it is 0.5357. But "vitamin d3 2000 iu tablets" against "vitamin d3 1000 iu
capsules" is 0.3421 — different products, different price points — and "usb c to hdmi adapter" against
"usb c to displayport cable" is 0.1842. A threshold low enough to merge the spelling variants (0.70)
would merge the flights (0.53) and nearly merge the vitamins (0.34); a threshold at 0.80 merges only
the spelling variants. This is the boundary where the near-duplicate machinery stops being a
de-duplicator and becomes a relevance bug, and it is why `V4.3.8` is used for *slot allocation* rather
than for string deletion.

**L4.4.3 — The z-gate that cannot work at small counts.** Requiring `μ + 3·sqrt(μ)` events to admit
means a multiplier of 2.50× at `μ = 4`, 2.00× at 9, 1.75× at 16, 1.50× at 36 and 1.30× at 100. At small
backgrounds the gate is not a significance test, it is a demand-doubling requirement, and it will
reject genuinely emerging strings. The correct response at `μ < 4` is not a smaller multiplier but a
longer window (which raises `μ`) or an independent signal, because no statistical gate can distinguish
"new" from "noise" at four expected events.

**L4.4.4 — Decay as a rate estimator under non-stationarity.** Candidate `C` ramps from 2 to 99
events/day. Its decayed rate is 43.93/day while its last seven days average 80.86/day: the estimator
understates the current level by a factor of 1.84. The counterexample kills the reading "r̂ is the
current rate". `r̂` is a *weighted average of the window*, centred 10.37 days in the past; for a ramping
candidate it is necessarily below the current level, and for a decaying one necessarily above. Any
threshold compared against `r̂` is therefore a threshold on a lagged quantity, which is acceptable only
if the lag is smaller than the dwell time the policy tolerates.

**L4.4.5 — The multiple-testing arithmetic that makes a per-candidate α meaningless.** Running an
admission test at `α = 1e-3` over 4,200,000 candidates yields 4,200 expected false admissions per run;
at `α = 1e-4`, 420; at `α = 0.01`, 42,000. A per-candidate significance level is only meaningful after
saying how many candidates are tested and what false-admission budget the surface tolerates. Section 8
uses this as a discriminating check between "the gate is too loose" and "the strings are genuinely
popular".

**L4.4.6 — Where the whole framework stops applying.** If the surface has fewer than ~10^4 daily events
total across all candidates, per-candidate statistics are hopeless and the honest design is an editorial
allowlist with a decay only for ordering. Conversely, if the surface has more than ~10^7 candidates
with events, the per-row state (56 bytes here) exceeds 560 MB and the funnel must prune before scoring
rather than score and then prune. Both ends are design boundaries, not tuning problems.

## 5. Worked Numerical Example

The example follows one prefix through the full pipeline: the funnel that produces the admitted set,
then the state for three candidates inside it, then the gate decisions, then expiry. Every number below
is `ILLUSTRATIVE`.

**Step 1 — the funnel.** 4,200,000 distinct normalised strings carry at least one event in the window.
Eight gates are applied in order, each with a constructed retention rate. The products are computed
exactly and rounded once per row.

| Gate | Rule | Retention | Survivors | Share of raw |
|---|---|---|---|---|
| G0 | raw distinct strings with ≥1 event in 42 days | 1.000 | 4,200,000 | 100.00% |
| G1 | hygiene: normalisation, length 1–80, no control characters | 0.890 | 3,738,000 | 89.00% |
| G2 | volume: ≥200 events and ≥28 distinct users in 28 days | 0.640 | 2,392,320 | 56.96% |
| G3 | concentration: ≥150 distinct users in 28 days | 0.412 | 985,636 | 23.47% |
| G4 | safety + legal lexicon, truncating matches | 0.9958 | 981,496 | 23.37% |
| G5 | bot share ≤ 0.35 and distinct-user ratio ≥ 0.15 | 0.969 | 951,070 | 22.65% |
| G6 | near-duplicate merge at Jaccard ≥ 0.80 | 0.915 | 870,229 | 20.72% |
| G7 | seasonality: residual z ≥ 4 against the day-of-week baseline | 0.880 | 765,802 | 18.23% |
| G8 | locale coverage: every token has a locale-appropriate analyser | 0.988 | 756,612 | 18.01% |

The funnel teaches two operational facts. First, the loss is not spread evenly: G2 and G3 together
retain `0.640 × 0.412 = 0.2637` of what reaches them, so 73.6% of the loss between G1 and G4 happens in
two gates, while the remaining six gates multiply to `0.890 × 0.7676 = 0.6831` — of which hygiene (G1,
89.0%) and seasonality (G7, 88.0%) are almost the whole cost. An engineer optimising G4 (99.58%) is
optimising a gate that removes 0.42% of its input; the same effort spent on G2's rule removes 36%.
Second, the ordering is load-bearing: applying G6 before G3 would merge families using counts that are
about to be thrown away, and the merge would then be wrong for the survivors. Gates that *aggregate*
(G6) must come after gates that *filter on per-string evidence* (G2, G3, G5).

**Step 2 — the state for three candidates.** All three are already admitted; the question is what the
next daily refresh does with them. Retention window `W = 28` days, half-life `H = 14`, so
`λ = 0.951695` and `Σw = 15.5264`.

```text
day index        1    2    3    4    5    6    7    8    9   10   11   12   13   14
weight        .263 .276 .290 .305 .320 .336 .354 .371 .390 .410 .431 .453 .476 .500
A events        48   46   45   43   42   40   38   37   35   34   32   30   29   27
B events        21   22   20   22   21   23   20   22   21   22   20   23   21   22
C events         2    3    1    3    4    3    5    4    6    5    8    9   12   15

day index       15   16   17   18   19   20   21   22   23   24   25   26   27   28
weight        .525 .552 .580 .610 .640 .673 .707 .743 .781 .820 .862 .906 .952 1.000
A events        26   24   22   21   19   17   16   14   12   11    9    7    6    4
B events        20   22   21   23   20   22   21   22   20   23   21   22   20   22
C events        18   22   27   33   39   46   53   60   68   75   82   88   94   99
```

Raw sums are `A = 734`, `B = 599`, `C = 884`. Weighted scores, computed in four 7-day blocks so the
arithmetic can be checked by hand:

```text
block        days      A block   A cum     B block   B cum     C block   C cum
1            1–7        91.805   91.805     45.629   45.629      6.616    6.616
2            8–14       96.031  187.836     65.422  111.051     26.613   33.229
3           15–21       87.392  275.228     91.316  202.367    150.787  184.016
4           22–28       52.602  327.830    129.889  332.256    498.038  682.054

rate  r̂ = S / 15.5264     A 21.114     B 21.399     C 43.929   events/day
variance = Σ y·w²         A 167.249    B 212.812    C 552.850
SE = sqrt(variance)       A 12.932     B 14.588     C 23.513
relative SE               A 3.94%      B 4.39%      C 3.45%
```

**Step 3 — independent arithmetic re-check.** The block sums above are a direct evaluation of
`Σ y_t λ^(28−t)`. The same totals must appear from the recursive form `e ← λ·e + y`, iterated once per
day from the oldest event to the newest, which never forms a product of more than two factors:

```text
recursion e_{t} = 0.951695·e_{t−1} + y_t
  A  →  327.8303        (block-sum result 327.830, difference 0.0003 = rounding in the table)
  B  →  332.2560        (block-sum result 332.256, difference 0.0000)
  C  →  682.0535        (block-sum result 682.054, difference 0.0005)
```

A second, independent check uses the *structure* of the answer rather than the arithmetic: `B` is
approximately flat at 21.4 events/day, so if the window were long relative to `H` its score should
approach `r/(1−λ) = 21.4/0.048305 = 443.0`. It is 332.256, which is 0.75 of that, and the ratio is
explained by the window truncation factor `1 − λ^28 = 0.75`: `443.0 × 0.75 = 332.3`. The two quantities
agree to 0.01%, which confirms both `Σw` and the recursion. The same check applied to `A` fails by
design — `A` is not stationary, and that failure is the diagnostic, not an error.

**Step 4 — gate decisions.** Entry threshold `τ_entry = 30` events/day, exit `τ_exit = 15`, equivalently
scores of 465.792 and 232.896. All three candidates are currently admitted, so the question is whether
each stays.

```text
candidate  r̂       S        z vs entry   z vs exit   decision
C          43.929  682.054   +9.20        +19.10      stay (well clear)
A          21.114  327.830   −10.67       +7.34       stay, flag declining
B          21.399  332.256   −9.15        +6.81       stay (band, flat)
```

`C` is admitted: its score exceeds the entry threshold by 216.26, which is 9.20 standard errors. `A`
and `B` are both inside the hysteresis band — above exit, below entry — so neither is retired and
neither would be admitted if it were new. This is exactly what a hysteresis band is for, and the
diagnostic value appears in the next step: the two candidates in the same band have opposite futures.

**Step 5 — ranking inside the band, and why it is not a decision.** `B` (332.256) outranks `A`
(327.830) by 4.43 events. The standard error of the difference is `sqrt(167.249 + 212.812) = 19.495`, so
the gap is `z = 0.227`. Ranking them is a coin flip dressed as a number, and the sheet's recommendation
is therefore not "rank them and move on" but "keep the previous order and break ties by a stable key",
because a daily re-shuffle of two statistically identical candidates reads to users as flicker and to
analysts as a ranking regression.

**Step 6 — the futures the scores do not show.** If no further events arrive, each score halves every 14
days, so each crosses the exit threshold after `14 · log2(S / 232.896)` days: `A` in 6.91 days, `B` in
7.18 days, `C` in 21.70 days. That single formula prices the entire retirement policy, and it exposes
the asymmetry that matters operationally: `A`'s trend (48 down to 4, a per-day ratio of 0.282 when
comparing its last seven days with its first twenty-one) means it will leave the surface within a week of
its last event, while `B` will sit in the band indefinitely because it keeps receiving 20–23 events a
day. A band with no trend term therefore has *two* populations inside it, and the reviewer needs the
trend column to tell them apart.

**Step 7 — sensitivity as a decision input.** The same three series under different half-lives:

| H (days) | λ | S(A) | S(B) | S(C) | order | C/A margin |
|---|---|---|---|---|---|---|
| 3 | 0.793701 | 50.09 | 103.55 | 368.17 | C > B > A | 7.35× |
| 5 | 0.870551 | 107.34 | 161.90 | 479.87 | C > B > A | 4.47× |
| 7 | 0.905724 | 167.25 | 212.81 | 552.85 | C > B > A | 3.31× |
| 10 | 0.933033 | 246.42 | 273.68 | 623.93 | C > B > A | 2.53× |
| 14 | 0.951695 | 327.83 | 332.26 | 682.05 | C > B > A | 2.08× |
| 21 | 0.967532 | 422.76 | 397.50 | 738.97 | C > A > B | 1.75× |
| 28 | 0.975549 | 482.73 | 437.54 | 770.90 | C > A > B | 1.60× |
| 60 | 0.988514 | 601.30 | 514.79 | 827.67 | C > A > B | 1.38× |

Three things are stable and one is not. Stable: `C` is always first, and its margin over `A` shrinks
monotonically as `H` grows (4.10× down to 1.76×). Unstable: the `A`/`B` order, which crosses at
`H = 14.88`. The policy consequence is that the *admission* decision is robust to `H` while the
*ordering* decision is not, which is the argument for expressing thresholds in rates and re-deriving
them whenever `H` changes.

## 6. Visual / Geometric Intuition

**The decay ladder.** Read the weights as a staircase from the oldest day to the newest; the shape is
what "two half-lives in the window" looks like geometrically.

```text
weight
1.00 |                                                     ####   day 28
0.95 |                                                ###
0.86 |                                           ###
0.78 |                                      ###
0.71 |                                 ###                     day 21 (0.707)
0.64 |                            ###
0.58 |                       ###
0.50 |                  ###                                   day 14 (0.500)
0.45 |             ###
0.37 |         ##
0.32 |      ##
0.26 |   ##                                                   day 1  (0.263)
     +----|----|----|----|----|----|----|----|----|----|----
      1    4    7   10   13   16   19   22   25   28   day
```

The area under that staircase is `Σw = 15.5264`, not 28. That gap — 44.5% of the nominal window — is
the entire content of the phrase "effective sample size", and the two marked levels (0.500 at day 14
and 0.263 at day 1) are the checkpoints to use when verifying an implementation by eye.

**The two-clock timeline.** Event time runs along the top, ingestion along the bottom; a bar chart of
"events attributed to day t" against "events *received* during day t" separates real demand from
pipeline behaviour.

```text
event time   d1   d2   d3   d4   d5   d6   d7
counts       20   22   25   60   22   21   20      ← real demand on d4
ingested     20   22   25   22   55   26   20      ← d4's extra 38 arrive during d5
score(t)     …    …    …    −?   +?   settle       ← the score wobbles before it settles
```

The 38 events that arrive a day late are weighted `λ = 0.951695` instead of 1, so the score computed at
`d4` understates the correct `d4` score by 4.83%; if they arrive three days late the understatement is
`1 − 0.951695³ = 13.80%`, and seven days late, 29.29%. The curve is the reason a backfill must be
replayed through the event-time bucket and not added to the current day.

**The hysteresis band as a corridor.** Two thresholds, one direction of travel per state, and a
corridor between them where nothing changes.

```text
rate (events/day)
 40 |  ·  C 43.9                                    admit ──►
 30 |==================================== τ_entry ==================
    |        A 21.1      B 21.4            (band: state preserved)
 15 |==================================== τ_exit  ==================
  0 |                                            ◄── retire
       dwell for a stopped candidate: 14·log2(30/15) = 14 days
       dwell 25→20 = 4.51 days        dwell 20→15 = 5.81 days
```

**The funnel as a Sankey.** Reading the eight gates as widths rather than numbers makes the marginal
yield obvious: the two volume gates remove nearly three quarters of the input, and of the remaining
gates only three cost more than 2% — hygiene (11.0%), near-duplicate merge (8.5%) and seasonality
(12.0%). The safety and locale gates cost 0.42% and 1.2% respectively, which is why an effort
allocation that spreads work evenly across gates is misallocated.

```text
4,200,000  ████████████████████████████████████████ 100.0%
3,738,000  ██████████████████████████████████████   89.0%   hygiene
2,392,320  ███████████████████████                  57.0%   volume
  985,636  █████████                                23.5%   concentration
  981,496  █████████                                23.4%   safety/legal
  951,070  █████████                                22.6%   bot share
  870,229  ████████                                 20.7%   near-dup
  765,802  ███████                                  18.2%   seasonality
  756,612  ███████                                  18.0%   locale
```

**The slot picture with a family cap.** Twelve candidates, ten slots, cap two per family. The cap
promotes the twelfth-ranked string into slot nine and leaves slot ten empty rather than filling it with
a third variant of the same intent.

```text
rank  score  family   served?          slots: 1 a | 2 b | 3 c | 4 d | 5 f
 1 a  43.93   f1       yes                     6 g | 7 h | 8 j | 9 l | 10 (empty)
 2 b  21.40   f2       yes
 3 c  21.11   f3       yes             score served  157.64
 4 d  18.60   f1       yes             score available 177.84 (top ten uncapped)
 5 e  17.20   f1       no  (cap)       loss 20.20 = 11.36%
 6 f  15.90   f4       yes
 7 g  12.40   f2       yes             slot-10 click rate ≈ 0.158 × slot 1
 8 h  10.10   f5       yes
 9 i   9.00   f1       no  (cap)
10 j   8.20   f6       yes
11 k   7.10   f2       no  (cap)
12 l   6.00   f4       yes  ← promoted by the cap
```

## 7. Comparison / Selection Table

Choose by *what the decision has to be right about*, not by which statistic is fashionable. The
"minimum sample" column is the rate at which the mechanism can support a 20% relative claim, derived
from section 4.2 or from the interval width in section 4.3.

| Mechanism | Decides | Statistic | Minimum sample | Failure signature | Update cost | Choose when |
|---|---|---|---|---|---|---|
| Lifetime count | order only | `Σ y_t` over all history | 0.28/day at 90 days (`25/90`) | old hit outranks new hit | trivial | surface is an archive, not a discovery tool |
| 28-day rectangular count | order only | `Σ y_t` over 28 days | 0.89/day (`25/28`) | order inverts on trend (A vs B) | trivial | demand is stationary and horizon is a month |
| Decayed score `H = 14` | order | `Σ y_t λ^(28−t)` | 1.03/day | lag bias 6.4%/day of ingestion lag | one multiply per row-day | default for a discovery surface |
| Decayed score + hysteresis | order + eligibility | `S` against two thresholds | 1.03/day | flapping if band < 4 SE | plus one comparison | any surface with a safety review of the admitted set |
| Wilson lower bound | engagement eligibility | `k/n` clicks per impression | `n ≥ 600` at 4% for 20% relative | point estimate admits luck | one sqrt per row | impressions are logged and CTR gates admission |
| Poisson upper tail vs background | burst detection | `P(X ≥ k` given `μ0) ≤ α` | `μ0 ≥ 9` for a 2× gate at 3σ | 4,200 false admits at α=1e-3 | one tail evaluation | separating real emergence from prefix-level noise |
| Banded MinHash (b=16, r=8) | family membership | Jaccard on 4-gram shingles | 0.86 recall at s=0.86 | merges distinct SKUs at low thresholds | 128 perms per string | slot diversity and duplicate suppression |
| Sequence-model prior | tie-break, cold start | `P(s` given `p)` | none (needs calibration) | confident nonsense | one forward pass | cold-start strings with no events, never as the only gate |
| Editorial allowlist | eligibility | human decision | none | decays invisibly, owner leaves | zero | regulated, legal, or launch surfaces |
| Hybrid: score gate + family cap + safety | all three | composed | 1.03/day | blame ambiguity across layers | sum of parts | production surfaces with a review obligation |

Two readings of the table are worth stating explicitly. First, the two "cheap" rows (lifetime and 28-day
counts) are not wrong — they are correct for the question "what is popular over a month", and the only
reason they fail here is that the product question is "what is popular now". Second, the hybrid row's
failure signature is *blame ambiguity*: when a suggestion that should have appeared does not, four
layers can each have suppressed it, and the only defence is that each layer logs its own veto with a
reason code. A system that logs only the final list cannot be debugged.

## 8. Failure Modes & Edge Cases

Each entry names a symptom, the competing causes that produce it, the check that discriminates between
them, and the correction. "Monitor the list" is not a check; a check must return a different answer for
different causes.

**F8.1 — A runaway suggestion (one string dominates a prefix).** Symptom: a single completion holds the
top slot for a high-traffic prefix for days, and its count grows faster than the prefix's total traffic.

*Competing causes.* (a) Genuine event: a news-driven spike. (b) Automated traffic: a scraper or a
keyboard-macro loop. (c) Exposure feedback: the string is surfaced first, so it is typed first.
(d) Double counting: the same query logged once per keystroke or once per retry.

*Discriminating check.* Compute the distinct-user ratio and the per-second bucket profile for the
string and for its prefix. Genuine demand has a distinct-user ratio near the prefix norm and a
per-second profile that follows the population's diurnal curve; automation shows a flat or
machine-periodic profile and a distinct-user ratio far below the norm; exposure feedback shows the
string's share rising *after* each ranking change and falling when it is demoted (a one-week reversal
test settles it); double counting shows events per session greater than 1.0 on the affected client
version only.

*Correction.* For (b), exclude by device and inter-arrival regularity, then recompute; for (c), add a
dedicated exploration slot so the top slot is not self-confirming; for (d), de-duplicate within session
before counting, which is a one-line change that alters every historic number and must be versioned.

*Prevention.* Gate G5 in the funnel (bot share ≤ 0.35, distinct-user ratio ≥ 0.15) is a crude version of
the check; the reversal test is the part that cannot be automated away.

**F8.2 — Staleness after an ingestion outage.** Symptom: after a 36-hour pipeline gap, every
mid-volume suggestion loses rank, and the admitted count falls by roughly the share of candidates near
the threshold.

*Competing causes.* (a) Real demand drop during the gap. (b) Missing buckets treated as zeros.
(c) Backfill applied to the wrong day bucket. (d) A half-life change shipped in the same window.

*Discriminating check.* Recompute the score twice, once with the gap days dropped and once with them
imputed from the neighbouring week's day-of-week average, and compare the difference to the predicted
lag bias. Missing days reduce `Σw` as well as `S`, so a correct implementation loses less than the
naive reading suggests; if the observed fall is larger than `1 − λ^Δ` predicts for `Δ` days of missing
data (4.83% for one day, 13.80% for three, 29.29% for seven), the cause is (c) or (d), not (a).

*Correction.* Replay the backfill through event-time buckets, recompute the affected prefix rows only,
and record the replay in the audit log. Never add backfilled events to "today"; that inflates the
newest bucket by construction and cannot be undone by decay.

*Prevention.* Lag monitoring on the p99 difference between event time and processing time, alerting at
`> H/4` (3.5 days at `H = 14`), because beyond that the newest bucket's bias exceeds
`1 − 0.951695^3.5 = 15.9%` — five times the 3.45% relative standard error of a high-volume candidate.

**F8.3 — Threshold flapping.** Symptom: a suggestion appears and disappears on consecutive days near a
boundary, usually at moderate volume.

*Competing causes.* (a) Poisson noise around the threshold. (b) Genuine day-to-day oscillation (weekly
seasonality). (c) A trend crossing the threshold. (d) Two writers computing the score with different
`W` or `H`.

*Discriminating check.* Express the noise in *rate* units and compare it to the band. At 20 events/day
the relative standard error is `1/sqrt(24.242 × 20) = 4.54%`, i.e. 0.91 events/day, so the 15-unit band
is 16.5 standard errors wide — Poisson noise cannot cause flapping at that volume. At 1 event/day the
relative error is 20.3% (0.20 events/day) and the band is still 74 standard errors wide. The check
therefore rules out (a) in almost every real case and points at (b), (c) or (d): inspect the day-of-week
residual (b), the seven-day slope (c), and the score of the same row from both writer paths (d), where
a disagreement of exactly `λ^(Δ)` indicates a stale `t_last`.

*Correction.* For (b) and (c), widen the band or add a dwell requirement of `H/2` days in the new state
before the flip takes effect. For (d), make the score a single derived value with one writer.

*Prevention.* Log every state transition with its score, threshold and writer version; a transition log
is the only artefact that can distinguish flapping from legitimate churn after the fact.

**F8.4 — Family capture of the completion list.** Symptom: eight of ten slots are variants of one
intent ("red running shoes", "red running shoes men", "red running shoes men wide"), and the list's
coverage of the prefix's other intents collapses.

*Competing causes.* (a) Genuinely one dominant intent. (b) Near-duplicate merge threshold too high, so
variants were never grouped. (c) Family cap applied before ranking, so the cap admits low-score
variants early. (d) The prefix itself is over-specific, so all ten slots *should* be variants.

*Discriminating check.* Compute family entropy of the served list and compare it to the entropy of the
top-100 candidates. If the served entropy is far below the candidate entropy, the list is captured; if
they match, cause (a) or (d) holds. Then recompute with the cap applied *after* a full sort (the correct
order) and compare — a difference identifies (c) immediately.

*Correction.* Apply the cap post-sort, as in `V4.3.7`; accept the 11.36% score loss measured there as
the price of diversity, and decide explicitly whether to fill the freed slot or leave it empty.

*Prevention.* Report family entropy as a first-class metric beside the top-slot click rate, because a
captured list can have a *better* click rate than a diverse one while being worse for discovery.

**F8.5 — Safety echo.** Symptom: a prefix that is itself sensitive completes to a string that repeats
or escalates the sensitivity, and the completion is admitted because it is genuinely popular.

*Competing causes.* (a) Lexicon miss: the term is absent from the blocklist. (b) Tokenisation: the term
is broken across a prefix boundary so a per-token match fails. (c) Truncation: the term is masked in the
served text but present in the logged string. (d) Popularity override: the gate is applied before the
score, so a high score reintroduces the string.

*Discriminating check.* Run a fixed adversarial prefix suite (the same 2,000 prefixes on every release)
and diff the served lists. A miss that appears only for multi-token inputs points at (b); a miss that
appears only for strings above the top-slot threshold points at (d); a miss that appears in logs but not
in the UI points at (c).

*Correction.* Apply the safety gate *last* in the pipeline so no later stage can reintroduce a blocked
string, and make the gate a hard filter on the served text rather than on the ranked list.

*Prevention.* The suite is a regression test, not a monitor: it must fail the build, and its prefix list
must be owned by trust and safety rather than by search engineering.

**F8.6 — Locale and tokenisation mismatch.** Symptom: a locale's admitted count is far below the funnel
prediction and its top completions are visibly wrong (diacritics lost, CJK strings split mid-character).

*Competing causes.* (a) Normalisation strips marks that carry meaning. (b) The analyser segments CJK per
character, so "東京" and "東京都" land in different nodes with split counts. (c) The locale's stopword
list removes tokens that are content in that locale. (d) The funnel's gate G8 is applied per string but
the index is built per prefix, so a string passes and its nodes are never created.

*Discriminating check.* Re-tokenise a 1,000-string sample with the *serving* analyser and compare node
counts to the offline counts. A node-count ratio near the number of extra tokens per string indicates
(b); a constant small deficit across all strings indicates (a) or (c); a deficit concentrated in strings
that pass every gate indicates (d).

*Correction.* Version the analyser with the index, so that a mismatch is a build error rather than a
silent count split; rebuild the locale's rows rather than patching them.

*Prevention.* One analyser artefact, imported by both the offline scorer and the serving path, and a
canary set per locale in the release checklist.

**F8.7 — Eviction race between catalog and score.** Symptom: a completion is served for a product that
was withdrawn, and clicking it lands on an empty results page.

*Competing causes.* (a) The catalog change has not propagated to the suggestion index. (b) The
suggestion is served from a cache whose TTL exceeds the catalog's own TTL. (c) The product is regionally
unavailable but the string is globally admitted. (d) The score expired the row but the serving index
still holds the id.

*Discriminating check.* Join the served list against the catalog snapshot by version, not by timestamp:
count ids present in the served list whose catalog version is older than the current snapshot. A
non-zero count localised to one cache tier identifies (b); a count spread across tiers identifies (a);
a count confined to one region identifies (c).

*Correction.* Make catalog liveness a filter at render time, not a step in the nightly job, and let the
id's absence be a cache invalidation event rather than a TTL expiry.

*Prevention.* Empty-result rate per completion, measured on click-through, is the cheapest detector: a
suggestion that is served but never clicked, or clicked and immediately abandoned, is usually dead
catalog.

**F8.8 — The policy change that silently re-ranks.** Symptom: after a half-life change from 21 to 14
days, the admitted count is unchanged but the top slots of mid-volume prefixes reshuffle, and a
stakeholder reports "the search got worse".

*Competing causes.* (a) The half-life change reordered candidates whose trends differ (the crossover at
`H = 14.88` in `L4.4.1`). (b) The change also altered which candidates clear the entry threshold,
because the threshold is expressed in `S` rather than in `r̂`. (c) A co-deployed change to the family cap
took effect at the same time. (d) Cache warm-up produced a one-day artefact.

*Discriminating check.* Recompute the admitted set and the ranking under the old and new parameters
from the same frozen event table. If the admitted set differs, the threshold is in score units and
cause (b) applies; if only the order changes, (a); if only the list *length* changes, (c).

*Correction.* Express thresholds in `r̂` before changing `H`, and re-derive them: at `H = 14` the entry
threshold of 30/day is a score of 465.792, while at `H = 21` the same rate is a score of
`30 × 18.577 = 557.3`. Keeping the *score* fixed at 465.792 across that change silently lowers the rate
threshold from 30/day to `465.792/18.577 = 25.07/day` — a 16.4% loosening that admits candidates the
policy was written to exclude.

*Prevention.* Parameter changes ship with a shadow-mode comparison over at least two half-lives, and
the diff is reviewed as a *list* diff, not as a metric diff.

**F8.9 — The cold-start string that no gate can see.** Symptom: a newly launched product with a
predictable name gets no completion for days, and the team adds a manual override that is never removed.

*Competing causes.* (a) Volume gate: genuinely below 200 events. (b) Concentration gate: a name typed
by many users once each still fails a 150-distinct-user rule if exposure is low. (c) Exposure: nobody
types it because it is never suggested, so the count never starts. (d) The string is filtered by a
lexicon rule intended for a different product family.

*Discriminating check.* Compute the string's count conditional on being *shown*; if it is shown rarely
or never, the count is measuring exposure rather than demand, and cause (c) applies. If it is shown and
not typed, (a) applies and no override is warranted.

*Correction.* For (c), a sequence-model prior or an editorial seed with an explicit expiry date; both
must carry an owner and a removal date in the row itself.

*Prevention.* Every manual admission carries an expiry field that the nightly job enforces, so a
cold-start override cannot become permanent infrastructure by neglect.

## 9. Complexity & Cost Analysis

**State size.** One row per admitted candidate at 56 bytes: 8 B `f64` score, 4 B last-update epoch-day,
4 B family id, 4 B flags, 8 B catalog backpointer, 28 B average normalised key. At 756,612 admitted
candidates that is `756,612 × 56 = 42,370,272` bytes = 42.37 MB = 40.41 MiB. The same population with
28 explicit daily counters is `756,612 × (28×4 + 40) = 756,612 × 152 = 115,005,024` bytes = 109.68 MiB:
2.71× the memory for the privilege of exactly reproducing a score the recursion already gives. The serving index
is a separate, smaller-per-row structure: 1,180,000 prefix nodes at `10 × 4 B` ids plus 12 B of header =
52 B each, i.e. 61.36 MB, holding only the top-ten decision and not the evidence.

**Update cost.** At 30 events per admitted candidate per day the event rate is
`756,612 × 30 = 22,698,360` events/day = 262.7 events/s. Lazy decay makes each event `O(1)` but adds one
exponentiation per *read*; with 480,000 state reads/s (40,000 queries/s × 12 candidates), the
exponentiation cost dominates. Eager nightly rollup makes each row `O(1)` per day (756,612 writes) and
each read a plain comparison. The break-even is
`reads_per_day / writes_per_day = 4.147e10 / 7.566e5 = 5.48e4`: eager wins by about 4.7 orders of
magnitude on this read mix, which is why lazy decay belongs in the pipeline (where it deduplicates and
handles late events) and not in the request path. State the general cost as `cost = E·c_eval + N·c_row`
where `c_eval` and `c_row` are measured on the actual stack; the honest planning range for `c_eval` is
0.2–2 µs, which puts the nightly event pass at 4.5–45 s of single-core time — measurable, not estimated.

**Near-duplicate cost.** 128 MinHash permutations over 35 shingles per string, at 2,100,000 strings, is
`2.1e6 × 128 × 8 B = 2,150,400,000 B = 2,050.78 MiB` for the signatures alone, plus
`2.1e6 × 35 × 4 B = 294,000,000 B = 280.38 MiB` for the shingle ids if they are materialised. The
banding cost is what bites: with `b = 16` bands of `r = 8` rows the average bucket holds
`2.1e6 / 2^8 = 8,203` strings, so one query string examines `16 × 8,203 = 131,248` candidates, and a
full pass examines about `2.1e6 × 131,248 / 2 = 1.38e11` pairs. That number is why banding parameters are
chosen from the *bucket* size and not only from the recall curve: raising `r` to 12 with `b = 16` cuts
the bucket from 8,203 to 513 and the pass from 1.38e11 to 8.61e9 pairs, at the cost of recall at
`s = 0.86` falling from 0.9966 to 0.9427. The intermediate `r = 9` is the useful setting: bucket 4,102,
pass 6.89e10, recall 0.9914 — 0.5% less recall than `(16, 8)` for half the work.

**Precision versus memory.** The same tolerance costs different amounts of window depending on the
half-life, because `Neff` saturates:

| W (days) | H (days) | Σw | Neff | r_min for 20% | S at r_min | state (MB at 56 B/row) |
|---|---|---|---|---|---|---|
| 28 | 7 | 9.944 | 17.836 | 1.402 | 13.94 | 42.37 |
| 28 | 14 | 15.526 | 24.242 | 1.031 | 16.01 | 42.37 |
| 28 | 21 | 18.577 | 26.166 | 0.955 | 17.75 | 42.37 |
| 28 | 30 | 20.856 | 27.064 | 0.924 | 19.27 | 42.37 |
| 90 | 30 | 38.310 | 67.329 | 0.371 | 14.22 | 42.37 |
| 90 | 60 | 56.281 | 82.684 | 0.302 | 17.02 | 42.37 |

The table separates two levers that are usually confused. Widening `W` at fixed `H` buys precision up to
the `Neff` ceiling of about 40 days at `H = 14`; lengthening `H` at fixed `W` buys very little, because
`Neff` saturates at `(1+λ)/(1−λ)` — 27.064 at `H = 30` with `W = 28`. Memory does not move at all in
this table, because the row is fixed-size; it is the per-day counters (`W` floats per row) that scale
with `W`, at `756,612 × 4 B = 3.03 MB` per extra day for the whole population.

**Serving cost.** 40,000 queries/s × 12 candidates = 480,000 state lookups/s, and 12 bytes of score and
flags per query = 0.48 MB/s of state bandwidth — negligible, which is why the state can live in-process
with a short-TTL cache. At a p99 of 12 ms, the concurrency implied by Little's law is
`40,000 × 0.012 = 480` in-flight requests: the number to size the cache and the connection pool
against, not the QPS.

**Sampling cost for evaluation.** Comparing two admission policies on clicks at a 4% baseline, a 0.5
percentage-point lift needs `2 · (1.96 + 0.84)² · 0.04 · 0.96 / 0.005² = 24,084` impressions per arm.
At 3,000 impressions/day per prefix that is eight days per prefix-arm, which is why policy comparison is
done on a matched canary set of prefixes rather than per prefix.

## 10. Key Theorems / Results Reference

**T10.1 — Recursive/closed-form equivalence.** For `λ ∈ (0,1)` and any finite series `y_0…y_{W−1}`,
`Σ_{j=0}^{W−1} y_j λ^{W−1−j}` equals the recursion `e ← λe + y` iterated over the same series starting
from `e = 0`. Validity: exact in exact arithmetic; in `f64` the difference is below `1e-12` relative for
`W ≤ 10^4`. Use: any disagreement between an offline SQL job and an online service means the two are
using different `W`, not different rounding.

**T10.2 — Geometric sum.** `Σ_{j=0}^{W−1} λ^j = (1−λ^W)/(1−λ)`, and as `W → ∞` it tends to `1/(1−λ)`.
Validity: `λ ≠ 1`. Use: `Σw` is the denominator of every rate, so an off-by-one in the exponent count
scales every threshold by `1/(1−λ) = 20.7` at `H = 14` — a bug that is invisible until a threshold is
compared across two code paths.

**T10.3 — Effective sample size.** `Neff = (Σw)²/Σw²`, which for geometric weights on `W` days equals
`(1−λ^W)²(1−λ²) / ((1−λ)²(1−λ^{2W}))`, tending to `(1+λ)/(1−λ)` as `W → ∞`. Values: 24.242 at
`W = 28, H = 14`; 40.404 as the limit. Validity: independent observations with equal variance. Use: it
is the number that converts a rate into a precision claim, and the ceiling that tells you when widening
the window has stopped helping.

**T10.4 — Poisson upper tail.** For `X ~ Poisson(μ)`, `P(X ≥ k | μ) = 1 − Σ_{i<k} e^{−μ} μ^i / i!`.
Values used above: 9.78e-11 at `(200, 90)`, 8.87e-6 at `(30, 12)`, 1.58e-4 at `(60, 36)`, 7.75e-3 at
`(5, 1.2)`. Validity: events independent and non-clustered; with clustering the tail is heavier and the
gate is looser than it looks.

**T10.5 — Wilson score interval.** `(p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)) / (1 + z²/n)`; it is the
inversion of the score test and stays inside `[0,1]` at small `n`, unlike the Wald interval. Values:
`[2.88%, 8.54%]` at 12/240, `[3.30%, 4.69%]` at 118/3000, `[3.63%, 5.09%]` at 129/3000, `[0, 11.35%]` at
0/30. Validity: binomial sampling with independent impressions; it does not correct for position bias,
which is a separate inversion problem.

**T10.6 — LSH collision probability.** For `b` bands of `r` rows, `P(collide) = 1 − (1 − s^r)^b` where
`s` is the Jaccard similarity. Values for `b=16, r=8`: 1.6e-7 at `s=0.1`, 0.0607 at 0.5, 0.6133 at 0.7,
0.9966 at 0.86. Validity: independent permutations, exact Jaccard, and a similarity threshold placed
where the curve is steep; a threshold in the flat region (below 0.4) makes the band configuration
irrelevant and the cost unbounded.

**T10.7 — Weighted-sum dispersion.** For a stationary Poisson source, `Var(S)/E(S) = (1+λ^W)/(1+λ)`,
which is 0.6405 at `W = 28, H = 14`. Validity: Poisson and stationary — both, not one. Use: it explains
why a decayed score of a stationary source is *less* noisy than a count of equal mean, and it is the
factor to convert a count threshold into a score threshold during a migration. It is *not* a
wholesale check on a real series: the observed ratios for the three candidates here are 1.266 (ramping
`C`), 1.000 (flat `B`) and 0.797 (declining `A`), so applying the 0.6405 factor to a trending candidate
understates its variance by up to 2×.

**T10.8 — Half-life ordering crossover.** For two series with different trends, the ordering of their
decayed scores is a function of `H` and can cross. The crossing for `A` and `B` above is at
`H = 14.88` days, found by bisection on `S_A(H) − S_B(H)`. Validity: any pair whose recent-to-old ratio
differs. Use: it is the proof that `H` is a ranking parameter, and the reason `H` changes must be
evaluated as a list diff.

**T10.9 — Little's law.** `L = Λ·W` relates in-flight requests `L`, arrival rate `Λ` and time in system
`W`. At 40,000 queries/s and 12 ms p99, `L = 480`. Validity: a stable system over the averaging window.
Use: sizing caches, pools and per-request state fetches, where a naive read of "40k QPS, 12 candidates"
suggests 480k concurrent lookups and the correct figure is 480 concurrent requests each doing 12.

## 11. Interview / Exam Questions

**Q1. A candidate's 28-day series is `[10, 12, 11, 9, 10, 12, 11]` with `H = 14` and `W = 28`. What is
its decayed rate, and what would you report as its uncertainty?**

*Answer.* With only the seven most recent days present, the weights are `λ^6 … λ^0` =
`0.743, 0.781, 0.820, 0.862, 0.906, 0.952, 1.000`, so `Σw = 6.0634` (not 15.5264) and `Σw² = 5.3036`.
The score is `S = 10(0.743) + 12(0.781) + 11(0.820) + 9(0.862) + 10(0.906) + 12(0.952) + 11(1.000)` =
`7.43 + 9.37 + 9.02 + 7.76 + 9.06 + 11.42 + 11.00 = 65.06`, so `r̂ = 65.06/6.0634 = 10.73` events/day.

Uncertainty: `Neff = 6.0634²/5.3035 = 6.932`. The variance of the score is `Σ y w² =
5.520 + 7.314 + 7.402 + 6.687 + 8.203 + 10.869 + 11.000 = 56.995`, so `SE(S) = sqrt(56.995) = 7.550` and
`relSE = 7.550/65.057 = 11.6%` — equivalently `1/sqrt(6.932 × 10.729) = 11.6%`, which is the independent
re-check.

*Why it is asked.* It tests whether the candidate divides by the weights actually present rather than by
`Σw` for a full 28-day window. The bug's direction depends on where it is made: dividing `S` by the
full-window 15.5264 reports 4.19 events/day against a true 10.73 — a 61% understatement that would retire
a healthy candidate — while failing to truncate the *numerator* (adding zeros for absent days) is
harmless and failing to truncate the denominator is not. The ratio `15.5264/6.0634 = 2.56` is the size of
the error.

**Q2. Two candidates score 327.830 and 332.256 with standard errors 12.932 and 14.588. May you rank
them?**

*Answer.* No. The difference is 4.426 and its standard error is `sqrt(12.932² + 14.588²) = 19.495`, so
`z = 0.227`. Keep the previous order and break ties deterministically. Reporting the ranking as a finding
is a claim the data cannot support.

*Why it is asked.* It separates people who compute a score from people who know what a score entitles
them to say. The follow-up is always "what would change your answer" — a longer window, lower
correlation between the two series, or a decision that is insensitive to the order.

**Q3. A pipeline delivers one day of events a day late. How wrong is the score, and does it matter?**

*Answer.* Wrong by `1 − λ^Δ` for `Δ` days of lag: 4.83% at one day, 13.80% at three, 29.29% at seven.
At `H = 14` a one-day lag is a 4.83% understatement of the newest bucket, which is inside the 3.45%
relative standard error of a high-volume candidate and outside it for a candidate near `r_min = 1.03/day`.
It matters exactly when the admission decision is near a threshold.

*Why it is asked.* The answer requires the decay identity, not an opinion, and it forces the candidate to
state when a bias is acceptable — which is the actual engineering judgement.

**Q4. You must choose between a 28-day count and a 14-day half-life for a "trending now" surface. What
decides it?**

*Answer.* The question the surface answers. A count has mean age 14.0 days and `Neff = 28`; the decayed
score has mean age 10.37 days and `Neff = 24.242`. For a stationary surface the count is strictly better
(more effective sample, no trend bias). For "now", the count ranks `A` (734 raw) above `B` (599 raw)
whereas the decayed score ranks `B` above `A` — and `A` is a candidate that has fallen from 48 to 4
events/day, so the count is ranking a corpse.

*Why it is asked.* It tests whether "recency weighting" is understood as a bias-variance trade rather
than as a quality improvement.

**Q5. Design the exit threshold for an entry threshold of 30 events/day at `H = 14`, and state the dwell
time you are buying.**

*Answer.* Any exit below 30 creates hysteresis; 15 is the natural choice because it halves the rate and
therefore buys exactly `14 · log2(30/15) = 14` days of dwell for a candidate whose events stop. The band
must be at least four standard errors wide in rate units at its own level: at 20/day the relative error
is 4.54%, i.e. 0.91/day, so the 15-unit band is 16.5 SE — comfortably wide. Reporting dwell without the
noise comparison is incomplete.

*Why it is asked.* It is a design question with an arithmetic answer, and the arithmetic exposes whether
the candidate understands that the band width and the half-life jointly set the policy's memory.

**Q6. Why is a per-candidate significance level of 1e-3 not a policy?**

*Answer.* Because 4,200,000 candidates are tested and `4.2e6 × 1e-3 = 4,200` false admissions are
expected per run. Either the level is set against the false-admission budget (`α ≤ 2.4e-7` for one
expected false admission) or the policy must be evaluated on the admitted set rather than on each
candidate independently.

*Why it is asked.* It is the multiple-testing question in its operational form, and the numeric answer is
memorable, which is the point.

**Q7. A prefix's completions all look like variants of one phrase. What do you check first, and what do
you change?**

*Answer.* First check family entropy of the served list against the entropy of the top-100 candidates, to
distinguish "one dominant intent" from "captured list". If captured, check whether the cap is applied
before or after the sort; before-sort capping is the usual bug. Then apply the cap post-sort and accept
the measured score loss — 11.36% on the constructed twelve-candidate list — as the price of coverage.

*Why it is asked.* It tests diagnosis before intervention. Jumping to "lower the cap" without the entropy
comparison can make a legitimately focused prefix worse.

**Q8. You have 12 clicks on 240 impressions for a new completion. Can you admit it at a 5% click rate?**

*Answer.* No. The 95% Wilson interval for 12/240 is `[2.88%, 8.54%]`; the lower bound is well below 5%,
so the data are consistent with a true rate of 2.9%. Either wait for more impressions or admit on a
different signal (the popularity gate) and let the click gate act as a demotion rule rather than an
admission rule.

*Why it is asked.* It forces the distinction between a point estimate and a decision, and the interval is
small enough to compute by hand once the formula is known.

**Q9. A string has 1 event per day. What precision can you claim, and what would you do instead?**

*Answer.* `relSE = 1/sqrt(24.242 × 1) = 20.3%`, exactly the tolerance at which `r_min = 1.03/day` is
defined. You can claim "about one event a day, give or take a fifth" — which supports ordering but not an
admission decision. The options are a longer window (`W = 90, H = 30` gives `Neff = 67.3` and
`r_min = 0.371`), a sequence-model prior, or an editorial decision with an expiry date.

*Why it is asked.* It is the precision floor in disguise, and it exposes whether the candidate treats
"low volume" as a reason for a lower threshold (wrong) or a reason for a different estimator (right).

**Q10. Choose banding parameters for 2.1 million strings when you need 0.99 recall at Jaccard 0.86 and
cannot spend more than one second per full pass.**

*Answer.* The recall requirement picks the row count: `r = 8` with `b = 16` gives
`1 − (1 − 0.86^8)^16 = 0.99662`, and `r = 9` with `b = 16` gives 0.99143 — both clear the 0.99 bar, and
`r = 12` falls to 0.94272, so 12 rows is already too many. Recall alone therefore allows `(16, 8)` at 128
bits (2.05 GiB of signatures) or `(16, 9)` at 144 bits.

Cost then decides, and it decides against both. With `r = 8` each band has `2^8 = 256` buckets, so the
average bucket holds `2.1e6/256 = 8,203` strings and one string is compared against
`16 × 8,203 = 131,248` others; a full pass is about `1.38e11` comparisons. With `r = 9` the bucket halves
to 4,102 and the pass to 6.89e10. Neither is a one-second job at any plausible comparison rate: at
`3e5` comparisons/s the `r = 9` pass takes `2.3e5` s. The requirement is unsatisfiable at full pairwise
scale, and the honest answers in order of preference are (i) only fingerprint new and changed strings and
compare them against the existing index, which reduces the pass from `N²` to `NΔ`; (ii) shard by first
token so families can only form within a shard; (iii) accept `(16, 9)` rather than `(16, 8)` and spend the
saved 2× on verification depth.

The false-positive side is the number to quote next: at `s = 0.2` the collision probability is `4.10e-5`
for `(16, 8)`, which over `2.2e12` pairs of a 2.1-million-string population is about 9e7 candidate pairs
that a verifier must then reject by exact Jaccard. A candidate who quotes only the 0.99662 has answered the
recall half of a two-half question.

*Why it is asked.* It contains a trap: the recall formula is easy and the cost formula is the actual
engineering. A candidate who answers only with 0.9966 has answered half the question.

**Q11. A product is discontinued today. Its suggestion has a score of 682.054. When does it stop being
served, and what must you log for that to be auditable?**

*Answer.* It falls below the exit threshold of `15 × 15.5264 = 232.896` after
`14 · log2(682.054/232.896) = 21.70` days, assuming no further events. For auditability you need the
transition logged with the score, both thresholds, the parameter version and the writer version — a
transition without the parameter version cannot be re-derived after the next `H` change.

*Why it is asked.* It is a one-line calculation with a policy consequence, and the auditability half
tests operational maturity.

**Q12. Why can threshold flapping almost never be blamed on Poisson noise?**

*Answer.* Because the noise, expressed in rate units, is `1/sqrt(Neff·r)`: 4.54% at 20 events/day
(0.91/day) against a band of 15 units, and 20.3% at 1 event/day (0.20/day) against the same band. The
band is 16.5 and 74 standard errors wide respectively, so a Poisson process cannot cross it repeatedly.
Flapping therefore indicates weekly seasonality, a trend crossing the threshold, or two writers
disagreeing about the parameters.

*Why it is asked.* It inverts the intuitive answer, and the arithmetic that does the inverting is short
enough to be reproduced at a whiteboard.

**Q13. The admitted count is unchanged after a half-life change, but stakeholders say quality dropped.
What happened?**

*Answer.* The ordering changed while the eligibility did not — precisely the situation `L4.4.1` predicts,
with a crossover at `H = 14.88` for two candidates whose trends differ. If the thresholds were expressed
in score units, the admitted set would also have changed, because the same rate threshold maps to
`30 × 15.526 = 465.8` at `H = 14` and `30 × 18.577 = 557.3` at `H = 21`. The diagnostic is to recompute
both the admitted set and the order from a frozen event table.

*Why it is asked.* It tests the separation of concerns from section 2 in a realistic incident shape, and
the two numeric mappings are the evidence that the candidate actually checked.

**Q14. A scraper doubles one prefix's counts for a week. Which gates catch it, and which do not?**

*Answer.* G5 (bot share, distinct-user ratio) catches it if the scraper is identifiable; the volume gate
G2 does not, because it only sees volume; the concentration gate G3 partially catches it, because one
scraper contributes few distinct users. The discriminating check is the per-second profile and the
session-level events-per-session count. Note the ordering consequence: the inflated score can promote
the string into the top slot, where exposure feedback then makes the inflation partly real — which is
why the check must run before the ranking change is accepted.

*Why it is asked.* It requires the candidate to reason about which gate sees which signal, and to notice
the feedback loop rather than treating the inflation as a one-time data-quality fix.

**Q15. What is the smallest change that makes an autocomplete surface auditable end to end?**

*Answer.* Log, for every served completion, the score, the parameter version, and one veto or admit
reason code per layer. Everything else — the decay arithmetic, the family cap, the safety filter — can be
re-derived from those four fields plus the event table; without them, a list diff cannot be attributed to
a layer, and every investigation becomes an argument.

*Why it is asked.* It is a judgement question with a defensible single answer, and it reveals whether the
candidate has operated a system or only built one.

## 12. Quick Reference Card

| Item | Value / rule | Check |
|---|---|---|
| Decay factor | `λ = 2^(-1/H)`; 0.951695 at `H = 14` | `λ^14 = 0.5`, `λ^28 = 0.25` |
| Weight sum | `Σw = (1−λ^W)/(1−λ)` = 15.5264 at `W=28, H=14` | equals the recursion's fixed-point denominator |
| Rate | `r̂ = S/Σw`; thresholds live in `r̂`, never in `S` | 30/day ↔ 465.792 at `H=14`, 557.3 at `H=21` |
| Effective sample | `Neff = (Σw)²/Σw²` = 24.242; ceiling `(1+λ)/(1−λ)` = 40.404 | `Neff ≤ W`, and 24.242/28 = 86.58% |
| Precision | `relSE = 1/sqrt(Neff·r)`; `r_min = 25/Neff` = 1.031/day at 20% | direct variance route recomputes 20.00% |
| Gate pair | entry 30, exit 15 → dwell `14·log2(2)` = 14 days | band ≥ 4 SE in rate units |
| Expiry | `t = H·log2(S/(τ_exit·Σw))` = 21.70 days from 682.054 | `A` 6.91 d, `B` 7.18 d from their scores |
| Lag bias | `1 − λ^Δ`: 4.83% at 1 day, 13.80% at 3, 29.29% at 7 | compare to relSE before blaming demand |
| Near-duplicate | `P = 1 − (1−s^r)^b`; `b=16, r=8` → 0.9966 at `s=0.86` | shingle Jaccard: 0.71 for spelling, 0.34 for distinct SKUs |
| Cost anchors | 56 B/row → 42.37 MB at 756,612 rows; 22.70M events/day = 262.7/s | 480 concurrent at 40k QPS × 12 ms |
