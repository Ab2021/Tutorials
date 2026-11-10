---
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
file_id: "01"
topic: "Anomaly detection triage"
level: 2
status: "draft"
version: "3.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in event schema, prevalence, investigator capacity, label maturity, or score model version."
notation_style: "plain-text"
prerequisites: ["Probability, distances, validation contracts"]
pattern_tags: ["anomaly-scoring", "capacity-triage", "delayed-labels", "false-positive-economics"]
source_status: "OPEN"
owner_role: "Risk analytics reviewer"
source_index: "INDEX.md"
---

# Anomaly Detection Triage

## 1. File Header

> **Purpose:** convert an unsupervised anomaly score into a capacity-bounded investigation policy, and decide whether an apparent anomaly is a behaviour change or a data/operational failure.
> **Prerequisites:** point probability and densities, distance and neighbourhood reasoning, feature scaling, and the time-safe validation contract taught in Module 11. You should be able to compute a conditional probability and read a cumulative distribution function.
> **Key outcome:** given a score distribution over eligible events, a review capacity `C`, a service time per review, a review cost, and delayed adjudications, you can (a) choose a threshold or ranking rule that fits capacity, (b) report yield with an honest denominator, (c) compute the queueing consequence of your triage policy, (d) compute the break-even detection probability that justifies a review, and (e) reject an anomaly claim that is actually a schema, scaling, or join failure.
> **This file does NOT cover:** proof of fraud or fault, causal attribution of an anomaly to a specific actor, supervised classifier design when labels are abundant, or any measured production deployment. The worked numbers are constructed teaching values.
> **Level:** 2 Core Methods (with production diagnostics in Sections 8–9).
> **Version / reviewed:** 3.0 / not independently reviewed; review is OPEN.
> **Evidence:** `DERIVED` ranking, precision, queueing and information identities; `VERIFIED` hand arithmetic in Sections 4–5; `ILLUSTRATIVE` event counts, costs and reviewer workloads; `SOURCED` bibliographic mechanism references only; `OPEN` any claim about a real deployment.
> **Retrieval cue:** if alerts rise abruptly, or a reviewer says "this cluster is obviously fraud", start at Section 8 and work back to Section 4.

The visible inventory lists an anomaly-detection sheet under the fraud cheat-sheet directory at [`directory.txt`](../../../../Templates/directory.txt). That filename supplies scope only; it does not expose source content, establish a production result, or certify any number used below.

## 2. Concept Map / Mental Model

### 2.1 The four layers of a triage system

An anomaly detection system is not one object. It is four layers with different failure modes, and most operational confusion comes from arguing about the wrong layer.

```text
LAYER 1  EVIDENCE          LAYER 2  SCORING           LAYER 3  POLICY            LAYER 4  LEARNING
------------------------   ------------------------   ------------------------   ------------------------
event at t0                reference population       eligibility filter         adjudication outcome
point-in-time features  -> fit (window, version)   -> rank by score         -> mature label
join correctness           score a(x)                 cut at capacity C          review-bias record
schema contract            monotone transform         dedupe to entities         drift / refit decision
     |                          |                          |                          |
     v                          v                          v                          v
 missing / shifted?  ---->  score distribution   ---->  queue + SLA      ---->  next policy, next threshold
```

The arrows are dataflow, not causation. Layer 3 cannot repair a Layer 1 defect: cutting capacity does not make a null-rate spike into fraud. Layer 4 cannot validate Layer 2 if the review set is selected by Layer 2's own scores, which is the identification problem developed in Section 5.

### 2.2 What "unsupervised" does and does not buy

Unsupervised methods estimate unusualness relative to a declared reference population. They produce an ordering. The ordering is useful even when no label exists, because a human reviewer can consume a ranked queue. But an ordering carries no absolute meaning:

- The score scale is model-specific and version-specific; `0.71` from one fit is not comparable to `0.71` from another.
- The score is not a probability. Converting it into "risk" requires either calibration against mature labels or an explicit decision-theoretic argument (Section 4.2, formula ANO-14).
- The ranking is invariant under any strictly increasing transform, so cosmetic rescaling changes nothing about who is reviewed, while it changes everything about a fixed numeric threshold copied from a dashboard.

### 2.3 The alert lifecycle

Triage is a state machine over an alert, and each transition has an owner and a cost.

```text
   [scored] --eligibility pass--> [candidate]
       |                              |
   ineligible                     dedupe(entity, window)
       |                              |
       v                              v
   [suppressed]                  [queued] --capacity--> [assigned] --service time s--> [adjudicated]
   (logged, not deleted)              |                        |                              |
                                  age > SLA                 abandon/abstain            mature after delay d
                                      |                        |                              |
                                      v                        v                              v
                                  [escalated]             [deferred]                 [reported in cohort]
```

Two properties matter operationally. First, suppression must be logged with a reason, otherwise the next investigation cannot reconstruct why an alert never reached a human. Second, "abstain" is a legitimate terminal state: a reviewer may conclude that the available evidence cannot decide, which is not the same as "not an anomaly".

### 2.4 The central invariant of triage

The invariant that organises this entire sheet is:

```text
investigations_started_per_period <= capacity_per_period
AND reported_yield_denominators_are_mature
AND every_alert_reduction_has_a_recorded_reason
```

Everything else — threshold choice, method choice, score normalisation — is subordinate to those three clauses. A detector that violates the first clause generates an unbounded queue. A report that violates the second makes the model look better or worse than it is. A pipeline that violates the third destroys the audit trail needed to diagnose Section 8 failures.

### 2.5 Where the decision actually lives

```text
business objective (loss avoided, cases found, time-to-detect)
        |
        v
decision rule (who is reviewed, in what order, with what SLA)
        |
        +--> cost inputs: review cost, error costs, delay cost
        +--> capacity inputs: reviewers, service time, shifts
        |
        v
threshold / top-k / tiering  <---- score ranking  <---- reference population & features
                                              |
                                              +---- data quality gates (fail closed)
```

The lower half of this map is engineering. The upper half is where the decision is owned. A common organisational failure is to let the modelling team own the threshold, because the threshold is a capacity statement and capacity belongs to the operations owner.

**Takeaway:** four layers, one invariant — if investigations exceed capacity or denominators are immature, no amount of scoring sophistication rescues the control.

## 3. Foundational Definitions

Every symbol below is defined on first use and reused consistently in Sections 4–11. Where a term has an established meaning in the outlier-detection literature and a different operational meaning in a review workflow, both are given and the distinction is marked.

| Term / symbol | Formal meaning | Units / shape | Valid domain | Common confusion | Operational check |
|---|---|---|---|---|---|
| event | one immutable observational record with an event-time `t0` | row | one row per business event after dedupe | "event" confused with "case" or "alert" | row count per day before and after dedupe |
| entity | the party an event belongs to (account, card, provider, device, merchant) | key | stable identifier with a lineage record | entity key drift after a system migration | key cardinality per day |
| eligibility filter | predicate deciding which events may be scored | boolean function | must be point-in-time computable | "eligible" silently equals "non-null score" | count of events failing the filter, by reason |
| point-in-time feature | feature whose value was knowable at `t0` | vector `x ∈ R^d` | no future information, no post-outcome fields | label-derived feature smuggled in as "context" | feature timestamp vs event timestamp audit |
| reference population | the declared distribution the score is relative to | set of vectors | fixed window, fixed eligibility, fixed model version | "the whole history" used implicitly | recorded window start/end and row count |
| anomaly score `a(x)` | scalar unusualness, larger = more unusual by convention | unitless, model-specific | monotone within one fitted model | read as probability or as risk | score direction documented per model |
| score direction | the sign convention mapping "unusual" to "high" | convention | must be uniform across the queue | reversed sign for distance-to-centre metrics | unit test on a known-unusual fixture |
| contamination `c` | assumed fraction of outliers used at fit time | probability | `0 < c < 0.5` typically | treated as the true prevalence | the value is stored with the model artifact |
| threshold `τ` | score cut separating selected from unselected | score units | any real value in score range | "the" threshold reused across populations | recompute per population and version |
| alert | a selected candidate that a human may review | case | after eligibility and dedupe | alert counted per row rather than per entity | alert-to-entity ratio |
| dedupe key | fields identifying "the same investigation" | tuple | stable across the review window | dedupe on event ID only, so repeats leak | duplicate rate per entity per day |
| review capacity `C` | investigations a team can start per period | investigations/day | measured, not aspirational | capacity quoted in "alerts" instead of investigations | time-and-motion sample of service time |
| service time `s` | mean reviewer minutes per investigation | minutes/investigation | measured with a distribution, not a single mean | mean used where tail matters for SLA | p50/p90 of observed handle time |
| precision@k | confirmed fraction among the top `k` mature reviewed alerts | dimensionless | `0 ≤ p@k ≤ 1`, `k ≥ 1` | reported with pending treated as negative | numerator, denominator, maturity date printed |
| alert rate `r` | selected alerts over eligible events in the same window | dimensionless | `0 ≤ r ≤ 1` | numerator and denominator from different windows | numerator/denominator source query |
| label maturity | adjudication is final and past any appeal window | boolean per alert | defined by a stated delay `d` | "reviewed" assumed to mean "mature" | maturity cohort table by review date |
| adjudication | human determination: confirmed, refuted, or abstain | categorical | mutually exclusive, with reason codes | abstain folded into refuted | reason-code distribution |
| drift | change in feature, score, or label distribution over time | distributional | compared on like-for-like windows | every drift treated as behaviour change | frozen-version replay |
| novelty | genuinely new legitimate behaviour unseen in the reference window | semantic | requires provenance evidence | novelty conflated with anomaly | launch/change log cross-check |
| segmentation | grouping of events into stable behavioural groups | partition | groups must be stable enough to act on | group labels treated as outcomes | label churn rate between refits |
| triage policy | the full rule: eligibility, ranking, cut, dedupe, SLA, escalation | object | must be versioned and owned | "threshold" used as a synonym for policy | policy document version in the case record |
| abstention | reviewer decision that evidence is insufficient | terminal state | must be counted separately | treated as a negative label | abstain rate per reviewer and cohort |
| break-even probability `p*` | calibrated detection probability at which review value equals review cost | probability | `0 ≤ p* ≤ 1` | confused with the chosen threshold quantile | recomputed when cost or value changes |

### 3.1 Three near-neighbour distinctions that cause most arguments

**Anomaly versus outlier versus novelty.** An outlier is a low-density point with respect to a fitted reference. An anomaly is an outlier that also violates an expectation someone cares about. A novelty is a legitimate new pattern. All three can look identical in feature space on the day of first observation; only provenance evidence separates them. The practical consequence: never write "anomaly" into a case record until the provenance check in Section 8 has run. Write "outlier, cause unresolved".

**Score versus probability versus rank.** The score is a model-internal ordering key. The probability is a calibrated expectation over events given a labelled population. The rank is the score's position within one population. Precision@k consumes ranks. The break-even rule consumes probabilities. Mixing them produces the classic error of comparing a raw isolation-forest score to a probability cutoff of `0.9`.

**Review capacity versus review throughput.** Capacity is the rate at which investigations can be *started* without unbounded queue growth. Throughput is what was actually completed. A team can start faster than it finishes for weeks, showing high throughput numbers while the queue silently accumulates. Section 9's queueing model exposes this.

### 3.2 Operational definitions that decide the arithmetic

Four further terms recur throughout the worked example and the failure table, and each one silently changes a denominator if it is defined loosely.

**Audit sample.** A set of events drawn from the *unreviewed* population by a sampling design that does not depend on the score, adjudicated by the same review process. Its purpose is to estimate the positives the queue missed. It is not an alert sample, it cannot be selected by score, and it must be held out of any training set whose future values it will later judge. A file that reports recall without an audit sample has no audit sample; it has a number.

**Reference window.** The date range and eligibility filter used to fit the model and to define "normal". Its length is a bias–variance trade: a long window includes stale regimes, a short window gives noisy quantile estimates. The quantile estimate's standard error is `sqrt(p(1−p)/n)` for a target tail probability `p` over `n` pooled observations. For a cut at the 99.5th percentile (`p = 0.005`) on a single day of `n = 12,000`, that is `sqrt(0.005×0.995/12,000) = 0.000644`, which is 12.9% relative to `p` and corresponds to a count uncertainty of `0.000644×12,000 = 7.73` alerts around a nominal 60. Pooling 28 days gives `n = 336,000` and reduces the relative error to `sqrt(0.005×0.995/336,000)/0.005 = 2.4%`, or about 1.5 alerts around 60. The reference window length is therefore a statistical decision with an operational consequence — a one-day reference makes a 7-alert day-to-day swing routine noise — and not a configuration default.

**Entity linkage.** The mapping from an event to the entity whose behaviour is being assessed. Linkage quality determines whether dedupe works and whether a case is coherent. Two failure shapes matter: under-linkage (one entity split across two keys, so its behaviour looks normal in both halves) and over-linkage (two entities merged, so each looks anomalous by inheriting the other's activity). Both are testable: under-linkage shows as a bimodal per-entity event count, over-linkage as implausibly high per-entity activity concentrated in one merge candidate.

**Suppression.** Any rule that removes a candidate from the queue before a human sees it: allowlists, expiry, business exclusions, duplicate collapse, or a hard rule that fires first. Suppression is legitimate. Suppression without a recorded reason is not, because the next investigation cannot distinguish "we knew this was legitimate" from "a join dropped it".

The distinction between these four terms is not pedantry: each one moves either the numerator or the denominator of a reported metric, and a metric whose denominator is defined by convention rather than by a written rule will drift as the team changes.

**Takeaway:** the vocabulary that prevents most triage disputes is entity versus event, score versus probability, and capacity versus throughput.

## 4. Core Formulas with Full Derivation

**Validity before use (applies to every formula in this section).** The ranking and precision quantities below require: a fixed evaluation window; a single model version; deduplication on a declared key; an eligibility filter applied identically to numerator and denominator; and labels that are mature, or an explicit maturity-adjusted denominator. They describe the observed reviewed population. They do not establish future prevalence, they do not establish recall without unreviewed sampling, and they do not identify a causal effect of the detector on any outcome.

Notation fixed for the whole section: `N` eligible events in the window; `a_i` score of event `i`; `C` investigations per window; `k` selected alerts (`k ≤ C` after dedupe); `y_i ∈ {0,1}` mature confirmed indicator; `π` the prevalence of true positives among eligible events; `s` mean service minutes per investigation.

### 4.1 Simplest / Base Case: selection, rate, and yield

The simplest case has a complete label set, unlimited capacity, and one model version. Even here the three quantities that matter are frequently conflated, so derive them from the definition.

**Selection count.** Define the selection function for a threshold `τ` as the indicator sum over eligible events. Because the indicator is a Bernoulli variable whose expectation is the survival probability of the score distribution, the count is a binomial draw when events are independent; the derivation is a direct application of the definition of the empirical distribution function.

```text
Validity: one window, one model version, deduplicated rows, independent events.
Derivation: A(tau) = sum_{i=1..N} 1[a_i > tau]
            = N * (1/N) * sum_i 1[a_i > tau]
            = N * S_hat(tau),  where S_hat is the empirical survival function.

FORMULA [ANO-01]: A(tau) = N * S_hat(tau)
WHERE: A(tau) selected alerts [alerts]; N eligible events [events];
       S_hat(tau) empirical survival probability [dimensionless].
CONSTRAINT: single version and window; ties at tau handled by the tie rule;
            independence is used only for the binomial variance, not the mean.
NUMERICAL EXAMPLE: N = 12,000 events, S_hat(0.71) = 0.0050;
                   A = 12,000 * 0.0050 = 60 alerts.
INTERPRETATION: capacity C = 60 implies the threshold must sit at the 99.5th
                percentile of this population's score distribution.
```

**Alert rate.** Dividing by the same `N` gives the operating rate. The derivation is a ratio of two counts from the same population, so the units cancel; the only way to get this wrong is to use a numerator from one window and a denominator from another, which is a join defect rather than a mathematical one.

```text
Validity: numerator and denominator from the same window and eligibility filter.
Derivation: r = A(tau) / N, both counts over the same eligible set.

FORMULA [ANO-02]: r = A(tau) / N
WHERE: r alert rate [alerts per event, dimensionless]; A alerts [alerts]; N [events].
CONSTRAINT: same window; dedupe applied before counting; N > 0.
NUMERICAL EXAMPLE: 60 / 12,000 = 0.005 = 0.5%.
INTERPRETATION: 0.5% sounds small but consumes the entire 60-review capacity,
                so "low rate" is not evidence of a well-controlled queue.
```

**Precision at k.** The yield of the queue is the confirmed fraction among the top `k`. Derive it as a sample mean of the label indicator over the selected prefix.

```text
Validity: labels mature for all of the top k; k <= number of reviewed alerts.
Derivation: TP_k = sum_{i=1..k} y_i  (ordered by descending score)
            p@k = TP_k / k  is a sample mean of y over the selected prefix.
            Its expectation under a fixed deterministic ranking is TP_k/k;
            under a random ranking with prevalence pi, E[p@k] = pi.

FORMULA [ANO-03]: p@k = (sum_{i=1..k} y_i) / k
WHERE: y_i mature label of the i-th ranked alert [0/1]; k prefix length [alerts].
CONSTRAINT: all k labels mature; ordering deterministic and reproducible;
            pending labels excluded rather than set to zero.
NUMERICAL EXAMPLE: 18 confirmed among k = 60 reviewed and mature -> 0.30.
INTERPRETATION: 0.30 is investigator yield in this queue at this capacity,
                not recall, not prevalence, and not a probability of fraud.
```

**The baseline that makes p@k interpretable.** A yield number alone is uninterpretable because the prevalence is unknown. The random-ranking baseline is a hypergeometric expectation: draw `k` rows without replacement from a population containing `πN` positives.

```text
Validity: ranking carries no information; sampling without replacement.
Derivation: TP_k ~ Hypergeometric(N, pi*N, k)
            E[TP_k] = k * pi
            Var[TP_k] = k * pi * (1-pi) * (N-k)/(N-1)
            The finite-population correction (N-k)/(N-1) appears because
            draws are without replacement.

FORMULA [ANO-04]: E[p@k | random] = pi ;  SD = sqrt(pi(1-pi)(N-k)/((N-1)k))
WHERE: pi prevalence [dimensionless]; N population; k prefix [alerts].
CONSTRAINT: exchangeable rows, single window, complete positive ascertainment.
NUMERICAL EXAMPLE: pi = 18/12,000 = 0.0015, k = 60, N = 12,000
                   E[TP] = 60 * 0.0015 = 0.09
                   Var = 60*0.0015*0.9985*(11,940/11,999) = 0.08942
                   SD(TP) = 0.2990, so SD(p@k) = 0.2990/60 = 0.00498
INTERPRETATION: a random top-60 would confirm about 0.09 events. Observed 18
                is a 200x lift, which is a strong ranking signal but is bounded
                by the assumption that pi is really 0.0015.
```

**Derivation check.** The variance computation is worth verifying by hand because it drives how many positives are needed before a lift claim is stable: `60 × 0.0015 = 0.09`; `0.09 × 0.9985 = 0.089865`; `11,940 / 11,999 = 0.995083`; product `0.089423`; square root `0.29904`; divided by `60` gives `0.004984`. All consistent with the block above.

### 4.2 General Case: maturity, capacity, queueing, and decision value

The general case adds the three features that make triage a hard problem: labels arrive late and non-randomly, capacity is a rate rather than a budget, and review value is not uniform across alerts.

**Maturity-adjusted precision.** Let `m` be the number of matured alerts in the top `k` and `δ = k − m` the pending count. The naive denominator `k` implicitly asserts that all pending alerts are negative. Derive the bias explicitly by decomposing the true confirmation count into matured and pending components.

```text
Validity: pending alerts are a well-defined stratum; the pending confirmation rate
          p_pend is unknown but bounded in [0,1].
Derivation: TP_full = TP_mature + TP_pending
            p_full = (TP_mature + delta * p_pend) / k
            p_mature = TP_mature / m
            p_mature >= p_full  <=>  TP_mature/m >= (TP_mature + delta*p_pend)/k
            which holds iff p_pend <= p_mature.
            So mature-only precision is optimistic exactly when pending cases are
            harder (lower confirmation rate) than already-decided cases.

FORMULA [ANO-05]: p_full = (TP_m + delta * p_pend) / k,  with p_pend in [0,1]
WHERE: TP_m matured confirmations [count]; delta pending [alerts]; k = m + delta.
CONSTRAINT: p_pend constant within the pending stratum; single review policy.
NUMERICAL EXAMPLE: TP_m = 18, m = 48, delta = 12, k = 60, p_pend = 0.25
                   p_mature = 18/48 = 0.375
                   p_full   = (18 + 12*0.25)/60 = 21/60 = 0.35
INTERPRETATION: report 0.375 as mature precision and 0.35 as the bounded
                estimate; the 0.025 gap is the size of the maturity claim.
```

The two bounds are the honest reporting form: `p_full ∈ [TP_m/k, (TP_m+δ)/k]` = `[0.30, 0.50]` in this example. That interval is wide, and its width is a direct measure of how much the maturity delay is costing the organisation in decision quality. Reducing `δ` is worth more than most modelling improvements of the same effort.

**Capacity arithmetic in investigations, not alerts.** Capacity is consumed per investigation. If alerts dedupe to `K_e` entities, labour is computed on `K_e`.

```text
Validity: service time measured on the same case type; no parallel work on one case.
Derivation: total minutes = K_e * s ; hours = K_e * s / 60 ;
            annual hours = 365 * K_e * s / 60 (calendar-day basis);
            annual labour cost = 365 * K_e * s / 60 * rate.

FORMULA [ANO-06]: H_day = K_e * s / 60 ;  Cost_day = H_day * rate
WHERE: K_e unique investigations [investigations/day]; s service minutes
       [min/investigation]; H_day [staff-hours/day]; rate [currency/staff-hour].
CONSTRAINT: mean service time; escalation and rework excluded; no leave/absence.
NUMERICAL EXAMPLE: K_e = 44, s = 20, rate = 45 currency/h
                   H_day = 44*20/60 = 14.6667 staff-hours
                   Cost_day = 14.6667*45 = 660.00 currency
                   With 18 confirmations, cost per confirmation = 36.67 currency
INTERPRETATION: the unit of the queue is the investigation, so dedupe is a
                capacity feature and not a cosmetic one.
```

**Queueing consequence.** A capacity figure that just barely covers demand produces a queue that grows without bound on any bad day. Model the review desk as `m` parallel reviewers with Poisson arrivals and exponential service (Erlang C). This is a `HEURISTIC` approximation: real service times are right-skewed and arrivals are bursty, so treat the output as an order-of-magnitude argument, not an SLA.

```text
Validity: Poisson arrivals, exponential service, m pooled reviewers,
          no abandonment, no priority, stable system (a < m).
Derivation: offered load a = lambda * s in reviewer-shift units;
            C_erlang(m,a) = [a^m/m! * m/(m-a)] /
                            [ sum_{j=0..m-1} a^j/j! + a^m/m! * m/(m-a) ];
            conditional mean wait given all servers busy is s/(m-a);
            so W_q = C_erlang * s/(m-a); W = W_q + s.

FORMULA [ANO-07]: W_q = C_erlang(m,a) * s/(m-a),  a = lambda*s,  rho = a/m
WHERE: lambda investigations/day; s in reviewer-days (1 shift = 480 min);
       m reviewer-shifts/day; W_q waiting time [days]; rho utilisation.
CONSTRAINT: a < m strictly; exponential service; single FIFO queue.
NUMERICAL EXAMPLE: lambda = 44/day, s = 20/480 = 0.041667 reviewer-day, m = 3
                   a = 44*0.041667 = 1.83333 ; rho = 0.61111
                   a^3/3! = 6.1620/6 = 1.02701 ; m/(m-a) = 3/1.16667 = 2.57143
                   numerator = 1.02701*2.57143 = 2.64088
                   sum_{j<3} a^j/j! = 1 + 1.83333 + 1.68056 = 4.51389
                   C_erlang = 2.64088/(4.51389+2.64088) = 2.64088/7.15477
                            = 0.36911
                   W_q = 0.36911 * 0.041667/1.16667 = 0.013183 day
                       = 0.013183*480 = 6.33 min
                   W   = 6.33 + 20 = 26.33 min
INTERPRETATION: at 61% utilisation the desk is comfortable; the same demand
                with two reviewers is not (see Section 5.6).
```

**Little's law as the consistency check.** Whatever the arrival and service distributions, the mean number in system equals arrival rate times mean time in system. This holds under weak assumptions and is the fastest way to catch a triage policy that quietly accumulates backlog.

```text
Validity: long-run stable system; consistent units; mean values.
Derivation: conservation of flow — in steady state the average number of
            entities in the system equals the average entry rate times the
            average residence time (Little, 1961).

FORMULA [ANO-08]: L = lambda * W
WHERE: L mean investigations in the system [count]; lambda [investigations/day];
       W mean residence time [days].
CONSTRAINT: steady state; same time unit on both sides.
NUMERICAL EXAMPLE: lambda = 44/day, W = 26.33/1440 = 0.018284 day
                   L = 44*0.018284 = 0.8045 investigations
INTERPRETATION: fewer than one case in flight on average is consistent with the
                Erlang calculation; a measured backlog far above this signals
                abandonment, rework, or an arrival process the model ignores.
```

**The break-even probability.** Review is worth doing when the expected value of the review exceeds its cost. This converts triage from a ranking exercise into a decision rule, and it is the only formula here that uses an actual probability rather than a rank.

```text
Validity: single alert review; value V accrues only on confirmation;
          review cost c is incurred regardless of outcome; risk-neutral objective.
Derivation: EV(review) = p*V - c ; EV(skip) = 0
            review iff p*V - c > 0 iff p > c/V
            With capacity binding, order candidates by EV which, for constant
            V and c, is the same order as p. Hence ranking by a calibrated
            probability is equivalent to ranking by expected value.

FORMULA [ANO-09]: p* = c / V
WHERE: p* break-even probability [dimensionless]; c review cost
       [currency/investigation]; V expected recoverable value per confirmed
       case [currency/case].
CONSTRAINT: constant c and V; no capacity constraint in the derivation
            (capacity then truncates the ranked list at C).
NUMERICAL EXAMPLE: c = 15 currency (20 min at 45/h), V = 1,800 currency
                   p* = 15/1800 = 0.008333
INTERPRETATION: any alert whose calibrated probability exceeds 0.83% is worth
                a review on value grounds alone; if the queue is still shorter
                than that rule implies, the binding constraint is headcount.
```

This is the single most useful number a triage team can compute, because it separates two very different remedies. If the observed precision at the cut is far above `p*`, the organisation is under-reviewing and should add capacity. If observed precision is below `p*`, adding capacity destroys value and the ranking must improve first.

**Calibration from rank to probability.** Unsupervised scores are ranks, so `p*` cannot be applied directly. Two workable conversions exist, and both need mature labels.

```text
Validity: a labelled cohort with scores from the same model version and population.
Derivation (binned): partition the score range into bins B_j; estimate
            p_hat_j = confirmations_j / reviewed_j within each bin; this is a
            histogram estimator of P(y=1 | a in B_j).
Derivation (isotonic): fit a non-decreasing function g minimising squared error
            subject to monotonicity; g(a) is the calibrated probability.

FORMULA [ANO-10]: p_hat_j = c_j / n_j ;  monotone g = isotonic(a -> y)
WHERE: c_j confirmations in bin j [count]; n_j mature reviews in bin j [count].
CONSTRAINT: n_j large enough for the binomial error to be tolerable;
            reviews must not be censored by score within the bin.
NUMERICAL EXAMPLE: top decile band [0.80,0.92): 21 confirmations / 60 reviews
                   p_hat = 0.35 ; band [0.60,0.80): 6/120 = 0.05
INTERPRETATION: the 0.35 band clears p* = 0.0083 by a wide margin; the 0.05 band
                also clears it, which is what makes the capacity argument.
```

The censoring caveat in ANO-10 is not decorative. If only the top 60 of a population are ever reviewed, every bin below the cut has `n_j = 0`, so calibration is impossible below the cut without a random audit sample. This is the same identification problem as Section 5, appearing in a second guise.

### 4.3 Key Special Cases / Variants

Each variant below changes the score construction, not the triage logic. The ranking, capacity, and maturity formulas apply unchanged, which is precisely why method selection should be driven by the failure mode you can observe rather than by benchmark folklore.

| Variant | What changes | Assumptions | Use when | Avoid when | Equation ID |
|---|---|---|---|---|---|
| Robust z-score (MAD) | score is scaled deviation from a robust centre | unimodal, roughly symmetric marginal per feature | one or few interpretable features; fast triage | multi-modal marginals; `MAD = 0` | ANO-11 |
| Mahalanobis | score is covariance-weighted distance | elliptical, approximately Gaussian cloud; invertible covariance | correlated continuous features, moderate `d` | `d` comparable to `n`; heavy tails | ANO-12 |
| Isolation forest | score from expected random-partition depth | anomalies are few and separable by axis splits | mixed types, non-linear, ranking needed | dense clusters of anomalies; strong masking | ANO-13 |
| LOF | score is local density ratio | local density meaning is meaningful | density varies across regions | high `d`; poor scaling; tiny `k` | ANO-14 |
| One-class SVM | score is distance to a learned boundary | boundary captures normal support | compact normal region, kernel well chosen | large `n`; unstable `nu` | ANO-15 |
| Reconstruction error (autoencoder) | score is residual magnitude | normal manifold is learnable and low-dimensional | high-dimensional correlated signals | leakage of the anomaly into training; no validation set | ANO-16 |
| Change point (CUSUM) | score is accumulated deviation of an aggregate | pre-change mean known or estimated | ordered aggregates, ramp shifts | per-event attribution required | ANO-17 |

**Robust z-score with MAD.** The median absolute deviation is a scale estimator with a 50% breakdown point, which is what makes it usable on contaminated data where the standard deviation is not.

```text
Validity: approximately unimodal marginal; MAD > 0; deviation intended per feature.
Derivation: for X ~ Normal(mu, sigma), MAD = median|X - mu| = 0.6745*sigma,
            so (x - median)/MAD is in units of sigma/0.6745; multiplying by
            0.6745 returns approximately sigma units.

FORMULA [ANO-11]: z = 0.6745 * (x - median(x)) / MAD,  MAD = median|x - median(x)|
WHERE: x feature value; z robust standardised deviation [sigma units approx].
CONSTRAINT: MAD > 0; single feature; unimodality for the sigma interpretation.
NUMERICAL EXAMPLE: median = 10, MAD = 2, x = 17
                   z = 0.6745 * 7 / 2 = 4.7215/2 = 2.361
INTERPRETATION: 2.36 robust sigma is unremarkable for a heavy-tailed feature and
                alarming for a tightly controlled sensor — the scale is relative.
```

**Mahalanobis distance.** Whitening turns an elliptical cloud into a sphere, after which squared distance to the centre has a known distribution under Gaussianity. That distribution is what makes a tail probability available at all.

```text
Validity: multivariate normal (or approximately elliptical) cloud; Sigma invertible;
          mean and covariance estimated on a reference window without the events scored.
Derivation: z = Sigma^{-1/2}(x - mu) has identity covariance;
            D^2 = ||z||^2 = (x-mu)^T Sigma^{-1} (x-mu);
            for Gaussian x, D^2 ~ chi-square with p degrees of freedom.

FORMULA [ANO-12]: D^2 = (x-mu)^T Sigma^{-1} (x-mu) ~ chi2_p
WHERE: x [p-vector]; mu [p-vector]; Sigma [p x p]; D^2 [squared sigma units].
CONSTRAINT: p < n for a stable inverse; no severe outliers in the estimate;
            elliptical symmetry for the chi-square tail to be meaningful.
NUMERICAL EXAMPLE: p = 4, 99.5th percentile of chi2_4 = 14.860
                   D_cut = sqrt(14.860) = 3.855
INTERPRETATION: a 4-feature Gaussian cloud gives a principled 0.5% cut; the
                same cut on a bimodal cloud flags an entire legitimate mode.
```

**Isolation forest score.** The mechanism is expected path length in an ensemble of random binary partitions. Short average depth means the point was isolated quickly, which is unusual.

```text
Validity: anomalies are few, and axis-aligned random splits isolate them sooner
          than normal points; subsample size psi fixed; ensemble averaged.
Derivation: for a binary search tree built on n points, the expected path length
            of an unsuccessful search is c(n) = 2H(n-1) - 2(n-1)/n, with H the
            harmonic number. Normalising the observed mean depth by c(n) and
            mapping through 2^{-x} puts the score in (0,1) with 0.5 at average depth.

FORMULA [ANO-13]: s(x,n) = 2^{-E[h(x)]/c(n)},  c(n) = 2H(n-1) - 2(n-1)/n
WHERE: h(x) path length [edges]; E[h] ensemble mean; n subsample size [points];
       s score in (0,1], larger = more isolated.
CONSTRAINT: single fit; average over t trees; n = psi not the full dataset.
NUMERICAL EXAMPLE: n = 256, H(255) = 6.1204, c = 12.2409 - 1.9922 = 10.2487
                   E[h] = 4  -> s = 2^{-4/10.2487} = 2^{-0.3903} = 0.763
                   E[h] = 2  -> s = 2^{-0.1952} = 0.873
                   E[h] = 10.2487 -> s = 0.5 exactly
INTERPRETATION: the score is a normalised depth, so a value of 0.763 means
                "isolated in 39% of the average depth", not "76% fraud".
```

**Local outlier factor.** LOF compares a point's local density to the densities of its neighbours, so it detects points that are unusual *relative to their own region* rather than unusual globally.

```text
Validity: k neighbours defined; distances comparable (scaled features);
          local density is a meaningful concept for the data.
Derivation: k-distance(x) = distance to the k-th nearest neighbour;
            reach-dist_k(x,o) = max(k-distance(o), d(x,o));
            lrd_k(x) = 1 / ( (1/k) * sum_{o in N_k(x)} reach-dist_k(x,o) );
            LOF_k(x) = (1/k) * sum_{o in N_k(x)} lrd_k(o) / lrd_k(x).

FORMULA [ANO-14]: LOF_k(x) = mean_{o in N_k(x)} lrd_k(o) / lrd_k(x)
WHERE: lrd local reachability density [inverse distance]; LOF dimensionless,
       LOF > 1 means locally sparser than neighbours.
CONSTRAINT: k >= 2 in practice; feature scaling applied; no ties at k-distance.
NUMERICAL EXAMPLE: k = 4, mean reach-dist from x = 2.0 -> lrd(x) = 0.500
                   each neighbour has lrd = 0.800
                   LOF = 0.800/0.500 = 1.60
INTERPRETATION: x sits in a region 60% less dense than its neighbours; inside a
                uniformly sparse region this is normal and LOF stays near 1.
```

**One-class SVM.** The method learns a boundary that encloses a chosen fraction of the training mass and scores points by their position relative to that boundary. The parameter `nu` is not a contamination guess in disguise; it is an upper bound on the training-set rejection fraction and a lower bound on the support-vector fraction, so it controls the shape of the solution as well as its volume.

```text
Validity: the normal class is compact in the chosen kernel space; features scaled;
          kernel bandwidth chosen from data scale, not default.
Derivation: the primal problem minimises (1/2)||w||^2 + (1/(nu n)) sum_i xi_i - rho
            subject to w.phi(x_i) >= rho - xi_i, xi_i >= 0. Solving via Lagrange
            multipliers alpha_i gives w = sum_i alpha_i phi(x_i), with
            0 <= alpha_i <= 1/(nu n) and the constraint sum_i alpha_i = 1.
            The support-vector count therefore satisfies
            (fraction with alpha_i > 0) >= nu, and the training rejection
            fraction is <= nu. The score is f(x) = sum_i alpha_i K(x_i,x) - rho.

FORMULA [ANO-15]: f(x) = sum_i alpha_i K(x_i,x) - rho ; anomaly if f(x) < 0
WHERE: alpha_i dual weights [dimensionless]; K kernel; rho offset [kernel units];
       f decision function [kernel units, sign meaningful only].
CONSTRAINT: scaled features; nu in (0,1]; kernel bandwidth comparable to the
            feature scale; the score magnitude is not a probability.
NUMERICAL EXAMPLE: n = 2,000, nu = 0.01 -> at most 20 training points may be
                   rejected, and at least 20 must carry alpha_i > 0;
                   decision threshold is f(x) = 0, not a value of 0.01 or 0.99.
INTERPRETATION: "1% of training data" sets the boundary volume; it says nothing
                about the false positive rate on tomorrow's population.
```

**Reconstruction error.** An autoencoder learns a compressed representation of normal behaviour and scores a record by how badly it fails to reconstruct. Its statistical weakness is that the reconstruction model has no notion of prevalence, so the score is a residual magnitude whose distribution must be calibrated empirically on a held-out normal window.

```text
Validity: training set contains predominantly normal records; a held-out normal
          window exists for threshold calibration; residual distribution is
          stable across windows.
Derivation: encoder e: R^d -> R^h, decoder g: R^h -> R^d, trained to minimise
            mean squared error on normal data. For a new x, the residual
            r = x - g(e(x)); the anomaly score is ||r||^2 (or a weighted sum).
            Under a Gaussian residual model with covariance Sigma_r,
            ||r||^2 follows a scaled chi-square, which supplies a tail cut.

FORMULA [ANO-16]: R(x) = ||x - g(e(x))||^2 ; anomaly if R(x) > q_{1-alpha}
WHERE: R reconstruction error [squared feature units]; e encoder; g decoder;
       q tail quantile of R on a held-out normal window.
CONSTRAINT: anomalies absent from training; feature scaling consistent between
            fit and score; residual distribution re-checked each reference window.
NUMERICAL EXAMPLE: held-out normal window n = 5,000 gives the 99.5th percentile
                   of R as 0.0412; a record with R = 0.0655 exceeds the cut and
                   is scored as an outlier at a 0.5% budget, not as an anomaly.
INTERPRETATION: the threshold is a quantile of a residual on normal data; any
                shift in that residual distribution invalidates the cut silently.
```

**CUSUM for ordered aggregates.** When the anomaly lives in an aggregate time series rather than an individual row, accumulation beats single-point scoring because a small persistent shift is invisible point-by-point.

```text
Validity: ordered observations; pre-change mean mu0 known or estimated from a
          stable reference window; single-sided shift direction.
Derivation: two-sided CUSUM tracks S_t = max(0, S_{t-1} + (x_t - mu0 - K));
            the reference value K = delta/2 is the half-shift convention, so
            deviations smaller than K do not accumulate and larger ones do.
            Alarm when S_t > h; (h, K) set the average run length, which for
            Gaussian data is obtained by simulation or Siegmund's approximation.

FORMULA [ANO-17]: S_t = max(0, S_{t-1} + x_t - mu0 - K) ; alarm if S_t > h
WHERE: mu0 pre-change mean [x units]; K reference shift [x units];
       h decision threshold [x units]; S_t cumulative sum [x units].
CONSTRAINT: stable pre-change regime; known sigma for ARL calibration;
            reset after alarm, and record the reset.
NUMERICAL EXAMPLE: mu0 = 10, K = 0.5, h = 4,
                   x = 10.2, 10.9, 11.4, 12.1, 11.8
                   increments: -0.3, 0.4, 0.9, 1.6, 1.3
                   S: 0, 0.4, 1.3, 2.9, 4.2 -> alarm at t = 5
INTERPRETATION: no single observation is extreme; the accumulated drift is what
                crosses the line, which is why per-event scoring misses ramps.
```

**Rank transformation as the universal adapter.** Any monotone transform leaves the review order unchanged. This is why you can standardise scores across model versions for reporting without changing a single decision, and why you cannot fix a bad threshold by rescaling.

```text
Validity: transform strictly increasing on the score support.
Derivation: if g is strictly increasing then a_i > a_j <=> g(a_i) > g(a_j);
            therefore every prefix of the ranking is identical, and p@k is identical.
            But for a fixed numeric tau, the selected set changes unless
            g(tau_new) = tau_old.

FORMULA [ANO-18]: rank-preserving if g strictly increasing;
                  p@k(g(a)) = p@k(a) for all k
WHERE: g monotone transform; a score vector; k prefix length.
CONSTRAINT: strict monotonicity; ties handled identically before and after.
NUMERICAL EXAMPLE: percentile transform u = rank/N: top 60 of 12,000 maps to
                   u in (11,940/12,000, 1.0] = (0.995, 1.0]; decisions unchanged,
                   the threshold moves from 0.71 to 0.995 - reporting only.
INTERPRETATION: normalising scores is a communication convenience; it is not a
                model improvement and must not be presented as one.
```

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

The purpose of this subsection is to make the failure boundaries executable. Each row is a check you can run, not a caveat to recite.

| Result / invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| `p@k` is a yield estimate | all `k` labels mature | maturity cohort count vs `k` | 12 of 60 pending; `18/60` reported as final | ANO-05 bounds, then wait for maturity |
| `p@k ≥ π` implies lift | `π` measured on the same population | audit sample of unreviewed rows | `π` assumed from `18/12,000` while labels are review-selected | stratified audit sample |
| ranking is threshold-free | comparisons stay inside one fit | model version in the case record | 0.71 reused after a refit with new features | recompute quantile per version |
| top-`C` selection respects capacity | `C` in investigations after dedupe | alert-to-entity ratio | 60 alerts, 44 entities, capacity quoted as 60 | ANO-06 on entities |
| queue is stable | `lambda*s < m` | utilisation vs 1 | 88 investigations/day, capacity 72/day | add staff or cut `k` |
| precision is prevalence-sensitive | none — this is definitional | compare two populations' precision at equal TPR/FPR | same detector, 30% in one segment, 8% in another | report per-segment |
| MAD scaling is valid | `MAD > 0` | count of zeros | constant sensor readings → all `z = 0` | use a floor or another method |
| Mahalanobis tail is calibrated | Gaussianity, `p < n` | `p/n` ratio and marginal plots | `p = 400`, `n = 500` → singular-ish inverse | shrinkage covariance, `p` reduction |
| score comparability across versions | monotone map known or labels available | frozen replay | thresholds copied across versions | recalibrate on a common cohort |
| recall is identifiable | unreviewed sampling exists | audit-sample size and design | recall quoted from reviewed alerts only | stratified sampling, Section 5 |

**The `k → N` limit.** As you select every event, `p@k` converges to the population prevalence `π` regardless of model quality. This is a useful sanity check on any yield curve: the far right of the curve must land on the base rate. If it does not, the label set is incomplete, because the model cannot manufacture prevalence.

**The `π → 0` limit with fixed capacity.** Precision at a fixed *rate* degrades linearly in the positive class while the number of false positives at that rate stays constant. Concretely, with a fixed operating point at false positive rate `f`, selecting `fN` events yields `fN(1−π)` false positives and `TPR·πN` true positives, so

```text
precision = TPR*pi / (TPR*pi + f*(1-pi))  ->  0  as  pi -> 0  when f > 0
```

```text
Validity: single threshold with fixed TPR and FPR; independent events.
Derivation: TP = TPR * pi * N ; FP = FPR * (1-pi) * N ;
            precision = TP/(TP+FP), divide numerator and denominator by N.

FORMULA [ANO-19]: precision = (TPR*pi) / (TPR*pi + FPR*(1-pi))
WHERE: TPR sensitivity [dimensionless]; FPR [dimensionless]; pi prevalence.
CONSTRAINT: same threshold for both classes; no post-hoc filtering.
NUMERICAL EXAMPLE: pi = 0.0015, TPR = 0.45, FPR = 0.005
                   numerator = 0.000675
                   denominator = 0.000675 + 0.0049925 = 0.0056675
                   precision = 0.1191
INTERPRETATION: a 45% sensitive detector at 0.5% FPR yields 11.9% precision at
                this prevalence; the base rate, not the model, sets the ceiling.
```

**The ties counterexample.** Discrete scores break the quantile rule. Suppose `N = 12,000`, `C = 60`, and the ordered scores are `a_(58) = 0.716`, `a_(59) = 0.714`, `a_(60) = 0.712`, `a_(61) = 0.712`, `a_(62) = 0.709`. The rule "cut at `a_(60)`" admits both 0.712 rows, giving 61 investigations against a capacity of 60. Two bounded responses exist: apply a deterministic secondary sort key (oldest unresolved, or highest expected value), or accept `C+1` and record the overrun. What must not happen is an untracked 1.7% overrun every day, which compounds into a standing backlog.

**The `k > n_reviewed` limit.** Requesting precision at a `k` larger than the number of mature reviews forces either an interpolation assumption or a silent negative fill. Both are inventions. Report the yield curve only up to the mature count and mark the rest "not identified".

**Takeaway:** every formula in this section has a hard validity edge — maturity, capacity units, prevalence, ties, and version identity — and each edge has a check that runs in minutes.

## 5. Worked Numerical Example

**Evidence label: ILLUSTRATIVE inputs, VERIFIED arithmetic.** The event counts, costs, handle times, and outcomes below are constructed teaching values for a generic review desk. They are not measured from any organisation, and the scenario is not a description of a real deployment.

**Workflow and decision.** A review desk scores 12,000 eligible events per day. It can start 60 investigations per day. Each investigation takes 20 minutes of a reviewer's time, loaded at 45 currency units per staff-hour. Each confirmed case is worth an assumed 1,800 currency units of recoverable loss. The desk must decide: how many alerts to review, whether to add staff, and what yield to report while 12 adjudications are still pending.

### 5.1 Step 1 — the small hand-checkable case

Ten consecutive alerts from one day, ranked by descending score, with dedupe visible:

| rank | score | entity | maturity | outcome |
|---:|---:|---|---|---|
| 1 | 0.912 | E31 | mature | confirmed |
| 2 | 0.884 | E07 | mature | confirmed |
| 3 | 0.861 | E12 | mature | refuted |
| 4 | 0.842 | E31 | mature | confirmed |
| 5 | 0.838 | E19 | mature | confirmed |
| 6 | 0.821 | E02 | mature | refuted |
| 7 | 0.803 | E44 | mature | confirmed |
| 8 | 0.795 | E07 | mature | refuted |
| 9 | 0.781 | E55 | pending | — |
| 10 | 0.774 | E08 | mature | refuted |

Hand arithmetic, alert level: `p@5 = 4/5 = 0.80`; `p@10` with pending excluded is `5/9 = 0.5556`; `p@10` if pending were wrongly treated as negative is `5/10 = 0.50`. The gap between 0.5556 and 0.50 is the size of one label, which is exactly why the denominator rule must be written down rather than remembered.

Entity level: the ten alerts cover entities `{E31, E07, E12, E19, E02, E44, E55, E08}` — eight unique investigations, so the alert-to-entity ratio is `10/8 = 1.25`. Two entities appear twice (E31, E07). Confirmed *entities* among the first five alerts are E31 and E19, so entity-level precision over the four entities reached is `2/4 = 0.50`, which differs from alert-level `0.80`. Neither number is wrong; they answer different questions and must never be mixed in one table.

Independent check: the duplicate count is verified by `alerts − unique = 10 − 8 = 2`, matching the two repeated keys. The outcome counts sum to `5 confirmed + 4 refuted + 1 pending = 10`. Both checks pass.

### 5.2 Step 2 — the daily aggregate

| Quantity | Value | Derivation |
|---|---:|---|
| eligible events `N` | 12,000 | eligibility query, one version |
| selected alerts `A` | 60 | threshold at the 99.5th percentile |
| alert rate `r` | 0.005 | `60 / 12,000` |
| unique investigations `K_e` | 44 | dedupe on entity within a 24 h window |
| alert-to-entity ratio | 1.364 | `60 / 44` |
| mature after 72 h | 48 | 18 confirmed, 30 refuted |
| pending | 12 | adjudication delay beyond 72 h |

The alert-to-entity ratio of `1.364` matters more than it looks. Capacity is 60 investigations, but only 44 distinct investigations exist, so the desk is at 73% of a naive alert-based load estimate before any dedupe at all. Conversely, quoting capacity in alerts while measuring load in investigations systematically overstates headroom by 36%.

### 5.3 Step 3 — the reported yield, three ways

Mature-only: `18/48 = 0.375`. Bounded full estimate with `p_pend = 0.25`: `(18 + 12×0.25)/60 = 21/60 = 0.35`. Hard bounds: `[18/60, (18+12)/60] = [0.30, 0.50]`.

The correct reporting sentence names the denominator and the bound: "mature precision 37.5% (18/48); full-cohort precision bounded in 30.0%–50.0% with 12 pending". Anything shorter is ambiguous, and ambiguity in a yield report propagates into staffing decisions.

### 5.4 Step 4 — labour and unit economics

```text
H_day      = 44 * 20 / 60            = 14.6667 staff-hours
Cost_day   = 14.6667 * 45            = 660.00 currency
Cost/conf  = 660.00 / 18             = 36.67 currency per confirmed case
p*         = (20/60)*45 / 1800
           = 15 / 1800               = 0.008333
```

Check the middle line by an alternate route: cost per investigation is `20/60 × 45 = 15.00`, so 44 investigations cost `44 × 15 = 660.00`. Both routes agree. Cost per confirmation uses the confirmed count, so if the pending 12 resolve with 3 more confirmations, cost per confirmation becomes `660/21 = 31.43`.

The break-even probability `p* = 0.83%` is the decision-relevant number. At the observed 35% precision, each review returns `0.35 × 1,800 = 630` in expected value against a cost of 15, a net `615` per review. The desk is not over-reviewing; it is under-reviewing.

### 5.5 Step 5 — how many alerts could the value case support?

If every alert at the current cutoff clears `p* = 0.0083`, then the value case supports reviewing until the calibrated probability falls below that level. Suppose the next score band `[0.60, 0.80)` contains 120 alerts with a calibrated probability of 0.05. Those still clear `p*` by a factor of six. So the economically justified volume is roughly `60 + 120 = 180` alerts per day, which is three times current capacity. That is a capacity finding, not a model finding, and the two must be reported separately or the organisation will buy a better model when it needs two more reviewers.

At 180 investigations with 20 minutes each: `180 × 20/60 = 60 staff-hours/day`, requiring `60/8 = 7.5` full-time equivalents, against 3 today. This is the kind of arithmetic that makes a triage review useful.

### 5.6 Step 6 — what happens when capacity is cut

Same 44 investigations per day, but two reviewers instead of three.

```text
a        = 44/24                  = 1.83333
m        = 2
m - a    = 0.16667
a^2/2!   = 3.36111/2              = 1.68056
m/(m-a)  = 2/0.16667              = 12.000
numerator= 1.68056*12             = 20.1667
sum_j<2  = 1 + 1.83333            = 2.83333
C_erlang = 20.1667/(2.83333+20.1667) = 20.1667/23.000 = 0.87681
W_q      = 0.87681 * (0.041667/0.16667) = 0.87681*0.25000
         = 0.21920 day           = 105.2 min
W        = 105.2 + 20             = 125.2 min
```

Halving the review team from three to two multiplies waiting time by `105.2/6.33 = 16.6`. That non-linearity is the entire reason to compute the queue instead of assuming "two thirds the staff, similar delay". The Erlang figure assumes exponential service and Poisson arrivals; real desks have bursts and vacations, so treat 105 minutes as a lower bound on the delay you would actually observe.

### 5.7 Step 7 — the identification problem, with numbers

The observed 18 confirmations in the top 60, with `π = 18/12,000 = 0.0015`, is internally inconsistent with an ordinary detector. Here is why.

```text
At C = 60, the false positive rate is bounded: FPR <= 60/11,982 = 0.005008.
Suppose TPR = 0.45 and FPR = 0.005; ANO-19 gives precision 0.1191.
To reach precision 0.30 with FPR = (60-18)/11,982 = 0.003505,
solve 0.30 = 0.0015*T / (0.0015*T + 0.9985*0.003505):
0.30*(0.0015T + 0.003500) = 0.0015T
0.00045T + 0.001050 = 0.0015T
0.00105 = 0.0015T - 0.00045T = 0.00105T
T = 1.00
```

So the 30% precision figure *requires* perfect recall: all 18 positives must be inside the top 60. A detector achieving perfect recall at a 0.5% review rate is not credible, which means the premise `π = 0.0015` is wrong. The 18 confirmations are not the population of positives; they are the positives *that happened to be reviewed*. The label mechanism censors everything outside the queue.

```text
Audit design: draw n_audit = 400 rows uniformly from the 11,940 unreviewed rows,
adjudicate them, count positives c_a.
Estimated missed positives      = (c_a / n_audit) * 11,940
Estimated total positives       = 18 + missed
Recall@60                       = 18 / total

With c_a = 2:  missed = (2/400)*11,940 = 59.7
               total  = 18 + 59.7 = 77.7
               recall@60 = 18/77.7 = 0.2317
Binomial SE of c_a/n: sqrt(0.005*0.995/400) = 0.003526
95% interval on the rate: 0.005 +/- 1.96*0.003526 -> (-0.0019, 0.0119)
Truncated to [0, 0.0119] -> missed in [0, 142] -> recall in [0.113, 1.000]
```

The audit is directionally informative and statistically useless at `n = 400`: the recall interval spans almost the whole range. Using the Hoeffding form from Section 10, to estimate a proportion to `±0.01` with 95% confidence requires `n = ln(40)/(2×0.0001) = 18,444` — infeasible for adjudication. The practical resolution is a *stratified* audit: sample heavily from the high-score band just below the cut, where missed positives are plausible, and lightly from the far tail where they are not. That is a design decision worth stating explicitly in any recall claim, and worth refusing to make silently.

**Wrong path, for contrast.** Treating pending as negative gives 0.30; treating `π` as `18/12,000` gives a 200× lift claim; both are defensible-looking numbers that a reviewer would accept without question. The discriminating question is "how could a positive outside the queue ever be counted?" If no answer exists, the lift and the recall are both unidentified.

### 5.8 Step 8 — production-scale change

At 180 alerts/day (the value-justified volume), dedupe at 1.364 gives 132 investigations, `132 × 20/60 = 44` staff-hours, `44/8 = 5.5` FTE. Case storage at 1.5 MB per case and 2-year retention: `132 × 365 × 2 × 1.5 MB = 144,540 MB ≈ 141 GiB`. These are order-of-magnitude figures that assume constant daily volume, no seasonality, and no rework; each of those assumptions is a measurement task, not a modelling task.

### 5.9 What this example does and does not establish

It establishes: the arithmetic of capacity, yield, maturity bounds, break-even value, queue sensitivity, and the identification failure. It does not establish that any particular detector, feature set, or cost figure is correct, and the 18/48 outcomes are constructed. The transferable part is the sequence — small hand check, aggregate, yield with bounds, unit economics, queue sensitivity, identification audit — not the specific numbers.

**Takeaway:** the worked chain shows that most triage questions are arithmetic with a named denominator, and that the hardest question (recall) is a sampling-design question rather than a modelling one.

## 6. Visual / Geometric Intuition

### 6.1 Score distributions and the selected region

```text
density
  |                                   normal mass
  |        .........................................
  |     ...                                         ...
  |   ..                                               ..
  |  .                                                   .
  | .                                                     .
  |.            |-------------------- selected ---------->|
  +---.---------.-------------------------------------------.----> score
    0.0       0.71                                      1.0
               ^ threshold at 99.5th percentile
               |  (area to the right = 0.005 = 60 of 12,000)
```

Caption (schematic, not to scale): the shaded tail area is the alert rate, which is a property of the score distribution and the threshold, not of the phenomenon. Reading this diagram in reverse is the key move: choose the tail area first from capacity, then read the threshold off the axis. Choosing the threshold first and discovering the tail area afterwards is how queues are created.

### 6.2 Why precision and AUC diverge

```text
                     score
              low            high
        +-------------+-------------+
  pos   |     FN      |     TP      |   P = pi*N
        +-------------+-------------+
  neg   |     TN      |     FP      |   N_neg = (1-pi)*N
        +-------------+-------------+
              ^ vertical line = threshold
   precision = TP / (TP + FP)   <-- both cells on the RIGHT
   recall    = TP / (TP + FN)   <-- both cells on the LEFT
   AUC       = P(score_pos > score_neg)  <-- rank order only, no cells
```

Caption: AUC depends only on where positives and negatives fall in the ordering, so it is invariant to prevalence. Precision depends on the absolute counts in the right-hand column, so it collapses when the negative population grows relative to the positive one. Two populations can share an AUC of 0.90 and differ in precision at the same operating point by a factor of ten. This is why a detector "validated" with AUC on a balanced research sample can fail on a 0.15% prevalence queue without anything being wrong with the code.

### 6.3 Isolation geometry

```text
+-----------------------------------+
|        .    .                     |
|   .  .        .        +----------+
|  .       .          .  |          |
|     .        .         |   RARE   |
|  .     .  .            |     *    |
|        .               +----------+
|   .   .       .                   |
|        .   .     .                |
+-----------------------------------+
  Dense region: many random axis splits needed to isolate a point.
  Rare region : a few splits separate * because it sits far from the mass.
```

Caption: the isolation forest mechanism asks how many random axis-aligned cuts are needed to separate a point. The consequence that matters in triage is masking: if a *cluster* of unusual points exists, each member has neighbours, so no member is isolated quickly and the whole cluster scores as normal. Global density methods and isolation methods share this blind spot; only a segmentation step that changes the reference population removes it.

### 6.4 Local density geometry and the LOF ring

```text
   sparse region                     dense region
   o      o                        ooooooooo
       x                            ooooooooo
   o        o                       ooooooooo
     (k-NN radius large)            (k-NN radius small)

   LOF(x) > 1  if x sits at the edge of a dense region
   LOF(x) ~ 1  if x sits anywhere inside a uniformly sparse region
```

Caption: LOF is a *ratio* of neighbourhood densities. A point in the sparse region is far from everything globally, yet has `LOF ≈ 1` because its neighbours are equally far. This inverts the usual priority: in a business setting, the sparse-region point may be more interesting (a new market, a new product) while LOF flags the boundary point.

### 6.5 The clustering-artefact picture

```text
true space                       projected/segmented view
  A  A  A                          [ A A A ][ B B ][ B ]
  A  A  A     B  B        -->      [ A A A ][ B B ][ B ]
  B  B        B  B                 boundary looks crisp, but the
                                   split was chosen by the projection
```

Caption: a segmentation is a hypothesis about structure, not an observation of it. If the boundary between two groups moves when the seed, the feature subset, or the scaling changes, then any "anomalous cluster" claim built on it is an artefact of the projection. The check is stability across re-runs, which Section 8 treats as a required diagnostic.

### 6.6 The queueing picture

```text
wait
 |                                  x  (rho -> 1)
 |                              x
 |                          x
 |                    x
 |            x
 |      x
 +--x------------------------------------> utilisation rho
   0.3   0.6   0.8   0.9   0.95   0.99   1.0
```

Caption (schematic): waiting time is convex in utilisation, so the last 10% of capacity absorbs a disproportionate share of delay. This is why "we are at 92% utilisation, we have headroom" is a statement about cost, not about service. At `rho = 0.83` with a single server the mean wait is roughly `5 ×` service time; at `rho = 0.95` it is roughly `19 ×`. Multi-server pooling flattens the curve, which is why three pooled reviewers beat three specialists with separate queues.

### 6.7 Reading a precision-yield curve

```text
precision
 1.0 |*
     | *
 0.6 |  *
     |    *
 0.4 |      *-.
     |          '-._
 0.2 |              '-.._
     |                    '-...._
 0.0 +------------|---------------|--------> k (alerts reviewed)
     0           60             180        N
                  ^ capacity      ^ break-even probability crossing
```

Caption: the curve falls steeply at first and then flattens toward the prevalence `π`. The decision-relevant crossing is not where precision "looks bad" but where it crosses `p* = c/V`. In this illustrative shape the crossing is far to the right of current capacity, which is a capacity finding rather than a ranking finding — exactly the distinction Section 5.5 formalises.

**Takeaway:** the four pictures that change decisions are the tail area (capacity sets the threshold), the confusion quadrants (prevalence sets precision), the isolation geometry (clusters mask themselves), and the convex wait curve (headroom is not linear).

## 7. Comparison / Selection Table

### 7.1 Method comparison under triage conditions

| Method | Preconditions | Quality / guarantee | Cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Explicit rule (invariant) | an invariant is known and stable | exact, auditable, zero false positives by construction | near zero | hard constraints, regulatory limits | behaviour is novel or rapidly changing | `DERIVED` |
| Robust z (MAD) | unimodal marginal, `MAD > 0` | interpretable per-feature flagging | `O(nd)` | fast screening, small `d` | multi-modal or heavy-tailed features | `DERIVED` |
| Mahalanobis | elliptical cloud, invertible `Σ`, `p << n` | calibrated tail under Gaussianity | `O(nd² + d³)` fit | correlated continuous features | `p` near `n`; mixture populations | `DERIVED` |
| Isolation forest | few anomalies, axis-separable | good ranking, no density estimate | `O(t·ψ log ψ)` fit, `O(t log ψ)` score | mixed types, large `n`, ranking needed | cluster anomalies; strong masking | `SOURCED` mechanism |
| LOF | meaningful local density, scaled features | local adaptation | `O(n²d)` naive | density varies by region | high `d`, tiny `k`, unscaled units | `SOURCED` mechanism |
| One-class SVM | compact normal support, good kernel | boundary-based, controllable `nu` | `O(n²)`–`O(n³)` | small-to-medium `n`, clear normal class | large `n`; unstable `nu`; shifting normal | `SOURCED` mechanism |
| Autoencoder residual | learnable normal manifold, clean training set | high capacity, handles correlations | `O(n d h)` per epoch | high-dimensional signals (sequences, images) | anomalies in training data; no threshold theory | `SOURCED` mechanism |
| Change point (CUSUM) | ordered aggregate, stable pre-period | detects small persistent shifts | `O(n)` streaming | ramps, regime shifts, KPIs | per-event attribution needed | `DERIVED` trace |
| Supervised classifier | enough mature labels | best precision when labels are valid and time-safe | `O(n d)` training per epoch | mature label base exists | labels are review-selected (feedback loop) | `DERIVED` |
| Human review only | judgement required, no model | flexible, handles context | `s` minutes per case | low volume, high stakes | volume exceeds capacity | `ILLUSTRATIVE` |

Two rows deserve emphasis. The explicit-rule row is the cheapest baseline and is routinely skipped: if a policy invariant exists ("refund exceeds original charge"), a rule beats every model on precision and auditability, and the model should be reserved for what the rule does not cover. The supervised row is often the right answer *and* the most dangerous one, because training on review-selected labels learns the reviewer's selection policy as much as the phenomenon.

### 7.2 Selection ladder

```text
IF a hard validity, safety, or regulatory constraint exists
    -> use an explicit rule; do not put a probabilistic model in the control path.
ELSE IF mature, time-safe labels exist for the population being scored
    -> train a supervised model, and evaluate on a time-forward holdout.
ELSE IF the anomaly is an aggregate shift over ordered time
    -> use a change-point detector on the aggregate, with a stable reference mean.
ELSE IF features are mixed-type, n is large, and ranking is the deliverable
    -> isolation forest, contamination set from measured review capacity.
ELSE IF density varies strongly across legitimate regions
    -> LOF with scaled features and a validated k, plus a stability check.
ELSE IF the normal class is compact and n is moderate
    -> one-class SVM with nu chosen by target false positive rate.
ELSE IF the signal is high-dimensional and correlated (sensor, sequence, image)
    -> reconstruction residual with a held-out normal validation set.
ELSE
    -> robust z screen plus human review, and invest in a label/audit sample.
If constraints conflict -> capacity and label maturity win; a method you cannot
evaluate is not a candidate, regardless of its published performance.
```

The ordering encodes one judgement: evaluability beats sophistication. A method whose failure you can observe tomorrow is worth more than a method with better published numbers that you cannot audit this quarter.

### 7.3 Threshold-rule comparison

| Rule | Behaviour under changing volume | Auditability | Failure mode | Use when |
|---|---|---|---|---|
| Fixed score threshold | volume swings with population | high | silent capacity breach | population is stable and monitored |
| Fixed alert rate (quantile) | volume constant, threshold moves | medium | threshold chases noise | capacity is the binding constraint |
| Top-`C` by score | volume exactly `C` | high | ties and stale ranking at the boundary | capacity genuinely fixed |
| Tiered (score band × value) | volume controlled per tier | medium | tier definitions drift | heterogeneous value across alerts |
| Two-stage (screen then rank) | cheap first stage absorbs volume | medium | first-stage miss unrecoverable | feature cost is high |
| Learned policy | adapts | low without monitoring | feedback loop amplification | mature labels and guardrails |

**Takeaway:** choose the method you can evaluate and the threshold rule that matches which constraint actually binds — and never pick a method on published performance alone.

## 8. Failure Modes & Edge Cases

Each entry below follows the same diagnostic shape: symptom, competing causes, the check that discriminates them, a bounded fix, and a verification that would confirm the fix. Provenance is `HYPOTHETICAL` teaching scenario unless stated; none of these is a reported incident from a named organisation.

### ANO-F01 — Alert volume triples overnight

| Element | Content |
|---|---|
| Trigger | daily alert count 60 → 183 while eligible events stay at 12,000 |
| Competing causes | (a) genuine behavioural shift; (b) upstream schema change producing nulls that route to a high-score branch; (c) feature scaling change (unit or precision); (d) model version swap with a different score distribution; (e) eligibility filter widened |
| Discriminating check | cross-tab alert count by `feed_version`, `null_rate`, and `model_version` for the same day. If the alert count is flat within each version and the version mix changed, it is (d) or (e). If nulls rose and alerts concentrate where nulls are, it is (b). Scale change shows as a shifted feature distribution with unchanged rank correlation to the prior day |
| Bounded fix | quarantine the affected feed version and replay the prior 24 h through the same code; do not retune the model until the data path is verified |
| Verification | replay of a known-good day reproduces the previous alert count within sampling variation, and the null-rate and version counters return to baseline |
| Prevention | a fail-closed gate that blocks scoring when null rate or schema hash deviates beyond a threshold, plus a version stamp on every alert |

The order of investigation here is not a stylistic preference. Behavioural explanations are the most expensive to test and the least common; metadata explanations are cheap and common. Checking metadata first costs one query and eliminates three causes.

### ANO-F02 — Precision collapses but nothing changed in the model

| Element | Content |
|---|---|
| Symptom | `p@60` falls from 0.375 to 0.11 across two weeks; score distribution is stable |
| Competing causes | (a) maturity delay: recent cohorts still pending and only easy cases matured; (b) population mix shift toward a segment with lower base rate; (c) reviewer drift (fatigue, policy change, or a new reviewer with different judgement); (d) genuine detector decay on a drifting population |
| Discriminating check | stratified precision by maturity cohort and by reviewer. If precision is stable within every cohort but the mix is young, it is (a). If precision is stable within cohorts and reviewers but the segment mix changed, it is (b). If precision differs by reviewer on the same score band, it is (c) |
| Bounded fix | report precision on a fixed-age cohort; hold the model; investigate reviewer disagreement on a shared sample of 40 alerts |
| Verification | cohort-stratified precision returns to a stable band, and inter-reviewer agreement on the shared sample exceeds the pre-registered threshold |
| Prevention | a maturity cohort table published alongside every yield number, with cohort age as a required column |

### ANO-F03 — Reviewers are overloaded while precision looks fine

| Element | Content |
|---|---|
| Symptom | queue age p90 rises from 40 min to 3 days; reviewer overtime grows; precision unchanged |
| Competing causes | (a) duplicates inflating investigations; (b) capacity reduced by leave or attrition; (c) escalation rework; (d) volume increase hidden by averaging across weekdays |
| Discriminating check | alert-to-entity ratio (duplicates), staff-hours available per day (capacity), reopened-case share (rework), and daily rather than weekly volume |
| Bounded fix | dedupe on entity within the review window, cap daily starts at measured capacity, and publish queue age as a primary metric |
| Verification | daily investigations started ≤ capacity for ten consecutive days, and queue age p90 falls below the SLA |
| Prevention | make queue age and utilisation first-class metrics with alerts, not reporting artefacts |

Queue age is the metric that catches this. Precision does not, because the same alerts eventually get reviewed — just late. Late detection is a distinct business cost from false positives, and it does not appear in any precision table.

### ANO-F04 — Pending adjudications are counted as negatives

| Element | Content |
|---|---|
| Symptom | precision reported as 0.30 in a weekly review, then revised to 0.375 and later to 0.35 |
| Competing causes | (a) denominator policy error; (b) deliberate optimism under reporting pressure; (c) unresolved maturity definition across teams |
| Discriminating check | recompute with three denominators (`k`, mature only, bounded full) and see whether the difference exceeds the decision threshold on the staffing question |
| Bounded fix | publish the bounds `[TP/k, (TP+δ)/k]` and label the mid estimate as conditional on `p_pend` |
| Verification | the next three reports list numerator, denominator, pending count, and bound width |
| Prevention | a reporting template with mandatory fields, and a check that refuses to export without them |

### ANO-F05 — A "cluster" is escalated as a finding

| Element | Content |
|---|---|
| Symptom | 120 alerts form one dense group in a two-feature projection; a stakeholder calls it an organised ring |
| Competing causes | (a) a genuinely distinct operational pattern; (b) projection artefact from a high-dimensional structure; (c) sampling artefact from a new upstream source; (d) duplicated records |
| Discriminating check | stability across seeds, feature subsets, and scaling; provenance of the cluster members (same source system, same timestamp batch); exact-duplicate rate within the group |
| Bounded fix | treat the group as a hypothesis; run one prospective check on new records before any action; do not merge cases into a "ring" case file |
| Verification | the group reconstitutes under a different algorithm and a different window with overlapping membership |
| Prevention | require stability evidence before any segmentation label is used in a narrative |

The phrase "organised ring" is a causal claim. Clustering produces a partition, not a cause; the leap from partition to cause is where most embarrassing escalations originate.

### ANO-F06 — Threshold oscillation

| Element | Content |
|---|---|
| Symptom | the daily alert count swings 35–95 while the population is stable; reviewers cannot plan |
| Competing causes | (a) threshold recomputed from a small window with noisy quantile estimates; (b) ties at the cut; (c) an adaptive rule that reacts to yesterday's review outcomes; (d) a genuine bursty arrival process |
| Discriminating check | hold the threshold fixed for seven days and compare volume variance; compute the binomial SD of the count at the fixed rate, `sqrt(N r (1-r)) = sqrt(12,000×0.005×0.995) = 7.73`, and compare with observed SD |
| Bounded fix | set the threshold from a trailing 28-day window, smooth with a floor and ceiling, and require a change-approval record |
| Verification | observed daily SD approaches 7.73 under a fixed rate; the change log shows each threshold movement with a reason |
| Prevention | versioned threshold policy with a control chart on the alert count itself |

That binomial calculation is a useful control limit: with a stable rate of 0.5% on 12,000 events, a day with 60 alerts has an SD of `sqrt(60×0.995) = 7.73`, so a day with 90 alerts is `+3.9 SD` and worth investigating, while a day with 68 is routine noise that should not trigger a model review.

### ANO-F07 — Feedback loop in the label set

| Element | Content |
|---|---|
| Trigger | a supervised re-ranker trained on past review outcomes keeps converging to the incumbent policy |
| Competing causes | (a) the model really learned the phenomenon; (b) the model learned reviewer selection (only high-score rows have labels); (c) the model learned reviewer bias (a specific reviewer confirms a specific pattern) |
| Discriminating check | train on a random audit sample only and compare rankings; check whether the top-`k` overlap with the incumbent exceeds what an independent signal would produce |
| Bounded fix | maintain a permanent random audit stream (even 200 rows/day) as an unbiased evaluation set; never train on the audit set's future rows |
| Verification | evaluation on the audit stream reproduces the offline precision within its confidence interval |
| Prevention | label-provenance column recording whether a label came from a queue review or an audit sample |

Without a random stream there is no unbiased evaluation set at all, and every reported improvement is conditional on the incumbent's selection. This is the same identification argument as Section 5.7, now applied to model iteration.

### ANO-F08 — Seasonality mistaken for anomaly

| Element | Content |
|---|---|
| Symptom | every Monday shows a score spike; every holiday shows a collapse |
| Competing causes | (a) genuine weekday effect in the phenomenon; (b) volume-driven artefact (more events, same rate); (c) reference window fitted on weekdays only; (d) upstream batch timing |
| Discriminating check | compute the alert *rate* rather than the count by day-of-week; compare per-day-of-week score quantiles; check the reference window's day-of-week composition |
| Bounded fix | fit or normalise per day-of-week, or include day-of-week in the reference stratification |
| Verification | day-of-week residual means are indistinguishable from zero over eight weeks |
| Prevention | a calendar feature or stratified reference as an explicit design decision recorded with the model |

### ANO-F09 — Score leakage through a feature that encodes the outcome

| Element | Content |
|---|---|
| Symptom | offline precision 0.62, live precision 0.14 |
| Competing causes | (a) genuine drift; (b) a feature populated after the outcome (settlement flag, dispute record); (c) label leakage into the reference window; (d) different population between offline and live |
| Discriminating check | per-feature availability timestamp versus event time; ablate each suspicious feature and re-evaluate; compare the offline population's eligibility filter with live |
| Bounded fix | drop post-outcome features, rebuild the reference window with a strict cutoff, re-evaluate on a time-forward holdout |
| Verification | offline precision falls into the live confidence interval, and the leaky feature's contribution drops to noise |
| Prevention | feature contracts with `available_at` timestamps and an automated as-of join test |

### ANO-F10 — Zero-variance feature destroys a distance score

| Element | Content |
|---|---|
| Symptom | every alert has the same top feature contribution; scores look degenerate |
| Competing causes | (a) `MAD = 0` for a constant feature; (b) a unit change making one feature dominate; (c) missing-value sentinel `-1` treated as a real value; (d) standardisation fitted on a window containing the anomaly |
| Discriminating check | per-feature variance and `MAD` table; contribution share of the top feature; sentinel-value frequency |
| Bounded fix | drop or floor zero-variance features, exclude sentinels explicitly, and fit scaling on a clean reference window |
| Verification | contribution shares are within a plausible range across features, and a fixture with a known injected anomaly is detected after the fix |
| Prevention | a pre-fit feature audit that fails the build on zero variance or sentinel dominance |

### ANO-F11 — Capacity quoted in alerts, load measured in investigations

| Element | Content |
|---|---|
| Symptom | the team "has 60 slots" but works 14.7 hours and cannot explain the gap |
| Competing causes | (a) dedupe not applied; (b) service time measured on easy cases only; (c) escalation excluded from the time study; (d) queue age causing rework |
| Discriminating check | time-and-motion sample of 100 investigations with p50/p90; count of escalations per investigation; duplicate rate |
| Bounded fix | restate capacity in investigations using measured `p90` service time, not the mean |
| Verification | planned starts × measured service time matches observed staff-hours within 10% over two weeks |
| Prevention | capacity model stored as a versioned artefact with its measurement date |

### ANO-F12 — Abstentions silently counted as refutations

| Element | Content |
|---|---|
| Symptom | reported precision drifts down over a quarter with no model change |
| Competing causes | (a) a new reviewer abstaining more; (b) evidence retention failures making cases undecidable; (c) genuinely harder alert mix; (d) a policy change adding appeal-stage evidence requirements |
| Discriminating check | abstain rate by reviewer and by week; evidence-completeness rate at case open |
| Bounded fix | count abstain separately in all precision tables; route abstentions to a second reviewer rather than to refuted |
| Verification | precision recomputed with a three-way outcome recovers the prior level, and the abstain rate is published |
| Prevention | schema constraint with three mutually exclusive terminal states and a required reason code |

**Takeaway:** diagnose the layer before tuning the score — most "model decay" is a maturity, dedupe, schema, or selection effect that a two-column cross-tab exposes.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic costs

Sizes: `n` scored events per period; `d` features; `t` trees; `ψ` subsample size per tree (isolation forest); `k` neighbours (LOF); `m` reviewers; `s` service minutes.

| Operation | Input sizes | Time | Space | I/O / communication | Dominant bottleneck |
|---|---|---|---|---|---|
| Feature materialisation | `n`, `d` | `O(n d)` | `O(n d)` | feature-store reads `n·d` values | join latency, not compute |
| MAD screen | `n`, `d` | `O(n d log n)` with sorting per feature | `O(n d)` | one pass | sorting per feature |
| Mahalanobis fit | `n`, `d` | `O(n d² + d³)` | `O(d²)` | one pass + inverse | `d³` inverse when `d` large |
| Mahalanobis score | `n`, `d` | `O(n d²)` | `O(d²)` | none | matrix-vector per row |
| Isolation forest fit | `ψ`, `t` | `O(t ψ log ψ)` | `O(t ψ)` | subsample reads | tree count and `ψ` |
| Isolation forest score | `n`, `t` | `O(n t log ψ)` | `O(t ψ)` | none | per-row tree traversal |
| LOF (naive) | `n`, `d`, `k` | `O(n² d)` | `O(n²)` if full matrix | none | pairwise distance matrix |
| LOF (indexed) | `n`, `d`, `k` | `O(n d log n)` expected | `O(n k)` | index build | index quality in high `d` |
| One-class SVM fit | `n`, `d` | `O(n²)`–`O(n³)` | `O(n²)` kernel | none | kernel matrix size |
| Autoencoder (per epoch) | `n`, `d`, `h` | `O(n d h)` | `O(d h)` params | none | matrix multiply |
| CUSUM update | `T` periods | `O(T)` | `O(1)` | none | none — streaming |
| Review service | `K_e` cases | `O(K_e s)` minutes | `O(K_e)` case records | evidence reads | human time |

The table's practical message: compute is not the constraint at these volumes. At `n = 12,000` per day with `d = 200`, feature materialisation is measured in megabytes while review labour is measured in staff-hours.

### 9b. Back-of-envelope sizing

**Workload.** `n = 12,000` events/day, `d = 200` float32 features, `t = 100` trees, `ψ = 256`, `K_e = 44` investigations/day, `s = 20` min, loaded rate 45/h, `V = 1,800` per confirmed case.

```text
Feature matrix/day      = 12,000 * 200 * 4 bytes
                        = 9,600,000 bytes = 9,375 KiB = 9.155 MiB
Isolation forest nodes  <= 2*psi - 1 = 511 per tree
Node payload            = feature index (2) + threshold (4) + left (4)
                          + right (4) + count (4) = 18 bytes
Model size              = 100 * 511 * 18 = 919,800 bytes = 0.877 MiB
Scoring comparisons/day ~ 12,000 * 100 * 8 (expected depth ~ log2(256))
                        = 9,600,000 comparisons
Review labour/day       = 44 * 20 / 60 = 14.6667 staff-hours
Labour cost/day         = 14.6667 * 45 = 660.00 currency
Annual labour cost      = 365 * 660.00 = 240,900 currency
Annual confirmations    = 18 * 365 = 6,570
Cost per confirmation   = 240,900 / 6,570 = 36.67 currency
Score storage/year      = 12,000 * 8 * 365 = 35,040,000 bytes = 33.4 MiB
Case evidence 2 years   = 44 * 365 * 2 * 1.5 MB = 48,180 MB = 47.0 GiB
```

Every figure above is `ILLUSTRATIVE` input with `VERIFIED` arithmetic. The comparisons figure is not a benchmark: it assumes eight depth comparisons per tree and ignores branch misprediction, memory hierarchy, and batch overhead, so any real timing must be measured.

**Bottleneck.** At this scale the bottleneck is human review and label turnaround, not compute. Doubling feature count doubles materialisation cost to about 18 MiB/day — a rounding error — while doubling alert volume quadruples staffing cost through the queueing nonlinearity of Section 5.6.

**Utilisation sweep (three pooled reviewers, 20-minute service).** This sweep is the quantitative version of the warning in Section 6.6, and it is `DERIVED` from the Erlang C formula with the usual modelling caveats.

| Investigations/day `λ` | Offered load `a` | Utilisation `ρ` | Mean wait `W_q` (min) | Mean residence `W` (min) | Cases in system `L` |
|---:|---:|---:|---:|---:|---:|
| 44 | 1.833 | 0.611 | 6.3 | 26.3 | 2.41 |
| 60 | 2.500 | 0.833 | 28.1 | 48.1 | 6.01 |
| 70 | 2.917 | 0.972 | 227.5 | 247.5 | 36.10 |
| 71 | 2.958 | 0.986 | 467.5 | 487.5 | 72.10 |
| 72 | 3.000 | 1.000 | unbounded | unbounded | unbounded |

Read the table as a warning about marginal headroom: the same three reviewers absorb a 36% volume increase (44 → 60) with a 48-minute residence time, but a further 18% (60 → 71) multiplies residence by ten. The last two rows show why "we can absorb a bit more" is not a valid inference from a headcount table. The `L` column is computed by Little's law and cross-checks every `W_q` in the row, which is the internal consistency test for this kind of sizing.

**Sensitivity.** Three quantities dominate: `K_e` (linear in labour), `p*` (linear in cost, inverse in value), and `rho` (convex in delay). Concretely, from the base case of `660.00` currency per day:

| Perturbation | New value | Change | Consequence |
|---|---:|---:|---|
| investigations +20% (44 → 52.8) | 792.00 | +20% | linear; requires capacity decision |
| service time +20% (20 → 24 min) | 792.00 | +20% | linear in labour, superlinear in delay |
| loaded rate +20% (45 → 54) | 792.00 | +20% | linear; procurement decision |
| `V` −20% (1,800 → 1,440) | `p*` 0.010417 | +25% | fewer alerts clear the value test |
| `c` +20% (15 → 18) | `p*` 0.010000 | +20% | same direction as value loss |

The first three rows are arithmetically identical in cost but completely different in remedy: one is a demand problem, one is a process problem, one is a contract problem. A sensitivity table that reports only the percentage is not actionable; it must name which lever moved. Note also that `p*` is more sensitive to a fall in value than to a rise in cost of the same percentage, because it moves inversely with `V` and directly with `c` while sitting at a small base — a 20% fall in `V` moves `p*` by 25%.

### 9c. Cost of the four error types

| Error type | Mechanism | Illustrative cost expression | Note |
|---|---|---|---|
| False positive reviewed | reviewer time spent on a refuted alert | `c = 15` per investigation | bounded, visible, easy to measure |
| False negative in the queue | a true positive never reviewed | `V = 1,800` foregone per case, times count | unbounded and unmeasured without an audit sample |
| Delay on a true positive | value decays with time | `f(age)` unknown; must be estimated per domain | often the largest term and the least measured |
| Suppression without a record | an alert never reaches review | auditability loss; no direct cost model | a governance defect, not a modelling one |

The asymmetry is the point: the cheap error is measured obsessively while the expensive one is estimated by assumption. Any triage review that reports only precision is reporting the small term.

### 9d. Cost per outcome, stated honestly

`Cost per confirmed case = total labour cost / confirmations`. In the worked example, `660/18 = 36.67` per day and `240,900/6,570 = 36.67` annualised — consistent because both numerator and denominator scale with 365. Two caveats must travel with the number: confirmations are capped by review capacity (so the denominator is a property of the triage policy, not of the phenomenon), and value `V` is assumed, not measured. Presenting `36.67` as "cost per case found" without those caveats overstates precision by roughly the same factor as the identification problem in Section 5.7.

**Takeaway:** at realistic triage volumes the algorithm is cheap and the denominator is expensive — measure the service time and the audit sample before optimising the model.

## 10. Key Theorems / Results Reference

| Result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Source |
|---|---|---|---|---|---|
| Neyman–Pearson | For a fixed false positive rate, the likelihood-ratio test maximises power | simple null vs simple alternative; known distributions | ranking by density ratio is optimal *for its assumed densities* | that an unsupervised score is a density ratio | Neyman & Pearson (1933) |
| Bayes cost rule | Decide positive iff `p(x)/p_0(x) > (c_FP/c_FN)·((1−π)/π)` | known likelihoods and prior; known costs | explains why the optimal cut moves with prevalence and cost | that costs are known in practice | standard decision theory |
| Precision identity | `precision = TPR·π/(TPR·π + FPR·(1−π))` | single threshold, both classes | precision is prevalence-bound; AUC is not | that TPR is estimable from reviewed alerts | `DERIVED` |
| Hypergeometric baseline | Under random ranking `E[TP_k] = kπ`, `Var = kπ(1−π)(N−k)/(N−1)` | exchangeable draws without replacement | gives the null for a lift claim | that `π` is known | `DERIVED` |
| Hoeffding bound | `P(|p̂ − p| ≥ ε) ≤ 2exp(−2kε²)` | independent bounded observations | sample-size planning for precision claims | tight intervals at small `k` | Hoeffding (1963) |
| Rank invariance | Strictly increasing `g` preserves every prefix and `p@k` | strict monotonicity | normalisation is cosmetic for decisions | score comparability across fits | `DERIVED` |
| Little's law | `L = λW` in steady state | stable system, consistent units | fastest backlog consistency check | any distributional detail | Little (1961) |
| Erlang C | `W_q = C(m,a)·s/(m−a)` with the stated recursion | Poisson arrivals, exponential service, pooled queue | exposes convexity of delay in utilisation | real SLA for bursty, skewed service | Erlang (1917) |
| Isolation normalisation | `c(n) = 2H(n−1) − 2(n−1)/n` for expected unsuccessful-search depth | random binary tree over `n` points | calibrates the score to (0,1] with 0.5 at average depth | anomaly probability | Liu, Ting & Zhou (2008) |
| Mahalanobis tail | `D² ~ χ²_p` for Gaussian `x` | multivariate normality, invertible `Σ` | principled threshold at a stated tail probability | validity for mixture or heavy-tailed data | standard multivariate statistics |
| AUC as probability | `AUC = P(a_pos > a_neg)` | continuous scores, independent pairs | prevalence-free summary | usefulness at a fixed capacity | Mann–Whitney identity |
| LOF definition | `LOF_k(x) = mean lrd(o)/lrd(x)` over `k` neighbours | meaningful local density | detects local, not global, unusualness | that ratio magnitude is a probability | Breunig et al. (2000) |

### 10.1 Derivation sketch: the Bayes cost threshold

Suppose the loss of a false positive is `c_FP` and of a false negative is `c_FN`, and the prior positive probability is `π`. The expected loss of declaring positive is `c_FP·(1−π)·p(x|0)`; the expected loss of declaring negative is `c_FN·π·p(x|1)`. Declaring positive is optimal when the first is smaller:

```text
c_FP * (1-pi) * p(x|0) < c_FN * pi * p(x|1)
=> p(x|1)/p(x|0) > (c_FP/c_FN) * ((1-pi)/pi)

EXAMPLE: c_FP = 15, c_FN = 1,800, pi = 0.0015
ratio = (15/1800) * (0.9985/0.0015)
      = 0.008333 * 665.667
      = 5.547
```

Interpretation: the likelihood ratio must exceed `5.547` before a positive decision is worth its cost. This is the same content as the break-even probability in ANO-09 but expressed on the likelihood-ratio scale, and the two must agree: `p* = c/(c+V)` in probability terms, `5.547` in ratio terms, both implying a low posterior threshold consistent with the earlier `0.83%`.

### 10.2 Derivation sketch: the Hoeffding sample size

Setting the right-hand side of the bound to `α` and solving for `k`:

```text
2*exp(-2k*eps^2) = alpha
=> exp(-2k*eps^2) = alpha/2
=> -2k*eps^2 = ln(alpha/2)
=> k = ln(2/alpha) / (2*eps^2)

eps = 0.10, alpha = 0.05:
k = ln(40)/(2*0.01) = 3.6889/0.02 = 184.4 -> 185 mature reviews

eps = 0.05, alpha = 0.05:
k = 3.6889/(2*0.0025) = 3.6889/0.005 = 737.8 -> 738 mature reviews
```

With 48 mature reviews per day, `185` is reachable in four days and `738` in sixteen. That is the honest answer to "how long before we can trust this precision number", and it is far longer than most reporting cycles assume.

### 10.3 What these results do not license

None of these results justifies treating a score as a probability, and none of them estimates recall. The Neyman–Pearson result is conditional on knowing both densities; an isolation forest supplies neither. The chi-square tail is exact only under Gaussianity. The Erlang formula is a model of a queue, not a warranty on a service level. Each row's "what it does not imply" column exists because each of these results is routinely over-claimed in one specific direction.

**Takeaway:** the useful theorems here are rank-invariance, the precision identity, the hypergeometric null, and Hoeffding — they bound what a triage report can and cannot claim.

## 11. Interview / Exam Questions

**Q1 (recall).** A team reports "precision at 60 alerts is 0.30". What three facts must accompany that number before it can be interpreted?

*Checked answer:* numerator and denominator policy (mature count, not `k`, if any alert is pending), the window with its eligibility filter and model version, and the review capacity that set `k`. *Reasoning:* `0.30` is compatible with `18/60`, `18/60` where 12 were pending, or `18/48` mature-only, and the staffing consequence differs across these. *Red flag:* the speaker cannot say whether pending alerts were excluded.

**Q2 (calculation).** `N = 12,000`, capacity `C = 60`, and the score is continuous. At what empirical percentile must the threshold sit, and what is the alert rate?

*Checked answer:* the 99.5th percentile; rate `60/12,000 = 0.005`. *Reasoning:* capacity fixes the tail area, and `1 − 0.005 = 0.995` is the quantile. *Follow-up:* what changes if scores are discrete and `a_(60) = a_(61)`? *Red flag:* answering "0.5%" without connecting it to capacity.

**Q3 (application).** A reviewer claims "these 120 alerts in one cluster prove an organised ring". What is the first professional response?

*Checked answer:* ask for stability evidence (seed, feature subset, scaling, window) and provenance (same source system, same timestamp batch, duplicate rate). *Reasoning:* a cluster is a partition, and a ring is a causal claim; partitions reproduce under reasonable perturbations only if the structure is real. *Red flag:* accepting the narrative because it is vivid.

**Q4 (calculation).** Service time is 20 minutes, loaded rate 45/h, and 44 investigations per day. Compute staff-hours, daily labour cost, and cost per confirmation with 18 confirmations.

*Checked answer:* `44×20/60 = 14.667` h; `14.667×45 = 660`; `660/18 = 36.67` per confirmation. *Reasoning:* capacity is consumed per investigation, so dedupe precedes the calculation. *Follow-up:* what if the pending 12 resolve with three more confirmations? Cost falls to `660/21 = 31.43`. *Red flag:* multiplying 60 alerts by service time when only 44 investigations exist.

**Q5 (judgement).** Offline precision is 0.62 and live precision is 0.14. Name the three most likely causes and one check each.

*Checked answer:* (a) post-outcome feature — check `available_at` timestamps against event time; (b) population mismatch — compare eligibility filters and segment mix; (c) reference-window leakage — rebuild with a strict cutoff and re-evaluate time-forward. *Reasoning:* leakage and population differences explain large gaps; genuine drift rarely produces a 4× gap in one step. *Red flag:* immediately proposing a better model.

**Q6 (application).** Why can precision not be estimated from reviewed alerts alone when recall is unknown?

*Checked answer:* because the reviewed set is selected by the score, so it contains no information about positives below the cut. *Reasoning:* in the worked example, precision `0.30` at a 0.5% review rate implies perfect recall, which is implausible; the resolution is that `π` is unknown and only an unreviewed sample identifies it. *Red flag:* quoting recall computed as `confirmed/reviewed`.

**Q7 (calculation).** Estimate the number of mature reviews needed to claim precision `±0.10` at 95% confidence using Hoeffding.

*Checked answer:* `ln(2/0.05)/(2×0.01) = 184.4 → 185`. *Reasoning:* bounded independent observations, so Hoeffding applies directly; the bound is conservative but the order is right. *Follow-up:* `±0.05` needs `738`, which is why weekly precision claims on 48 reviews are noise. *Red flag:* quoting a precision to two decimals from 30 observations.

**Q8 (diagnosis).** Alert volume triples overnight with stable event volume. Give an ordered diagnostic sequence.

*Checked answer:* (1) null rate and schema hash by feed version; (2) model version and score quantiles; (3) eligibility filter row counts; (4) feature scaling and sentinel values; (5) behavioural explanation last. *Reasoning:* metadata causes are cheap to test and much more common than behavioural shifts of that magnitude. *Red flag:* starting with a hypothesis about attacker behaviour.

**Q9 (selection).** When is LOF a better choice than an isolation forest, and what must be true for LOF to work?

*Checked answer:* LOF when density varies strongly across regions and the question is "unusual relative to neighbours"; isolation forest when the question is global separability and ranking is the deliverable. LOF requires scaled features, a validated `k`, and meaningful neighbourhood distances. *Reasoning:* LOF is a density ratio, so a uniformly sparse region yields `LOF ≈ 1` even for globally extreme points. *Red flag:* applying LOF with unscaled mixed units.

**Q10 (calculation).** Utilisation is `0.83` with one server and mean service 20 minutes. Why is a two-hour delay plausible?

*Checked answer:* because wait time is convex in utilisation; at `rho = 0.83` the `M/M/1` mean wait is `rho*s/(1−rho) = 0.83×20/0.17 = 97.6` minutes, plus 20 minutes of service. *Reasoning:* `W_q = rho·s/(1−rho)` follows from Little's law applied to the queue. *Follow-up:* pooling three reviewers instead of one changes the curve dramatically (Section 5.6). *Red flag:* treating 83% utilisation as "17% spare, so 17% more work fits".

**Q11 (judgement).** Precision at the cut is 0.35 and the break-even probability is 0.0083. What is the correct recommendation?

*Checked answer:* the binding constraint is capacity, not ranking; the organisation should expand review capacity (or tier the queue by value) rather than invest first in a better model. *Reasoning:* the margin between 0.35 and 0.0083 is roughly 42×, so nearly every alert is economically worth reviewing. *Red flag:* recommending a model upgrade when the arithmetic says headcount.

**Q12 (application).** What does contamination mean, and what happens if it is set from an assumed fraud rate?

*Checked answer:* contamination is the assumed outlier fraction used at fit time; setting it from an assumed rate makes the model's selected volume an artefact of that assumption rather than a measurement. *Reasoning:* the parameter changes the fit's notion of the boundary, so a wrong value silently shifts the score distribution. *Red flag:* treating contamination as the observed alert rate.

**Q13 (diagnosis).** Precision falls from 0.375 to 0.11 in two weeks with a stable score distribution. What is the first check?

*Checked answer:* maturity-cohort stratification — how old are the decided cases, and is precision stable within cohorts? *Reasoning:* early-decided cases skew easy or hard, and a young cohort mix produces a fake decline. *Follow-up:* reviewer-stratified precision on a shared sample is the second check. *Red flag:* refitting the model.

**Q14 (judgement).** An audit sample of 400 unreviewed rows finds 2 positives. What can you claim?

*Checked answer:* an estimated missed positive count of about `60` with a very wide interval (roughly 0–142), so recall is bounded near `0.11–1.00` and no decision-grade recall claim is possible. *Reasoning:* binomial standard error `0.0035` on a rate of `0.005` gives a near-degenerate interval at this sample size. *Follow-up:* stratify the audit and increase `n` in the high-score band below the cut. *Red flag:* reporting `recall ≈ 23%` as if it were measured.

**Q15 (judgement).** You must choose between an interpretable robust z-score and a supervised model with a 4× higher offline precision, trained on review outcomes. What decides it?

*Checked answer:* whether an unbiased evaluation set exists. If all labels come from the review queue, the supervised model's offline advantage may partly encode the incumbent selection policy; without a random audit stream, the comparison is not identified. *Reasoning:* this is the feedback loop of ANO-F07, and it is the dominant risk in triage settings with scarce labels. *Red flag:* accepting offline precision as evidence when the label provenance is one-directional.

**Takeaway:** the questions that separate candidates are about denominators, prevalence, and identification — not about which algorithm has the better paper.

## 12. Quick Reference Card

| Trigger | Rule | Assumption | Avoid | Verification | Owner |
|---|---|---|---|---|---|
| capacity known, volume stable | cut at the `(1 − C/N)` empirical quantile | continuous scores, one version | copying a threshold across versions or segments | alert count equals `C` for five days | operations |
| alerts reported as yield | `p@k` on mature labels; state numerator, denominator, pending | maturity definition agreed | pending treated as negative | cohort table published with the number | risk analytics |
| precision looks low | check `p* = c/V` before blaming ranking | constant review cost and value | model rebuild when capacity binds | `p*` recomputed and compared with observed precision | decision owner |
| alert spike | audit null rate, schema hash, feed version, model version first | version stamps exist on alerts | behavioural narrative before metadata check | known-good day replays to the prior count | data engineering |
| queue age rising | utilisation `ρ = λs/m`, wait via Erlang, backlog via Little | pooled FIFO queue, stable rates | averaging volume over weekdays | measured backlog matches `L = λW` | operations |
| new category flagged | check launch/change log before acting | provenance log exists | permanent allowlist without expiry | sampled review after expiry | risk analytics |
| duplicate alerts | dedupe on entity within the window; count investigations | stable entity key | capacity quoted in alerts | alert-to-entity ratio reported daily | operations |
| cluster escalated | test stability across seed, features, scaling | reproducible pipeline | calling a partition a cause | membership overlap across re-runs | risk analytics |
| recall requested | stratified audit of unreviewed rows; report interval | audit sampling design pre-registered | recall from reviewed alerts only | interval width stated with the estimate | risk analytics |
| scores normalised | treat as reporting only; decisions invariant to monotone maps | strict monotonicity | presenting normalisation as improvement | same top-`k` before and after | risk analytics |

Final checks before this card is used to justify a decision: capacity is expressed in investigations after dedupe; every precision statement names its denominator and pending count; `p*` is computed from current cost and value; the alert-to-entity ratio is published; queue age and utilisation are visible; an unreviewed audit stream exists or the recall claim is marked unidentified; and the model version, reference window, and eligibility filter are stamped on the alert record. If any of these is missing, the correct action is to collect it — not to choose a different algorithm.
