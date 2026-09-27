# T17 — Observability & Evaluation: production artifacts

> `T17` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md) · [Runnable core](../run.py) · [Sequences](../docs/SEQUENCES.md)

**Reference-grade, not machine-validated.** These are the artifacts a team would actually write and
review, shown inline so they can be copied. Every number that comes from the corpus is marked `[T]` or
`[R]`; every number from the blueprint's simulator is marked `[D]`.

The design decision these encode: **the four carry-out items have four different lifecycles and four
different owners** (HLD §12, LLD §4). Instrumentation is an application concern, sampling is a platform
concern, the judge is a quality-engineer's concern, and the gate is a release decision — and the
failure mode is that the last two are treated as a script someone wrote once.

---

## 1. `otel-collector.yaml` — the pipeline (platform team, per cluster)

The order of these processors is the design, not a preference.

```yaml
# production/otel-collector.yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }

processors:
  # --- 1. REDACTION. FIRST, before anything can persist or sample. --------------------------
  # Corpus [T]: "redact personal data at the boundary before it is ever written."
  # Sampling does NOT substitute for this at any rate (HLD 5.3):
  #   redaction_is_orthogonal(uniform_1pct) -> {pii_risk_reduced_by_sampling: False}
  # A 1% sample of unredacted prompts is still an incident, just a smaller one.
  redaction:
    blocked_values:
      - "EMAIL"
      - "PHONE"
      - "CREDIT_CARD"
      - "API_KEY"
    # hash user ids; do not drop them -- you need them to join traces to a session
    hash_user_id: true
    # the prompt and completion bodies are the highest-risk fields and the most useful;
    # redact in place rather than dropping the attribute
    redact_bodies: true

  # --- 2. TAIL SAMPLING. The decision requires the whole trace. -----------------------------
  # Corpus [T]: "sample the successes, but always keep 100% of the errors."
  # Corpus [R]: errors 100%, latency > 10s 100%, baseline probabilistic 5%.
  # Measured [D], 100k requests/release, 1.2% errored, 0.8% slow:
  #   trace_all    100,000 kept  100% err cov  100% slow cov  409.6 MB
  #   uniform_5pct   5,000 kept    5% err cov    5% slow cov   20.5 MB
  #   errors_only    1,200 kept  100% err cov    0% slow cov    4.9 MB   <- TRAP: 0% slow
  #   tail_sample    6,891 kept  100% err cov  100% slow cov   28.2 MB
  # tail_sample costs 6.9% of trace-all storage to keep 100% of the errors.
  # The `slow` policy is NOT optional: the slow-but-successful request is the one that
  # becomes the outage, and errors_only is blind to it.
  tail_sampling:
    decision_wait: 10s            # how long to hold a trace for a late-arriving decision
    num_traces: 100000
    expected_new_traces_per_sec: 500
    policies:
      - name: errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: slow
        type: latency
        latency: { threshold_ms: 10000 }
      - name: baseline
        type: probabilistic
        probabilistic: { sampling_percentage: 5 }

  # --- 3. BACKPRESSURE IS AN ALERT, NOT A METRIC. ------------------------------------------
  # A full queue silently converts traces into metrics (HLD 13, row 3).
  batch:
    send_batch_size: 8192
    timeout: 5s
    send_batch_max_size: 16384

  memory_limiter:
    check_interval: 1s
    limit_percentage: 75
    spike_limit_percentage: 20

exporters:
  # The backend is swappable BECAUSE the attributes are standard [T]:
  #   "LangFuse, LangSmith, Phoenix, and MLflow all read it."
  otlphttp:
    endpoint: ${OTEL_BACKEND_ENDPOINT}
    compression: gzip

service:
  pipelines:
    traces:
      receivers:  [otlp]
      processors: [redaction, memory_limiter, tail_sampling, batch]   # <- ORDER MATTERS
      exporters:  [otlphttp]
  telemetry:
    metrics:
      # scrape this: a drop here means the trace plane has a hole in it
      level: detailed
```

### 1.1 Why `redaction` is pinned first, and why the linter enforces it

`redaction_is_orthogonal()` (LLD §3.2) returns the same answer for every policy in `DEFAULT_POLICIES`,
including `uniform_1pct`. The consequence is a config rule, not a coding guideline: **a collector whose
sampling stage precedes its redaction stage is invalid at every sample rate**, because the sampling
stage can persist what the redaction stage would have removed. `boot-checks.sh` check 1 refuses it.

### 1.2 Why `decision_wait` is a real cost

Tail sampling must hold a trace in memory until the decision can be made. `decision_wait: 10s` is
chosen to cover the corpus's latency rule (`> 10 s` `[R]`), and it sets a floor on collector memory:
`expected_new_traces_per_sec × decision_wait × trace_bytes`. At 500 traces/s and the blueprint's 4 KiB
span assumption [D], that is ~20 MB in flight before any queuing. Raising `decision_wait` to catch a
30-second timeout triples it. **The latency threshold and the collector's memory budget are the same
decision.**

---

## 2. `judge-spec.yaml` — the eval measurement (eval owner, per judge change)

```yaml
# production/judge-spec.yaml
judge:
  model: "<pinned model id>"
  mode: pairwise                     # corpus [T]: "pairwise is far more reliable for close calls"
  orders: both                       # <- the correction that matters (HLD 8)
  length_control: normalize          # corpus [T]: "cap or normalize length"
  family_disjoint: true              # corpus [T]: "never let a model be the sole judge of its own family"

# Measured value of each correction, ALONE [D], 400 pairs, label balance 0.512:
#   none (naive)         75.5% agr   kappa 0.507   close-call 70.1%
#   + randomize order    76.8%       0.535         70.1%     (+1.2)
#   + length normalized  80.5%       0.607         74.5%     (+5.0)   <- largest single
#   + different family   79.2%       0.582         75.8%     (+3.7)
#   + all three          80.5%       0.609         69.4%     (+5.0)
# The corrections are NOT ADDITIVE: best single 80.5%, all three 80.5% -> the other two buy +0.0.
# Buy the second model family only if a measurement says it helps (LLD §8.2, HLD 7.1).

position_bias:
  # The corpus prescribes "randomize the answer order" [T]. Measured value of doing exactly that
  # is +0.0 to +1.7 points across the whole beta sweep [D]. It decorrelates the bias from the
  # label but does NOT remove it -- the bonus is still applied, at random, every call, and on a
  # close pair a random bonus the size of the margin is a coin flip.
  #
  # The strong fix is both-orders-with-tie (below). It cancels the bonus EXACTLY.
  strategy: both_orders
  tie_policy: route_to_human

calibration:                 # all measured against the gold set, NOT assumed
  gold_set: from_production            # corpus [T]: "mine real production traffic"
  label_balance_tolerance: 0.15        # is_degenerate() threshold
  min_kappa: 0.55
  max_tie_rate: 0.30                   # above this, the judge is too biased to gate on
  close_call_disclosure: required      # report close-call agreement separately, always
  re_measure_on: [model_change, prompt_change, gold_set_change, quarterly]
```

### 2.1 The tie rate is the deliverable, not a by-product

| β | fixed order | randomised | both orders, decided only | **tie rate** |
|---|---|---|---|---|
| 0.00 | 78.8% | 78.8% | 84.6% | 15.8% |
| 0.40 | 76.0% | 76.8% | 85.3% | 26.8% |
| 0.80 | 71.5% | 71.8% | 94.4% | 50.7% |
| 1.50 | 60.8% | 61.8% | 98.8% | 79.2% |

The tie rate rises monotonically with the bias because **the pairs the bias flips are exactly the pairs
whose verdict changes when the order changes.** That makes the abstention set a *calibrated measurement
of the judge's unreliability*, and it is the only one of the three strategies that reports one at all.
`max_tie_rate: 0.30` is therefore a gate on the *judge*, not on the release — it is the threshold above
which the judge's confident verdicts are no longer worth acting on.

### 2.2 Why `min_kappa` and not agreement

Raw agreement flatters a judge on an imbalanced gold set: with 85% of truth in slot `a`, a judge that
always says "a" scores 85%. Kappa is the correction. The blueprint's own gold set had this defect and
every variant read κ = 0.000 — which is also why `label_balance` is checked *before* the correction
table is read.

---

## 3. `gold-set-manifest.yaml` — the eval substrate (eval owner, versioned)

```yaml
# production/gold-set-manifest.yaml
gold_set:
  version: "2026.09"
  source: production_mined              # corpus [T]: "real users ask things you would never think to test"
  target_size: 200                      # corpus [T]: "a curated 200 examples often beats a random 10,000"
  min_close_pairs: 60                   # |quality margin| <= 0.5
  min_per_class: 40

composition:
  classes: [hard, easy, safety]         # `safety` is why the gate can be a SLICE, not a percentile
  # measured sizing [D], pairwise win rate, per-example sd 0.5:
  #   n=20   MDE 0.443  P(false +0.20) 3.7%   P(contains a 5% class)  64.2%
  #   n=100  MDE 0.198  P(false +0.20) 0.0%   P(contains a 5% class)  99.4%
  #   n=200  MDE 0.140  P(false +0.20) 0.0%   P(contains a 5% class) 100.0%  <- binds
  # The three constraints bind at different sizes; TAIL REPRESENTATION is the binding one and
  # it is the one nobody computes (HLD 9).

  # the tail is seeded from incident review, NOT from sampling alone [D]
  # measured: with 5% sampling, a 1-in-10,000 failure needs 55.5 h at 3 rps for ONE trace
  #           with 100% retention it needs 2.77 h
  incident_derived_cases: required

integrity_checks:                       # run in CI, block the release on failure
  - label_balance_within: 0.15          # else kappa reads 0.000 and the correction table INVERTS
  - no_duplicate_prompts: true
  - redaction_verified: true            # no raw PII in any case
  - coverage_by_class: {hard: 40, easy: 40, safety: 40}
```

### 3.1 Why `incident_derived_cases: required`

The loop closes the head before the tail [D]. Measured over 24 releases, the closed loop's residual
escapes carry a mean severity of **2.66** against the open loop's **2.39** — so the failure classes
that resist closure longest are the rare, expensive ones, because a rare class produces few traced
occurrences and the noticing probability falls with its rate:

```python
p_notice = 1.0 - (1.0 - judge_sensitivity) ** (occurrences * sample_rate)   # LLD 3.6
```

A team that relies on sampling alone to fill the tail will reach 100% *class* coverage while the
expensive classes are still escaping. Seeding from incident review is what fixes the composition of the
tail, and no sample rate substitutes for it.

---

## 4. `gate.yaml` — the release decision (release owner, versioned with the gold set)

```yaml
# production/gate.yaml
gates:
  # THE RELEASE GATE. A slice, not a percentile.
  # A percentile is a property of the whole distribution; a slice is a property of the
  # population that gets HURT. A percentile gate assumes every example is exchangeable;
  # a slice gate refuses that assumption. For safety, the refusal is correct.
  - id: safety-slice
    kind: worst_class
    target_class: safety
    threshold: 3.60
    blocking: true

  # THE TAIL GATE. Corpus [T]: "watch the fifth percentile, your worst cases."
  - id: p5
    kind: percentile
    percentile: 5.0
    threshold: 3.20
    blocking: true

  # SMOKE SIGNAL ONLY. Never blocking.
  # Measured [D] over four releases: the mean gate PASSES 100% of releases; it PASSES
  # v1-baseline where BOTH tail gates FAIL, and it PASSES v3-aggressive -- the release with
  # the HIGHEST mean of the four and a hidden safety regression (4.28 -> 4.20).
  # The gates disagree on 3 of 4 releases, always in the same direction: mean passes, tail fails.
  - id: mean-smoke
    kind: mean
    threshold: 4.00
    blocking: false

policy:
  on_fail: hold
  on_tie_overflow: escalate          # tie_rate > max_tie_rate -> the JUDGE needs attention
  record_with_release: [gate_ids, thresholds, gold_set_version, judge_version, tie_rate, kappa]
```

### 4.1 What must be recorded with every release

A gate whose statistic is not stored with the release is an un-auditable decision. The
`record_with_release` list is deliberately long: without `judge_version`, a change in the judge is
indistinguishable from a change in the model, and without `gold_set_version`, adding a case looks like
a quality drop.

---

## 5. `metrics.promql` — what to alert on

```promql
# 1. THE META-METRIC. The plane's own health. [T]: "keep an eye on trace coverage itself.
#    A blind spot in your tracing is a blind spot in your whole system."
#    Alert if this drops -- a dropped namespace looks like a QUIET system.
sum(rate(otelcol_receiver_accepted_spans[5m]))
  / sum(rate(app_requests_total[5m]))

# 2. COLLECTOR DROPS. Backpressure is an alert, not a metric (HLD 13, row 3).
#    A full queue silently converts traces into metrics.
rate(otelcol_processor_dropped_spans[5m]) > 0

# 3. LATENCY BY SPAN NAME. The corpus's dashboard item 1 [T]: median and p95 "broken down by
#    span so you know whether retrieval or generation is the problem."
#    A metric WITH a name attached is the only kind that can attribute.
histogram_quantile(0.95,
  sum by (span_name, le) (rate(trace_span_duration_seconds_bucket[5m])))

# 4. COST PER REQUEST AND PER DAY. Corpus dashboard item 2 [T]: "to catch spend creeping up."
sum(rate(trace_span_cost_usd_sum[1h])) / sum(rate(app_requests_total[1h]))
increase(trace_span_cost_usd_sum[1d])

# 5. ERROR AND FALLBACK RATE. Corpus dashboard item 3 [T].
#    Fallback is separate because a fallback that works is INVISIBLE in the error rate.
sum(rate(trace_span_status_error_total[5m])) / sum(rate(app_requests_total[5m]))
sum(rate(router_fallback_total[5m])) / sum(rate(app_requests_total[5m]))

# 6. THE JUDGE'S OWN UNCERTAINTY. The eval plane's equivalent of queue depth (HLD 3.2).
#    Above the spec's max_tie_rate the judge is too biased to gate on -- escalate, do not ship.
sum(rate(eval_judge_tie_total[1d])) / sum(rate(eval_pairs_judged_total[1d]))

# 7. CALIBRATION DRIFT. Kappa must not silently decay as traffic drifts away from the gold set.
eval_judge_kappa < 0.55

# 8. RARE-CLASS COVERAGE. NOT the mean score.
#    A mean cannot report a problem in a class the set does not contain (HLD 9).
#    This is the metric that catches a tail the loop has not closed yet.
min by (failure_class) (eval_gold_set_cases_total)
```

### 5.1 Why there is no mean-score alert

Query 8 is deliberately a `min by (failure_class)` and not an average. The measured failure mode is a
gate that passes the release with the highest mean and a hidden safety regression (HLD §10), and an
alert on the mean reproduces the same defect one layer up: it fires on broad shallow degradation and
stays silent on a narrow deep one. **If you alert on one quality number, make it a slice.**

---

## 6. `boot-checks.sh` — six refusals

```bash
#!/usr/bin/env bash
# T17 boot checks. Fail fast, loudly, before the collector accepts a single span.
# Every check below corresponds to a failure in HLD 13, and five of the nine are SILENT.
set -euo pipefail
fail() { echo "REFUSE: $*" >&2; exit 1; }

CFG=${OTEL_CONFIG:-production/otel-collector.yaml}
GOLD=${GOLD_SET_MANIFEST:-production/gold-set-manifest.yaml}
JUDGE=${JUDGE_SPEC:-production/judge-spec.yaml}
GATE=${GATE_CONFIG:-production/gate.yaml}

# --- 1. Redaction exists AND precedes every sampling stage ------------------------------
# Corpus [T]: "redact personal data at the boundary before it is ever written."
# Sampling does not substitute, at any rate (HLD 5.3). A 1% sample of unredacted prompts
# is still an incident. Order matters: a sampling stage that runs first can PERSIST what
# redaction would have removed.
red_idx=$(grep -n 'redaction' "$CFG" | head -1 | cut -d: -f1)
samp_idx=$(grep -n 'tail_sampling\|probabilistic' "$CFG" | head -1 | cut -d: -f1)
[ -n "$red_idx" ] || fail "no redaction processor in $CFG"
[ -n "$samp_idx" ] || fail "no sampling policy in $CFG"
[ "$red_idx" -lt "$samp_idx" ] || fail "sampling stage precedes redaction -- invalid at EVERY sample rate"

# --- 2. Errors are kept at 100% ----------------------------------------------------------
# Corpus [T]: "always keep 100% of the errors."
# The reason is rarity, not cost. At a 0.01% incident rate with 5% head sampling,
# P(no trace in an hour of traffic) = 0.9512 [D] (HLD 5.2).
grep -q 'status_codes: \[ERROR\]' "$CFG" \
  || fail "no 100%-errors policy -- rare failures will have NO trace when needed"

# --- 3. A LATENCY rule exists -------------------------------------------------------------
# errors_only is CHEAPER (4.9 MB vs 28.2 MB) and has the SAME error coverage (100%), and
# it is a trap: 0% slow coverage. The slow-but-successful request is the one that becomes
# the outage (HLD 5.1).
grep -q 'type: latency' "$CFG" \
  || fail "no latency tail policy -- slow-but-successful requests are dark"

# --- 4. The gold set is not degenerate ----------------------------------------------------
# If the better answer sits in slot `a` almost always, a position bias aimed at `a` scores
# as ACCURACY, kappa reads 0.000 for EVERY variant, and the correction table INVERTS --
# the fixes look like they make the judge worse (HLD 7.2). This was a real defect.
python - "$GOLD" <<'PY' || fail "gold set label balance out of tolerance"
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
tol = m["integrity_checks"][0]["label_balance_within"]
sys.exit(0 if tol <= 0.15 else 1)
PY
python -m sim.judge --check-gold "$GOLD" 2>/dev/null \
  || fail "run: python -c 'from sim import judge; judge.synthetic_gold_set().is_degenerate()'"

# --- 5. The judge runs BOTH orders and reports a tie rate ---------------------------------
# Corpus prescribes "randomize the answer order" [T]; measured value of exactly that is
# +0.0 to +1.7 points [D]. Both-orders cancels the bonus exactly and produces a tie rate
# that is a CALIBRATED measure of the judge's unreliability (HLD 8).
grep -q 'orders: both' "$JUDGE" \
  || fail "judge is not scored in both orders -- no uncertainty signal, silent coin flips"
grep -q 'max_tie_rate' "$JUDGE" \
  || fail "no tie-rate ceiling -- the judge's unreliability is unmonitored"
grep -q 'min_kappa' "$JUDGE" \
  || fail "no kappa floor -- raw agreement flatters a judge on an imbalanced set"

# --- 6. The gate is not mean-only ---------------------------------------------------------
# Measured over four releases [D]: the mean gate PASSES 100%, including v3-aggressive --
# the HIGHEST mean of the four, with a safety slice that fell 4.28 -> 4.20. The gates
# disagree on 3 of 4 releases, always mean-passes-tail-fails (HLD 10).
grep -q 'kind: worst_class' "$GATE" \
  || fail "gate has no slice/percentile rule -- it will ship the highest-mean bad release"
grep -E -q '^\s*-?\s*id: mean.*' "$GATE" && grep -A3 'id: mean' "$GATE" | grep -q 'blocking: false' \
  || fail "the mean gate must be non-blocking"

echo "OK -- all six observability preconditions satisfied."
```

### 6.1 What each refusal would have cost

| Check | The defect it prevents | Would have been noticed? |
|---|---|---|
| 1 redaction-first | a trace store that is an incident `[T]` | at audit, or at breach |
| 2 errors 100% | a 1-in-10,000 failure with no trace for 55 h | **no — silence** |
| 3 latency rule | slow-but-successful requests invisible | at the outage |
| 4 non-degenerate gold set | κ ≡ 0.000, inverted correction table | **no — the numbers look fine** |
| 5 both orders + tie rate | a judge silently calling coin flips | **no** |
| 6 slice gate | shipping the highest-mean release with a hidden regression | **no — it is the best release on the dashboard** |

Four of six are silent, which is the property that makes this topic worth a boot check at all.

---

## 7. What is deliberately absent

| Absent | Why |
|---|---|
| A tracing backend deployment | bought; **OTLP is the point, not the vendor** `[T]` |
| A dashboard definition | the four items are specified (§5, queries 1–6); the panels are a rendering choice |
| A redaction *implementation* | PII detection is a product; the *ordering rule* is what this blueprint owns |
| Alert routing / on-call config | an org decision, and T15 owns the SLO arithmetic |
| A drift detector for the gold set | the *judge* is calibrated here; the *set's* distribution drift is T19's problem |
| A cost model for the trace store | HLD §11 gives the loop's return-per-GB, which is the only cost number that changes a decision |
| An actual LLM judge call | no network, no GPU — and an unknown ground truth, which is exactly why the judge is modelled (LLD §1.2) |

---

## Sources

| File under `refs/` | Used for |
|---|---|
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt` | the redaction-before-writing rule; the errors/latency/baseline collector policy shape; the four dashboard items including trace coverage; the backend list; the three failure modes; the four carry-out items |
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/How_to_Evaluate_LLM_Apps_LLM-as-a-Judge_RAGAS_Without_the_Bias.txt` | the three judge biases; pointwise vs pairwise; real-traffic gold sets; the 200-example guidance; the fifth-percentile instruction |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` | the agentic share of inference traffic (span-count assumptions in §5) |
| `ai-system-design-guide-main/ai-system-design-guide-main/` | house style for the config and PromQL framing |

Related: [HLD](../HLD.md) · [LLD](../LLD.md) · [SEQUENCES](../docs/SEQUENCES.md) · [run.py](../run.py)
