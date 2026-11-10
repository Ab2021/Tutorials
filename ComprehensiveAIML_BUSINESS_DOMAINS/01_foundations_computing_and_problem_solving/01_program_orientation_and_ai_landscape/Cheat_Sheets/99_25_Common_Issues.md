---
title: "25 Common Issues — Program Orientation and AI Landscape"
topic: "Diagnosing recurring defects in AI use-case selection, vendor claims, and landscape decisions"
collection: "01_program_orientation_and_ai_landscape"
level: "foundation"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
file_id: "99"
owner_role: "curriculum maintainer"
source_index: "INDEX.md"
review_trigger: "Any change to the unit's decision contract or to the vendor-claim review procedure"
notation_style: "plain-text"
source_status: "DERIVED"
prerequisites: ["01_AI_Use_Case_Charter", "02_AI_Market_Map", "03_Technology_Due_Diligence"]
pattern_tags: ["troubleshooting", "decision-hygiene", "numerical-checking"]
---

# 25 Common Issues — 01_program_orientation_and_ai_landscape

> **Purpose:** give a practitioner a repeatable way to separate a real defect from a plausible story when an AI programme decision looks wrong.
> **Evidence status:** every issue below is a *constructed teaching case*. The arithmetic is derived and re-checked here; the organisational settings, ticket volumes, prices and latencies are ILLUSTRATIVE values chosen so the numbers can be recomputed by hand. No vendor, employer, or deployment is being described, and nothing in this file is a measured company result.
> **Reader contract:** you may copy a check, a formula, or a threshold. You may not copy a number into a business case without relabelling its provenance.

## 1. File Header

### 1.1 What this file covers

This unit sits at the front of an AI programme: before pilots, before architecture, at the point where someone
decides *which* problem to attack, *whose* product to consider, and *what evidence* counts as sufficient to spend money.
The defects that hurt most at that stage are not model bugs. They are contract defects (a number means something
different from what the reader assumes), arithmetic defects (a plausible figure computed on the wrong base), and
authority defects (a score acting as an approver).

The catalogue therefore targets four categories only:

- **VIOLATED_ASSUMPTION** — a precondition of the method was true in the study and false in the deployment.
- **NUMERICAL_EDGE** — the formula was applied correctly, but to a degenerate, skewed, or mis-united input.
- **SCALE_FAILURE** — the mechanism is sound until volume, tenancy, or retention changes its economics.
- **CONCEPT_CONFUSION** — two different quantities share one word, so a decision compares unlike things.

Anything that is purely a software bug (a null pointer, a serialisation mismatch) is out of scope here; those belong
to the engineering collection. Anything that needs a live incident timeline is out of scope too, because this file will
not invent one.

### 1.2 How to use the catalogue

Use it as a triage sequence, not as a checklist to read top to bottom.

1. Write the symptom in one sentence, using the number that is wrong and the unit it is in.
2. Classify the symptom into one of the four categories above. Classification is a hypothesis, not a verdict.
3. Open every candidate issue in that category and run its discriminating check. A check that cannot fail is not a check.
4. Contain harm before correcting cause: defer, cap, or route to a human, and record why.
5. Correct the mechanism, then replay the record that triggered the symptom plus one boundary neighbour.
6. Write the verification result into the decision ledger, so the next reviewer sees evidence rather than memory.

The sequence matters because steps 3 and 4 are frequently reversed. A team that "fixes" first and diagnoses second
usually corrects the loudest cause and leaves the mechanism intact; the symptom returns one reporting cycle later,
now with a change history that makes the real cause harder to see.

### 1.3 Severity vocabulary

Severity here is the consequence *within the declared scope of the decision being made*. It is not an incident
severity and must not be pasted into an incident system.

| Level | Meaning in this file | Typical response window |
|---|---|---|
| S1 | The issue can authorise spending or an automated action on a false premise | Same day; block the decision |
| S2 | The issue changes a ranking between two viable options | Before the next review gate |
| S3 | The issue distorts reporting but not the direction of the decision | Next scheduled review |
| S4 | The issue is cosmetic or documentation-only | Batch with routine maintenance |

### 1.4 Provenance policy

Three labels appear in this file and they are not interchangeable.

- **DERIVED** — the arithmetic is shown in this file and can be recomputed from the stated inputs.
- **ILLUSTRATIVE** — the inputs were constructed for teaching. The arithmetic on them is correct; the inputs are not measurements.
- **OPEN** — the cause cannot be determined from the available record, and the file says so instead of guessing.

A reviewer who converts an ILLUSTRATIVE input into a sourced fact has created a new defect of exactly the kind
CI-01 describes. The label travels with the number.

### 1.5 What this file deliberately excludes

- No incident post-mortems: this file has no timelines, no affected-customer counts, no dates of a real outage.
- No vendor comparisons: products are referenced by capability class only, never by a claimed performance ranking.
- No regulatory interpretation: retention and residency issues are described as contract terms, not legal advice.
- No claim that any of the 25 issues is common in the statistical sense. "Common" means "recurring in the teaching
  record of this unit", which is a statement about the syllabus, not about industry frequency.

### 1.6 Maintenance rule

An issue is retired only when its discriminating check can no longer fail, which in practice means the contract that
allowed it has changed. Retiring an issue requires three edits in one commit: the catalogue row, the detailed entry,
and the index line that points here. Deleting a row without a note is the failure mode this rule exists to prevent.

## 2. Concept Map / Mental Model

### 2.1 The five-stage triage loop

```text
            +---------------------------------------------------+
            |  1. STATE the symptom with number + unit + window |
            +-------------------------+-------------------------+
                                      |
                    +-----------------v------------------+
                    | 2. CLASSIFY (assumption / numeric / |
                    |    scale / concept) - hypothesis    |
                    +-----------------+------------------+
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
     +--------v--------+    +---------v---------+   +---------v---------+
     | 3a CHECK input  |    | 3b CHECK formula  |   | 3c CHECK authority|
     |    contract     |    |    and units      |   |    and owner      |
     +--------+--------+    +---------+---------+   +---------+---------+
              |                       |                       |
              +-----------------------+-----------------------+
                                      |
                    +-----------------v------------------+
                    | 4. CONTAIN: defer, cap, or route   |
                    +-----------------+------------------+
                                      |
                    +-----------------v------------------+
                    | 5. CORRECT + REPLAY + LEDGER entry |
                    +------------------------------------+
```

The loop is deliberately asymmetric: containment (stage 4) is cheap and reversible, correction (stage 5) is
expensive and leaves history. Teams that skip containment pay for their diagnosis window in duplicated effects.

### 2.2 The three tracks

Every issue in the catalogue belongs to exactly one track, and the track tells you which artefact to open.

- **Selection track** — the question is "should we do this at all?". Artefacts: use-case charter, value model, baseline
  measurement. Typical failure: the baseline was never measured, so any improvement claim is unfalsifiable.
- **Landscape track** — the question is "what class of tool could do this, and at what cost shape?". Artefacts: market
  map, capability matrix, price sheet with all line items. Typical failure: comparing list prices that exclude the
  dominant cost driver.
- **Diligence track** — the question is "does the specific product meet the specific claim?". Artefacts: evaluation
  protocol, stratified test set, contract terms. Typical failure: the evaluation protocol is written after the result
  is known, which makes it a rationalisation rather than a test.

A symptom that appears to span two tracks usually means the artefact from one track is being used as evidence in the
other. That is not a fourth category; it is CI-25.

### 2.3 Evidence ladder

Place every statement on this ladder and note the rung. The rung determines what kind of decision the statement can
support, regardless of how confident the speaker sounds.

| Rung | Statement form | Supports |
|---|---|---|
| E0 | "The vendor says so" | Nothing; a hypothesis to test |
| E1 | "A published benchmark says so" | Shortlisting, never sizing |
| E2 | "Our own stratified sample says so, n stated" | A pilot decision with a stated interval |
| E3 | "Our pilot on live traffic says so, with a comparator" | A production decision with guardrails |
| E4 | "Our production run says so, with a monitored denominator" | A scaling decision |

Movement up a rung is bought with measurement effort, and the cost is visible in §9. The ladder also explains why
an E1 statement cannot be repaired by repeating it more confidently.

### 2.4 The stop rule

Diagnosis has a budget. Stop and escalate when any of these is true:

- two competing causes survive the same discriminating check, which means the check is not discriminating enough;
- the record needed to decide was never captured, so the honest answer is OPEN and the fix is to change what gets logged;
- the correction would change a contract that other teams depend on, which makes it a decision for the owner, not a fix.

Continuing past the stop rule produces confident-sounding narrative instead of evidence. That is the single most
expensive output a programme can generate at orientation stage, because it is consumed by people who cannot check it.

### 2.5 Why the categories are diagnostic and not cosmetic

A NUMERICAL_EDGE fixed as if it were a CONCEPT_CONFUSION produces a guard that fires on legitimate input.
A CONCEPT_CONFUSION fixed as if it were a NUMERICAL_EDGE produces a tolerance that hides the disagreement.
A SCALE_FAILURE fixed as if it were a VIOLATED_ASSUMPTION produces a rewritten policy that the system ignores.
In each pairing the fix is plausible, cheap, and wrong, which is why the classification step is not optional.

## 3. Foundational Definitions

Each term is defined as it is used in this file. Where a term is contested in industry usage, the definition states
the operational consequence rather than trying to settle the dispute.

| Term | Definition used here |
|---|---|
| Symptom | An observable output that disagrees with a stated expectation, expressed with a number, a unit, and a window. |
| Cause | The mechanism that produces the symptom; a cause is falsifiable, a story is not. |
| Discriminating check | An observation whose outcome differs between two competing causes. If both causes predict the same outcome, the check is useless. |
| Containment | A reversible action that limits harm while diagnosis continues: defer, cap, route to a human, or return a prior result. |
| Correction | A change to the mechanism, a contract, or a policy, such that replay of the triggering record behaves differently. |
| Verification | A repeatable procedure with a pass condition, run on the triggering record and at least one boundary neighbour. |
| Decision ledger | The append-only record of what was decided, by whom, on which inputs, under which policy version. |
| Cut-off timestamp | The instant after which an input may no longer influence a decision; distinct from the input's own event time. |
| Population | The complete set of items a metric is meant to describe; every rate needs one stated. |
| Stratified sample | A sample drawn to over-represent rare but decision-relevant strata, then reweighted to the live mix before reporting. |
| Base rate (prevalence) | The share of the population that truly belongs to the positive class, before any model output. |
| Sensitivity | The share of true positives that the system flags; a property of the system at a threshold. |
| Specificity | The share of true negatives that the system leaves unflagged; also threshold-dependent. |
| Precision | The share of flagged items that are truly positive; a joint property of the system and the base rate. |
| Containment rate | The share of interactions the workflow resolves without human handling; a business metric, not a model metric. |
| Benchmark | A fixed task set with a fixed scoring rule, used for comparison across systems; not a forecast of field performance. |
| Pilot | A bounded deployment on a stated slice of live work, with a comparator and a stop condition declared in advance. |
| Unit economics | Cost and benefit expressed per transaction, per tenant, or per month, with the volume at which fixed costs are recovered. |
| Idempotency key | A stable identifier attached to a request so that a repeat produces one effect rather than two. |
| Effect count | The number of durable side effects produced by a logical request; the quantity that matters when retries occur. |
| p95 latency | The 95th percentile of a latency distribution over a stated window; a statement about the tail, not about the mean. |
| SLO | A service-level objective: a target for a named percentile over a named window, with a consequence if missed. |
| Chained availability | The product of the availabilities of services that must all succeed for the end-to-end operation. |
| Review capacity | The human hours per period available to examine escalated items; a hard constraint that models do not relax. |
| Threshold | A numeric cut on a score; a modelling parameter, never an authority. |
| Authority | The named role permitted to take the action; recorded with the decision, not inferred from the score. |
| Evidence rung | The position of a claim on the E0–E4 ladder in §2.3, used to bound what the claim can support. |

## 4. Core Formulas with Full Derivation

Four families carry almost every quantitative argument in this unit: backlog and drain, decision quality at a base
rate, cost accumulation, and time-to-tail composition. Each is derived in the five-part anatomy: formula, where it
applies, the constraint that keeps it valid, a numerical example recomputed by hand, and the interpretation that the
example licenses. §4.4 then states where each formula stops being true.

### 4.1 Base case

#### F1 — Backlog accumulation and drain

```text
FORMULA [F1-BACKLOG]: B = max(0, lambda - mu) * t      then  T_drain = B / (mu_d - lambda_d)
WHERE: B [requests], lambda, mu, mu_d, lambda_d [requests/s], t [s], T_drain [s]
CONSTRAINT: empty initial queue, constant rates, no loss, no retries, single server class
NUMERICAL EXAMPLE: lambda = 128, mu = 110, t = 240 -> B = 18 * 240 = 4,320 requests
                  drain with mu_d = 150, lambda_d = 90 -> 4,320 / 60 = 72 s
INTERPRETATION: the backlog is an integral of the rate gap; draining needs a strictly positive surplus
```

The interpretation step is where most arguments go wrong. B = 4,320 does not mean 4,320 requests will be late; it
means 4,320 requests are queued at the end of the window, all of which will be late by at least 72 s even under the
best recovery rate stated. Late *count* depends on arrival timing within the window, which the formula does not model.

#### F2 — Precision at a base rate

```text
FORMULA [F2-PRECISION]: P = Se * p / (Se * p + (1 - Sp) * (1 - p))
WHERE: P, Se, Sp, p are dimensionless probabilities in [0,1]; p is prevalence, not the flag rate
CONSTRAINT: single threshold, no abstention class, labels are ground truth, calibration irrelevant to this identity
NUMERICAL EXAMPLE: Se = 0.88, Sp = 0.94, p = 0.03
  TP-rate = 0.88 * 0.03 = 0.0264
  FP-rate = 0.06 * 0.97 = 0.0582
  P = 0.0264 / (0.0264 + 0.0582) = 0.0264 / 0.0846 = 0.3121 -> 31.2%
INTERPRETATION: at a 3% base rate, nearly seven of ten escalations are false alarms despite 94% specificity
```

The identity is just Bayes' rule with a normalising denominator, and it contains no free parameter. That is exactly
why it is useful: it shows that a claim about Se and Sp alone can never settle a claim about P. The flag rate here is
0.0846, i.e. 8.46% of all items are escalated, of which 0.0264/0.0846 = 31.2% are real.

#### F3 — Linear cost per request

```text
FORMULA [F3-COST]: c_r = (n_in * p_in + n_out * p_out) / 1e6 + c_fixed / N
WHERE: n_in, n_out [tokens]; p_in, p_out [currency per 1e6 tokens]; c_fixed [currency/month]; N [requests/month]
CONSTRAINT: prices linear in tokens, no volume discount, no retry overhead, fixed platform cost constant in N
NUMERICAL EXAMPLE: n_in = 3400 at 3.00, n_out = 480 at 15.00
  = (3400 * 3.00 + 480 * 15.00) / 1e6 = (10200 + 7200) / 1e6 = 0.0174 per request
  at N = 5200 and c_fixed = 2400 -> 2400 / 5200 = 0.4615 fixed share
  c_r = 0.0174 + 0.4615 = 0.4789 per request
INTERPRETATION: at pilot volume the platform bill, not the token bill, sets the unit price
```

The formula is linear in tokens and *hyperbolic* in volume. That one asymmetry generates most of the pricing
arguments in this unit: a vendor quote per token is a marginal number being discussed as if it were an average one.

#### F4 — Little's law for concurrency

```text
FORMULA [F4-LITTLE]: L = lambda * W
WHERE: L [requests in flight], lambda [requests/s], W [s per request], same measurement window for both
CONSTRAINT: stable system, no growth or drain inside the window, all arrivals eventually served
NUMERICAL EXAMPLE: lambda = 20, W = 1.7 -> L = 34 concurrent requests; check 34 / 1.7 = 20
INTERPRETATION: concurrency is a product, so a 30% latency increase raises worker count by 30% at fixed arrival rate
```

L is the minimum steady-state concurrency. Provisioning below L guarantees a growing queue regardless of how fast
each individual worker is, which is the arithmetic behind CI-15.

### 4.2 General case

#### F5 — Tail of a sum of stages

```text
FORMULA [F5-TAILSUM]: T_total ~ N(mu_1 + mu_2 + c, sqrt(s1^2 + s2^2 + 2*rho*s1*s2))
  then p95(T_total) = mean_total + 1.645 * sigma_total ; and p95(X) + p95(Y) >= p95(X + Y) when tails are light
WHERE: mu_k [ms] stage means, s_k [ms] stage standard deviations, rho in [-1,1], c [ms] constant stages
CONSTRAINT: approximately normal stage times; independence or a stated rho; no shared hard timeout
NUMERICAL EXAMPLE: retrieval mu = 200, s = 80 ; generation mu = 850, s = 300 ; post = 40 constant
  sum-of-percentiles: (200 + 1.645*80) + (850 + 1.645*300) + 40 = 331.6 + 1343.5 + 40 = 1715.1 ms
  independent: sigma = sqrt(6400 + 90000) = sqrt(96400) = 310.5
               p95 = 1090 + 1.645 * 310.5 = 1090 + 510.7 = 1600.7 ms
  rho = 0.8 : sigma = sqrt(96400 + 2*0.8*80*300) = sqrt(134800) = 367.1
              p95 = 1090 + 603.9 = 1693.9 ms
INTERPRETATION: adding component p95s over-provisions by 7.1% when independent and by 1.3% when highly correlated
```

Over-provisioning by 114 ms sounds harmless until it is multiplied by worker count. Under-provisioning happens when
stages share a resource (same host, same connection pool, same garbage-collection pause), and that is the case the
sum-of-percentiles method silently assumes away.

#### F6 — Retry amplification across layers

```text
FORMULA [F6-RETRY]: A = (1 / (1 - q))^k
WHERE: q = per-attempt failure probability, k = number of independent retry layers, A = attempts per logical request
CONSTRAINT: failures independent across layers, retries capped at one each, no jitter-free synchronisation
NUMERICAL EXAMPLE: q = 0.04, k = 3 -> 1/0.96 = 1.0417 ; 1.0417^3 = 1.1304 -> 13.0% extra attempts
INTERPRETATION: three retry layers turn a 4% per-attempt failure rate into a 13% capacity surcharge
```

The surcharge lands exactly when the system is already degraded, because q is not constant — it rises with load.
A capacity plan built on the healthy-q figure therefore fails at the moment it is needed, which is CI-12's mechanism.

#### F7 — Context re-sending in multi-turn sessions

```text
FORMULA [F7-SESSION]: n_in_session = sum over i = 1..T of (b + (i - 1) * o) = T*b + o*T*(T-1)/2
WHERE: b [tokens] base prompt, o [tokens] per-turn added output, T [turns], n_in_session [tokens]
CONSTRAINT: every turn re-sends the full history, no server-side caching, no truncation window
NUMERICAL EXAMPLE: b = 3400, o = 480, T = 12
  = 12*3400 + 480*(12*11/2) = 40800 + 480*66 = 40800 + 31680 = 72480 input tokens
  cost = 72480*3.00/1e6 + 12*480*15.00/1e6 = 0.21744 + 0.0864 = 0.30384 per session
  naive per-turn estimate = 12*(0.0102 + 0.0072) = 0.2088 -> understates by 31.3%
INTERPRETATION: the quadratic term is invisible in a single-turn cost model and dominates long sessions
```

The understatement is 0.30384 / 0.2088 = 1.4552, i.e. the true cost is 45.5% higher; framed the other way, the naive
model recovers only 68.7% of the cost. Both framings are correct and the distinction is the whole point of CI-13.

#### F8 — Chained availability

```text
FORMULA [F8-AVAIL]: A_chain = product over k of A_k ; downtime_minutes = (1 - A_chain) * minutes_in_period
WHERE: A_k dimensionless in (0,1]; period = 43200 minutes for a 30-day month
CONSTRAINT: all stages required for success, independent failures, no retry or failover, synchronous chain
NUMERICAL EXAMPLE: 0.9995 * 0.999 * 0.998 = 0.9985005 * 0.998 = 0.9965035
  downtime = (1 - 0.9965035) * 43200 = 0.0034965 * 43200 = 151.0 minutes/month
  compare single stage 0.999 -> 0.001 * 43200 = 43.2 minutes/month
INTERPRETATION: the chain produces 3.5x the downtime of a single 99.9% stage (151.0 vs 43.2 minutes)
```

### 4.3 Variants

#### F9 — Sample size to test a claim

```text
FORMULA [F9-N]: n = p*(1-p) * (z / m)^2
WHERE: p estimate of the proportion, m margin of error in proportion units, z = 1.96 at 95% confidence, n [items]
CONSTRAINT: simple random sample from the target population, independent items, n*p and n*(1-p) both >= 10
NUMERICAL EXAMPLE: p = 0.88, m = 0.03 -> 0.88*0.12*(1.96/0.03)^2 = 0.1056*4268.44 = 450.7 -> 451 items
                  m = 0.05 -> 0.1056*1536.64 = 162.3 -> 163 items
INTERPRETATION: tightening the margin from 5 to 3 points nearly triples the labelling bill
```

The 451 figure is the *unstratified* case. If the workflow has four strata of unequal decision weight, the sample must
be allocated across them and reweighted, which raises the effective n (the cost consequence is worked in §9.3).

#### F10 — Expected value of diligence

```text
FORMULA [F10-EV]: EV_diligence = (P_before - P_after) * L - C
WHERE: P_* failure probabilities in [0,1], L loss given failure [currency], C cost of the diligence [currency]
CONSTRAINT: single loss magnitude, no partial-failure outcomes, diligence result actually changes the go/no-go
NUMERICAL EXAMPLE: (0.40 - 0.22) * 480000 - 22000 = 0.18*480000 - 22000 = 86400 - 22000 = 64400
  ratio = 86400 / 22000 = 3.93
INTERPRETATION: diligence pays only when it can flip the decision; with no authority to stop, EV is -C
```

#### F11 — Amdahl bound on pipeline speed-up

```text
FORMULA [F11-AMDAHL]: S = 1 / ((1 - f) + f / s)
WHERE: f = share of wall time in the accelerated stage, s = speed-up factor on that stage, S = total speed-up
CONSTRAINT: fixed workload, no overlap, the remaining stages unchanged, single-threaded critical path
NUMERICAL EXAMPLE: f = 0.72, s = 2 -> S = 1 / (0.28 + 0.36) = 1 / 0.64 = 1.5625
                  s -> infinity -> S_max = 1 / 0.28 = 3.571
INTERPRETATION: halving the dominant stage buys 56% throughput and no configuration change buys more than 257%
```

#### F12 — Memory and unit conversion for vector stores

```text
FORMULA [F12-VECTOR]: bytes = n_chunks * d * w ; GB_decimal = bytes / 1e9 ; GiB = bytes / 2^30
WHERE: n_chunks [items], d [dimensions], w [bytes per component: 4 for float32, 2 for float16]
CONSTRAINT: dense flat index, no compression, no metadata overhead, one replica per copy
NUMERICAL EXAMPLE: 1400000 * 384 * 4 = 2,150,400,000 bytes
  = 2.1504 GB decimal = 2.1504e9 / 1.073741824e9 = 2.0027 GiB
  a dashboard reporting 2.00 GiB understates a per-GB egress bill by 2.1504/2.0027 - 1 = 7.4%
INTERPRETATION: decimal and binary prefixes differ by 7.4%, which is larger than most negotiation margins
```

### 4.4 Limits and counterexamples

Each formula above has a stated validity condition. The counterexamples below are the ones that survive review
because they look like the base case.

- **F1 fails when the queue is capped or lossy.** With a hard cap at 2,000 and B = 4,320, the excess 2,320 is rejected,
  not queued; the observed late count falls while the rejection count rises. A team reading only the late-count series
  concludes the incident ended.
- **F2 fails when the labels are the same judgements the model was trained to imitate.** Se and Sp then describe
  agreement with an annotator policy, not with truth, and P inherits the annotator's error. Check by re-labelling a
  subset blind to the model output.
- **F3 fails at the volume where discounts or caching start.** At 250,000 requests/month the fixed share collapses to
  2400/250000 = 0.0096, so c_r = 0.0270; extrapolating the pilot figure of 0.4789 upward overstates cost by 17.7x.
- **F4 fails while the system is draining.** During a drain, L is not lambda*W because arrivals and departures are
  imbalanced; the formula describes the steady state only.
- **F5 fails when a hard timeout truncates the tail.** Timeouts replace long latencies with fast failures, so the
  measured p95 improves while the success rate falls. Always report both or neither.
- **F6 fails when retries are not independent.** A shared dependency makes q equal across layers, so the layers
  multiply attempts without multiplying success probability — pure load, no benefit.
- **F7 fails when the provider caches the prefix.** Cached input tokens are usually priced at a fraction of fresh
  input, which can remove most of the quadratic term; the model must then split cached and uncached tokens.
- **F8 fails when stages share a failure domain.** Co-located services in one zone give correlation close to one, and
  the product understates joint downtime by a wide margin.
- **F9 fails when the sample is drawn from a convenience pool.** A 451-item sample of easy tickets has the same n and
  none of the information; the interval is precise about the wrong population.
- **F10 fails when diligence cannot change the decision.** If the sponsor has already committed, P_after = P_before
  and EV = -C exactly, however good the analysis is.
- **F11 fails when the accelerated stage overlaps others.** With pipelining, f is not a serial share and S can exceed
  the Amdahl bound; check whether stages run concurrently before quoting the bound.
- **F12 fails with quantisation or HNSW overhead.** float16 halves the payload but adds graph and metadata structures
  that can exceed the vector bytes at small dimensions.

## 5. Worked Numerical Example

### 5.1 The setting

ILLUSTRATIVE programme: a policy-question assistant for a service desk. A sponsor wants to approve a 12-month
subscription on the strength of a vendor benchmark that quotes 94% answer accuracy. The programme office must decide
whether the unit economics work at the desk's actual volume and review capacity.

Constructed inputs, all ILLUSTRATIVE:

- requests: N = 5,200 per month, one request per ticket
- tokens: n_in = 3,400, n_out = 480; prices $3.00 and $15.00 per million tokens
- fixed retrieval and orchestration platform: $2,400 per month
- handling time before the assistant: 5.5 minutes per ticket
- adoption: 65% of tickets take the assisted path; time saving on those tickets: 28%
- loaded agent cost: $62.00 per hour
- escalation: prevalence of the "needs specialist" class p = 0.03; Se = 0.88, Sp = 0.94
- review cost per escalated item: 4.0 minutes at the same $62.00 hourly rate

### 5.2 Step 1 — variable cost

```text
c_var = (3400*3.00 + 480*15.00)/1e6 = (10200 + 7200)/1e6 = 0.0174 per request
monthly variable = 0.0174 * 5200 = 90.48
```

### 5.3 Step 2 — unit cost including the platform

```text
c_fixed_share = 2400 / 5200 = 0.461538
c_r = 0.0174 + 0.461538 = 0.478938 -> 0.4789 per request
monthly total = 90.48 + 2400 = 2490.48
check: 2490.48 / 5200 = 0.478938
```

The platform bill is 96.4% of the monthly cost. Any argument about token pricing is therefore an argument about
4% of the bill at this volume, which is why the first thing to test is volume, not price.

### 5.4 Step 3 — labour benefit

```text
saved_minutes_per_ticket = 5.5 * 0.28 = 1.54 min
saved_hours = 5200 * 0.65 * 1.54 / 60 = 3380 * 1.54 / 60 = 5205.2 / 60 = 86.753 h
benefit = 86.753 * 62.00 = 5378.70
```

Independent re-check by a different route, to catch a unit slip:

```text
per_ticket_benefit = 1.54 min = 1.54/60 h = 0.0256667 h ; 0.0256667 * 62.00 = 1.591333
eligible_tickets = 5200 * 0.65 = 3380
3380 * 1.591333 = 5378.71  (agrees with 5378.70 to rounding)
```

### 5.5 Step 4 — escalation load and review cost

```text
flag_rate  = Se*p + (1 - Sp)*(1 - p) = 0.0264 + 0.0582 = 0.0846
flagged    = 0.0846 * 5200 = 439.9 -> 440 items/month
true_pos   = 0.0264 * 5200 = 137.3 -> 137 items/month
precision  = 137 / 440 = 0.3114 (matches F2's 31.2% within rounding)
review_h   = 440 * 4.0 / 60 = 29.333 h
review_cost = 29.333 * 62.00 = 1818.67
```

### 5.6 Step 5 — net position

```text
net = benefit - variable - fixed - review
    = 5378.70 - 90.48 - 2400.00 - 1818.67
    = 5378.70 - 4309.15
    = 1069.55 per month   (ILLUSTRATIVE)
annualised = 1069.55 * 12 = 12834.60
benefit_cost_ratio = 5378.70 / 4309.15 = 1.248
```

The proposal looked like a 2.2x return when the review load was omitted (5378.70 / 2490.48 = 2.16). Including the
escalation the same design produces, the ratio is 1.25. One line of arithmetic reversed the strength of the case,
and no model behaviour changed.

### 5.7 Step 6 — latency feasibility

```text
budget_p95 = 2000 ms
model_p95_independent = 1600.7 ms  (F5)
headroom = 2000 - 1600.7 = 399.3 ms = 19.96% of budget
with rho = 0.8 headroom = 2000 - 1693.9 = 306.1 ms = 15.3% of budget
```

A 15% headroom is not a safety margin at a percentile boundary; it is roughly the size of one deployment's worth of
variance. Feasibility therefore depends on the correlation assumption, which is an architectural fact, not a
statistical preference.

### 5.8 Step 7 — break-even volume

```text
margin_per_ticket = 1.591333 (benefit) - 0.0174 (variable) = 1.573933
break_even_N = 2400 / 1.573933 = 1524.8 -> 1525 tickets/month (ignoring review cost)
including review at 0.0846 flags/ticket * 4/60 h * 62.00 = 0.0846*4*1.03333 = 0.3497 per ticket
breakeven_incl_review = 2400 / (1.573933 - 0.3497) = 2400 / 1.224233 = 1960.4 -> 1961 tickets/month
share_of_current_volume = 1961 / 5200 = 37.7%
```

So the programme needs at least 1,961 tickets a month at these parameters before it stops losing money. That number,
not the vendor's accuracy, is the decision-relevant quantity.

### 5.9 Sensitivity table

| Parameter moved | New value | Net per month | Direction |
|---|---|---|---|
| Adoption 65% -> 40% | 0.40 | 5200*0.40*1.591333 = 3310.0 ; 3310.0 - 4309.15 = -999.15 | Loss |
| Saving 28% -> 18% | 0.18 | 5.5*0.18/60*62 = 1.023 per ticket ; 3380*1.023 = 3457.74 ; -4309.15 -> -851.41 | Loss |
| Prevalence 3% -> 10% | p = 0.10 | flag 0.088+0.054 = 0.142 ; review 0.142*5200*4/60*62 = 3052.05 ; net = -163.83 | Loss |
| Volume 5,200 -> 20,800 | 4x | benefit 21514.80 ; var 361.92 ; fixed 2400 ; review 7273.34 ; net = 11479.54 | Strong gain |
| Review time 4 -> 2 min | 2.0 | review 440*2/60*62 = 909.33 ; net = 1978.89 | Gain |

Read the table as a ranking of *what to measure first*. Volume and prevalence move the sign; token price does not
appear at all, because at this scale it cannot.

### 5.10 What the example licenses

The example supports one decision: "test prevalence and adoption before approving a subscription, and price the
review capacity into the case". It does not support any claim about a real product, and the 1,961-ticket break-even
is a property of the constructed inputs, not of the workflow. Anyone quoting 1,961 without the parameter set is
committing CI-01.

## 6. Visual / Geometric Intuition

### 6.1 The base-rate plane

```text
precision
1.0 |                                    * Se=.95 Sp=.99
    |                            *
0.8 |                    *
    |              *
0.6 |         *
    |      *
0.4 |   *
    | *
0.2 *
    +----+----+----+----+----+----+----+-----> prevalence p
    0   .05  .10  .20  .30  .40  .50  .70
```

The curve is flat only at the high-prevalence end. Every argument that quotes precision without stating p is
quoting a point on a curve as if it were a constant. When two teams disagree about precision, the first question is
whether they are standing at different points on the x-axis.

### 6.2 Cost shape against volume

```text
cost/request
0.50 |*
0.25 | *
0.10 |  *
0.05 |    *  *
0.02 |          *   *    *     *      (asymptote = c_var)
     +---+---+---+---+---+---+---> N (thousand requests/month)
        1   5  10  20  50 100 250
```

At N = 1,000 the fixed share is 2.40; at N = 250,000 it is 0.0096. The curve has no minimum and no optimum — it is a
monotone approach to the marginal cost. Any "per-request price" quoted without the volume is therefore not a price.

### 6.3 Tail composition

```text
stage A  |============|                      mean 200, s 80
stage B          |===================|       mean 850, s 300
sum      |=======================================|  p95 1600.7 independent
p95-sum  |==========================================|  p95 1715.1 additive
                                            ^ budget 2000
                                            |<-headroom->|
```

The gap between the two bars is the cost of assuming independence. The gap between the additive bar and the budget
line is the entire engineering margin, and it is smaller than it looks because latency budgets are consumed by
deployments, not by averages.

### 6.4 Triage as a decision tree

```text
symptom
  |
  +-- number changed but inputs same? ------ yes --> NUMERICAL_EDGE (unstable input or unit)
  |                                   \----- no  --> continue
  |
  +-- input distribution changed? --------- yes --> VIOLATED_ASSUMPTION
  |
  +-- volume/tenancy/retention changed? --- yes --> SCALE_FAILURE
  |
  +-- two words for one quantity? --------- yes --> CONCEPT_CONFUSION
  |
  +-- none of the above -----------------> capture more evidence; do not guess
```

The last branch is the important one. A triage tree that always produces an answer teaches people to invent one.

## 7. Comparison / Selection Table

### 7.1 Which check to run first

The table is ordered by how cheaply the check can falsify a competing explanation. Cost is expressed as reviewer
minutes, assuming a $48/hour reviewer and no new instrumentation.

| Observed symptom | Category first tried | Cheapest first check | Cost | Disqualifier that ends the search |
|---|---|---|---|---|
| Precision in production far below the pilot figure | VIOLATED_ASSUMPTION | Compare prevalence in the pilot frame against live prevalence | 20 min | Prevalence differs by more than 2x; nothing else needs checking |
| Two vendors' accuracy figures differ by a few points | CONCEPT_CONFUSION | Read both task definitions and scoring rules side by side | 30 min | The task sets differ in composition; the comparison is void |
| Cost per request quoted by two vendors differs 3x | NUMERICAL_EDGE | Recompute both from token counts and the same price basis | 45 min | Tokenisers differ enough to explain the gap |
| Monthly bill far above the modelled bill | SCALE_FAILURE | Decompose the bill into fixed, variable, overage lines | 25 min | An overage line appears that the model omitted |
| Latency SLO breached at p95 while mean looks fine | NUMERICAL_EDGE | Rebuild the percentile from raw samples, not from per-stage p95s | 40 min | A hard timeout is truncating the tail |
| Queue depth rising steadily at constant traffic | SCALE_FAILURE | Measure lambda, mu and the backlog slope over two windows | 35 min | lambda minus mu is positive; concurrency is under-provisioned |
| Two reports of the same accuracy disagree | CONCEPT_CONFUSION | Compare the weighting each report applied to strata | 25 min | Different mixes; pooled numbers are not comparable |
| An automated action was taken on a model score | CONCEPT_CONFUSION | Audit the actor field and policy version in the ledger | 15 min | The actor is a role, but its mandate did not cover this action |
| Duplicate tickets or duplicate emails after an error spike | SCALE_FAILURE | Count effects per logical request key | 30 min | Effect count > 1 for one key; deduplication is absent |
| Pilot result cannot be reproduced | VIOLATED_ASSUMPTION | Diff the pilot prompt, version, and data slice | 20 min | Prompt or slice changed between runs |

### 7.2 Category against containment choice

Containment is chosen by blast radius, not by category. The same category can need different containment on
different days, and a team that fixes containment to category will over-contain small defects and under-contain
large ones.

| Blast radius | Preferred containment | Why not the alternative |
|---|---|---|
| One record, one key | Return the prior result and log the reason | Re-running silently doubles the effect |
| One tenant, many records | Cap intake for that tenant only | Global caps spread one tenant's defect across all |
| One workflow, all tenants | Route to human review with a visible backlog counter | Silent deferral hides the workload from the owner |
| Spend-authorising output | Block and escalate to the named approver | Any automated fallback becomes an authority claim |
| Reporting only | Publish with a stated defect note | Withholding the number invites an unofficial substitute |

### 7.3 Evidence rung required per decision

| Decision | Minimum rung | What a lower-rung artefact cannot do |
|---|---|---|
| Shortlist two vendors | E1 | Cannot size the spend |
| Approve a paid pilot | E2 | Cannot claim a production saving |
| Approve production rollout | E3 | Cannot support a multi-year commitment |
| Commit to a volume-based contract | E4 | Cannot support a break-even claim at scale |
| Retire an incumbent process | E3 plus a comparator run | Cannot prove the replacement is better, only that it exists |

### 7.4 Diagnostic against correction authority

| Diagnostic conclusion | Who may correct it | Wrong owner's likely action |
|---|---|---|
| Model scores poorly on a stratum | Evaluation owner | Rewrite the vendor contract, which does not help |
| Prevalence assumption wrong | Business owner of the baseline | Retrain the model, which changes nothing |
| Cost model omitted a term | Finance partner | Add a guard that defers work, which adds cost |
| Score used as authority | Process owner plus risk | Lower the threshold, which increases the harm |
| Retry duplicates effects | Integration owner | Add a monitor, which records the duplicates |

## 8. Failure Modes & Edge Cases

### 8.1 Catalogue

Each row is the index into the detailed entries that follow. The verification column states what must be observed
after the correction, so a reviewer can tell a fix from a postponement.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | Field precision far below the pilot's precision | Prevalence changed, or the pilot frame was enriched | Compare live prevalence with pilot-frame prevalence | Restate performance at the live base rate; re-baseline | Publish precision with its prevalence and frame | Recomputed precision on live-mix sample within 2 points of prediction |
| CI-02 | CONCEPT_CONFUSION | Vendor "accuracy" cannot be reproduced on our tickets | Different task, different scoring rule, or different prompt | Run both protocols on the same 100 tickets | Rebuild the comparison on one protocol | Contract the task definition and scoring rule in writing | Same protocol, same tickets, gap explained or gone |
| CI-03 | NUMERICAL_EDGE | Price per request differs 3x between two vendors | Tokeniser difference, output-length difference, or hidden line items | Count tokens with each vendor's own tokeniser | Normalise to cost per completed task | Compare on cost per task, never on price per token | Recomputed cost per task within 10% after normalisation |
| CI-04 | NUMERICAL_EDGE | Per-request price collapses when volume grows | Fixed platform cost divided by a pilot denominator | Split the bill into fixed and variable lines | Quote marginal and average cost at the real volume | State the volume with every unit price | Average cost recomputed at production volume |
| CI-05 | NUMERICAL_EDGE | End-to-end p95 exceeds the sum of stage p95s | Percentiles added, correlation between stages, or truncation | Rebuild the percentile from raw end-to-end samples | Budget on the measured joint distribution | Record per-stage samples, not only summaries | Measured p95 within 10% of the model's prediction |
| CI-06 | CONCEPT_CONFUSION | "List price" and "our price" differ by 40% | Commitment tier, overage rate, or support line | Read the invoice, not the price sheet | Model effective price including overage | Require a three-scenario price model in the business case | Invoice reproduction within 5% from the model |
| CI-07 | SCALE_FAILURE | Break-even sits above current volume | Fixed cost spread over too few units | Recompute break-even at pilot and production volumes | Stage the commitment to volume | State break-even volume on every business case | Break-even recomputed with actual invoice data |
| CI-08 | VIOLATED_ASSUMPTION | Hours saved do not appear in the budget | No demand to absorb freed time, or salaried staff | Check queue length and hiring plan for the freed hours | Convert to wait-time or capacity benefit | Declare the conversion rule before the pilot | Finance confirms the benefit line or reclassifies it |
| CI-09 | VIOLATED_ASSUMPTION | Benefit overstated relative to live usage | Adoption assumed at 100%, or eligibility overstated | Measure assisted share of eligible tickets | Recompute with measured adoption | Pilot the adoption curve before the benefit claim | Adoption measured on four consecutive weeks |
| CI-10 | VIOLATED_ASSUMPTION | Review capacity becomes the binding constraint | Escalation cost omitted from the model | Compare flagged items per month with reviewer hours | Price the review line into the case | Model review as a first-class cost | Review hours per month match the model within 15% |
| CI-11 | NUMERICAL_EDGE | A better model produces a worse total cost | Threshold selected on accuracy, not on cost | Compute cost per 100 items at several thresholds | Choose the threshold that minimises expected cost | Publish the cost matrix with the threshold | Cost per 100 recomputed at the chosen threshold |
| CI-12 | SCALE_FAILURE | Capacity surcharge appears only during degradation | Retry layers multiplied, failure not independent | Count attempts per logical request during an incident | Collapse retries to one layer | Budget capacity at degraded failure rates | Attempt multiplier measured under induced failures |
| CI-13 | NUMERICAL_EDGE | Session cost far above per-turn estimate | History re-sent each turn, quadratic growth | Sum input tokens per session from the log | Model the quadratic term | Price sessions, not turns, in the business case | Recomputed session cost within 5% of billed |
| CI-14 | SCALE_FAILURE | Queue grows though each request is fast | Concurrency below lambda times W | Compute lambda times W and compare with worker count | Provision to measured concurrency | Derive worker count from Little's law | Queue depth flat over a two-hour constant-rate window |
| CI-15 | CONCEPT_CONFUSION | Team debates "latency" and disagrees on the number | Waiting time versus service time versus percentile | Separate queue time from service time in the trace | Report both, named | Name the quantity in every SLO | Two named metrics on the dashboard, no ambiguity |
| CI-16 | CONCEPT_CONFUSION | End-to-end availability misses the target badly | Component SLA quoted as chain availability | Multiply the required stages and convert to minutes | Set targets per stage from the chain budget | State the chain, not the weakest link | Chain product tracked monthly against the budget |
| CI-17 | NUMERICAL_EDGE | Storage and egress bills exceed the estimate | Decimal versus binary prefixes | Convert bytes under both prefixes | Quote in the billing unit | Put the unit in the column header | Bill reproduced from bytes within 2% |
| CI-18 | CONCEPT_CONFUSION | One report shows version A better, another version B | Pooled versus per-segment comparison, unequal segment sizes | Recompute per segment and per pool separately | Report per segment with weights shown | Publish segment table with every pooled figure | Segment and pooled tables both present and consistent |
| CI-19 | NUMERICAL_EDGE | Stratified results differ from live experience | Wrong weights applied to strata | Recompute with live stratum shares | Reapply the live mix | Store the weights with the result | Weighted estimate within 2 points of a live check |
| CI-20 | VIOLATED_ASSUMPTION | Tight interval but the estimate is wrong | Sample drawn from a convenience frame | Compare frame composition with production mix | Redraw from the production frame | Declare the frame before sampling | Frame composition within 5 points of production |
| CI-21 | VIOLATED_ASSUMPTION | Accuracy drifts without any model change | Label definition drift or annotator turnover | Double-label a subset blind and compare | Re-pin the label guide and re-baseline | Version the label guide with the test set | Inter-annotator agreement restored above the floor |
| CI-22 | CONCEPT_CONFUSION | An action was taken that nobody approved | Score threshold treated as an approval | Read actor and policy version in the ledger | Require a named approver for the action class | Separate the score from the mandate in the design | Ledger shows approver for every action in the class |
| CI-23 | SCALE_FAILURE | Duplicate side effects after an error spike | Missing or unstable idempotency key | Replay one request key twice and count effects | Persist deduplication before the effect | Test the retry path with the effect counter | Effect count of one per key across a forced retry |
| CI-24 | NUMERICAL_EDGE | A 1-point difference is called an improvement | Sample too small, or intervals overlapping | Compute intervals for both estimates | Require non-overlap before claiming a win | State n and interval with every comparison | Decision recorded with its interval and rule |
| CI-25 | CONCEPT_CONFUSION | A benchmark is cited as evidence of field performance | Rung confusion, or deliberate compression | Ask which population and which task the figure describes | Relabel to the correct rung | Tag every number with its rung | Every figure in the case carries a rung tag |

### 8.2 Detailed entries

#### CI-01 — Precision quoted from an enriched pilot frame

**Symptom.** The pilot reported 86% precision; the first live month shows 31% of escalations are correct. Nothing in
the model changed, and the error is on the business side of the decision: the number was true and the sentence
around it was false.

**Mechanism.** A pilot is usually built from a curated set where the interesting class is over-represented, because
that is how you get enough positives to measure anything. Precision depends on prevalence through Bayes' identity,
so a frame with 30% positives and a live mix with 3% positives produce numbers that share a formula and nothing else.
The pilot was measuring the model; the business case needed a measurement of the population.

```text
se = 0.88, sp = 0.94
pilot frame  p = 0.30 : TP 0.2640, FP 0.0420, flag 0.3060, precision 0.863
live mix     p = 0.03 : TP 0.0264, FP 0.0582, flag 0.0846, precision 0.312
ratio of precisions = 0.863 / 0.312 = 2.77
```

**Competing causes.** The first is a genuine degradation: the model may be scoring differently on live input because
the input distribution shifted. The second is the frame difference above, which needs no model explanation at all.
Both produce "precision fell", and they are separated by one observation: reconstruct the flag rate on live data.

**Discriminating check.** Compare the flag rate with the predicted flag rate. A model that still behaves the same
flags 8.46% of live items; if the observed flag rate is 30%, then the live population resembles the pilot frame and
the model is fine. If the flag rate is 8.5% but precision is 31%, the frame explanation holds and the model is fine
too — the number was simply reported at the wrong base rate.

**Containment.** Keep the escalation path as it is, but stop quoting precision in any approval document until the
prevalence-stated version exists. The containment is editorial, which is correct here: the harm is a wrong decision,
not a wrong action.

**Correction.** Recompute expected precision at the live prevalence, publish the pair (precision, prevalence), and
re-baseline the business case with the corrected figure. If the corrected number breaks the case, that is the
finding, not an inconvenience.

**Prevention.** Make prevalence a mandatory field in the evaluation record. A precision figure without a prevalence
and a frame description is not a result; treat it as a draft.

**Verification.** Draw a live-mix sample, label it blind, and check that observed precision lands within two
percentage points of the prediction. A prediction that misses by more than that means the model's Se or Sp are also
different, and CI-02 becomes the next issue to open.

**Escalation.** If the corrected precision still supports the decision, record both numbers in the ledger. Deleting
the pilot figure destroys the audit trail that shows the correction was made.

#### CI-02 — Benchmark accuracy used as field accuracy

**Symptom.** A vendor reports 91% on a published question-answering benchmark; on the desk's own tickets the same
system answers 64% acceptably. Both numbers are honest and neither describes the other's population.

**Mechanism.** A benchmark fixes a task, a prompt convention, a scoring rule, and a population. Field use fixes none
of those unless you fix them yourself. The four differences that usually dominate are: question length (tickets are
longer and messier), answer format (benchmark expects a short span, the desk needs a procedure), reference material
(benchmark is closed-book, the desk is retrieval-augmented), and grading (exact match versus a human accept/reject).

**Competing causes.** Either the system is weaker out of domain, or the harness is weaker. The harness explanation is
embarrassing and common: a retrieval pipeline that was not indexed correctly, a prompt that omits the policy version,
a grader that rejects correct answers on formatting. The system explanation requires a domain gap.

**Discriminating check.** Run the system on 100 of our tickets twice: once with our intended pipeline, once with the
vendor's documented prompt and no retrieval. If the second run scores near the benchmark figure, the harness explains
the gap. If both runs score near 64%, the gap is the domain, and the benchmark is simply not evidence about us.

**Containment.** Do not stop the evaluation; stop the comparison. Two numbers measured under different protocols are
not a comparison, and the containment is to remove the vendor figure from the decision table.

**Correction.** Rebuild the comparison on one protocol, one ticket set, one grader, with the grader's instructions
published. Where the vendor cannot run your protocol, treat their number as a shortlisting signal only.

**Prevention.** Write the task definition and scoring rule into the evaluation protocol before the first run, and
require both parties to sign the same document. The protocol is the artefact; the score is a by-product.

**Verification.** Re-run with the agreed protocol and inspect the grader's confusion counts, not just the headline.
A gap that survives an identical protocol is a real capability difference and belongs in the market map.

**Escalation.** If the vendor resists protocol parity, that resistance is itself decision-relevant information about
the diligence relationship, and belongs in the ledger as a stated limitation.

#### CI-03 — Price per token compared across tokenisers

**Symptom.** Vendor A quotes $2.50 per million input tokens and vendor B $10.00. The naive reading is a 4x
difference, which survives until both are priced per completed ticket.

**Mechanism.** Tokenisers segment text differently. A tokeniser with a smaller vocabulary needs more tokens for the
same characters, and domain vocabulary (part numbers, policy codes) can inflate the count substantially. Output
length is also a design choice: a system instructed to answer in one sentence emits fewer tokens than one that
reasons first, and the two may reach the same accuracy.

```text
ILLUSTRATIVE ticket: 4,200 input tokens under tokeniser A, 5,460 under tokeniser B (ratio 1.30)
A: 4200 * 2.50 / 1e6 = 0.0105 ; output 600 * 10.00 / 1e6 = 0.0060 ; total 0.0165
B: 5460 * 10.00 / 1e6 = 0.0546 cannot be compared directly - B's price is not the same product
recompute B at A's token count: 5460 * 2.50/1e6 = 0.01365 - already 30% above A on input alone
```

The third line is the trap: dividing one vendor's price by the other vendor's token count produces a number that
belongs to neither. Cost per task must be measured, not divided.

**Competing causes.** A real price difference, a tokenisation difference, or a task-scope difference where one vendor
counts retrieval context and the other does not. The third is the most expensive to discover late.

**Discriminating check.** Count the same 50 tickets with each vendor's own tokeniser, then price each count at that
vendor's rate. If the per-task costs converge, the headline difference was tokenisation. If they do not, ask what
each count includes.

**Containment.** Quote cost per completed task in the decision table and strike per-token figures from it. Nothing is
lost: per-token price is a procurement detail, not a decision input.

**Correction.** Build the cost model on measured tokens per task for each candidate, with output length held at the
same target quality.

**Prevention.** Compare on cost per task at a fixed quality bar, which also removes the incentive to win on price by
answering worse.

**Verification.** Reconcile a month of invoices against the per-task model. A residual above 10% means a line item
or a token-count assumption is still wrong.

**Escalation.** When the gap survives normalisation, the cheaper vendor may still lose on review cost; run CI-10 and
CI-11 before concluding.

#### CI-04 — Unit price computed at pilot volume

**Symptom.** The business case quotes $0.48 per request; the sponsor reads it as a marginal price and plans a
250,000-request year at roughly $120,000, then receives a bill near $7,000 plus platform fees.

**Mechanism.** Any fixed cost divided by a small denominator inflates the average. The pilot ran 5,200 requests
against a $2,400 platform line, so the fixed share was $0.4615 of a $0.4789 total. At production volume the same
platform line contributes $0.0096.

```text
c_var = 0.0174
N = 5200    : fixed share 2400/5200   = 0.4615 ; average 0.4789 ; fixed share of total 96.4%
N = 250000  : fixed share 2400/250000 = 0.0096 ; average 0.0270 ; fixed share of total 35.6%
overstatement factor if the pilot average is applied at scale = 0.4789 / 0.0270 = 17.7
```

**Competing causes.** A pricing tier change at volume (some vendors price down, some add support fees), or the
denominator error above. The first is discoverable in the contract; the second is discoverable in one division.

**Discriminating check.** Ask whether the quoted figure moves when volume changes. A marginal price does not; an
average price does, and the shape of the move identifies the fixed component.

**Containment.** Present both numbers in the case, labelled, until the contract fixes which one applies. A single
unlabelled unit price is the defect.

**Correction.** Restate the case as marginal cost plus a fixed block, then compute the average at the contracted
volume. The two-scenario presentation also makes the commitment decision visible.

**Prevention.** Require every unit price to carry its volume and its cost basis. A template field is enough; the
failure is cultural, not analytical.

**Verification.** Rebuild last month's invoice from the model at actual volume. Agreement within 5% means the fixed
and variable split is right.

**Escalation.** If the fixed line is shared across several programmes, apportion it explicitly; an unapportioned
shared cost makes every individual programme look unprofitable and the portfolio look impossible.

#### CI-05 — Stage percentiles added together

**Symptom.** Two stages each report a 900 ms p95, the end-to-end dashboard reports a 2,100 ms p95, and the SLO at
2,000 ms is breached while every stage claims to meet its own target.

**Mechanism.** Percentiles are not additive. The sum of stage p95s is an upper bound that holds only when the stage
tails align; the true joint percentile depends on the correlation. With light-tailed, roughly independent stages the
sum over-provisions; with shared resources it can under-provision.

```text
stage means 300 and 300, sd 120 each, constant 0
additive      : 2 * (300 + 1.645*120) = 2 * 497.4 = 994.8
independent   : mu 600, sigma sqrt(2*14400) = sqrt(28800) = 169.7
                p95 = 600 + 1.645*169.7 = 600 + 279.2 = 879.2
gap additive minus independent = 994.8 - 879.2 = 115.6 ms (13.1% of the independent figure)
```

**Competing causes.** Genuine tail correlation from a shared host or connection pool, or a measurement artefact where
each stage's p95 was computed over a different window than the end-to-end number. The window mismatch is common when
stages are owned by different teams.

**Discriminating check.** Rebuild the end-to-end percentile from the raw per-request traces. If the recomputed p95
matches the additive figure, the stages really do move together; if it matches the independent figure, the dashboard
is adding percentiles somewhere upstream.

**Containment.** Publish the measured joint p95 as the SLO metric and remove the summed figure from the dashboard.
Two conflicting latency numbers on one screen guarantee an unresolvable argument.

**Correction.** Budget from the measured joint distribution, and keep per-stage distributions for attribution. The
per-stage numbers answer "who is slow"; only the joint number answers "are we inside the SLO".

**Prevention.** Store per-request stage timings for a sample of traffic, not just per-stage summaries. Summaries
cannot be recombined; samples can.

**Verification.** Compare the model's predicted p95 with a fresh measured p95 over a week of traffic. A 10% miss
indicates the correlation assumption or the tail shape has changed.

**Escalation.** If a hard timeout sits near the SLO, latency and success rate must be read together — truncation
lowers measured latency by converting slow successes into fast failures.

#### CI-06 — List price treated as effective price

**Symptom.** The price sheet says $0.50 per thousand requests; the invoice implies $0.71. The difference is a
committed-tier discount that was never reached, plus an overage rate applied to the excess.

**Mechanism.** Tiered contracts price the first block high and later blocks low, then apply a punitive rate above the
commitment. The effective price therefore depends on the volume distribution, not on the volume total. A year that
averages 20,000 requests a month but peaks at 40,000 pays overage in the peak months and never earns the discount
that the annual total appears to justify.

**Competing causes.** A billing error, a tier missed by a small margin, or a genuinely different line item (support,
egress, storage) folded into the same per-request figure.

**Discriminating check.** Read the invoice lines in order and classify each as fixed, tiered, or overage. An overage
line present in the peak months settles the question without any modelling.

**Containment.** Stop quoting a single price. Use three scenarios — pilot, expected, peak — with the overage rate
applied in the peak.

**Correction.** Model effective price from the monthly volume distribution, not from the annual mean, and include the
commitment decision as an explicit choice with its downside.

**Prevention.** Require the vendor's price schedule to be reproduced as a piecewise function in the business case.
If the schedule cannot be reproduced, the case is not ready.

**Verification.** Rebuild an invoice from the piecewise model. A 5% residual indicates a missing line or a wrong
tier boundary.

**Escalation.** Where the overage rate exceeds twice the tier rate, the contract is pricing variance, not volume, and
the portfolio owner should decide whether to smooth demand or renegotiate.

#### CI-07 — Break-even computed at pilot volume

**Symptom.** A programme with a genuine per-ticket margin is declared unviable because break-even was computed while
the fixed cost was spread over 1,200 tickets.

**Mechanism.** Break-even volume is fixed cost divided by contribution margin. Change the volume in the denominator
of the *average* cost and you change the answer, but the correct procedure solves for volume instead: N* = C / m,
where m is the per-ticket contribution and C the fixed block. Using the pilot's average cost as if it were m
double-counts the fixed block.

```text
m = 1.5739 (benefit 1.5913 minus variable 0.0174)
C = 2400
N* ignoring review      = 2400 / 1.5739 = 1524.8 -> 1525 tickets/month
with review m' = 1.2242 : 2400 / 1.2242   = 1960.4 -> 1961 tickets/month
current volume 5200 -> headroom 5200 - 1961 = 3239 tickets/month (62.3% above break-even)
```

**Competing causes.** A genuinely thin margin, or a mis-specified margin that includes fixed cost. The two are
separated by asking whether the quantity changes when volume changes: contribution margin does not.

**Discriminating check.** Compute break-even twice, once with the average cost from the pilot and once with the
contribution margin. If the two disagree on viability, the pilot's denominator is in the wrong place.

**Containment.** Withhold the viability verdict, not the programme. A verdict built on the wrong denominator is worse
than a delay, because it is actionable and wrong.

**Correction.** Present break-even as a volume, with its parameter set, alongside the current volume. The distance
between the two is the real risk statement.

**Prevention.** Make break-even volume a mandatory line in the business case template. It forces the fixed and
variable split that most cases skip.

**Verification.** Recompute break-even with the actual invoice for fixed costs and measured adoption for the margin.
A move of more than 20% means a parameter, not arithmetic, was wrong.

**Escalation.** If break-even sits within 15% of current volume, the programme is a volume bet; the owner should
decide it as such rather than as a technology bet.

#### CI-08 — Saved hours booked as cash

**Symptom.** The case claims $5,379 per month of benefit from 86.8 saved hours; finance finds no line item that
changes, because the agents are salaried and the queue still exists.

**Mechanism.** Time saved converts to money only through one of four channels: reduced overtime or contract hours,
reduced headcount at the next hiring cycle, absorbed growth without new hires, or reduced customer wait that has a
measured value. Without a named channel the benefit is a capacity statement, which is useful and not cash.

```text
saved hours per month = 5200 * 0.65 * 1.54 / 60 = 86.753 h
cash value at 62.00/h  = 5378.70
cash value if only 30% of freed time lands in a paid channel = 0.30 * 5378.70 = 1613.61
difference = 5378.70 - 1613.61 = 3765.09 per month, i.e. 70% of the claimed benefit
```

**Competing causes.** The benefit is real but in a channel finance does not book (waiting time), or the benefit is
real and the channel exists but the volume is too small to cross a hiring threshold. The second produces a step
function, which is why a linear benefit claim misleads even when the saving is genuine.

**Discriminating check.** Ask which budget line changes next quarter if the saving disappears. If no line changes in
either direction, the saving is capacity, and it should be booked as capacity.

**Containment.** Split the benefit line into cash and capacity in the case document, with the conversion rule stated.
Reviewers can then weight each part appropriately.

**Correction.** Recompute the case with only the cash channel in the payback calculation, and present capacity as a
secondary benefit with its own condition (a hiring cycle that does not happen).

**Prevention.** Declare the conversion rule before the pilot, because after the pilot the team has an incentive to
choose the most favourable one.

**Verification.** Finance confirms the specific line, or the case is reissued with the cash benefit removed. Either
outcome is a pass; the failure mode is an unverified benefit that persists in a slide deck.

**Escalation.** Capacity benefits that require a hiring decision belong to the workforce owner, not the programme
owner; route the claim accordingly.

#### CI-09 — Adoption assumed at one hundred percent

**Symptom.** Benefit was modelled on every eligible ticket; four weeks of live data show 41% of eligible tickets take
the assisted path, and the payback period doubles.

**Mechanism.** Adoption is a behaviour, not a property of the tool. It depends on interface friction, shift patterns,
trust after the first bad answer, and whether the assisted path is the default. A tool that is technically correct
and one click away still loses to habit.

```text
eligible tickets = 5200 * 0.65 = 3380
benefit at 100% adoption = 3380 * 1.591333 = 5378.71
benefit at 65% adoption  = 2197 * 1.591333 = 3496.16
benefit at 41% adoption  = 1385.8 * 1.591333 = 2205.4
claimed-to-measured ratio at 41% = 5378.71 / 2205.4 = 2.44
```

**Competing causes.** Adoption genuinely low, or the eligibility definition wrong so that the denominator includes
tickets the tool cannot help with. The second is a measurement artefact that makes a successful launch look like a
failed one.

**Discriminating check.** Measure adoption within the *strict* eligible set — tickets where the tool demonstrably
had the needed reference material. If adoption is high there and low overall, the eligibility definition is the
defect.

**Containment.** Report the pilot on the strict set while the eligibility definition is repaired, with both
denominators visible.

**Correction.** Recompute the case using the measured adoption curve and its trajectory, not its first-month value.
Adoption curves usually have a ramp; a single early point is a poor forecast in both directions.

**Prevention.** Instrument the assisted path before launch so adoption is measurable from day one. Without that
instrument, adoption can only be estimated, and estimates are what produced the defect.

**Verification.** Four consecutive weeks of adoption data with the strict and broad denominators, both stable enough
to forecast from.

**Escalation.** If adoption is flat below 25% after eight weeks despite a working tool, the limiting factor is
process or trust; further model work will not move it.

#### CI-10 — Review capacity omitted from the cost model

**Symptom.** A design with good model metrics costs 42% more per month than the model predicted, and the gap equals
the human review line that nobody modelled.

**Mechanism.** Every escalation is work. At a 3% base rate with Se 0.88 and Sp 0.94, 8.46% of items are flagged, so a
5,200-ticket month generates 440 reviews. Four minutes each is 29.3 hours, which is 0.73 of a full-time reviewer and
is not free merely because it is small.

```text
flag rate = 0.0264 + 0.0582 = 0.0846
flagged per month = 0.0846 * 5200 = 439.9 -> 440
review hours = 440 * 4 / 60 = 29.333
review cost = 29.333 * 62.00 = 1818.67 per month
share of total monthly cost = 1818.67 / 4309.15 = 42.2%
```

**Competing causes.** Review cost genuinely absent from the design, or review counted in a different budget so it
does not appear in the programme's own figure. The second is an accounting boundary, and it matters because someone
still pays.

**Discriminating check.** Multiply the flag rate by volume and by minutes per review, then compare with the reviewer
roster. A roster that exists but is unbilled is the boundary case; a roster that does not exist is the design case.

**Containment.** Publish the review hours per month as a first-class metric. It converts an invisible constraint into
a visible one, and it caps the design honestly.

**Correction.** Add the review line to the cost model and, where the line is large, reduce the flag rate rather than
the review time. Flag rate is a threshold decision; review time is a workload consequence.

**Prevention.** Model review as a cost with its own sensitivity row. Any case where review is under 5% of total cost
at a low base rate should be re-examined, because that combination is unusual.

**Verification.** Compare modelled review hours against the actual roster hours for a month. Agreement within 15% is
adequate; a larger gap means either the flag rate or the minutes per review is wrong.

**Escalation.** When review hours exceed one full-time equivalent, the programme has a staffing decision to make and
should not proceed on a technology decision alone.

#### CI-11 — Threshold chosen on accuracy instead of cost

**Symptom.** Version C scores the highest measured accuracy of three candidates and is the most expensive one to
operate, because the accuracy metric weights a 4.10 false alarm and a 400.00 missed case as if they were equally bad.

**Mechanism.** Accuracy counts errors, cost prices them. At a 3% base rate the two error populations also differ in
size by a factor of 32, so the accuracy metric's implicit weights (0.03 for a miss, 0.97 for a false alarm) are almost
the exact inverse of the cost weights (400.00 for a miss, 4.10 for a false alarm).

```text
per 100 items at p = 0.03, cost of a miss 400.00, cost of a false alarm 4.10
A: se .88 sp .94 -> misses 0.36, alarms 5.82  -> 144.00 + 23.86 = 167.86 ; accuracy 0.9382
B: se .96 sp .80 -> misses 0.12, alarms 19.40 -> 48.00 + 79.54 = 127.54 ; accuracy 0.8048
C: se .75 sp .99 -> misses 0.75, alarms 0.97  -> 300.00 + 3.98 = 303.98 ; accuracy 0.9828
```

Accuracy ranks C > A > B. Cost ranks B < A < C. The order is exactly reversed, and the reversal holds at p = 0.15 as
well, where the three costs are 740.91, 309.70 and 1503.48. A scorecard sorted by accuracy therefore recommends the
most expensive configuration with confidence, and no amount of extra evaluation data changes that — the metric is
the defect, not the sample.

The crossover between A and B is worth knowing because it sets which side of the argument a programme is on: B beats
A when 0.24 * c_miss exceeds 13.58 * 4.10 = 55.68, i.e. when c_miss exceeds 232.00. Below that miss cost the more
specific configuration wins; above it, the more sensitive one does.

**Competing causes.** The cost weights are wrong, or the cost weights are right and the metric is wrong. The second
is far more common, and it is invisible when only one metric is reported.

**Discriminating check.** Plot cost per 100 against the threshold for the candidate configurations. If the minimum
cost threshold differs from the maximum-accuracy threshold, the metric is the defect.

**Containment.** Freeze the current threshold until the cost matrix is agreed. Moving a threshold without the matrix
is how false-positive floods get shipped.

**Correction.** Adopt the cost-minimising threshold, publish the cost matrix next to it, and re-derive the review
capacity for the new flag rate.

**Prevention.** Require the two error costs in the evaluation record, even when they are estimates with a stated
range. An unpriced error cannot be optimised for.

**Verification.** After two weeks at the new threshold, compare realised review hours and missed-case counts with the
matrix's prediction. A large miss means a cost weight, not the threshold, needs revisiting.

**Escalation.** When the two error costs are contested, present the threshold as a function of their ratio rather than
as a single number; that converts a political argument into a sensitivity curve.

#### CI-12 — Retry amplification metered as steady-state load

**Symptom.** Capacity planning at a 4% per-attempt failure rate holds; during a partial dependency failure the service
saturates at a load that should have been survivable.

**Mechanism.** Each retry layer multiplies attempts by 1/(1-q). With three independent layers and q = 0.06, attempts
per logical request are 1.204 — a 20% surcharge exactly when everything is already slower. The surcharge is not the
main effect: the extra attempts consume the same pool that the healthy requests need, so the effective q rises, which
increases the multiplier again.

```text
q = 0.06 : per layer 1/0.94 = 1.0638 ; three layers 1.0638^3 = 1.2041
extra attempts at 5200 requests = 0.2041 * 5200 = 1061.3
variable cost of those extra attempts = 1061.3 * 0.0174 = 18.47 per month at steady state
during a q = 0.30 episode : 1/0.70 = 1.4286 ; ^3 = 2.9155 -> 9,960 attempts for the same 5,200 requests
```

**Competing causes.** A genuine capacity shortfall, or the amplification above. Both present as saturation with
rising latency, and they are separated by counting attempts per logical request rather than requests.

**Discriminating check.** Divide total downstream attempts by total logical requests during the episode. A ratio near
3 identifies amplification; a ratio near 1 with high latency identifies shortfall.

**Containment.** Disable retries at all but one layer, and add jitter. Removing the redundant layers costs almost
nothing in success probability once the surviving layer has backoff.

**Correction.** Keep a single retry owner per call path, cap attempts explicitly, and budget capacity at the degraded
failure rate rather than the healthy one.

**Prevention.** Put the attempt multiplier in the capacity model as a variable, not a constant. The number is a
function of q, and q is the thing that moves during degradation.

**Verification.** Induce a 10% dependency failure in a test environment and measure the attempt multiplier. A measured
multiplier near the model's prediction validates the capacity budget.

**Escalation.** Where the multiplier exceeds 2 at the design failure rate, the budget must include it or the service
will shed load by timing out, which converts a latency incident into an availability incident.

#### CI-13 — Multi-turn session cost modelled per turn

**Symptom.** The case prices 12 turns at the single-turn rate and is out by 45% of the session cost; the monthly bill
exceeds the model by a figure that grows as sessions get longer.

**Mechanism.** When every turn re-sends the accumulated history, input tokens grow linearly per turn and quadratically
per session. The per-turn model sees the first term and misses the second. Prompt caching removes most of the term
when the provider supports it, which is why the same arithmetic can be right for one vendor and wrong for another.

```text
base prompt 3400 tokens, per-turn addition 480, 12 turns
input tokens = 12*3400 + 480*(12*11/2) = 40800 + 31680 = 72480
session cost  = 72480*3.00/1e6 + 12*480*15.00/1e6 = 0.21744 + 0.0864 = 0.30384
naive model   = 12*(3400*3.00/1e6 + 480*15.00/1e6) = 12*(0.0102 + 0.0072) = 0.2088
understatement = 1 - 0.2088/0.30384 = 31.3% of the true cost; the true cost is 45.5% above the model
```

**Competing causes.** Longer sessions than assumed, or the quadratic term missing from the model. Separating them
takes one log query: compute the mean turn count and compare it with the 12 used in the case.

**Discriminating check.** Sum input tokens per session directly from the request log and divide by session count.
If the measured mean is well above 12 times the base prompt, the quadratic term is real regardless of assumed turns.

**Containment.** Cap session history at a stated window and reset context at task boundaries. The cap is a product
decision as much as a cost decision, because it changes answer quality on long sessions.

**Correction.** Replace the per-turn model with the session model, and price the truncation policy as a quality
trade-off rather than a free saving.

**Prevention.** Model cost per session, at a stated turn distribution. A mean turn count with no distribution hides
exactly the sessions that dominate the bill.

**Verification.** Reconcile one month of sessions against the model, segmented by turn count. Agreement within 5%
overall is not sufficient if the long-session segment is off by more than 20%.

**Escalation.** When the long-session segment is the majority of spend, the pricing structure favours a different
architecture — summarise history rather than re-send it — and that is a design decision for the owner.

#### CI-14 — Worker count provisioned without Little's law

**Symptom.** Every request completes in under two seconds, the mean latency is healthy, and yet the queue depth rises
monotonically through the working day and drains only overnight.

**Mechanism.** Concurrency needed in steady state is the product of arrival rate and time in system. At 20 requests
per second and 1.7 seconds per request, 34 workers are required merely to keep the queue flat. Provisioning 28 gives
a shortfall of 6 request-slots; the queue grows at roughly the rate at which arrivals exceed the achievable service
rate, which is not obvious from per-request latency at all.

```text
required concurrency L = lambda * W = 20 * 1.7 = 34
provisioned = 28 -> effective service rate = 28 / 1.7 = 16.47 req/s vs 20 req/s arriving
growth rate = 20 - 16.47 = 3.53 req/s ; over a 5-hour day = 3.53 * 18000 = 63,540 queued requests
```

That last figure is why the metric looks catastrophic at the end of the day and fine at the start: the backlog is an
integral, and the per-request latency never registers it.

**Competing causes.** A slow dependency, a worker leak, or the concurrency shortfall. The first two show up as rising
service time; the third shows rising queue depth at constant service time.

**Discriminating check.** Compare service time (time from worker pickup to completion) across the day. If it is flat
while queue depth rises, concurrency is short. If it rises with the queue, the dependency is degrading under load.

**Containment.** Cap admission rate to the sustainable service rate and reject or defer the excess with a reason
code. This converts an unbounded queue into a bounded one with visible loss, which is the lesser harm.

**Correction.** Provision from the measured product, then add headroom for the p95 service time rather than the mean,
because a queue is drained by tail capacity.

**Prevention.** Derive worker count in the capacity document from lambda and W, both measured. A number chosen by
feel is a number that cannot be defended during a growth review.

**Verification.** Hold a constant arrival rate for two hours and confirm queue depth is flat. A slow drift upward
means W was measured under lighter conditions than production.

**Escalation.** If the required concurrency exceeds the budget, the decision is admission control or a latency target
change; adding workers indefinitely converts a design problem into a cost problem.

#### CI-15 — Three different quantities called latency

**Symptom.** A review meeting spends forty minutes with two teams quoting 300 ms and 1,700 ms for the same path,
both correct, neither measuring the same thing.

**Mechanism.** Waiting time (queue), service time (processing), and user-perceived time (client to render) are three
different random variables with three different distributions. Percentiles of different variables cannot be compared,
and the discrepancy is largest exactly when the system is under strain, because queue time inflates first.

```text
service time p50 = 260 ms, queue wait p50 = 40 ms -> user-perceived p50 ~ 300 ms
under load, queue wait p95 = 1,400 ms, service p95 = 300 ms -> perceived p95 ~ 1,700 ms
ratio of perceived to service p95 = 1700 / 300 = 5.67
```

**Competing causes.** Measurement boundary differences (client timer versus server timer), percentiles from different
windows, or genuine queueing. All three present as "the numbers disagree", and only the third is a capacity problem.

**Discriminating check.** Instrument the boundary explicitly: timestamp at request receipt, at worker pickup, at
response write, and at client render. Four timestamps turn a dispute into a subtraction.

**Containment.** Freeze both numbers on the dashboard with their names attached, side by side. Naming is the
containment because the argument is about vocabulary, not about the system.

**Correction.** Adopt one metric per promise: an SLO on user-perceived p95, an internal alert on queue wait, and a
service-time budget per stage. Each has an owner and a threshold.

**Prevention.** Define every latency metric by its two timestamps in the metric catalogue. A metric defined by a name
alone will be redefined by whoever is under pressure.

**Verification.** Sum the three intervals for a sample of traces and confirm they equal the end-to-end client figure
within clock-skew tolerance. A residual larger than a few milliseconds means a missing segment.

**Escalation.** When queue wait dominates, the fix is capacity, not code optimisation; optimising service time under a
queueing constraint produces a visible improvement in a metric that does not move the user's experience.

#### CI-16 — Component SLA quoted as chain availability

**Symptom.** Three dependencies each advertise 99.9%, the composite fails its 99.9% target, and the shortfall is
attributed to bad luck rather than arithmetic.

**Mechanism.** Availability of a synchronous chain is the product of its stages. Three at 99.9% give 99.70%, which is
2.7 times the downtime of one. Adding a fourth at 99.5% gives 99.20%, which is nearly eight times.

```text
0.999 * 0.997 * 0.995 = 0.991023
downtime = (1 - 0.991023) * 43200 = 387.8 minutes/month
single 99.9% reference = 43.2 minutes/month
ratio = 387.8 / 43.2 = 8.98
budget for 99.9% composite = 43.2 minutes/month, so the chain is over budget by 344.6 minutes
```

**Competing causes.** Correlated failures in one zone (which make it worse than the product), or asynchronous stages
that do not actually chain (which make it better). The architecture, not the SLAs, decides which.

**Discriminating check.** Draw the dependency graph for the success path and mark each edge synchronous or
asynchronous. Only the synchronous path multiplies; a queue between stages converts a chain into a set of
independently available stages.

**Containment.** Set the composite target from the chain product and state the gap explicitly, rather than promising
a number the architecture cannot deliver.

**Correction.** Either remove a synchronous stage (cache, queue, or degrade gracefully) or accept the arithmetic and
publish the resulting target.

**Prevention.** Compute chain availability in the architecture document, with the per-stage numbers and their source.
A composite target without the product is a wish.

**Verification.** Track measured composite availability monthly and compare with the product prediction. Measured
below predicted points at correlation, which needs a failure-domain review rather than more redundancy.

**Escalation.** When a stage's SLA is the binding constraint, that is a vendor-management conversation; when two
stages share a failure domain, it is an architecture conversation. They have different owners and different fixes.

#### CI-17 — Decimal and binary prefixes in the same column

**Symptom.** A storage estimate of 4.2 GB is challenged as "only 3.9 GB" by an engineer reading the console, and the
egress bill is 7.4% above the estimate for reasons nobody investigates.

**Mechanism.** GB is 10^9 bytes and GiB is 2^30 bytes, a ratio of 1.0737. Cloud billing commonly uses decimal
prefixes while operating systems and many consoles display binary ones. The gap is larger than most negotiation
margins, so a cost model built on console readings systematically under-forecasts.

```text
4.2e9 bytes = 4.2 GB decimal = 4.2e9 / 1.073741824e9 = 3.9116 GiB
understatement if the console figure is used as the billing figure = 4.2 / 3.9116 - 1 = 7.4%
on a 9,000 currency unit monthly storage bill the gap is 666 currency units per month
```

**Competing causes.** Replication factor, compression, or the prefix confusion. Replication is a multiplicative
factor of two or three, so it is rarely mistaken for 7%; the confusion is the usual explanation for a small
persistent gap.

**Discriminating check.** Convert the same byte count under both prefixes and see which matches the invoice. That is
a two-line calculation and it settles the question.

**Containment.** Put the unit in the column header of every cost table, and never carry a bare number between systems.

**Correction.** Convert all storage figures to the billing unit at the point of entry, so downstream arithmetic has
one convention.

**Prevention.** Store bytes as integers internally and format at the edge. Intermediate formatted values are where
prefix errors enter.

**Verification.** Reconcile one month's bill from raw byte counts. Agreement within 2% confirms the convention;
a 7% residual indicates the prefix was applied twice in one direction.

**Escalation.** Where a vendor quotes in one prefix and bills in another, write both into the contract note; the
ambiguity is a commercial risk, not a rounding error.

#### CI-18 — Pooled comparison hiding a segment reversal

**Symptom.** Report A shows version X at 77% and version Y at 63%, so X is declared better; a per-segment table shows
Y ahead on both segments.

**Mechanism.** When the two versions were evaluated on different mixes of segments, the pooled figure is a weighted
average whose weights differ. If the segments also differ in difficulty, the pooled ordering can invert the
segment-wise ordering — Simpson's reversal. It is not a paradox; it is a weighted average doing exactly what
weighted averages do.

```text
segment H (large, easy): X 900 items at 0.80, Y 100 items at 0.90
segment L (small, hard): X 100 items at 0.50, Y 900 items at 0.60
pooled X = (900*0.80 + 100*0.50)/1000 = 770/1000 = 0.77
pooled Y = (100*0.90 + 900*0.60)/1000 = 630/1000 = 0.63
segment-wise: Y leads by 0.10 and 0.10; pooled: X leads by 0.14
```

**Competing causes.** Different mixes (as above), or a genuine segment interaction where a version is better
everywhere but the evaluation sets differ in size by an order of magnitude.

**Discriminating check.** Recompute both pooled and per-segment figures from the same rows of counts. If the
segment table and the pooled table come from different row sets, the comparison is void before any statistics.

**Containment.** Remove the pooled headline from the decision table until the table is rebuilt with a single row set.

**Correction.** Report per-segment results with the segment sizes, and give the pooled figure only alongside the
weights. If the decision must be pooled, reweight to the live mix (CI-19) rather than to the evaluation mix.

**Prevention.** Publish the segment table with every pooled figure, as a template rule. The table is cheap; the
reversal is expensive because it usually picks the wrong vendor.

**Verification.** Confirm that the pooled figure can be reconstructed exactly from the segment rows. Any residual
means an unlisted segment exists, which is the defect that produced the reversal.

**Escalation.** When the live mix is dominated by one segment, that segment's performance is the decision; the pooled
figure is a reporting convenience and should not carry the decision alone.

#### CI-19 — Stratified estimate weighted to the wrong mix

**Symptom.** The evaluation reports 81% agreement; live sampling finds 75%. Both numbers are correct applications of
the same per-segment results, with different weights.

**Mechanism.** A stratified sample reports per-stratum results with weights. If the weights come from the evaluation
design, which necessarily over-samples rare strata, the estimate describes the evaluation population. Applying the
live weights re-targets the estimate to the deployed population.

```text
stratum A share in eval 0.75, result 0.86
stratum B share in eval 0.25, result 0.66
eval-weighted = 0.75*0.86 + 0.25*0.66 = 0.645 + 0.165 = 0.810
live shares A 0.45, B 0.55 -> 0.45*0.86 + 0.55*0.66 = 0.387 + 0.363 = 0.750
gap = 0.810 - 0.750 = 0.060 -> 6.0 points
```

**Competing causes.** Wrong weights, or a genuine change in live segment performance. The distinction matters because
only the second needs model work.

**Discriminating check.** Recompute the per-segment results on live data. If they match the evaluation's per-segment
results, only the weights were wrong; if they moved, the model is behaving differently on live input.

**Containment.** Report both the evaluation-weighted and live-weighted estimates with their weights visible, and
pause any decision that depends on the difference between them.

**Correction.** Store weights with every stratified result, and recompute the headline from live weights at each
reporting cycle.

**Prevention.** Make the weights a stored field, not a computed afterthought. A result without weights cannot be
retargeted and will be quoted at the wrong population indefinitely.

**Verification.** Compare the live-weighted estimate with a small blind live sample. Agreement within two points
means the weights and the per-segment results are both current.

**Escalation.** Where the two estimates straddle a decision threshold, the decision should be deferred or made
provisionally with a review date; either is honest, and choosing the favourable weight is not.

#### CI-20 — Convenience sample with a confident interval

**Symptom.** An evaluation on 300 items reports a 4.1-point margin of error, and the result is wrong by 12 points
against live performance. The interval is not miscalculated; the frame is wrong.

**Mechanism.** Every sampling formula assumes the sample was drawn from the target population. A convenience frame —
last quarter's tickets, one team's queue, the items an engineer found interesting — satisfies the arithmetic and not
the assumption. The interval quantifies sampling noise and says nothing about frame bias, which is usually larger.

```text
frame: evaluation set 72% short tickets, production 9% short tickets
interval for n = 300 at p = 0.84 : 1.96 * sqrt(0.84*0.16/300) = 1.96 * 0.02149 = 0.0421 -> +-4.2 points
observed live gap = 12 points, i.e. 12 / 4.2 = 2.86 times the stated margin
```

A gap 2.86 times the sampling interval is the signature of a frame defect rather than bad luck. That relation is the
useful part: intervals bound noise, and a miss beyond them points at the design.

**Competing causes.** Genuine sampling noise (which would be within the interval), frame bias (as above), or a
production change between evaluation and now.

**Discriminating check.** Compare the composition of the sample against production on three dimensions — length,
segment, and source system. A composition gap larger than the claimed effect size makes the estimate unusable.

**Containment.** Withhold the estimate from the decision, and state the frame limitation explicitly rather than
widening the interval.

**Correction.** Redraw from the production frame, stratified, with the composition checked before the labelling work
is paid for.

**Prevention.** Declare frame and composition targets in the protocol. A sample that fails the composition check is
rejected before annotation, which is where the money is.

**Verification.** Composition within five points of production on the declared dimensions, with the check recorded
alongside the result.

**Escalation.** Where the production frame cannot be enumerated — for example, items never logged — the estimate is
unavailable and the correct output is a logging requirement, not a number.

#### CI-21 — Accuracy drift with no model change

**Symptom.** Measured accuracy falls four points over three months while the model, prompt, and retrieval index are
unchanged, and no deployment happened.

**Mechanism.** The measurement changed even though the system did not. Label definition drift, annotator turnover,
and a reference set that no longer matches the current policy all move the number. A reference set frozen eighteen
months ago encodes an older policy, so a good answer to today's policy scores as wrong.

**Competing causes.** Model drift through input distribution change, label drift, and reference staleness. All three
produce a gradual decline and a stable-looking system.

**Discriminating check.** Double-label a 50-item subset under the current guide and again under the historic guide.
If agreement between the two guides falls below the floor, the label definition moved; if both guides agree and the
system's answers disagree, the system is answering an outdated policy.

**Containment.** Annotate the reported figure with the guide version it was measured under. That alone stops the
number being compared across incompatible cycles.

**Correction.** Version the label guide with the reference set, re-baseline under the current guide, and record the
conversion so the historic series remains interpretable.

**Prevention.** Review the reference set on a schedule tied to policy change, and require a fresh double-label audit
at each review.

**Verification.** Inter-annotator agreement restored above the declared floor, and the re-baselined figure reproduced
by a second annotator pair.

**Escalation.** When the policy itself changed, the old reference set is obsolete and its history should be archived
rather than reconciled; reconciling it produces a false continuous series.

#### CI-22 — Threshold acting as an approval

**Symptom.** A refund was issued automatically because the model's confidence cleared the threshold, and the ledger
records the model as the actor. Nobody signed, and the process owner's mandate did not cover automated refunds.

**Mechanism.** A threshold is a numeric cut on a score. Authority is a property of a role and a policy version. Wiring
the first to an action without the second means the policy is de facto defined by a modelling parameter, which nobody
reviewed and which changes whenever the model changes.

**Competing causes.** A policy configuration error (the action class was misclassified), or a design that never had a
separate approval concept. The second is the common one, because the threshold path is easy to build and the approval
path is not.

**Discriminating check.** Read the ledger for one action: actor, policy version, approver field. An actor field
containing a model identifier, or an empty approver field, settles it.

**Containment.** Disable the automated action class and route those items to human review. This is a hard containment
because an unapproved action has already occurred once; a second occurrence is a governance failure, not a defect.

**Correction.** Separate the score from the mandate: the score produces a recommendation, a named role approves, and
the ledger records both. The threshold still helps prioritise; it no longer authorises.

**Prevention.** Define action classes explicitly, with the role permitted for each, and treat any new action as a
policy change requiring approval. The design should make an unapproved action impossible to express.

**Verification.** Sample twenty recent actions in the class and confirm every one has a named approver and a policy
version consistent with the approval record.

**Escalation.** Any occurrence involving money, health, legal status, or an external party goes to the risk owner
immediately, regardless of magnitude; the pattern, not the amount, is what matters.

#### CI-23 — Retries creating duplicate effects

**Symptom.** After a dependency hiccup, four tickets receive two replies each and one refund is issued twice, while
the request count in the logs looks normal.

**Mechanism.** A retry that repeats a request without a stable key repeats the effect. Under 2,400 requests with a
0.4% retry rate, the expected duplicate count is 2400 * 0.004 = 9.6 — about ten duplicates per hiccup, each one a
customer-visible defect and each one cheap to prevent and expensive to explain.

**Competing causes.** Missing deduplication, or an idempotency key that is unstable (contains a timestamp or a random
component, so the retry looks like a new request). The second is more insidious because the code appears to implement
deduplication.

**Discriminating check.** Replay one request key twice and count durable effects. Then inspect the key's construction
for time or randomness. Both observations are needed: the first proves the defect, the second locates it.

**Containment.** Route duplicate-prone action classes to a manual confirmation step while the key is repaired. Manual
confirmation is slow and safe, which is the right trade during an open defect.

**Correction.** Persist the deduplication record before performing the effect, keyed on a stable identifier derived
from the logical request. Order matters: writing the record after the effect leaves a crash window that produces the
duplicate.

**Prevention.** Test the retry path with an effect counter in continuous integration. A retry test that asserts only
on the response body will pass while duplicates are produced.

**Verification.** Force a retry on a keyed request and confirm effect count equals one, including a kill between
record and effect.

**Escalation.** If duplicates reached an external system, the correction needs a reconciliation run, not just a code
change; unreconciled duplicates persist in downstream records indefinitely.

#### CI-24 — A one-point difference called an improvement

**Symptom.** Version B scores 86% against version A's 84% and is declared superior; on a larger run the gap is 0.6
points with heavily overlapping intervals.

**Mechanism.** The standard error of a proportion shrinks with the square root of n, so a comparison made on 300
items cannot resolve a two-point difference. The interval at n = 300 is about 4.1 points, which is twice the claimed
effect.

```text
n = 300,  p = 0.84 : 1.96*sqrt(0.84*0.16/300) = 1.96*0.021166 = 0.0415 -> +-4.1 points
n = 1200, p = 0.86 : 1.96*sqrt(0.86*0.14/1200) = 1.96*0.010017 = 0.0196 -> +-1.96 points
required n for a 2-point margin at p = 0.85 : 0.85*0.15*(1.96/0.02)^2 = 0.1275*9604 = 1224.5 -> 1,225 items per arm
```

**Competing causes.** A real but small effect, or noise. The distinguishing feature is reproducibility: a real effect
of 2 points reproduces on a second independent sample, and noise does not.

**Discriminating check.** Compute both intervals and check for overlap. Overlapping intervals do not prove equality,
but they do forbid a directional claim at the same confidence.

**Containment.** Label the comparison "not resolved at this sample size" and keep both versions in the running. That
is a decision, and it is cheaper than a rollback.

**Correction.** Either increase n to the required figure (1,225 per arm for a 2-point margin at 95% confidence) or
restate the claim as a range.

**Prevention.** Declare the minimum detectable effect and the required n before the run. Post-hoc power discussion is
how one-point differences become strategy.

**Verification.** Re-run on a fresh sample and check whether the gap persists within the predicted interval. A
vanishing gap confirms the original was noise.

**Escalation.** Where a two-point difference would change a business decision, the evaluation budget must be sized to
resolve it; asking a 300-item test to settle a 2-point question is a budget decision dressed as an analysis.

#### CI-25 — Benchmark cited as field evidence

**Symptom.** A slide states "the system is 91% accurate" with no population, task, or rung attached, and the number is
used to justify a workflow change in a domain the benchmark never covered.

**Mechanism.** Numbers lose their provenance when they move between documents. A published benchmark figure is an E1
statement: it can shortlist a candidate and cannot size an outcome. The compression happens because the qualifier is
long and the slide is short, and the incentive to compress is strongest for the most favourable numbers.

```text
benchmark 91% (E1, closed-book, short answers, exact match)
field      64% (E2, n = 300, retrieval-augmented, human accept/reject)
gap = 27 points; ratio 0.91/0.64 = 1.42
decision that a 91% figure supports: include this vendor in a shortlist
decision that it does not support: replace a human review step
```

**Competing causes.** Deliberate compression, or an honest belief that a benchmark implies field performance. The
second is more common and is fixed by the rung vocabulary, not by admonition.

**Discriminating check.** Ask what population, task, and scoring rule the figure describes, and which rung it sits on.
If the answer is "a published set, exact match", the figure cannot support a field claim, however accurate it is.

**Containment.** Tag every number in the case with its rung. A slide where each figure carries a tag resists the
compression because the omission becomes visible.

**Correction.** Retitle the claim to its rung and add the field measurement as a separate line, so the shortlisting
signal and the sizing signal are both present and clearly different.

**Prevention.** Make the rung tag part of the number's format in the template. Format is stronger than policy, because
it fails visibly.

**Verification.** Sample ten figures from the finished case and check each has a rung and a population. Any untagged
figure is a defect even if the figure is correct.

**Escalation.** Where a decision already rests on an E1 number, the correction is a re-decision, not an annotation;
annotating a decision that has been made on the wrong rung leaves the decision wrong and the slide accurate.

## 9. Complexity & Cost Analysis

### 9.1 Cost model for a diagnostic check

Diagnosis is work with a price, and the price determines whether a check is worth running before a decision. The
model below uses two rates, both ILLUSTRATIVE: a reviewer at 48.00 per hour and an engineer at 72.00 per hour.

```text
FORMULA [G1-CHECKCOST]: C_check = r * h + q * m * r
WHERE: r [currency/h] the role rate, h [h] elapsed hours, q [items] records examined, m [min] per record
CONSTRAINT: records are already available; if instrumentation must be built, add its build cost separately
NUMERICAL EXAMPLE: reviewer, 1.5 h, 40 records at 2 min
  = 48.00 * 1.5 + 40 * (2/60) * 48.00 = 72.00 + 64.00 = 136.00
INTERPRETATION: half the cost of a record-reading check is reading, so record count is the lever
```

The record-reading term dominates quickly: 40 records at two minutes is 1.33 hours, close to the framing time. A
check that requires 400 records costs ten times the reading term and should be run once, stored, and reused.

### 9.2 Cost of the four check archetypes

| Archetype | Typical elapsed | Records read | Cost at the stated rates | Reuse |
|---|---|---|---|---|
| Arithmetic re-derivation (F1–F12) | 0.5 h reviewer | 0 | 24.00 | Permanent; the formula is the artefact |
| Log query and recomputation | 1.0 h engineer | 1,000 rows by query | 72.00 | Re-runnable at zero marginal cost |
| Document cross-read (protocols, price sheets) | 1.5 h reviewer | 2 documents | 72.00 | Permanent until a document version changes |
| Blind re-labelling | 0.5 h engineer + q * m | 451 at 4 min | 36.00 + 1443.20 = 1479.20 | Only until the guide changes |

The fourth row is 62 times the cost of the first and answers a different question. This asymmetry is the practical
argument for running arithmetic checks first: they are cheap, they are permanent, and they eliminate a large share
of candidate causes before any annotation budget is committed.

### 9.3 Labelling cost scales with the square of precision required

Sample size grows as 1/m², so the labelling bill grows the same way. The table shows the consequence for the 88%
proportion used throughout this file, at 4 minutes per label and 48.00 per hour.

| Margin m | n (F9) | Labelling hours | Cost | Cost relative to m = 0.05 |
|---|---|---|---|---|
| 0.05 | 163 | 10.87 | 521.60 | 1.00 |
| 0.03 | 451 | 30.07 | 1443.20 | 2.77 |
| 0.02 | 1015 | 67.67 | 3248.00 | 6.23 |
| 0.01 | 4057 | 270.47 | 12982.40 | 24.89 |

```text
check for m = 0.03: n = 0.88*0.12*(1.96/0.03)^2 = 0.1056 * 4268.44 = 450.7 -> 451
hours = 451 * 4 / 60 = 30.0667 ; cost = 30.0667 * 48.00 = 1443.20
ratio to m = 0.05 cost 521.60 = 2.767
```

A 2.5x accuracy improvement in the margin costs 6.2x. That is the correct frame for the question "can we settle this
with a bigger test?", and it is the reason CI-24's fix is sometimes "restate as a range" rather than "collect more".

### 9.4 Search cost through the catalogue

Triage is a search over a category of at most seven issues. The expected number of checks run is the position of the
true cause in the chosen order, plus one for the misclassification case.

```text
FORMULA [G2-SEARCH]: E[checks] = (n + 1) / 2 with the true cause uniform over a category of size n
WHERE: n = category size; a misclassification adds one full pass of n checks before the right category
NUMERICAL EXAMPLE: n = 7 -> E[checks] = 4 ; a 25% misclassification rate adds 0.25 * 7 = 1.75
  total expected checks = 4 + 1.75 = 5.75
  at 0.75 h average and 48.00/h -> 5.75 * 36.00 = 207.00 per symptom
INTERPRETATION: ordering the category by cheapness halves the search; the misclassification term sets its floor
```

This is why §7.1 is ordered by cost rather than by conceptual elegance. Ordering by elegance puts the interesting
issue first and the cheap one fifth, roughly doubling the expected spend per symptom.

### 9.5 Cost of the defect versus cost of the prevention

| Issue | Prevention cost (engineer hours at 72.00) | Defect cost per occurrence | Break-even occurrences |
|---|---|---|---|
| CI-10 review capacity omitted | 6.0 h = 432.00 | 1818.67 per month of unmodelled review | 0.24 months |
| CI-23 duplicate effects | 8.0 h = 576.00 | 9.6 duplicates at 20.00 handling = 192.00 per episode | 3.0 episodes |
| CI-22 unapproved action | 12.0 h = 864.00 | One governance review, estimated 1200.00 | 0.7 occurrences |
| CI-01 prevalence not stated | 2.0 h = 144.00 | One wrong go/no-go; magnitude undefined | Below one |
| CI-17 prefix confusion | 1.0 h = 72.00 | 666.00 per month on a 9,000 storage bill | 0.11 months |

The row with an undefined magnitude is the important one. CI-01's defect cost cannot be computed because it is a
decision, not an expense, and decision errors are not amortised against engineer hours. Prevention is therefore
justified by asymmetry, not by arithmetic, and it should be stated that way rather than with a fabricated figure.

### 9.6 Volume scaling of the review loop

At 5,200 tickets and a 4-minute review, the loop consumes 29.3 reviewer hours; at 20,800 it consumes 117.3. The
review load scales linearly while the platform cost is fixed, so the *mix* of cost changes even though the per-ticket
labour constant does not.

```text
5,200 tickets : flags 439.9, review hours 29.333, review cost 1818.67, review share of total 42.2%
20,800 tickets: flags 1759.7, review hours 117.312, review cost 7273.35
  variable 361.92, fixed 2400.00 -> total 10035.27, review share 72.5%
review share growth = 72.5% - 42.2% = 30.3 points
```

A design that is labour-dominated at pilot volume is *more* labour-dominated at scale, because the fixed platform
line grows sublinearly. Any scaling plan that holds review minutes constant per flag while ignoring the flag rate's
dependence on volume will underprice the growth phase.

### 9.7 Cost of a missing decision ledger

The ledger is a fixed cost with an unusual property: its absence is invisible until a defect needs attribution, at
which point the cost is the full re-derivation of the original analysis. Using the check archetype costs from §9.2,
attributing one defect without a ledger requires re-running all four archetypes:

```text
re-derivation 24.00 + log recomputation 72.00 + document cross-read 72.00 + blind re-labelling 1479.20 = 1647.20
with a ledger, the same attribution is a read: 0.25 h reviewer = 12.00
avoided cost per attribution = 1635.20
```

Five attributions across a programme pay for the ledger several times over. This is the strongest available argument
for recording reason codes at the moment of decision, and it is arithmetic rather than principle.

### 9.8 Complexity summary

| Activity | Complexity in the decision size | Practical bound |
|---|---|---|
| Arithmetic check on a stated number | O(1) | One recomputation |
| Log-based recomputation | O(rows) with a query | Bounded by retention window |
| Blind re-labelling | O(n) where n grows as 1/m² | Bounded by budget, not by method |
| Full triage pass | O(issues in category) | Bounded by category size, at most 7 here |
| Ledger read | O(1) per decision | Requires the record to exist |

## 10. Key Theorems / Results Reference

### 10.1 Conservation of flow

**Statement.** In a stable system with no loss, the mean number in the system equals the arrival rate times the mean
time in the system. In discrete form, backlog change over a window equals arrivals minus departures.

**Conditions.** A closed accounting boundary, consistent units, and a single definition of "in the system". Requests
that are retried inside the boundary are one arrival; requests that are dropped are departures with an error label.

**What it forbids.** It forbids any plan in which arrivals exceed sustainable service while claiming a flat queue,
and it forbids adding latency without adding either capacity or admission control. It also forbids the arithmetic
in CI-14: 28 workers at 1.7 s serve 16.47 requests per second, so a 20-per-second arrival cannot leave the queue flat,
whatever the per-request latency chart shows.

### 10.2 Little's law

**Statement.** For a stable system measured over the same window, mean concurrency L = lambda * W.

**Conditions.** Both quantities measured over the identical window; stability, meaning arrivals and departures balance
within it. During a drain the law still holds instantaneously but L is no longer a provisioning number.

**What it forbids.** It forbids sizing a worker pool from throughput tests alone, and it forbids treating a latency
improvement as a capacity improvement without checking whether W was the binding term. It also gives the cheapest
correct capacity estimate available: multiply two measured numbers.

### 10.3 Bayes' identity for precision

**Statement.** P = Se*p / (Se*p + (1 - Sp)(1 - p)), where p is prevalence.

**Conditions.** One threshold, no abstention, ground-truth labels, and a sample drawn from the population the reader
cares about. If labels encode an annotation policy, P describes agreement with that policy.

**What it forbids.** It forbids quoting precision without prevalence (CI-01), and it forbids reading a specificity
figure as a promise about alarms. At p = 0.03 with Se 0.88 and Sp 0.94, alarms are 69% wrong; at p = 0.35 the same
system is 89% right. Same system, same formula, different population.

### 10.4 Chain availability

**Statement.** A_chain is the product of per-stage availabilities for stages that must all succeed.

**Conditions.** Synchronous dependency, independent failures, no failover or retry. Asynchrony converts a chain into
independent stages; shared failure domains make the product optimistic.

**What it forbids.** It forbids promising a composite target equal to the best component's SLA, and it forbids adding
a stage without recomputing the target. Three stages at 99.9% yield 99.70%, which is 129.5 minutes per month against
a 43.2-minute budget — a gap that no amount of monitoring closes.

### 10.5 Amdahl's bound

**Statement.** S = 1 / ((1 - f) + f/s), and the bound as s grows is 1/(1 - f).

**Conditions.** Fixed workload, serial critical path, no overlap. With pipelining the fraction is not a serial share
and the bound does not apply.

**What it forbids.** It forbids claiming a speed-up larger than 1/(1 - f) from any improvement to the accelerated
stage, and it forbids spending a budget on the cheap stage. With f = 0.72 and s = 2, the gain is 56%, not 100%; the
ceiling is 257%.

### 10.6 Simpson's reversal

**Statement.** A pooled comparison can order two options differently from every segment it pools.

**Conditions.** Segment sizes differ between the compared groups, and segment difficulty differs. Reversal requires
both; neither alone is sufficient.

**What it forbids.** It forbids a pooled verdict when the evaluation mixes differ, and it forbids presenting a pooled
figure without its weights. In question Q10 below, X wins pooled by 3.5 points while Y wins both segments by 8 and 7.

### 10.7 Retry amplification

**Statement.** With k independent retry layers and per-attempt failure q, attempts per logical request are
(1/(1 - q))^k.

**Conditions.** Independence across layers and a single retry per layer. Shared dependencies break independence and
turn the multiplier into pure load with no success benefit.

**What it forbids.** It forbids capacity plans built on healthy-state failure rates, and it forbids stacking retries
at several layers "for reliability". Two layers at q = 0.10 cost 23.5% extra attempts; at q = 0.30 the same two
layers cost 104% extra.

### 10.8 Sampling error bound

**Statement.** n = p(1 - p)(z/m)² for a proportion with margin m at confidence determined by z.

**Conditions.** Simple random sample from the target frame, independent items, and n*p as well as n*(1 - p) at least
10. Violating the frame invalidates the bound regardless of n (CI-20).

**What it forbids.** It forbids declaring a win smaller than the interval, and it forbids the belief that a bigger
sample repairs a biased frame. A 4,057-item sample from the wrong frame has an interval of 1 point around the wrong
number.

### 10.9 Unit-prefix identity

**Statement.** 1 GiB = 2^30 bytes = 1.0737e9 bytes, so a decimal GB and a binary GiB of the same numeric value differ
by 7.37%.

**Conditions.** Exactly one conversion per figure; the billing convention known and stated.

**What it forbids.** It forbids carrying formatted sizes between systems, and it forbids a reconciliation tolerance
tighter than the prefix gap. A 2% reconciliation target cannot distinguish a rounding error from a prefix error, so
the tolerance must be set with 7.4% in mind.

### 10.10 Base-rate fallacy

**Statement.** A low false-positive rate applied to a large negative population can outnumber true positives produced
by a high true-positive rate applied to a small positive population.

**Conditions.** Prevalence well below 50%, and error rates that are not degenerate.

**What it forbids.** It forbids judging a screening system by its error rates rather than its alarm composition, and
it forbids staffing a review queue from a sensitivity estimate. At p = 0.02 with Se 0.90 and Sp 0.99, 64.8% of alarms
are correct; drop Sp to 0.90 and only 15.5% are, a fourfold change in the review queue from one specificity point.

## 11. Interview / Exam Questions

Each question is answerable from this file. Where arithmetic is involved the working is shown, because the number
without the method is the defect this file exists to prevent.

**Q1. A classifier has Se 0.90 and Sp 0.95 at a 4% prevalence. What share of its alarms is correct, and how would you
present the figure to a sponsor who wants to staff a review queue?**

TP rate = 0.90 * 0.04 = 0.036; FP rate = 0.05 * 0.96 = 0.048; flag rate = 0.084; P = 0.036/0.084 = 0.4286, so 42.9%
of alarms are correct. Present it as "of every 100 alerts, 43 are real and 57 are review work", because that phrasing
is the quantity the sponsor is staffing for. Quoting Se and Sp alone would invite the assumption that alarms are
mostly real.

**Q2. At 5,200 requests a month with the parameters in §5, how many items enter human review, and what does that cost
at 62.00 per hour with four minutes per item?**

Flag rate = 0.0264 + 0.0582 = 0.0846; flagged = 0.0846 * 5200 = 439.9, so 440 items. Review hours = 440 * 4/60 =
29.333; cost = 29.333 * 62.00 = 1818.67 per month. The reviewer load is 0.73 of a full-time equivalent, which is the
number a staffing conversation actually needs.

**Q3. A vendor quotes 0.4789 per request at pilot volume. Why is applying that figure to 250,000 requests wrong, and
what is the correct figure?**

The pilot figure is an average including a 2,400 monthly fixed line spread over 5,200 requests: fixed share
2400/5200 = 0.4615 of the 0.4789 total. At 250,000 the fixed share is 2400/250000 = 0.0096, so the average is
0.0174 + 0.0096 = 0.0270. The pilot figure overstates by 0.4789/0.0270 = 17.7x. Quote marginal cost plus a fixed
block, then compute the average at the contracted volume.

**Q4. You must show a 3-point margin at 95% confidence around an estimated 88%. How many labels at four minutes
each, and what does it cost at 48.00 per hour?**

n = 0.88 * 0.12 * (1.96/0.03)² = 0.1056 * 4268.44 = 450.7, so 451 labels. Hours = 451 * 4/60 = 30.07; cost =
1443.20. If the workflow has four unequal strata, allocate within strata and reweight, which raises the effective n —
the unstratified figure is a floor, not a plan.

**Q5. Arrivals are 45 per second and a request holds a worker for 2.2 seconds. How many workers are needed, and what
happens at 1.2 times that latency?**

Required concurrency L = 45 * 2.2 = 99. At 1.2x latency, W = 2.64 and L = 118.8, so 119 workers. The second figure
explains why a latency regression with no traffic change causes queue growth: capacity is a product, and either
factor moves it.

**Q6. Three synchronous stages advertise 99.9%, 99.8% and 99.5%. What composite availability results, and how much
monthly downtime does that allow?**

0.999 * 0.998 * 0.995 = 0.992017. Downtime = (1 - 0.992017) * 43200 = 344.9 minutes per month. A single 99.9%
component allows 43.2 minutes, so the chain consumes eight times the budget of its strongest link. Either remove a
synchronous dependency or publish the composite as the target.

**Q7. A ten-turn session starts with a 3,000-token prompt and adds 420 tokens per turn, all history re-sent. How many
input tokens does the session consume, and what does it cost at 2.00 and 12.00 per million?**

Input = 10 * 3000 + 420 * (10*9/2) = 30000 + 420 * 45 = 30000 + 18900 = 48,900 tokens. Cost = 48900 * 2.00/1e6 =
0.0978 input + 10 * 420 * 12.00/1e6 = 0.0504 output, total 0.1482 per session. A per-turn model would have priced
10 * (0.0060 + 0.00504) = 0.1104, which recovers only 74.5% of the true session cost.

**Q8. Two retry layers each retry once, and the per-attempt failure rate is 10%. What fraction of extra attempts does
that create, and what changes at 30%?**

Per layer 1/0.90 = 1.1111; two layers 1.1111² = 1.2346, so 23.5% extra attempts. At q = 0.30, 1/0.70 = 1.4286 and
1.4286² = 2.0408, so 104.1% extra. The second figure is the one that matters, because q rises precisely when the
system is already loaded.

**Q9. Fixed cost is 3,000 per month and contribution margin is 1.42 per ticket. What volume breaks even, and how
should the answer be presented?**

N* = 3000 / 1.42 = 2112.68, so 2,113 tickets per month. Present it as a volume with its parameter set, and place the
current volume beside it: the distance between them is the risk statement. A per-ticket average cost cannot answer
this question, because it embeds the fixed line it is trying to solve for.

**Q10. Two versions are compared on unequal mixes. X scores 0.82 on 700 items and 0.55 on 300; Y scores 0.90 on 300
and 0.62 on 700. Which is better, and what does the pooled figure show?**

Pooled X = (700*0.82 + 300*0.55)/1000 = (574 + 165)/1000 = 0.739. Pooled Y = (300*0.90 + 700*0.62)/1000 =
(270 + 434)/1000 = 0.704. Pooled favours X by 3.5 points, while Y leads on both segments by 8 and 7 points. Report
per segment with sizes; the pooled figure is a weighted average of the mixes, not a quality statement.

**Q11. At p = 0.04, a miss costs 35.00 and a false alarm costs 5.00. What does configuration Se 0.90, Sp 0.95 cost
per 100 items, and how would you use the cost matrix?**

Misses = 0.10 * 0.04 * 100 = 0.4 items at 35.00 = 14.00; alarms = 0.05 * 0.96 * 100 = 4.8 at 5.00 = 24.00; total
38.00 per 100. A more sensitive configuration reduces the 14.00 and multiplies the 24.00, so the matrix explains why
"more accurate" can mean "more expensive". Publish it with the threshold; it converts an opinion into a trade-off.

**Q12. A dashboard reports 2.4 GB; the invoice prices GiB. What is the difference in bytes and in money at a stated
rate, and what tolerance should reconciliation use?**

2.4e9 bytes = 2.4 GB decimal = 2.4e9/2^30 = 2.2352 GiB. The console figure understates the billing quantity by
2.4/2.2352 - 1 = 7.4%. Reconciliation tolerance must therefore sit well outside 7.4%, or every month produces a false
flag; compare raw bytes instead of formatted sizes and the tolerance can return to 2%.

**Q13. Diligence costs 18,000 and moves failure probability from 0.45 to 0.20 on a 350,000 loss. What is the
expected value, and when is it negative?**

EV = (0.45 - 0.20) * 350000 - 18000 = 87500 - 18000 = 69,500, a 3.86x return on the diligence spend. It is negative
whenever the analysis cannot change the decision, because then P_after = P_before and EV = -18,000 exactly. So the
first question is not "how good is the analysis" but "who is permitted to act on it".

**Q14. You estimate 50% from 600 items, then from 2,400. What are the two intervals, and what does the comparison
teach about resolving a 2-point question?**

n = 600: 1.96 * sqrt(0.5*0.5/600) = 1.96 * 0.0204 = 4.0 points. n = 2,400: 1.96 * 0.0102 = 2.0 points. Quadrupling
the sample halves the interval, so resolving a 2-point difference needs roughly 2,400 items per arm — which is the
arithmetic behind declining a 300-item comparison of two close versions.

**Q15. A stage occupying 72% of wall time is made twice as fast. What speed-up results, and what is the ceiling?**

S = 1/((1 - 0.72) + 0.72/2) = 1/(0.28 + 0.36) = 1/0.64 = 1.5625, so 56.25% faster throughput. The ceiling is
1/0.28 = 3.571. A 300% improvement claim for the same change is impossible, and any plan exceeding 257% must be
reducing the non-accelerated share rather than speeding the stage up.

## 12. Quick Reference Card

| Field | Value |
|---|---|
| Scope | Selection, landscape and diligence defects for an AI programme before pilots and architecture |
| First move | State the symptom with a number, a unit and a window; classify before fixing |
| Cheapest check set | Recompute the arithmetic (24.00), then query the log (72.00), then cross-read documents (72.00) |
| Precision rule | P = Se*p/(Se*p + (1-Sp)(1-p)); a precision figure without prevalence is a draft |
| Capacity rule | L = lambda*W; provision above the product or the queue grows at constant service time |
| Cost rule | Quote marginal cost plus the fixed block, with the volume; the average falls as 1/N |
| Availability rule | Composite = product of synchronous stages; 0.999×0.998×0.995 = 99.20% = 344.9 min/month |
| Threshold rule | Pick the cost-minimising threshold, publish the error costs, never let a score approve an action |
| Verification and stop | Replay the triggering record plus one boundary neighbour and ledger the result; two surviving causes means the check is not discriminating, and no record means the cause is OPEN |
| Catalogue | 25 issues: 6 VIOLATED_ASSUMPTION, 8 NUMERICAL_EDGE, 4 SCALE_FAILURE, 7 CONCEPT_CONFUSION |
