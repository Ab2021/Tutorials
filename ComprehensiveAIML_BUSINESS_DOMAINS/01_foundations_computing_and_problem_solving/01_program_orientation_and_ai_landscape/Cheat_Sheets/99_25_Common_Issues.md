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
INTERPRETATION: three excellent stages produce 3.5x the downtime of the weakest one alone
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
be allocated across them and reweighted, which raises the effective n (see §5's re-check for the cost consequence).

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
| Adoption 65% -> 40% | 0.40 | 5378.70*(0.40/0.65) - 4309.15 = 3310.0 - 4309.15 = -999.15 | Loss |
| Saving 28% -> 18% | 0.18 | 5378.70*(0.18/0.28) - 4309.15 = 3457.7 - 4309.15 = -851.45 | Loss |
| Prevalence 3% -> 10% | p = 0.10 | flag 0.088+0.054 = 0.142; review 0.142*5200*4/60*62 = 3050.9; net = 5378.70 - 90.48 - 2400 - 3050.9 = -162.68 | Loss |
| Volume 5,200 -> 20,800 | 4x | benefit 21514.8; var 361.9; fixed 2400; review 7274.7; net = 11478.2 | Strong gain |
| Review time 4 -> 2 min | 2.0 | review 909.3; net = 1978.9 | Gain |

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
