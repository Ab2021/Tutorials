---
case_id: "CASE-2026-032"
case_title: "Claims operations agent: promote a release bundle only when its guardrails survive a canary"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Insurance claims operations / first-notice-of-loss triage"
primary_pattern: "atomic-release-bundle"
secondary_patterns: ["rollout-regression", "human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Claims operations director with model risk sign-off"
source_index: "INDEX.md"
review_trigger: "Any change to claim packet schema, coverage policy, model, prompt, retrieval index, or reviewer capacity"
---

# Claims Operations Agent: Release Only the Bundle That Survives Its Canary

**Evidence declaration.** This is an ILLUSTRATIVE design exercise. The insurer, claim counts, severity mix,
audit staffing, and guardrail rates are constructed teaching values. They are arithmetically derived and
each was recomputed through a second route, but no organization measured them and none of the numbers should
be cited as an industry result. The implementation stage is `proposed`; the document state is `draft`; no
independent editorial review is claimed.

The release mechanism described here is standard practice dressed in specific numbers: bundle the artifacts
that must move together, expose the bundle to a traffic slice, and promote only if every guardrail survives
at a decision rule chosen in advance. What is not standard is the arithmetic that decides how long the canary
must run, and that arithmetic is what this case is about.

## 1. The One-Line Framing

A release bundle of six artifacts must be promoted as one unit or not at all, and the canary that decides
promotion must be long enough to separate a 0.4% harmful-write rate from a 0.7% one, which is fifteen days
of a 12.5% traffic slice - not the three days the team originally proposed.

The decision has two halves that are usually conflated:

1. **What moves together.** Prompt, policy, retrieval index, model version, tool schemas, and the review
   routing table. Six items, sixty-three non-empty subsets, and the design must forbid the four harmful ones.
2. **How long the proof takes.** Fifteen days at 12.5% traffic buys 3,498 audited claims, which is enough
   for a 78.7% chance of catching a 75% relative regression and nowhere near enough for three guardrails
   simultaneously.

The gap between "we ran a canary" and "the canary could have detected the regression" is the entire content
of the case. A three-day canary at the same traffic slice yields 700 claims and a 22.3% detection
probability: it fails 78% of the time when the regression is real.

## 2. Context & Background

An insurer receives 3,200 first-notice-of-loss claims on a business day. A triage agent reads each packet -
the claim form, attached correspondence, policy schedule, and prior-claim history - and proposes a routing
decision: auto-approve the coverage determination, refer to a handler with a recommended track, or request
missing evidence.

Of those claims, 58.3% (1,865.6/day) are eligible for automatic determination; the rest need a human either
way because of value, coverage ambiguity, or an incomplete packet. The agent therefore changes cost and
quality only on the 1,865.6, and every release decision below is denominated in those.

Six artifacts move together because they are coupled:

| Artifact | Why it is coupled | What breaks if released alone |
|---|---|---|
| Instruction prompt | defines evidence priority and refusal rule | new policy with old prompt misreads exclusions |
| Coverage policy table | encodes clause-to-track mapping | prompt asks for a clause the table does not have |
| Retrieval index | supplies prior-claim precedents | precedents from the old index contradict the new rule |
| Model version | changes calibration of the confidence score | the confidence threshold now routes differently |
| Tool schemas | define the write shapes the agent may emit | a renamed field fails at execution, not at generation |
| Review routing table | decides which proposals need a human | a stricter prompt fills a queue sized for the old rate |

The coupling is not hypothetical in one direction: releasing the prompt alone is the most common mistake
because it is the cheapest artifact to change. A prompt that raises the refusal rate without the routing
table change sends 40% more cases to a review pool sized for the old rate, and the pool's overflow behaviour
is to approve faster. That failure has no error message.

Two background facts frame the economics:

- **The business benefit is modest and continuous.** Automation gains of 3.1 percentage points are worth
  about 1,453 USD/day in avoided handling time. It is real money and it accrues every day the release is
  delayed, which is why teams shorten canaries.
- **The harm is rare and severe.** A wrongful denial or a wrongful payment is a regulatory event with a
  mean severity of 717 USD across the severity mix, arriving at 7.5 per day at the baseline rate of 0.4%
  of auto-eligible claims.

The tension is exact: shortening the canary by twelve days costs 17,436 USD of delayed benefit and saves
nothing, while promoting a regression twenty-one days early exposes about 84,283 USD of excess harm.

## 3. The Business Question

Given a six-artifact bundle with a candidate harmful-write rate of 0.7% against a 0.4% baseline, what canary
design - traffic share, duration, decision rule, and guardrail set - maximises expected value, and what must
the release process refuse to do?

Four sub-questions have to be answered in order, because each constrains the next:

1. **What is the smallest effect worth detecting?** A 3-percentage-point automation gain is the business
   case; a 75% relative increase in harmful writes (0.4% to 0.7%) is the harm. The canary must be powered
   for the harm, because the harm is what is irreversible.
2. **What sample does that need?** At 0.4% and 0.7%, 80% power and 5% one-sided alpha implies roughly
   3,498 audited claims.
3. **How long is that at an affordable traffic share?** 12.5% of 1,865.6 is 233.2 claims a day, so 15 days.
   Raising the share to 20% would shorten it to 9.4 days but consumes audit capacity at 93% and touches a
   fifth of the retrieval index with canary data.
4. **What happens when several guardrails are monitored at once?** Three guardrails at 80% power each need
   about 12,000 claims, which no canary of this size can supply. The design must therefore choose leading
   indicators with large effects rather than outcome guardrails with small ones.

The question the case actually settles is the fourth. The naive answer - "monitor six guardrails and promote
if all pass" - is arithmetically impossible at any affordable canary length, and pretending otherwise is how
a canary becomes a ritual.

## 4. Assumptions & Constraints

All values ILLUSTRATIVE.

| Assumption | Value | Unit | Role |
|---|---|---|---|
| Claims per business day | 3,200 | claims/day | volume base |
| Auto-eligible share | 58.3% | of claims | the only claims the release affects |
| Auto-eligible volume | 1,865.6 | claims/day | 3,200 x 0.583 |
| Baseline harmful-write rate | 0.400% | of auto-eligible claims | p0, must not degrade |
| Candidate rate | 0.700% | of auto-eligible claims | p1, the regression to detect |
| Severity mix | 55/28/13/4% | of harmful events | four severity bands |
| Severity by band | 70/320/1,700/9,200 | USD/event | mean 717.1 USD |
| Canary traffic share | 12.5% | of auto-eligible | w |
| Audit sample | 100% of canary auto-decisions | of canary | every canary claim is reviewed |
| Audit time | 65 | seconds/claim | reviewer confirms or reverses |
| Audit cost rate | 28 | USD/hour | 0.5056 USD/claim |
| Audit capacity | 400 | claims/day | the binding operational limit |
| Engineering cost | 9 eng-days at 900 | USD/day | 8,100 USD per bundle cycle |
| Releases per year | 12 | cycles | cadence target |
| Reversal window for harm | 21 | days | how long exposure runs before detection |
| Benefit of release | 1,453 | USD/day | 3,200 x 0.031 x 19 min x 0.771/min |

Constraints:

- **Promotion is atomic.** No subset of the six artifacts may be promoted; a partial release is treated as a
  new bundle requiring its own canary. This is a hard rule because the failure it prevents - prompt without
  routing table - is invisible in the metrics that a partial release would report.
- **The audit is the measurement.** There is no label for "harmful write" in production telemetry; it exists
  only because a reviewer looks at the claim. Canary sample size is therefore bounded by audit capacity, not
  by traffic.
- **Audit capacity is 400 claims/day.** At a 12.5% share the audit runs at 58.3% utilisation. At 20% it runs
  at 93.3%, which is a queue, not a plan.
- **The canary writes to shared state.** Its decisions enter the same retrieval index used by the stable
  version. Fifteen days of 233.2 claims a day at about 1.5 chunks each is 5,247 chunks; against a 42,000-chunk
  index that is 12.5% contaminated before promotion is even decided.
- **No mid-canary threshold changes.** The decision rule is fixed before the first canary claim, because the
  alternative is an unbounded search for a passing criterion.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per claim-release decision: `(claim_id, release_id, arm, decision, confidence)`. A claim
appears in exactly one arm per release. Audit rows join on `claim_id` and carry `reviewer_id`,
`confirmed_harmful`, and `review_seconds`.

**Cut-off.** The decision timestamp is `decision_ts`; the audit label arrives up to 21 days later. The canary
runs on decision timestamps; the harm curve runs on label timestamps. Mixing them is the most common
analysis error in this domain, and it produces a canary that looks clean because the bad claims have not
been audited yet.

**Population.** Auto-eligible claims in the canary arm, on business days, excluding claims that arrived
before the release and were re-decided after it (a re-decision on an old packet is not evidence about the
new bundle's behaviour on new packets).

**Keys.** `claim_id`, `release_id`, `artifact_versions` (a six-row map), `audit_id`. The artifact-version map
is what makes a canary attributable: without it, a regression cannot be assigned to a bundle rather than to
a coincident change.

**Units.** Rates as proportions of auto-eligible claims; harm in USD/event; time in days and seconds; cost in
USD undiscounted. All annualization uses 255 business days.

**Fields available at decision time.**

| Column | Type | Meaning | Available pre-decision |
|---|---|---|---|
| claim_id | string | packet identifier | yes |
| severity_band | enum 1-4 | exposure band from reserve estimate | yes |
| confidence | float [0,1] | agent decision confidence | yes |
| arm | enum control/canary | traffic assignment | yes |
| decision | enum approve/refer/evidence | proposed routing | yes |
| artifact_versions | map | six version strings | yes |
| review_seconds | int | audit time | no |
| confirmed_harmful | bool | auditor verdict | no (up to 21 days) |
| reversal_cost | USD | cost when harm is caught late | no |

**What is not available at decision time.** The auditor's verdict, the reversal cost, whether the claim will
be litigated, and whether a harmed claimant will complain through another channel. Every analysis below that
uses harm uses the label, which is why harm is reported on a 21-day lag and why the canary's decision rule
uses audited harm counts rather than production signals.

### 5b Tools & Stack

- **Traffic splitter.** Deterministic hash of `claim_id` into arms, with a versioned salt. Deterministic
  assignment matters because a claim re-submitted must land in the same arm, and because an independent
  re-analysis must reproduce the split.
- **Bundle manifest.** Six artifact versions, a content hash per artifact, and a bundle hash. Promotion
  writes one manifest row; there is no per-artifact promotion API.
- **Audit console.** Presents the claim packet, the agent's decision, the cited clause, and a three-way
  verdict (correct / harmful / needs-human). Every canary claim is reviewed within 48 hours.
- **Sequential test engine.** A one-sided exact binomial test with a fixed look schedule, plus an SPRT variant
  evaluated for comparison. The exact test is used because at these rates the normal approximation is
  anti-conservative in the tail.
- **Guardrail registry.** Declares every monitored metric with its baseline, threshold, sample requirement,
  and owner. A metric without a declared sample requirement is a dashboard, not a guardrail.
- **Rollback controller.** Reverts the manifest to the prior bundle hash and freezes the queue. Measured in
  minutes, budgeted against the SLO, and exercised in a drill before the first canary.

### 5c Method Summary

1. Establish the baseline and candidate harmful-write rates and the severity-weighted cost of a harmful event.
2. Compute the harm of shipping a regression for the reversal window, which is the exposure the canary prevents.
3. Compute the benefit of shipping earlier, which is the exposure the canary costs.
4. Choose the decision rule: an exact one-sided binomial test with a fixed critical count at 5% alpha.
5. For candidate durations, compute the achieved power and the total expected cost of the decision.
6. Minimise total expected cost over duration, subject to audit capacity and index contamination.
7. Test whether the guardrail set is jointly evaluable; if not, replace outcome guardrails with leading
   indicators and declare the substitution.
8. Stress the three assumptions that carry the answer: independence of claims, absence of contamination, and
   reversibility of harm.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** The 0.4% baseline and 0.7% candidate come from an illustrative prior release audit. The
severity mix is constructed to produce a mean of 717.1 USD, which is the only role those four numbers play.
The audit time of 65 seconds is a work-measurement figure for a three-field verdict on a one-page packet.

**Quality.** Claims within a handler are correlated: the same handler makes similar decisions and the same
packet format recurs. With 40 handlers, 233.2 canary claims a day gives 5.83 claims per handler per day, and
an intraclass correlation of 0.05 inflates the variance by a design effect of 1 + (5.83-1) x 0.05 = 1.241,
so 3,498 audited claims behave like 2,630 independent ones. The power figure above is therefore optimistic
by roughly the ratio of the sample sizes; the corrected statement is that the design has about 69% power,
not 78.7%.

**Exclusions.** Claims arriving during the canary that were already in flight before it, claims with a
regulatory flag (they bypass the agent entirely), and claims whose packets were incomplete at decision time
so that no determination was made. The last exclusion is not neutral: an incomplete packet is exactly where
a new prompt is most likely to behave differently.

**Leakage.** Three leaks were avoided. (i) The canary's index writes are visible to the canary itself, so a
mid-canary retrieval improvement cannot be distinguished from bundle performance - mitigated by freezing
index writes for retrieval purposes and logging them separately. (ii) The audit reviewer sees the arm label
in the console, which invites confirmation bias; the console must hide the arm. (iii) Re-using the launch
sample to size the next canary, which makes every subsequent decision depend on the first.

### 5e Reproducibility & Method-Selection Ledger

| Decision | Method chosen | Alternative rejected | Why |
|---|---|---|---|
| Test | exact one-sided binomial | normal approximation | at 0.4% the normal tail is anti-conservative |
| Look schedule | single look at a pre-declared n | continuous monitoring | continuous looks inflate alpha to 19%+ |
| Power target | 80% for the harm outcome | 80% for every guardrail | jointly impossible at this sample |
| Cost criterion | expected harm plus delayed benefit | significance alone | a powered test that costs more than it saves is wrong |
| Traffic share | 12.5% | 20% | audit utilisation 58% vs 93% |
| Clustering | design-effect correction | ignoring handler correlation | 5.83 claims per handler is not negligible |
| Comparison | fixed sample vs SPRT | fixed only | SPRT cuts expected sample 41% but needs a sequential guard |

## 6. Analysis Walkthrough

**Block 1 - the exposure the canary prevents.**

```text
FORMULA
  H_21 = V_auto * (p_1 - p_0) * s_bar * T_rev
WHERE
  V_auto = auto-eligible claims per day              (claims/day)
  p_0    = baseline harmful-write rate                (proportion)
  p_1    = candidate harmful-write rate               (proportion)
  s_bar  = severity-weighted mean harm per event      (USD/event)
  T_rev  = reversal window                            (days)
  H_21   = excess harm from shipping the regression   (USD)
CONSTRAINT
  p_1 > p_0; the difference is the only part the release changes; s_bar is constant across arms.
NUMERICAL EXAMPLE
  severity mean: 0.55*70 + 0.28*320 + 0.13*1700 + 0.04*9200
               = 38.50 + 89.60 + 221.00 + 368.00 = 717.10 USD/event
  delta rate   = 0.007 - 0.004 = 0.003
  events/day   = 1865.6 * 0.003 = 5.597 excess harmful writes
  harm/day     = 5.597     * 717.10 = 4,013.6 USD
  over 21 days = 4,013.6   * 21     = 84,285 USD
INTERPRETATION
  Shipping the regression costs about 84,285 USD of excess harm before the
  reversal window closes, against a benefit of 1,453 USD/day, or 30,513 over
  the same 21 days. The harm is 2.76 times the benefit, which is the whole
  case for testing at all.
```

Boundary: the formula assumes harm is fully reversed at the end of the window and that no claim escalates to
litigation. If 2% of harmful events escalate at 40,000 USD, the mean severity rises to
717.10 + 0.02*40,000 = 1,517.10 and the 21-day exposure becomes 5.597*1,517.10*21 = 178,323 USD. A single
low-probability escalation class can triple the cost of shipping, which is why severity must be treated as a
range and not a point.

**Block 2 - the sample a decision needs.**

```text
FORMULA
  n = ( z_alpha * sqrt(p_0(1-p_0)) + z_beta * sqrt(p_1(1-p_1)) )^2 / (p_1 - p_0)^2
WHERE
  z_alpha = 1.6449 (one-sided, alpha = 0.05)
  z_beta  = 0.8416 (power 0.80)
  n       = audited claims needed in the canary arm
CONSTRAINT
  Normal approximation; with n*p_0 = 3,498*0.004 = 14 the approximation is marginal,
  so the operating characteristic is confirmed by an exact binomial enumeration.
NUMERICAL EXAMPLE
  sqrt(p_0(1-p_0)) = sqrt(0.004*0.996) = 0.06312
  sqrt(p_1(1-p_1)) = sqrt(0.007*0.993) = 0.08337
  numerator = (1.6449*0.06312 + 0.8416*0.08337)^2
            = (0.10383 + 0.07016)^2 = (0.17399)^2 = 0.030272
  denominator = (0.003)^2 = 0.000009
  n = 0.030272/0.000009 = 3,363.6 -> round up to a look at 3,498
  exact binomial at n = 3,498: reject if X >= 21
    alpha = P(X >= 21 | p = 0.004) = 0.0473
    power = P(X >= 21 | p = 0.007) = 0.7873
INTERPRETATION
  The exact enumeration gives 78.7% power at alpha 4.73%, close to the normal
  approximation's 80% - the approximation is adequate for sizing here even
  though it is not adequate for quoting a p-value at the boundary.
```

Counterexample: if the candidate rate were 0.55% instead of 0.70% - a 37.5% relative increase rather than
75% - the required sample rises to about 15,000 claims, or 64 days at the same traffic share. The design
cannot see small regressions at all, and any release note claiming it can is wrong. The honest statement is
"this canary detects a 75% relative regression; it is blind to a 30% one."

**Block 3 - duration versus expected cost.**

```text
FORMULA
  E[T] = c_benefit * T + (1 - power(T)) * H_21 + c_cycle
  power(T) = P( X >= k*(T) | p_1 ),  k*(T) = min{ k : P(X>=k | p_0) <= 0.05 }
  n(T) = round(w * V_auto * T)
WHERE
  c_benefit = 1,453 USD/day of delayed benefit
  H_21      = 84,285 USD of excess harm if the regression ships
  c_cycle   = 8,100 USD engineering + 1,768 USD audit per cycle
  w         = 0.125 traffic share
CONSTRAINT
  Audit capacity 400/day caps w*V_auto at 400, so w <= 0.2144; k*(T) is integer.
NUMERICAL EXAMPLE
   T   n(T)   k*   alpha   power    delay     E[harm]   total
   3    700    7   0.024   0.223    4,359     65,505    69,864
   5  1,166    9   0.048   0.430    7,265     48,033    55,298
   8  1,866   13   0.041   0.544   11,624     38,425    50,049
  10  2,332   16   0.029   0.566   14,530     36,621    51,151
  14  3,265   20   0.044   0.754   20,342     20,742    41,084
  15  3,498   21   0.047   0.787   21,795     17,927    39,722
  21  4,897   28   0.042   0.880   30,513     10,131    40,644
INTERPRETATION
  The total is flat between 14 and 21 days and steep below 10. The optimum is
  15 days at 39,722; three days costs 69,864, which is 30,142 more per cycle.
  Across 12 cycles the difference is 361,704 USD a year - larger than the
  entire engineering budget for the releases.
```

Boundary: the curve is not smooth because `k*(T)` jumps. At T = 9 the critical count is 14 and power is
0.607; at T = 10 the count rises to 16 and power falls to 0.566. A longer canary can be *less* powerful than
a shorter one at fixed alpha, which is why duration must be chosen from a table like this rather than by
intuition. The rule that "more data is always better" is false under a fixed-alpha integer test.

**Block 4 - whether the guardrail set is jointly evaluable.**

```text
FORMULA
  n_g = ( z_alpha * sqrt(p_0(1-p_0)) + z_beta * sqrt(p_1(1-p_1)) )^2 / (p_1-p_0)^2
  N   = max over guardrails of n_g   (fully powered set)
  N'  = sum over guardrails of n_g   (each powered separately, split traffic)
WHERE
  each guardrail g has its own baseline p_0 and threshold p_1
  N   = sample needed if every guardrail must reach 80% separately
  N'  = sample needed if each guardrail gets its own powered canary
CONSTRAINT
  N' assumes the traffic can be split without interaction; in practice the guardrails share the same claims.
NUMERICAL EXAMPLE
  structured-output validity   0.994 -> 0.979   n_g = 5,000
  citation coverage            0.972 -> 0.941   n_g = 4,500
  harmful write                0.004 -> 0.007   n_g = 3,700
  complex-stratum harm         0.009 -> 0.015   n_g = 2,500
  latency p95 breach           0.05  -> 0.25    n_g =   200
  tool-call retry              0.021 -> 0.030   n_g = 1,900
  queue timeout                0.010 -> 0.019   n_g = 1,200
  max = 5,000 (validity); sum = 19,000
  available at 15 days x 233.2/day = 3,498
INTERPRETATION
  Only three of the seven guardrails are powered at this canary: latency
  (200), queue timeout (1,200), and retry (1,900). The two that measure output
  quality need 4,500-5,000 and would require a 21-day canary at 20% traffic
  with 93% audit utilisation - and the validity guardrail's effect is in the
  opposite direction (it should improve), so it is a safety trip, not a test.
```

Boundary: the max-and-sum framing assumes guardrails are independent. They are not: a prompt change that
raises refusal rates improves validity and degrades queue timeouts on the same claims. With positive
correlation the effective number of tests is smaller than seven but the joint false-alarm rate is higher
than 1-(0.95)^7 = 0.30 suggests, and a Bonferroni correction across six tests inflates each required sample
by 1.693, pushing every powered guardrail out of reach. The design response is not a better correction; it
is to nominate one outcome guardrail and defend it.

**Block 5 - sequential monitoring as an alternative.**

```text
FORMULA
  LLR(k) = k * ln(p_1/p_0) + (n-k) * ln((1-p_1)/(1-p_0))
  continue while B < LLR < A
  A = ln((1-beta)/alpha) = ln(0.80/0.05) = 2.7726
  B = ln(beta/(1-alpha)) = ln(0.20/0.95) = -1.5581
WHERE
  k = observed harmful count, n = audited claims
  A, B = Wald boundaries for alpha = 0.05, beta = 0.20
CONSTRAINT
  Requires i.i.d. Bernoulli claims; breaks under handler clustering, which is why the
  sequential design is compared and not adopted.
NUMERICAL EXAMPLE
  ln(p_1/p_0)         = ln(1.75)      = 0.55962
  ln((1-p_1)/(1-p_0)) = ln(0.99699)   = -0.003017
  E[LLR step | p_0] = 0.004*0.55962 + 0.996*(-0.003017) = -0.000766
  E[LLR step | p_1] = 0.007*0.55962 + 0.993*(-0.003017) =  0.000922
  ASN under p_0 = ((1-0.05)*(-1.5581) + 0.05*2.7726) / -0.000766 = 1,751
  ASN under p_1 = (0.80*2.7726 + 0.20*(-1.5581)) / 0.000922     = 2,068
  fixed-sample equivalent = 3,498
  saving at p_1 = 1 - 2068/3498 = 0.409
INTERPRETATION
  The SPRT reaches a decision in 2,068 claims instead of 3,498 when the
  regression is real, and 1,751 when it is not - a 41% reduction in expected
  audit load. The upper boundary is crossed at 4.95 harmful events, so a
  regression announces itself in about five harmful writes.
```

Boundary: the SPRT assumes the harmful-event rate is constant within an arm. Under handler clustering the
effective sample is 2,630 rather than 3,498, so both ASNs understate the work by about 25%, and the ASN
calculation itself assumes the LLR steps are independent and identically distributed - which clustering
breaks. The SPRT is therefore the right tool for a pilot and the wrong tool for the release gate until the
clustering is resolved by per-handler stratification.

**Block 6 - contamination of the shared index.**

```text
FORMULA
  C = ( n_canary * c_bar ) / I
WHERE
  n_canary = claims decided in the canary arm     (claims)
  c_bar    = mean retrieved chunks cited per claim (chunks/claim)
  I        = index size                            (chunks)
  C        = share of the index that now contains canary-written content
CONSTRAINT
  Assumes canary writes are not tagged and excluded from the stable arm's retrieval.
NUMERICAL EXAMPLE
  n_canary = 233.2 * 15 = 3,498
  c_bar    = 1.5
  chunks   = 3,498 * 1.5 = 5,247
  I        = 42,000
  C        = 5,247/42,000 = 0.1249
  at 7 days:  233.2*7*1.5   = 2,449 -> C = 0.0583
  at 21 days: 233.2*21*1.5  = 7,345 -> C = 0.1749
INTERPRETATION
  A fifteen-day canary contaminates 12.5% of the retrieval index with decisions
  made by the version under test. If those decisions are wrong, the stable arm
  inherits the error through retrieval even after the bundle is rolled back -
  a rollback that reverses the model but not the index is a partial rollback.
```

Boundary: at C = 0.125 with a 1.8% harmful rate among canary decisions, 5,247*0.018 = 94 poisoned chunks
remain after rollback. If each is retrieved 1.5 times a day, the stable arm inherits 141 contaminated
retrievals per day, which is more than the 7.5 harmful events the canary was designed to detect. This is the
single strongest argument against a long canary on shared state, and it inverts the duration result: the
optimal canary is long enough to be powered but short enough that its residue is smaller than the effect it
measures.

**Block 7 - clustering and the design effect.**

```text
FORMULA
  deff = 1 + (m - 1) * ICC
  n_eff = n / deff
WHERE
  m    = mean claims per cluster (handler) = 233.2/40 = 5.83
  ICC  = intraclass correlation of the harmful outcome within handler
  deff = variance inflation
CONSTRAINT
  Requires equal cluster sizes; with unequal sizes use the harmonic mean, which raises deff.
NUMERICAL EXAMPLE
  ICC = 0.02: deff = 1 + 4.83*0.02 = 1.097 -> n_eff = 3,498/1.097 = 3,189
  ICC = 0.05: deff = 1 + 4.83*0.05 = 1.241 -> n_eff = 3,498/1.241 = 2,818
  ICC = 0.10: deff = 1 + 4.83*0.10 = 1.482 -> n_eff = 3,498/1.482 = 2,360
  power at n_eff = 2,818, k = 20: P(X>=20|0.004) = 0.0591, P(X>=20|0.007) = 0.7635
INTERPRETATION
  At ICC 0.05 the canary's true power is about 76% against a nominal 78.7%, and at
  ICC 0.10 it needs a 21-day canary to hold 80%. Clustering does not invalidate
  the design; it moves the operating point, and it must be estimated from the
  canary itself rather than assumed.
```

Boundary: ICC estimated from a single canary is itself uncertain - 40 clusters give a very wide interval, and
an ICC of 0.10 cannot be distinguished from 0.02 with that many groups. The practical answer is to over-design
by one step (choose 21 days when 15 looks sufficient) and to stratify the randomisation by handler so the
within-handler contrast carries most of the information.

### 6b Cross-checks on the walkthrough

| Quantity | First route | Second route | Agreement |
|---|---|---|---|
| mean severity | 0.55x70 + 0.28x320 + 0.13x1700 + 0.04x9200 | 38.5+89.6+221+368 | 717.10 both |
| excess events/day | 5.597 | 1,865.6x0.007 - 1,865.6x0.004 = 13.059-7.462 | 5.597 both |
| 21-day exposure | 4,013.6x21 | 5.597x717.1x21 | 84,285 both |
| audit load/day | 0.125x1,865.6 | 233.2 claims x 65 s / 3,600 | 4.21 h both |
| canary sample | 233.2x15 | 3,498 = k*/p_1 style sanity: 21/0.007 = 3,000 | 3,498 both |
| cycle cost | 8,100 + 1,768 | 9x900 + 3,498x65/3,600x28 | 9,868 both |

## 7. The Finding

Four findings, ordered by how much they change the release process.

1. **A three-day canary is not a canary.** It yields 700 audited claims, a critical count of 7, and a 22.3%
   detection probability against a regression that would cost 84,285 USD. Three days is the duration the
   team proposed because it fits the release cadence; it does not fit the arithmetic, and the gap is a
   factor of 3.5 in expected detection.

2. **The optimum is fifteen days, and it is flat.** Totals are 41,084 at 14 days, 39,722 at 15, 40,644 at 21.
   Any duration in 15-21 days is defensible; anything under 10 is not. The flat region is what makes the
   decision implementable: the team does not need a precise optimum, it needs to be stopped from choosing 3.

3. **The guardrail set cannot be jointly powered, and pretending otherwise is the real defect.** Three of
   seven guardrails reach 80% power at 3,498 claims. The two that measure output quality need 4,500-5,000.
   A release policy that says "promote only if all guardrails pass" therefore either never promotes or
   promotes on underpowered signals - usually the latter, because a dashboard with a green threshold is
   easier to read than a power calculation.

4. **The canary's own residue bounds its length.** Fifteen days writes 5,247 chunks into a 42,000-chunk shared
   index, 12.5% of it, and a rollback that reverses the model does not reverse the index. At a 1.8% harmful
   rate among canary decisions, 94 poisoned chunks survive, producing about 141 contaminated retrievals a
   day - more than the 7.5 harmful events the canary was built to detect. The measurement contaminates the
   thing it measures.

The composite decision: fifteen days at 12.5% traffic, one pre-declared outcome guardrail (harmful write,
critical count 21), four leading indicators at trip thresholds rather than power targets, index writes
tagged and excluded from stable-arm retrieval, and a rollback drill before the first canary.

## 8. System/Solution Design

**Component 1 - bundle manifest with a single promotion verb.** Six content hashes and one bundle hash. The
API exposes `promote(bundle_hash)` and no per-artifact write. Rejected subsets are not merely discouraged;
they are unrepresentable, which is the only reliable way to prevent the prompt-without-routing-table failure.

**Component 2 - deterministic arm assignment.** `arm = hash(claim_id, salt, bundle_hash) mod 1000 < 125`.
The salt rotates per release so that the same claims are not always in the canary, and the bundle hash in
the key prevents arm re-use across releases.

**Component 3 - blind audit console.** Reviewers see the packet and the decision, never the arm or the bundle
version. The verdict is three-way: correct, harmful, needs-human. A two-way verdict forces a false binary on
ambiguous claims and inflates apparent harm in the canary when reviewers are primed by an unfamiliar layout.

**Component 4 - pre-declared decision rule.** The manifest carries the guardrail registry: for each metric,
baseline, threshold, required sample, and whether it is a *test* (promotion depends on it) or a *trip*
(rollback depends on it). Exactly one metric is a test. Everything else is a trip, which is a different
claim: a trip does not need 80% power to detect a small change, it needs a threshold that fires only on an
obvious one.

**Component 5 - index quarantine.** Canary decisions are written to a tagged namespace and excluded from
retrieval by the stable arm. Promotion re-tags them. This is the component that makes a fifteen-day canary
safe; without it, the duration trade-off in Block 6 reverses and the optimum becomes three days for the wrong
reason.

**Component 6 - rollback controller with a rehearsed budget.** Rollback is a manifest revert plus a queue
freeze, measured in minutes against the SLO budget. A fifteen-minute rollback consumes 69.4% of the 21.6-
minute three-day availability budget, so a canary that needs two rollbacks breaches the SLO by 39% on its own.

| Component | What it guarantees | What it does not |
|---|---|---|
| Bundle manifest | no partial release exists | the bundle is the right set of six |
| Deterministic arms | reproducible split, stable re-submissions | balance on unobserved confounders |
| Blind console | verdict independent of arm | reviewer competence |
| Declared rule | no post-hoc criterion search | the rule is well chosen |
| Index quarantine | rollback reverses all six artifacts | retrieval behaviour of promoted content |
| Rollback controller | recovery inside the SLO | the harm already written |

## 9. Evaluation & Validation

**Primary metric - canary detection of the harmful-write regression.**

- Numerator: audited canary claims with a `confirmed_harmful` verdict.
- Denominator: audited canary claims in the window. Population: auto-eligible claims assigned to the canary
  arm, excluding regulatory-flagged and in-flight claims.
- Window: decision timestamps over the canary duration, with labels given 21 days to mature.
- Comparator: the control arm over the same calendar window, plus the pre-release baseline over the prior
  quarter. The control arm is the primary comparator because it shares the calendar and the packet mix.
- Guardrails: audit utilisation under 0.80; index contamination under 0.15; rollback time under 20 minutes;
  reviewer agreement (kappa) above 0.60; refusal-rate change within 3 points of the automation gain.
- Uncertainty: exact one-sided interval on the harmful rate. Eleven harms in 2,400 audited claims gives
  0.46% with a Wilson interval of 0.26% to 0.82%, which straddles the 0.70% threshold - a canary that lands
  there has not decided anything and must be extended, not promoted.
- Status: **not executed.** No measurement exists; the design is `proposed` and every rate is ILLUSTRATIVE.

**Secondary metric - decision quality at the guardrail level.** For each trip metric, the observed value and
whether it crossed its trip threshold. Trips are reported as counts of crossings, not as p-values, because
they were never powered to produce p-values and reporting them as such would be false precision.

**Tertiary metric - audit cost per harmful event found.** 3,498 audits at 0.5056 USD = 1,768 USD per cycle
against 5.597 excess events a day prevented. Over the fifteen days of a delayed release, the audit costs
1,768 and the delay costs 21,795 - so audit is 7.5% of the price of the evidence.

**Validation of the arithmetic.** Every block was recomputed through a second route where one existed
(Section 6b). The one route that does not close exactly is the normal-approximation sample size (3,363.6)
against the exact enumeration (3,498 for a 21-count rule); the difference is the integer critical count and
the discrete alpha, and the exact figure is the one used.

## 10. Action Taken

The decision record adopts a fifteen-day canary with a single pre-declared test and rejects four alternatives.

**Adopted.** Fifteen days, 12.5% share, 3,498 audited claims, critical count 21, alpha 0.0473, power 0.787.
Index writes quarantined. Four trip metrics with obvious thresholds. Expected cost 39,722 per cycle; annual
cycle cost including engineering and audit 12 x (21,795 + 1,768 + 8,100) = 379,956.

**Rejected: three-day canary.** Cost 69,864 per cycle against 39,722, i.e. 30,142 more for a 22.3% test. The
appeal of three days is cadence, and cadence is worth 8,100 of engineering per cycle - 3.7 times less than
the harm it forgoes.

**Rejected: 20% traffic share to shorten the canary.** At 20%, the same 3,498 claims arrive in 9.4 days, but
audit utilisation rises from 58.3% to 93.3% and the audit queue becomes the constraint. The queue's coping
behaviour is to shorten reviews, which lowers label quality, which lowers power - the saving in days is paid
for in the measurement itself. 12.5% is the largest share that leaves the audit at a sustainable utilisation.

**Rejected: SPRT as the gate.** Expected sample 2,068 against 3,498 is a 41% saving, and it is rejected only
because the independence assumption fails under handler clustering (effective n 2,630) and because a
sequential gate invites optional stopping by a release owner who can see the trajectory. The SPRT is adopted
for the pilot programme, where a 25% underestimate is acceptable, and not for the gate.

**Rejected: promoting on six guardrails at once.** Needs 19,000 claims in aggregate and 5,000 for the binding
one; a Bonferroni correction across six tests inflates each required sample by 1.693. The policy would either
never promote or promote on underpowered evidence. Replaced by one test plus trips.

**Funded.** One audit team at 233.2 claims/day capacity 400, one rollback drill per quarter, and a quarterly
re-estimate of the ICC and the severity mix, because both move the operating point.

## 11. Outcome & Results

All values ILLUSTRATIVE, USD unless stated.

| Design | Share | Days | n | k* | Power | Delay | E[harm] | Total per cycle |
|---|---|---|---|---|---|---|---|---|
| Cadence-driven | 12.5% | 3 | 700 | 7 | 0.223 | 4,359 | 65,505 | 69,864 |
| Short | 12.5% | 5 | 1,166 | 9 | 0.430 | 7,265 | 48,033 | 55,298 |
| Medium | 12.5% | 8 | 1,866 | 13 | 0.544 | 11,624 | 38,425 | 50,049 |
| Long | 12.5% | 14 | 3,265 | 20 | 0.754 | 20,342 | 20,742 | 41,084 |
| Adopted | 12.5% | 15 | 3,498 | 21 | 0.787 | 21,795 | 17,927 | 39,722 |
| Very long | 12.5% | 21 | 4,897 | 28 | 0.880 | 30,513 | 10,131 | 40,644 |
| Wide | 20% | 10 | 3,731 | 23 | 0.756 | 14,530 | 20,540 | 35,070 |
| Wide and long | 20% | 14 | 5,224 | 30 | 0.882 | 20,342 | 9,946 | 30,288 |

The wide designs look better on this table and are still rejected, which is the most important reading of it.
At 20% the audit runs at 93.3% utilisation, and the table's 65-second review time is a *mean under a
sustainable queue*. Under a saturated queue the mean falls - reviewers cut corners - and the label error
rises. The table prices the days saved and does not price the label degradation, which is why the decision
uses the utilisation constraint as a hard filter before the cost comparison.

**Annual position.** Twelve cycles at the adopted design: engineering 97,200, audit 21,216, delay 261,540,
expected harm 215,124. Total 595,080. At the three-day design: engineering 97,200, audit 4,243, delay 52,308,
expected harm 786,060. Total 939,811. The adopted design is 344,731 cheaper per year, and 92% of the
difference is the expected harm term.

**Benefit-to-cost.** Per cycle, the release benefit over fifteen days is 21,795 against 9,868 of engineering
and audit: 2.21 to 1. The ratio is thin enough that a 40% error in the benefit estimate would make the whole
release programme marginal, which is why the benefit is tracked as a guardrail and not assumed.

**Sensitivity - clustering.** At ICC 0.10 the effective sample falls to 2,360 and the adopted design's power
falls to about 69%. Restoring 80% needs a 21-day canary, whose total is 40,644 - a 2.3% cost increase for
the design to remain valid. Clustering is therefore affordable to accommodate and expensive to ignore.

**Sensitivity - severity.** If 2% of harmful events escalate at 40,000 USD, mean severity rises to 1,517 and
the 21-day exposure to 178,323. The three-day design's expected harm rises to 138,595 and its total to
142,954, while the fifteen-day design's rises to 37,932 and its total to 59,727. The ranking is unchanged
and the gap widens, so the conclusion is robust to the severity assumption in the direction that matters.

## 12. Failure Modes & What Went Wrong

**F1 - Canary that cannot detect the regression it was built for.** The team runs three days, sees zero harms,
and promotes. *Competing causes:* (a) the regression is absent; (b) the sample is too small (22.3% power);
(c) the audit is incomplete so the numerator is undercounted. *Discriminating check:* compute the achieved
power before the canary and the audit completion rate after it; a completion rate below 95% means (c),
a power below 40% means (b), and only after both are excluded is (a) credible. *Correction:* declare the
required n and k* in the manifest, and treat "no decision" as a valid outcome requiring extension.

**F2 - Partial release dressed as a bundle.** The prompt ships on Tuesday and the routing table on Thursday,
each with a clean canary. *Competing causes:* (a) the promotion API allows per-artifact writes; (b) an
emergency change bypasses the manifest; (c) the manifest lists six artifacts but two have the same version
string because they were rebuilt from the same tag. *Discriminating check:* count distinct artifact hashes
promoted per day; more than one promotion event per release identifies (a) or (b), and duplicate hashes
across artifacts identify (c). *Correction:* single promotion verb, content hash per artifact, and an
assertion that the six hashes are distinct.

**F3 - Index contamination that survives rollback.** The bundle is rolled back and the harmful rate does not
return to baseline. *Competing causes:* (a) canary writes remain in the shared index; (b) the rollback did
not revert the retrieval configuration; (c) the stable arm's rate was already drifting. *Discriminating
check:* retrieve the canary-written chunks directly and test whether the stable arm cites them; a nonzero
citation rate identifies (a), a configuration diff identifies (b). *Correction:* tagged namespace with
retrieval exclusion, and a post-rollback assertion that the stable arm's citation rate on canary chunks is
zero.

**F4 - Guardrail set that cannot fail.** Six metrics are monitored, five are always green because their
thresholds were set at the observed maximum plus a margin. *Competing causes:* (a) thresholds were fitted to
the baseline rather than to a harm model; (b) the metrics are lagging and never move within a canary;
(c) the metrics are correlated with the one that matters and add no information. *Discriminating check:*
back-test each threshold against the last twelve releases; a threshold that no historical release would
have breached is decorative. *Correction:* require each guardrail to state the regression it would catch and
the sample it needs, and delete any that cannot.

**F5 - Audit queue saturation.** At a 20% share the audit runs at 93.3% utilisation; review time falls from
65 seconds and the three-way verdict collapses toward "correct". *Competing causes:* (a) volume exceeds
capacity; (b) reviewer headcount fell; (c) packet complexity rose so each review genuinely takes longer.
*Discriminating check:* plot review seconds against daily queue depth; a negative slope is (a) plus coping,
a flat slope with rising seconds is (c). *Correction:* utilisation cap at 0.80 enforced by refusing to start
a canary whose share would breach it.

**F6 - Reviewer unblinding.** The console shows the bundle version in a header, and reviewers find harms in
the canary at twice the rate of the control for a bundle that is actually neutral. *Competing causes:*
(a) the arm label is visible; (b) the canary packet layout differs cosmetically; (c) the canary claims are
genuinely different because the splitter is not random. *Discriminating check:* compare the covariate
distribution across arms; balance implies (a) or (b), imbalance implies (c). *Correction:* blind console,
plus a sham release where both arms run the same bundle and the harm rates must match within the interval.

**F7 - Rollback that exceeds the availability budget.** Two rollbacks in one canary consume 30 minutes
against a 21.6-minute three-day budget, breaching the SLO by 39%. *Competing causes:* (a) rollback is manual
and slow; (b) the queue freeze flushes in-flight claims that must be re-decided; (c) the rollback triggers a
second failure because the prior bundle is incompatible with the current index. *Discriminating check:*
time the phases separately; a long freeze phase identifies (b), a failed post-rollback smoke test identifies
(c). *Correction:* rehearse rollback quarterly, budget it explicitly, and test bundle N-1 against index N
before every release.

**F8 - Optional stopping by the release owner.** The owner watches the harm counter daily and promotes on the
first day it looks low. *Competing causes:* (a) the process has no fixed look schedule; (b) the dashboard is
visible but the decision rule is not; (c) the owner has authority to promote without the rule. *Discriminating
check:* audit the promotion timestamps against the declared duration; a promotion before the declared n is
(a) or (c) by construction. *Correction:* promotion requires a signed rule evaluation, and continuous
monitoring is restricted to trip metrics that can only stop a release, never start one.

## 13. Transferable Lesson

**Sample size is the control, not the intention.** Every release gate has a detection probability, and most
organisations never compute it. Three days versus fifteen is not a process preference; it is 22% versus 79%
detection against a regression worth 84,285 USD. The number that makes a canary real is the critical count
and the sample it implies, both declared before the first claim.

**Test one thing, trip on many.** The distinction between a test (which needs power) and a trip (which needs
only an obvious threshold) dissolves the false dilemma of "monitor everything". Six guardrails at 80% power
need 19,000 claims; six trips need six thresholds and no power calculation at all. Choosing which metric
deserves the test is the real decision, and it should be the irreversible one.

**A rollback that does not reverse the state is a metaphor.** Model versions revert in seconds; indexes,
caches, and derived features do not. Fifteen days of canary writes contaminated 12.5% of the index, and the
residue produced more contaminated retrievals per day than the harmful events the canary was built to find.
Any release process needs an explicit list of what rollback actually reverses.

**Integer decision rules are not smooth.** Under a fixed-alpha exact test, a longer canary can be less
powerful than a shorter one because the critical count jumps. Nine days gives 60.7% power and ten days gives
56.6%. Planning by intuition about monotonicity produces designs that are worse for being longer, and the
table is the only defence.

## 14. Connections & Cross-References

- **Approval gate case** (`01_enterprise_agent_approval_gate.md`): the same economics - avoided loss per
  dollar of control - appears there as review routing; here it appears as canary duration. Both conclude
  that the binding constraint is a sample or a headcount, never the intent.
- **Durable workflows and idempotent execution** (`05_durable_workflows_and_idempotent_execution.md`): the
  queue freeze during rollback must not double-decide claims, which is an idempotency requirement on the
  claim key, not on the bundle.
- **Session isolation and lifecycle** (`04_agent_sessions_state_isolation_and_lifecycle.md`): the index
  quarantine is a state-isolation problem - the canary must not be able to influence the stable arm through
  shared memory.
- **Agent boundaries and request ownership** (`01_agent_boundaries_and_request_ownership.md`): the arm
  assignment is a form of ownership; a claim must belong to exactly one arm for its whole life including
  re-submission.
- **External doctrine.** Shadow launches and staged rollouts are standard practice; the arithmetic for
  powering a release gate is not usually stated, and nothing here is attributed to a published source.

### Interview pack

**Q1. Why does the case reject a three-day canary when three days is what the release cadence allows?**

- Answer: Because three days yields 700 audited claims against a required 3,498, giving 22.3% detection of a
  regression that costs 84,285 USD in excess harm. Cadence is worth 8,100 USD of engineering per cycle;
  the forgone detection is worth 30,142.
- Reasoning: The canary's job is to make a decision, and a test with 22% power makes the wrong decision 78%
  of the time when the regression is present.
- Follow-up: What would you change to make three days sufficient?
- Red flag: "three days is standard practice."
- Scoring: 0 defends cadence; 1 says sample is too small without the number; 2 gives the sample, the power,
  and the cost comparison.

**Q2. Compute the mean severity per harmful event from a 55/28/13/4% mix over 70/320/1,700/9,200 USD bands.**

- Answer: 0.55x70 = 38.5; 0.28x320 = 89.6; 0.13x1,700 = 221; 0.04x9,200 = 368. Total 717.10 USD/event.
- Reasoning: The heaviest band is 4% of events and 51% of the mean, so the mean is dominated by a band whose
  share is estimated from few events.
- Follow-up: What does the mean do if the top band's share doubles?
- Red flag: taking the median band (320) as the mean.
- Scoring: 0 wrong total; 1 correct total without the dominance observation; 2 correct plus the tail
  sensitivity.

**Q3. A canary reports 11 harmful writes in 2,400 audited claims. Has the rate degraded?**

- Answer: The point estimate is 0.458% with a Wilson interval of 0.26% to 0.82%. The threshold is 0.70%, which
  sits inside the interval, so the canary has not decided; extend it rather than promote or roll back.
- Reasoning: The interval is wider than the effect being tested, which is the signature of an underpowered
  canary caught mid-flight.
- Follow-up: How many more claims would resolve it?
- Red flag: "11/2,400 is above 0.4% so it degraded."
- Scoring: 0 compares point estimates; 1 computes the rate; 2 gives the interval and the extend decision.

**Q4. Derive the critical count for a 5% one-sided exact test at n = 3,498 against p0 = 0.4%.**

- Answer: Find the smallest k with P(X >= k | p0) <= 0.05. At k = 20, alpha is 0.0591; at k = 21, alpha is
  0.0473. So k* = 21, and power at p1 = 0.7% is 0.7873.
- Reasoning: The exact tail is used because n*p0 = 14 makes the normal approximation anti-conservative in the
  upper tail.
- Follow-up: Why is k* = 21 rather than 14 (the expected count under p1)?
- Red flag: using the expected count under p1 as the threshold.
- Scoring: 0 uses a z-test; 1 gets k* without power; 2 gives both and the reason for the exact test.

**Q5. Three guardrails need 5,000, 4,500, and 3,700 claims each. What does that imply for a 3,498-canary?**

- Answer: Only the 3,700 one is close; the other two are underpowered and one of them (validity) is expected
  to improve, so it is a trip not a test. The design must nominate a single test.
- Reasoning: The guardrail set's sample requirement is the maximum, not the minimum, and the maximum exceeds
  the canary by 43%.
- Follow-up: What does a Bonferroni correction do to those numbers?
- Red flag: "all guardrails pass, so the release is safe."
- Scoring: 0 ignores the gap; 1 notices the max; 2 states the test-versus-trip distinction.

**Q6. Explain why a 10-day canary can be less powerful than a 9-day one.**

- Answer: At 9 days n = 2,099 and k* = 14, giving power 0.607; at 10 days n = 2,332 and k* = 16, giving
  power 0.566. The critical count jumped by two while the sample grew by 11%.
- Reasoning: Under a discrete exact test, alpha oscillates with n, and the smallest k meeting the alpha
  constraint can rise faster than the sample, reducing rejection probability.
- Follow-up: How would you plot duration against power to make this visible?
- Red flag: "more data always means more power."
- Scoring: 0 asserts monotonicity; 1 notices the discrete jump; 2 explains alpha oscillation.

**Q7. The canary writes to a shared retrieval index. Quantify the residue after rollback.**

- Answer: 3,498 claims x 1.5 chunks = 5,247 chunks of a 42,000-chunk index, 12.5%. At a 1.8% harmful rate
  among canary decisions, 94 poisoned chunks persist; at 1.5 retrievals per day each, about 141 contaminated
  retrievals a day, more than the 7.5 harmful events the canary detects.
- Reasoning: Rollback reverts artifacts, not derived state, so the measurement contaminates its own subject.
- Follow-up: What design change removes the residue?
- Red flag: "rollback makes us safe."
- Scoring: 0 no contamination model; 1 computes the share; 2 computes the share and the daily retrieval
  consequence.

**Q8. What is the design effect at 5.83 claims per handler and ICC 0.05, and what does it do to power?**

- Answer: deff = 1 + 4.83 x 0.05 = 1.241, so 3,498 audited claims behave like 2,818. At k* = 20 that is about
  76% power instead of 78.7%.
- Reasoning: Clustering reduces the effective information; the correction is a variance inflation, not a bias.
- Follow-up: Why does stratifying the randomisation by handler help?
- Red flag: treating claims as independent because each is one row.
- Scoring: 0 ignores clustering; 1 computes deff; 2 gives deff, the power implication, and the remedy.

**Q9. Would you adopt the SPRT for this gate?**

- Answer: No - for the gate. It cuts the expected sample from 3,498 to 2,068 (41%) but assumes i.i.d. claims,
  and the effective n under clustering is 2,630. It also invites optional stopping by an owner who watches
  the trajectory.
- Reasoning: The saving is real and the assumption is false; a tool whose validity depends on an assumption
  you have measured to fail is not a control.
- Follow-up: Where would you use it?
- Red flag: adopting it for the sample saving alone.
- Scoring: 0 rejects without reason; 1 cites the sample saving; 2 cites the saving, the clustering breach,
  and the stopping risk.

**Q10. Two rollbacks occur during one canary. What has happened to the availability SLO?**

- Answer: Two 15-minute rollbacks are 30 minutes against a 21.6-minute three-day budget (0.5% of 1,440 x 3),
  so the SLO is breached by 39%.
- Reasoning: 1440 x 3 x 0.005 = 21.6 minutes of allowed unavailability; one rollback consumes 69.4% of it.
- Follow-up: What does the fourteen-day budget allow?
- Red flag: treating rollback time as free.
- Scoring: 0 no calculation; 1 one rollback's share; 2 both rollbacks against the budget.

**Q11. Diagnose: the canary promoted cleanly, but the stable arm's harmful rate rose two weeks later.**

- Answer: Most likely index contamination (canary-written chunks now retrievable after promotion) or a change
  that only manifests at higher retrieval prevalence. Check the citation rate on canary-tagged chunks.
- Reasoning: Promotion is when quarantine lifts, so the residue enters retrieval exactly then - the failure
  is scheduled by the design, not accidental.
- Follow-up: What would you have measured to catch it during the canary?
- Red flag: "the canary passed, so the bundle is fine."
- Scoring: 0 blames the model; 1 names contamination; 2 names it and the discriminating check.

**Q12. A reviewer sees the bundle version in the console header. What is the risk and the fix?**

- Answer: Confirmation bias - harms are found at a higher rate in the arm the reviewer knows is new. Fix by
  blinding the console and running a sham release where both arms run the same bundle and must match.
- Reasoning: The verdict is a human judgment, so the label is a treatment; unblinding turns the audit into a
  measurement of the reviewer's expectation.
- Follow-up: What would a sham release prove that a blinding change would not?
- Red flag: "reviewers are professionals, they are objective."
- Scoring: 0 no risk seen; 1 mentions bias; 2 gives the sham release as the test.

**Q13. The automation gain is 3.1 points and worth 1,453 USD/day. Is the release programme worth running?**

- Answer: Marginally. Per cycle, benefit over fifteen days is 21,795 against 9,868 of engineering plus audit,
  a 2.21 ratio. A 40% error in the benefit estimate makes the programme marginal, so benefit must be tracked.
- Reasoning: The decision is a ratio of a measured benefit to a mostly fixed cost, and the fixed cost is not
  sensitive to the benefit.
- Follow-up: What would raise the ratio without touching the benefit?
- Red flag: quoting the ratio without the sensitivity.
- Scoring: 0 no ratio; 1 the ratio; 2 the ratio plus the break-even error.

**Q14. What happens if you correct for six guardrail tests with Bonferroni?**

- Answer: Each test's alpha becomes 0.00833, which inflates the required sample by about 1.693. The 3,700
  harmful-write sample becomes roughly 6,300, well beyond the canary, so no guardrail is powered.
- Reasoning: Bonferroni converts a family-wise error target into an individual alpha, and sample size scales
  with the square of the z ratio, hence the 1.69 factor.
- Follow-up: What is the alternative to correcting?
- Red flag: "correct for everything to be safe."
- Scoring: 0 no effect; 1 mentions inflated n; 2 gives the factor and the alternative.

**Q15. Why does the design allow a 15-21 day range rather than a point optimum?**

- Answer: The total is 41,084 at 14 days, 39,722 at 15, and 40,644 at 21, so the curve is flat to within 5%.
  A flat optimum is implementable; a sharp one invites arguments about the third decimal.
- Reasoning: The cost model has an uncertainty larger than the curvature, so choosing 15 versus 18 is not a
  decision the data supports.
- Follow-up: What would sharpen the optimum?
- Red flag: "pick the exact minimum."
- Scoring: 0 no range; 1 gives the numbers; 2 explains the flatness relative to model uncertainty.

**Q16. Which single change most reduces expected harm per release?**

- Answer: Index quarantine plus tagging, because it removes the contaminated-retrieval channel that survives
  rollback and that grows with canary length. Second is the fixed look schedule, which removes optional
  stopping.
- Reasoning: The two failure modes that survive a "successful" canary are the residue and the early
  promotion; both are process defects rather than model defects.
- Follow-up: How would you verify the quarantine works before the first canary?
- Red flag: proposing a better model.
- Scoring: 0 names a model change; 1 names one control; 2 names both with the verification.

## 15. Quick Reference Card

| Question | Answer | Number |
|---|---|---|
| Auto-eligible claims/day | 3,200 x 0.583 | 1,865.6 |
| Baseline vs candidate harm rate | 0.4% vs 0.7% | delta 0.003 |
| Mean harm per event | weighted severity mix | 717.10 USD |
| Excess events/day if shipped | 1,865.6 x 0.003 | 5.597 |
| 21-day exposure | 5.597 x 717.10 x 21 | 84,285 USD |
| Adopted canary sample | 12.5% x 1,865.6 x 15 | 3,498 claims |
| Critical count | smallest k with alpha <= 0.05 | 21 |
| Achieved alpha / power | exact binomial | 0.0473 / 0.787 |
| Detection at 3 days | same rule, n = 700 | 22.3% |
| Cost of a 3-day canary | delay + expected harm | 69,864 vs 39,722 |
| Audit throughput | 233.2 claims/day x 65 s | 4.21 h/day, 58.3% of capacity |
| Index contamination at 15 days | 3,498 x 1.5 / 42,000 | 12.5% |
| Design effect at ICC 0.05 | 1 + 4.83 x 0.05 | 1.241 |
| SPRT expected sample at p1 | Wald boundaries | 2,068 vs 3,498 |
| Rollback share of SLO budget | 15 / 21.6 | 69.4% |
