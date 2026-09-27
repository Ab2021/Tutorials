# T19 — FinOps, Token Economics & Sovereignty: production artifacts

> `T19` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md) · [Sequences](../docs/SEQUENCES.md) · [Runnable core](../run.py) · [Case study](../../../01-case-studies/T19-finops-sovereignty.md)

**Reference-grade, not copy-paste.** Every artifact here encodes one of this topic's measured findings, and the finding is quoted above the artifact so a reviewer can see which decision the file is carrying. A config whose rationale is not written down is a config that will be edited back to the default.

| # | Artifact | Owner | Encodes |
|---|---|---|---|
| 0 | `unit-memo.md` | finance + the owning team | **the decision that precedes every measurement** |
| 1 | `bill.yaml` | finance | the decomposition is the model; a derived line is re-derived |
| 2 | `levers.yaml` | platform | one pricing slot, ranked by `banked` not by `claim` |
| 3 | `sovereignty.yaml` | security + legal | the **floor**, the prerequisite cap, residency as a filter |
| 4 | `attribution.yaml` | platform | coverage reported **beside** growth; reconcile before chargeback |
| 5 | `metrics.promql` | SRE | the six readings a general cost dashboard omits |
| 6 | `cost-event.schema.json` | platform | the four fields that make a cost record evidence |
| 7 | `bootcheck.py` | platform | refuse to start on posture and residency violations |

Artifact **0** is first because the build order is the design (LLD §10). It contains no configuration — it is the only artifact in this directory that is a *decision*, and every other file here is downstream of it.

---

## 0. `unit-memo.md` — the denominator (finance + owning team)

```markdown
# production/unit-memo.md

# WHY THIS FILE EXISTS
# "Companies are doing tokens per output as one of the metrics -- you need to change the metrics.
#  You need to start looking at it as a business outcome rather than the token spend."      [T]
#
# MEASURED [D], the corpus's own two data points (30c -> 16c, 630 -> 91 lines, 8.2 -> 3.6 files):
#     per artifact   0.5333   -46.7%   cheaper          <- the headline unit
#     per file       1.2148   +21.5%   MORE expensive
#     per line       3.6923  +269.2%   MORE expensive   <- the delivered-work unit
# SAME TWO ROWS. THREE SIGNS. The unit is chosen before the arithmetic and decides its sign.

programme: inference-cost-reduction-2026
signed_by: [finance, platform, the owning team]
signed_on: 2026-09-27           # BEFORE any optimising starts. This date is the whole point.

primary_unit:
  name: cost_per_resolved_case
  denominator: cases closed as resolved          # not "cases handled", not "calls"
  owner_of_denominator: customer-operations      # NOT the team being measured
  why: >
    The corpus's preferred direction [R]. The denominator is owned by the consuming function, so the
    measured team cannot improve its own number by doing less work.

companion_metric:
  name: cost_per_line_delivered
  denominator: lines merged to main
  why: >
    The counterpart that caught the Feb->Apr reversal. If the primary unit improves while this one
    degrades, the programme is shipping less work per case -- which the primary unit cannot see.

explicitly_rejected_units:
  - name: cost_per_call
    why: "a team can split one task into more calls. 1% of calls are 29.2% of the bill [D]."
  - name: tokens_per_output
    why: >
      reasoning tokens are billed at the output rate and absent from the response -- a call with 300
      visible and 2,000 thinking tokens scores 1.0x on this metric and costs 7.67x [D]. The metric is
      SYSTEMATICALLY OPTIMISTIC because the way to improve it is to move work out of view.
  - name: cost_per_task
    why: >
      gameable by making the task smaller, which is exactly what the Feb->Apr figures show happened.
      Allowed only if `task_size_p50` is reported beside it.

review_cadence: quarterly
re_derive_from: the metering ledger, raw counts only     # a unit that cannot be re-derived cannot be
                                                          # defended when a team disputes a number
```

### 0.1 Why the denominator's owner is not the measured team

The whole design turns on one sentence in this file: `owner_of_denominator: customer-operations`. If
the team whose cost is being measured also owns the denominator, the metric is a target it can move
without changing the business — and the corpus's own figures show that this is not hypothetical, it is
what happened between February and April. The **line count fell to 14.4%** of its February value while
the cost fell only to 53.3% **[D]**.

---

## 1. `bill.yaml` — the decomposition (finance)

```yaml
# production/bill.yaml
# The decomposition IS the model. Every lever's worth is computed against these shares, because
#     lever saving = reachable_share_of_layer x discount x share_of_bill
# and the last factor is what converts a DISCOUNT into an AMOUNT.
#
# A lever's claimed multiple is not its worth. MEASURED [D] on this bill:
#   prompt_caching     claim 10x   ->  banked 23.2%   (#1 by banked, #3 by claim)
#   distillation       claim 40x   ->  banked  2.4%   (#1 by claim,  #8 by banked)
# If these shares are wrong, every ranking downstream is wrong. Replace the six illustrative lines
# with measured values BEFORE ranking any lever.

version: 2026-Q3
assert_sums_to: 1.00        # enforced at load. A bill that does not sum is a modelling error.

layers:
  - name: system_prompt
    share: 0.3105
    derived: true           # re-derived at load from the two inputs below, never trusted as typed
    derive_from:
      prefix_share_of_input: 0.69    # [R] Datadog 2026, via the guide
      input_share_of_bill: 0.45      # [D] illustrative
  - name: model_tier
    share: 0.2000
    derived: false
  - name: output_length
    share: 0.1600
    derived: false
  - name: reasoning_tokens
    share: 0.1500
    derived: false
    note: "billed at the OUTPUT rate and absent from the response"
  - name: retrieved_context
    share: 0.0795
    derived: false
  - name: conversation_memory
    share: 0.0700
    derived: false
  - name: retry_overhead
    share: 0.0300
    derived: false

cache:
  read_discount: 10.0       # [T] "roughly ten times cheaper". Providers quote 50-90% OFF, which is
                            # a 2x-10x multiple -- DIFFERENT UNITS. 90% off == 10x, not 90x.
  write_premium: 1.25       # [R]
  cached_share_today: 0.28  # [R]
  cached_share_target: 0.90

  # MEASURED [D], the curve and its floor:
  #   discount   2x -> banked 11.2% |  5x -> 19.8% | 10x -> 23.2%
  #             20x -> 24.9%       | 50x -> 26.0% | 100x -> 26.4%
  #             INFINITE -> 26.7%  <-- THE ASYMPTOTE. A free cache read cannot do better.
  # The uncached 10% of calls still pays full price, and nothing outside the prefix is touched.
  asymptote_banked: 0.267   # assert this at load; a plan projecting more than this is invalid

  # MEASURED [D], the write premium:  break-even reads = 1.25 / (1 - 1/10) = 1.39
  #   read ONCE:  uncached 1.000 vs cached 1.350  ->  +0.350, A BILL INCREASE
  #   read TWICE: uncached 2.000 vs cached 1.450  ->  -0.550, worth it
  min_read_write_ratio: 2.0 # break-even is 1.39; operate at 2 and DISABLE below it, per prefix
  disable_scope: per_prefix # NOT global. A global switch cannot express a per-prefix rule.
```

### 1.1 Why `derived: true` and `assert_sums_to` are load-bearing

The 31.05% line is the largest line in the bill and it is **derived**, not chosen: `0.69 × 0.45`. Two
failure modes are prevented by declaring it as such. First, a reviewer who disagrees with the bill can
argue with the two inputs rather than with the product. Second, the number cannot drift — if the guide
revises the 69% prefix share, the derivation fails loudly instead of silently ranking caching first
against a stale share.

`assert_sums_to: 1.00` catches the other error: a lever's worth is computed as a share of the total, so
a bill that sums to 0.97 makes every lever look 3% weaker than it is, uniformly, and no individual
number looks wrong.

### 1.2 Why the discount and the percentage are called out as different units

`cached_share_target: 0.90` and `read_discount: 10.0` sit three lines apart and are **not** the same
kind of number — and providers advertise the first kind while the corpus's lever is stated in the
second. A "90% off" cache is a **10×** discount. Conflating them makes the prefix line look nine times
cheaper than it is. The runner has an explicit `percent_to_multiple()` for exactly this conversion, and
it is the only place in the codebase where a percentage becomes a multiple.

---

## 2. `levers.yaml` — the registry (platform)

```yaml
# production/levers.yaml
#
# MEASURED [D], this bill, every lever at its full claimed discount, disjoint shares:
#   CEILING                                       60.61%   = 2.54x
#   headline [T] "roughly a tenth of the cost"    90.00%   = 10.00x    GAP 29.39 points
#
#   best ordering  (saving_first, pricing on value) 59.30%
#   free_first     (easy reduction, pricing last)   56.55%
#   worst ordering (ease_first)                     55.17%   SPREAD 4.15 points
#
#   the ladder's most aggressive claim, at its stated reach: 2.02 points OF THE CEILING.
#   It would have to be applied to the whole non-prefix bill to matter -- which is the exact
#   application its own source calls a failure mode: distillation "wins on narrow, high-volume
#   tasks and fails on open-ended long-tail work" [R], and the long tail is where the bill is
#   going (see `mix` below).
#
# ORDER OF OPERATIONS -- not negotiable:
#   1. Decide the PRICING lever. There is ONE slot per traffic slice.
#   2. Sort the REDUCTION levers by `banked`, never by `claim`.
#   3. Quote the CEILING, never the headline.

ceiling: 0.6061
committed_target: 0.55          # below the ceiling, with margin. NOT 0.90.

pricing:                        # one slot: these two reprice the SAME tokens
  - name: reserved_capacity
    share: 0.45
    discount: 0.35              # 15-70% [R]
    banked: 0.1575
    chosen: true
    ease: 2
    caveat: "a commitment. It costs whether or not the traffic arrives -- mirror it into the policy
             store. Conflicts with difficulty_routing: never commit to capacity for traffic you
             intend to route away."
  - name: batch_lane
    share: 0.20
    discount: 0.50              # ~50% with a ~24h ceiling [R]
    banked: 0.1000
    chosen: false               # skipped by conflict -- and the skip IS REPORTED, not dropped
    ease: 4

reduction:
  - name: prompt_caching
    target: [[system_prompt, 1.00, 0.746]]
    claim_multiple: 10.0
    claim_notation: multiple
    banked: 0.2316              # #1 by banked, #3 by claim
    ease: 5
    caveat: >
      Only a STABLE prefix. "Cache something that changes each call and you gain nothing" [T].
      A nonce, a timestamp or a reordered tool list at the front invalidates everything downstream.
      Read the write premium before enabling: break-even is 1.39 reads, and a prefix read once is a
      bill INCREASE.
  - name: reasoning_gating
    target: [[reasoning_tokens, 1.00, 0.60]]
    claim_multiple: 15.0
    claim_notation: multiple
    banked: 0.0900              # #2 by claim, #4 by banked
    ease: 4
    eval_on: tasks_gated_off    # NOT tasks_gated_on. The failure is a task that needed to think.
  - name: difficulty_routing
    target: [[model_tier, 0.55, 0.50]]
    claim_multiple: 2.0
    claim_notation: percentage
    banked: 0.0550
    ease: 3
    conflicts: [distillation]
  - name: output_caps
    target: [[output_length, 1.00, 0.30]]
    banked: 0.0480
    ease: 5
    caveat: "a schema, not a character count. A cap truncates; a schema steers."
  - name: context_discipline
    target: [[retrieved_context, 1.00, 0.30], [conversation_memory, 1.00, 0.30]]
    banked: 0.0449
    ease: 4
  - name: distillation
    target: [[model_tier, 0.15, 0.80]]
    claim_multiple: 40.0
    claim_notation: multiple    # 5-40x [R]
    banked: 0.0240              # #1 BY CLAIM (40x), #8 BY BANKED. The whole point of this file.
    ease: 1
    conflicts: [difficulty_routing]
    caveat: "distil the easy path, NEVER the tail. The tail is where the bill is going."
  - name: quantisation
    target: [[model_tier, 0.30, 0.40]]
    claim_multiple: 2.0
    claim_notation: none        # UNQUANTIFIED in the corpus. The 2.0 is the AUTHOR'S READING.
    banked: 0.0240
    ease: 2
    caveat: "'nearly free on many models, but on some it quietly drops accuracy -- always rerun your
              evals' [T]. Self-hosted only."
  - name: bounded_retries
    target: [[retry_overhead, 1.00, 0.50]]
    claim_multiple: 1.0
    claim_notation: none        # UNQUANTIFIED
    banked: 0.0150
    ease: 4
    caveat: "the ceiling must sit INSIDE the loop, not in an after-the-fact alert."

mix:
  growth_factor: 4.3            # [D] the scenario's YoY token growth
  note: >
    MEASURED [D]: today the agentic population is 1.0% of calls and 29.2% of the bill, at 29.2x the
    mean call. One growth cycle makes it 63.9% of the bill. RE-RANK THE LEVERS QUARTERLY -- an order
    derived from today's average is the wrong order for next year's bill.
```

### 2.1 Why `chosen` and `chosen: false` are both present

`batch_lane` is in the file with `chosen: false` and its numbers intact, rather than deleted. A lever
that is removed from a plan gets re-added by the next person, and the re-addition is where the
double-count happens — the two pricing levers would each report their full banked saving against the
same tokens. Keeping it visible, with its value and its conflict, makes the trade auditable:
15.75 points against 10.00 points, decided once, on value.

### 2.2 Why `claim_notation: none` is a required field

Three levers are unquantified in the corpus. A registry that forced a number on them would invent a
claim and give it the same authority as the corpus's own. `none` marks the multiple as the *author's
reading*, so an interviewer, a reviewer or a successor can see which of the ten numbers are citations
and which are inferences.

---

## 3. `sovereignty.yaml` — the posture (security + legal)

```yaml
# production/sovereignty.yaml
#
# MEASURED [D], claimed against effective, with the mean and the FLOOR:
#   posture             claimed  effective   capped      mean    floor   gap
#   compliance_only       0.438      0.438   --         0.438   0.100   0.338
#   regional_attested     0.825      0.812   trust      0.812   0.750   0.062
#   own_accelerators      0.887      0.887   --         0.887   0.850   0.037
#   own_stack             0.975      0.975   --         0.975   0.950   0.025
#   api_with_dpa          0.288      0.288   --         0.288   0.050   0.238
#   vendor_tee_lease      0.500      0.413   TRUST      0.413   0.300   0.113
#
# THE POSTURES THAT SOUND MOST DEFENSIBLE ON A SLIDE HAVE THE LARGEST mean-to-FLOOR GAP.
# A contract and a regional endpoint do not make the model yours.

prerequisites:                  # DATA, not code. A reviewer can veto a rule; not a function.
  trust: control                # you cannot attest an environment you did not choose
  continuity: control           # you cannot switch vendors if the vendor picks the model

thresholds:
  floor_min: 0.60               # FAIL if any single dimension is below this
  mean_min: null                # deliberately null. A mean threshold IS the failure mode.

current_posture: regional_attested
postures:                       # kept in-file so the rejected options and their numbers stay visible
  - name: api_with_dpa
    cost_index: 0.85
    scores: { control: 0.55, trust: 0.05, economics: 0.30, continuity: 0.25 }
    rejected_because: "floor 0.05 -- the cheapest posture and the widest claim-to-evidence gap"
  - name: vendor_tee_lease
    cost_index: 1.05
    scores: { control: 0.50, trust: 0.85, economics: 0.35, continuity: 0.30 }
    rejected_because: >
      claims 0.85 trust and can evidence 0.50. The attestation is TRUE and about SOMEBODY ELSE'S
      MACHINE: the silicon and the model are not ours, so it proves a property of an environment we
      did not choose. This entry is why the prerequisite rule exists.
  - name: own_stack
    cost_index: 1.60
    scores: { control: 1.00, trust: 0.95, economics: 0.95, continuity: 1.00 }
    rejected_because: "cost index 1.60 for a floor of 0.95 we do not need"

residency:
  # A FILTER, never a score. MEASURED [D], gap 1.0 / noise 1.0:
  #   score-based router  ->  15.87% leak rate  ->  1,586.6 leaks per 10,000 requests
  #   filter              ->  0 leaks
  # A legal property has NO error tolerance: 99.84% correct residency is NON-COMPLIANT.
  mode: filter
  filter_field: allowed_regions     # the out-of-region option is REMOVED from the candidate set
  leak_rate_allowed: 0.0            # exactly zero. asserted in test, NOT monitored in prod.
  fail_open: false                  # an unattributable region is a rejection, not a fallback

  # WHY `mode: filter` AND NOT `mode: score, threshold: 0.99`:
  # a score-based implementation has a PARAMETER for tolerance. A filter does not, and that absence
  # is the design. Do not add a threshold to this block.

continuity:
  reading: minimum                  # NOT mean
  layers: [accelerator, model_family, serving_engine, cloud_region]
  min_options_per_layer: 2          # 1 is not continuity
  # MEASURED [D]: three of four example stacks score a MEAN of 0.75 and a FLOOR of 0.0.
  #   {acc 2, model 3, engine 2, region 1}  -> floor 0.0, binding cloud_region
  #   {acc 2, model 2, engine 1, region 3}  -> floor 0.0, binding serving_engine
  #   {acc 1, model 4, engine 3, region 3}  -> floor 0.0, binding ACCELERATOR
  strategy: >
    The cheapest continuity purchase in most stacks is NOT a second cloud -- it is a second SERVING
    ENGINE on the same weights. Accelerator choice is fixed earliest and is therefore the most
    expensive layer to change. "The choice of accelerator is extremely important... it should be
    part of the implementation strategy" [T].

attestation:
  attested_share: 0.40              # attest the regulated slice, not everything
  throughput_penalty: 0.20          # MEASURED [T]: "it is not going to run as fast as what it was
                                    # running before. But it is going to run more securely."
  cost_uplift: 0.25                 # = 1/(1-0.20) - 1 . NEVER put the penalty in this key.
  applied_to: post_optimisation_bill  # optimisation and attestation hit the SAME tokens
  # MEASURED [D], the uplift table, so a reviewer can check the conversion:
  #   penalty  5% -> uplift  5.3%     penalty 20% -> uplift 25.0%
  #   penalty 10% -> uplift 11.1%     penalty 30% -> uplift 42.9%
  #   penalty 40% -> uplift 66.7%
  # A 20% penalty is a 25% cost uplift because CAPACITY is what you buy.
  # Getting this wrong in the direction of the penalty undersizes every TEE budget by a fifth.

open_weights:
  requires: [weights, training_code, training_data, reproducible]
  # "when people say open source model they'll just publish open weights. They don't show the source
  #  code which was used to train, and they don't showcase the silicon..." [T]
  # MEASURED openness:  closed API 0.00 | open weights 0.25 | + training code 0.50 | open source 1.00
  # Only `reproducible` counts as a continuity answer. Everything else is a supplier's goodwill.
```

### 3.1 Why `cost_uplift` is a separate key from `throughput_penalty`

Every business-case error in this area is entering the penalty where the uplift belongs, and the error
is always in the same direction — the sovereign option looks cheaper than it is. Two keys force the
conversion to be an act performed by a person, rather than a number copied from a vendor deck. The
uplift table is quoted in-file so the act can be checked.

### 3.2 Why `leak_rate_allowed: 0.0` and not `0.001`

A tolerance of 0.001 is a tolerance, and the number it tolerates is a legal exposure. The runner's
`residency_compliance()` returns compliant **iff** the rate is exactly zero, and there is no threshold
parameter to set — a future engineer cannot configure an acceptable leak rate into this system without
noticing that the function has no parameter for one.

---

## 4. `attribution.yaml` — the ledger (platform)

```yaml
# production/attribution.yaml
#
# "Tag every call by team, feature, tenant, model, route and environment. Without attribution there
#  is no way to compute unit economics." [R]  -- showback BEFORE chargeback, and chargeback only
#  "once the tags are trustworthy".
#
# MEASURED [D], the finding that shapes this file: coverage FALLS as the business succeeds.
#   before:  coverage 90.46%   untagged 9.54%    bill 1,000,000
#   after:   coverage 82.43%   untagged 17.57%   bill 1,916,667   (one growth cycle)
#   NOBODY TAGGED WORSE. The untagged spend is concentrated in the team growing 6.5x.
#   Untagged spend is NOT a random sample of the bill.

required_tags: [team, feature, tenant, model, route, environment]
tags_are_mandatory: true          # an untagged request is REJECTED at the gateway, not defaulted

coverage:
  min_spend_weighted: 0.90
  report_beside: [team_growth, untagged_spend]     # NON-OPTIONAL
  # WHY: a team at 55% coverage growing 6.5x is the finding. It is invisible in the global coverage
  # figure AND in the per-team percentage. The PAIR is the signal; either alone is not.
  alert_rule: "coverage < 0.90 AND growth > 2.0"
  report_per_team: true

governance:
  stage: showback                 # showback -> chargeback -> per-run caps, in that order [R]
  chargeback_requires: trusted_tags
  # "Showback before chargeback" is a sequencing instruction and the reason is political as well as
  # technical: chargeback on untrusted tags produces a fight about the numbers instead of a plan.

  reconciliation:
    enabled: true                 # MUST exist BEFORE chargeback, not after
    cadence: hourly               # not quarterly. This is the only off-gateway detector.
    compare: [gateway_spend, provider_invoice]
    alert_residual_pct: 0.01
    # MEASURED [D]: gateway 880,000 vs invoices 1,000,000 -> residual 120,000 = 12.0%.
    # A 12% residual is not a rounding error; it is traffic that left the gateway. Once spend is
    # billed to a team, the cheapest route FOR THAT TEAM is an account the gateway never sees.

caps:
  per_run_ceiling: 5.00
  derive_from: p99_task_cost      # NOT the mean
  # "Runaway agents have burned tens of thousands of dollars over a single weekend" [R] is an
  # argument for a ceiling. It is NOT an argument for a SMALL one.
  # MEASURED [D]: mean task 0.42c, p99 12.24c. A $5 ceiling from the TAIL kills nothing legitimate
  # and still catches a runaway two orders of magnitude above the mean. From the MEAN, it terminates
  # legitimate long tasks -- the mean is not a smaller tail, it is a different population.
  terminate_on_breach: true
  alert_source_statistic: true    # a termination from a mean-derived cap is itself an alert
```

### 4.1 Why `report_beside` is in the file

`coverage: 90.46%` passes the threshold, and `coverage: 82.43%` also passes it. Neither number, alone,
says anything is wrong — the failure is a *drift* with no threshold crossing, and it is only visible
when the growth rate is printed next to it. Making `report_beside` a required key means a dashboard
that omits the companion quantity fails to load rather than passing review.

### 4.2 Why `reconciliation` precedes `chargeback`

Chargeback is the rung that *creates* the incentive to leave the gateway. Instrumenting the residual
after enabling chargeback means a period during which spend can leave invisibly and nothing detects
it. The ordering in this file — reconciliation enabled and firing at 1%, before `stage: chargeback` is
reached — is the design, not a preference.

---

## 5. `metrics.promql` — the six readings a general dashboard omits

```promql
# production/metrics.promql
# A general-purpose cost dashboard emits one reading: spend per unit. This file emits that AND the
# five companions that turn it into a decision. Each companion below is the ONLY signal for one of
# the failure modes in HLD 8.1.

# ---------------------------------------------------------------------------------------------
# 1. THE PRESCRIBED ONE -- and the instruction NOT to alert on it.
#    cost_per_unit is the corpus's direction [R] and it is where every programme starts.
#    It is the metric T19 exists to warn about, and it is present so #2 has something to compare to.
# ---------------------------------------------------------------------------------------------
# alert: NEVER DIRECTLY. See #2.

# ---------------------------------------------------------------------------------------------
# 2. THE SIGN DISAGREEMENT. THE ONE THAT MATTERS.
#    MEASURED [D]: 30c -> 16c is -46.7% per artifact, +21.5% per file, +269.2% per line delivered.
#    A single unit CANNOT show this. Two units can.
# ---------------------------------------------------------------------------------------------
# ALERT: the flag is true for two consecutive periods
rate(unit_sign_disagreement[30d]) > 0

# ---------------------------------------------------------------------------------------------
# 3. THE CACHE WRITE PREMIUM, PER PREFIX. The only lever in the ladder that can RAISE the bill.
#    MEASURED [D]: break-even is 1.39 reads at a 10x read discount and a 1.25x write price.
#    A prefix read once costs +0.350 units. A dashboard reports "caching: ON" either way.
# ---------------------------------------------------------------------------------------------
sum by (prefix) (rate(cache_reads_total[1h]))
  / sum by (prefix) (rate(cache_writes_total[1h]))
# ALERT: any prefix below 2 for a full period

# ---------------------------------------------------------------------------------------------
# 4. THE PREFIX CACHED SHARE, AGAINST ITS ASYMPTOTE.
#    MEASURED [D]: the lever's banked saving is capped at 26.7% at an INFINITE discount, because
#    the uncached 10% of calls still pays full price. A programme chasing more than this is paying
#    for a discount that cannot be spent.
# ---------------------------------------------------------------------------------------------
sum(rate(cache_read_tokens_total[1h]))
  / sum(rate(input_tokens_total[1h]))
# ALERT: below target. AND: refuse to approve a business case projecting > 26.7% from caching alone.

# ---------------------------------------------------------------------------------------------
# 5. BANKED SAVING PER LEVER, AGAINST ITS CLAIM.
#    MEASURED [D]: prompt_caching 23.2% banked from a 10x claim; distillation 2.4% from a 40x one.
#    EIGHT OF TEN LEVERS change rank between claim order and banked order.
# ---------------------------------------------------------------------------------------------
sum by (lever) (lever_banked_saving_ratio)
# ALERT: a lever whose banked saving is more than 5 points below the claim in levers.yaml

# ---------------------------------------------------------------------------------------------
# 6. THE CEILING GAP. What the programme committed to against what the ladder can reach.
#    MEASURED [D]: ceiling 60.61% (2.54x). The corpus's headline is 90% (10x). GAP 29.39 points.
#    Alert on the GAP, not on the achievement -- a widening gap means the target is drifting from
#    the bound, which no individual lever's metric can show.
# ---------------------------------------------------------------------------------------------
ceiling_ratio - committed_target_ratio
# ALERT: the gap widens for two consecutive quarters -> re-baseline the target

# ---------------------------------------------------------------------------------------------
# 7. PRICING LEVERS SKIPPED BY CONFLICT. A double-count caught before it ships.
#    MEASURED [D]: without conflict enforcement the stack reports 66.98% against a 60.61% ceiling.
# ---------------------------------------------------------------------------------------------
sum(rate(pricing_lever_conflicts_skipped_total[1h]))
# ALERT: ANY non-zero value in a deployed plan

# ---------------------------------------------------------------------------------------------
# 8. COVERAGE *BESIDE* GROWTH, PER TEAM. They are ONE metric.
#    MEASURED [D]: coverage 90.46% -> 82.43% under one growth cycle, untagged 9.54% -> 17.57%.
#    A global 90% is not "90% of the picture" -- it is the whole picture minus the part about to
#    matter. The pair is the signal.
# ---------------------------------------------------------------------------------------------
attribution_coverage   # by (team)
team_growth            # by (team)
# ALERT: coverage < 0.90 AND growth > 2.0, for the same team

# ---------------------------------------------------------------------------------------------
# 9. THE GATEWAY-INVOICE RESIDUAL. The only detector for spend that left the gateway.
#    MEASURED [D]: 880,000 gateway vs 1,000,000 invoiced -> 12.0% residual.
#    This is the failure mode chargeback INTRODUCES, so it must be live before chargeback is.
# ---------------------------------------------------------------------------------------------
(provider_invoice_amount - gateway_spend_amount) / provider_invoice_amount
# ALERT: above 0.01

# ---------------------------------------------------------------------------------------------
# 10. WHERE THE CAP CAME FROM. A termination is an event; its SOURCE STATISTIC is the metric.
#     MEASURED [D]: mean task 0.42c vs p99 12.24c. A mean-derived $5 cap kills legitimate work.
# ---------------------------------------------------------------------------------------------
sum by (source) (rate(runaway_terminations_total[1h]))   # source: mean | tail
# ALERT: any termination where source="mean"

# ---------------------------------------------------------------------------------------------
# 11. THE SOVEREIGNTY FLOOR -- not the mean.
#     MEASURED [D]: compliance_only mean 0.438 / floor 0.100 (gap 0.338);
#                    api_with_dpa    mean 0.288 / floor 0.050 (gap 0.238).
# ---------------------------------------------------------------------------------------------
min by (posture) (sovereignty_dimension_score)
# ALERT: below floor_min. The MEAN is never the alert -- it is what a slide shows.

# ---------------------------------------------------------------------------------------------
# 12. CLAIMED AGAINST EFFECTIVE, PER POSTURE. The cap is the metric.
#     MEASURED [D]: vendor_tee_lease claims 0.85 trust and evidences 0.50 -- it is attesting a
#     property of an environment it did not choose.
# ---------------------------------------------------------------------------------------------
sovereignty_claimed_mean - sovereignty_effective_mean
# ALERT: any posture with a non-empty `capped` set in production

# ---------------------------------------------------------------------------------------------
# 13. RESIDENCY LEAK RATE. A DEFECT, never a warning.
#     MEASURED [D]: a score-based router at gap 1.0 / noise 1.0 leaks 15.87% -> 1,586.6 per 10,000.
#     A filter leaks zero. A legal property has no error tolerance.
# ---------------------------------------------------------------------------------------------
sum(rate(residency_out_of_region_total[1h]))
# ALERT: ANY non-zero value. This is a page, not a ticket -- and the fix is the candidate set,
# not a better score.

# ---------------------------------------------------------------------------------------------
# 14. THE CONTINUITY FLOOR, WITH ITS BINDING LAYER.
#     MEASURED [D]: three of four example stacks score a MEAN of 0.75 and a FLOOR of 0.0.
# ---------------------------------------------------------------------------------------------
min by (layer) (continuity_option_count) >= 2
# ALERT: floor at 0 while the mean is >= 0.70 -- the exact signature of a mean hiding a floor

# ---------------------------------------------------------------------------------------------
# 15. THE INVISIBLE BILL LAYER.
#     MEASURED [D]: a call with 300 visible and 2,000 reasoning tokens scores 1.0x on
#     "tokens per output" and costs 7.67x. Reasoning is billed at the OUTPUT rate.
# ---------------------------------------------------------------------------------------------
sum(rate(reasoning_tokens_total[1h]))
  / (sum(rate(output_visible_tokens_total[1h])) + sum(rate(reasoning_tokens_total[1h])))
# ALERT: a RISING share while the visible-token metric is flat -- that is work moving out of view
```

### 5.1 The alert that is deliberately absent

There is no alert on metric 1, and the absence is the file's most important design decision. `cost per
unit` is the metric the entire topic exists to interrogate; alerting on it would make the dashboard
complicit in the error it is meant to catch. It is emitted, because #2, #5 and #6 all need something
to compare against — but a threshold on it would be a threshold on the wrong quantity.

---

## 6. `cost-event.schema.json` — the record that is evidence

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "t19.cost_event.v1",
  "title": "T19 cost event",
  "type": "object",
  "required": ["schema", "ts", "request_id", "tenant", "team", "route", "tokens", "unit",
               "pricing", "cost", "tags_complete"],
  "properties": {
    "schema":     { "const": "t19.cost_event.v1" },
    "ts":         { "type": "string", "format": "date-time" },
    "request_id": { "type": "string" },
    "tenant":     { "type": "string" },
    "team":       { "type": "string" },
    "feature":    { "type": "string" },

    "route": {
      "type": "object",
      "required": ["tier", "residency", "filtered_out"],
      "properties": {
        "tier":         { "enum": ["small", "mid", "frontier"] },
        "residency":    { "type": "string" },
        "filtered_out": {
          "type": "array", "items": { "type": "string" },
          "description": "WHY THIS FIELD EXISTS: residency is a FILTER, so the decision is what was REMOVED. A score-based router cannot emit this field -- which is the point."
        }
      }
    },

    "tokens": {
      "type": "object",
      "required": ["input_uncached", "input_cached_read", "input_cache_write",
                   "output_visible", "output_reasoning"],
      "properties": {
        "input_uncached":    { "type": "integer" },
        "input_cached_read": { "type": "integer" },
        "input_cache_write": { "type": "integer",
          "description": "the numerator of the write premium. Without it, a prefix read once looks free." },
        "output_visible":    { "type": "integer" },
        "output_reasoning":  { "type": "integer",
          "description": "BILLED AT THE OUTPUT RATE AND ABSENT FROM THE RESPONSE. Omit this and the 'tokens per output' denominator is systematically optimistic by up to 7.67x." }
      }
    },

    "unit": {
      "type": "object",
      "required": ["name", "denominator_label"],
      "properties": {
        "name":              { "type": "string" },
        "numerator":         { "type": "number" },
        "denominator_label": { "type": "string" }
      },
      "description": "THE DENOMINATOR TRAVELS WITH THE EVENT. This is what makes the unit memo (artifact 0) enforceable rather than aspirational -- the unit is not a dashboard definition that can quietly change."
    },

    "pricing": {
      "type": "object",
      "required": ["lane"],
      "properties": {
        "lane":           { "enum": ["on_demand", "batch", "reserved"] },
        "commitment_id":  { "type": ["string", "null"] },
        "batch_eligible": { "type": "boolean" }
      },
      "description": "A reserved commitment is a cost whether or not the traffic arrives -- which is why the commitment id is on the event and mirrored into the policy store."
    },

    "sovereignty": {
      "type": "object",
      "properties": {
        "attested":                   { "type": "boolean" },
        "attestation_id":             { "type": ["string", "null"] },
        "attested_share_of_traffic":  { "type": "number" },
        "throughput_penalty":         { "type": "number" },
        "cost_uplift_applied":        { "type": "number",
          "description": "= 1/(1-penalty) - 1. Per event, so 'what did sovereignty cost us' is answerable from the ledger rather than from a spreadsheet." }
      }
    },

    "cost": {
      "type": "object",
      "required": ["input", "output", "total"],
      "properties": {
        "input":  { "type": "number" },
        "output": { "type": "number" },
        "uplift": { "type": "number" },
        "total":  { "type": "number" }
      }
    },

    "tags_complete": {
      "type": "boolean",
      "description": "false is a REJECTION at the gateway, not a defaulted value. Coverage is a fraction, and a defaulted tag makes the fraction lie."
    }
  }
}
```

### 6.1 The four fields that make this evidence rather than logging

**`route.filtered_out`** records the *removal*, not the outcome. A residency implementation that scores
cannot produce it, so this field is the schema-level test for whether residency is a filter.

**`tokens.output_reasoning`** is the token class that is billed and invisible. It is the only field in
the record that a general-purpose LLM observability schema omits, and without it the most common cost
metric in the industry is systematically optimistic.

**`unit`** carries the denominator **on the event**. Dashboards can be redefined; an event's unit is a
record of what was measured at the time. This is what lets a team win a dispute about a number that
was computed under a denominator they later changed.

**`sovereignty.cost_uplift_applied`** stores the *uplift*, not the penalty, per event. It makes the
conversion in artifact 3 auditable after the fact rather than only at planning time.

---

## 7. `bootcheck.py` — refuse to start

```python
#!/usr/bin/env python3
"""production/bootcheck.py -- refuse to start.

Five conditions. Each one is a state where the system would run, report green, and be measuring the
wrong thing -- or be non-compliant while reporting a high score. A boot check rather than an alert,
because all five are SILENT: nothing crosses a threshold, so nothing pages.

    python production/bootcheck.py --config production/ [--ledger-dir path]

Exit 0 = safe to serve. Exit 1 = at least one condition failed, with the reason printed.
Adapted from T18's pattern (a control that can degrade without reporting must fail at boot).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FAILURES: list[str] = []


def fail(cond: str, detail: str) -> None:
    FAILURES.append(f"{cond}: {detail}")


def check_bill_sums(cfg: dict) -> None:
    """1. THE BILL MUST SUM TO 1.00.

    Every lever's worth is a share of the total, so a bill summing to 0.97 makes every lever look 3%
    weaker than it is -- uniformly, so no individual number looks wrong.
    """
    total = sum(l["share"] for l in cfg["layers"])
    if abs(total - 1.0) > 1e-9:
        fail("BILL_SUMS", f"layers sum to {total:.6f}, must be 1.00")


def check_prefix_derivation(cfg: dict) -> None:
    """2. A DERIVED LINE MUST RE-DERIVE.

    system_prompt is 0.3105 = 0.69 x 0.45. If either input moves, the product must move with it --
    otherwise the largest line in the bill silently ranks caching against a stale share.
    """
    for l in cfg["layers"]:
        if not l.get("derived"):
            continue
        d = l["derive_from"]
        want = d["prefix_share_of_input"] * d["input_share_of_bill"]
        if abs(want - l["share"]) > 1e-6:
            fail("DERIVED_DRIFT", f"{l['name']}: declared {l['share']}, derives to {want:.6f}")


def check_pricing_slot(cfg: dict) -> None:
    """3. EXACTLY ONE PRICING LEVER MAY BE CHOSEN.

    Two pricing levers reprice the same tokens -- one saving, counted twice. Without this check the
    stack can report a reduction that EXCEEDS its own ceiling (66.98% against 60.61%).
    """
    chosen = [p["name"] for p in cfg["pricing"] if p.get("chosen")]
    if len(chosen) != 1:
        fail("PRICING_SLOT", f"{len(chosen)} pricing levers chosen: {chosen}. There is one slot.")


def check_committed_target_below_ceiling(cfg: dict) -> None:
    """4. THE COMMITTED TARGET MUST BE ACHIEVABLE.

    ceiling 60.61% (2.54x). The corpus's headline is 90% (10x). A committed target above the
    ceiling is a target the ladder cannot reach, and committing to it is how a programme starts
    inventing savings.
    """
    if cfg["committed_target"] > cfg["ceiling"]:
        fail("TARGET_ABOVE_CEILING",
             f"target {cfg['committed_target']:.4f} > ceiling {cfg['ceiling']:.4f}")


def check_sovereignty_floor(cfg: dict) -> None:
    """5a. THE FLOOR, NOT THE MEAN.

    A posture with one dimension at 0.05 and a mean of 0.288 passes a mean test and is not a
    posture. The alert threshold is the MINIMUM.
    """
    PREREQ = cfg["prerequisites"]
    DIMS = ("control", "trust", "economics", "continuity")
    postures = {p["name"]: p for p in cfg["postures"]}
    cur = postures[cfg["current_posture"]]
    eff = {d: min(cur["scores"][d], cur["scores"].get(PREREQ.get(d, d), 1.0)) for d in DIMS}
    floor = min(eff.values())
    if floor < cfg["thresholds"]["floor_min"]:
        fail("SOVEREIGNTY_FLOOR",
             f"{cfg['current_posture']} floor {floor:.3f} < {cfg['thresholds']['floor_min']}")
    capped = [d for d in DIMS if eff[d] < cur["scores"][d]]
    if capped:
        fail("PREREQ_CAPPED", f"{cfg['current_posture']} claims above its prerequisite: {capped}")


def check_continuity_floor(cfg: dict) -> None:
    """5b. CONTINUITY IS A MINIMUM.

    Two accelerators, two model families and two clouds with ONE serving engine is a single-vendor
    system. The mean says 0.75; the floor says zero. The floor is the reading that ships.
    """
    c = cfg["continuity"]
    counts = c.get("current_option_counts", {})
    if counts:
        bad = [layer for layer in c["layers"] if counts.get(layer, 0) < c["min_options_per_layer"]]
        if bad:
            fail("CONTINUITY_FLOOR", f"layers with fewer than {c['min_options_per_layer']} options: {bad}")


def check_residency_is_a_filter(cfg: dict) -> None:
    """5c. RESIDENCY MUST BE A FILTER, AND MUST TOLERATE ZERO.

    A score-based router leaks 15.87% of requests at gap 1.0 / noise 1.0 (1,586.6 per 10,000).
    A filter leaks zero. A legal property has no error tolerance, so ANY non-zero tolerance
    configured here is a boot failure rather than a warning.
    """
    r = cfg["residency"]
    if r["mode"] != "filter":
        fail("RESIDENCY_MODE", f"mode is {r['mode']!r}; residency must be a filter")
    if r.get("leak_rate_allowed", 0.0) != 0.0:
        fail("RESIDENCY_TOLERANCE",
             f"leak_rate_allowed is {r['leak_rate_allowed']}; a legal property has no tolerance")


def check_attribution_preconditions(cfg: dict, ledger_dir: Path | None) -> None:
    """5d. COVERAGE MUST BE REPORTED BESIDE GROWTH, AND RECONCILIATION MUST PRECEDE CHARGEBACK.

    Coverage drifts without crossing a threshold (90.46% -> 82.43% in one growth cycle), so a
    coverage figure without the growth rate beside it cannot detect its own failure. And
    chargeback creates the incentive to leave the gateway, so the reconciliation must already
    be live when it is enabled.
    """
    a = cfg["attribution"]
    if "team_growth" not in a["coverage"]["report_beside"]:
        fail("COVERAGE_BESIDE", "coverage.report_beside must include team_growth")
    if a["governance"]["stage"] in ("chargeback", "caps") and not a["governance"]["reconciliation"]["enabled"]:
        fail("RECONCILE_BEFORE_CHARGEBACK",
             f"stage is {a['governance']['stage']} with reconciliation disabled")
    if a["caps"]["derive_from"] != "p99_task_cost":
        fail("CAP_SOURCE", f"caps derive from {a['caps']['derive_from']}; must be the tail, not the mean")

    # Coverage, if a ledger is supplied.
    if ledger_dir and (ledger_dir / "coverage.json").exists():
        cov = json.loads((ledger_dir / "coverage.json").read_text())
        sw = cov.get("spend_weighted_coverage", 1.0)
        if sw < a["coverage"]["min_spend_weighted"]:
            fail("COVERAGE", f"spend-weighted coverage {sw:.4f} below {a['coverage']['min_spend_weighted']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ledger-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = json.loads((args.config / "t19.config.json").read_text())

    check_bill_sums(cfg)
    check_prefix_derivation(cfg)
    check_pricing_slot(cfg)
    check_committed_target_below_ceiling(cfg)
    check_sovereignty_floor(cfg)
    check_continuity_floor(cfg)
    check_residency_is_a_filter(cfg)
    check_attribution_preconditions(cfg, args.ledger_dir)

    if FAILURES:
        print("REFUSING TO START. Conditions failed:")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nAll of these are SILENT failures: nothing crosses a threshold, so nothing pages.")
        return 1

    print("boot check passed: bill closed, one pricing slot, target under ceiling, posture above its")
    print("floor, continuity above zero at every layer, residency a filter with zero tolerance,")
    print("coverage reported beside growth, reconciliation live before chargeback.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 7.1 Why a boot check and not an alert

All eight conditions above are **silent**: none of them crosses a threshold at run time, so no alert
can fire on any of them. A bill summing to 0.97, a second pricing lever, a target above the ceiling, a
posture whose trust exceeds its control, a residency implementation that scores — each one produces a
system that runs cleanly and reports green while measuring the wrong thing or being non-compliant.

The boot check converts each of them from a *state* into an *event*, at the only moment when somebody
is paying attention: deployment. It is the same pattern T18 applies to guardrail rails, applied here to
the measurement plane — and the reason it belongs in this directory rather than in the HLD is that it
is the one artifact that makes all the others enforceable.

---

## Sources

| File under `refs/` | Used for |
|---|---|
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | the ladder and its ordering rule; "exhaust the free wins"; "roughly a tenth of the cost"; cache reads ~10× cheaper; the stable-prefix caveat; the quantisation accuracy warning; the buy-vs-build guidance |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | the Feb→Apr artifact study behind artifact 0; "tokens per output" as the metric to change; attestation; open-weights-vs-open-source |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Sovereign_AI_Inference_Own_Your_AI._Control_Your_Data.txt` | the four dimensions; the system-property framing; the TEE trade-off sentence behind `cost_uplift`; accelerator choice as implementation strategy |
| `Agentic_AI_Infra_transcripts_2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery_Building_the_Full_Stack_of_A.txt` | the 10–100× agentic multiplier behind the mix block |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/04-finops-and-token-economics.md` | the eight-layer decomposition; the 69%/28% anchor; provider caching discount ranges; batch and provisioned pricing; attribution-first and showback-before-chargeback; the runaway-agent case behind the cap rule |
| `ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` | cascade tiering; distillation's long-tail failure; the reasoning multiplier range |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/03-ai-gateways-and-model-routing.md` | the gateway as attribution and budget-enforcement point |

Related: [HLD](../HLD.md) · [LLD](../LLD.md) · [Sequence diagrams](../docs/SEQUENCES.md) · [Runnable core](../run.py) · [Case study](../../../01-case-studies/T19-finops-sovereignty.md)
