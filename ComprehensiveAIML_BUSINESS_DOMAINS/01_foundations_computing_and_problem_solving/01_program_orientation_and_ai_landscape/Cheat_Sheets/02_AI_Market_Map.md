---
collection: "program-orientation-ai-landscape"
file_id: "02"
topic: "AI market map and deployment modes"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Material change to a candidate system, its data rights, or the applicable operating constraints"
notation_style: "plain-text"
prerequisites: ["01_AI_Use_Case_Charter.md"]
pattern_tags: ["problem-framing", "build-buy", "human-oversight"]
source_status: "OPEN — directory inventory consulted; named source bodies unavailable"
owner_role: "product and technical decision owner"
source_index: "INDEX.md"
---

# AI Market Map: Capability, Control, and Deployment Choice

> **Purpose:** Classify an AI proposal by the decision it supports, the asset it consumes, and the control boundary it changes before comparing vendors or models.
> **Prerequisites:** [AI Use-Case Charter](01_AI_Use_Case_Charter.md); ability to distinguish a rate from a count and a measured result from an estimate.
> **Key outcome:** Given a workflow and constraints, select a smallest credible capability/deployment category and name the evidence that could overturn it.
> **This file does NOT cover:** model training recipes, procurement law, or a claim that a named vendor has any listed capability.
> **Level:** 1 Foundations
> **Version / reviewed:** 1.0 / not reviewed
> **Evidence:** DERIVED classification rules; ILLUSTRATIVE arithmetic; OPEN source-body boundary.
> **Retrieval cue:** If a proposal says “we need AI” without naming the decision or control boundary, start at Section 2.

**Source inventory and boundary.** `Train/Templates/directory.txt` records unavailable containers `GLCC`, `learn`, `raw`, and `speechtotxt`; it establishes neither their contents nor a product claim. The primary references below support only general method definitions. Every workload and dollar value in this sheet is illustrative.

## 2. Concept Map / Mental Model

```text
[Workflow trigger and eligible unit]
             |
             v
[Decision: retrieve / classify / forecast / generate / act]
             |
             +-- deterministic policy? --> [rule or search baseline]
             |
             +-- predictive signal? -----> [statistical/ML candidate]
             |
             +-- unstructured assistance? -> [generative candidate]
             |
             v
[Control boundary: local -> managed API -> managed workflow -> autonomous action]
             |
             v
[Data rights, latency, cost, audit, human approval]
             |
             +-- failed constraint --> [narrow task or retain baseline]
             v
[Evaluation and operating owner]
```

A market map is not a popularity chart. It is a decision map: an offering belongs where its input, output, control boundary, and proof burden belong. “LLM” is a model-family label; it says nothing about retrieval quality, authorization, reliability, or who may act on an output.

**Takeaway:** Choose a category only after the decision and control boundary are explicit.

## 3. Foundational Definitions

| Term | Meaning | Unit/type | Valid use | Nonexample |
|---|---|---|---|---|
| Eligible unit | One item on which a decision can be made | ticket, document, request, account | denominator for coverage and cost | a vague “user base” |
| Capability | Transformation from declared input to output | contract | compare alternatives with same task | a vendor brand |
| Deployment mode | Where inference/state runs and who operates it | control boundary | assess data, latency, change control | model quality score |
| Retrieval | Find and present relevant existing evidence | ranked records | source-grounded lookup | invented answer without evidence |
| Prediction | Estimate a defined target before the outcome is known | probability/value | thresholded action with label definition | post-outcome scoring |
| Generation | Produce candidate text/code/media from context | candidate artifact | assisted work with review | evidence of factual correctness |
| Automation | Execute a state-changing workflow step | side effect | only with identity, idempotency, and approval policy | sending an unreviewed suggestion |
| Human-in-the-loop | A named person can inspect, change, reject, or escalate | ownership contract | consequence exceeds validated autonomy | decorative “review available” |

Four market layers are useful but overlap: **assets** (data, labels, compute, identity), **capability** (retrieve, predict, generate, optimize), **delivery** (library, hosted API, application, managed service), and **operation** (monitoring, incident ownership, access control, release process). A comparison that skips a layer hides risk.

**Takeaway:** Capability, delivery, and autonomy are independent axes.

## 4. Core Formulas with Full Derivation

### 4.1 Base: eligible coverage

Validity before use: each eligible unit is counted once in a fixed window and the eligibility rule is frozen. This measures reach, not correctness.

Derivation: if `n_eligible` units meet the inclusion rule and `n_served` receive a usable system response, the fraction served is served units divided by eligible units.

```text
FORMULA [MAP-01]: c = n_served / n_eligible
WHERE: c is coverage [fraction]; n_served and n_eligible are counts [units]
       over the same population and time window.
CONSTRAINT: 0 <= n_served <= n_eligible; retries must not create extra units.
NUMERICAL EXAMPLE: n_eligible=2,000 requests/day; n_served=1,500 requests/day;
                   c=1,500/2,000=0.75 = 75%.
INTERPRETATION: a quality score on served traffic cannot describe the 500 unserved requests.
```

Boundary/counterexample: if a safety filter excludes difficult requests, coverage can fall while measured quality on the remaining subset rises. Report both, by material slice.

### 4.2 General: expected incremental value

Validity before use: the baseline and intervention are evaluated on comparable eligible units; benefit and harm are monetary estimates, not observed outcomes unless sourced.

Derivation: incremental value is the sum of expected per-unit incremental benefit minus expected per-unit harm, less fixed and variable operating costs. Let `p_g` be probability of a beneficial outcome and `p_h` probability of a harmful outcome under the proposed intervention; `b` and `h` are their respective monetary consequences.

```text
FORMULA [MAP-02]: V = N*(p_g*b - p_h*h - k) - F
WHERE: V [currency/window]; N eligible units [units/window]; p_g,p_h [fractions];
       b [currency/beneficial unit]; h [currency/harmful unit];
       k variable cost [currency/unit]; F fixed cost [currency/window].
CONSTRAINT: probabilities and consequences must refer to the same action and window;
            do not count savings already present in the baseline.
NUMERICAL EXAMPLE: ILLUSTRATIVE N=10,000/month, p_g=0.20, b=$8,
                   p_h=0.01, h=$50, k=$0.20, F=$3,000/month.
                   V=10,000*(0.20*8 - 0.01*50 - 0.20)-3,000
                    =10,000*(1.60-0.50-0.20)-3,000=$6,000/month.
INTERPRETATION: the proposal has a positive estimate only under the stated probabilities,
                values, and costs; it is not a measured ROI claim.
```

Independent arithmetic check: beneficial value is `$16,000`; expected harm `$5,000`; variable cost `$2,000`; subtracting fixed `$3,000` gives `$6,000`. Units resolve to currency/month.

### 4.3 Variant: review-capacity-limited autonomy

A system may be accurate enough yet unsafe to scale if uncertain outputs exceed reviewers. Let `q` be the fraction routed to review, `N` eligible units/day, and `r` review capacity in units/day.

```text
FORMULA [MAP-03]: q*N <= r
WHERE: q [fraction], N [units/day], r [units/day].
CONSTRAINT: each routed unit consumes one review slot; queueing, rework, and absence are excluded.
NUMERICAL EXAMPLE: N=2,000/day and r=240/day implies q <= 240/2,000=0.12.
INTERPRETATION: a policy routing 20% would create at least 160 unreviewed arrivals/day
                under this simplified steady-state model.
```

Sensitivity: if capacity falls to 180/day, the threshold is 9%, not 12%. Increasing model confidence thresholds can reduce `q` but may also reduce coverage or miss harmful cases; test the joint curve.

### 4.4 Limits and counterexample

A high offline score is not a deployment mode. It fails to answer whether data may leave a boundary, whether output is fresh enough, and whether an action can be reversed. If two options have identical score but one needs a prohibited external transfer, selection is decided by the constraint, not score.

**Takeaway:** Economics and coverage are useful only with a frozen unit, comparator, and operating constraint.

## 5. Worked Numerical Example

**Scenario — illustrative, not measured.** A support team considers three modes for 10,000 eligible requests/month: (A) search plus a human response, (B) hosted generation with mandatory citation and review, (C) an autonomous action workflow. The action changes account state, so C has a higher proof burden.

| Quantity | A: search | B: assisted generation | C: autonomous action |
|---|---:|---:|---:|
| Served units/month | 9,500 | 8,000 | 7,000 |
| Variable cost/unit | $0.05 | $0.35 | $0.45 |
| Proposed review fraction | 1.00 | 0.10 | 0.02 |
| Fixed cost/month | $500 | $3,000 | $7,000 |
| Reversible action? | yes | yes | no for some actions |

For B, coverage is `8,000/10,000 = 80%`. Variable cost is `8,000*$0.35=$2,800`; total declared operating cost is `$5,800/month`. Its review load is `8,000*0.10=800 requests/month`. If reviewers have 40 requests/day for 20 working days, capacity is `800/month`, exactly the modelled load; there is no absence or rework headroom.

The arithmetic is checked by reverse multiplication: `800/8,000=10%`; `$2,800+$3,000=$5,800`. A 20% increase in eligible volume at unchanged review staffing yields 960 reviews/month, exceeding capacity by 160. Therefore the proposal cannot call B scalable without a queue policy, staffing change, or a threshold adjustment evaluated for quality and harm.

**Takeaway:** A deployment choice is rejected when its review and reversibility contract fails, even if generation looks attractive.

## 6. Visual / Geometric Intuition

Think of a proposal as a point in a three-dimensional space:

```text
                 higher consequence
                        ^
                        |       autonomous action
                        |             x
                        |   assisted generation x
                        | search x
                        +----------------------------> broader data/control boundary
                       low review burden -------> high review burden
```

Move right when external data transfer, vendor dependency, or privilege expands. Move up when a wrong output changes money, rights, safety, or a customer record. Move toward high review burden when ambiguity is common. As distance from the origin grows, require stronger evidence, narrower authorization, better rollback, and a named operational owner.

**Takeaway:** More capable is not automatically more suitable; control and consequence move the proof burden.

## 7. Comparison / Selection Table

| Condition | First choice | Why | Do not choose when | Fallback |
|---|---|---|---|---|
| Answer exists in controlled documents | retrieval with citations | preserves source inspection | corpus is stale or authorization cannot be enforced | manual search |
| Output is a stable numeric target | statistical/ML prediction | explicit target and calibration can be tested | target is delayed/invalid or inputs leak future data | rule/triage queue |
| Drafting is valuable but a person owns final output | generative assistance | compresses drafting work | output cannot be reviewed at required granularity | templates and search |
| Operation changes a record or sends an instruction | bounded automation | can reduce handoffs if reversible | identity, idempotency, approval, or rollback is missing | propose-only workflow |
| Data cannot cross an external boundary | local/controlled deployment or no deployment | boundary is a hard constraint | vendor contract does not meet it | retain baseline |
| Demand is bursty and cost-sensitive | queued/batched assistance where delay permits | exposes budget and capacity controls | deadline needs synchronous response | smaller deterministic path |

**Takeaway:** A selection table must include a disqualifier and a fallback, not a single “best” column.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|---|
| MAP-F01 | “AI” requirement has no measurable success | solution-first framing; hidden political goal | ask for decision owner, baseline, action, and harm | pause model comparison; write charter | owner signs a falsifiable decision contract |
| MAP-F02 | Offline quality is high but operations reject output | leakage; evaluation population differs; policy constraint | time-availability audit and slice-by-slice eligibility comparison | rebuild split and narrow scope | held-out, point-in-time evaluation passes declared gate |
| MAP-F03 | Review queue grows | volume increase; routing threshold too low; review time underestimated | compare arrivals, routing fraction, completion timestamps | cap intake or add staffing; do not silently drop work | backlog trend is non-positive at peak workload |
| MAP-F04 | Vendor comparison uses incomparable scores | different task, dataset, latency, or refusal policy | require shared test protocol and raw output samples | run controlled acceptance evaluation | protocol records equal inputs, window, and scoring |
| MAP-F05 | Hosted deployment violates data boundary | undocumented subprocessor; overly broad prompt logging; identity mapping | map every field and retention path | remove sensitive field or choose permitted boundary | security/owner review of actual contract and configuration |
| MAP-F06 | Automation repeats a harmful action | timeout retry; non-idempotent endpoint; duplicate event | replay same idempotency key and inspect state history | add idempotency key and approval state | duplicate replay produces one durable effect |

A mitigation can introduce harm: tighter filters reduce coverage, batching adds delay, and removing fields can degrade decision quality. Record the tradeoff rather than calling the failure fixed.

**Takeaway:** Diagnose mechanism before changing a threshold or buying another tool.

## 9. Complexity & Cost Analysis

| Operation | Input size | Time / cost model | Space/state | Important assumption |
|---|---|---|---|---|
| Keyword/document retrieval | `D` indexed documents, `k` returned | index-dependent; measure p95 latency | index plus query state | authorization filtering occurs before display |
| Batch prediction | `N` units | O(N) model invocations | batch/feature buffer | no retries or cache misses in estimate |
| Generative assistance | `N` requests, `t_in+t_out` tokens/request | provider/local token cost times N | context and log retention | output tokens and retries are budgeted |
| Human review | `N*q` routed units | reviewer-minutes/unit times routed units | queue and audit trail | service-time distribution has a tail |

For B in Section 5, if each review averages 6 minutes, 800 reviews need 4,800 minutes = 80 reviewer-hours/month before rework. At 20 workdays this is four hours/day, not “10% of traffic” in an operational vacuum. The estimate excludes context switching and escalation; measure those before approval.

**Takeaway:** Convert a routing fraction into reviewer-hours and queue headroom before a scale claim.

## 10. Key Theorems / Results Reference

| Result | Use | Conditions / limit | Source |
|---|---|---|---|
| Expected value linearity | aggregate stated per-unit expected gains/costs | does not establish probabilities or causal effects | DERIVED from expectation linearity |
| Queue stability intuition | arrival rate must not exceed sustainable completion rate over the relevant window | variable arrivals/service need queueing analysis, not only means | DERIVED workload model |
| Calibration concept | probability estimates should match observed frequencies on declared population | sample size and distribution shift matter | [NIST/SEMATECH e-Handbook](https://www.itl.nist.gov/div898/handbook/) — method reference, accessed 2026-11-09 |
| Least privilege | authorization should grant only required permissions | implementation needs a system-specific threat model | [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final), 2020, accessed 2026-11-09 |

These sources do not validate an illustrative workload or vendor claim.

**Takeaway:** A named principle narrows a design argument; it does not replace acceptance evidence.

## 11. Interview / Exam Questions

1. **Why is “use an LLM” not a problem statement?** It omits the user decision, baseline, action, harm, and operating boundary. Start with those; an LLM may later be a candidate.
2. **Calculate coverage for 840 served out of 1,200 eligible units.** `840/1,200=0.70`, or 70%. State the eligibility rule and excluded 360 units.
3. **A vendor reports 92% accuracy. What must you ask?** Task, population, label definition, class mix, split/time boundary, abstentions, metric numerator/denominator, latency/cost, and raw error slices.
4. **When does retrieval beat generation?** When the task is locating controlled, inspectable facts and a source-grounded answer is sufficient; it loses when required facts are absent or the retrieval corpus is stale.
5. **What is the strongest baseline for an automation proposal?** The current workflow measured on the same eligible units, including human time, errors, delays, and escalations—not an arbitrary weaker model.
6. **A review queue is growing despite 95% offline quality. Diagnose.** Compare arrivals to completions, routing policy, eligibility, reviewer service-time tail, and rework; quality alone does not set capacity.
7. **How does a nonreversible action change the design?** Require narrower authorization, pre-action validation, approval/segregation where needed, durable audit, and an explicit escalation path; a high score is insufficient.
8. **What changes if data must remain inside a controlled boundary?** Deployment becomes a hard constraint; compare permitted local/controlled options or retain the baseline before optimizing model score.
9. **Explain the $6,000 estimate in MAP-02.** It is expected incremental value under illustrative probabilities and costs. It is not revenue until the comparator and probabilities are observed or credibly estimated.
10. **What would falsify a “save reviewer time” claim?** A frozen comparison showing equal or higher reviewer-minutes after verification, correction, and escalation, or insufficient capacity at expected arrivals.
11. **Name a plausible wrong method for vendor selection.** Selecting the highest published benchmark. It fails because tasks, data, refusal policies, cost, and operational controls may differ.
12. **If volume doubles, what must be rechecked?** Review capacity, token/call cost, concurrency/latency, rate limits, error budget, data retention, and whether quality holds for new slices.

**Takeaway:** Defend a choice by its decision contract and falsification path, not by model vocabulary.

## 12. Quick Reference Card

| Cue | Rule | Use when | Critical assumption | Avoid when |
|---|---|---|---|---|
| Start | Name unit, decision, baseline, owner | any AI proposal | workflow is observable | outcome is undefined |
| Classify | Separate capability, delivery, autonomy | comparing offerings | task is stable enough to describe | labels replace evidence |
| Bound | Treat data/control constraints as gates | sensitive or regulated workflow | boundary is known | contract/configuration is unknown |
| Size | `q*N <= r` for review feasibility | human review is required | stable-rate simplification | tail/service variation is ignored |
| Value | `V=N(p_g*b-p_h*h-k)-F` | preliminary prioritization | comparable baseline and stated values | called measured ROI |
| Promote | Test quality, harm, cost, capacity, rollback | moving beyond prototype | versioned evaluation exists | demo is the only evidence |
| Stop | Retain baseline or narrow scope | a hard constraint fails | owner accepts the boundary | risk is merely relabeled |
