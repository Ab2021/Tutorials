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

## 1. File Header

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

### 11.1 Roles, adoption, and model-family decision drills

**Role drill — executive sponsor.** The sponsor owns the business decision, prioritization, and acceptance of residual business risk.
The sponsor does not certify a model’s security or independently score evaluation cases.
The sponsor must be able to explain the cost of retaining the baseline.

**Role drill — product owner.** The product owner turns user goals into an eligible-unit and output contract.
The product owner must not redefine success after reviewing candidate performance.
The product owner owns user-facing fallback behavior with operations.

**Role drill — domain reviewer.** The reviewer checks whether an output supports a correct workflow action.
The reviewer is not interchangeable with a click-through approver.
Review capacity, authority, and evidence access must be explicit.

**Role drill — data owner.** The data owner determines provenance, access, retention, and decision-time availability.
Access to a field does not by itself authorize external transfer or model use.
The owner records conditions under which a source becomes stale or invalid.

**Role drill — ML or analytics owner.** This role defines target, split, calibration, uncertainty, and monitoring for predictive use.
It must expose leakage and label delay rather than optimizing an offline score.
For generative use, it defines an evaluation protocol rather than claiming a universal accuracy.

**Role drill — platform owner.** This role owns deployment boundary, identity, logs, version pinning, capacity, and recovery.
It cannot accept business harm on behalf of the decision owner.
A model endpoint without an operator is an unfinished deployment mode.

**Role drill — security and privacy counterpart.** This role maps data flow, access, retention, and threat assumptions.
It does not replace a system-specific authorization test with a generic certification label.
Its decision can be a hard gate, a requested control, or an OPEN evidence request.

**Role drill — procurement or finance partner.** This role checks commercial terms, volume assumptions, exit exposure, and cost variance.
It must model review and integration work rather than comparing only token or seat prices.
It should state what price, usage, or contractual change reopens the decision.

### 11.2 Adoption stages and their evidence

| Stage | Permitted claim | Evidence required | Not yet established |
|---|---|---|---|
| Discovery | a workflow hypothesis exists | owner interview and current-path map | value or technical feasibility |
| Baseline | current performance is measured on a defined population | versioned measurement protocol | intervention improvement |
| Prototype | output can be generated in a constrained setting | input/output trace and exclusions | safe workflow integration |
| Offline evaluation | candidate passes a frozen test under stated rubric | cases, scorer, version, limitations | field outcome |
| Shadow mode | candidate runs without changing workflow action | production-like inputs and discrepancy log | user adoption or safe automation |
| Assisted pilot | named users review output before action | approval trail, capacity and fallback evidence | scalable autonomy |
| Limited production | bounded population receives controlled service | monitoring, incident path, rollback | universal suitability |
| Scale decision | broader scope is considered | slice results, cost, capacity, risk recheck | permanence or immunity to drift |

Adoption is a sequence of narrower claims becoming better evidenced.
No stage authorizes a claim from a later stage.
A polished prototype cannot substitute for an eligibility definition.
An offline score cannot substitute for a review-capacity calculation.
An assisted pilot cannot establish safety for nonreversible autonomy.
The appropriate next stage may be “retain baseline” when evidence does not support expansion.

### 11.3 Model-family scope map

| Family | Input-to-output contract | Strong fit | Primary boundary | First acceptance question |
|---|---|---|---|---|
| Rule system | explicit condition to deterministic output | stable policy and validation | exceptions and policy change | are rules complete enough? |
| Search/retrieval | query to ranked authorized evidence | evidence lookup | corpus freshness and permissions | does the needed evidence exist and rank? |
| Classical statistical model | structured observations to forecast or estimate | interpretable stable relationships | assumptions and uncertainty | is target available at decision time? |
| Supervised ML | features to predicted label, score, or value | repeatable labeled decision | leakage, drift, calibration | are labels valid and timely? |
| Embedding model | content to vector representation | similarity, retrieval, clustering | semantic mismatch and index version | what relevance judgment validates neighbors? |
| Generative language model | context to candidate text or structured output | drafting, transformation, constrained extraction | unsupported output and prompt/data route | can every consequential output be reviewed? |
| Multimodal model | text, image, audio, or video to output | modality-dependent analysis | alignment, privacy, environment shift | are timestamps and source rights valid? |
| Optimization method | objective and constraints to feasible allocation | explicit tradeoffs | infeasibility and unmodeled constraints | who owns objective and overrides? |
| Agentic workflow | state plus tools to multi-step action | bounded tool sequence with recoverable steps | permissions, injection, idempotency, stopping | can each tool action be authorized and recovered? |

A model family is not a deployment mode.
The same language model can run under local, managed, or hosted control boundaries.
The same retrieval method can use different data rights and retention policies.
The same agentic pattern can be proposal-only or state-changing automation.
Selection must therefore name family, delivery, authority, and operation separately.

### 11.4 Adoption measurement definitions

Adoption numerator is the number of eligible users or units that actually use the intervention as defined.
Adoption denominator is the population offered a usable intervention opportunity in the same window.
Activation is a first meaningful use, not merely account creation.
Retention is repeat meaningful use over a declared interval, not continued entitlement.
Override rate is the fraction of intervention outputs materially changed or rejected by an authorized reviewer.
Abstention rate is the fraction of eligible units deliberately routed away from an automated response.
Workload displacement is measured baseline effort minus intervention-path effort including correction.
Outcome movement must be distinguished from adoption; users can adopt a tool that does not improve the decision.
Novelty can inflate short-term use; evaluate at a predeclared later window where appropriate.
Forced use is not evidence of user value or workflow fit.

```text
FORMULA [MAP-04]: a = n_meaningful_use / n_offered
WHERE: a is adoption [fraction]; n_meaningful_use [users or units/window]
       meets a declared meaningful-use rule; n_offered [same unit/window]
       had a usable opportunity to use the intervention.
CONSTRAINT: define opportunity before observing uptake; exclude access outages
            from neither numerator nor denominator without recording them.
NUMERICAL EXAMPLE: ILLUSTRATIVE 360 of 600 eligible analysts use a cited
                   retrieval result in a weekly workflow; a=360/600=60%.
INTERPRETATION: this does not establish that the use was beneficial or that
                the other 240 rejected the tool for quality reasons.
```

Competing causes of low adoption include poor output quality, missing workflow integration, unavailable access, insufficient training, distrust, and lack of relevant cases.
The discriminating check is a reason-coded observation plus availability logs and representative user interviews.
Raising a usage target without this diagnosis can turn adoption into a compliance metric.

### 11.5 Model-family comparison failure cases

**MAP-F07 — Rule system dismissed too early.**
The symptom is a complex model proposal for an explicitly deterministic policy.
Competing causes are unmeasured exceptions, prestige bias, or a genuine need for ranking.
Check exception volume and decision inconsistency before selecting a learned method.
Use rules or validators first when policy is stable and exception handling is observable.
Verify by measuring residual exception workload and policy-change maintenance.

**MAP-F08 — Embedding similarity mistaken for authorization.**
The symptom is a semantically relevant document shown to an unauthorized user.
Competing causes are post-retrieval filtering, stale ACL index, or identity mapping failure.
Test access enforcement before ranking display with revoked and cross-role cases.
Fix the authorization path rather than tuning embedding similarity.
Verify that prohibited records cannot be retrieved or cited under the test identity.

**MAP-F09 — Generative fluency mistaken for source-grounded correctness.**
The symptom is a persuasive answer containing unsupported operational advice.
Competing causes are missing retrieval, stale evidence, weak source selection, or rubric that rewards style.
Inspect citation entailment and source freshness separately from readability.
Require abstention or manual search when evidence is absent.
Verify with adversarial missing-source cases and human rubric review.

**MAP-F10 — Predictive score treated as a decision policy.**
The symptom is a calibrated or discriminative score shipped without a threshold or capacity policy.
Competing causes are missing cost matrix, unexamined class imbalance, or lack of reviewer capacity.
Plot outcome, harm, and workload across thresholds on the held-out population.
Choose a threshold jointly with action owner and capacity owner.
Verify the actual routed volume and error slice after controlled rollout.

**MAP-F11 — Agentic workflow treated as a chat interface.**
The symptom is an unbounded tool sequence triggered by a casual prompt.
Competing causes are overbroad permissions, absent stopping rules, untrusted context, or retry behavior.
Trace every tool contract, input source, authority, idempotency key, and recovery path.
Constrain the workflow to proposal-only until each action passes separate evidence gates.
Verify that injection-like content cannot expand authority or bypass approval.

**Takeaway:** Adoption and family selection are measurable workflow decisions, not popularity contests or model-label comparisons.

### 11.6 Career-track boundary map

| Track | Primary decision contribution | Core evidence produced | Does not own alone | First skill to demonstrate |
|---|---|---|---|---|
| Business analyst | frames workflow and baseline | eligible-unit map and metric definitions | production access control | distinguish a count from a rate |
| Product manager | sets user outcome and tradeoffs | decision contract and rollout gate | independent risk acceptance | write a falsifiable success condition |
| Data analyst | measures baseline and slices | reproducible query or measurement record | causal claim beyond design | reconcile numerator and denominator |
| Data scientist | estimates/predicts under assumptions | target definition, split, uncertainty analysis | permission to deploy action | identify leakage in a proposed feature |
| ML engineer | packages model behavior into service | versioned artifact, test, monitor, rollback path | business prioritization | reproduce a candidate run |
| Data engineer | makes governed inputs available | lineage, freshness, schema, access contract | model-quality claim | trace event time through a pipeline |
| Software engineer | integrates bounded behavior | input/output contract and failure recovery | label validity | make retries idempotent |
| Platform engineer | operates shared deployment surfaces | identity, capacity, observability, incident controls | workflow benefit claim | size a rate and a backlog |
| Security engineer | tests threat and authorization boundaries | scoped control evidence and risk finding | product usefulness | demonstrate least-privilege enforcement |
| Privacy or records specialist | checks purpose and retention boundaries | field-flow and retention decision record | quality scoring | distinguish collection from permitted use |
| Domain expert | defines correct action and harmful error | rubric and escalation policy | infrastructure reliability | adjudicate a realistic boundary case |
| Operations lead | owns queue and fallback behavior | service workflow, staffing, and recovery drill | model training objective | explain peak-load capacity |
| Procurement partner | evaluates commercial commitments | comparable quote, term, and exit record | technical security conclusion | model price and volume sensitivity |
| Risk or compliance partner | identifies non-negotiable constraints | gate rationale and escalation route | generic “approval” for all risk | articulate scope and limitation |

Career tracks overlap in practice.
The table is a responsibility map, not a hierarchy or a staffing prescription.
One person may cover several roles in a small team only when conflicts and competence boundaries are made explicit.
The decision owner must still be named when technical contributors have broad responsibilities.
No career title proves readiness for an unbounded autonomous action.

### 11.7 Value-chain map

```text
source data -> governed preparation -> model or rule asset -> serving interface
    -> workflow integration -> human/automated action -> outcome measurement
    -> monitoring, incident response, version and evidence maintenance
```

Data value comes from permitted, timely, interpretable information, not merely volume.
Preparation value comes from reliable identity, schema, and availability semantics.
Model value comes from a task-specific transformation evaluated against a baseline.
Serving value comes from latency, cost, reliability, and controlled version behavior.
Integration value comes from placing output at a decision point with authority and fallback.
Outcome value comes from a credible comparison of action consequences.
Operations value comes from detecting drift, failures, and evidence expiry before harm grows.
Each link can be the binding constraint even when another link appears technically impressive.

### 11.8 Deployment-mode decision records

**Local controlled inference.**
Use when a permitted environment, predictable data route, or offline capability is a hard constraint.
Measure hardware capacity, update ownership, isolation, patching, and recovery.
Do not assume local execution automatically creates adequate access control or audit.
Fallback can be a manual procedure or a smaller controlled service.

**Managed inference API.**
Use when the data route, contractual terms, latency, rate limits, and service operation fit the workflow.
Measure actual input/output usage, retry behavior, version controls, and retention configuration.
Do not treat a managed API as transferring accountability for business decisions.
Fallback must handle provider error, rate limits, version changes, and unavailable responses.

**Self-hosted service.**
Use when control over runtime, version, or integration is required and operating capability exists.
Measure deployment reproducibility, capacity, incident ownership, and patch cadence.
Do not choose it merely to avoid a quoted API price while ignoring operating labor.
Fallback must cover infrastructure failure and model artifact rollback.

**Embedded capability.**
Use when latency, connectivity, privacy, or device constraints require local execution at the interaction point.
Measure device variation, model size, thermal/power boundaries, and update path.
Do not assume laboratory latency transfers to a device fleet.
Fallback should be safe degraded behavior rather than indefinite retry.

**Batch processing.**
Use when decisions tolerate delay and aggregating work improves cost or throughput.
Measure freshness window, failure recovery, partial completion, and rerun idempotency.
Do not use batch simply because synchronous latency has not been measured.
Fallback should make stale output visible and provide manual prioritization.

**Workflow platform integration.**
Use when the deployment value is coordinating state, approvals, and tools rather than a single model call.
Measure authority propagation, state ownership, duplicate handling, and tool failure behavior.
Do not call a chained prompt sequence a safe workflow without these controls.
Fallback should stop side effects and preserve enough state for deliberate recovery.

### 11.9 Unresolved-source handling protocol

The visible inventory identifies `GLCC`, `learn`, `raw`, and `speechtotxt` as unavailable containers.
Their names are provenance identifiers, not subject-matter evidence.
Record four unresolved source nodes, one for each visible container.
Do not calculate a hidden-topic completion ratio from those node names.
The current decision is whether exact content is required for a new teaching claim.
If it is required, request a lawful content or filesystem export with the node identity preserved.
If it is not required, finish visible-topic work and retain an OPEN limitation.

```text
FORMULA [MAP-05]: u = n_unresolved / n_source_nodes
WHERE: u is inventory-level unresolved-node fraction [fraction];
       n_unresolved [nodes] is the count of visible source nodes without
       accessible bodies; n_source_nodes [nodes] is all visible nodes included
       in the stated inventory, not a count of hidden lessons.
CONSTRAINT: u describes inventory access only; it must never be interpreted
            as hidden-topic completeness, quality, or curriculum coverage.
NUMERICAL EXAMPLE: ILLUSTRATIVE n_unresolved=4 named containers and
                   n_source_nodes=40 visible inventory nodes; u=4/40=10%.
INTERPRETATION: 10% of listed nodes are unresolved by this artificial inventory;
                zero hidden-topic conclusions follow from that percentage.
```

The independent check is `0.10*40=4` unresolved nodes.
If another inaccessible node is found, the numerator and denominator must be recomputed together.
The failure boundary is any use of node count as a surrogate for unknown content volume.
The mitigation is an explicit OPEN record, not a guessed syllabus.

**Takeaway:** Deployment and career choices are made at different links in the value chain; inaccessible source containers remain inventory limitations, not invented curriculum evidence.

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
