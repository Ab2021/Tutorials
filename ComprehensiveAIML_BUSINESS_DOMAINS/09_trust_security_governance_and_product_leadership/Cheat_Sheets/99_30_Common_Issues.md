---
collection: "part09-trust-to-value"
file_id: "99"
topic: "30 cross-module trust, governance, product, economics and leadership issues"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-08"
reviewed: null
review_trigger: "Changed authority, user population, provider contract, release bundle or measurement denominator"
notation_style: "plain-text"
prerequisites: ["../49_ai_security_privacy_safety_and_red_teaming/INDEX.md", "../50_responsible_ai_governance_compliance_and_audits/INDEX.md", "../51_ai_product_management_ux_and_human_ai_collaboration/INDEX.md", "../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md", "../53_communication_leadership_and_ai_transformation/INDEX.md"]
pattern_tags: ["eligibility-before-value", "evidence-bound-decision", "recoverable-delegation"]
source_status: "DERIVED / ILLUSTRATIVE; original source bodies unavailable"
owner_role: "Cross-functional AI release owner"
source_index: "INDEX.md"
---
# Thirty cross-module Common Issues: when trust, value and execution disagree

## 1. File Header

**Purpose:** Diagnose a failed handoff between security, governance, product, economics and leadership rather than optimizing one team's metric in isolation.
**Outcome:** Given a disputed release or investment decision, construct competing explanations, choose a discriminating check, contain safely and define a falsifiable correction.
**Scope:** Thirty distinct cross-module issues, numbered P09-001 through P09-030, with complete diagnostic records in Section 8.
**Provenance:** Every issue is HYPOTHETICAL; none is represented as an observed company incident or a reproduced implementation failure.
**Prerequisites:** Read the five [module homes](../INDEX.md#modules) in index order; know fractions, units, conditional decisions and basic access-control terminology.
**Inventory:** Consulted [directory.txt](../../../Templates/directory.txt), especially execution/approval surfaces, security/governance, practical ML governance and visible safety-training titles.
**Source crosswalk:** [Exact inventory accounting](../../00_SOURCE_TOPIC_CROSSWALK.md) separates visible technical commitments from unavailable original contents.
**Focused briefs:** [Approval/trust boundaries](../../00_SOURCE_COVERAGE_AGENTS.md#agent-trust-approvals) and [safety/alignment distinctions](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-safety).
**Template basis:** [Course numerical standard](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md); this file preserves all twelve sheet sections and places its card last.
**Not covered:** Penetration-test execution, legal applicability determinations, clinical safety certification and current provider prices require specialized evidence outside this diagnostic guide.
**Retrieval cue:** If the headline score improves but someone cannot authorize, reproduce, afford or reverse the decision, start with the Section 2 boundary map.
**Evidence policy:** A hypothetical signal is an expected diagnostic result, not a log that was collected.
**Action policy:** Proposed checks below are acceptance oracles; no check is claimed executed.
**Review status:** Authoring draft; independent technical, numerical and editorial publication review remains unrecorded.
**Audience:** Product owners, security engineers, governance leads, procurement teams and executives sharing responsibility for one AI workflow.
**Decision boundary:** A release recommendation is not permission to process data, spend funds or take an external action.
**Operational boundary:** A successful request can still be an unsuccessful task if the user must repeat work elsewhere.
**Statistical boundary:** A zero count in a finite test does not establish a zero failure probability.
**Economic boundary:** Labor time released is not automatically an avoidable cash expense.
**Leadership boundary:** Naming an owner is insufficient unless that role can stop the relevant action and obtain repair resources.
**Takeaway:** Diagnose the boundary where evidence ceases to justify authority, value or recovery.

## 2. Concept Map / Mental Model

```text
[User need + affected people]
             |
             v
[Authority / data / purpose eligibility] --fails--> [Deny or redesign]
             |
             v
[Versioned evidence + uncertainty] --missing--> [Collect or abstain]
             |
             v
[Useful task + human workload] --overload--> [Reduce scope / queue safely]
             |
             v
[Full cost + counterfactual value] --negative--> [Baseline / defer]
             |
             v
[Accountable approval + reversible rollout]
             |
             v
[Measured outcomes + adverse effects] --> [Reassess / stop / retire]
```

The arrows describe a proposed decision workflow, not a causal model showing that each stage guarantees the next.
Eligibility is evaluated before utility ranking because an impermissible action is not made permissible by a larger projected benefit.
The workflow carries a common release identity so that different teams are not approving different versions under one familiar product name.
Recovery is a separate path because revoking future access does not reverse a message already sent or data already exported.
A dashboard is an observation surface; it is not the complete state of the underlying business process.
The same symptom may cross several boundaries, so start by locating the earliest point where the claimed invariant no longer holds.
An apparent cost overrun may originate in low task success, retries, review work, contract minima or a changed denominator.
An apparent safety gain may originate in blanket refusal, a less demanding test set or a monitor that ignores tool effects.
An apparent adoption failure may originate in poor product value, inaccessible interaction, delayed provisioning or fear of punishment.
An apparent governance delay may originate in a genuinely missing control, a reviewer bottleneck or ambiguous decision authority.
An apparent transformation success may originate in reclassification of ordinary work rather than a real change in outcomes.

| Symptom family | First boundary to inspect | Competing mechanism | Diagnostic artifact |
|---|---|---|---|
| Great score, blocked release | Eligibility | Genuine hard constraint versus missing evidence | Requirement-to-control map |
| Safe-looking text, harmful result | Effect boundary | Model refusal versus external action | Tool effect ledger |
| Busy reviewers, slow service | Human capacity | Excess referrals versus slow handling | Arrival/service trace |
| Cheap calls, expensive tasks | Outcome denominator | Retry amplification versus low useful completion | Task-linked bill ledger |
| High usage, weak value | Counterfactual | Novelty versus useful substitution | Eligible task cohort |
| Approved, then inconsistent | Version boundary | Mixed release versus population shift | Signed release manifest |
| Fast growth, no accountability | Authority | Ownership gap versus missing resources | Decision-rights register |
| Stopped service, continuing harm | Recovery | In-flight effects versus stale credentials | Action reconciliation log |

A useful diagnosis names what result would disprove the favorite explanation.
A useful mitigation reduces current exposure without pretending to establish the missing causal explanation.
A useful correction changes the mechanism rather than merely suppressing its symptom.
A useful verification includes an adversarial or boundary case that failed under the old mechanism.
A useful prevention step has an owner, trigger and observable signal.
A useful escalation asks a role with the actual authority needed for the disputed decision.
**Takeaway:** Follow the request, money, evidence and authority together; optimizing only one flow hides the failure.

## 3. Foundational Definitions

| Term / symbol | Definition | Units / type | Valid range | Nearby nonexample |
|---|---|---|---|---|
| Eligible task | A task meeting stated user, purpose, data and time rules before treatment | Boolean per task | True or false | A task retained because it succeeded |
| Useful completion, U | A completed task meeting the predeclared user-outcome rubric | Count of tasks | Nonnegative integer | A model response returned successfully |
| Attempt, A | One invocation or execution attempt linked to a task | Count of attempts | Nonnegative integer | A unique business task |
| Total cost, C | All costs inside the declared accounting boundary and window | Illustrative USD | Nonnegative scalar here | Provider inference bill alone |
| Review arrival, lambda | Referrals entering the human queue per unit time | Cases/hour | Nonnegative | Number of users provisioned |
| Review service, mu | Completed reviews per unit time at the assumed mix | Cases/hour | Nonnegative | Nominal reviewers employed |
| Backlog, B | Arrived but not completed review work | Cases | Nonnegative | Sum of completed reviews |
| Harm event | A defined adverse effect in the chosen threat/workflow boundary | Boolean per exposed unit | Definition-specific | A negative sentiment label alone |
| Release identity | Immutable bundle identifier for model, prompt, tools, policies and data interfaces | Structured identifier | Exact match required | A marketing product name |
| Counterfactual | Outcome under the alternative decision for the same target population | Potential outcome | Generally not jointly observed | The last month's outcome automatically |
| Authority | Permission to decide or act for a bounded identity, scope and purpose | Policy relation | Context-dependent | Ability to generate convincing text |
| Residual risk | Risk remaining after specified controls and their limitations | Structured assessment | Not inherently a probability | A compliance checklist score |

A task ID groups attempts, reviews and eventual output; it is not a credential and must not authorize an action.
A release ID identifies a bundle; it is not sufficient evidence that every deployed component actually matches the bundle.
A recorded timestamp states when the system learned something; an effective timestamp states when the fact or rule applies.
A policy expiration terminates a grant under its specified semantics; it does not necessarily delete historical records.
A deletion request identifies data to remove; the scope must account for replicas, derived indexes and contractual exceptions.
A guardrail is an outcome or constraint that can disqualify a release despite improvement in the primary objective.
A proxy is useful only to the extent its relationship to the desired outcome survives the proposed intervention.
A sampled audit measures the selected sample; extrapolation requires a defensible sampling frame and assumptions.
A confidence interval summarizes a repeated-sampling procedure under its model; it is not a guarantee against an adaptive attacker.
An accountable role accepts or rejects a decision; responsible implementers carry out the work and need not own residual risk.
A stop authority can disable new work; a repair authority can restore or compensate for already committed effects.
An avoided cost changes a payable or committed resource requirement; an allocated cost is an accounting attribution.
A unit price multiplies a billable quantity under a contract; a useful-task cost divides complete cost by useful completions.
A positive net estimate does not override hard authorization, safety or legal constraints.
**Takeaway:** Define the unit, population, version and authority before interpreting a metric.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

**Validity first:** The same fixed window and accounting boundary must be used in cost and useful-task counts.
Start with the definition of an average: allocate the total included cost equally across the counted useful completions.
This allocation does not claim every task consumed the same resources.
It becomes undefined when no task meets the usefulness rubric.

```text
FORMULA [P09-COST]: c_useful = C / U
WHERE: c_useful [USD/useful task]; C total included cost [USD];
       U useful completions [tasks] within the same population/window.
CONSTRAINT: U > 0; consistent accounting boundary and task definition.
NUMERICAL EXAMPLE: C=12000 USD, U=3000 tasks;
                   c_useful=12000/3000=4 USD/useful task.
INTERPRETATION: compare against a baseline measured on the same tasks.
```

A plausible wrong calculation divides cost by all attempts, making repeated failures look like productive volume.
With 6,000 attempts, that wrong denominator produces 2 USD/attempt, a different metric rather than a cheaper useful task.
Independent dimensional check: USD divided by tasks is USD/task; multiplying 4 by 3,000 reconstructs 12,000 USD.
Boundary: doubling attempts without changing useful outcomes leaves c_useful unchanged if cost is unchanged.
If the extra attempts increase cost, c_useful rises even while the call-level success dashboard improves.

### 4.2 General Case

**Validity first:** Cost categories must be mutually exclusive within the accounting boundary to avoid double counting.
Partition total included cost into invocation, human-review, platform and incremental incident/remediation costs.
Each category is summed before dividing by the same useful-completion count.
Shared fixed cost may be allocated, but the allocation rule must be declared separately from marginal cost.

```text
FORMULA [P09-FULL]: C = C_call + C_review + C_platform + C_remediation
WHERE: each C_* is included cost [USD] for the declared window.
CONSTRAINT: no duplicate charges across categories; taxes and excluded items stated.
NUMERICAL EXAMPLE: C_call=2400; C_review=6000; C_platform=2600;
                   C_remediation=1000; C=12000 USD.
INTERPRETATION: the review workflow can dominate a low inference bill.
```

Independent check: 2,400 + 6,000 = 8,400; 2,600 + 1,000 = 3,600; 8,400 + 3,600 = 12,000 USD.
The formula is an accounting identity once the partition is defined, not an empirical estimate of a provider's charges.
A missing category cannot be set to zero because no invoice has arrived; record it as unpriced or outside scope.
Sensitivity: increasing review cost by 50% adds 3,000 USD and raises cost per useful task to 5 USD if U stays 3,000.
This sensitivity holds task mix and useful outcomes fixed; a changed review process can change both.

**Validity first:** A constant-rate, no-loss review flow can be analyzed by conservation of cases.
Arrivals add cases and completed reviews remove cases; integrate their difference over the stated interval.
The fluid model ignores within-hour burstiness, priority classes and reviewer setup time.

```text
FORMULA [P09-QUEUE]: B_end = max(0, B_start + (lambda-mu)*t)
WHERE: B_start and B_end [cases]; lambda and mu [cases/hour]; t [hours].
CONSTRAINT: constant rates, work-conserving service, no loss, adequate buffer;
            when the queue empties, service cannot create negative backlog.
NUMERICAL EXAMPLE: B_start=0; lambda=90; mu=72; t=8;
                   B_end=(90-72)*8=144 cases.
INTERPRETATION: a review-required policy is infeasible without adequate service.
```

The maximum-with-zero expression is valid here because constant rates prevent a later reversal within the interval.
For time-varying rates, update short intervals with reflection at zero; do not subtract hypothetical idle service from future backlog.
Independent check: 720 arrivals minus 576 completions equals 144 remaining cases.
Boundary: when lambda equals mu, an existing positive backlog persists in this fluid model rather than drains.
Sensitivity: 96 cases/hour service against 90 cases/hour arrivals removes only six backlog cases/hour.
Draining 144 cases then requires 24 hours, assuming that spare rate persists and priorities do not change.

### 4.3 Key Special Cases / Variants

**Validity first:** The following bound assumes independent Bernoulli trials with one fixed failure probability and reliable labels.
For n trials with zero failures, the chance of that observation at failure probability p is (1-p)^n.
Set this probability to alpha, solve for p, and obtain an exact one-sided upper endpoint for the zero-failure case.
The model is inappropriate for adaptive attacks with changing strategies or strongly clustered observations.

```text
FORMULA [P09-ZERO]: p_upper = 1 - alpha^(1/n)
WHERE: p_upper failure probability [dimensionless]; alpha tail probability;
       n independent trials [count].
CONSTRAINT: n>0; 0<alpha<1; zero observed failures; fixed Bernoulli p.
NUMERICAL EXAMPLE: alpha=0.05; n=100;
                   p_upper=1-0.05^0.01, approximately 0.029513.
INTERPRETATION: zero failures in 100 trials does not establish zero risk.
```

The endpoint is approximately 2.9513%, not a claim that the actual rate equals that value.
Independent reconstruction: (1-p_upper)^100 = 0.05 by substitution into the defining equation.
Boundary: with n=1, the upper endpoint is 0.95; one passing trial provides little rate information.
Sensitivity: at n=1,000 the corresponding endpoint is approximately 0.299125% under the same assumptions.
Reporting many prompts from the same template as independent can materially overstate evidence.
Neither endpoint estimates the success probability of an adversary outside the test distribution.

| Variant | What changes | Valid use | Avoid condition |
|---|---|---|---|
| Marginal economics | Include only costs changed by the decision | Short-run routing choice | Ignoring committed capacity for investment decisions |
| Full allocation | Include declared shared-cost allocation | Portfolio affordability | Treating allocation as avoidable cash |
| Capacity gate | Compare referral flow to service by class | Human review planning | Treating average balance as deadline proof |
| Zero-failure bound | Quantify one narrow sampling uncertainty | Fixed independent test suite | Adaptive or correlated attacks |
| Exact-action authorization | Evaluate one effect-bearing payload | Write-capable workflow | Assuming an earlier text approval covers changed effects |

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

A validity constraint is checked as a predicate, not blended into a weighted score that a high benefit can offset.
Input contract: request identity, tenant, purpose, action payload, destinations, policy version and freshness context are available.
State: candidate, eligible, evidence-ready, approved, executing, committed, reconciled, rejected or revoked.
Transition: candidate becomes eligible only if every mandatory predicate holds for that exact request context.
Invariant: no committed external effect is authorized solely by the model's assertion that permission exists.
Justification: the effecting service independently evaluates identity and policy immediately before the protected action.
Worked trace: candidate export names twelve accounts; allowed set has five; eligibility intersection contains five accounts.
If approval was for a twelve-account payload, the five-account revision is a different payload and requires the documented approval semantics.
A policy can deliberately permit safe subset execution, but that rule must exist before the request rather than be inferred afterward.
Expiration trace: approval expires at 14:00; an attempt at 14:01 cannot use it even if the text was generated at 13:59.
Termination: rejection records a reason and returns a bounded error; it does not loop until some provider complies.
Recovery: committed effects are reconciled separately; revocation prevents future action but may not undo prior disclosure.
Failure boundary: a compromised enforcement service or misbound identity invalidates this local argument.
An application can satisfy this invariant while still producing wrong or harmful authorized decisions; authorization is not correctness.
A utility score of 0.99 cannot compensate for a failed tenant-isolation predicate.
As useful completions approach zero, c_useful grows without bound for positive fixed C; a call-price comparison becomes less meaningful.
As review arrivals approach service from below, spare capacity vanishes; small shocks can keep a queue from recovering.
As test counts grow, a fixed-distribution confidence bound narrows; coverage of untested harms does not automatically broaden.
**Takeaway:** Use identities, conservation and conditional bounds only within their stated validity domains.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE cross-module decision exercise; all inputs are constructed and no organizational deployment is implied.
**Workflow:** A proposed analyst assistant drafts case summaries, escalating selected cases to human reviewers before completion.
**Decision:** Continue a bounded pilot, expand to all eligible tasks, or retain the manual baseline.
**Primary unit:** One eligible analyst task, not an invocation, employee login or generated paragraph.
**Observation window:** An illustrative eight-hour working day for capacity; a separate illustrative month for costs.
**Separation rule:** Daily queue inputs must not be mixed directly with monthly useful-outcome counts.

| Input | Value | Units | Meaning |
|---|---:|---|---|
| Monthly invocation cost | 2400 | USD/month | All attempts, including failed attempts |
| Monthly review labor | 6000 | USD/month | Included review work only |
| Monthly platform allocation | 2600 | USD/month | Declared full-cost allocation |
| Monthly remediation labor | 1000 | USD/month | Included incremental correction work |
| Monthly useful tasks | 3000 | tasks/month | Meets predeclared task rubric |
| Monthly attempts | 6000 | attempts/month | Task-linked attempts, not useful outputs |
| Daily referral arrival rate | 90 | cases/hour | Constant illustrative daytime arrivals |
| Daily service capacity | 72 | cases/hour | Completed comparable reviews |
| Overload interval | 8 | hours | Starts with an empty queue |
| Fixed safety trials | 100 | trials | Independent Bernoulli model assumed only for arithmetic |
| Safety failures observed | 0 | failures | Constructed test outcome, not an executed test |

Step 1: Sum the four monthly costs using P09-FULL to obtain 12,000 USD/month.
Step 2: Divide by 3,000 useful tasks/month using P09-COST to obtain 4 USD/useful task.
Step 3: Report 2 USD/attempt separately; do not use it as the useful-task cost.
Step 4: Subtract 72 from 90 to get 18 cases/hour of review backlog growth.
Step 5: Multiply by eight hours using P09-QUEUE to obtain 144 cases left at close of day.
Step 6: Report safety test evidence under P09-ZERO as an upper endpoint near 2.9513%, conditional on the trial model.
Step 7: Check the hard boundary: no external write or export is eligible merely because the cost is attractive.
Step 8: Recommend against unqualified expansion because the assumed human workflow accumulates work even before burstiness is considered.
Step 9: Separate that recommendation from empirical proof; the constructed workload establishes only the arithmetic consequence of the assumptions.
Step 10: Assign the product owner to measure referral mix and reviewer handling distributions before choosing a revised pilot envelope.

Independent cost check: the four costs sum in two pairs to 8,400 + 3,600 = 12,000 USD.
Independent cost check: 3,000 useful tasks times 4 USD/task reconstructs the declared total.
Independent queue check: eight hours produces 720 referrals and 576 completions, leaving 144 cases.
Independent queue check: 18 cases/hour times eight hours produces the same result.
Independent probability check: raising 0.05^(1/100) to the 100th power recovers 0.05.
Rounding policy: retain full precision for any programmatic probability calculation; round the reported percentage to four decimal places.
Wrong path: reporting 100% safety from zero observed failures confuses a finite sample with a population guarantee.
Wrong path: reporting 2 USD/useful task silently replaces useful tasks with attempts.
Wrong path: adding another dashboard does not increase service capacity or reduce referrals.
Wrong path: doubling review utilization beyond feasible hours does not create real staffed capacity.

### Sensitivity that changes the decision

If service rises from 72 to 96 cases/hour while arrivals remain 90, a 144-case backlog drains in 24 hours.
That result is a fluid estimate; it says nothing about the oldest case's deadline under priority scheduling.
If referral arrivals fall to 60 cases/hour with service at 72, spare capacity is twelve cases/hour.
At that revised mix, 144 queued cases drain in twelve hours if the same handling cost and service rate remain valid.
If review labor rises 50% with useful outcomes fixed, cost per useful task rises from 4 to 5 USD.
If useful outcomes fall to 2,400 with total cost fixed at 12,000 USD, cost becomes 5 USD/useful task.
Those two sensitivities can happen together; then cost is 15,000/2,400 = 6.25 USD/useful task.
Independent joint check: 2,400 tasks multiplied by 6.25 USD/task equals 15,000 USD.
If policy prohibits the chosen data destination, no positive business estimate makes that destination an eligible option.
If tests are clustered by template, the simple zero-failure endpoint no longer describes the full dependence correctly.
The next measurement should target the parameter capable of reversing the decision, not the easiest metric to collect.
A pilot can collect valuable evidence without being financially positive in its first window, but the learning purpose must be explicit.
A release owner should distinguish an evidence-gathering pilot from a broad rollout presented as already beneficial.
**Takeaway:** One apparently cheap assistant can fail capacity, evidence and authority gates simultaneously.

## 6. Visual / Geometric Intuition

```text
Backlog [cases], schematic constant-rate overload
144 |                             * end of 8-hour day
    |                          /
 72 |              *          /   slope = 18 cases/hour
    |           /            /
  0 |*---------+------------+--------------------> time [hours]
     0         4            8

At a later 96-case/hour service rate and 90-case/hour arrival rate:
144 |* recovery begins
    | \
 72 |             * after 12 recovery hours
    |              \
  0 |---------------------------*---------------> recovery time [hours]
     0                         24
```

Both drawings are schematic, not measured latency curves and not to scale across the two panels.
The key points follow the case-conservation calculation, not a queueing-distribution fit.
The first slope is positive eighteen cases/hour; the second slope is negative six cases/hour while backlog remains positive.
A steeper recovery slope requires more genuine spare capacity or fewer arriving cases.
The area under the curve has units case-hours and relates to accumulated waiting work, but no additional formula is needed here.
A flat backlog curve at positive height can hide unacceptable waiting time even though the growth alarm is quiet.
A stable count can also conceal a changing mix of difficult cases; monitor age and service classes separately.

```text
Feasible action set, schematic set relationship
[All technically callable actions]
  [Actions allowed for this identity and purpose]
    [Actions supported by current evidence]
      [Actions affordable and operationally recoverable]
        * selected action
```

Nesting is a teaching picture for sequential filters, not a claim that evidence quality is naturally ordered as a scalar.
A cheaper callable action can lie outside the allowed subset and must remain excluded.
A review-ready action can leave the feasible set if the approval expires before execution.
A formerly useful action can become uneconomic if useful completions decline or contract terms change.
The selected action need not maximize a score over the outer set; it must satisfy the inner decision contract.
**Takeaway:** Watch slopes, set membership and state transitions; attractive averages do not repair invalid boundaries.

## 7. Comparison / Selection Table

| Diagnostic approach | Preconditions | What it can establish | Cost / overhead | Use when | Avoid when |
|---|---|---|---|---|---|
| Reconcile identifiers | Task/release/effect IDs retained lawfully | Whether teams describe the same objects | Join and storage work | Metrics contradict across teams | IDs can be forged or fields expose unnecessary personal data |
| Stratified sample review | Defined population and rubric | Sampled error mechanisms and slice differences | Reviewer labor | Aggregate quality hides a subgroup problem | Convenience sample is presented as population rate |
| Synthetic permission test | Controlled identities and safe fixtures | Enforcement behavior for tested boundaries | Fixture maintenance | Authority or tenant leakage suspected | Production secrets or real harmful effects are required |
| Task-linked cost ledger | Billable units and outcome labels reconcile | Included cost per defined useful task | Accounting integration | Cheap calls coexist with expensive operations | Unpriced work is silently treated as free |
| Queue conservation trace | Arrival/completion timestamps and classes | Backlog balance and flow mismatch | Low-volume event logging | Human review deadlines fail | Fluid averages are used as tail guarantees |
| Controlled rollout comparison | Valid assignment, interference handling and mature outcomes | Causal effect under the design assumptions | Exposure and analytical cost | Product value is uncertain after eligibility | Unsafe exposure is required to learn a prohibited harm |

Start with immediate containment if there is a credible unauthorized effect; do not wait for a definitive economic diagnosis.
If identity or release IDs disagree, reconcile them before comparing scores or invoices.
If the claimed improvement depends on excluding tasks, reconstruct the denominator using rules fixed before outcome observation.
If a capacity equation predicts accumulation, reduce the intake envelope or add measured service before expanding coverage.
If no hard boundary fails and the remaining uncertainty concerns incremental value, choose a bounded evaluation capable of reversing the investment decision.
If evidence collection itself exposes sensitive data, minimize or aggregate the evidence and arrange authorized access for needed detail.
When priorities conflict, authority and safety constraints are resolved first; value comparisons operate inside that admissible region.
A temporary manual baseline is a credible option when automation cannot satisfy the action contract.
A human reviewer is not a free or infallible fallback; evaluate workload, information and decision authority.
A vendor assertion is a candidate evidence item, not a substitute for task-specific eligibility and acceptance tests.
Escalate unresolved legal applicability to the authorized legal owner with dated source context rather than asking a model for definitive permission.
Escalate irreversible effects to incident response and repair ownership, not merely the product backlog.
**Takeaway:** Choose the least intrusive check that distinguishes the consequential hypotheses and respects the same boundaries as the product.

## 8. Failure Modes & Edge Cases

### Issue index

| ID | Category | Symptom | Root-cause hypothesis | Diagnostic / expected signal | Fix | Provenance | Related home |
|---|---|---|---|---|---|---|---|
| [P09-001](#p09-001) | CONCEPT_CONFUSION | High score overrides prohibition | Hard constraint treated as weight | Inspect score inputs; prohibited option remains ranked | Filter before scoring | HYPOTHETICAL | Module 49/52 |
| [P09-002](#p09-002) | VIOLATED_ASSUMPTION | Approved request changes effect | Approval binds text, not payload | Compare canonical payload digests | Bind exact action context | HYPOTHETICAL | Module 49/50 |
| [P09-003](#p09-003) | VIOLATED_ASSUMPTION | Tenant sees wrong specialization | Cache identity omits tenant/base | Cross-tenant interleaving trace | Tenant-scoped release keys | HYPOTHETICAL | Module 49/51 |
| [P09-004](#p09-004) | CONCEPT_CONFUSION | Safe response accompanies unsafe action | Text-only safety metric | Compare tool ledger to response label | Evaluate effects | HYPOTHETICAL | Module 49/50 |
| [P09-005](#p09-005) | VIOLATED_ASSUMPTION | Wrong rule cited confidently | Eligibility metadata stale | Boundary-date replay | Date/jurisdiction eligibility | HYPOTHETICAL | Module 50/51 |
| [P09-006](#p09-006) | VIOLATED_ASSUMPTION | Withdrawn data persists | Retention scope misses derivatives | Trace deletion lineage | Purpose-aware deletion workflow | HYPOTHETICAL | Module 49/50 |
| [P09-007](#p09-007) | VIOLATED_ASSUMPTION | Audit cannot reconstruct decision | Mutable evidence references | Resolve historical manifest | Immutable bounded evidence | HYPOTHETICAL | Module 50/53 |
| [P09-008](#p09-008) | NUMERICAL_EDGE | Success jumps after filtering | Outcome-dependent denominator | Rebuild eligible cohort | Freeze inclusion rules | HYPOTHETICAL | Module 50/51 |
| [P09-009](#p09-009) | CONCEPT_CONFUSION | Safety gains destroy usefulness | Blanket refusal | Benign/prohibited confusion matrix | Joint safety/utility gates | HYPOTHETICAL | Module 49/51 |
| [P09-010](#p09-010) | VIOLATED_ASSUMPTION | Confidence misleads new users | Calibration population changes | Reliability by language/task | Recalibrate or abstain | HYPOTHETICAL | Module 50/51 |
| [P09-011](#p09-011) | SCALE_FAILURE | Review queue misses deadlines | Referrals exceed service | Case conservation and age | Intake/capacity redesign | HYPOTHETICAL | Module 51/52 |
| [P09-012](#p09-012) | VIOLATED_ASSUMPTION | Human approval rarely catches errors | Reviewer lacks information or time | Blinded seeded-error review | Decision-support redesign | HYPOTHETICAL | Module 50/51 |
| [P09-013](#p09-013) | VIOLATED_ASSUMPTION | Pilot benefit disappears at scale | Volunteer selection | Compare eligible pilot/nonpilot mix | Representative evaluation | HYPOTHETICAL | Module 51/53 |
| [P09-014](#p09-014) | CONCEPT_CONFUSION | Local time falls, total work rises | Work exported downstream | End-to-end task timing | Include all affected labor | HYPOTHETICAL | Module 51/52 |
| [P09-015](#p09-015) | CONCEPT_CONFUSION | Savings miss finance plan | Time valued as cash | Reconcile payroll and capacity | Separate capacity/cash value | HYPOTHETICAL | Module 52/53 |
| [P09-016](#p09-016) | NUMERICAL_EDGE | Unit economics collapse | Attempts omitted from cost | Task-linked retry bills | Complete cost numerator | HYPOTHETICAL | Module 49/52 |
| [P09-017](#p09-017) | VIOLATED_ASSUMPTION | Failover crosses residency boundary | Eligibility not reapplied | Fault replay with route audit | Constrained fallback set | HYPOTHETICAL | Module 49/52 |
| [P09-018](#p09-018) | VIOLATED_ASSUMPTION | Vendor benchmark winner disappoints | Test/tuning contamination | Sealed task replay | Predeclared evaluation | HYPOTHETICAL | Module 50/52 |
| [P09-019](#p09-019) | NUMERICAL_EDGE | Cheap contract becomes expensive | Commitment cliffs ignored | Volume sensitivity at tiers | Full contract model | HYPOTHETICAL | Module 52/53 |
| [P09-020](#p09-020) | VIOLATED_ASSUMPTION | Multi-vendor plan fails together | Shared dependency | Dependency fault tree | Independent recovery evidence | HYPOTHETICAL | Module 49/52 |
| [P09-021](#p09-021) | CONCEPT_CONFUSION | Portfolio value exceeds total opportunity | Duplicate benefit attribution | Trace shared task population | Reconcile benefit ownership | HYPOTHETICAL | Module 52/53 |
| [P09-022](#p09-022) | SCALE_FAILURE | Central AI team blocks delivery | Unbounded dependency queue | Arrival/service by work class | Explicit platform contract | HYPOTHETICAL | Module 51/53 |
| [P09-023](#p09-023) | VIOLATED_ASSUMPTION | Everyone approves, nobody repairs | Authority/resource gap | Incident tabletop | Assign repair authority | HYPOTHETICAL | Module 50/53 |
| [P09-024](#p09-024) | CONCEPT_CONFUSION | Training complete, skills absent | Attendance substitutes for competence | Task-based assessment | Practice and capability evidence | HYPOTHETICAL | Module 51/53 |
| [P09-025](#p09-025) | NUMERICAL_EDGE | Board trusts precise fragile ROI | Point estimates hide switching values | Recompute threshold ranges | Communicate decision sensitivity | HYPOTHETICAL | Module 52/53 |
| [P09-026](#p09-026) | VIOLATED_ASSUMPTION | Rollback fails policy consistency | Mixed artifact versions | Replay old/new bundle pairs | Atomic compatibility manifest | HYPOTHETICAL | Module 49/50 |
| [P09-027](#p09-027) | SCALE_FAILURE | Stop command leaves effects running | No in-flight reconciliation | Pending/committed effect trace | Stop plus compensate | HYPOTHETICAL | Module 49/53 |
| [P09-028](#p09-028) | VIOLATED_ASSUMPTION | Few complaints hide growing harm | Reporting incentives suppress evidence | Independent anonymous sampling | Protected reporting channel | HYPOTHETICAL | Module 50/53 |
| [P09-029](#p09-029) | VIOLATED_ASSUMPTION | Adoption hides excluded users | Access barriers remove denominator | Eligible-user accessibility study | Inclusive workflow alternatives | HYPOTHETICAL | Module 51/53 |
| [P09-030](#p09-030) | SCALE_FAILURE | Retired product retains exposure | Exit omits data/credentials | Asset and grant reconciliation | End-to-end retirement | HYPOTHETICAL | Module 49/52/53 |

<a id="p09-001"></a>
### P09-001 — A weighted score buys its way through a prohibition
Category: CONCEPT_CONFUSION; subsystem: investment-to-authorization handoff.
Provenance: HYPOTHETICAL; derived counterexample to replacing a hard predicate with a finite score penalty.
Environment: Procurement ranks candidate model routes using quality, latency, cost and a security score.
Trigger: An ineligible destination receives enough quality points to beat an eligible destination.
Symptom: The selected route has the highest spreadsheet score but violates the declared destination rule.
Decision at risk: Whether the product may use that route at all, before asking whether it is attractive.
Hypothesis A: The rule is genuinely hard, but implementation encodes it as a weighted feature.
Hypothesis B: The score is correct, but the requirement owner has changed the authorized scope without updating the register.
Hypothesis C: The selected route is mislabeled and actually targets a different deployment boundary.
Discriminating input: The exact requirement version, route configuration and independently resolved destination identity.
Expected signal: A route known to fail the predicate remains in the candidate set presented to the ranking function.
Negative result: If the route passes the current predicate, investigate metadata currency rather than blaming weights.
Safe diagnostic: Replay synthetic route records without sending user content to any candidate provider.
Immediate containment: Remove the disputed route from the eligible set while authority is clarified.
Mitigation risk: Overbroad removal can interrupt legitimate service and should return a clear unavailable response.
Correction: Evaluate every mandatory predicate before scoring and retain the exclusion reason.
Prevention: The requirement owner approves predicate changes; the routing owner implements versioned policy evaluation.
Verification oracle: Give an ineligible route arbitrarily high utility; the selected output must never be that route.
Boundary test: An empty eligible set must lead to abstention or an explicitly authorized manual path.
Recovery: Reconcile historical exposures if the route was actually used; changing weights does not reverse disclosure.
Owner: Security policy owner for eligibility; procurement owner for ranking among eligible choices.
Escalation: Legal or data owner when the requirement's applicability or exception authority is disputed.
Evidence gap: No real route, contract or exposure record is provided in this teaching example.
Cost consequence: Excluding a cheap route can raise projected cost; that is a tradeoff inside the valid contract.
Decision consequence: A positive projected return is not a permission grant.
Related sheet: [Module 52 economics and procurement](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [NIST SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) supports explicit security practices, not this scenario's rule.

<a id="p09-002"></a>
### P09-002 — The approved text and executed payload are different objects
Category: VIOLATED_ASSUMPTION; subsystem: approval-to-effect binding.
Provenance: HYPOTHETICAL; motivated by the visible approval-gate inventory and its exact-action learning outcome.
Environment: An assistant drafts an export request and a human approves a rendered preview.
Trigger: A later transformation changes recipients, records, amount or destination before execution.
Symptom: Logs contain an approval, yet the committed action differs from what the approver saw.
Decision at risk: Whether the recorded approval authorizes the exact effect-bearing payload.
Hypothesis A: Approval is bound to a conversation ID rather than the canonical action representation.
Hypothesis B: The preview omits fields that were already present and material to the action.
Hypothesis C: A timezone or expiration mismatch invalidates an otherwise matching payload.
Discriminating input: Approved canonical payload, execution payload, policy version, identity and expiration context.
Expected signal: One material field differs while the execution path still accepts the earlier approval token.
Negative result: If payloads and time match, inspect approver authority and policy scope instead of assuming mutation.
Safe diagnostic: Use inert export fixtures and synthetic destinations so the comparison cannot leak real data.
Immediate containment: Suspend disputed execution and require a fresh approval for the exact current payload.
Mitigation risk: Reapproval can create duplicate work if an earlier effect already committed; reconcile first.
Correction: Bind approval to action semantics, destination, tenant, scope, version and validity interval.
Prevention: Render the same canonical representation that the effecting service verifies.
Verification oracle: Mutating a material destination or record set must invalidate the approval.
Boundary test: At expiration, execution follows the documented inclusive/exclusive time rule consistently.
Recovery: Reconcile committed actions by idempotency/effect key before any retry or compensation.
Owner: Tool-service owner implements enforcement; policy owner defines which mutations require approval.
Escalation: Incident response when an unintended destination may already have received data.
Evidence gap: This file contains no production approval token, payload digest or observed mutation trace.
Cost consequence: Exact binding adds engineering and review work but prevents approval evidence from becoming misleading.
Decision consequence: An approval record proves little until its object and authority are defined.
Related sheet: [Module 49 security and red teaming](../49_ai_security_privacy_safety_and_red_teaming/INDEX.md).
Source support: [Approval focused brief](../../00_SOURCE_COVERAGE_AGENTS.md#agent-trust-approvals), not inaccessible original code.

<a id="p09-003"></a>
### P09-003 — Tenant customization becomes a cross-tenant cache race
Category: VIOLATED_ASSUMPTION; subsystem: product personalization-to-isolation handoff.
Provenance: HYPOTHETICAL; the tenant-adapter interview brief is reference material, not deployment evidence.
Environment: A serving layer loads specialized adapters for several tenant assistants.
Trigger: Two tenants request adapters with the same short display name or incompatible base versions.
Symptom: A tenant receives another tenant's specialization or an unexplained regression during adapter swaps.
Decision at risk: Whether customization can be served without crossing tenant and compatibility boundaries.
Hypothesis A: The cache key omits tenant identity or base checkpoint identity.
Hypothesis B: Session state is not pinned and a concurrent swap changes the effective adapter mid-request.
Hypothesis C: Output contamination originates in retrieval permissions rather than adapter selection.
Discriminating input: Tenant-scoped request trace, adapter/base digest, cache key and retrieval authorization log.
Expected signal: Interleaving synthetic tenant requests changes the selected adapter despite identical authorized input.
Negative result: Stable adapter selection does not rule out cross-tenant retrieval or shared conversation memory.
Safe diagnostic: Use synthetic canary facts that belong to no real customer and cannot disclose customer content.
Immediate containment: Disable affected customization or pin each session to a known isolated release bundle.
Mitigation risk: Removing adapters can reduce domain-task usefulness; communicate the fallback rather than hiding it.
Correction: Scope cache/session identity by tenant, compatible base, adapter version and relevant policy context.
Prevention: Treat the whole tenant release bundle as an immutable selection unit.
Verification oracle: Cross-tenant negative tests must not alter selected bundle identity or reveal tenant canary content.
Boundary test: A missing or incompatible adapter returns an explicit authorized fallback or failure, never another tenant's adapter.
Recovery: Invalidate affected cache entries and investigate past outputs by release identity within lawful retention limits.
Owner: Serving-platform owner for cache/session semantics; tenant security owner for isolation acceptance.
Escalation: Data incident response if any actual tenant content may have crossed boundaries.
Evidence gap: Original tenant counts, service objectives and implementation details remain unspecified.
Cost consequence: Pinning and isolation can reduce cache reuse; price the actual workload rather than assuming free customization.
Decision consequence: An adapter changes behavior, not authorization, and cannot establish tenant isolation by itself.
Related sheet: [Module 49 security home](../49_ai_security_privacy_safety_and_red_teaming/INDEX.md).
Source support: [Interview-reference plan](../../00_REFERENCE_INFORMED_CASE_PLAN.md) defines a supporting curriculum commitment only.

<a id="p09-004"></a>
### P09-004 — A polite refusal masks an already executed unsafe effect
Category: CONCEPT_CONFUSION; subsystem: safety-evaluation-to-action boundary.
Provenance: HYPOTHETICAL; no actual unsafe effect was executed for this guide.
Environment: A release suite scores the assistant's final text for refusal and helpfulness.
Trigger: The assistant invokes an effecting tool and later emits a refusal-style response.
Symptom: The safety judge marks the task safe while the effect ledger shows a prohibited action.
Decision at risk: Whether the release satisfies the prohibited-action boundary, not merely a text-style rubric.
Hypothesis A: Evaluation ignores intermediate tool calls and committed effects.
Hypothesis B: The tool call was proposed but never committed, and the ledger conflates those states.
Hypothesis C: The action was allowed, but a stale prohibited-action label misclassifies it.
Discriminating input: Separate proposed, authorized, attempted and committed tool events with policy versions.
Expected signal: A prohibited committed effect precedes the supposedly safe final response in the same task trace.
Negative result: A proposal-only event does not establish harm; investigate evaluator state semantics before counting failure.
Safe diagnostic: Use a sandboxed effect sink with synthetic payloads and an independent policy oracle.
Immediate containment: Block the affected tool path or release promotion while the effect semantics are reviewed.
Mitigation risk: Blanket tool disabling can make benign tasks impossible and increase manual workload.
Correction: Evaluate full task traces and effect outcomes, with text scoring as a separate dimension.
Prevention: Release manifests enumerate critical action suites and fail closed when a required suite is missing.
Verification oracle: A synthetic prohibited effect cannot receive an overall pass even if the final text refuses.
Boundary test: A denied proposal with no effect is distinguished from an executed prohibited effect.
Recovery: Reconcile and repair real effects separately from changing the response evaluator.
Owner: Safety-evaluation owner defines labels; tool-service owner supplies trustworthy effect states.
Escalation: Incident owner if a prohibited action may have escaped the sandbox or affected real users.
Evidence gap: No real evaluation suite or judge calibration artifact is available here.
Cost consequence: Trace-level review costs more than text-only review; sample design must retain critical failures.
Decision consequence: Safe wording is neither necessary nor sufficient evidence of safe external behavior.
Related sheet: [Module 50 governance and audits](../50_responsible_ai_governance_compliance_and_audits/INDEX.md).
Source support: [Safety focused brief](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-safety) explicitly separates empirical safety evidence.

<a id="p09-005"></a>
### P09-005 — A regulatory answer ranks an inapplicable version first
Category: VIOLATED_ASSUMPTION; subsystem: governance-content-to-product-answer eligibility.
Provenance: HYPOTHETICAL architecture exercise; not legal advice or a current applicability determination.
Environment: A retrieval assistant answers questions for different jurisdictions and historical question dates.
Trigger: A highly relevant amendment is retrieved without checking its effective interval or territorial scope.
Symptom: The answer cites authentic text that was not eligible for the question's date and jurisdiction.
Decision at risk: Which sources may support the answer before any relevance ranking occurs.
Hypothesis A: Eligibility metadata is absent or stale at ingestion.
Hypothesis B: Metadata is correct, but the query path applies relevance before eligibility filtering.
Hypothesis C: The user's question is ambiguous about date or jurisdiction and needs clarification.
Discriminating input: Authoritative version history, effective interval, recorded-time history and exact question context.
Expected signal: A boundary-date replay returns a citation whose eligible interval excludes the requested date.
Negative result: If the citation is eligible, investigate interpretation and claim entailment rather than version filtering.
Safe diagnostic: Use public authoritative texts and synthetic questions; route interpretation disputes to the legal owner.
Immediate containment: Abstain on affected scope and display the unresolved applicability question.
Mitigation risk: Broad abstention can reduce utility; do not silently substitute a different jurisdiction's rule.
Correction: Maintain bitemporal provenance where needed and filter eligible source versions before ranking.
Prevention: Source-owner review is triggered by amendments, source corrections and metadata changes.
Verification oracle: Tests immediately before, at and after effective boundaries return only eligible versions.
Boundary test: Missing date or jurisdiction leads to explicit clarification or an authorized general-information response.
Recovery: Identify affected answers by source/version and issue corrections through the product's documented workflow.
Owner: Content pipeline owner maintains metadata; authorized legal owner determines material applicability.
Escalation: Legal review when eligible sources conflict or the question implies consequential legal action.
Evidence gap: No maintained regulatory corpus or jurisdiction matrix is provided by the original inventory.
Cost consequence: Metadata maintenance and expert review are operating costs, not one-time indexing work.
Decision consequence: Authentic citations can still be inapplicable and should not be treated as legal authorization.
Related sheet: [Module 50 governance home](../50_responsible_ai_governance_compliance_and_audits/INDEX.md).
Source support: [IR013 planning commitment](../../00_REFERENCE_INFORMED_CASE_PLAN.md) requires dated eligibility, not a legal conclusion.

<a id="p09-006"></a>
### P09-006 — Withdrawal removes a record but leaves its derived exposure
Category: VIOLATED_ASSUMPTION; subsystem: privacy-purpose-to-data-lifecycle handoff.
Provenance: HYPOTHETICAL; obligations and exceptions require context-specific authoritative review.
Environment: A product uses source records, document chunks, embeddings, caches and audit records.
Trigger: A valid purpose change or authorized withdrawal/deletion instruction affects source data.
Symptom: The source row disappears, yet the product still retrieves a chunk or cached summary derived from it.
Decision at risk: Whether the lifecycle action covers all in-scope representations without destroying required evidence.
Hypothesis A: Data lineage does not connect the source object to derived retrieval artifacts.
Hypothesis B: The deletion job is queued or failed, while the interface incorrectly reports completion.
Hypothesis C: A documented exception requires some retention, but the product exposes retained data to ordinary retrieval.
Discriminating input: Authorized request scope, lineage graph, retention rule, job state and retrieval permissions.
Expected signal: A synthetic source identifier remains reachable in a derivative after the workflow reports completion.
Negative result: Source absence alone does not establish removal from backups, exports or external processors.
Safe diagnostic: Use synthetic lifecycle fixtures; inspect real personal content only under appropriate access authority.
Immediate containment: Suppress affected objects from new retrieval while the authorized lifecycle scope is reconciled.
Mitigation risk: Overbroad suppression can break legitimate workflows and evidence retention if applied indiscriminately.
Correction: Propagate purpose/retention state through derivation lineage and separate retained-for-audit from available-for-use.
Prevention: Every new store or processor declares its deletion, expiry and exception semantics before adoption.
Verification oracle: A completed synthetic removal cannot be retrieved through supported product paths within the promised scope.
Boundary test: Legally or contractually retained records are access-restricted and not silently treated as general-purpose data.
Recovery: Reconcile external exports and processor confirmations; local deletion cannot recall an already disclosed copy.
Owner: Data lifecycle owner coordinates stores; privacy/legal owner resolves applicable obligations and exceptions.
Escalation: Privacy incident workflow if exposure continued beyond the authorized purpose boundary.
Evidence gap: No actual consent basis, deletion instruction, contractual retention term or jurisdiction is established here.
Cost consequence: Lifecycle operations require ongoing engineering and processor coordination, which belongs in full TCO.
Decision consequence: A privacy control is an end-to-end lifecycle mechanism, not one database operation.
Related sheet: [Module 49 privacy/security home](../49_ai_security_privacy_safety_and_red_teaming/INDEX.md).
Source support: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) is voluntary risk methodology, not an applicability ruling.

<a id="p09-007"></a>
### P09-007 — The audit link resolves, but the historical evidence has changed
Category: VIOLATED_ASSUMPTION; subsystem: evidence-to-governance decision traceability.
Provenance: HYPOTHETICAL; no real audit deficiency is asserted.
Environment: Release approvals link to dashboards, shared documents and model registry aliases.
Trigger: A dashboard query, document or alias is updated after the release decision.
Symptom: Reviewers cannot reconstruct which evidence supported the earlier approval despite every URL still opening.
Decision at risk: Whether the approval's historical basis is identifiable and consistent with the deployed bundle.
Hypothesis A: References point to mutable latest views instead of decision-time snapshots.
Hypothesis B: Snapshots exist, but retention or permissions prevent authorized review access.
Hypothesis C: The release identity was reused for a different configuration, making the evidence ambiguous.
Discriminating input: Decision timestamp, immutable manifest, source query/configuration and evidence access policy.
Expected signal: Reopening the historical approval yields current results rather than the recorded decision-time result set.
Negative result: Stable snapshots do not establish that sampling, labels or conclusions were correct.
Safe diagnostic: Compare metadata and digests first; avoid bulk copying sensitive evaluation content into audit packets.
Immediate containment: Mark the disputed historical claim unverified and preserve available artifacts under authorized retention.
Mitigation risk: Emergency evidence copies can proliferate sensitive data; define access and expiry before copying.
Correction: Store immutable evidence identifiers, query versions, sampling frames and release manifests with bounded access.
Prevention: Approval tooling refuses mutable-only references for material claims unless their limitations are explicit.
Verification oracle: A later dashboard update cannot alter the artifact resolved by a historical approval record.
Boundary test: Expired access returns a clear evidence-unavailable state, not a misleading current substitute.
Recovery: Reconstruct only what surviving records support and record irrecoverable gaps rather than inventing history.
Owner: Governance evidence owner defines required records; platform owners implement artifact identity and retention.
Escalation: Audit/legal owner if missing evidence affects a consequential attestation or required record.
Evidence gap: No real approval packet or historical snapshot is available in this teaching exercise.
Cost consequence: Immutable storage and controlled access have costs; indiscriminate full-data retention is not the default answer.
Decision consequence: A working URL is not sufficient historical provenance.
Related sheet: [Module 50 documentation/audits](../50_responsible_ai_governance_compliance_and_audits/INDEX.md).
Source support: [NIST SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) supplies secure-development vocabulary, not audit certification.

<a id="p09-008"></a>
### P09-008 — The success rate improves because failures leave the denominator
Category: NUMERICAL_EDGE; subsystem: product-metric-to-governance reporting.
Provenance: HYPOTHETICAL; numerical examples are constructed, not user-study results.
Environment: A pilot reports useful completions divided by eligible tasks.
Trigger: Timeouts, unsupported inputs or escalations are relabeled as ineligible after their outcomes are known.
Symptom: The reported success fraction rises while the user population receives no corresponding improvement.
Decision at risk: Whether a metric increase supports expansion or merely reflects outcome-dependent exclusion.
Hypothesis A: Inclusion rules changed after outcome observation.
Hypothesis B: The actual eligible population changed legitimately because scope was narrowed prospectively.
Hypothesis C: Event joins dropped failures because their terminal records lack expected fields.
Discriminating input: Predecision eligibility specification, full arrival ledger and exclusion reason/timestamp.
Expected signal: Failed tasks are disproportionately removed after terminal outcomes under a new exclusion rule.
Negative result: If scope narrowed prospectively, report the new population and do not compare unadjusted headline rates.
Safe diagnostic: Rebuild counts from task IDs and eligibility-time fields without exposing task content unnecessarily.
Immediate containment: Freeze promotion decisions based on the disputed rate and publish both old and reconstructed counts.
Mitigation risk: A rushed backfill can duplicate tasks; reconcile unique business IDs and repeated attempts.
Correction: Evaluate eligibility before treatment and retain outcome-independent exclusions with clear reasons.
Prevention: Dashboards display arrival, eligible, attempted, completed, useful and excluded counts together.
Verification oracle: Adding failed tasks to the arrival ledger cannot improve the eligible-task success rate through silent exclusion.
Boundary test: Zero eligible tasks produces an undefined/not-applicable rate with explicit count, never an artificial 100%.
Recovery: Restate affected reports and decisions using the reconciled population and window.
Owner: Product analytics owner defines the metric; governance owner challenges material denominator changes.
Escalation: Experiment owner when exclusions alter randomization balance or causal interpretation.
Evidence gap: No actual task ledger or observed rate is supplied here.
Cost consequence: Failed and excluded work can still incur cost and must remain in the declared cost boundary.
Decision consequence: A rate is inseparable from its numerator, denominator, eligibility time and population.
Related sheet: [Module 51 adoption and UX](../51_ai_product_management_ux_and_human_ai_collaboration/INDEX.md).
Source support: [Google SLO methodology](https://sre.google/workbook/implementing-slos/) distinguishes good events from total events and measurement coverage.

<a id="p09-009"></a>
### P09-009 — A safety improvement is actually blanket refusal
Category: CONCEPT_CONFUSION; subsystem: safety-to-product utility tradeoff.
Provenance: HYPOTHETICAL; aligned with the visible safety brief's separate benign and prohibited prompt sets.
Environment: A team tunes a refusal policy and promotes the model with fewer unsafe completions.
Trigger: Policy changes reduce both prohibited completions and legitimate task completions.
Symptom: The safety metric improves while users abandon benign tasks or route around the assistant.
Decision at risk: Whether the intervention improves the intended boundary rather than disabling the product indiscriminately.
Hypothesis A: The safety policy overgeneralizes prohibited categories to benign requests.
Hypothesis B: The benign test labels are ambiguous and require adjudication.
Hypothesis C: Retrieval or tool outages cause failure that is being mislabeled as safety refusal.
Discriminating input: Separate benign/prohibited labels, refusal decisions, tool errors and human-adjudicated ambiguous examples.
Expected signal: Benign false-refusal rate rises while the unsafe-compliance rate falls under the same held-out protocol.
Negative result: Stable benign refusal does not rule out lower answer quality on nonrefused tasks.
Safe diagnostic: Use a labeled held-out suite with inert effects and independent review of disputed labels.
Immediate containment: Limit the changed policy to the supported task scope or restore the prior authorized baseline.
Mitigation risk: Relaxing refusal thresholds without effect-level controls can reintroduce harmful actions.
Correction: Gate on both unsafe compliance and benign task usefulness, with explicit uncertainty and critical-failure rules.
Prevention: Maintain separate slices for benign lookalikes, prohibited requests and ambiguous boundary cases.
Verification oracle: A model that refuses every prompt cannot pass the declared useful-task gate.
Boundary test: Clarification or human escalation can be appropriate for ambiguity and should not be scored as arbitrary failure.
Recovery: Explain changed behavior to users and track workarounds that may create new exposure outside the managed product.
Owner: Safety owner defines prohibited boundaries; product owner defines legitimate task outcomes.
Escalation: Policy adjudicator when label disagreement changes the intervention decision.
Evidence gap: The source exercise's 8/100 and 12/100 counts are illustrative and not a release result.
Cost consequence: Over-refusal can push users into expensive manual work even if inference spend falls.
Decision consequence: Safety and usefulness are separate outcomes, not interchangeable complements.
Related sheet: [Module 49 safety and alignment](../49_ai_security_privacy_safety_and_red_teaming/INDEX.md).
Source support: [Safety focused brief](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-safety) requires false-refusal and unsafe-compliance reasoning.

<a id="p09-010"></a>
### P09-010 — Confidence displays transfer poorly to a new population
Category: VIOLATED_ASSUMPTION; subsystem: calibration-to-human reliance.
Provenance: HYPOTHETICAL; no particular model confidence API is assumed.
Environment: A product presents a reliability cue calibrated on one language, task family and user group.
Trigger: Expansion introduces a different language, input format or decision consequence.
Symptom: Users treat a high-confidence cue as reliable, but correctness among those cases is lower in the new slice.
Decision at risk: Whether the cue supports informed reliance for the actual deployment population.
Hypothesis A: Calibration shifted because the task or data distribution changed.
Hypothesis B: The confidence cue is a transformed model score with no demonstrated probability interpretation.
Hypothesis C: Outcome labels are delayed or inconsistent across the new population.
Discriminating input: Score definition, mature outcome labels, reliability by slice and the cue's displayed semantics.
Expected signal: Cases in a displayed confidence band have materially different observed correctness by relevant population.
Negative result: Overall calibration can conceal subgroup miscalibration; aggregate agreement is not sufficient.
Safe diagnostic: Review authorized samples with a shared correctness rubric and blinded adjudication where feasible.
Immediate containment: Remove or qualify the misleading cue for the unsupported slice and offer an explicit uncertainty path.
Mitigation risk: Removing cues without explanation can increase confusion or shift users to equally unsupported heuristics.
Correction: Recalibrate on representative evidence or replace numeric confidence with bounded task-specific guidance.
Prevention: Treat population/task changes as calibration-review triggers rather than cosmetic localization work.
Verification oracle: The displayed wording matches what the score actually estimates under the evaluated population.
Boundary test: Unsupported languages or task categories receive clarification, abstention or clearly labeled unverified guidance.
Recovery: Revisit consequential decisions made under the misleading cue through the product's review process.
Owner: Model evaluation owner for calibration evidence; UX owner for presentation and reliance behavior.
Escalation: Domain reviewer when incorrect reliance can produce consequential user harm.
Evidence gap: No scored dataset, calibration curve or measured reliance study is provided here.
Cost consequence: Calibration evidence and translated review are operating costs of expansion.
Decision consequence: A number that looks precise can increase misuse when its meaning changes across populations.
Related sheet: [Module 51 calibration/transparency](../51_ai_product_management_ux_and_human_ai_collaboration/INDEX.md).
Source support: [Human-AI interaction guidelines](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/) support interaction design inquiry, not this model's calibration.

<a id="p09-011"></a>
### P09-011 — Mandatory review creates an unstable service
Category: SCALE_FAILURE; subsystem: governance-control-to-human capacity.
Provenance: HYPOTHETICAL; the queue arithmetic is derived from constructed constant rates.
Environment: A release requires human review for a fraction of arriving tasks.
Trigger: The escalation policy increases referrals without changing measured reviewer capacity.
Symptom: Cases age, users retry, duplicate referrals arrive and the nominal safety control becomes a delay mechanism.
Decision at risk: Whether the required review can finish within the task's consequential deadline.
Hypothesis A: Referral arrivals exceed sustainable completed-review rate.
Hypothesis B: Total capacity is adequate, but a priority class or difficult-case mix causes deadline misses.
Hypothesis C: Duplicate referrals inflate counts because task retries are not deduplicated.
Discriminating input: Task-linked arrival/completion timestamps, service classes, duplicates and staffed productive hours.
Expected signal: P09-QUEUE predicts positive backlog growth and the event ledger reconciles the growth.
Negative result: Balanced average rates do not rule out bursts, class starvation or unacceptable tail waiting.
Safe diagnostic: Analyze aggregate flow and synthetic replay before changing approval requirements.
Immediate containment: Reduce intake or narrow pilot scope to the measured service envelope while preserving critical review.
Mitigation risk: Raising thresholds purely to reduce referrals can silently expose unreviewed high-risk cases.
Correction: Redesign referral rules, case preparation and staffing against task-level risk and deadline requirements.
Prevention: Review-policy changes include an arrival/service estimate and an overload response owner.
Verification oracle: Under the declared maximum intake, backlog and oldest-case age remain within the scenario's accepted limits.
Boundary test: At lambda=mu, a positive backlog does not drain under the constant-rate model.
Recovery: Deduplicate and prioritize existing cases using documented policy; tell users when deadlines cannot be met safely.
Owner: Operations owner controls service; product/governance owners jointly control intake and review policy.
Escalation: Business owner when safe demand exceeds available review capacity or deadlines require redesign.
Evidence gap: Handling-time distributions, burstiness and real staffing availability are not measured here.
Cost consequence: Human review can dominate full cost even when provider inference is inexpensive.
Decision consequence: Human-in-the-loop is an operational system, not a free safety adjective.
Related sheet: [Module 51 review workflow](../51_ai_product_management_ux_and_human_ai_collaboration/INDEX.md).
Source support: [Google SLO methodology](https://sre.google/workbook/implementing-slos/) supports user-facing measurement, not a universal review deadline.

<a id="p09-012"></a>
### P09-012 — Reviewers approve without an independent basis for judgment
Category: VIOLATED_ASSUMPTION; subsystem: human-control-to-effective oversight.
Provenance: HYPOTHETICAL; no claim about a real reviewer team's behavior is made.
Environment: A system presents a fluent recommendation and asks a reviewer to approve it quickly.
Trigger: Review volume grows or the interface hides source evidence and alternative explanations.
Symptom: Approval rates remain high even when plausible synthetic errors are introduced into controlled review exercises.
Decision at risk: Whether review meaningfully checks the relevant error mechanism rather than recording a click.
Hypothesis A: Automation bias or anchoring makes reviewers follow the recommendation despite contrary evidence.
Hypothesis B: Reviewers lack time, authority or necessary source information to reject it.
Hypothesis C: Seeded errors are unrealistic or the assessment rubric does not match the actual task.
Discriminating input: Blinded review exercises, evidence availability, time budget, reviewer comments and escalation authority.
Expected signal: Error detection changes when recommendations are hidden or evidence access/time is improved.
Negative result: A high approval rate alone does not prove ineffective review when cases are mostly correct.
Safe diagnostic: Use controlled synthetic tasks; do not deliberately seed harmful errors into live consequential workflows.
Immediate containment: Reduce review volume or hold high-consequence actions where reviewers cannot establish a basis.
Mitigation risk: Extra mandatory clicks can increase fatigue without improving detection.
Correction: Design review around the decision, relevant evidence, disagreement and a usable reject/escalate path.
Prevention: Assess review capability and information sufficiency when workload or interface changes.
Verification oracle: Reviewers can identify specified error classes and explain the evidence supporting rejection.
Boundary test: When evidence is unavailable, the reviewer can defer instead of being forced to approve or guess.
Recovery: Reassess previously approved high-consequence cases according to risk and lawful access limits.
Owner: Operations/UX owners design the review task; domain owner defines what competent judgment requires.
Escalation: Governance owner when review is a release condition but cannot be performed reliably.
Evidence gap: No measured reviewer accuracy, experimental assignment or task timing is available here.
Cost consequence: Effective oversight requires paid time and skills; throughput targets must account for that work.
Decision consequence: Human involvement is not equivalent to human control.
Related sheet: [Module 51 human-AI collaboration](../51_ai_product_management_ux_and_human_ai_collaboration/INDEX.md).
Source support: [Human-AI interaction guidelines](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/) are a design reference, not proof of oversight quality.

<a id="p09-013"></a>
### P09-013 — Volunteer-pilot value does not survive representative deployment
Category: VIOLATED_ASSUMPTION; subsystem: product-experiment-to-transformation inference.
Provenance: HYPOTHETICAL; no causal benefit is reported.
Environment: An enthusiastic volunteer group uses a copilot and reports better completion speed.
Trigger: Leadership extrapolates that effect to all roles, teams and task types.
Symptom: Broader rollout produces weaker benefit and more support demand than the pilot forecast.
Decision at risk: Whether the pilot estimates the effect for the proposed deployment population.
Hypothesis A: Volunteers had stronger skills, easier tasks or more time to adapt.
Hypothesis B: The intervention genuinely works, but rollout support or system performance changed.
Hypothesis C: Pilot measurement excluded hidden review/rework time that broader users now report.
Discriminating input: Eligibility frame, task/role mix, assignment mechanism, support exposure and full task timing.
Expected signal: Pilot participants differ materially from eligible nonparticipants on effect-relevant characteristics.
Negative result: Matching observed characteristics does not rule out unobserved motivation or local process differences.
Safe diagnostic: Use a bounded representative evaluation with appropriate assignment and guardrails after eligibility is established.
Immediate containment: Limit extrapolated benefit claims and expand only within the supported task/user envelope.
Mitigation risk: Restricting access can create fairness concerns; explain the evidence-gathering rationale and alternatives.
Correction: Estimate effects for the target population using a defensible study design and explicit interference assumptions.
Prevention: Pilot plans state the target population and the evidence needed for each expansion boundary.
Verification oracle: Report participant flow and task mix from invitation through mature outcomes, not only satisfied-user anecdotes.
Boundary test: A subgroup with no observations remains unsupported rather than inheriting the overall estimate.
Recovery: Rebaseline the investment plan using representative outcomes and support costs.
Owner: Product experimentation owner for inference; transformation owner for scope and support resources.
Escalation: Executive sponsor when the commercial commitment assumes evidence beyond the pilot's support.
Evidence gap: No recruitment, randomization, outcome dataset or support log exists for this teaching case.
Cost consequence: Scaling changes training, support, review and infrastructure costs as well as user counts.
Decision consequence: A pilot is an evidence source with a population boundary, not a universal expansion license.
Related sheet: [Module 53 transformation measurement](../53_communication_leadership_and_ai_transformation/INDEX.md).
Source support: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) supports contextual risk assessment, not this pilot's external validity.

<a id="p09-014"></a>
### P09-014 — The copilot saves one role's time by exporting work downstream
Category: CONCEPT_CONFUSION; subsystem: local-productivity-to-end-to-end value.
Provenance: HYPOTHETICAL; no measured workflow saving is asserted.
Environment: Analysts draft reports that are then checked by quality staff and acted on by an operations team.
Trigger: Draft generation becomes faster but produces more verification, correction or exception handling downstream.
Symptom: Analyst time falls while total labor per accepted report rises.
Decision at risk: Whether the intervention improves the entire task outcome rather than shifting effort between budgets.
Hypothesis A: The metric boundary excludes downstream review and rework.
Hypothesis B: A temporary learning effect raises review work but may decline with experience.
Hypothesis C: Report volume or difficulty changed independently of the copilot.
Discriminating input: End-to-end task IDs, role-specific time, accepted outputs, rework loops and case mix.
Expected signal: Savings in the initiating role are smaller than additional downstream work for comparable tasks.
Negative result: A single time-study window may not distinguish persistent burden from adaptation cost.
Safe diagnostic: Observe authorized workflow timing and use aggregate role-level measures without intrusive employee surveillance.
Immediate containment: Stop claiming total productivity gains from one role's timing and disclose the measured boundary.
Mitigation risk: Removing the tool can lose legitimate benefits; compare a bounded baseline before a broad reversal.
Correction: Define useful completion at accepted business output and include all materially affected labor.
Prevention: Product requirements include downstream stakeholders and adverse workload guardrails.
Verification oracle: A task cannot count as useful before required review/rework has reached its defined completion state.
Boundary test: Faster draft creation with twice the correction work is reported as a tradeoff, not automatic improvement.
Recovery: Reconcile labor assumptions in the business case and adjust team capacity plans.
Owner: End-to-end process owner arbitrates tradeoffs across analyst, quality and operations teams.
Escalation: Finance and workforce leads when claimed savings affect staffing commitments.
Evidence gap: No observed time study or accepted-output dataset is provided here.
Cost consequence: Rework is part of the full cost boundary and may dominate invocation savings.
Decision consequence: A local metric is not the same as system value.
Related sheet: [Module 52 unit economics](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [Green Book appraisal](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026) considers broad impacts and uncertainty in its own public-appraisal scope.

<a id="p09-015"></a>
### P09-015 — Released capacity is booked as cash that cannot be avoided
Category: CONCEPT_CONFUSION; subsystem: economic-case-to-financial-plan handoff.
Provenance: HYPOTHETICAL; illustrative labor values are not salary or market-price claims.
Environment: A business case multiplies estimated minutes saved by a loaded hourly labor rate.
Trigger: Leadership reduces the budget by the resulting estimate without changing payable commitments or productive output.
Symptom: The tool saves time on some tasks, but the expected cash reduction does not appear.
Decision at risk: Whether benefit is capacity, service improvement, avoided hiring or actual cash saving.
Hypothesis A: Allocated labor value was incorrectly treated as immediately avoidable expense.
Hypothesis B: Time was released but fragmented, so it could not be redeployed to useful work.
Hypothesis C: Savings were offset by new support, review or licensing costs omitted from the financial plan.
Discriminating input: Payroll/contract commitments, vacancy plans, work allocation and full incremental costs.
Expected signal: Paid hours and headcount commitments remain unchanged despite reported task-minute savings.
Negative result: No cash reduction does not prove no value; throughput or quality may have improved.
Safe diagnostic: Reconcile aggregate finance/workforce records rather than making unsupported employee-level performance judgments.
Immediate containment: Restate the benefit category and stop presenting capacity release as realized cash.
Mitigation risk: Forcing short-term headcount cuts can destroy service capacity or the adoption capability needed for value.
Correction: Model explicit realization mechanisms for each benefit and attach a responsible business owner.
Prevention: Finance signs off on cash, capacity and avoided-growth categories before investment approval.
Verification oracle: A claimed cash saving links to an actual reduced payable or approved avoided commitment in the stated window.
Boundary test: If no resource commitment changes, report released capacity or improved service, not cash saved.
Recovery: Revise budgets and portfolio forecasts to remove unachievable realization assumptions.
Owner: Finance owner validates realization; process owner decides how released capacity is redeployed.
Escalation: Executive sponsor if investment approval depended on a cash target that lacks a mechanism.
Evidence gap: No payroll, hiring, invoice or throughput evidence is available for this scenario.
Cost consequence: A financially negative pilot can still be useful learning, but that purpose must be explicit.
Decision consequence: Value categories answer different questions and must not be added indiscriminately.
Related sheet: [Module 52 economics](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [Green Book appraisal](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026) separates appraisal considerations; no company accounting rule is inferred.

<a id="p09-016"></a>
### P09-016 — Retries make low call prices an expensive task
Category: NUMERICAL_EDGE; subsystem: agent-reliability-to-unit-economics.
Provenance: HYPOTHETICAL; P09-COST provides the declared accounting identity.
Environment: A multi-step assistant retries retrieval, generation or tools before a task completes.
Trigger: An upstream reliability regression increases attempts while the price per invocation stays fixed.
Symptom: Procurement reports stable low call price, but monthly spend and cost per useful task rise.
Decision at risk: Which route or workflow is economical for successful end-to-end tasks.
Hypothesis A: Failed and retried attempts are omitted from the cost numerator.
Hypothesis B: Useful completion declined, shrinking the denominator even with stable attempt cost.
Hypothesis C: Human recovery and support costs increased outside the provider bill.
Discriminating input: Task-attempt linkage, all billable events, useful completion labels and review/remediation labor.
Expected signal: Cost rises with attempts per task or falls less than call-price reductions suggest.
Negative result: Stable attempt counts do not rule out longer prompts, output tokens or contract minimum charges.
Safe diagnostic: Reconcile invoices and synthetic traces without replaying effecting actions against live systems.
Immediate containment: Bound attempt budgets and return an explicit failure/escalation path rather than retry indefinitely.
Mitigation risk: Tighter retry budgets can reduce useful completion; evaluate the joint cost/success tradeoff.
Correction: Compare routes using complete cost per useful task under the same task mix and deadline constraints.
Prevention: Every retry carries the task ID, reason and remaining budget; cost telemetry includes failed work.
Verification oracle: Six thousand attempts and three thousand useful tasks cannot use six thousand as the useful-task denominator.
Boundary test: With zero useful tasks, report undefined useful-task cost and the incurred spend separately.
Recovery: Reconcile possibly duplicated effects before repeating actions that lacked idempotency protection.
Owner: Platform owner controls retry semantics; finance/product owners define full cost and useful outcomes.
Escalation: Incident owner when retries amplify load, duplicate effects or exceed spend authority.
Evidence gap: No real invoices, token counts or retry traces are provided here.
Cost consequence: A fixed call price is only one input to task economics.
Decision consequence: Cheap attempts can be expensive outcomes.
Related sheet: [Module 52 model/cloud cost](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [IR002 planning commitment](../../00_REFERENCE_INFORMED_CASE_PLAN.md) requires bounded budgets and successful-task cost.

<a id="p09-017"></a>
### P09-017 — An outage fallback forgets the original eligibility constraints
Category: VIOLATED_ASSUMPTION; subsystem: resilience-to-procurement/security boundaries.
Provenance: HYPOTHETICAL; no provider capability, region or availability guarantee is asserted.
Environment: A routing service can switch between provider deployments when one is unavailable.
Trigger: The primary eligible route times out and a generic fallback list is used.
Symptom: Availability improves while a destination, capability, budget or deadline constraint is violated.
Decision at risk: Whether fallback is allowed for the same tenant, purpose and payload as the original request.
Hypothesis A: Constraints are applied only to primary selection, not fallback selection.
Hypothesis B: Provider metadata changed and the cached eligibility record is stale.
Hypothesis C: The apparent alternate destination is an internal proxy and actual processing location needs verification.
Discriminating input: Exact fallback route, current contract/capability evidence, policy context and fault trace.
Expected signal: A replayed primary failure sends the request to a route excluded by the original hard predicates.
Negative result: A successful network request does not prove contractual or processing-boundary compliance.
Safe diagnostic: Use synthetic payloads and provider stubs only to test routing logic, clearly not current provider behavior.
Immediate containment: Restrict fallback to independently eligible routes; abstain when none remains.
Mitigation risk: Constraint-preserving abstention lowers apparent availability but preserves the declared boundary.
Correction: Reapply eligibility and remaining deadline/spend budgets at every route transition.
Prevention: Fallback contracts include allowed destinations, stale-metadata behavior and bounded retry semantics.
Verification oracle: Inject a primary timeout; every chosen fallback still satisfies the original predicates.
Boundary test: Empty eligible fallback set produces explicit unavailability rather than unconstrained best effort.
Recovery: Investigate historical route logs if actual requests may have crossed the boundary.
Owner: Routing owner implements transitions; procurement/security owners validate capability and contract metadata.
Escalation: Data/legal owner if outage pressure creates a request to override a binding constraint.
Evidence gap: Current provider terms, regions, capabilities and prices require fresh primary evidence.
Cost consequence: A lawful fallback may cost more; compare inside the constrained set rather than weakening the constraint silently.
Decision consequence: Resilience is correct only if the recovered service still satisfies the original contract.
Related sheet: [Module 52 constrained routing](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [IR002 reference plan](../../00_REFERENCE_INFORMED_CASE_PLAN.md) is a teaching brief, not service assurance.

<a id="p09-018"></a>
### P09-018 — A vendor wins the benchmark by seeing the test
Category: VIOLATED_ASSUMPTION; subsystem: procurement-evidence-to-selection validity.
Provenance: HYPOTHETICAL; no supplier misconduct is alleged.
Environment: Vendors receive sample tasks and teams repeatedly tune prompts or settings against a shared benchmark.
Trigger: The same tasks become both development material and the final selection test.
Symptom: A vendor wins the visible suite but underperforms on newly sampled eligible business tasks.
Decision at risk: Whether the benchmark predicts comparative performance on the target workload.
Hypothesis A: Test tasks leaked into tuning or demonstration preparation.
Hypothesis B: The original suite was unrepresentative even without leakage.
Hypothesis C: The deployment configuration differs from the evaluated configuration.
Discriminating input: Task lineage, access history, development/test separation and exact configuration manifests.
Expected signal: Performance drops on a sealed representative holdout while visible-suite gains persist.
Negative result: A holdout drop alone does not prove deliberate leakage; workload shift remains a competing cause.
Safe diagnostic: Use authorized nonconfidential or properly protected task samples and consistent evaluation access rules.
Immediate containment: Delay irreversible procurement commitments based on the contaminated or mismatched comparison.
Mitigation risk: Repeatedly creating and inspecting holdouts can contaminate the replacement evidence as well.
Correction: Predeclare tasks, rubric, hard constraints, versions, tuning budget and sealed final-evaluation protocol.
Prevention: Separate development examples from final selection data and log all evaluation adaptations.
Verification oracle: The final holdout is not used to choose prompts or settings before the one declared comparison.
Boundary test: Missing mandatory capability or security evidence disqualifies an option regardless of benchmark rank.
Recovery: Reopen the selection decision if contractual commitments allow and evidence materially changes the ranking.
Owner: Procurement evaluation owner controls fair protocol; domain reviewers adjudicate useful-task outcomes.
Escalation: Commercial/legal owner when new evidence conflicts with signed acceptance terms.
Evidence gap: No actual vendor submissions, contracts or benchmark runs are available here.
Cost consequence: Evaluation labor and secure data handling belong in procurement TCO.
Decision consequence: A benchmark is a measurement procedure, not a leaderboard detached from its access history.
Related sheet: [Module 52 vendor evaluation](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [NIST SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) supports supplier communication, not comparative model performance.

<a id="p09-019"></a>
### P09-019 — Commitment tiers reverse the apparent cheapest option
Category: NUMERICAL_EDGE; subsystem: commercial-contract-to-economic comparison.
Provenance: HYPOTHETICAL; no real rate card or current discount is quoted.
Environment: One offer has low usage rates with a minimum commitment; another has higher variable rates and no minimum.
Trigger: Adoption or useful completion falls below the forecast volume used in procurement.
Symptom: The nominally cheaper offer costs more per useful task because unused commitment remains payable.
Decision at risk: Which contract is affordable across plausible volume and quality scenarios.
Hypothesis A: The comparison applies a discounted unit price without the minimum-spend condition.
Hypothesis B: Usage shifted between billable classes or regions that have different terms.
Hypothesis C: The cost increase comes from task failures rather than contract structure.
Discriminating input: Signed pricing terms, billable unit definitions, volume scenarios and useful-task counts.
Expected signal: Recomputing total payable cost at low volume reverses the rank of the offers.
Negative result: Similar total cost at one volume does not establish equal flexibility, exit cost or security suitability.
Safe diagnostic: Model contracts using illustrative or authorized commercial terms; avoid publishing confidential prices.
Immediate containment: Stop presenting a single unit-price figure as the complete investment comparison.
Mitigation risk: Avoiding commitments entirely may sacrifice worthwhile discounts when stable demand is well evidenced.
Correction: Compare total payable cost, capacity risk, termination and migration obligations across scenarios.
Prevention: Approval includes switching values for volume, useful completion and contract term assumptions.
Verification oracle: The model charges the minimum when consumption falls below the contractual floor.
Boundary test: Zero usage can still produce positive payable cost under a minimum commitment.
Recovery: Reforecast spend and negotiate within actual contract rights rather than assuming cancellation is free.
Owner: Procurement owns terms; finance owns affordability; product owns realistic usage and success forecasts.
Escalation: Commercial authority when downside exposure exceeds the approved investment envelope.
Evidence gap: Actual prices, commitments, taxes and termination rights are unavailable in this exercise.
Cost consequence: A lower marginal price can coexist with a higher total and useful-task cost.
Decision consequence: Contract conditions are part of the formula, not small-print annotations after selection.
Related sheet: [Module 52 AI TCO](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [Green Book appraisal](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026) motivates sensitivity and switching-value analysis within its stated scope.

<a id="p09-020"></a>
### P09-020 — Multiple vendors share the failure domain
Category: VIOLATED_ASSUMPTION; subsystem: vendor-diversification-to-recovery assurance.
Provenance: HYPOTHETICAL; no vendor dependency architecture is asserted.
Environment: A business case claims resilience because two providers are contracted.
Trigger: Both routes depend on a shared identity service, network path, cloud region, data source or internal gateway.
Symptom: Both supposedly independent options fail together during a dependency outage.
Decision at risk: Whether diversification provides the intended failure isolation for the specific critical workflow.
Hypothesis A: The suppliers or internal routes share an unmodeled dependency.
Hypothesis B: The fallback is independent, but common local configuration prevents access to both.
Hypothesis C: Capacity exhaustion propagates through retries and overwhelms an otherwise independent fallback.
Discriminating input: Dependency graph, failure-domain evidence, authentication paths, quotas and fault timeline.
Expected signal: A controlled dependency failure disables both routes through the same shared node.
Negative result: One successful failover exercise does not establish independence for every failure mode.
Safe diagnostic: Use authorized controlled fault tests or tabletop dependency review; do not disrupt third-party services.
Immediate containment: Reduce retry amplification and provide the documented manual or unavailable path.
Mitigation risk: Moving traffic abruptly can exceed fallback quotas or violate task-specific eligibility constraints.
Correction: Design recovery around evidenced independent paths and the actual bottleneck, not supplier-count alone.
Prevention: Procurement and architecture reviews jointly record shared dependencies and tested recovery assumptions.
Verification oracle: A fault in each declared primary failure domain leaves the stated fallback path usable under load.
Boundary test: Common identity failure is considered explicitly instead of assuming another model endpoint solves it.
Recovery: Reconcile incomplete tasks and committed effects before replaying work through a restored route.
Owner: Reliability/platform owner maps dependencies; vendor owner obtains supported supplier evidence.
Escalation: Business continuity owner when no independent safe path satisfies the deadline and authority constraints.
Evidence gap: No real dependency map, failover run or capacity measurement is available here.
Cost consequence: True independence can require duplicated integration, reserve capacity and operational skill.
Decision consequence: Supplier diversity is not equivalent to failure-domain diversity.
Related sheet: [Module 52 vendor management](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [Google SLO methodology](https://sre.google/workbook/implementing-slos/) explains user-visible dependency considerations, not these suppliers' resilience.

<a id="p09-021"></a>
### P09-021 — Several projects claim the same pool of saved work
Category: CONCEPT_CONFUSION; subsystem: use-case-economics-to-portfolio aggregation.
Provenance: HYPOTHETICAL; no actual portfolio return is reported.
Environment: Summarization, search and drafting projects each forecast savings on the same analyst workflow.
Trigger: Leadership sums each project's standalone benefit without accounting for overlap or sequencing.
Symptom: Portfolio savings exceed the total baseline effort available to remove or redeploy.
Decision at risk: Which combination of projects creates incremental value within budget and shared constraints.
Hypothesis A: Multiple projects attribute the same task-minute saving to themselves.
Hypothesis B: Projects are complementary, but benefits depend on delivering the whole chain rather than summing parts.
Hypothesis C: Baseline effort estimates use inconsistent populations or time windows.
Discriminating input: Shared task population, baseline process map, benefit ownership and incremental scenario definitions.
Expected signal: The same baseline work unit appears in several additive benefit rows.
Negative result: Unique labels do not prove unique benefits if the underlying work overlaps.
Safe diagnostic: Reconcile aggregate workflow units and assumptions without treating forecasts as measured employee performance.
Immediate containment: Present the unreconciled sum as an upper-bound scenario, not an approved portfolio benefit.
Mitigation risk: Assigning all value to one project can hide genuine complements; model the joint intervention explicitly.
Correction: Define mutually consistent baseline and incremental scenarios with one accountable owner per realized outcome.
Prevention: Portfolio review checks shared populations, dependencies and benefit realization mechanisms.
Verification oracle: The combined forecast cannot eliminate more of a fixed baseline work pool than that pool contains.
Boundary test: A foundational platform can be necessary for value without being assigned a fictitious standalone share of every outcome.
Recovery: Restate portfolio priorities and affordability using reconciled value and dependency assumptions.
Owner: Portfolio/finance owner integrates benefits; process owner validates the shared work boundary.
Escalation: Executive sponsor when project incentives reward duplicate attribution.
Evidence gap: No baseline work census or observed joint intervention is supplied.
Cost consequence: Shared infrastructure costs can also be double counted or omitted; reconcile both sides.
Decision consequence: Additive spreadsheets require additive quantities, not merely differently named projects.
Related sheet: [Module 52 portfolio strategy](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [Green Book policy hierarchy discussion](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026) explains why component value need not be separately apportioned.

<a id="p09-022"></a>
### P09-022 — The central AI team becomes an unbounded dependency queue
Category: SCALE_FAILURE; subsystem: team-topology-to-product-delivery flow.
Provenance: HYPOTHETICAL; organizational choices are conditional design recommendations.
Environment: Every product team must ask a small central team for prompts, evaluation, data access and deployment changes.
Trigger: Demand grows while intake, service classes and self-service boundaries remain undefined.
Symptom: Delivery waits grow and teams bypass the central process to meet local deadlines.
Decision at risk: Which work should remain centralized and which capabilities need explicit reusable interfaces or local ownership.
Hypothesis A: Demand exceeds the central team's sustainable service capacity.
Hypothesis B: Requests are underspecified and repeated clarification dominates handling time.
Hypothesis C: Governance approvals, not technical implementation, are the actual blocked dependency.
Discriminating input: Request arrival/completion timestamps, wait states, work classes and blocked-reason history.
Expected signal: Queue age grows in a defined work class, with bottleneck time localized to one ownership boundary.
Negative result: A busy team does not prove capacity shortage if work is waiting on external decisions.
Safe diagnostic: Analyze aggregate flow and request contracts rather than ranking individual workers by superficial throughput.
Immediate containment: Limit concurrent commitments, publish priority rules and provide explicit escalation for critical work.
Mitigation risk: Self-service without guardrails can replicate security and evidence gaps across teams.
Correction: Define a platform service contract, supported paved paths and clear local versus central responsibilities.
Prevention: New controls are designed with service demand, ownership and support capacity in view.
Verification oracle: A product team can complete a supported standard path without undocumented central-team intervention.
Boundary test: Novel high-risk cases still escalate; self-service is not permission to bypass risk review.
Recovery: Reprioritize the existing queue transparently and retire duplicate or obsolete requests with requester agreement.
Owner: Platform leader defines service boundaries; product leaders own local outcomes and request quality.
Escalation: Transformation sponsor when central demand exceeds the funded operating model.
Evidence gap: No actual queue, staffing plan or measured handling distribution exists here.
Cost consequence: Centralization saves duplication only when coordination and waiting costs remain acceptable.
Decision consequence: A team topology is a flow design with queues and interfaces, not an org-chart aesthetic.
Related sheet: [Module 53 team topology](../53_communication_leadership_and_ai_transformation/INDEX.md).
Source support: [Google SLO methodology](https://sre.google/workbook/implementing-slos/) supports explicit service expectations, not a universal team structure.

<a id="p09-023"></a>
### P09-023 — Approval responsibility does not include repair authority
Category: VIOLATED_ASSUMPTION; subsystem: governance-accountability-to-incident execution.
Provenance: HYPOTHETICAL; the failure is organizational mechanism analysis, not an incident report.
Environment: A release has product, security, legal and executive approvals recorded in a responsibility matrix.
Trigger: A user-impacting failure requires rapid suspension, customer correction and funding for remediation.
Symptom: Several people signed the release, but none can authorize all necessary repair actions.
Decision at risk: Who can stop, investigate, compensate and restore the complete workflow.
Hypothesis A: The matrix assigns accountability without actual decision rights or resources.
Hypothesis B: Rights exist, but responders cannot locate the current owner or delegate out of hours.
Hypothesis C: Repair spans incompatible contracts or systems that no single role controls.
Discriminating input: Incident tabletop, access grants, spend authority, escalation contacts and recovery dependencies.
Expected signal: A concrete repair step stalls because no identified role has the required authority.
Negative result: A named owner who can answer the phone still may lack technical or financial execution capability.
Safe diagnostic: Conduct a tabletop with inert artifacts and record who can perform each required action.
Immediate containment: Assign a temporary incident commander and explicit bounded emergency authorities under organizational policy.
Mitigation risk: Broad emergency grants can create new exposure; scope, expire and audit them.
Correction: Align accountable roles with stop, repair, customer communication and funding authority.
Prevention: Operational readiness requires a successful authority-path walkthrough, not only a signed matrix.
Verification oracle: Every critical recovery action has an available authorized executor and a tested escalation route.
Boundary test: Absence of the primary owner activates a documented delegate rather than indefinite delay.
Recovery: Reconcile emergency actions, remove temporary grants and document unsupported repair limitations.
Owner: Business service owner owns end-to-end repair; specialist owners control their bounded enforcement domains.
Escalation: Executive sponsor when repair requires cross-budget or cross-contract authority.
Evidence gap: No actual matrix, role grant or exercise record is supplied here.
Cost consequence: Repair reserves and operational staffing are part of sustainable product cost.
Decision consequence: Accountability without authority is a reporting label, not an operating mechanism.
Related sheet: [Module 53 operating model](../53_communication_leadership_and_ai_transformation/INDEX.md).
Source support: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) is contextual risk-management guidance, not proof that any role is empowered.

<a id="p09-024"></a>
### P09-024 — Training attendance is mistaken for usable capability
Category: CONCEPT_CONFUSION; subsystem: learning-program-to-transformation outcome.
Provenance: HYPOTHETICAL; no real training program effectiveness is asserted.
Environment: A transformation dashboard tracks workshop completion and issued certificates.
Trigger: Leadership treats attendance targets as evidence that teams can safely perform AI-enabled work.
Symptom: Completion is high, yet teams cannot diagnose errors, apply policy boundaries or recover failed tasks.
Decision at risk: Whether the workforce has the specific capability required for the proposed operating scope.
Hypothesis A: The measurement records exposure to training, not demonstrated task performance.
Hypothesis B: Skills were learned but the actual tools, permissions or workflows differ from training conditions.
Hypothesis C: Managers provide no time or incentive to practice, so skills decay or remain unused.
Discriminating input: Role-specific practical tasks, tool access, observed task reasoning and supported practice opportunities.
Expected signal: Attendance and performance diverge on tasks that represent the real decision boundary.
Negative result: A single assessment does not establish sustained competence under workload or changing versions.
Safe diagnostic: Use synthetic scenarios and constructive feedback rather than hidden surveillance or fabricated performance claims.
Immediate containment: Limit unsupported high-consequence use and provide a usable manual/escalation path.
Mitigation risk: Blanket restriction can discourage learning; distinguish supervised practice from unsupervised authority.
Correction: Define capability outcomes, realistic practice, feedback and reassessment tied to task/version changes.
Prevention: Training plans specify what learners must derive, perform, reject and diagnose, not merely attend.
Verification oracle: Learners complete representative tasks and explain failure boundaries under changed constraints.
Boundary test: A learner who can repeat terminology but cannot reject an unsafe action is not cleared for that action.
Recovery: Target support to observed skill gaps and update the rollout envelope without inventing a completion rate.
Owner: Capability lead designs assessment; business owner defines required competence and grants actual work authority.
Escalation: Transformation sponsor when schedules assume capability that the assessment does not support.
Evidence gap: No learner records, assessment rubric or longitudinal outcomes are available here.
Cost consequence: Practice and assessment require time, facilitation and tool access beyond workshop delivery.
Decision consequence: Learning activity is an input; capability and business outcomes require separate evidence.
Related sheet: [Module 53 learning and change](../53_communication_leadership_and_ai_transformation/INDEX.md).
Source support: [Human-AI interaction research](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/) motivates user-centered evaluation, not a training efficacy claim.

<a id="p09-025"></a>
### P09-025 — Precise ROI hides an assumption that reverses the decision
Category: NUMERICAL_EDGE; subsystem: analytical-evidence-to-executive narrative.
Provenance: HYPOTHETICAL; no investment recommendation for a real organization is made.
Environment: A board paper reports a single return estimate to two decimal places.
Trigger: Adoption, useful completion, review cost or realization assumptions are weak but their uncertainty is omitted.
Symptom: Small plausible changes reverse the recommendation despite the apparent precision of the headline number.
Decision at risk: Whether the investment is robust, conditional or worth only a bounded evidence-gathering tranche.
Hypothesis A: The model uses optimistic point inputs without sensitivity or switching values.
Hypothesis B: The formula is correct but the narrative confuses forecast, target and realized result.
Hypothesis C: Costs and benefits use different time windows or incompatible accounting boundaries.
Discriminating input: Full input ledger, units, windows, evidence labels and one-at-a-time/joint sensitivities.
Expected signal: A plausible parameter change crosses the investment threshold while the paper presents an unconditional approval.
Negative result: Stability to one parameter does not establish robustness to correlated or structural uncertainty.
Safe diagnostic: Recompute the declared model transparently; do not substitute a different model without explaining the change.
Immediate containment: Qualify the recommendation and disclose the assumptions that can reverse it.
Mitigation risk: Overloading executives with unprioritized ranges can obscure rather than improve the decision.
Correction: Present the decision, dominant uncertainty, switching values, downside limits and evidence-gathering option.
Prevention: Executive templates require baseline, counterfactual, sensitivity and a clear evidence/stage distinction.
Verification oracle: A reviewer can identify which input change would reverse the recommendation and who owns measuring it.
Boundary test: If a hard constraint fails, no favorable sensitivity rescues eligibility.
Recovery: Reopen approval when realized inputs cross the declared recheck trigger rather than waiting for a large loss.
Owner: Analytical owner maintains the model; executive decision owner accepts the residual uncertainty explicitly.
Escalation: Finance/governance owner when a forecast is being represented as a measured outcome.
Evidence gap: No actual board paper, price quote or measured benefit exists in this exercise.
Cost consequence: Buying information can be preferable to committing the full portfolio under fragile assumptions.
Decision consequence: Precision of arithmetic is not certainty of inputs or validity of the decision model.
Related sheet: [Module 53 executive communication](../53_communication_leadership_and_ai_transformation/INDEX.md).
Source support: [Green Book appraisal](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026) explicitly distinguishes uncertainty and switching-value analysis.

<a id="p09-026"></a>
### P09-026 — Model rollback leaves new tools, policies and evidence behind
Category: VIOLATED_ASSUMPTION; subsystem: release-bundle-to-recovery consistency.
Provenance: HYPOTHETICAL; no particular deployment platform is assumed.
Environment: A release changes model, prompt, tool schema, retrieval index and policy configuration.
Trigger: A quality regression leads operators to restore only the previous model alias.
Symptom: The rolled-back service still behaves differently, fails tool calls or produces evidence tied to incompatible versions.
Decision at risk: Whether rollback restores a coherent and authorized service, not merely a model checkpoint.
Hypothesis A: Release components are versioned independently without a compatibility manifest.
Hypothesis B: Schema/data migrations are irreversible or not backward compatible with the older bundle.
Hypothesis C: Cached sessions continue using the new prompt/tools despite the restored model alias.
Discriminating input: Actual per-request component versions, migration state, session pinning and approved bundle manifest.
Expected signal: A post-rollback request combines old and new component identities outside a declared compatible set.
Negative result: Matching bundle identities do not establish correctness; the older version can still be unsuitable for new data.
Safe diagnostic: Replay synthetic representative requests in an isolated compatibility environment before live rollback.
Immediate containment: Stop unsafe effects and route to a known coherent authorized path if one exists.
Mitigation risk: Blind rollback can corrupt state or reintroduce a security flaw that motivated the new release.
Correction: Define atomic release identities with compatibility rules, migration boundaries and rollback/roll-forward plans.
Prevention: Approval evidence binds to the whole bundle and includes recovery acceptance cases.
Verification oracle: Every request after recovery reports a permitted bundle combination and passes its schema/action contract.
Boundary test: An incompatible state migration forces an explicit roll-forward or repair decision, not a false rollback-success claim.
Recovery: Reconcile in-flight tasks, caches and persisted state before declaring the service restored.
Owner: Release/platform owner coordinates bundle state; policy owner verifies restored authorization behavior.
Escalation: Incident commander when no coherent safe bundle can operate on the current state.
Evidence gap: No real deployment manifest, migration script or recovery execution is provided here.
Cost consequence: Recovery compatibility work is a lifecycle cost, not an optional postlaunch enhancement.
Decision consequence: The deployment unit should match the unit of reasoning, evaluation and approval.
Related sheet: [Module 50 release evidence](../50_responsible_ai_governance_compliance_and_audits/INDEX.md).
Source support: [NIST SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) supports integrating secure practices across the lifecycle, not this platform's atomicity.

<a id="p09-027"></a>
### P09-027 — The kill switch stops intake but not in-flight effects
Category: SCALE_FAILURE; subsystem: stop-authority-to-effect-reconciliation.
Provenance: HYPOTHETICAL; no harmful live action was used as a teaching test.
Environment: An agent accepts tasks, queues tool work and commits external actions asynchronously.
Trigger: Operators disable the user interface or primary worker during an incident.
Symptom: Queued or separately executing actions continue after the team reports the system stopped.
Decision at risk: What the stop mechanism actually prevents and which prior actions need cancellation or compensation.
Hypothesis A: The switch controls only new intake, not queued execution or delegated credentials.
Hypothesis B: Cancel requests race with commits, and task state does not distinguish attempted from committed effects.
Hypothesis C: Another worker or schedule continues acting under a different control path.
Discriminating input: Intake state, queue leases, worker identities, credential validity and effect commit timestamps.
Expected signal: A synthetic action commits after intake stop through an execution path not covered by the switch.
Negative result: No new UI traffic does not establish absence of in-flight or scheduled work.
Safe diagnostic: Use inert effect sinks and controlled queues; do not test cancellation by issuing real irreversible actions.
Immediate containment: Revoke or block effect authority at the service boundary and freeze/reconcile queued work.
Mitigation risk: Abrupt revocation can leave partial transactions or required compensations incomplete.
Correction: Define stop semantics across intake, planning, dispatch, execution and committed-effect reconciliation.
Prevention: Every effect-capable component participates in incident control and reports its outstanding work.
Verification oracle: After the declared enforcement point, no new unauthorized commit occurs; prior commits remain identifiable.
Boundary test: A commit racing with cancellation has a documented authoritative state and repair path.
Recovery: Resume only reconciled tasks with valid authority; do not blindly replay the entire queue.
Owner: Incident commander invokes stop; effect-service owner enforces authority; business owner authorizes compensation.
Escalation: External-system owner when cancellation or repair requires rights beyond the AI platform.
Evidence gap: No actual queue semantics, cancellation API or effect ledger is established here.
Cost consequence: Reconciliation and compensation consume labor and can create contractual liabilities not reflected in call cost.
Decision consequence: Stopping generation is not the same as stopping business effects.
Related sheet: [Module 49 incident response](../49_ai_security_privacy_safety_and_red_teaming/INDEX.md).
Source support: [Approval focused brief](../../00_SOURCE_COVERAGE_AGENTS.md#agent-trust-approvals) motivates explicit execution surfaces and blast radius.

<a id="p09-028"></a>
### P09-028 — Low complaint counts reflect silence rather than low harm
Category: VIOLATED_ASSUMPTION; subsystem: leadership-incentive-to-risk-evidence.
Provenance: HYPOTHETICAL; no real organization's reporting culture is characterized.
Environment: Teams are rewarded for adoption, launch speed and a low incident count.
Trigger: Users or employees expect blame, delay or loss of access when reporting AI failures.
Symptom: Official complaints remain low while independent interviews or downstream corrections reveal recurring problems.
Decision at risk: Whether the observed report count is a valid indicator of underlying adverse outcomes.
Hypothesis A: Reporting incentives or friction suppress detection and disclosure.
Hypothesis B: Harm is genuinely rare and independent reports describe isolated misunderstandings.
Hypothesis C: Different teams classify the same failure differently, fragmenting the count.
Discriminating input: Anonymous/independent sampling, reporting funnel, classification rules and downstream repair records.
Expected signal: Independent inquiry identifies materially more eligible failure events than the official reporting channel.
Negative result: A discrepancy does not by itself identify intent; friction and definition mismatch can explain it.
Safe diagnostic: Protect confidentiality and use authorized aggregate findings rather than exposing individual reporters.
Immediate containment: Provide a protected escalation route and suspend punitive interpretations of raw report counts.
Mitigation risk: An untriaged new channel can overwhelm responders or collect unnecessary sensitive data.
Correction: Separate detection incentives from blame, standardize event definitions and fund triage/repair capacity.
Prevention: Leadership reviews reporting coverage and response quality, not only a target of fewer reports.
Verification oracle: A synthetic report reaches an authorized responder, gets a trackable decision and cannot disappear silently.
Boundary test: An increase in reports after improving access is not automatically interpreted as worsening underlying safety.
Recovery: Reassess prior low-risk claims and investigate material unreported harms through appropriate process.
Owner: Governance/risk owner maintains reporting; leadership owns incentives; operations owns response execution.
Escalation: Independent assurance or ethics channel when ordinary management controls the disputed reporting incentives.
Evidence gap: No real survey, report dataset or retaliation evidence is provided here.
Cost consequence: Better detection can initially increase visible remediation cost while improving evidence quality.
Decision consequence: Absence of reports is not evidence of absence when detection and incentives are weak.
Related sheet: [Module 53 responsible adoption](../53_communication_leadership_and_ai_transformation/INDEX.md).
Source support: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) provides voluntary contextual risk-management framing, not culture measurement.

<a id="p09-029"></a>
### P09-029 — Adoption rates exclude the people who cannot use the interface
Category: VIOLATED_ASSUMPTION; subsystem: UX-access-to-transformation measurement.
Provenance: HYPOTHETICAL; no accessibility audit or user result is claimed.
Environment: A product reports active users divided by provisioned users as its adoption headline.
Trigger: Provisioning, language, assistive-technology compatibility or workflow access prevents some eligible people from participating.
Symptom: Adoption looks strong among provisioned users while the full eligible population cannot complete the intended task.
Decision at risk: Whether the product provides usable value to the stated population and whether exclusions are acceptable.
Hypothesis A: The denominator omits eligible people blocked before provisioning or first use.
Hypothesis B: Users have access but choose the manual workflow because the AI product lacks value.
Hypothesis C: The task is unavailable or irrelevant for some roles, requiring a narrower prospectively defined population.
Discriminating input: Eligibility frame, provisioning funnel, accessibility/task study and role-specific need.
Expected signal: Material eligible segments fail before the active-user denominator is formed.
Negative result: Equal provisioning does not establish equal usability, useful completion or benefit.
Safe diagnostic: Conduct consented accessibility and workflow research with representative participants and appropriate accommodations.
Immediate containment: Offer a usable non-AI or assisted path and disclose the measured population boundary.
Mitigation risk: A separate manual path can become slower or stigmatizing; evaluate service parity and user control.
Correction: Design inclusive interaction and measurement from eligibility through useful completion, not only login events.
Prevention: Product requirements name accessibility, language and alternative-path needs before rollout.
Verification oracle: Representative eligible participants can complete the real task or use an equivalent supported alternative.
Boundary test: A zero count in an excluded segment is not evidence of zero demand or zero harm.
Recovery: Reconcile adoption claims and prioritize barriers according to consequence and affected population.
Owner: UX/product owner owns task accessibility; transformation owner owns equitable access and support resources.
Escalation: Accessibility/domain/legal owner when exclusion conflicts with binding organizational or applicable requirements.
Evidence gap: No actual user population, accessibility test or legal applicability finding is established here.
Cost consequence: Inclusive design, support and alternative paths are part of product cost rather than optional extras.
Decision consequence: Adoption among reachable users does not establish value for the population promised.
Related sheet: [Module 51 human-AI UX](../51_ai_product_management_ux_and_human_ai_collaboration/INDEX.md).
Source support: [Human-AI interaction guidelines](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/) support design inquiry, not accessibility certification.

<a id="p09-030"></a>
### P09-030 — Retirement ends the subscription but leaves authority and data alive
Category: SCALE_FAILURE; subsystem: procurement-exit-to-security/lifecycle closure.
Provenance: HYPOTHETICAL; no actual vendor exit or data-return completion is reported.
Environment: A product is retired after weak value, a replacement decision or contract expiry.
Trigger: The team cancels its primary license but does not reconcile credentials, exports, indexes and scheduled integrations.
Symptom: Old jobs continue running, data remains accessible or costs continue after leadership reports the product closed.
Decision at risk: Whether retirement has removed ongoing exposure and transferred needed business capability safely.
Hypothesis A: The asset inventory omits derivative stores, service identities or downstream processors.
Hypothesis B: Contractual minimums or retention duties continue legitimately but were not communicated.
Hypothesis C: A replacement integration still depends on the old service and keeps it active.
Discriminating input: Asset/credential inventory, processor records, contract exit terms, scheduled work and replacement dependencies.
Expected signal: An allegedly retired service still has usable grants, reachable in-scope data or active scheduled work.
Negative result: Cancelled billing alone does not establish deletion, revocation or workflow continuity.
Safe diagnostic: Inspect authorized metadata and synthetic access checks; avoid redisclosing data during export validation.
Immediate containment: Disable unnecessary authority and scheduling while preserving required records under controlled access.
Mitigation risk: Abrupt closure can destroy needed records or interrupt a dependent business workflow.
Correction: Treat exit as an end-to-end migration and lifecycle plan with acceptance evidence for data, authority, cost and service continuity.
Prevention: Procurement requires exit rights and ownership before selection; architecture records downstream dependencies.
Verification oracle: Retired identities cannot perform effects, scheduled work is reconciled and retained data has a declared purpose/access state.
Boundary test: A required retention exception is documented and isolated rather than mislabeled as complete deletion.
Recovery: Resolve residual bills, grants and stores and record any third-party limitations that cannot be independently verified.
Owner: Product service owner coordinates retirement; procurement, security, data and records owners close their bounded obligations.
Escalation: Legal/commercial owner when exit terms prevent the assumed data return, deletion or migration timeline.
Evidence gap: Actual terms, processor confirmations and asset records are unavailable in this exercise.
Cost consequence: Migration, export, retained evidence and decommissioning belong in lifecycle TCO.
Decision consequence: A product is not retired until its effects, authority, data and dependencies have a controlled end state.
Related sheet: [Module 52 vendor lifecycle](../52_ai_economics_strategy_procurement_and_vendor_management/INDEX.md).
Source support: [NIST SSDF](https://csrc.nist.gov/pubs/sp/800/218/final) supports lifecycle security practices, not vendor-specific exit guarantees.

**Takeaway:** The thirty issues are distinct mechanisms; each requires its own discriminating signal and bounded correction.

## 9. Complexity & Cost Analysis

### 9a. Diagnostic operation model

Let N be the number of task records, A the number of attempt records, E the number of effect records and R the number of release bundles in the declared window.
Let K be the number of mandatory eligibility predicates per candidate route and V the number of candidate routes.
These are counts, not rates; they must not be substituted for cases/hour in P09-QUEUE.
Assume fixed-size keys and average constant-time hash lookup only for the following expected-operation discussion.
A task-attempt-effect reconciliation can use expected O(N+A+E) processing with hash-indexed joins when keys are well formed.
Auxiliary memory can be O(N) for a task index, plus retained output and any materialized unmatched records.
External sorting or database execution changes I/O and memory behavior; the bound is not a deployment benchmark.
A naive nested comparison of all task and attempt rows can cost O(N*A), making an avoidable diagnostic bottleneck.
Eligibility evaluation costs O(V*K) predicate checks before scoring if all predicates are evaluated for every route.
Short-circuiting can reduce work but must not omit required audit reasons or evaluate predicates with unsafe side effects.
Release compatibility review depends on declared combinations, not merely R independent labels; avoid testing every arbitrary combination when a manifest restricts valid bundles.
Human review cost scales with sampled cases and handling time, not only log volume.
Privacy minimization can reduce stored content while preserving task IDs, aggregate counters and authorized evidence pointers.
A hash digest helps identify content equality only under its cryptographic and canonicalization assumptions; it does not prove semantic authorization.
Retaining all raw content indefinitely is not a necessary consequence of traceability and may create additional risk.

| Operation | Sizes | Time / work | Space / I/O | Limitation |
|---|---|---|---|---|
| Task/attempt/effect reconciliation | N,A,E | Expected linear under indexed joins | Task index plus streamed records | Key skew and missing IDs require explicit handling |
| Eligibility checks | V,K | O(V*K) predicate evaluations | Route metadata | Predicate lookup latency is not covered by operation count |
| Stratified review | Sample sizes by slice | Sum of actual review handling work | Controlled sample records | Reviewer agreement and label quality need evidence |
| Historical evidence resolution | R manifests and linked artifacts | Depends on artifact graph | Immutable identifiers and bounded access | Working links do not establish claim correctness |
| Retirement reconciliation | Asset and grant inventory | Inventory traversal plus external confirmations | Audit trail | Third-party deletion may remain externally attested |

### 9b. Worked diagnostic budget

**Evidence:** ILLUSTRATIVE labor budget, not a market rate, actual invoice or measured handling-time distribution.
Assume 120 sampled tasks and eight minutes of review per task under a fixed rubric.
Multiplication gives 960 reviewer-minutes, which is sixteen reviewer-hours.
At an illustrative internal planning rate of 75 USD/hour, the initial review allocation is 1,200 USD.
Independent arithmetic check: 120 tasks times 8/60 hours/task times 75 USD/hour equals 1,200 USD.
If 20% of tasks need ten additional minutes of adjudication, 24 tasks add 240 minutes or four hours.
That adjudication adds 300 USD at the same illustrative rate, bringing review labor to 1,500 USD.
The estimate excludes recruitment, secure sample preparation, engineering, manager time, tooling and any external expert fees.
If handling time doubles because source evidence is harder to inspect, initial review labor doubles to 2,400 USD before adjudication.
If two reviewers independently inspect every task, the initial review allocation also doubles; do not conflate that with adjudication of disagreements.
The budget should be tied to the decision it can change; a cheap check that cannot distinguish hypotheses is not necessarily cost-effective.
Conversely, a large diagnostic program is unnecessary when an exact authorization mismatch already disqualifies the action.
The appropriate sample design depends on the target claim and failure consequences, not the convenience of a round count.
Critical prohibited effects may require exhaustive checking of a known bounded set rather than averaging them into a quality score.
These budget calculations establish resource consequences of assumptions, not review accuracy or causal validity.
**Takeaway:** Diagnose with explicit operation and labor models, and keep evidence quality separate from the money spent collecting it.

## 10. Key Theorems / Results Reference

| Result | Type | Statement and conditions | Practical implication | Does not imply | Support |
|---|---|---|---|---|---|
| P09-COST | Definition | Included cost divided by positive useful-task count within one boundary | Compare complete task economics | Equal per-task consumption or realized savings | Section 4 derivation |
| P09-QUEUE | Conservation model | Constant arrivals minus service change backlog while work remains, reflected at zero | Identify a capacity mismatch | Tail latency or reviewer accuracy | Section 4 assumptions |
| P09-ZERO | Exact zero-failure endpoint under Bernoulli model | Invert probability of zero failures for fixed independent trials | Quantify narrow sampling uncertainty | Adversarial safety guarantee | Section 4 inversion |
| Eligibility invariant | Conditional mechanism argument | Effect service independently evaluates exact-action authority before commit | Prevent score-based override of declared predicates | Correctness, legal applicability or compromised-enforcer resistance | Section 4 trace |
| SLI good/total structure | Sourced methodology | User-centered good events divided by declared total events | Make measurement boundary explicit | Universal SLO threshold | Google SRE workbook, Chapter 2 |

No theorem in this file proves that an AI product is safe, fair, lawful or valuable in every deployment.
The Bernoulli endpoint is not a robust bound for arbitrary dependence or adaptive adversaries.
A conservation identity remains true when counts are trustworthy, but missing arrivals or duplicate completions can corrupt its measured inputs.
A mechanism invariant depends on correct enforcement, identity binding and coverage of all effect paths.
A statistical confidence procedure cannot repair a biased sampling frame by increasing nominal sample size alone.
A positive forecast is a conditional projection, not an empirical finding about realized business outcomes.

### Primary reference and claim ledger

| Source ID | Primary source | Publication / version | Claim supported here | Access date | Limitation / trigger |
|---|---|---|---|---|---|
| S-01 | [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | AI RMF 1.0 released 2023-01-26; landing page accessed as current | Voluntary risk-management framing across AI lifecycle | 2026-11-08 | Recheck revision; not legal certification or outcome evidence |
| S-02 | [NIST SP 800-218](https://csrc.nist.gov/pubs/sp/800/218/final) | SSDF 1.1, February 2022 | Integrating secure-development practices and supplier communication | 2026-11-08 | Not proof of a product's security or supplier implementation |
| S-03 | [Implementing SLOs](https://sre.google/workbook/implementing-slos/) | Site Reliability Workbook, Chapter 2 | Good-event/total-event measurement and stakeholder-owned objectives | 2026-11-08 | Examples are not imported as universal targets |
| S-04 | [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/) | Amershi et al., CHI 2019 | Human-AI interaction is a design/evaluation concern; guidelines are a practitioner resource | 2026-11-08 | Does not validate this product, accessibility or reviewer accuracy |
| S-05 | [The Green Book 2026](https://www.gov.uk/government/publications/the-green-book--2/the-green-book-2026) | Updated 2026-10-28 | Appraisal compares options, uncertainty and switching values; component benefits need not be separately apportioned | 2026-11-08 | Public-sector appraisal context; no private accounting or legal applicability inferred |
| S-06 | [Visible inventory](../../../Templates/directory.txt) | Local reconstructed source tree | Topic provenance and unresolved hidden content | 2026-11-08 | Names alone do not establish original lesson content |

The NIST landing page states that AI RMF 1.0 is being revised; this file does not characterize an unreleased replacement.
The Green Book source is used for the stated appraisal concepts, not for any discount rate or regulatory requirement in this fictional product.
Original source bodies, hidden code, proprietary operations and company outcome measurements remain unavailable.
Source access is not independent reproduction of a source's empirical study.
No invented citation, customer name, deployment gain or reviewer approval is needed to teach the bounded mechanisms above.
**Takeaway:** Treat identities, models, methodology and evidence claims as different kinds of support with different boundaries.

## 11. Interview / Exam Questions

### Q1 — Level 1: What is the unit of a useful-task cost?
Prompt: A team reports 12,000 USD, 6,000 attempts and 3,000 useful completions for one month.
Clarify: Costs and completions must use the same window and declared population; ask what costs are included.
Answer: P09-COST gives 12,000/3,000 = 4 USD/useful task, while 12,000/6,000 = 2 USD/attempt is a separate metric.
Reasoning: An attempt is not a completed user outcome and repeated failures must not enlarge the useful denominator.
Sanity check: 4 USD/task times 3,000 tasks reconstructs the total cost.
Acceptable alternative: Report marginal task cost as well if the cost boundary is explicitly different and labeled.
Follow-up: If no task is useful, report incurred cost and an undefined useful-task ratio rather than dividing by zero.
Red flag: Calling the 2 USD figure useful-task cost silently changes the metric definition.
Scoring: 0 wrong denominator; 1 arithmetic only; 2 correct with boundary; 3 distinguishes marginal/full cost and zero case.
Answer support: Sections 3–5, P09-COST and P09-FULL.

### Q2 — Level 1: Does approval of a conversation authorize a tool effect?
Prompt: A reviewer approved a chat transcript and the tool later changes the destination.
Clarify: Ask which exact object, identity, scope, version and validity interval the approval covers.
Answer: Not automatically; effect authority must bind to the documented action semantics and the material current payload.
Reasoning: A conversation can contain many possible actions, and approval of earlier text does not establish permission for a changed effect.
Trace check: Mutate the destination in a synthetic request and require the prior approval to fail if the change is material under policy.
Acceptable alternative: A predeclared policy may authorize a safe subset transformation, but that rule must be explicit and independently enforced.
Follow-up: If an effect already committed, reapproval does not undo it; reconcile and repair through the authorized path.
Red flag: Trusting the model's claim that the reviewer would have approved the changed action.
Scoring: 0 assumes blanket authority; 1 notices change; 2 binds exact action; 3 includes time and committed-effect recovery.
Answer support: P09-002 and Section 4 mechanism trace.

### Q3 — Level 1: What does zero failures in a safety test establish?
Prompt: A fixed suite of 100 independent trials has zero failures under a binary reliable label.
Clarify: Confirm fixed failure probability, independence, test population and whether the outcome is illustrative or measured.
Answer: Under that model, the one-sided 95% zero-failure upper endpoint is 1-0.05^(1/100), approximately 2.9513%.
Reasoning: The zero-failure observation has probability (1-p)^100; inversion gives the endpoint.
Sanity check: Substituting the endpoint recovers probability 0.05 for zero failures.
Acceptable alternative: State a clearly labeled approximation near 3/n only if its approximation status and assumptions are retained.
Follow-up: Clustered or adaptive attacks invalidate the simple independent-trial interpretation and require a different evidence argument.
Red flag: Reporting population safety as 100% or presenting the endpoint as the true failure rate.
Scoring: 0 claims zero risk; 1 finite-sample caution; 2 correct endpoint/conditions; 3 distinguishes adversarial coverage.
Answer support: P09-ZERO and P09-004/P09-009.

### Q4 — Level 2: Diagnose the overloaded review workflow.
Prompt: An empty queue receives 90 cases/hour, completes 72 cases/hour and runs for eight hours at constant rates.
Clarify: Assume no losses, retries, class changes or idle service while work waits; distinguish the fluid model from a latency distribution.
Answer: Backlog grows at eighteen cases/hour and ends at 144 cases; arrivals are 720 and completions 576.
Reasoning: Conservation of cases gives arrivals minus completions, consistent with P09-QUEUE.
Sanity check: Both 18*8 and 720-576 produce 144 cases.
Acceptable alternative: An event-level simulation can handle variable arrivals if the actual trace is supplied; no such trace is given here.
Follow-up: At service 96 and arrivals 90, the 144-case backlog needs 24 hours to drain under the new constant-rate assumptions.
Red flag: Declaring the system stable because daily averages look acceptable while the stated rates imply accumulation.
Scoring: 0 incorrect units; 1 correct growth; 2 full calculation; 3 recovery and tail-latency caveat.
Answer support: Sections 4–6 and P09-011.

### Q5 — Level 2: Why can a cheaper provider produce a more expensive outcome?
Prompt: One route lowers per-attempt price but increases attempts and review work for the same useful tasks.
Clarify: Ask for task-linked attempts, complete billable quantities, review labor and useful completion definitions.
Answer: Compare P09-FULL/P09-COST over the same task mix; cheaper attempts can be outweighed by more attempts and human recovery.
Reasoning: Unit price affects one cost component, whereas full useful-task cost also depends on failures and the useful denominator.
Sanity check: Ensure every invoice row maps to the declared window and no failed work disappears from the numerator.
Acceptable alternative: For a short-run choice, show marginal economics separately from full allocated economics.
Follow-up: Add a minimum commitment and evaluate low-volume scenarios before assuming variable-price savings are realizable.
Red flag: Ranking solely by published token price or treating human review as free.
Scoring: 0 call-price only; 1 mentions retries; 2 complete task ledger; 3 marginal/full boundary and contract sensitivity.
Answer support: P09-016 and P09-019.

### Q6 — Level 2: Identify the hidden denominator change.
Prompt: A dashboard's success rate rises after timed-out tasks are reclassified as ineligible at the end of the task.
Clarify: Ask when eligibility was determined and whether the new rule was prospectively applied to the population.
Answer: Reconstruct the original eligible cohort using decision-time rules; report timeout outcomes rather than excluding them after observation.
Reasoning: Outcome-dependent exclusion can create an apparent improvement without any user benefit.
Sanity check: Reconcile arrival, eligibility, attempt, completion, usefulness and exclusion counts by unique task ID.
Acceptable alternative: A genuine prospective scope change can be valid, but rates must be labeled as different populations.
Follow-up: If the experiment randomized users, check whether exclusions also break the intended comparison or assignment balance.
Red flag: Explaining the rise only as product quality improvement without examining inclusion logic.
Scoring: 0 accepts headline; 1 suspects bias; 2 reconstructs cohort; 3 handles prospective scope and causal validity.
Answer support: P09-008 and Section 3 definitions.

### Q7 — Level 3: Select a route when utility and authority disagree.
Prompt: Route A has the best quality/cost score but fails a mandatory destination predicate; route B is eligible but more expensive.
Clarify: Confirm the predicate's authority, currency and exception process; do not invent an exception.
Answer: Exclude A before ranking; choose B only if it satisfies all remaining constraints and objectives, otherwise abstain or use the authorized baseline.
Reasoning: A hard condition defines the feasible set rather than a preference that utility can offset.
Sanity check: Give A an arbitrarily high score in a synthetic replay; it must remain unselectable.
Acceptable alternative: Seek an authorized prospective policy change if the business requires it, without sending data before approval.
Follow-up: If B fails, the fallback must reapply the original predicates and remaining budgets.
Red flag: Increasing the security penalty until the preferred route happens to lose; another score can still override it.
Scoring: 0 score override; 1 notices constraint; 2 filter/abstain; 3 authority and fallback semantics.
Answer support: P09-001 and P09-017.

### Q8 — Level 3: Diagnose high approval with low error detection.
Prompt: Reviewers approve almost everything, but the task population is believed to be mostly correct.
Clarify: High approval alone is ambiguous; ask about error prevalence, evidence access, time budget and reviewer authority.
Answer: Compare competing causes using safe blinded synthetic error exercises and evidence/time variations before concluding automation bias.
Reasoning: Reviewers may be appropriately approving correct work, may lack information, or may be anchored by recommendations.
Sanity check: Verify seeded errors match real error mechanisms and the exercise does not create live harmful effects.
Acceptable alternative: Expert review of naturally occurring labeled cases can work if sampling and truth labels are reliable.
Follow-up: If review is overloaded, reduce intake or redesign case preparation rather than adding another mandatory checkbox.
Red flag: Blaming reviewers solely from approval percentage or treating human involvement as proof of control.
Scoring: 0 single-cause blame; 1 multiple hypotheses; 2 discriminating safe check; 3 capacity, authority and measurement boundaries.
Answer support: P09-011 and P09-012.

### Q9 — Level 3: Defend a pilot claim under a changed population.
Prompt: Volunteers report faster work; leadership wants the same gain from every employee next month.
Clarify: Identify target population, recruitment, task mix, support and all affected labor before making an inference.
Answer: The volunteer result supports only its observed context; design a representative bounded evaluation before extrapolating to unobserved roles and tasks.
Reasoning: Selection, support differences and downstream work can explain a deployment gap without proving the tool has no value.
Sanity check: Report eligible-participant flow and unmeasured segments, and distinguish association from causal effect.
Acceptable alternative: A staged rollout with defensible assignment and guardrails can gather representative evidence if exposure is permissible.
Follow-up: If expansion is contractually committed already, disclose the evidence gap and revise the rollout envelope rather than inventing a benefit.
Red flag: Multiplying a volunteer time-saving average by total headcount without a realization or transport argument.
Scoring: 0 extrapolates directly; 1 notes bias; 2 representative design; 3 includes workflow cost and decision scope.
Answer support: P09-013 through P09-015.

### Q10 — Level 3: Explain why rollback did not restore the old service.
Prompt: Operators restored an old model alias, but failures persist after a release that also changed tools and retrieval.
Clarify: Obtain the actual per-request component identities and state migration status; a marketing version is insufficient.
Answer: Investigate mixed bundles, incompatible schemas and session caches; restore a coherent authorized combination or choose an explicit repair/roll-forward path.
Reasoning: The evaluated and approved unit is the combined workflow, not one model pointer.
Sanity check: Replay both successful and failed synthetic requests against the candidate recovery bundle.
Acceptable alternative: If rollback is unsafe after a migration, freeze effects and roll forward with a reviewed compatible correction.
Follow-up: Reconcile in-flight effects before replay so recovery does not duplicate external actions.
Red flag: Declaring recovery complete because the registry alias changed while runtime components remain unverified.
Scoring: 0 model-only reasoning; 1 notices dependencies; 2 coherent-bundle recovery; 3 state/authority/effect reconciliation.
Answer support: P09-026 and P09-027.

### Q11 — Level 3: Communicate a fragile investment recommendation.
Prompt: A precise positive return estimate reverses when review cost rises 50% or useful completion falls.
Clarify: Determine whether the estimate is forecast, target, simulation or observed result and identify the decision threshold.
Answer: Present a conditional recommendation, dominant switching assumptions, downside limits and a bounded evidence-gathering option.
Reasoning: Arithmetic precision does not remove uncertainty in workload, adoption, cost or benefit realization.
Sanity check: The executive reader can state what observation would reverse the decision and who owns that measurement.
Acceptable alternative: Defer investment when a cheap decisive measurement is available and delay is less costly than committing under uncertainty.
Follow-up: If a hard constraint fails, do not rescue the investment with a favorable economic scenario.
Red flag: Hiding all uncertainty in an appendix while the headline says the benefit has been achieved.
Scoring: 0 unconditional ROI; 1 broad caveat; 2 switching/decision structure; 3 evidence value and hard-boundary distinction.
Answer support: P09-025 and Section 5 sensitivity.

### Q12 — Level 3: Verify retirement rather than subscription cancellation.
Prompt: A license is cancelled, but scheduled integrations, service identities and derived data may remain.
Clarify: Identify contracts, retention duties, downstream dependencies and authorized deletion/revocation scope.
Answer: Reconcile assets, grants, scheduled work, retained data and replacement continuity; record externally attested limitations honestly.
Reasoning: Billing closure is only one state transition and does not prove exposure has ended.
Sanity check: Synthetic access attempts fail for retired identities and remaining records have a declared controlled purpose.
Acceptable alternative: A documented retained archive can remain if required and access-restricted; do not label it deleted.
Follow-up: If exit terms block deletion or migration, escalate to the commercial/legal owner and revise the timeline.
Red flag: Reporting complete closure from a cancelled invoice while usable credentials and active jobs remain.
Scoring: 0 billing-only; 1 names data; 2 end-to-end reconciliation; 3 handles retention, authority and continuity constraints.
Answer support: P09-006, P09-023 and P09-030.

### Runnable arithmetic inputs for integration review

The following is an oracle specification, not a claim that a runtime check was executed during authoring.
Use ordinary arithmetic for cost and queue values; use a floating-point exponentiation with adequate precision for the confidence endpoint.
Input cost categories: 2400, 6000, 2600 and 1000 USD; expected total: 12000 USD.
Input useful tasks: 3000; expected full cost: 4 USD/useful task.
Input attempts: 6000; expected separate attempt allocation: 2 USD/attempt.
Input arrivals/service/time: 90 cases/hour, 72 cases/hour and 8 hours; expected backlog: 144 cases.
Input recovery service/arrival: 96 and 90 cases/hour; expected drain time for 144 cases: 24 hours.
Input alpha/n: 0.05 and 100; expected endpoint: approximately 0.0295130496.
Input alpha/n sensitivity: 0.05 and 1000; expected endpoint: approximately 0.0029912495.
Input review sample/time/rate: 120 tasks, 8 minutes/task, 75 USD/hour; expected initial labor: 1200 USD.
Input adjudication: 20% of 120 tasks at 10 minutes/task and 75 USD/hour; expected additional labor: 300 USD.
Input joint sensitivity: total cost 15000 USD and useful completions 2400; expected cost: 6.25 USD/useful task.
No numerical oracle establishes actual workload, reviewer accuracy, safety, legal applicability or production benefit.
**Takeaway:** Strong answers separate the decision, its evidence, its authority and the test that could falsify the preferred explanation.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Thirty cross-module trust-to-value diagnostic issues; all HYPOTHETICAL |
| Core formula | Included cost / positive useful-task count; exact-action eligibility before utility ranking |
| Key assumption | Counts, identities, release versions, authority and population boundaries describe the same workflow |
| Complexity | Indexed task/attempt/effect reconciliation expected O(N+A+E); actual I/O and review labor require measurement |
| Use when | Security, governance, product, finance and leadership metrics disagree at a consequential handoff |
| Avoid when | A diagnostic would itself exceed authorization or a general guide is substituted for legal/domain determination |
| Go deeper | [Module homes](../INDEX.md#modules), then the canonical [case collection](../Case_Studies/INDEX.md) |
| Diagnostic | State competing causes, expected discriminating signal, safe containment and a falsifiable correction oracle |
| Evidence boundary | No observed incidents, executed experiments, vendor guarantees or realized business outcomes are claimed |
