---
collection: "healthcare-life-sciences-biomedical-ai"
file_id: "03"
topic: "Human factors, alert design, and recovery"
level: 2
status: "draft — substantive expansion required before publication"
version: "0.2"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in clinical workflow, alert policy, staffing, user interface, or escalation ownership"
notation_style: "plain-text"
prerequisites: ["01_Clinical_Decision_Support_Evidence_and_Handoff.md"]
pattern_tags: ["human-factors", "alerts", "recovery"]
source_status: "OPEN — no original source body or local workflow study available"
owner_role: "Clinical operations and safety lead"
source_index: "INDEX.md"
---

# Human Factors: Alert Design, Review, and Recovery

> **Purpose:** Design an assistive alert so a person can identify its reason, choose or reject an action, and recover when the alert or its delivery fails.
>
> **Prerequisites:** [clinical decision-support evidence](01_Clinical_Decision_Support_Evidence_and_Handoff.md) and a mapped local workflow.
>
> **Key outcome:** Given an alert rate, review time, and shift capacity, you can calculate burden, define an acknowledgment/recovery invariant, and distinguish alert fatigue from transport failure.
>
> **This file does NOT cover:** clinical policy, autonomous action, or measured effectiveness of a particular interface.
>
> **Level:** 2 — Core Methods.
>
> **Evidence:** DERIVED workload relation; VERIFIED illustrative arithmetic; OPEN workflow evidence.

## 1. File Header

An alert is an interruption plus a decision request, not merely a model output.
Its design must identify the recipient, urgency basis, evidence, available actions, ownership transfer, acknowledgment semantics, timeout, escalation, and audit record.
An “acknowledged” alert is not proof that the proposed action was appropriate.
This sheet teaches workflow engineering and arithmetic; it does not recommend a clinical threshold, treatment, staffing level, or response time.
Every numeric scenario below is constructed and labeled ILLUSTRATIVE.
Arithmetic and state traces are VERIFIED only against the explicit dependency-free oracle recorded in Section 9.
Local policy, recipient authority, observed workload, usability, accessibility, and clinical outcomes remain OPEN.
The source body and local workflow study named in the metadata were unavailable during authoring.
Consequently, no statement here describes a measured hospital deployment or a validated interface.
The design object is an alert episode: a policy decision plus all related presentations, dispositions, escalations, and recovery records.
A raw event is different from an episode because several events may support one reviewable episode.
A presentation is different from an episode because one episode may appear on several devices.
An action is different from an acknowledgment because receipt and completion have separate evidence.
Use stable identifiers for the source events, alert episode, policy version, owner, presentation, and terminal disposition.
These identifiers permit replay without relying on message text or timestamps alone.
The alert contract should be reviewable before implementation and testable after implementation.
Its minimum fields are eligibility, suppression scope, evidence snapshot, recipient rule, permitted dispositions, deadline basis, fallback, and closure rule.
Urgency must have a locally approved rationale; this sheet cannot supply that rationale.
Actionability must be evaluated at presentation time, not only when a model score is produced.
An authorized recipient needs both the organizational role and a feasible action path.
A visible button without authority or supporting resources is not actionability.
A timeout is a workflow trigger, not evidence that the first recipient ignored the alert.
A retry is another delivery attempt, not automatically another alert episode.
A terminal disposition closes work under a documented rule; it does not certify clinical benefit.
The operational owner owns unclosed work even when an automated component fails.
The technical owner owns transport repair, but repair alone cannot discharge clinical or operational ownership.
The safety review owner evaluates policy and evidence changes rather than inferring safety from alert counts.
Audit entries should record state transitions and reasons rather than overwrite a single mutable status.
Append-only history supports reconstruction of who knew what, when, and under which policy version.
Time fields need a common clock convention and an explicit source for each timestamp.
Client presentation time, server receipt time, and action completion time answer different questions.
Late-arriving audit records require reconciliation rules instead of silently rewriting the original sequence.
The recovery design must tolerate duplicate messages, unavailable recipients, stale evidence, and partial interface failure.
It must also permit safe rejection and deferral where local policy allows them.
A forced acceptance path would destroy the distinction between system suggestion and human judgment.
This sheet uses “safe” only for a specified state or ownership invariant, never as a claim of patient safety.
Evidence labels are DERIVED for relations, VERIFIED for executed constructed checks, ILLUSTRATIVE for scenarios, and OPEN for unavailable observations.
No generic best alert frequency or universally optimal timeout is asserted.
The worked examples expose denominators, assumptions, and failure boundaries so a local team can replace inputs with measured values.
Changing an input can reverse a design choice, especially the choice between interruption and a passive worklist.
The final decision must therefore include workflow observation and locally authorized review, neither of which is complete here.

**Takeaway:** An alert is safe only when its workflow and recovery behavior are as explicit as its prediction rule.

## 2. Concept Map / Mental Model

```text
[eligible event] -> [policy/rule] -> [alert with reason/version]
                                      | unavailable recipient
                                      v
                               [named fallback/escalation]
                                      |
                                      v
[user sees] -> [acknowledge / defer / reject / act] -> [audit outcome]
                                      |
                                      v
                         [timeout, duplicate, delivery failure recovery]
```

The invariant is: every eligible alert reaches either a responsible recipient with a recorded disposition or a monitored exception queue.
“Sent” is not delivery, “delivered” is not seen, and “seen” is not a clinical decision.
The lifecycle begins with an eligible source event, not with a UI popup.
Creation binds the evidence snapshot and policy version to an episode identifier.
Routing resolves a role and current assignment into one accountable active owner.
Delivery establishes that a transport accepted or handed off the notification under a defined protocol.
Presentation establishes that an interface rendered the alert; it still does not establish comprehension.
Acknowledgment records a specified user signal with a timestamp and actor identity.
Disposition records the selected workflow outcome, such as completed, rejected, deferred, or escalated under policy.
Closure verifies that no active owner, open timer, or unresolved exception remains.
These states form a directed history rather than a single rank-ordered score.
For example, an exception can be assigned to a fallback and later delivered without losing its earlier failure history.
A terminal state must have no outgoing operational transition except a separately audited correction or reopening policy.
This sheet treats a correction as a new linked record, not deletion of history.
Ownership is singular for accountability at any instant, even when several people collaborate.
Collaborators can be listed separately without making responsibility ambiguous.
Transfer requires an explicit release/accept or an authorized automatic assignment rule.
Merely sending a copy to a second recipient does not transfer ownership.
If two owners can act independently, the design needs conflict detection and a defined reconciliation rule.
The simplest teaching model avoids concurrent active ownership and preserves read-only observers.
Recipient authorization has three distinct checks: identity, current role assignment, and permission for the offered action.
Authentication alone answers only who the actor is.
A role directory entry can be stale, so the routing record needs an effective interval or assignment version.
Permission also depends on the action type and context, not only on a broad job title.
The interface should hide or disable unavailable actions while preserving the evidence needed to request an authorized handoff.
A read-only recipient can acknowledge receipt only if policy explicitly assigns that meaning.
The system must not relabel read-only viewing as acceptance of an action.
If no authorized recipient resolves, routing moves to an exception queue with a named operational owner.
If authorization expires between presentation and action, action submission must fail closed and offer the permitted handoff path.
The prior acknowledgment remains factual history but does not authorize the later action.
Evidence presentation should answer four questions: what triggered the alert, what supports it, what is uncertain or missing, and what can the recipient do next.
The evidence snapshot needs units, observation time, source identifier, and freshness.
A score without a threshold rationale or supporting inputs is weak decision support.
A long unstructured rationale can also hide the requested action.
Place the reason and available action near each other, with supporting details available without losing episode context.
Missing evidence should be visibly missing rather than replaced by a reassuring default.
Stale evidence should trigger the locally specified refresh, deferral, or fallback rule.
Deduplication belongs before interruption only when members share both a review decision and an action scope.
Grouping by similar text alone is not sufficient.
Each group must retain member identifiers, arrival times, evidence versions, and an ungroup operation for audit and correction.
Escalation is a transition caused by a specified condition, normally timeout or failed routing.
It has a bounded attempt count, a named next owner, and a terminal exception path.
Dedup and escalation interact: retries should reuse the episode identifier rather than manufacture independent episodes.
A genuine new event may instead require a new episode even if its text matches an earlier one.
Accessibility is part of presentation correctness.
An alert conveyed only by color, sound, hover, or fine pointer motion can exclude recipients.
Keyboard operation, readable labels, scalable text, sufficient contrast, and non-audio equivalents need local testing.
No specific compliance certification is claimed here.
Interruption cost includes switching away from the current task and reconstructing its context.
Therefore an interruptive design requires a stronger justification than a passive worklist.
Recovery returns the episode to a valid owned state after failure; it does not erase the failed transition.
Measured evidence comes from actual traces or studies; unmeasured evidence consists of proposed checks, constructed traces, and assumed parameters.
Keeping those categories separate prevents a successful replay test from masquerading as observed human performance.

**Takeaway:** Separate creation, transport, presentation, acknowledgment, and action in both metrics and failure handling.

## 3. Foundational Definitions

| Term | Meaning | Boundary |
|---|---|---|
| alert burden | alerts requiring attention in a window | not the same as model volume if alerts are deduplicated |
| actionable alert | alert with an authorized recipient, evidence, and feasible action | not merely a high score |
| acknowledgment | recorded receipt/disposition signal | not agreement or action completion |
| escalation | time-bounded transfer when owner/path fails | not repeated notification forever |
| override/rejection | documented disagreement with system suggestion | not automatically an error |
| fatigue | degraded attention or response under interruption load | requires measurement, not assumed from count alone |

Alert episode means the full lifecycle for one policy-created unit of review.
Source event means an input occurrence that may create or join an episode.
Eligibility means the rule-defined condition under which an event enters the alert process.
Suppression means a policy decision not to create or present an episode under specified conditions.
Suppression is not deletion; the decision and rule version remain auditable.
Recipient means the role-resolved actor assigned responsibility for the next workflow step.
Observer means an actor permitted to view but not own or act.
Authorization means a current permission for a particular action and context.
Authentication means verification of identity and does not imply authorization.
Routing means resolving policy, role, assignment, and availability into an owner.
Delivery means a defined transport handoff, whose exact semantics must be documented.
Presentation means a recorded rendering event from the intended interface.
Visibility means an opportunity to perceive, which still cannot be inferred perfectly from a rendering log.
Comprehension means understanding the reason and choices and requires user evidence beyond logs.
Acknowledgment means the narrow signal defined by local policy, such as “I received this.”
Action completion means a separately recorded workflow action, not merely a clicked acknowledgment.
Rejection means a permitted decision not to take the suggested action.
Deferral means an owned temporary delay with a deadline and subsequent transition.
Cancellation means policy-based closure of an episode that is no longer applicable.
Cancellation requires a reason and must not be used to hide unresolved work.
Exception queue means a monitored set of episodes lacking normal progression or ownership.
Terminal disposition means a state with no remaining active timer or owner obligation under the lifecycle contract.
Deadline means a timestamp or duration derived from a documented urgency policy.
Escalation means an authorized transfer after a specified trigger.
Retry means reattempting delivery for the same episode and attempt lineage.
Duplicate delivery means repeated transport or presentation of the same episode.
Duplicate event means a distinct input record that may or may not represent the same review episode.
False merge means joining source events that require independent dispositions.
False split means creating separate interruptions for events that genuinely share one disposition.
Evidence snapshot means the immutable or versioned inputs available when the alert was created.
Freshness means elapsed time or version distance between supporting evidence and presentation.
Action affordance means an interface control representing an available transition.
An affordance is valid only if the backend rechecks authorization and current episode state.
Audit lineage means the linked identifiers that allow a complete episode replay.
Workload means demand for human attention over a specified window and alert class.
Capacity means attention minutes available under the same window and exclusions.
Backlog means unfinished demand, expressed either in episodes or minutes with units stated.
Utilization means demand divided by capacity under a stated interval and model.
Tail latency means a high quantile of elapsed time, not the mean and not a deterministic bound.
Interruption means a demand to switch attention before the current task naturally ends.
Passive worklist means a recipient-initiated queue rather than an unsolicited interruption.
Accessibility means perceivable, operable, and understandable interaction across relevant user needs and assistive technologies.
Usability means effectiveness, efficiency, and satisfaction in a specified context, which cannot be inferred from implementation alone.
Failure replay means applying recorded or constructed transitions to the lifecycle rules and checking invariants.
Measured evidence means observations tied to a stated population, period, interface version, and collection method.
Unmeasured evidence means assumptions, proposed telemetry, and constructed examples that still require local collection.
The following lifecycle contract supplies precise operational meanings for the states.

| State | Entry condition | Required record | Valid exit |
|---|---|---|---|
| created | eligibility accepted | episode, evidence, policy | routed or route_failed |
| routed | owner resolved | owner and assignment version | delivered or route_failed |
| delivered | transport handoff observed | transport timestamp | presented |
| presented | interface rendering observed | presentation identifier | acknowledged or timed_out |
| acknowledged | receipt signal recorded | actor and timestamp | completed, rejected, or deferred |
| deferred | delay permitted | reason and next deadline | completed, rejected, or timed_out |
| timed_out | deadline expires unresolved | deadline basis | escalated or exception_queued |
| route_failed | owner/path resolution fails | failure category | exception_queued |
| exception_queued | monitored recovery begins | exception owner | fallback_assigned or exception_closed |
| fallback_assigned | fallback accepts ownership | new owner/version | delivered |
| escalated | timeout transfers path | attempt and new route | delivered or exception_queued |
| terminal | authorized closure | disposition and reason | none in ordinary operation |

A mechanism invariant follows from this table.
Precondition: every episode has one current lifecycle state and one accountability record.
Input: a timestamped transition with episode, actor or component, and reason.
Output: an appended transition if the edge and actor are valid; otherwise a rejected transition and exception signal.
Transition: apply only an allowed edge, transfer ownership explicitly, and cancel obsolete timers on terminal closure.
Invariant: at most one active owner exists, and every nonterminal episode has an owner or monitored exception owner.
Justification: these rules prevent silent loss and uncontrolled simultaneous action without asserting anything about clinical correctness.
Worked trace: `created → routed → delivered → presented → acknowledged → action_completed` has one owner throughout and ends terminal.
Recovery trace: `created → route_failed → exception_queued → fallback_assigned → delivered` restores normal progression while retaining failure history.
Termination: completed, rejected, policy-cancelled, or explicitly exception-closed states end the ordinary lifecycle.
Failure boundary: missing timestamps, ambiguous actor authority, or malformed lineage can prevent reconstruction even when a final status exists.
This mechanism is VERIFIED only for the constructed replay cases in Section 9.
Local state names may differ, but equivalent distinctions must remain interpretable.

**Takeaway:** Terms that collapse delivery, attention, and action hide the very failures the interface must detect.

## 4. Core Formulas with Full Derivation

### 4.1 Base: review demand, acknowledgment, and conservation

**Validity before use:** Workload arithmetic assumes a defined alert class, fixed average review time, and no overlap; it is a planning estimate, not a clinical staffing recommendation.
The base workload relation multiplies a count by average minutes per count and converts minutes to hours.

```text
FORMULA [HF-01]: H = A * m / 60
WHERE: H = review hours/shift [h/shift]; A = actionable alerts/shift [alerts/shift];
       m = average review minutes/alert [min/alert]; 60 = minutes/hour [min/h].
CONSTRAINT: fixed alert class and window; representative average; interruptions, escalation,
            and documentation excluded unless included in m; no measured staffing claim.
NUMERICAL EXAMPLE: A=180 alerts/shift; m=3 min/alert;
                   H=180*3/60=9 h/shift.
INTERPRETATION: this constructed demand exceeds one constructed 8-hour capacity.
```

Derivation: `A [alerts] * m [min/alert] = A*m [min]`; division by `60 [min/h]` leaves hours.
The reciprocal units matter: multiplying by minutes per alert is valid, whereas multiplying by alerts per minute would be wrong.
If delivery requires acknowledgment by deadline `d`, completion rate is `acknowledged by d / eligible alerts`; report exception reasons separately.
A high rate can hide inappropriate acknowledgment if recipients use a bulk-dismiss path.

```text
FORMULA [HF-02]: p_ack = N_ack(d) / N_eligible
WHERE: p_ack = acknowledgment proportion [dimensionless]; N_ack(d) = eligible episodes
       acknowledged by deadline d [episodes]; N_eligible = eligible episodes [episodes].
CONSTRAINT: same cohort and deadline policy; 0 <= N_ack(d) <= N_eligible;
            suppressed episodes and ineligible events are not silently added to the denominator.
NUMERICAL EXAMPLE: N_ack(d)=150; N_eligible=180;
                   p_ack=150/180=5/6=0.833333...=83.333...%.
INTERPRETATION: 30 eligible episodes were not acknowledged by d, without identifying why.
```

Derivation: a proportion is the count satisfying a condition divided by the full eligible cohort count.
The deadline must be compared per episode, not by mixing different urgency classes with different `d`.
An exception taxonomy is required because the residual `N_eligible-N_ack` combines several mechanisms.
The action mix below demonstrates why `p_ack` cannot substitute for completed-action evidence.

```text
FORMULA [HF-03]: N_ack = N_complete + N_reject + N_defer + N_other
WHERE: all N values are acknowledged episodes by disposition [episodes];
       N_other is an explicitly defined residual category, not an unexamined bucket.
CONSTRAINT: mutually exclusive dispositions at one measurement cutoff; exhaustive categories;
            later state changes require a new cohort snapshot rather than retroactive relabeling.
NUMERICAL EXAMPLE: complete=90; reject=20; defer=25; other=15;
                   90+20+25+15=150 acknowledged episodes.
INTERPRETATION: all 150 acknowledgments reconcile, but only 90 have completed action at cutoff.
```

The proportion completing action among all eligible episodes is `90/180=0.5=50%`.
The proportion among acknowledged episodes is `90/150=0.6=60%`.
Both are arithmetically correct but answer different questions and use different denominators.
Neither establishes that the action was appropriate or improved an outcome.

### 4.2 General: funnel residuals, backlog, and capacity

A funnel check uses conservation of episode counts across sequential observations.
Not every lifecycle is strictly linear, so the simple subtraction applies only to a cohort snapshot with defined inclusion at each stage.
For example, fallback episodes must be counted once even if they pass through delivery more than once.

```text
FORMULA [HF-04]: R_j = N_j - N_(j+1)
WHERE: R_j = residual between adjacent funnel stages j and j+1 [episodes];
       N_j = unique eligible episodes observed at stage j [episodes].
CONSTRAINT: same episode cohort and cutoff; nested stage membership; retries and observers
            counted once per episode; R_j >= 0 under strictly nested membership.
NUMERICAL EXAMPLE: created=180, routed=176, delivered=170, presented=165, acknowledged=150;
                   residuals=(4,6,5,15), totaling 30 created-not-acknowledged episodes.
INTERPRETATION: the residual locates missing progression but does not prove its root cause.
```

Derivation: if stage `j+1` is a subset of stage `j`, set difference has size `N_j-N_(j+1)`.
If late arrivals or alternate paths violate nesting, use a state-transition ledger rather than forcing a funnel.
Backlog uses conservation of work rather than count alone.
Let `B_k` be unfinished review minutes at the end of discrete interval `k`.
Arrivals during the interval create `a_k*m` minutes of demand, while at most `C_k` minutes can be completed.
The recurrence includes unfinished work from the prior interval.

```text
FORMULA [HF-05]: B_k = max(0, B_(k-1) + a_k*m - C_k)
WHERE: B_k = backlog after interval k [min]; a_k = arriving actionable alerts [alerts];
       m = average review time [min/alert]; C_k = usable review capacity [min];
       B_0 = initial backlog [min].
CONSTRAINT: no abandoned work, capacity cannot be banked across idle intervals, common m,
            no simultaneous service above C_k, and all demand/capacity use the same interval.
NUMERICAL EXAMPLE: B_0=0; a=(50,45,35,30); m=3 min/alert; C_k=120 min each hour;
                   demands=(150,135,105,90) min;
                   B=(30,45,30,0) min at the four hour ends.
INTERPRETATION: peak constructed backlog is 45 min and the queue clears in hour four.
```

Derivation: before service, available work is `B_(k-1)+a_k*m`.
Completed work is the smaller of available work and `C_k`.
Subtracting that completed amount gives exactly `max(0,B_(k-1)+a_k*m-C_k)`.
The `max` encodes the fact that unused capacity cannot become negative backlog or stored future service.
For variable review times, replace `a_k*m` with the sum of individual review-minute demands.
For different alert classes, use a vector of counts and class-specific times, then sum minutes only after defining all classes.
Average utilization over a window follows from total demand and total nominal capacity.

```text
FORMULA [HF-06]: U = sum_k(a_k*m) / sum_k(C_k)
WHERE: U = average capacity utilization [dimensionless]; sums cover identical K intervals.
CONSTRAINT: positive total capacity; fixed m or replaced by actual demand sums;
            U is an average and does not bound hourly backlog or waiting time.
NUMERICAL EXAMPLE: demand=150+135+105+90=480 min; capacity=4*120=480 min;
                   U=480/480=1=100%.
INTERPRETATION: full average utilization coexists with a 45-minute transient peak backlog.
```

This counterexample rejects the claim that `U=1` implies no backlog.
It also rejects the claim that average utilization below one proves timely response.
Recovery requires spare service after arrivals decline.

```text
FORMULA [HF-07]: T_clear = B_start / (C_r - a_r*m)
WHERE: T_clear = time to clear backlog [h if rates use hours]; B_start = starting backlog [min];
       C_r = recovery capacity [min/h]; a_r = recovery arrivals [alerts/h];
       m = review time [min/alert].
CONSTRAINT: constant recovery rates, C_r > a_r*m, no new outage or abandonment,
            and divisible fluid work; discrete episode completions need a trace.
NUMERICAL EXAMPLE: B_start=30 min; C_r=120 min/h; a_r=30 alerts/h; m=3 min/alert;
                   new demand=90 min/h; T_clear=30/(120-90)=1 h.
INTERPRETATION: 30 min/h spare capacity clears 30 backlog-minutes in one constructed hour.
```

Derivation: net backlog change per hour is `a_r*m-C_r`.
When negative, backlog declines at `C_r-a_r*m` minutes per hour, so starting backlog divided by decline rate gives time.
If `C_r=a_r*m`, a positive backlog remains constant under this model.
If `C_r<a_r*m`, it grows and no finite clearance time exists.
Workload formulas should therefore be paired with deadline and backlog-age observations, not used in isolation.

### 4.3 Variants: interruptions, concurrency, and audit storage

Initial review is rarely the only labor.
Escalations, documentation, and context switching add terms that must be measured separately.

```text
FORMULA [HF-08]: H_total = (A*m + E*e + D*d + I*i)/60
WHERE: H_total = total modeled labor [h]; A = alerts [count]; m = review [min/alert];
       E = escalations [count]; e = added minutes/escalation [min/escalation];
       D = documented episodes [count]; d = added minutes/document [min/document];
       I = interruptions [count]; i = added switching minutes/interruption [min/interruption].
CONSTRAINT: categories are incremental and nonoverlapping; rates and averages describe the same cohort;
            switching time is not inferred here and must be locally measured.
NUMERICAL EXAMPLE: A=160, m=3; E=10, e=6; D=160, d=0 (included in m); I=40, i=0.5;
                   H_total=(480+60+0+20)/60=560/60=9.333333... h.
INTERPRETATION: escalation and switching add 80 minutes beyond initial review.
```

Setting `d=0` here does not claim documentation is free; it explicitly avoids double counting because the example defines it as included in `m`.
A local analysis that measures documentation separately must use a consistent `m` excluding that work.
The cost expression is linear only because averages are fixed; it is not a queueing model for waiting time.
Interruptions can have nonlinear effects on error or fatigue, which this formula deliberately does not assert.
Audit storage is a separate engineering estimate.

```text
FORMULA [HF-09]: S = A * r * b
WHERE: S = payload-only audit bytes [bytes]; A = episodes [episodes];
       r = audit records/episode [records/episode]; b = payload bytes/record [bytes/record].
CONSTRAINT: constant record size for estimate; excludes indexes, replication, backups,
            timestamps already excluded only if absent from b, compression, and retention policy.
NUMERICAL EXAMPLE: A=160 episodes; r=6 records/episode; b=512 bytes/record;
                   S=160*6*512=491520 bytes=480 KiB (1 KiB=1024 bytes).
INTERPRETATION: payload is small in this scenario, but actual storage remains unmeasured.
```

Derivation: episodes times records per episode gives records; records times bytes per record gives bytes.
The conversion is `491520/1024=480`, not a decimal-kilobyte conversion.
If record sizes vary, sum the actual sizes rather than using an unsupported average.
For retry schedules, cumulative elapsed time is a sum of delays, not the last delay alone.

```text
FORMULA [HF-10]: T_n = sum_(j=1..n) delta_j
WHERE: T_n = elapsed schedule time after n waits [min]; delta_j = j-th wait [min].
CONSTRAINT: waits are sequential, start from the same trigger, and exclude processing time unless added;
            schedule is locally configured rather than clinically recommended here.
NUMERICAL EXAMPLE: delta=(5,10,20) min; T_1=5, T_2=15, T_3=35 min.
INTERPRETATION: a 20-minute third wait ends at minute 35, not minute 20.
```

This schedule is illustrative and is not a recommendation for any alert class.
A retry cap must still specify where work goes after the last allowed escalation.

### 4.4 Limits and counterexamples

**Boundary:** Reducing `A` by raising a threshold may lower burden while increasing missed opportunities.
Workload cannot choose the threshold alone.
A deduplication example exposes another limit.
Suppose six raw events correctly form four episodes with memberships `{e1,e2}`, `{e3}`, `{e4,e5}`, and `{e6}`.
At three minutes per episode, correct grouping produces `4*3=12` review minutes.
An incorrect collision merges the first two groups into `{e1,e2,e3}` and produces only three episodes.
The arithmetic estimate becomes `3*3=9` minutes, a constructed three-minute reduction.
That lower workload is not a benefit if `e3` needs an independent disposition.
The audit check must detect the false merge by membership and action scope, not celebrate fewer alerts.
Acknowledgment ratios also have a denominator trap.
Dropping undelivered episodes from the eligible denominator can raise the reported proportion without improving delivery.
The original cohort must remain reconstructable.
Backlog formulas assume work is not abandoned.
If alerts expire or are cancelled, those removals need an explicit separate term and an auditable reason.
Treating abandoned work as completed service would understate backlog and overstate capacity.
Average review time can conceal a heavy tail or distinct classes.
A class-stratified model is necessary when review times differ materially.
Fixed capacity can also be unrealistic under competing work, meetings, or absence.
The examples use capacity only as a constructed constraint, not an instruction about staffing.
Concurrency makes minute totals insufficient.
Two people cannot necessarily divide one indivisible review, and simultaneous action can conflict.
The state oracle therefore complements rather than replaces workload arithmetic.
Finally, arithmetic verification does not verify semantic correctness.
An oracle can confirm `150/180` while the underlying acknowledgment log is incomplete or misleading.
Local measurement, usability review, and authorized policy review remain OPEN.

**Takeaway:** Quantify interruption demand and pair it with missed-alert and disposition evidence.

## 5. Worked Numerical Example

**ILLUSTRATIVE; arithmetic VERIFIED.** A worklist produces 180 actionable alerts per shift.
At three minutes of initial review each, burden is nine hours.
If an eight-hour capacity is binding, excess is one reviewer-hour.
If 20 alerts are duplicated and deduplication safely collapses them, `A=160` and estimated initial burden becomes eight hours.
That change must be tested against the risk that “duplicates” are distinct events needing separate action.
The 20 removed alerts are a constructed input, not an observed deduplication yield.
The phrase “safely collapses” is an assumption requiring membership and action-scope checks.
A five-minute acknowledgment target with 150 acknowledgments among 180 eligible alerts gives `150/180=83.3%` when rounded to one decimal.
The exact fraction is `5/6`; repeating decimal expansion is `83.333...%`.
The target duration is a scenario parameter and is not endorsed by this sheet.
The 30 exceptions must be separated into no delivery, no presentation, no recipient, delayed acknowledgment, and deliberately deferred review before treating the result as a user-performance metric.
For a compatible funnel snapshot, use created `180`, routed `176`, delivered `170`, presented `165`, and acknowledged `150`.
The adjacent residuals are four unrouted, six routed-not-delivered, five delivered-not-presented, and fifteen presented-not-acknowledged.
Their sum is `4+6+5+15=30`.
This decomposition does not prove that four episodes shared one root cause or that fifteen users deliberately ignored alerts.
It tells investigators where to look next.
Now assign the 150 acknowledged episodes a mutually exclusive cutoff disposition mix: 90 completed, 20 rejected, 25 deferred, and 15 other.
The mix sums to 150.
Completed action per eligible episode is `90/180=50%`.
Completed action per acknowledged episode is `90/150=60%`.
A report that selects only 60% would obscure the 30 eligible episodes without acknowledgment.
Neither proportion reveals whether the 90 actions were correct.
Rejection may be appropriate judgment and must not automatically count as failure.
Deferral remains nonterminal unless a later transition is observed.
The 15 “other” records require a defined reason taxonomy before operational conclusions are drawn.
Next, distribute the 160 deduplicated episodes across four hours as arrivals `50,45,35,30`.
The total is `50+45+35+30=160`.
At three minutes each, hourly demand is `150,135,105,90` minutes.
With 120 capacity minutes each hour and zero initial backlog, hour-one backlog is `max(0,0+150-120)=30` minutes.
Hour-two backlog is `max(0,30+135-120)=45` minutes.
Hour-three backlog is `max(0,45+105-120)=30` minutes.
Hour-four backlog is `max(0,30+90-120)=0` minutes.
Thirty capacity minutes are unused in hour four under this divisible-work model.
Total demand and total capacity both equal 480 minutes.
Thus average utilization is 100% even though backlog peaks at 45 minutes and later clears.
This demonstrates why an average alone cannot establish deadline performance.
To isolate recovery, start with 30 backlog-minutes and assume 30 arrivals per hour at three minutes each.
New demand is 90 minutes per hour.
With 120 recovery capacity minutes per hour, spare capacity is 30 minutes per hour.
Clearance is `30/30=1` hour.
If arrivals instead consume all 120 minutes per hour, spare capacity is zero and the backlog does not clear.
If they consume more, it increases.
Now test sensitivity to review time while retaining 160 total arrivals and 480 capacity minutes.
At two minutes each, total demand is `160*2=320` minutes or `5.333...` hours.
Average utilization becomes `320/480=2/3`, approximately 66.7%.
At four minutes each, total demand is `160*4=640` minutes or `10.666...` hours.
Demand exceeds capacity by 160 minutes, so average utilization is `640/480=4/3`, approximately 133.3%.
The original conclusion that eight hours exactly matches capacity is therefore fragile to a one-minute change in average review.
This sensitivity is arithmetic, not an observed distribution of review times.
For added work, assume ten escalations at six incremental minutes each and 40 interruptions at 0.5 incremental minutes each.
Initial review after deduplication consumes 480 minutes.
Escalations add 60 minutes and switching adds 20 minutes.
Total modeled labor is 560 minutes, or `9.333...` hours.
The scenario uses zero incremental documentation minutes only because documentation is defined as included in initial review.
Using a nonzero documentation term without removing it from review would double count.
For audit payload, six records per episode at 512 bytes each across 160 episodes gives 960 records.
Payload is `960*512=491520` bytes, equal to 480 KiB.
Indexes, replicas, retention, and backups remain outside the estimate.
For escalation timing, waits of 5, 10, and 20 minutes produce cumulative times 5, 15, and 35 minutes.
The retry-cap trace allows the first two timeout attempts to escalate and sends attempt three to the exception queue.
This rule prevents an unbounded loop in the constructed state machine.
A real implementation still needs a monitored exception owner and explicit closure criteria.
The end-to-end decision from this example is limited: workload is sensitive, funnel exceptions need taxonomy, and acknowledgment cannot substitute for action.
It does not select a threshold, validate a timeout, or establish that eight hours is an appropriate local capacity.
The executed oracle and its limitations appear in Section 9 so readers can distinguish checked arithmetic from OPEN observations.

**Takeaway:** A denominator and exception taxonomy prevent a superficially good acknowledgment number from concealing transport or staffing failure.

## 6. Visual / Geometric Intuition

Imagine alert load as arrivals to a bounded attention queue.
When arrivals exceed safe review capacity, either waiting time grows or alerts are ignored, escalated, grouped, or displaced by other work.
The UI controls which evidence is visible at the decision point; it cannot compensate for absent ownership or invalid inputs.
A queue diagram should show work entering, work completed, and backlog retained.

```text
Hour:          1        2        3        4
Arrivals:     50       45       35       30
Demand min:  150      135      105       90
Capacity:    120      120      120      120
Backlog min:  30   ->   45   ->   30   ->   0
               rising  peak     draining clear
```

The backlog line rises whenever the demand bar exceeds the capacity line.
It falls only when spare capacity exceeds new demand after accounting for existing work.
Equal total demand and capacity do not imply a flat backlog line.
That is the geometric content of the 45-minute peak.
A funnel diagram answers a different question.

```text
Created       180  ##################################
Routed        176  #################################-
Delivered     170  ################################--
Presented     165  ###############################---
Acknowledged  150  #############################-----
```

Each step downward represents unresolved progression between stages, not necessarily permanent loss.
A delayed episode may later enter the next stage.
A single cutoff snapshot cannot distinguish delay from permanent failure without timestamps and final state.
The bars count unique episodes rather than messages, preventing retries from inflating apparent coverage.
The lifecycle state diagram adds paths that a funnel hides.

```text
created -> routed -> delivered -> presented -> acknowledged -> terminal
   |                                  |              |
   v                                  v              v
route_failed                      timed_out       deferred
   |                                  |              |
   v                                  v              v
exception_queued <- fallback <- escalated       later disposition
```

The drawing is simplified: exception closure and policy cancellation also require explicit terminal rules.
Its main point is that recovery loops return to valid owned states without deleting earlier history.
An ownership timeline makes transfer visible.

```text
Time       Owner                 Evidence
t0         unresolved            episode created
t1         role queue            route selected
t2         reviewer R            delivery accepted
t3         reviewer R            presentation recorded
t4         escalation queue      deadline expired
t5         fallback F            transfer accepted
t6         none; terminal        disposition recorded
```

At no nonterminal instant should the owner column be empty.
The role queue and escalation queue can be owners only if monitored and assigned accountability.
“System” is not an adequate owner label unless a named operational responsibility stands behind it.
Evidence presentation can be visualized as layered disclosure.

```text
Layer 1: reason + evidence time + urgency basis + permitted next step
Layer 2: supporting inputs + units + missingness + policy version
Layer 3: full lineage + prior dispositions + recovery/audit history
```

Layer one supports the immediate workflow question without overwhelming the recipient.
Layer two permits checking rather than blind acceptance.
Layer three supports review and replay without forcing every recipient to parse audit internals.
All layers need accessible names and equivalent keyboard or assistive-technology paths.
A color-only urgency stripe lacks a redundant textual label.
A sound-only alert lacks a visual equivalent and may be imperceptible in some environments.
An icon-only action can be ambiguous without an accessible name.
Interruption geometry involves two task contexts rather than one queue.

```text
Task A context --interrupt--> Alert B --return--> reconstruct Task A context
                 |<------ switching and recovery cost ------>|
```

The diagram does not assume a numeric switching penalty.
It identifies where a local time-and-motion or usability study could measure one.
Grouping alerts changes the number of context switches only when grouping is semantically valid.
False merges reduce visible work geometrically while potentially hiding independent obligations.
The state machine is therefore a correctness overlay on every queue, funnel, and interface diagram.

**Takeaway:** Alert fatigue is a queue-and-workflow problem as well as an interface problem.

## 7. Comparison / Selection Table

| Pattern | Use when | Advantage | Disqualifier | Recovery requirement |
|---|---|---|---|---|
| interruptive alert | urgent, actionable event | rapid attention | no timely action exists | acknowledged recipient and timeout escalation |
| passive worklist | prioritization is sufficient | fewer interruptions | time-critical event | stale-item and ownership monitoring |
| grouped/deduplicated alert | repeated evidence is genuinely same action | lowers burden | events have independent consequences | reversible grouping and audit members |
| abstain/fallback | input or authority is inadequate | prevents unsupported guidance | fallback unavailable | exception owner and baseline process |

Selection begins with the required response and actionability, not with available UI components.
An interruptive alert is a candidate only when interruption has a locally justified benefit over passive presentation.
If no recipient can act, an interruption merely transfers demand without resolving it.
A passive worklist reduces unsolicited switching but can accumulate stale items.
It requires visible ownership, age, and deadline indicators.
Grouping is appropriate only when one disposition genuinely covers every member.
If members need independent action, separate episodes or explicit subitem dispositions are required.
Fallback is appropriate when evidence or authority is inadequate for the normal path.
Fallback must be a real process with capacity and ownership, not a label for dropping work.
The next table separates ownership mechanisms.

| Ownership mechanism | Strength | Failure risk | Required check |
|---|---|---|---|
| named individual | clear accountability | absence or stale assignment | current assignment and fallback |
| monitored role queue | shares work across eligible actors | queue assumed to self-own | queue owner and acceptance record |
| dual active owners | redundancy | conflicting or duplicate action | conflict rule or avoid concurrency |
| automatic escalation | bounded waiting | wrong target or loops | authorization, cap, terminal exception |
| exception owner | captures abnormal paths | exception becomes invisible backlog | age monitor and closure audit |

The simplest default for the teaching state machine is one active owner with read-only observers.
A local design may choose concurrency only with a explicit conflict-resolution mechanism.
Recipient selection must also distinguish identity from permission.

| Authorization question | Evidence needed | Failure response |
|---|---|---|
| Who is acting? | authenticated actor identity | reject unidentified action |
| Is the assignment current? | assignment version/effective interval | reroute or fallback |
| May this actor perform this action? | action-specific permission | disable/reject and hand off |
| Is the episode still actionable? | current state and evidence version | refresh, defer, or cancel by policy |

Backend checks remain necessary even when a button appears enabled.
UI state can be stale or manipulated, and authorization can expire after presentation.
The next table compares evidence presentations.

| Presentation choice | Helps | Can harm | Verification |
|---|---|---|---|
| reason-first summary | rapid interpretation | omit important limitations | comprehension and completeness review |
| score-only display | compact ranking | hide basis and missing inputs | require evidence and version access |
| layered detail | supports checking | bury urgent action under detail | task-based usability test |
| raw record dump | completeness | overload and ambiguous relevance | curated evidence view |
| freshness label | exposes stale support | false reassurance if clock wrong | clock/source consistency test |

Evidence needs source time, units, missingness, and policy version.
More text is not automatically more actionable.
The interface should preserve context when a recipient opens supporting detail.
Acknowledgment designs also differ.

| Ack semantics | What it proves | What it cannot prove |
|---|---|---|
| received | actor recorded receipt | understanding, agreement, action |
| reviewed | actor attested to review | review quality or appropriate decision |
| accepted | actor agreed to proposed next step | completion or correctness |
| rejected | actor declined suggestion | poor model or user error |
| deferred | actor retained owned delay | eventual completion |

A single ambiguous “OK” button should not be allowed to carry all these meanings.
Action completion needs its own event and evidence.
Escalation policies can likewise be compared by their terminal behavior.

| Policy shape | Benefit | Boundary |
|---|---|---|
| retry same route | handles transient transport failure | cannot fix absent authority |
| escalate to fallback | changes ownership/path | fallback must exist and accept |
| cap then exception | prevents infinite loops | exception needs monitoring and closure |
| unlimited rebroadcast | superficial persistence | duplicates, loops, unclear ownership |

Measured and unmeasured evidence require different decision weight.
A constructed replay can verify allowed transitions but not recipient comprehension.
A rendering log can measure presentation events but not actual attention.
A usability study can measure task performance in its tested population but not automatically generalize across settings.
A local trace can reveal outage patterns but cannot by itself establish clinical outcomes.
Selection documentation should state which evidence exists and which remains OPEN.
It should record the rejected alternatives and the constraint that would reverse the decision.
For example, a passive worklist choice reverses if a verified response requirement makes recipient-initiated review too late.
An interruptive choice reverses if no feasible timely action or authorized recipient exists.
Grouping reverses when audit shows false merges.
Fixed single-owner routing reverses when assignments are too unstable, but the alternative still needs explicit accountability.
No selection here is a clinical recommendation.

**Takeaway:** Choose the least interruptive mechanism that still meets the defined response requirement.

## 8. Failure Modes & Edge Cases

| ID | Competing causes | Discriminating signal | Bounded correction / verification |
|---|---|---|---|
| HF-F01 alert fatigue | excessive volume; poor precision; staffing loss | high dismiss/defer rate with delivery intact, stratified by shift | change policy only after reviewing missed-event trade-off; observe disposition distribution |
| HF-F02 silent delivery failure | queue outage; address mapping error; client failure | created count exceeds delivered/presented count | route exception queue; test failed transport and recovery trace |
| HF-F03 duplicate suppression harms | repeated message; independent recurring event | inspect grouped event IDs and actions | make grouping reversible; sample audit before rollout |
| HF-F04 acknowledgment without review | bulk action; incentive; unclear UI | implausibly short latencies or repeated patterns | distinguish seen/acknowledged/action; usability review |
| HF-F05 escalation loop | absent owner; bad timeout; retry bug | repeated alerts without terminal disposition | cap retries; assign named escalation and verify closure |

Each entry below separates plausible causes, checks, mitigation risk, correction, prevention, and verification.
These are diagnostic hypotheses, not observed incidents.
All local frequencies and clinical consequences remain unmeasured.

### HF-F01: rising dismissal or deferral under interruption load

Category: `CONCEPT_CONFUSION` with possible `SCALE_FAILURE`.
Provenance: constructed diagnostic entry; no local user study available.
Observation to investigate: recipients dismiss or defer more often, or response latency rises.
Competing cause A is excessive interruption volume relative to available capacity.
Competing cause B is low relevance or poor evidence, which makes rejection reasonable rather than fatigue.
Competing cause C is reduced staffing or competing work, which changes capacity without any policy change.
Competing cause D is a transport or rendering defect that changes the apparent response distribution.
Expected signals include intact creation/delivery/presentation traces, shift-stratified disposition changes, rising backlog age, and increased deferral.
Discriminating checks are to reconcile the funnel first, then compare arrivals, capacity, evidence quality, and disposition reasons over the same windows.
A latency-only check cannot separate workload from interface confusion.
Bounded mitigation is to review grouping, urgency classification, or work allocation within authorized policy.
Mitigation risk is missed independent events, delayed review, or simply moving burden to a fallback queue.
Correction requires preserving the original cohort and reviewing dispositions rather than optimizing dismissal rate.
Prevention requires capacity monitoring, actionability review, and a change gate for interruption policy.
Verification requires a subsequent measured distribution of dispositions, backlog age, and missed independent work under the same definitions.
OPEN evidence: no baseline, fatigue scale, user observation, or outcome study is available here.

### HF-F02: created alerts disappear before presentation

Category: `VIOLATED_ASSUMPTION` or `SCALE_FAILURE`, depending on the trace.
Provenance: constructed diagnostic entry; no local transport logs available.
Observation to investigate: created count exceeds delivered or presented count for the same cohort.
Competing cause A is queue or network outage.
Competing cause B is stale recipient mapping or expired authorization.
Competing cause C is client rendering failure after successful transport.
Competing cause D is telemetry loss, which can mimic missing delivery even when a user saw the alert.
Expected signals include stage residuals, transport error codes, unresolved routing identities, or missing client presentation events.
Discriminating checks join episode IDs across producer, transport, and client logs and compare timestamp conventions.
A producer success response alone cannot discriminate these causes.
Bounded mitigation routes affected episodes to a monitored exception queue and invokes the authorized fallback.
Mitigation risk is duplicate presentation when delayed original delivery later succeeds.
Correction therefore uses idempotent episode identity and records both attempts rather than creating unrelated episodes.
Prevention requires end-to-end heartbeats or synthetic checks where appropriate, assignment validation, and replayable audit lineage.
Verification injects a constructed failed-route or failed-presentation trace and confirms exception ownership, fallback delivery, and one terminal disposition.
The executed oracle checks route-failure and fallback states, not real queue behavior.
OPEN evidence: actual transport semantics, client instrumentation, and failure rates remain unavailable.

### HF-F03: deduplication merges independent obligations

Category: `CONCEPT_CONFUSION` or `VIOLATED_ASSUMPTION`.
Provenance: constructed six-event grouping example; no production dedup audit available.
Observation to investigate: alert count falls after grouping but reviewers report missing context or separate actions become difficult.
Competing cause A is correct removal of repeated delivery for the same episode.
Competing cause B is false merging of distinct source events based on text or proximity.
Competing cause C is false splitting, which increases interruptions even though no independent obligation was removed.
Competing cause D is stale group membership after evidence updates.
Expected signals include member IDs with incompatible action scopes, repeated reopen requests, or grouped events carrying distinct dispositions.
Discriminating checks inspect every member, evidence version, and required action rather than only the rendered group title.
The constructed bad merge `{e1,e2,e3}` is detectable because `e3` belongs to an independent correct group.
Bounded mitigation disables the narrow grouping rule for the affected pattern and preserves reversible membership.
Mitigation risk is a temporary increase in interruptions and backlog.
Correction ungroups or creates linked explicit subitem dispositions according to local policy without deleting audit history.
Prevention requires shared-action criteria, member-level lineage, and test cases for collision and recurrence.
Verification replays correct and bad groups and confirms that independent obligations remain separately closable.
Lower alert count alone is not verification.
OPEN evidence: local duplicate prevalence, false-merge rate, and action consequences remain unmeasured.

### HF-F04: acknowledgment is mistaken for review or action

Category: `CONCEPT_CONFUSION`.
Provenance: constructed acknowledgment/disposition example; no local usability observation available.
Observation to investigate: acknowledgment rate is high while action completion, rejection, or deferral evidence is absent or implausible.
Competing cause A is a bulk-dismiss control.
Competing cause B is an incentive tied only to acknowledgment speed.
Competing cause C is ambiguous button wording that implies receipt to the user but acceptance to the system.
Competing cause D is legitimate rapid rejection because evidence is clearly irrelevant.
Expected signals include very short presentation-to-acknowledgment latencies, repeated actor patterns, and a missing subsequent disposition.
Discriminating checks inspect event semantics and task flows, then sample whether recipients can explain the alert reason and chosen action.
Very short latency is a clue, not proof of inadequate review.
Bounded mitigation relabels and separates receipt from disposition while retaining a permitted rejection path.
Mitigation risk is extra clicks, longer review, or workarounds.
Correction reconciles historical metrics under their old semantics rather than retroactively treating acknowledgments as actions.
Prevention requires explicit event definitions, separate backend action checks, and task-based usability review.
Verification confirms that the state `acknowledged` is nonterminal and that action has a distinct authorized event.
The state oracle performs this constructed check only.
OPEN evidence: actual comprehension, click latency distribution, and appropriateness of rejection remain unmeasured.

### HF-F05: escalation retries never reach terminal ownership

Category: `SCALE_FAILURE` or `VIOLATED_ASSUMPTION`.
Provenance: constructed retry-cap trace; no local escalation incident available.
Observation to investigate: repeated timeout and escalation events accumulate without a terminal disposition.
Competing cause A is no authorized fallback recipient.
Competing cause B is a timeout reset bug that creates a new cycle indefinitely.
Competing cause C is duplicate episode creation for each retry, defeating a per-episode cap.
Competing cause D is an unmonitored exception queue that receives episodes but never closes them.
Expected signals include repeated episode IDs or, worse, many new IDs with identical lineage, increasing attempt counts, and no owner acceptance.
Discriminating checks reconstruct attempt lineage, timer creation/cancellation, fallback authorization, and exception age.
Counting notifications cannot distinguish retry from new episode creation without identifiers.
Bounded mitigation applies a per-episode attempt cap and transfers to a named monitored exception owner.
Mitigation risk is delayed review while waiting for exception handling or concentration of backlog in the fallback path.
Correction resolves the owner/path problem and records an authorized closure or resumed normal path.
Prevention requires cap tests, terminal-state tests, fallback assignment checks, and exception-age monitoring.
Verification uses attempts one, two, and three: escalate, escalate, then exception queue; a subsequent fallback trace reaches terminal disposition.
The oracle verifies this rule but not actual timing or staffing.
OPEN evidence: real timeout configuration, retry frequency, and exception clearance remain unmeasured.

### HF-F06: evidence looks actionable but is stale or incomplete

Category: `VIOLATED_ASSUMPTION`.
Provenance: constructed interface failure entry; no source-body or local interface evidence available.
Observation to investigate: recipients cannot determine what changed, or an action fails because supporting evidence has changed.
Competing cause A is missing source timestamp or unit.
Competing cause B is stale evidence displayed without a freshness indicator.
Competing cause C is unavailable detail hidden behind an inaccessible interaction.
Competing cause D is an authorization change between presentation and submission.
Expected signals include evidence-version mismatches, missing fields, expired assignments, and action rejection despite an enabled control.
Discriminating checks compare creation and presentation snapshots, actor assignment versions, and backend rejection reasons.
A screenshot alone cannot establish backend state or freshness.
Bounded mitigation blocks unsupported submission and offers refresh, deferral, or authorized handoff.
Mitigation risk is added delay or additional interruptions after refresh.
Correction records the failed transition and retains the earlier acknowledgment as history, not permission.
Prevention requires explicit freshness rules, accessible evidence layers, and backend state/authorization checks.
Verification uses a replay with stale version or expired actor permission and confirms fail-closed behavior plus owned recovery.
That replay has not been executed against a local implementation and remains a proposed verification.
OPEN evidence: actual freshness thresholds, authorization policy, and usability remain unavailable.

Across all entries, mitigation is bounded because an apparent fix can move rather than remove risk.
Reducing volume can hide independent obligations.
Adding escalation can add interruptions and fallback overload.
Adding confirmation clicks can increase burden without improving understanding.
Disabling an action can leave unresolved work unless handoff is owned.
Verification therefore needs both the targeted signal and a guardrail for transferred work.

**Takeaway:** Every alert failure needs a signal that distinguishes a human-workload problem from a technical-delivery problem.

## 9. Complexity & Cost Analysis

For `A` alerts, immutable audit writes and disposition reconciliation are `O(A)` records under ordinary append semantics.
The material cost is attention: `A*m` minutes plus escalation and documentation.
Storage, paging, and staffing prices are not measured here.
Monitor backlog age and unassigned count, not only model score distribution.
If each episode has at most `r` records, audit append work is `O(A*r)`.
When `r` is bounded by a fixed lifecycle and retry cap, this is linear in episodes.
Without a retry cap, both records and notifications can grow with repeated attempts rather than unique work.
Deduplication requires comparing candidate memberships.
A naive all-pairs comparison over `E` source events costs `O(E^2)` comparisons, although an implementation may index candidates differently.
No particular indexing algorithm or measured runtime is asserted here.
State transition validation is constant time per event when allowed edges are represented as a fixed lookup table.
Replaying `R` transition records costs `O(R)` edge checks plus the cost of reading and storing those records.
Checking one active owner and terminal closure per replayed episode is also linear in records under straightforward iteration.
Storage for the constructed 160-episode example is 960 audit payloads and 480 KiB before overhead.
Actual database storage, replication, backup, and retention costs are OPEN.
Human attention remains nonfungible: two available hours on paper do not prove timely capacity at the required instant.
Average utilization can be 100% while backlog transiently reaches 45 minutes, as Section 5 shows.
Tail response requires timestamp distributions or discrete simulation, not only average demand.
Context-switch costs are not derivable from alert count and must be observed locally.
Accessibility and usability work are design and evaluation costs, not optional cosmetic additions.
Their resource requirements and effectiveness are unmeasured in this sheet.

### Executed dependency-free arithmetic and state-oracle smoke

Execution status: **VERIFIED for constructed arithmetic and explicit transition rules only**.
The check used Python's standard `fractions.Fraction` for exact rational arithmetic and no third-party dependencies.
It executed during authoring on 2026-11-09.
It did not connect to a database, interface, queue, clinical system, or user study.
The arithmetic oracle evaluated the exact scenarios in Sections 4 and 5.
The state oracle encoded allowed outgoing edges and terminal states, then replayed three valid traces, one nonterminal acknowledgment trace, and one invalid backward transition.
Retry-cap attempts one, two, and three were evaluated independently.
Actual output is reproduced below without upgrading its scope.

```text
ARITHMETIC_ORACLE
initial_review_hours = 9
dedup_alerts = 160
dedup_review_hours = 8
ack_fraction = 5/6 ack_percent = 83.33333333333333
eligible_not_ack = 30
ack_action_mix_sums = 150
action_complete_per_eligible_percent = 50.0
action_complete_per_ack_percent = 60.0
funnel_gaps_created_routed_delivered_presented_ack = [4, 6, 5, 15]
not_presented_plus_presented_not_ack = 30
hourly_demand_minutes = [150, 135, 105, 90]
hour_end_backlog_minutes = [30, 45, 30, 0]
peak_backlog_minutes = 45
total_demand_minutes = 480
total_capacity_minutes = 480
recovery_hours_from_30min_backlog = 1
sensitivity_4min_total_hours = 32/3
sensitivity_4min_excess_minutes = 160
sensitivity_2min_review_hours = 16/3
audit_payload_bytes = 491520
audit_payload_KiB = 480
escalation_cumulative_minutes = [5, 15, 35]
dedup_raw_events = 6
dedup_correct_groups = 4
dedup_bad_collision_groups = 3
dedup_bad_collision_has_false_merge = True
STATE normal_completed: terminal=True; ack_does_not_imply_action=True; states=6
STATE fallback_rejected: terminal=True; ack_does_not_imply_action=True; states=8
STATE escalated_completed: terminal=True; ack_does_not_imply_action=True; states=10
STATE ack_only_nonterminal: terminal=False; ack_does_not_imply_action=False; states=5
ack_only_terminal = False
invalid_transition_rejected = ('invalid_ack_to_presented', 'acknowledged', 'presented')
retry_cap_outcomes = ['escalated', 'escalated', 'exception_queued']
SMOKE_RESULT = PASS; scope = constructed arithmetic and transition rules only
```

The label `ack_does_not_imply_action=True` in a valid terminal trace means an action/rejection event exists after acknowledgment.
In the acknowledgment-only trace it is false, and terminal is also false.
The oracle rejects `acknowledged → presented` because returning to presentation is not an allowed ordinary transition.
A real reopening or correction policy would need explicit new semantics rather than an undocumented backward edge.
The normal completed trace confirms a valid path to action completion.
The fallback rejected trace confirms that route failure can recover and end in a permitted rejection.
The escalated completed trace confirms that timeout, escalation, and later completion can coexist in one lineage.
The retry cap confirms escalation on attempts one and two and exception queueing on attempt three.
The arithmetic output confirms exact workload, fraction, backlog, sensitivity, storage, and timing values.
It also confirms that the deliberately bad deduplication group contains a false merge.
These checks do not prove that the chosen state model matches local policy.
They do not test concurrent actors, clock skew, persistence failures, security, accessibility, or actual user behavior.
They do not establish clinical validity, appropriate timing, or measured effectiveness.
No real log replay was available.
That absence remains an OPEN evidence limitation rather than a passed test.

### Failure-replay protocol for local evidence

A useful local replay starts with an episode ID and an immutable ordered transition extract.
Record the source of each timestamp and the actor or component associated with each transition.
Check that the first state is created and every subsequent edge is allowed.
Check authorization at each action and ownership at every nonterminal point.
Check that retries retain episode lineage and do not exceed the cap.
Check that deduplicated members retain independent obligations where required.
Check that terminal states have no active timers or unresolved exception ownership.
Report malformed or ambiguous records separately instead of silently repairing them.
A successful replay establishes consistency with encoded rules, not correctness of the rules themselves.
A failed replay identifies a transition or invariant violation, but competing implementation and telemetry causes still require investigation.
Measured replay evidence should include period, population, interface version, missing-record handling, and counts.
Unmeasured proposed checks must be labeled as such.
No such measured local replay is claimed here.

**Takeaway:** The dominant cost of an alert system can be scarce reviewer attention, which must be explicitly budgeted.

## 10. Key Theorems / Results Reference

- **Conservation of work (DERIVED):** under the simple model, alert review minutes equal count times per-alert minutes.
- **End-to-end delivery principle:** a producer success response does not establish recipient presentation or acknowledgment.
- **Authority boundary:** an interface affordance must not create an action beyond the user’s assigned authority.

**Backlog recurrence (DERIVED):** with no abandonment and nonbankable capacity, `B_k=max(0,B_(k-1)+a_k*m-C_k)`.
Its proof is the subtraction of completed work from available work, with completed work capped by both availability and capacity.
Its boundary is variable demand, expiry, cancellation, or concurrent service, each requiring explicit model changes.
**Finite-recovery condition (DERIVED):** under constant recovery rates, a positive backlog clears only if `C_r>a_r*m`.
Equality leaves backlog unchanged and a negative surplus grows it.
This result concerns the fluid workload model, not a guaranteed real recovery time.
**Funnel residual result (DERIVED):** for nested membership over one cohort, adjacent residuals sum to the first-minus-last count.
In the worked cohort, `4+6+5+15=180-150=30`.
If episodes can enter through alternate paths, membership may not be nested and the simple funnel is invalid.
**Disposition partition result (DERIVED):** mutually exclusive and exhaustive acknowledged dispositions sum to total acknowledgments.
This is accounting consistency, not proof that any disposition was appropriate.
**Acknowledgment non-implication (VERIFIED for encoded state model):** the acknowledgment-only trace is nonterminal and lacks action evidence.
A policy definition, not arithmetic, determines what acknowledgment means.
**Terminal ownership invariant (VERIFIED for constructed traces):** valid terminal traces end in a declared terminal state; abnormal nonterminal paths retain an exception or active owner.
The implementation must enforce this locally before claiming the invariant in production.
**Retry boundedness (DERIVED and VERIFIED for three constructed attempts):** a per-episode cap of two escalations sends the third timeout attempt to exception handling.
This prevents unlimited escalation edges only if episode identity and cap enforcement are correct.
**Deduplication conservation boundary:** grouping reduces presentation count, but independent member obligations must remain closable.
A lower episode count is not sufficient evidence of correct deduplication.
**Authority invariant:** every action transition requires current actor identity, assignment, and action permission.
A previously valid acknowledgment cannot authorize a later action after permission expires.
**Evidence freshness boundary:** an evidence snapshot is historical unless refresh or version equivalence is established.
Presentation of a snapshot does not establish current truth.
**Accessibility non-inference:** implementation of labels or keyboard handlers alone does not establish usable accessibility.
Local testing with relevant tasks and users remains required.
**Measurement non-inference:** a rendering event does not prove attention, and an acknowledgment does not prove comprehension.
Stronger claims need stronger study designs and direct evidence.
**Arithmetic/semantic separation:** the executed oracle can check numerical and encoded state consistency while inputs and rules remain wrong for the local setting.
This separation is the central limitation of all constructed verification here.
**Rejection neutrality:** a rejection can be appropriate, so rejection rate alone is not an error metric.
Interpretation needs reason categories and review.
**Exception-queue persistence:** transferring work to an exception queue changes location, not the need for ownership and closure.
Exception age and unresolved count remain necessary operational measures.
**Deadline semantics:** timeout means an unresolved deadline occurred, not that a recipient is at fault.
Transport, presentation, capacity, and policy causes must be separated.
**Clock boundary:** cross-component latency requires synchronized or otherwise interpretable timestamps.
Unexplained negative latencies are data-quality signals rather than evidence of instantaneous response.
**Recovery non-erasure:** successful fallback adds valid later states but does not erase earlier failures.
Audit lineage must preserve both for diagnosis.
**Evidence status:** no external theorem, local production property, original source result, or clinical outcome is claimed beyond these bounded derivations and constructed checks.

**Takeaway:** Instrument the whole path from eligibility through terminal disposition, not just the alert generator.

## 11. Interview / Exam Questions

1. **Compute review hours for 180 alerts at three minutes each.** Nine hours; state the fixed-average assumption.
   Reasoning: `180*3=540` minutes and `540/60=9` hours.
   The answer excludes unmeasured escalation and switching work and is not a staffing recommendation.

2. **Why is “alert sent” inadequate?** Transport, presentation, ownership, and acknowledgment can still fail.
   Reasoning: producer success is only one component's observation.
   An adequate answer names episode IDs and end-to-end states rather than relying on notification count.

3. **What invariant should a critical alert have?** Terminal disposition by a responsible owner or a monitored exception path.
   Reasoning: every nonterminal episode needs accountable ownership, and closure needs an explicit authorized state.
   “Critical” does not supply a clinical timeout or policy; those require local approval.

4. **How can deduplication be dangerous?** Similar-looking events may require independent actions; preserve members and reversibility.
   Application: merging correct groups `{e1,e2}` and `{e3}` reduces four episodes to three but creates a false merge.
   The nine-minute workload versus twelve minutes is not a benefit if `e3` loses its independent disposition.

5. **Does acknowledgment mean the recommendation was correct?** No; it records a workflow event, not validity or outcome.
   In the worked cohort, 150 acknowledgments contain only 90 completed actions at cutoff, plus rejection, deferral, and other dispositions.
   Even action completion does not prove appropriateness.

6. **How do you distinguish fatigue from transport failure?** Compare created, delivered, presented, and disposition traces before interpreting response behavior.
   Diagnosis: intact presentation with rising deferral and backlog suggests workload or relevance hypotheses; stage residuals suggest routing, transport, or client hypotheses.
   Telemetry loss remains a competing cause, so no single metric is decisive.

7. **What changed constraint reverses a passive-worklist choice?** A verified time-critical response requirement may require interruptive escalation.
   Judgment: the reversal depends on feasible authorized action and local urgency policy.
   If no action is available, adding interruption does not solve the underlying problem.

8. **What should happen on missing recipient identity?** Do not silently drop; route to an exception owner/fallback.
   The trace `created → route_failed → exception_queued → fallback_assigned → delivered` preserves failure history and restores ownership.
   Rebroadcasting without identity resolution is not an adequate answer.

9. **What is OPEN?** Actual user study, local capacity, policy, and source-body evidence.
   Also open are transport logs, accessibility evaluation, clinical consequences, and measured deduplication behavior.
   The executed smoke verifies constructed arithmetic and state rules only.

10. **Why track rejection?** It can expose interface, data, or policy mismatch and supports accountable review.
    Rejection may be correct judgment and must not automatically be penalized.
    Reason categories and independent review are needed before interpretation.

11. **Calculate the four hourly backlog values for arrivals 50, 45, 35, and 30, three minutes per alert, and 120 capacity minutes per hour.**
    Answer: `30,45,30,0` minutes, starting from zero.
    The recurrence is `B_k=max(0,B_(k-1)+3*a_k-120)`; unused capacity cannot be banked.

12. **A cohort has 180 eligible, 150 acknowledged, and 90 completed-action episodes. Which action percentages are valid?**
    Answer: 50% of eligible and 60% of acknowledged, for different denominators.
    Neither is a measure of correct action, and the 30 unacknowledged episodes still require exception analysis.

13. **An actor was authorized when an alert appeared but loses permission before clicking the action. What should happen?**
    Answer: the backend rejects the action, records the failed transition, and offers an authorized handoff or fallback.
    The earlier acknowledgment remains history but grants no permission; hiding the button alone is insufficient.

14. **Average utilization is 100% and final backlog is zero. Does that prove deadlines were met?**
    Answer: no.
    In the worked trace, backlog peaks at 45 minutes before clearing; averages and final state omit backlog age and per-episode deadline performance.

15. **What does the executed state oracle establish, and what does it leave unresolved?**
    Answer: it verifies the three valid constructed traces, nonterminal acknowledgment-only state, invalid backward-edge rejection, retry-cap outcomes, and arithmetic outputs.
    It does not establish real concurrency handling, persistence, timing, user comprehension, accessibility, local policy validity, or clinical benefit.

**Takeaway:** Interview answers should distinguish technical state transitions from human decisions.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Alert contract | recipient, urgency, evidence, action choices, owner, timeout |
| Core sizing | `H=A*m/60`; disclose excluded work |
| Delivery states | created → delivered → presented → acknowledged → terminal disposition |
| Safety invariant | no eligible alert silently loses ownership |
| Duplicate rule | group only if action is truly shared and membership is auditable |
| Escalation | bounded retry, named fallback, monitored exception queue |
| Audit | policy/version, evidence, timestamps, dispositions, recovery events |
| Avoid | interpreting acknowledgment or lower volume as clinical benefit |
| Evidence limit | ILLUSTRATIVE inputs; arithmetic/state smoke VERIFIED; local workflow and source body OPEN |
