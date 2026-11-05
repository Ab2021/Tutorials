# Decision Rights and Escalation

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 9 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06, and 12; topics 4, 7, and 8 recommended  
> Version 2026.5

## 1. Why this topic exists

A control tower can identify a shipment that is likely to miss an outbound departure.
That observation does not authorize anyone to buy capacity, change a route, or amend a customer commitment.
The person who sees the risk often does not own the budget, the carrier contract, the inventory, or the customer relationship.
If those rights are unclear, teams either wait for a meeting or make unauthorized changes.
Both choices can destroy value.

Consider a shipment of temperature-sensitive components moving to a factory.
A carrier reports an arrival forecast after the receiving cutoff.
A premium vehicle might recover the schedule, but the alternate carrier has not demonstrated the required temperature controls.
A supervisor has budget to expedite but cannot waive product-quality requirements.
Meanwhile, a salesperson proposes a new delivery date before the factory has confirmed capacity.
The tower's problem is not a lack of alerts; it is a lack of an agreed route from evidence to a valid decision.

A **decision right** specifies who may make a defined decision, for what scope, under which limits, and during what period.
An **escalation** routes a decision or a risk to someone who can resolve it when the current owner cannot.
Accountability remains visible even when a task is delegated.
The tower may recommend an action without being authorized to execute it.
A carrier may accept a booking without being authorized to change the customer's contract.

This lesson designs a practical decision-rights system for rerouting, expediting, and making or changing customer promises.
It connects economic analysis with hard constraints, monetary authority, separation of duties, and time-sensitive escalation.
The worked case and thresholds are illustrative company policy, not universal industry norms.
The reader should adapt them to signed contracts, local law, operating hours, and actual delegation documents.

Good governance makes decisions faster because it resolves predictable questions before an incident.
It does not guarantee that the forecast is correct or that every chosen action succeeds.
A decision can be reasonable at the time and still lead to a bad outcome.
Review the evidence available at decision time, not just the outcome.

## 2. Learning outcomes

After this topic you can:

1. **Distinguish** a forecast, an operating plan, a service target, and a customer commitment in an exception record; exercise 1.
2. **Assign** authority, execution, consultation, and accountability for a reroute, an expedite, and a customer promise; exercises 2 and 8.
3. **Apply** scoped spending limits, hard constraints, and escalation deadlines without treating an alert as approval; exercises 3 and 4.
4. **Calculate** incremental expected cost for feasible responses and identify the authorized choice; exercise 5.
5. **Design** a time-bounded escalation and handoff, including an unavailable approver and an expired intervention window; exercise 6.
6. **Critique** decision evidence, uncertainty, conflicts, and proposed exceptions to policy; exercise 7.
7. **Audit** execution, customer communication, and outcome against the approved decision; exercise 8.

A complete answer names an individual or role, a decision, a time zone, and the applicable limit.
It explains what happens if the next person does not acknowledge the escalation.
It identifies which values are observed and which are forecasts or management assumptions.

## 3. The core mental model

```text
observed event + forecast + order/contract context
                         |
                         v
            qualify impact and latest action time
                         |
                         v
          enumerate options and hard constraints
                         |
                         v
         compare incremental cost and residual risk
                         |
                         v
     identify decision right + scope + financial limit
              /                          \
     within delegated authority       outside authority
              |                          |
              v                          v
     approve or decline           escalate with deadline
              |                          |
              +------------+-------------+
                           v
             record approval and execute once
                           |
                           v
       independently confirm physical and commercial outcome
                           |
                           v
              close, learn, and review policy
```

The diagram has two distinct gates.
A **feasibility gate** asks whether an action is allowed and operationally possible.
An **authority gate** asks whether the person approving that particular action may bind the organization.
A low-cost option can fail the first gate because it violates a safety requirement.
A high-value option can pass the first gate but fail the second until a higher authority approves it.
A recommendation cannot make an infeasible option feasible or create an absent delegation.

The inputs differ in status.
An event is a recorded observation, subject to source quality and correction.
An estimated time of arrival (ETA) is a forecast, not proof that a vehicle arrived.
The operating plan expresses the intended sequence of moves.
A service target expresses an aspiration, not necessarily an enforceable commitment.
A customer commitment is a promise under defined commercial terms.
An updated plan or forecast does not itself amend that commitment.

Decision rights attach to specific actions, not to general seniority.
A transport manager may change a route between approved facilities without changing product ownership.
A procurement manager may approve a carrier under a contract but not promise inventory to a customer.
A customer-service manager may communicate a revised proposal but not unilaterally amend a contract that requires customer acceptance.
Authority to spend money is not the same as authority to override compliance, allocate scarce inventory, or alter delivery terms.

A responsible person performs work.
An accountable owner answers for the decision and its outcome.
A consulted specialist contributes required input; consultation is not automatically a veto.
An approver grants a specifically assigned right.
An informed party receives the outcome.
Organizations can use a responsible, accountable, consulted, and informed (RACI) matrix, but it must not hide who has final approval.
Where two functions must approve, name both and define the order and timeout.
Where one function has a blocking safety right, identify that right explicitly.

Escalation is a timed control, not a synonym for forwarding an email.
A useful packet says what is needed, by when, what happens if no response arrives, and who has the next right to decide.
It carries the options, economics, constraints, and evidence version.
A named backup can decide only if the delegation covers the same action and scope.
Silence is not approval unless an authorized policy and the relevant contract explicitly establish that rule; this lesson assumes it is not.

## 4. Definitions and vocabulary

| Term | Meaning and unit or identifier | Common misuse |
|---|---|---|
| Control tower | Operating capability connecting events, decisions, and follow-through | Equating a dashboard with authority |
| Exception | A condition requiring review against an operational rule | Assuming every deviation requires intervention |
| Event | Recorded occurrence with source and time | Treating a received message as verified physical fact |
| ETA | Forecast arrival time with time zone and uncertainty | Treating the point estimate as a promise |
| Forecast | Estimate of future outcomes, possibly probabilistic | Calling the most likely outcome a guaranteed outcome |
| Plan | Intended operating sequence, quantity, and timing | Assuming the plan was executed |
| Target | Desired performance level over a stated period | Treating a target as a signed contract |
| Commitment | Promise to a counterparty on defined terms | Assuming internal intention changes external terms |
| Promise date | Delivery or availability date communicated on specified terms | Omitting whether customer acceptance is required |
| Decision right | Defined authority to approve an action in a scope | Assigning rights by title alone |
| Delegation | Documented transfer of specified authority for a period | Assuming a person automatically inherits all rights |
| Delegation of authority (DOA) | Organizational record of authorized actions and limits | Using an outdated chart as a live authorization check |
| Scope | Boundaries such as entity, site, customer, mode, and period | Applying one site's limit across the network |
| Financial limit | Maximum authorized amount, in currency and defined basis | Comparing a per-day limit with one invoice only |
| Incremental cost | Change in cost relative to a stated baseline, in currency | Including cost common to every alternative |
| Total commitment | Full amount the organization becomes obligated to pay | Checking only the first payment or a single shipment |
| Expedite | Premium action to accelerate or protect service | Assuming premium transport always arrives sooner |
| Reroute | Change to the planned path, facility, or service | Ignoring site, customs, or carrier restrictions |
| Allocation | Assignment of scarce supply to orders or demand | Letting transport choose the commercial priority implicitly |
| Hard constraint | Requirement an option must satisfy before selection | Trading a prohibited action against a financial benefit |
| Soft constraint | Preference that may be traded off if authorized | Calling a policy preference a legal prohibition |
| Escalation | Timed transfer of a decision or risk to an appropriate authority | Forwarding a message without owner or deadline |
| Acknowledgment | Explicit acceptance of a handoff by the receiving role | Treating message delivery as acceptance |
| Decision deadline | Latest time an approval can preserve a specified option | Using the customer delivery time as the approval deadline |
| Action cutoff | Latest execution time for a particular operational option | Assuming approval can occur at the same instant as departure |
| Residual risk | Risk remaining after the proposed action | Reporting gross risk avoided without what remains |
| Scenario | Internally consistent possible future with stated assumptions | Adding incompatible worst cases together |
| Probability | Assessed chance of a stated event, from 0 to 1 | Presenting judgment as a calibrated model |
| Expected cost | Probability-weighted cost for defined scenarios, in currency | Treating an average as a guaranteed outcome |
| Avoided loss | Baseline loss minus residual loss attributable to an action | Counting revenue and contribution margin as separate losses |
| Incremental net benefit | Avoided expected loss minus incremental action cost | Comparing against a baseline that already includes the action |
| Reservation | Temporarily held capacity subject to stated expiry and terms | Assuming a hold is free or final |
| Approval | Authorized agreement to a defined action, scope, and amount | Treating a general conversation as a specific approval |
| Execution | Placing or changing the operational instruction | Assuming approval means the carrier accepted it |
| Confirmation | Independent evidence of the actual booking or outcome | Equating an instruction sent with an instruction accepted |
| Audit trail | Time-stamped evidence of facts, policy, decision, and result | Keeping only a final status field |
| Segregation of duties | Separation of incompatible steps such as request and approval | Assuming any two users necessarily provide independent control |
| Emergency authority | Explicitly preapproved, bounded delegation for defined conditions | Inventing a safety waiver during an emergency |
| Policy version | Identifier and effective dates of a rule set | Re-evaluating an old decision using only today's policy |
| Idempotency key | Identifier allowing repeated requests to be recognized as one intended action | Retrying a booking as a new order after a timeout |

## 5. The mechanics, step by step

### Step 1 — Establish the event and the decision to be made

**Input:** the exception identifier, latest events, order reference, operational plan, and relevant customer terms.
**Output:** a concise decision statement with an owner and a decision deadline.

Identify the shipment, order lines, quantity, and locations affected.
Give each event's occurrence time and receipt time, with the applicable time zone.
Check whether a correction, duplicate, or older event has changed the picture.
A forecast can be late because the event feed is stale rather than because the vehicle is late.
Record the source and the age of the last reliable observation.
Ask the carrier or facility for confirmation when that uncertainty could change the choice.

Write the decision as a verb and an object.
For example, “approve a premium linehaul for order O-184 before 10:00 UTC” is actionable.
“Manage shipment risk” is not a decision statement.
Separate the decision to purchase transport from the decision to make a new customer commitment.
The same incident may need both decisions, owned by different functions.

State what the existing plan says and what the customer has actually been promised.
Record whether a proposed new date is an internal forecast, a customer proposal, or an accepted amendment.
Do not infer customer acceptance from a sent email.
A commercial owner must inspect the applicable terms before describing a change as binding.

### Step 2 — Quantify impact and preserve the baseline

**Input:** current plan, inventory, demand, capacity, customer terms, and event forecast.
**Output:** a baseline scenario and a bounded estimate of consequences.

Determine which orders and quantities are actually exposed.
Inventory at another facility may cover part of the demand, but only if it is available, allocated, and deliverable in time.
An unconfirmed transfer is an option, not existing coverage.
A facility appointment or loading cutoff can turn a small inbound delay into a long service delay.
Trace the dependency rather than applying the same delay to every downstream milestone.

Choose a baseline, such as keeping the current carrier and appointment without additional intervention.
Name the observation and plan versions used for the baseline.
Estimate loss only for the specified horizon and affected quantity.
Possible components include incremental handling, contractual charges, and lost contribution margin if supported.
Do not count gross revenue as lost profit when variable costs are avoided.
Do not count an estimated contractual charge twice in a bundled loss estimate.
Do not assume a forecast late arrival automatically establishes a breach.

Identify the decision deadline by working backward from each option's cutoff.
Subtract the time required for approval, carrier confirmation, warehouse release, and execution.
Use a conservative buffer where those durations are variable.
If an option expires at 10:30 UTC and execution requires 20 minutes, approval after 10:10 UTC cannot reliably preserve it.
The customer's delivery window may be days later, but the choice closes much earlier.

### Step 3 — Enumerate distinct actions

**Input:** baseline, network capacity, contract constraints, and stakeholder needs.
**Output:** a short list of feasible alternatives and a no-change baseline.

For rerouting, identify the origin, alternate path, destination, carrier, mode, quantity, and revised milestone.
Check whether the new route changes customs status, border crossings, temperature exposure, or insurance coverage.
Check whether the destination can receive and process the quantity at the projected time.
A route is not executable merely because a map shows a road.

For expediting, identify the premium service, quoted price, capacity-hold expiry, and what is included.
Ask whether the carrier quote is binding and whether cancellation or demurrage costs apply.
Check that pickup, handling, delivery, and release activities fit within the remaining time.
A faster linehaul cannot repair an unavailable unloading slot.

For a customer promise, distinguish notification of risk from an offer to amend terms.
List the quantity, delivery location, date range, conditions, and approvals required.
A partial delivery may preserve part of the service outcome but can create extra handling or customer disruption.
A proposed later date should not be described as accepted until the customer agrees where agreement is required.

Consider an option to wait only if its decision window and consequences are explicit.
Waiting for a better observation can have value, but may also cause capacity to disappear.
State the latest next review time and what action will follow each plausible observation.
If waiting has no explicit trigger or deadline, it is not an actionable option.

### Step 4 — Screen hard constraints before ranking

**Input:** candidate actions, safety requirements, approved-provider status, contract terms, and applicable rules.
**Output:** a feasibility record for each action, including unresolved checks.

Separate prohibited actions from costly but permitted actions.
A product-release hold cannot be removed merely because an expedite has a positive expected benefit.
A route through a restricted location cannot be chosen without the required compliance approval.
A carrier with unverified temperature capability cannot be substituted if qualification is mandatory.
A vehicle that cannot meet the required pickup window is not a feasible recovery option.

For each hard constraint, identify the evidence source and the responsible specialist.
A policy exception is available only if the policy actually permits one and the designated authority grants it.
A legal requirement or a mandatory product-safety control cannot be waived by a transport manager's spending limit.
If an essential fact is unknown, record the option as pending verification, not as cleared.
Escalate to the function that can resolve the unknown before the operational cutoff.
If the check cannot be completed in time, use the defined safe fallback.

Keep the evidence at the level of the actual move.
An approved carrier may lack an approved subcontractor or vehicle type.
A destination may accept the product but not the hazardous-goods class or temperature range in question.
A route may be contractually permitted in general but excluded for a particular customer.
A budget code may be open while the relevant entity lacks authority to incur the obligation.

### Step 5 — Compare incremental economics and risk

**Input:** feasible options, action cost, scenario probabilities or bounds, and scenario losses.
**Output:** comparable net costs and the assumptions behind the recommendation.

For action `a`, define:

```text
expected incremental cost(a)
  = action cost(a) + sum over scenarios [probability(s) × residual loss(a, s)]
```

Use the same baseline, horizon, currency, and tax treatment across alternatives.
Include action costs that arise only if the action is taken.
If the action changes probabilities, use the probabilities appropriate to that action rather than reusing baseline probabilities blindly.
If action cost depends on the scenario, place it inside the probability-weighted sum.
The simple formula assumes the action cost is certain and incurred regardless of outcome.

A positive expected net benefit relative to baseline is not authorization.
It is one input to the decision and must pass the constraint and authority gates.
A company may prefer a more expensive action when it materially reduces a severe tail outcome or protects a critical service.
That preference needs an explicit policy or authorized judgment; do not conceal it in an invented probability.
If the loss estimate is uncertain, calculate a break-even threshold and a low/high range.
Identify which assumption would reverse the recommendation.

Do not treat expected cost as the maximum possible loss.
For a rare catastrophic safety or compliance outcome, a simple expected-value comparison may be inappropriate.
Document the scenario that management wants to avoid and the applicable hard restriction.
When probabilities cannot be defended, use scenarios without pretending they add to a known distribution.

### Step 6 — Map the actual decision right

**Input:** current delegation, entity, site, action, commercial terms, amount basis, and effective time.
**Output:** named approver or approvers and the evidence that their rights cover the proposed action.

Use the delegation of authority in effect when the decision is made.
Record its version, effective dates, scope, limit, and the identity of the authorized person.
Check whether the person is on duty or has an approved delegate.
A job title in an old directory is not sufficient proof of an active delegation.

A limit needs a basis.
It might apply to incremental cost per incident, total contractual commitment, spend per day, or a rolling budget.
The same number has different consequences under each basis.
Determine whether taxes, cancellation exposure, options, and related commitments count.
Apply all relevant limits, not only the one that permits action.
Never split a single commercial commitment into multiple bookings to evade a threshold.

Check whether the action requires several rights.
A reroute may require transport approval, quality clearance, and a commercial allocation decision.
An expedite may require financial approval and an approved carrier.
A changed promise may require an internal commercial approver and the customer's acceptance.
Financial approval does not stand in for the other rights.

Distinguish approval from execution.
A tower analyst can assemble the business case and issue an instruction after approval if the operating procedure permits.
The carrier's acceptance confirms a booking; it does not create the buyer's authority retroactively.
Maintain independent review of the approval and payment where company policy requires segregation of duties.
A single employee holding multiple roles can defeat a nominal two-role control.

### Step 7 — Apply thresholds and escalate

**Input:** decision rights, requested amount, constraints, timing, and service risk.
**Output:** an authorized decision or a time-bounded escalation to the next appropriate role.

The following is an illustrative internal policy, not a recommended universal threshold.
Amounts are incremental United States dollars (USD) per incident, excluding tax, and assume no additional aggregate cap.
The threshold applies to the entire related recovery action, not an individual invoice.
Other contractual and compliance approvals remain mandatory.

| Trigger | Illustrative right or route | Required condition |
|---|---|---|
| Up to and including USD 2,000 incremental | Transport supervisor may approve an approved reroute or expedite | Approved route and carrier; no change to customer terms |
| Above USD 2,000 through USD 7,500 | Transport manager may approve the physical recovery action | Same safety and contract gates; total cost recorded |
| Above USD 7,500 through USD 20,000 | Supply-chain director may approve the physical recovery action | Finance consulted; no hard constraint waived |
| Above USD 20,000 | Executive on-call approver may decide under active delegation | Finance and relevant control functions consulted |
| Any proposed change to a customer commitment | Commercial account owner approves proposal under its commercial delegation | Customer acceptance captured when the contract requires it |
| Any product-safety or legal uncertainty | Quality or compliance owner resolves or blocks the proposed action | No monetary threshold substitutes for clearance |
| No response by acknowledgement deadline | Named on-call backup receives the packet | Backup must have the required active authority |
| Decision deadline missed | Use the preapproved safe fallback and notify the owner | Do not execute an expired or unauthorized recovery |

At exactly USD 2,000, the supervisor tier applies under this illustrative table.
At USD 2,000.01, the manager tier applies.
A request of USD 7,500 is within the manager tier; USD 7,500.01 is not.
If actual exposure is uncertain, use a defensible upper bound or obtain approval for a not-to-exceed amount.
A higher final amount requires a fresh approval before the new obligation is incurred.
An approved cap is not a license to spend the full cap without need.

Escalation triggers should cover more than money.
A high-value customer, a potential safety issue, uncertainty about a mandatory approval, or a customer commitment change may require an appropriate specialist regardless of cost.
Use a documented trigger for repeated incidents or cumulative exposure if the organization has an aggregate policy.
A low-cost action can still be out of scope because the destination, carrier, or customer is excluded.

### Step 8 — Build the escalation packet

**Input:** feasible alternatives, decision deadline, requested right, and current evidence.
**Output:** a compact, versioned packet addressed to a named approver and backup.

Include the incident and order identifiers, affected quantity, and latest verified event.
State the current customer commitment separately from the latest ETA forecast.
Say what decision is requested and why it cannot be made under the sender's delegation.
Give the latest acknowledgment time and latest decision time, each with a time zone.
Name the person who will execute the decision and the method of confirmation.

List the baseline, the proposed action, and at least one meaningful alternative where time permits.
For each, show cost, residual risk, operational feasibility, and assumptions.
Include the relevant policy version and the amount requested for approval.
Call out unresolved constraints and their designated owners.
Identify the safe fallback if the deadline passes without valid approval.
Provide a direct route to the underlying contract or source data, subject to access rights.

A useful request reads: “Approve up to USD 4,800 incremental for qualified carrier B by 10:10 UTC; capacity hold expires at 10:30 UTC; quality clearance is recorded; otherwise retain carrier A and notify the account owner.”
It is stronger than “urgent: please advise.”
The fallback should be safe and authorized, not an implied permission to take the requested action.
If the existing plan is itself unsafe, the fallback is to stop or hold the operation and contact the appropriate safety owner.

### Step 9 — Manage acknowledgments, handoffs, and clock expiry

**Input:** escalation packet, on-call roster, working calendar, and deadlines.
**Output:** an accepted owner or a documented next escalation and safe state.

Send through a channel the approver monitors during the relevant shift.
A notification receipt does not establish that the approver read or accepted the request.
Require a positive acknowledgment by the stated time.
If the first recipient is unavailable or does not acknowledge, contact the named backup.
If the backup lacks the relevant authority, continue to the next authorized on-call role.
Do not grant rights simply because the backup is reachable.

The sender remains responsible for the case until the receiving owner explicitly accepts it.
After acceptance, the new owner may request additional evidence, decide, or escalate further.
Record each transfer and the reason for it.
A handoff should include open tasks and time remaining, not just a link to the original alert.
At shift change, explicitly transfer cases that remain open.

When the deadline passes, mark the specific option as expired.
Recalculate remaining options using current capacity and forecasts.
Do not assume a capacity hold is still valid because an approval arrived late.
Notify affected parties of the fallback and the remaining risk.
A late approval may authorize a different action only if its scope and price still match.

### Step 10 — Approve precisely and execute once

**Input:** a valid decision, confirmed constraints, and a current executable quote.
**Output:** a recorded instruction and evidence of acceptance or a new exception.

The approval should identify the action, carrier, route, amount or cap, quantity, and expiry.
Record any conditions, such as quality release or customer acceptance.
A verbal approval under emergency procedures should be captured promptly with identity, time, and scope.
Do not backdate the approval to make the record look complete.

The executor checks the latest state before issuing the instruction.
If the load has already departed, the route has changed, or the quote expired, stop and reassess.
Use a unique action identifier to distinguish an intentional new booking from a retry.
After a network timeout, reconcile the carrier's booking state before sending a new request.
An idempotency key can help where both systems support the same semantics.
It does not prove that a carrier accepted a booking or that capacity exists.

Capture the carrier confirmation number, accepted route, price, and pickup window.
If confirmation differs materially from the approval, do not silently substitute terms.
Return the changed terms to the appropriate approver.
Inform warehouse, carrier, and commercial teams of the execution outcome.
Avoid disclosing unnecessary customer or shipment details to parties without a need to know.

### Step 11 — Manage the customer communication separately

**Input:** verified operational status, authorized commercial position, and contractual communication requirements.
**Output:** a traceable notification, proposal, or accepted commitment amendment.

A risk notice can explain uncertainty without making a new promise.
A proposed date should be labeled as a proposal and include any conditions.
If the contract requires customer acceptance, record that acceptance and its scope.
If a customer rejects the proposed date, preserve the original contractual position and escalate the service risk.
Do not claim that the original commitment vanished because the organization sent an update.

The account owner checks quantity, location, delivery window, and the customer's operating calendar.
The physical plan must support the proposed promise with an appropriate confidence margin.
The account owner should know whether inventory is released and transport capacity is confirmed.
If only a forecast is available, use language that accurately communicates uncertainty.
Commercial authority may also depend on contractual remedies or concessions, not only the revised date.

### Step 12 — Close on evidence and learn

**Input:** approval record, execution confirmations, physical events, invoices, and customer response.
**Output:** a closed incident or an explicitly owned residual exception.

Close the action after confirming what actually happened, not merely after sending a carrier request.
Reconcile the booked amount with the eventual invoice and investigate material variance.
Compare the actual route and delivery with the approval conditions.
Check whether the customer accepted any proposed change and whether it was fulfilled.
If the outcome remains pending, transfer it to a named owner with a next review time.

Review process measures separately from outcomes.
A prompt, well-supported decision can fail because the carrier suffers an unforeseen breakdown.
An unauthorized decision can appear successful by luck.
Measure acknowledgment time, approval lead time, expired opportunities, unauthorized actions, and rework.
Segment by severity and shift so a low overall average does not hide a night-shift failure.

Periodically test whether limits match the organization's risk appetite and operating speed.
Do not set limits from one anecdote or from the number of alerts alone.
Use a sample of incidents to inspect decision quality, feasibility checks, and actual financial exposure.
Revise the policy through the normal governance process and communicate the effective date.
Keep historical decisions tied to the version in force at the time.

## 6. Worked example

### Case and facts

Northlake Components, a fictional manufacturer, sends 240 temperature-controlled kits from its distribution center to a customer assembly plant.
The customer commitment is delivery of all 240 kits by 16:00 UTC on Tuesday.
The existing plan uses carrier A and a transfer terminal before the customer plant.
A carrier event received at 09:10 UTC reports a delayed inbound leg.
The tower forecasts that the kits may miss the terminal's 12:00 UTC departure.
The carrier has not yet confirmed the final arrival time.

The tower assesses three decision-relevant scenarios for the existing plan.
The probabilities are illustrative expert judgments, not measured model calibration.
A “late” outcome means delivery after the stated customer commitment within the defined assessment horizon.
The potential loss estimates combine documented incremental handling and estimated lost contribution, with no separate contractual penalty added.
The amounts exclude taxes and use USD per incident.

| Existing-plan outcome | Probability | Residual business loss if it occurs |
|---|---:|---:|
| On time | 0.40 | USD 0 |
| One day late | 0.45 | USD 12,000 |
| Two days late | 0.15 | USD 26,000 |

Expected baseline loss is `(0.40 × 0) + (0.45 × 12,000) + (0.15 × 26,000)`.
That is `0 + 5,400 + 3,900 = USD 9,300`.
This is an expected value across the stated scenarios, not an invoice or a guaranteed saving.
The worst modeled loss is USD 26,000, and outcomes outside the assessment horizon are not priced.

### Alternatives and constraints

Carrier B can move the same 240 kits directly to the plant for an incremental USD 4,800.
The quote is a firm incremental price, excludes tax, and is assumed payable even if the recovery fails.
The capacity hold expires at 10:30 UTC.
The warehouse needs 20 minutes after approval to release and load the kits, so the latest approval is 10:10 UTC.
Quality has confirmed carrier B's approved temperature controls for this lane and product.
The receiving plant has confirmed a feasible delivery slot.
The transport manager's active delegation covers this lane and an incremental cost up to USD 7,500 per incident.

Carrier C quotes an incremental USD 2,900 and offers a slightly earlier ETA.
Its temperature-controlled subcontractor has not been qualified for the kits.
Qualification cannot be completed before the hold expires.
Because qualification is a mandatory product-quality condition in this case, carrier C is infeasible.
Its lower price does not enter the comparison of executable options.
The tower may request qualification for future moves but cannot waive it for this decision.

A third feasible option is a partial recovery with carrier D at an incremental USD 2,100.
It delivers 120 kits early and leaves 120 kits on carrier A.
The plant confirms it can use a split delivery, and carrier D is qualified.
The relevant account owner must approve the split proposal if it changes the customer's agreed delivery terms.
The scenario loss estimates below already include the cost of the late half and extra customer handling.
Do not add another late-half loss on top of them.

| Action | Incremental action cost | On-time probability and loss | One-day-late probability and loss | Two-day-late probability and loss |
|---|---:|---|---|---|
| Keep A | USD 0 | 0.40; USD 0 | 0.45; USD 12,000 | 0.15; USD 26,000 |
| Use B for all kits | USD 4,800 | 0.85; USD 0 | 0.12; USD 12,000 | 0.03; USD 26,000 |
| Split with D | USD 2,100 | 0.60; USD 0 | 0.30; USD 7,000 | 0.10; USD 15,000 |

For B, expected residual loss is `(0.85 × 0) + (0.12 × 12,000) + (0.03 × 26,000)`.
That equals `0 + 1,440 + 780 = USD 2,220`.
Expected total incremental cost is `4,800 + 2,220 = USD 7,020`.
Expected net benefit versus keeping A is `9,300 - 7,020 = USD 2,280`.

For the split, expected residual loss is `(0.60 × 0) + (0.30 × 7,000) + (0.10 × 15,000)`.
That equals `0 + 2,100 + 1,500 = USD 3,600`.
Expected total incremental cost is `2,100 + 3,600 = USD 5,700`.
Expected net benefit versus keeping A is `9,300 - 5,700 = USD 3,600`.
The split has the lowest expected cost of the three feasible options under these assumptions.
It also leaves a 0.40 probability of a late outcome, compared with 0.15 for carrier B.
The decision maker must consider both expected cost and any separately approved service-risk policy.

### Authority and decision

At 09:20 UTC, the tower analyst prepares the case and sends it to the transport manager.
The split's USD 2,100 incremental cost exceeds the supervisor's USD 2,000 limit in the illustrative policy.
The transport manager can approve up to USD 7,500 for the physical action if all required gates are satisfied.
The analyst cannot approve it solely because the business case is positive.
The manager cannot waive carrier qualification or change customer terms with the transport delegation.

At 09:25 UTC, the commercial account owner confirms that a split delivery is permitted by the existing customer agreement and that no amendment is needed.
The owner communicates the operational change to the plant under the existing notification procedure.
This communication is not presented as a new commitment.
The manager confirms the cost basis and signs the USD 2,100 physical action at 09:35 UTC.
The executor books carrier D, receives a confirmation for 120 kits, and records the booking reference.
Carrier A confirms the remaining 120 kits.
The tower checks both actual deliveries and closes only after reconciling the invoice and customer outcome.

If the agreement instead required customer acceptance, the split would be pending until that acceptance was received.
The manager could not assume the customer's silence amended the agreement.
If acceptance could not be obtained before the action deadline, the team would compare the remaining authorized options and escalate the residual customer risk.
If carrier D's quote expired or increased beyond the approved cap, the executor would obtain a fresh decision before committing to new terms.
If no valid approval were obtained by 10:10 UTC, the defined fallback would be to retain carrier A and notify the account owner; no premium booking would be placed.

### Sensitivity and decision quality

The split's expected loss reduction before action cost is `9,300 - 3,600 = USD 5,700`.
Its break-even incremental price is therefore USD 5,700, holding scenario probabilities and losses fixed.
The quoted USD 2,100 is USD 3,600 below that break-even price.
This does not justify buying the action at any price up to USD 5,700 without the relevant approval.

The split is cheaper in expectation than B by `7,020 - 5,700 = USD 1,320`.
If the split's residual expected loss rose by more than USD 1,320 with all other values unchanged, B would become the lower expected-cost option.
That is a one-variable sensitivity, not a claim that all assumptions can change independently.
The scenario probabilities for both actions require validation against actual carrier and lane outcomes.
The decision record should preserve the judgment and the evidence available at 09:35 UTC.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It calculates the worked example's expected costs and applies the illustrative incremental-spend tiers.
It explicitly excludes the unqualified carrier before ranking.
It does not approve a booking, evaluate commercial authority, check a live delegation, or waive a constraint.
All probabilities, losses, prices, and qualification flags are illustrative inputs.
The `Decimal` type avoids binary floating-point surprises at monetary boundaries.

```python
from decimal import Decimal

D = Decimal

OPTIONS = [
    {
        "name": "keep A",
        "action_cost": D("0"),
        "qualified": True,
        "scenarios": [
            (D("0.40"), D("0")),
            (D("0.45"), D("12000")),
            (D("0.15"), D("26000")),
        ],
    },
    {
        "name": "all B",
        "action_cost": D("4800"),
        "qualified": True,
        "scenarios": [
            (D("0.85"), D("0")),
            (D("0.12"), D("12000")),
            (D("0.03"), D("26000")),
        ],
    },
    {
        "name": "split D",
        "action_cost": D("2100"),
        "qualified": True,
        "scenarios": [
            (D("0.60"), D("0")),
            (D("0.30"), D("7000")),
            (D("0.10"), D("15000")),
        ],
    },
    {
        "name": "unqualified C",
        "action_cost": D("2900"),
        "qualified": False,
        "scenarios": [],
    },
]


def expected_cost(option):
    scenarios = option["scenarios"]
    if sum((prob for prob, _ in scenarios), D("0")) != D("1"):
        raise ValueError("Scenario probabilities must sum to one")
    if any(prob < 0 or loss < 0 for prob, loss in scenarios):
        raise ValueError("Probabilities and losses must be nonnegative")
    residual = sum((prob * loss for prob, loss in scenarios), D("0"))
    return option["action_cost"] + residual


def required_tier(incremental_cost):
    if incremental_cost <= D("2000"):
        return "transport supervisor"
    if incremental_cost <= D("7500"):
        return "transport manager"
    if incremental_cost <= D("20000"):
        return "supply-chain director"
    return "executive on call"


eligible = [option for option in OPTIONS if option["qualified"]]
baseline = expected_cost(eligible[0])
for option in eligible:
    cost = expected_cost(option)
    benefit = baseline - cost
    print(
        f"{option['name']}: expected USD {cost:.2f}; "
        f"benefit USD {benefit:.2f}; "
        f"tier {required_tier(option['action_cost'])}"
    )

best = min(eligible, key=expected_cost)
print(f"Lowest expected cost: {best['name']}")
print("Excluded: unqualified C")
```

Expected output:

```text
keep A: expected USD 9300.00; benefit USD 0.00; tier transport supervisor
all B: expected USD 7020.00; benefit USD 2280.00; tier transport manager
split D: expected USD 5700.00; benefit USD 3600.00; tier transport manager
Lowest expected cost: split D
Excluded: unqualified C
```

The tier shown for the unchanged baseline is only the numeric tier for zero incremental cost.
It does not assert that continuing the existing plan requires a new supervisor approval.
The program assumes each action's fixed cost is incurred in every scenario.
Its qualification flag stands in for a completed real-world review; the code cannot establish that review occurred.
It also assumes the same currency, horizon, affected quantity, and loss basis across options.
A production implementation would need current data, identity and delegation checks, evidence retention, and transaction controls.
Those controls are not supplied by this teaching calculation.

To check a limit manually, compare the entire related incremental obligation with the applicable tier.
For USD 2,000 exactly, the illustrative supervisor limit is sufficient for the physical action if every other requirement is met.
For USD 2,000.01, request manager approval.
The program does not encode tax, cumulative exposure, emergency authority, or other organizational policies.

## 8. Data, standards, and regulatory context

### A decision record is more than an event

An event feed can establish that a source reported a shipment event at a time.
It cannot by itself establish that an employee had authority to approve a commercial action.
The decision record needs an incident identifier, relevant event versions, decision time, policy version, and approver identity.
It also needs the action scope, cost basis, required clearances, and execution result.
Keep the approval distinct from the notification sent to a carrier and the carrier's acceptance.

Minimum fields for a physical recovery include order and shipment identifiers, affected quantity, locations, and mode.
Store the quote, currency, expiry, incremental cost, and any total obligation relevant to policy.
Capture the precise conditions of approval and the effective delegation.
Where a customer commitment changes, link the proposal and any required customer acceptance.
A later correction to an event should not erase the evidence the approver actually saw.
Retain both the original decision context and the corrected record with timestamps.

### Visibility standards do not confer authority

GS1's Electronic Product Code Information Services (EPCIS) standard defines a framework for sharing visibility event data about objects and business processes.
It can carry information about what, when, where, why, and how an event occurred.
It does not determine a company's financial delegation or customer-contract amendment process.
An EPCIS event may support a decision; an appropriately authorized approval must be recorded through the organization's controls.
Consult the applicable GS1 standard and trading-partner implementation agreement for event definitions.

Electronic data interchange (EDI) messages can communicate shipment status, orders, or transaction information between partners.
A message type's existence does not establish that a specific sender may change commercial terms.
The actual agreement and implementation guide determine the meaning of the exchanged message in that relationship.
Do not infer legal acceptance merely because a technical acknowledgment was received.
A technical acknowledgment may establish receipt or syntax acceptance, not agreement to a new delivery promise.

### Financial controls and delegation

The Committee of Sponsoring Organizations of the Treadway Commission (COSO) Internal Control—Integrated Framework describes principles for designing and evaluating internal control.
It is a framework, not a universal statutory schedule of transport-approval thresholds.
Segregation of duties, authorization, and monitoring can support reliable financial control when adapted to the organization.
A documented delegation should define who may commit the relevant legal entity and how authority is suspended or delegated.
Finance and legal teams should determine which controls are mandatory in the applicable jurisdiction and entity.

The US Sarbanes–Oxley Act of 2002, section 404, concerns management assessment of internal control over financial reporting for covered issuers.
It does not prescribe the illustrative USD 2,000, USD 7,500, or USD 20,000 limits in this lesson.
Whether a particular expedite approval is material to a company's financial-reporting controls depends on its circumstances and control design.
This lesson is educational and does not determine an issuer's legal obligations.

### Safety, trade, and privacy

Transporting dangerous goods, food, medicines, or other regulated products may require specific controls on routing, handling, qualifications, and records.
Requirements vary by product, mode, route, and jurisdiction.
A generic decision matrix cannot determine whether a particular alternate carrier satisfies those requirements.
Consult the applicable regulator, contract, and responsible quality or compliance function before execution.
Do not treat an executive spending approval as a waiver of a legal obligation.

An international reroute may change borders, customs procedures, sanctions exposure, or the party acting as importer.
Check the actual trade lane and current restrictions with the appropriate specialist.
A favorable transit estimate does not establish that a border crossing is legally or operationally available.
If the shipment is held pending clearance, describe the option as pending, not approved.

Decision records may contain personal information, such as employee names and contact details.
They may also contain commercially sensitive prices, customer commitments, and locations.
Apply the organization's access, retention, and deletion policies and applicable law.
Where the EU General Data Protection Regulation (GDPR) applies, its principles include data minimization and storage limitation; applicability and lawful processing require a specific assessment.
Do not distribute an entire customer contract to every carrier when a narrower instruction is sufficient.

### Identity, timestamps, and historical evidence

Use a reliable identity source for approvers and delegated backups.
Preserve when an approval was granted and the action to which it applied.
An approval token copied to a different order, entity, amount, or expired quote should not be treated as valid for the new action.
Use time zones and record daylight-saving transitions unambiguously.
UTC is useful for a cross-border incident, while local time may be necessary for facility calendars.

Record event time separately from processing time and decision time.
A late-arriving event can explain why the decision maker did not know a fact at the time.
An approval made after an instruction is not equivalent to a prior approval, even if later ratification is permitted by policy.
A retrospective review should distinguish an authorized emergency exception from an undocumented bypass.

## 9. Industry practice and case evidence

### What published frameworks establish

COSO's internal-control framework makes authorization, control activities, and monitoring relevant to an organization's control environment.
It does not prescribe a single control-tower operating model or prove that a particular dollar threshold produces better service.
The practical implication is to define authority at the point where a financial or operational commitment is made.
That implication is an application of the framework, not a reported result from a named control-tower deployment.

GS1 publishes EPCIS as a visibility-event standard for sharing information across enterprises.
The standard can help parties agree on event semantics and exchange evidence.
It does not create a common commercial decision maker among independent firms.
A carrier, shipper, and customer retain their own contracts, constraints, and decision rights.
The implication for design is to link shared event evidence to separately governed approvals.

The US Federal Emergency Management Agency's National Incident Management System (NIMS) describes incident-management concepts, including defined roles, coordination, and the transfer of command.
NIMS addresses emergency management, not ordinary freight recovery spending.
Its useful analogy is that handoffs need explicit responsibility, communication, and a clear transfer point.
This analogy does not mean a commercial control tower must adopt NIMS command structures.

### Public company evidence and inference

In its 2023 annual report, A.P. Møller - Mærsk describes integrated logistics and the importance of its customers' supply-chain needs.
That public reporting supports the existence of multi-service logistics offerings and the importance of coordinating transport and logistics.
It does not disclose or validate the illustrative decision thresholds in this lesson.
It also does not establish that a particular customer incident was resolved through the process described here.

[INFERENCE] In an integrated logistics arrangement, a service provider may see events across multiple legs while the cargo owner retains commercial commitments to its customer.
The operating agreement therefore needs to say which changes the provider may make independently and which require the owner's consent.
That is a general governance inference, not a claim about Mærsk's internal approvals or any named customer contract.
A control tower operated by an external provider should not assume that visibility implies authority to bind the cargo owner.

A useful contract design exercise is to define the permitted response for an approved lane and service level.
Specify whether the provider may use a different carrier, reroute through another country, or incur premium cost.
State the financial ceiling, the scope of customer communication, and the required notification time.
Define a contact and an authorized backup for out-of-scope decisions.
Require evidence of carrier acceptance and the final cost.
These are design recommendations, not claims about any particular company's contract.

### Comparing operating models

A centralized tower can pool expertise and provide a consistent queue across sites.
It may also be distant from local facility constraints or customer history.
A local site can know the receiving calendar but may not see network-wide alternatives.
A hybrid model gives the tower visibility and standard triage while retaining specified local, commercial, and quality rights.
None of these models removes the need for an explicit accountable owner.

Centralization can shorten an escalation path when a qualified approver is on duty.
It can lengthen that path if all decisions must wait for a single office in another time zone.
Local empowerment can speed low-risk routine actions but create inconsistent pricing or unauthorized commitments without clear limits.
The relevant comparison is the time to a valid executable decision, not the number of management layers alone.
Measure missed cutoffs, approval quality, unauthorized spend, and customer outcomes together.

### Evidence limits

Public case descriptions often emphasize dashboards, visibility, and service improvement without publishing incident-level counterfactuals.
A reported service improvement cannot automatically be attributed to a new approval matrix.
It may reflect changes in carrier capacity, inventory, customer demand, or measurement.
A credible internal evaluation compares similar incidents, documents policy changes, and examines both speed and control failures.
Where randomization is not feasible, state the comparison's selection and confounding limitations.

An action that was approved within minutes can still fail if the warehouse did not execute it.
A carrier confirmation can still be followed by a missed pickup.
Customer acceptance of a revised date does not prove the delivery occurred on that date.
Each of these is a different stage and requires separate evidence.
Avoid reporting an approval as a realized service recovery.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Alert treated as authority | Analyst books directly from queue | Detection and approval are conflated | Unauthorized spend or route | Explicit approval gate and action audit |
| Stale delegation | Approver is absent from current roster | Cached role survives revocation | Invalid approval | Check effective delegation at decision time |
| Limit basis unclear | Staff compare invoice only | Incremental and total commitments differ | Threshold bypass | Define basis, aggregation, and scope |
| Split purchasing | Multiple small related bookings | One commitment is divided across requests | Evasion of approval limit | Aggregate related recovery actions |
| Quality gate bypass | Low quote lacks qualification evidence | Financial benefit dominates a hard constraint | Unsafe or nonconforming product | Quality-owned clearance and safe hold |
| Customer promise invented | Forecast copied into customer email as guarantee | ETA and commitment are conflated | Dispute or unmet obligation | Separate forecast, proposal, and accepted amendment |
| Unacknowledged escalation | Message sent without recipient response | Sender assumes transfer occurred | Decision missed | Timed acknowledgment and named backup |
| Expired hold | Approval arrives after capacity expiry | Deadline is not tied to execution time | Unavailable or repriced option | Reconfirm capacity and seek fresh approval |
| Approval too vague | Record says only “proceed” | Scope, amount, and conditions are missing | Wrong action or amount | Capture action, cap, carrier, and expiry |
| Duplicate booking | Retry after timeout creates two orders | Request state is unknown | Duplicate spend or capacity | Idempotency and external reconciliation |
| Wrong stakeholder | Transport approves commercial concession | Different decision rights are conflated | Unauthorized contract change | Separate physical and commercial approvals |
| Aggregate loss double counted | Penalty added to a loss figure already containing it | Components overlap | Overstated recovery benefit | Loss-component ledger and baseline |
| False precision | Probabilities reported to many decimals without evidence | Judgment looks calibrated | Misleading ranking | State provenance and sensitivity range |
| False closure | Ticket closed when instruction is sent | No confirmation or outcome check | Unresolved service failure | Confirm booking, delivery, and customer status |
| Backup lacks scope | On-call person can only approve another site | Availability is confused with authority | Delayed or invalid decision | Validate backup delegation in advance |
| Unsafe default | Silence triggers an unapproved reroute | Timeout is treated as consent | Safety or legal exposure | Safe fallback and explicit approval |
| Repriced action | Carrier confirms a higher amount | Approval applies to old quote | Spend beyond authorization | Stop and reapprove changed terms |
| Information leakage | Full order data broadcast to all carriers | No need-to-know filter | Privacy and commercial exposure | Role-based access and minimal packet |
| Hindsight blame | Review cites a later correction as known earlier | Evidence timeline is lost | Unfair assessment, hidden problems | Preserve decision-time snapshot |
| Hidden queue aging | Average response time looks healthy | Severe night-shift cases are masked | Repeated missed cutoffs | Segment by shift, severity, and deadline |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Delegation is current | Determines who can bind the organization | Revocation or entity change | Check effective version and identity at decision time |
| Limit is incremental per incident | Makes tier comparison unambiguous | Policy uses total commitment or daily aggregate | Read actual delegation and calculate all applicable bases |
| All related spend is aggregated | Prevents threshold evasion | Multiple shipments or invoices hide one recovery | Link actions to one incident and inspect total exposure |
| Quote remains valid | Makes chosen cost executable | Capacity hold expires or carrier reprices | Record expiry and reconfirm before booking |
| Probabilities sum to one | Makes expected cost interpretable | Missing scenarios or overlapping categories | Define exhaustive mutually exclusive outcomes |
| Probabilities are decision-relevant | Supports option ranking | Historical lane mix differs from current disruption | Compare forecast and actuals by lane and condition |
| Loss estimates do not overlap | Avoids double counting | Penalty included in both bundled and separate loss | Reconcile the loss-component ledger |
| Fixed action cost is certain | Supports the simple formula | Cancellation or cost depends on scenario | Put contingent cost inside scenario calculation |
| Quality clearance is valid | Makes alternate carrier feasible | Different subcontractor, lane, or product | Verify clearance at actual move level |
| Plant can receive the split | Makes the split operationally feasible | Slot is withdrawn or customer cannot use partial supply | Obtain current facility confirmation |
| Contract permits split | Avoids an unauthorized amendment | Terms require acceptance or prohibit partial delivery | Have commercial owner review actual terms |
| Fallback is safe | Allows a bounded timeout response | Existing plan is itself unsafe | Stop or hold and escalate to safety owner |
| Time estimates include execution | Makes approval deadline meaningful | Warehouse or carrier lead time varies | Use observed lead times and a buffer |
| Actions are mutually exclusive in calculation | Makes each expected cost comparable | Both carriers are committed or partial overlap occurs | Model combined actions and related costs explicitly |
| Decision horizon captures material loss | Makes comparisons meaningful | Customer harm extends beyond the modeled days | Extend horizon or state unpriced tail exposure |
| Access records are trustworthy | Supports audit and accountability | Shared credentials or edited logs | Use individual identities and controlled evidence retention |

Expected cost is a planning tool, not a substitute for legal interpretation, quality clearance, or management judgment.
The worked probabilities and USD amounts are invented for teaching and are not benchmarks.
The best expected-cost option can differ from the option preferred under a documented tail-risk or customer-service policy.
The lesson does not prescribe a single organizational chart or an emergency waiver procedure.
Actual authority depends on the relevant legal entity, delegation, contract, and effective time.

## 12. Exercises

### Exercise 1 — Classify the statement

**Time:** 10 minutes.
**Deliverable:** classify each sentence as observation, forecast, plan, target, or commitment, with a one-sentence reason.

1. The carrier's gate scan records arrival at 14:12 UTC.
2. The model predicts arrival at 16:30 UTC with a 90% interval of 15:45–18:00 UTC.
3. The warehouse intends to load the order at 17:00 UTC.
4. The network aims for 98% on-time delivery each month.
5. The customer agreement specifies delivery by Friday at 12:00 UTC.

**Acceptance criteria:** all five labels are correct; the answer does not call a forecast an accepted customer amendment.

### Exercise 2 — Assign distinct rights

**Time:** 20 minutes.
**Deliverable:** a five-row matrix assigning the accountable decision owner, executor, and consulted party for a carrier change, quality clearance, spend approval, customer notification, and contract amendment.

Assume a qualified transport team, a quality function, finance, an account owner, and the customer.
Do not assume any one role has every right.
**Acceptance criteria:** each decision has a named accountable role; the matrix distinguishes customer acceptance from internal approval and includes a safety block.

### Exercise 3 — Apply threshold boundaries

**Time:** 15 minutes.
**Deliverable:** identify the minimum illustrative physical-action approval tier for USD 1,999.99, USD 2,000.00, USD 2,000.01, USD 7,500.00, and USD 7,500.01 incremental cost per incident.

Assume the section 5 table applies and no other financial limit is binding.
Then explain whether a supervisor may authorize two related USD 1,400 bookings separately.
**Acceptance criteria:** all boundary tiers are correct; the related bookings are aggregated and nonfinancial gates remain explicit.

### Exercise 4 — Screen constraints

**Time:** 15 minutes.
**Deliverable:** classify three options as feasible, pending, or infeasible and name the missing check or reason.

Option A is an approved carrier on an approved lane with a confirmed receiving slot.
Option B uses an unqualified temperature-controlled subcontractor where qualification is mandatory.
Option C uses an approved carrier, but the receiving slot has not yet been confirmed.
**Acceptance criteria:** the answer does not rank B solely on price; it treats C as pending rather than automatically safe.

### Exercise 5 — Calculate and compare

**Time:** 20 minutes.
**Deliverable:** calculate expected incremental cost and net benefit versus baseline for a new option E.

Baseline expected loss is USD 9,300.
E costs USD 3,000 regardless of outcome.
Its three mutually exclusive outcomes have probabilities 0.70, 0.20, and 0.10 with losses USD 0, USD 8,000, and USD 20,000 respectively.
Compare E with the worked split option at USD 5,700 expected cost.
**Acceptance criteria:** show probability-weighted residual loss, total cost, benefit, and the cheaper option; do not imply the result grants authority.

### Exercise 6 — Design a timed escalation

**Time:** 20 minutes.
**Deliverable:** write an escalation packet of at most 150 words and a three-event handoff timeline.

A qualified premium option costs USD 8,200 incremental.
Capacity expires at 15:00 UTC and the warehouse needs 25 minutes after approval.
The transport manager is unreachable; the director and an authorized backup are on call.
The current plan is safe but may miss the customer's commitment.
**Acceptance criteria:** decision deadline is no later than 14:35 UTC; packet requests the correct physical approval, includes acknowledgment and fallback, and does not promise customer acceptance.

### Exercise 7 — Critique a decision

**Time:** 20 minutes.
**Deliverable:** identify at least five defects and propose one corrective control for each.

“The model says the truck will arrive Friday, so I emailed the customer a guaranteed Friday delivery.
The carrier did not respond to our first booking, so I sent another booking without checking.
I approved USD 6,000 because I can approve USD 6,000 at my own site, although this move belongs to another entity.
The alternate carrier is cheaper, so we will review its required cold-chain qualification after dispatch.
I closed the exception when the second booking request was sent.”

**Acceptance criteria:** cover promise status, duplicate execution, entity scope, hard constraint, and closure evidence.

### Exercise 8 — Design and audit a local policy

**Time:** 35 minutes.
**Deliverable:** write a one-page decision matrix and a six-field minimum decision record for an organization operating across two shifts.

Include reroute, expedite, allocation, customer communication, quality clearance, and customer amendment.
Specify a named on-call backup, authority limits and their basis, timeout behavior, and at least one segregation-of-duties control.
Explain how an auditor would trace one decision from incident to execution and customer outcome.
**Acceptance criteria:** every action has an accountable approver; no mandatory control is waived by spend; the record preserves policy version and decision-time evidence; handoffs and closure are testable.

## 13. Self-check answers

### Exercise 1

1. Observation: the scan records an event, although its source and timestamp still need quality checks.
2. Forecast: the model describes an uncertain future arrival, not an accepted delivery promise.
3. Plan: the warehouse intends an action but has not demonstrated execution.
4. Target: 98% is an aspiration measured over a month.
5. Commitment: the agreement contains an external obligation on defined terms.

A gate scan may later be corrected; recorded observation is not infallible truth.
A forecast may support a proposal, but it does not amend the agreement by itself.

### Exercise 2

A defensible matrix follows; actual authority depends on the organization's delegation and contract.

| Decision | Accountable owner | Executor | Consulted or required party |
|---|---|---|---|
| Select permitted carrier change | Transport manager | Tower or transport operator | Site and carrier |
| Clear required product controls | Quality owner | Quality specialist | Transport and product owner |
| Approve incremental spend | Authorized financial approver | Transport operator after approval | Finance where policy requires |
| Send customer risk notice | Commercial account owner | Customer-service representative | Transport for verified status |
| Amend customer commitment | Authorized commercial owner for internal approval | Account representative | Customer acceptance where required; legal as needed |

Quality can block an action that fails a mandatory requirement even if a financial approver favors it.
The customer is not merely an informed internal role when acceptance is required for an amendment.
The person executing a transaction must not be mistaken for its authorizing owner.

### Exercise 3

USD 1,999.99 and USD 2,000.00 fall within the supervisor tier.
USD 2,000.01 and USD 7,500.00 require the transport manager tier.
USD 7,500.01 requires the supply-chain director tier under the illustrative table.
Two related USD 1,400 bookings total USD 2,800 and must be treated as one incident obligation under the stated policy.
They therefore require the manager tier, not two supervisor approvals.
All results remain subject to site, entity, safety, commercial, and other relevant constraints.

### Exercise 4

A is feasible on the facts given, subject to applicable approvals and any other required checks.
B is infeasible before the deadline because the mandatory subcontractor qualification is absent.
C is pending because a receiving slot is necessary but unconfirmed.
If C's slot cannot be confirmed before its decision cutoff, exclude it from executable options for that decision.
A cheaper price cannot override a mandatory quality gate.

### Exercise 5

E's expected residual loss is `(0.70 × 0) + (0.20 × 8,000) + (0.10 × 20,000)`.
That is `0 + 1,600 + 2,000 = USD 3,600`.
Adding USD 3,000 produces expected total incremental cost of USD 6,600.
Its expected net benefit against the USD 9,300 baseline is USD 2,700.
The worked split costs USD 5,700 in expectation, USD 900 less than E.
A decision still requires feasible operations, appropriate authority, and any applicable customer approval.

### Exercise 6

The decision deadline is 15:00 minus 25 minutes, or 14:35 UTC; a buffer would make it earlier.
A sample packet is:

> Approve the qualified USD 8,200 incremental recovery for incident X by 14:35 UTC.
> Capacity expires at 15:00 UTC and the warehouse needs 25 minutes to execute.
> The current safe plan risks the existing customer commitment.
> Please acknowledge by 14:15 UTC; if unavailable, the authorized backup is next.
> Without timely approval, retain the safe current plan and notify the account owner of the risk.
> Customer terms remain unchanged unless the commercial process establishes otherwise.

One acceptable timeline is: request director acknowledgment at 14:00 UTC; contact the authorized backup at 14:15 UTC if unacknowledged; decide by 14:35 UTC or apply the fallback.
The director tier applies because USD 8,200 exceeds USD 7,500 and is at most USD 20,000.
The backup must hold equivalent relevant authority, not merely be reachable.

### Exercise 7

The model forecast is not a guarantee; the account owner must communicate uncertainty and confirm authority before making a commitment.
A second booking may duplicate the first; reconcile carrier state and use an agreed idempotency mechanism before retrying.
A site-specific right may not cover another entity; check the applicable delegation and escalate to an authorized person.
Cold-chain qualification is a mandatory pre-dispatch gate in the exercise; hold the move until quality clears it or choose another feasible option.
A sent request is not accepted execution or delivery; keep the case open until confirmation and appropriate outcome evidence exist.
Other valid observations include the absence of a price cap, deadline, customer terms, and decision-time evidence.

### Exercise 8 rubric

Award full credit for a matrix that names one accountable decision owner per action and shows all additional mandatory approvals.
The matrix must distinguish allocation from transport execution and a customer proposal from an accepted amendment.
The spending basis must identify currency, scope, aggregation, and applicable period.
The on-call backup must have an active relevant delegation, and a missed deadline must lead to a safe, authorized fallback.
Segregation of duties should separate a requester or executor from a required approver where policy calls for it.

A minimum record can combine fields, but it should preserve incident and order identifiers; decision-time evidence; action and cost; approver and delegation version; approval time and conditions; and execution plus customer outcome references.
A strong audit trace also includes source timestamps, quote expiry, clearance evidence, and any amendment acceptance.
A weak design assigns “the tower” to every role, allows silence to approve a premium action, or closes the case on message transmission.

## 14. Interview questions

1. **How does visibility differ from decision authority?**
   A strong answer says event access supports assessment but does not confer the right to spend, reroute, or amend a contract.
   A weak answer says the person who detects an exception may automatically execute the cheapest action.

2. **How would you define a spending threshold?**
   A strong answer specifies currency, incremental or total basis, incident aggregation, entity, scope, period, and boundary inclusivity.
   A weak answer provides only a number and title.

3. **What would you do when the financially best carrier is not qualified?**
   A strong answer excludes or holds the option pending the mandatory clearance and evaluates feasible alternatives.
   A weak answer trades the quality requirement against expected savings without an authorized basis.

4. **When does an ETA become a customer promise?**
   A strong answer separates a forecast, a communicated proposal, and a commitment under the actual contract.
   A weak answer treats an updated dashboard date as an automatically accepted amendment.

5. **How do you determine an escalation deadline?**
   A strong answer works backward from a capacity cutoff through approval and execution lead times, including a buffer.
   A weak answer uses the customer's final delivery date and ignores booking expiry.

6. **What does a useful escalation packet contain?**
   A strong answer names the decision, approver, options, constraints, cost, deadline, evidence, and safe fallback.
   A weak answer forwards an alert with “urgent” and no requested action.

7. **How would you handle an unavailable approver?**
   A strong answer requires acknowledgment, an authorized on-call backup, a timed escalation, and a safe fallback.
   A weak answer assumes silence or the availability of an unrelated manager grants approval.

8. **How can you compare an expedite with doing nothing?**
   A strong answer uses a common baseline and horizon, incremental action cost, probability-weighted residual loss, and sensitivity.
   A weak answer compares the expedite price with gross revenue and ignores outcome uncertainty.

9. **What evidence proves an action was executed?**
   A strong answer distinguishes approval, instruction, carrier acceptance, physical event, invoice, and customer outcome.
   A weak answer treats a sent booking request as confirmed pickup and delivery.

10. **How should an incident review judge a decision that failed?**
    A strong answer evaluates authority and the evidence reasonably available at the decision time, then examines execution and outcome separately.
    A weak answer assumes every bad outcome proves the original decision was negligent.

## 15. Key terms

- **Accountability:** ownership of a decision and its consequences; see [mental model](#3-the-core-mental-model).
- **Acknowledgment:** explicit acceptance of a handoff; see [handoffs](#step-9--manage-acknowledgments-handoffs-and-clock-expiry).
- **Action cutoff:** latest executable time for an option; see [deadlines](#step-2--quantify-impact-and-preserve-the-baseline).
- **Allocation:** assignment of limited supply to demand; see [definitions](#4-definitions-and-vocabulary).
- **Approval:** authorized agreement to a scoped action; see [execution](#step-10--approve-precisely-and-execute-once).
- **Audit trail:** retained decision and execution evidence; see [data context](#8-data-standards-and-regulatory-context).
- **Baseline:** comparison state without the proposed incremental action; see [economics](#step-5--compare-incremental-economics-and-risk).
- **Commitment:** promise made under specified terms; see [customer communication](#step-11--manage-the-customer-communication-separately).
- **Constraint:** requirement or preference governing an option; see [screening](#step-4--screen-hard-constraints-before-ranking).
- **Decision deadline:** latest approval time that preserves an option; see [worked case](#6-worked-example).
- **Decision right:** scoped authority for a specific decision; see [mapping](#step-6--map-the-actual-decision-right).
- **Delegation:** documented grant of authority; see [mapping](#step-6--map-the-actual-decision-right).
- **Escalation:** timed routing to a person who can resolve a decision; see [thresholds](#step-7--apply-thresholds-and-escalate).
- **ETA:** estimated time of arrival, a forecast; see [vocabulary](#4-definitions-and-vocabulary).
- **Execution:** issuing the authorized operational instruction; see [execution](#step-10--approve-precisely-and-execute-once).
- **Expected cost:** probability-weighted loss plus relevant action cost; see [calculation](#7-code-query-or-calculation).
- **Expedite:** premium recovery intended to protect or accelerate service; see [alternatives](#step-3--enumerate-distinct-actions).
- **Fallback:** authorized safe response when the requested decision is unavailable; see [handoffs](#step-9--manage-acknowledgments-handoffs-and-clock-expiry).
- **Forecast:** estimate of an uncertain future; see [mental model](#3-the-core-mental-model).
- **Hard constraint:** condition that cannot be traded against financial benefit without valid authority; see [screening](#step-4--screen-hard-constraints-before-ranking).
- **Idempotency key:** identifier for recognizing retries of one intended action; see [execution](#step-10--approve-precisely-and-execute-once).
- **Incremental cost:** difference from a defined baseline; see [economics](#step-5--compare-incremental-economics-and-risk).
- **Plan:** intended operating sequence; see [mental model](#3-the-core-mental-model).
- **Reroute:** change to an operational path or facility; see [alternatives](#step-3--enumerate-distinct-actions).
- **Residual risk:** exposure remaining after a response; see [worked case](#6-worked-example).
- **Segregation of duties:** separation of incompatible responsibilities; see [controls](#financial-controls-and-delegation).
- **Target:** desired performance rather than an external promise; see [mental model](#3-the-core-mental-model).
- **Threshold:** defined trigger for a different approval or escalation route; see [thresholds](#step-7--apply-thresholds-and-escalate).

### Review cards

These short prompts reinforce the links between evidence, authority, and execution.
Use them without referring to the worked calculations, then check the referenced section.

1. A carrier has confirmed a route, but the buyer's approval has not arrived.
   The confirmation establishes a carrier response; it does not establish valid buyer authorization.
   See [execution](#step-10--approve-precisely-and-execute-once).
2. A manager approves USD 4,000, and the confirmed quote rises to USD 4,300.
   The executor must obtain the appropriate fresh approval before accepting the new obligation.
   See [thresholds](#step-7--apply-thresholds-and-escalate).
3. A quality owner approves carrier B for product X, but the load contains product Y.
   The existing approval does not necessarily cover the actual product.
   Check the applicable qualification before dispatch.
4. A customer receives a proposal but has not responded.
   Do not record an accepted amendment when the applicable terms require acceptance.
   Preserve the original commitment and manage the open risk.
5. An authorized backup is on call but is delegated only for another legal entity.
   Contact the next person whose active delegation covers this entity and action.
   Availability does not expand scope.
6. The approval arrives after the carrier's capacity hold expires.
   Reconfirm availability and price before executing.
   If the action has changed, route the new terms for approval.
7. A late event correction shows the shipment was already moving when the manager decided.
   Retain both the original decision-time view and the correction.
   Evaluate the decision against the evidence then available.
8. The customer accepts a new delivery date but the recovery carrier rejects the booking.
   The commercial amendment and physical execution are separate facts.
   Reassess the plan and communicate the new risk through the authorized owner.

### Compact decision-record example

The following example is fictional and shows the structure of a record, not an actual approval.

| Field | Entry |
|---|---|
| Incident | NL-184; order O-184; 240 kits |
| Evidence | Carrier A delay report received Tuesday 09:10 UTC; forecast version 2 |
| Existing commitment | Customer delivery by Tuesday 16:00 UTC |
| Requested physical action | Carrier D for 120 kits; carrier A retains 120 kits |
| Cost basis | USD 2,100 incremental for the complete incident, excluding tax |
| Decision authority | Transport manager; active lane and entity delegation, version 4 |
| Other clearance | Quality qualification confirmed; commercial owner confirms split permitted |
| Decision | Approved at 09:35 UTC, subject to confirmed quoted terms |
| Execution | Carrier D booking and carrier A remaining-load confirmation recorded |
| Outcome | Actual delivery and invoice reconciliation remain open until evidence arrives |

The entry's last row deliberately does not assert success at the moment of approval.
An outcome should be filled with the actual event reference and customer status when available.
A correction should be appended or versioned rather than replacing the record of what the decision maker saw.
The precise retention period and access controls follow the organization's applicable policy and law.

## 16. Sources

The sources below support the standards, control, regulatory, and public-case statements in this lesson.
They do not substantiate the fictional case, its probabilities, or its illustrative approval limits.

1. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis), including EPCIS 2.0 resources. Visibility-event concepts and standard scope; not financial authority.
2. GS1, [EPCIS and CBV implementation guideline](https://www.gs1.org/standards/epcis-and-cbv-implementation-guideline/current-standard). Event implementation context; trading partners still need their own agreements.
3. COSO, [Internal Control—Integrated Framework](https://www.coso.org/internal-control). Internal-control framework and principles; not a prescribed control-tower dollar matrix.
4. US Congress, [Sarbanes–Oxley Act of 2002, Public Law 107-204](https://www.govinfo.gov/content/pkg/PLAW-107publ204/pdf/PLAW-107publ204.pdf), section 404. Primary statutory text; applicability requires entity-specific assessment.
5. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially article 5. Primary legal text; applicability and obligations depend on processing context.
6. Federal Emergency Management Agency, [National Incident Management System](https://www.fema.gov/emergency-managers/nims). Incident coordination and roles; used here as an analogy, not as a commercial freight rule.
7. A.P. Møller - Mærsk, [Annual reports](https://www.maersk.com/investors/financial-results/annual-reports), 2023 annual report. Public company context on integrated logistics; no claim about the fictional decision case.
8. NIST, [Security and Privacy Controls for Information Systems and Organizations, SP 800-53 Rev. 5](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final). Reference for access and audit controls; applicability depends on the organization.

Legal, tax, safety, and contractual interpretation require qualified advice and current jurisdiction-specific sources.

## 17. Links

- [Module 19 index](_index.md).
- [Module 19 survey and lab sections](../../../course/19_control_towers_visibility_events.md).
- [ETA prediction](04_eta-prediction.md) for uncertain arrival forecasts and promise implications.
- [Delay propagation](05_delay-propagation.md) for downstream constraints and scenario losses.
- [Exception management and queue design](07_exception-queue.md) for case ownership and queue state.
- [Alert economics](08_alert-economics.md) for the cost of attention and missed exceptions.
