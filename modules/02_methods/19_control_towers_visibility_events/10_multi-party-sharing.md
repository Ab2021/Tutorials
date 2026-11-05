# Multi-Party Data Sharing and Trust

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06, and 12; Module 19 topics 02, 07, and 09  
> Version 2026.5

## 1. Why this topic exists

A control tower can see a late shipment only if someone produces, transmits, and permits use of a relevant event.
The shipment may cross a supplier, a carrier, a port terminal, a customs broker, and a buyer.
Each participant knows a different part of the journey.
Each also has a legitimate interest in limiting disclosure.

The buyer wants enough warning to reserve an alternate production slot.
The carrier wants a realistic appointment and fewer calls from the buyer.
The supplier wants proof that it tendered the shipment on time.
The terminal wants a manageable queue, not a flood of speculative requests.
A tower operator wants a consistent event history, but may not have rights to use that history for every product it offers.

A request for “all available data” confuses technical availability with permission.
Vehicle coordinates can expose driver behavior and carrier routes.
Purchase-order lines can expose prices and customer relationships.
A projected arrival time can be mistaken for a contractual promise.
A combined dataset can reveal commercially sensitive patterns even when each input looked innocuous.

Insufficient sharing also has a cost.
Without the carrier's exception event, the buyer may book labor for a truck that cannot arrive.
Without a shipment identifier, the event may attach to the wrong order.
Without a named owner, nobody knows who can correct an inaccurate arrival event.
A tower that merely collects data does not solve these decisions.

The design question is therefore not “How do we centralize everything?”
It is “Which participant may use which fact, for which decision, for how long, and with what accountability?”
The answer combines a business case, an agreement, a data contract, technical controls, and a process for resolving disputes.

**Decision in this lesson:** select the smallest sharing arrangement that lets a buyer and carrier manage missed delivery appointments while respecting each participant's rights.
The worked example evaluates a bounded pilot, not a general authorization to share every shipment or use data to train a model.

**Audience:** tower product owners, operations managers, procurement staff, data stewards, security practitioners, carrier managers, and analysts.
Legal and privacy specialists must review actual agreements and applicable law.
This lesson is educational and is not legal advice.

## 2. Learning outcomes

After this lesson you can:

1. **Map** the participants, decisions, data contributions, and incentives for a sharing use case.
   - Exercise 1 asks you to identify a missing actor and its incentive.
   - A good map distinguishes the party producing data from the party authorizing a use.
2. **Specify** a purpose-limited data-sharing agreement with field-level rights and retention.
   - Exercise 2 tests whether a new proposed use falls within that agreement.
   - A data specification is not a substitute for a signed agreement.
3. **Choose** controls for access, onward sharing, correction, revocation, and audit.
   - Exercises 3 and 4 examine row-level authorization and revocation.
   - A control must identify an owner, a trigger, and evidence of execution.
4. **Calculate** each participant's pilot economics and distinguish a positive joint result from a fair deal.
   - Exercise 5 asks you to compute incremental net benefits.
   - A positive total does not establish that every party will participate.
5. **Evaluate** event standards and relevant privacy obligations without treating a standard as legal permission.
   - Exercise 6 requires a standards-versus-rights distinction.
   - The correct answer is specific to the event and its use.
6. **Design** an implementation and exit plan with measurable service levels and escalation.
   - Exercise 7 requires operational controls and an exit test.
   - A viable exit includes downstream copies and documented exceptions.
7. **Critique** aggregation, derived data, and model-use proposals for leakage and incentive effects.
   - Exercise 8 uses a small-group aggregate.
   - A technical transformation does not by itself settle contractual rights.

## 3. The core mental model

Treat shared visibility as a set of **conditional, accountable information flows**, not a single shared database.
A useful flow has five components: a decision, a permitted purpose, a minimum data product, enforcement, and a feedback loop.

```text
Operational decision: reschedule an at-risk delivery
            |
            v
Buyer <--- permitted shipment status ---> Carrier
  |                                          |
  | purpose, fields, parties, expiry         | source event and corrections
  +-------------------+----------------------+
                      v
              Tower operator
       identity -> policy -> filter -> delivery
                     |           |
               audit trail    exception owner
                     |           |
               access review  correction/revocation
```

The tower may act as an intermediary rather than as the owner of the underlying commercial relationship.
Its ability to receive an event does not imply permission to sell it, publish it, train on it, or give it to a second customer.
Different recipients can see different views of the same physical event.

Start with a decision and a time window.
For a missed delivery slot, a predicted arrival window and the relevant shipment identifier may be sufficient.
Continuous vehicle position, driver identity, freight rate, and the carrier's other customers may not be necessary.
Ask the user who will act, when the decision closes, and which information would change the action.

Then draw the rights boundary.
A boundary is not only a network firewall.
It includes the agreement's scope, the audience, the requested purpose, the retention period, the rules for derived products, and a way to correct or withdraw access.
Make those rules executable where feasible and review them where human judgment is required.

Finally close the feedback loop.
A partner can report that a record is wrong, a recipient is no longer entitled, or an event is missing.
The tower records the request, investigates the source, communicates the outcome, and applies the agreed correction or revocation.
An audit trail supports accountability but must itself be protected and retained only as justified.

**Trust is earned by predictable behavior.**
It increases when parties receive a useful return, can see their obligations, can contest errors, and can verify access.
It is not equivalent to assuming all participants are benign, nor to installing a blockchain.
Cryptographic or distributed technology can strengthen selected evidence properties, but cannot determine business purpose, identity, accuracy, or legal entitlement on its own.

### Four separations to preserve

| Separation | Why it matters | Example |
|---|---|---|
| Possession versus permission | A system can technically receive a field it may not be entitled to use. | A carrier sends driver name; the agreed view excludes it. |
| Source versus recipient | The source can correct an event even when the recipient has made an operational decision from it. | Carrier corrects an erroneous arrival. |
| Forecast versus commitment | An estimate expresses uncertainty; a promise allocates responsibility. | ETA is 14:00–15:00; the committed appointment is 15:30. |
| Joint value versus partner value | A positive total can still leave the data provider worse off. | Buyer saves labor while carrier pays all integration cost. |

### A useful counterfactual

Suppose a buyer can decide whether to move an appointment at 10:00 local time.
An arrival estimate received at 10:05 is not necessarily useless, but it is too late for that particular decision cutoff.
An estimate received at 09:30 is not useful if it refers to another shipment or cannot be acted upon under the agreement.
Timeliness, correctness, authorization, and decision relevance must all hold.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse or relevant unit |
|---|---|---|
| Control tower | A coordinated capability for visibility, exception handling, and decision execution. | Treating a dashboard alone as a tower. |
| Data subject | An identified or identifiable natural person in contexts such as the EU General Data Protection Regulation (GDPR). | Assuming business-to-business data cannot contain personal data. |
| Data controller | Under GDPR, an entity determining purposes and means of processing personal data. | Assigning the role solely by a vendor's job title. |
| Data processor | Under GDPR, an entity processing personal data on a controller's behalf. | Assuming every data recipient is automatically a processor. |
| Data provider | Party that makes a dataset or event available. | Assuming the provider has all necessary third-party rights. |
| Data recipient | Party authorized to receive specified data. | Treating its employees, affiliates, and vendors as automatically included. |
| Source system | System from which a record originates. | Equating the first tower system with the original source. |
| System of record | Agreed authoritative source for a specified fact and period. | Assuming a single system is authoritative for all facts. |
| Event | A record that something happened, or was asserted to have happened, at a time. | Treating a predicted event as an observed event. |
| Event time | Time of the underlying occurrence, with time zone or UTC offset. | Substituting arrival at the tower. |
| Processing time | Time the system handled the event. | Assuming it proves when the physical activity occurred. |
| Event identifier | A source-scoped or globally unique identifier for an event. | Using shipment ID alone to deduplicate distinct events. |
| Shipment identifier | An identifier for the movement being monitored. | Assuming it identifies a purchase-order line uniquely. |
| Estimated time of arrival (ETA) | Predicted arrival time or interval at a specified location. | Treating an estimate as a commitment. |
| Commitment | An accepted obligation or promise, such as a confirmed appointment. | Calling a model prediction a confirmed slot. |
| Data contract | Technical and operational specification for a data product, including schema and service expectations. | Treating a schema file as the legal sharing agreement. |
| Sharing agreement | Documented terms governing parties, purposes, rights, obligations, and remedies. | Assuming one generic clause resolves every downstream use. |
| Purpose limitation | Restriction of processing or use to stated purposes, subject to applicable law and terms. | Assuming a broad business label permits unrelated uses. |
| Data minimization | Selecting only data needed for a stated purpose. | Deleting identifiers that are essential to perform the authorized decision. |
| Attribute-based access control (ABAC) | Authorization based on attributes such as recipient, shipment relationship, purpose, and time. | Assuming a user's job title alone proves entitlement. |
| Row-level security | Filtering which records an identity can access. | Believing column masking also filters rows. |
| Column masking | Restricting visibility of selected fields. | Believing masked output prevents all inference. |
| Onward sharing | Disclosure by a recipient to another party. | Ignoring contractors and exports. |
| Retention | Period and conditions for keeping data. | Assuming deletion of the live table deletes backups and exports. |
| Revocation | Removal or narrowing of a party's future access under agreed terms. | Assuming revocation automatically erases every historical copy. |
| Correction | Rectification or supersession of a record with provenance. | Silently rewriting history without notifying recipients. |
| Lineage | Record of source, transformations, and downstream destinations. | Confusing a data catalog label with proof of every export. |
| Audit log | Protected record of relevant actions and policy decisions. | Logging every sensitive payload without a retention rationale. |
| Aggregation | Combining records into a summary. | Claiming a summary cannot reveal a participant in a small group. |
| Pseudonymization | Replacing direct identifiers while retaining a means of linkage. | Calling the resulting data anonymous by default. |
| De-identification | Techniques intended to reduce identification risk. | Assuming a fixed transformation eliminates all re-identification risk. |
| Derived data | A result computed from input data, such as a delay score. | Assuming its commercial or legal rights are self-evident. |
| Service-level objective (SLO) | Measurable desired service performance for a defined population and window. | Calling an aspiration an enforceable service-level agreement. |
| Electronic Data Interchange (EDI) | Structured exchange of business documents between systems. | Treating an EDI message as an authorization mechanism. |
| Electronic Product Code Information Services (EPCIS) | GS1 standard for exchanging visibility event data. | Treating an EPCIS event as proof of truth or permission. |
| Application programming interface (API) | Defined interface through which software exchanges requests or data. | Treating authentication to an API as authorization for all records. |
| Data residency | Requirements or commitments concerning where data is stored or processed. | Treating storage location alone as proof that transfers are compliant. |
| Clean room | Controlled environment for joint analysis under specified restrictions. | Assuming the environment itself makes every output safe. |
| Differential privacy | A mathematical privacy framework that bounds the effect of an individual's inclusion under specified parameters. | Calling ordinary aggregation differentially private. |

## 5. The mechanics, step by step

### Step 1 — Name the decision and its deadline

**Input:** an operational problem, its decision-maker, and the cost of acting late.
**Output:** a one-sentence decision statement and decision cutoff.

1. Identify a concrete action, such as moving a dock appointment or expediting a component.
2. Name the person or role authorized to take it.
3. Identify when the action stops being useful.
4. Identify the shipment population for which the action applies.
5. Record the baseline process and the information available without the proposed sharing.
6. State which decisions remain outside the use case.

For a missed appointment, the buyer's receiving planner may need an ETA interval before 10:00 local time.
A carrier dispatcher may need the resulting slot change before the driver reaches the facility.
These are two distinct information flows with different recipients.

**Branch:** if no operational action would change, do not collect another field merely because the platform supports it.
If a second action is proposed, assess it as a separate purpose rather than quietly expanding the first.

### Step 2 — Map the parties and their incentives

**Input:** the decision statement and existing commercial relationships.
**Output:** a participant map with gains, costs, and rights dependencies.

1. List the entity producing each fact and the entity controlling the source system.
2. List the operational recipient and any tower operator processing on its behalf.
3. Identify affected parties not directly connected to the interface, including drivers and subcontracted carriers where relevant.
4. Identify subcontractors, cloud providers, and affiliates that may see or process the data.
5. List each party's expected benefit, integration work, recurring cost, and principal risk.
6. Ask who can veto the arrangement and who can sponsor it.
7. Confirm whether a party has authority to provide the data and authorize the proposed use.

A carrier may improve dock turn time yet fear that a buyer will use raw location histories to benchmark unrelated lanes.
A buyer may reduce overtime yet fear that supplier order details will leak to a competitor.
A tower operator may want cross-customer analytics, while participants may consent only to shipment execution.
A neutral facilitator should document these tensions instead of assuming data sharing is costless.

**Branch:** if the provider receives no credible benefit, offer a reciprocal operational flow, a service payment, lower manual reporting burden, or another explicit allocation.
If no arrangement makes all necessary participants willing, change the scope or do not launch.
A positive aggregate business case is not consent.

### Step 3 — Inventory data and classify sensitivity

**Input:** candidate source fields, source owners, and decision statement.
**Output:** field inventory with classification and disposition.

1. For each candidate field, record definition, source, granularity, refresh frequency, and expected quality.
2. Identify identifiers linking an event to a shipment, partner, location, person, or commercial relationship.
3. Mark fields that may be personal data or commercially confidential, subject to specialist review.
4. Consider combinations: location plus time may identify a driver even if the name is omitted.
5. Record the proposed recipient, purpose, retention, and allowed transformations for each field or field group.
6. Label fields as required, conditionally required, or excluded.
7. Document why an excluded field is not needed for the decision.

| Candidate | Decision use | Proposed treatment |
|---|---|---|
| Shipment ID | Match the event to an active load. | Share with entitled buyer and carrier. |
| Event type | Distinguish departed, delayed, and arrived. | Share using agreed codes. |
| Event time and offset | Establish when the event occurred. | Share with provenance and offset. |
| ETA interval | Decide whether a slot change is needed. | Share estimate and calculation time. |
| Driver name | Not necessary for appointment planning in this case. | Exclude from buyer feed. |
| Continuous GPS trace | Not necessary if ETA and exception status suffice. | Keep outside this buyer feed. |
| Freight rate | Not needed for this decision. | Exclude. |
| Delay reason | May support action but can disclose sensitive detail. | Use an agreed coarse code; restrict free text. |

**Branch:** when a field is useful but sensitive, first ask whether a lower-granularity alternative preserves the decision.
A coarse delay reason may suffice where a detailed driver note would expose a person's health information.
Do not claim that coarse codes are anonymous without a contextual assessment.

### Step 4 — Define the permitted purpose and party boundary

**Input:** inventory, participant map, and legal review.
**Output:** a negotiated rights matrix.

1. Name the provider, recipient, tower operator, and permitted subcontractors by legal entity or an agreed mechanism.
2. Define the permitted operational purpose in terms a reviewer can test.
3. Define the covered shipments, geography, dates, and data categories.
4. State permitted actions: receive, display, store, correct, and distribute within the authorized audience.
5. State prohibited or separately approved actions: resale, competitor benchmarking, unrelated marketing, and model training where applicable.
6. Define the treatment of aggregates, derived data, incident evidence, and backups.
7. Specify whether and how recipients can export data and onward-share it.
8. Agree on access termination, return or deletion, exceptions, and evidence of completion.
9. Allocate responsibilities for notices, legal bases, cross-border transfers, security, incidents, and audits as appropriate.
10. Obtain qualified legal review; do not infer controller or processor roles from a diagram alone.

The agreement and the technical policy must match.
A contract allowing a buyer to view only its own shipments is not implemented by a dashboard with an unrestricted download button.
Conversely, an overly restrictive filter may prevent a party from performing the agreed service.

**Branch:** when the purpose changes, pause the new use until the relevant parties approve and the applicable legal requirements are addressed.
Changing a policy label after the fact is not approval.

### Step 5 — Agree the event and quality contract

**Input:** rights matrix and source capabilities.
**Output:** versioned event specification and operational service objectives.

1. Define event names and their business meaning.
2. Distinguish observed occurrences from estimates and commitments.
3. Require an event ID, shipment ID, source, event time, and processing or receipt time.
4. Specify the time-zone representation, units, and code lists.
5. Define how the recipient handles duplicates, late events, reversals, and corrected events.
6. Agree on validation errors and a route for repair.
7. State expected coverage, freshness, and accuracy measures with denominators and exclusions.
8. Specify monitoring frequency, accountable owner, and escalation path.
9. Version the schema and announce breaking changes before deployment.
10. Provide sample messages that use synthetic or approved data.

For example, measure freshness as the fraction of valid in-scope events received within 15 minutes of event time.
The denominator is all valid in-scope events for which the source can establish an event time, not just records arriving through the API.
Measure missing coverage against an independently reconciled shipment manifest where possible.
A source clock error can make a freshness metric look better or worse; check clock synchronization and negative delays.

**Branch:** if an event is late but still materially changes a decision, deliver it with its original event time and a late flag.
If a correction arrives, retain its relationship to the original event and notify affected recipients under the agreed workflow.

### Applying the event contract to difficult situations

The schema and the service objective need decisions for conditions that do not fit the happy path.
Use the following scenarios during partner onboarding.
Each scenario distinguishes the physical occurrence, the source assertion, and the tower's processing.

**A shipment is tendered but never appears in the feed.**

- The buyer's manifest shows the load as in scope.
- The carrier feed has no event for the load.
- The event-freshness metric alone cannot describe this absence.
- Record the load as missing against the manifest-based coverage measure.
- Ask the carrier whether the load was accepted, canceled, subcontracted, or incorrectly mapped.
- Do not fabricate a departure event from the planned departure time.
- Route a missing-data exception to the carrier's data steward.
- Tell the receiving planner that no observed movement is available.
- The planner may use a fallback call under the agreed operating procedure.
- Close the exception only when the record is corrected or a valid exclusion is documented.

**A departure event arrives after an arrival event.**

- Preserve both source event times and tower receipt times.
- Order the physical timeline by event time only if the source timestamps are credible.
- Show the late arrival of information separately from the earlier physical departure.
- Do not overwrite the arrival event merely because the departure message came later.
- Recompute any affected state under the agreed late-event rules.
- If the change alters a previously communicated decision, notify the responsible recipient.
- Record whether the correction was timely enough to alter an action.
- Investigate repeated delays in the carrier's transmission pipeline.

**A duplicate message is delivered after a retry.**

- Compare a stable source event identifier and source identity.
- Treat a retransmission of the same event as one business occurrence.
- Preserve technical delivery metadata if needed for troubleshooting.
- Do not use shipment ID alone as the deduplication key.
- One shipment can have several valid departure, stop, and arrival events.
- If the same event ID has a changed payload, do not silently discard the difference.
- Apply the agreed correction or conflict workflow.
- Ask the source to explain whether it was a correction or an identifier error.

**A source reports an impossible clock sequence.**

- An event may appear to occur after its receipt time because a clock is fast.
- A local timestamp may be misread as UTC.
- An offset can also be missing or wrong at a daylight-saving transition.
- Mark the timestamp quality issue instead of manufacturing a corrected time.
- Compare against other documented events and source clock monitoring.
- Ask the source owner to provide an authoritative correction.
- Exclude invalid timestamps from a freshness calculation only under documented rules.
- Report the exclusion count alongside the metric.
- An improving freshness percentage accompanied by rising exclusions is a warning, not proof of improvement.

**The ETA interval changes several times.**

- Each ETA is a forecast made at a particular calculation time.
- Retain the issue time and the predicted arrival interval.
- The latest forecast should not erase what the planner knew earlier.
- Track whether a new estimate crosses a decision threshold.
- Avoid sending repeated alerts when no operational decision changes.
- The planner's accepted appointment remains a separate commitment.
- Compare forecasts with actual arrival for quality assessment.
- Restrict any cross-partner model evaluation to the permitted scope.

### Practical service-level design

A service objective needs a precise numerator, denominator, measurement period, and owner.
The following illustrative targets apply only to a negotiated pilot.
They are targets for discussion, not universal industry benchmarks.

| Measure | Numerator | Denominator | Example target | Owner |
|---|---|---|---|---|
| Shipment coverage | In-scope shipments with at least one valid required status. | All in-scope shipments in the reconciled manifest. | At least 97% per week. | Carrier integration lead. |
| Timely events | Valid events received within 15 minutes of event time. | Reconciled valid events with reliable source time. | At least 95% per week. | Carrier integration lead. |
| Decision lead time | Actionable exceptions received before the agreed cutoff. | All confirmed at-risk shipments with an observable cutoff. | At least 90% per week. | Tower operations lead. |
| Correction acknowledgement | Disputes acknowledged within one business day. | All valid correction requests received. | At least 95% per month. | Source data steward. |
| Access removal | Terminated grants disabled by the effective time. | All grants scheduled to terminate in the period. | 100% per month. | Identity administrator. |

Coverage and timeliness answer different questions.
A feed can deliver every event it sends within one minute while failing to send half of the required events.
A feed can have complete coverage but provide its events too late for a slot decision.
A high decision-lead-time rate can coexist with poor event coverage if the denominator excludes unknown shipments.
Review all three measures together.

The manifest is also imperfect.
It may contain canceled shipments, duplicates, and loads assigned to a different carrier.
Reconciliation therefore needs an agreed cutoff, exclusion reason, and appeal path.
Neither party should unilaterally redefine the denominator after seeing an unfavorable result.
Record both the original and revised measure if an error is corrected.

For each objective, distinguish a target from a contractual remedy.
An objective may trigger operational investigation without creating a financial penalty.
If a service-level agreement includes credits or other remedies, legal and commercial owners must define them.
An operational target is not automatically a supplier warranty.

### Designing the rights matrix for implementation

The matrix needs enough precision for an engineer to enforce it and for a business owner to recognize an incorrect result.
Use a row per meaningful combination of party, data group, and purpose.
Do not put a single unrestricted “shared data” entry in the agreement.

For each row, record:

1. The provider and any upstream party whose rights constrain disclosure.
2. The receiving legal entity, not just a brand or application name.
3. The permitted individual roles or service identities.
4. The specific operational purpose and covered shipment population.
5. The fields or field group available to that audience.
6. The allowed operations, such as display, export, correction, and onward delivery.
7. The effective start and end, including the time zone.
8. The maximum retention and the treatment of backups or mandatory records.
9. The permitted use, if any, of derivatives and aggregates.
10. The owner who can approve a change and the evidence of that approval.

The resulting policy must be tested with concrete requests.
A carrier dispatcher can see the carrier's assigned load but not a competing carrier's load.
A buyer planner can see the buyer's covered load but not another customer's load.
The tower support role may have restricted diagnostic access, subject to a separate process.
An analyst does not gain permission to export all rows merely because the dashboard displays some of them.
Where an emergency override exists, its justification, approver, expiry, and review should be explicit.

### Onboarding handoff checklist

Before the first production event, the business and technical owners jointly check the following.
The checklist is an example of evidence to request, not proof that every applicable requirement has been met.

| Check | Evidence | If absent |
|---|---|---|
| Purpose is approved. | Signed scope and named business owner. | Do not activate the feed. |
| Source authority is assessed. | Provider and upstream rights review. | Resolve the authority gap. |
| Fields are minimized. | Approved field inventory and sample projection. | Remove or justify excess fields. |
| Identity is established. | Verified partner and service identities. | Do not issue active credentials. |
| Entitlements are populated. | Covered-shipment mapping and effective periods. | Deny access until mapping is available. |
| Contract is implementable. | Policy rules mapped to agreement clauses. | Resolve conflicting or ambiguous terms. |
| Test events are safe. | Synthetic or expressly approved examples. | Replace unapproved samples. |
| Time semantics are clear. | Offset, event time, receipt time, and code list. | Correct the interface specification. |
| Quality has an owner. | Named source steward and escalation route. | Assign an accountable owner. |
| Corrections are supported. | Linked correction example and notification path. | Complete the workflow before relying on the feed. |
| Exit is defined. | Termination triggers and destination inventory. | Record and approve an exit procedure. |
| Incident response is coordinated. | Contact roster and agreed notification process. | Align incident handling before launch. |

The same discipline applies when a partner adds a subcontractor or a new geographic region.
A previously approved data flow does not automatically cover a new recipient, population, or purpose.
Review the change against the agreement before provisioning access.

### Incident and dispute decision paths

Not every incorrect event is a security incident.
Not every access denial is evidence of an attack.
The operating team needs separate but connected workflows.

For an incorrect or disputed event:

1. Capture the event identifier, affected shipment, disputed field, and reporter.
2. Restrict the report to people who need it for investigation.
3. Mark the event disputed without removing evidence of the original assertion.
4. Ask the source owner to examine the underlying operational record.
5. Record the source's conclusion and any linked correction.
6. Notify authorized recipients whose decisions could be affected.
7. Close the dispute with the reason, owner, and time.

For a suspected unauthorized disclosure:

1. Preserve relevant evidence without making unnecessary additional copies.
2. Alert the designated security and privacy contacts under the incident plan.
3. Contain the access path where safe to do so.
4. Determine which identities, records, destinations, and time periods are involved.
5. Follow contractual and applicable legal notification requirements.
6. Document containment, remediation, and any residual uncertainty.
7. Review whether the same policy gap affects other access paths.

The specific notification deadlines and affected-party obligations depend on the agreement and applicable law.
An operational service objective must not be presented as a substitute for legal incident obligations.

### Economics beyond the initial pilot

At scale, separate fixed integration cost from per-shipment operating cost.
A carrier may pay once to build an interface but incur ongoing support cost for each new buyer.
A platform may incur fixed security costs and variable storage or messaging costs.
A buyer may realize benefits only when saved labor hours can be redeployed or paid overtime actually falls.

Consider an illustrative annual extension with 4,000 shipments.
Suppose the expected avoided-miss rate is 4 percentage points compared with a credible baseline.
That implies 4,000 × 0.04 = 160 avoided misses per year.
At USD 245 of combined external saving per avoided miss, gross external saving is USD 39,200 per year.
If incremental annual external costs total USD 24,000, combined net is USD 15,200 per year.
This projection assumes the per-miss savings and causal effect persist at the larger scale.
It excludes incident losses and opportunity costs unless they are included in the stated cost figure.

If the effect falls to 2 percentage points, the model yields 80 avoided misses.
Gross saving then becomes USD 19,600 and net becomes −USD 4,400.
The break-even rate is USD 24,000 / (4,000 × USD 245), or approximately 2.45 percentage points.
These calculations do not determine the allocation of costs between buyer, carrier, and tower operator.
They indicate which assumptions should be measured before signing a multiyear agreement.

### Implementation sequence for a small partner

A small carrier may not have an always-on API or a dedicated integration team.
That constraint does not justify bypassing the rights and quality design.
It changes the implementation sequence and may change the achievable service level.

1. Agree the same purpose, audience, and event semantics used for larger partners.
2. Determine whether a secure scheduled file can meet the decision cutoff.
3. Provide a simple template with controlled codes and an explicit time offset.
4. Authenticate the sender and restrict the destination.
5. Validate each file and return actionable errors to a named contact.
6. Reconcile the file against the in-scope shipment manifest.
7. Provide a manual correction route with an auditable case identifier.
8. Measure whether batch timing is sufficient for the receiving decision.
9. Account for the carrier's manual workload in the partner economics.
10. Move to a different interface only when benefits justify the transition.

If a daily file arrives after every relevant appointment cutoff, the file may still help historical reporting.
It does not meet the real-time appointment-management objective.
The parties should describe the limitation honestly rather than labeling the feed real time.

### Step 6 — Implement policy at every access path

**Input:** signed terms, rights matrix, user and partner identities, and event contract.
**Output:** enforceable policies and evidence of access decisions.

1. Authenticate the calling application or user using the platform's approved identity mechanism.
2. Resolve the legal entity, role, permitted purpose, and active shipment relationship from authoritative records.
3. Apply default-deny authorization when a required entitlement is absent or expired.
4. Filter rows to the shipments the recipient may access.
5. Mask or omit fields outside that recipient's permitted view.
6. Check that the same rules apply to APIs, dashboards, file exports, support tools, and scheduled reports.
7. Record who requested access, the evaluated policy version, the decision, the event scope, and the time.
8. Protect logs from ordinary modification and limit access and retention.
9. Review privileged access separately, including support and emergency access.
10. Exercise both allow and deny cases before onboarding a partner.

An identity credential identifies a caller; it does not establish that the caller may view every customer's loads.
An API key that belongs to a carrier must not be accepted as proof that a particular buyer is entitled to a shipment.
A shipment relationship must have an effective period and a clear source of authority.

**Branch:** if the entitlement service is unavailable, use an agreed safe failure mode.
For this use case, the default is to deny new requests and alert the owner; do not silently serve an unrestricted cached view.
Any emergency exception requires a named approver, a time limit, and subsequent review.

### Step 7 — Control copies, derivatives, and secondary uses

**Input:** field rights, retention schedule, and data-flow inventory.
**Output:** a tracked lifecycle for copies and derived outputs.

1. Record each destination, including analytical stores, exports, caches, backups, and subcontractors.
2. Set expiration or deletion controls per destination where practical.
3. Separate the operational feed from model-training and cross-customer analytics environments.
4. Record whether the agreement permits derived metrics and which parties may receive them.
5. Evaluate whether an aggregate reveals a single carrier or customer through small groups or repeated queries.
6. Restrict query access and outputs when inference risk remains unacceptable.
7. Document a process for downstream correction or deletion requests.
8. Track contractual and legal retention exceptions rather than claiming universal immediate deletion.

An ETA model may combine carrier events with buyer appointments.
The resulting score can still encode commercially sensitive behavior.
The fact that a score is mathematically derived does not settle ownership, confidentiality, personal-data status, or permissible use.
Those questions require the agreement and applicable law.

**Branch:** if partners approve a later benchmarking product, negotiate population, suppression, permitted audiences, retention, and compensation separately.
If they do not approve, keep the operational purpose boundary.

### Step 8 — Design correction, disputes, and revocation

**Input:** source-of-truth assignments, contact directory, and termination terms.
**Output:** a time-bounded workflow for contested records and access changes.

1. Provide a channel to challenge an event and attach its identifier and reason.
2. Acknowledge receipt and assign an owner.
3. Preserve the original assertion and its provenance while marking it disputed where appropriate.
4. Ask the responsible source to investigate and issue a correction or explain why the original remains valid.
5. Link a correction to the original, including time, source, and reason code.
6. Notify recipients and identify decisions already made from the old value.
7. Handle a revoked partner by ending future access, invalidating relevant credentials, and disabling scheduled exports.
8. Inventory existing copies and apply the agreed return, deletion, or retention exception process.
9. Record evidence, outstanding exceptions, and completion approval.
10. Escalate unresolved factual, commercial, and legal disputes through named contacts.

Revoking access is not the same as deleting all historical records.
For example, an incident investigation or a legal obligation might justify retaining a restricted copy.
The agreement must describe the applicable exception and review process; the tower should not invent one when a partner exits.

### Step 9 — Pilot, measure, and decide whether to scale

**Input:** approved design, baseline, and pilot population.
**Output:** a go, change, or stop decision with participant-specific evidence.

1. Limit the initial cohort and duration.
2. Confirm onboarding, mapping, security, and user training before counting benefits.
3. Compare outcomes with a credible baseline, using a comparable group where feasible.
4. Measure decision timeliness and event coverage as well as the technical uptime of the interface.
5. Track implementation and recurring costs by participant.
6. Review access denials, correction requests, incidents, and unapproved uses.
7. Record operational side effects, such as carrier dispatch workload or misdirected slot changes.
8. Agree who verifies the benefits and who resolves contested measurements.
9. Decide whether to expand, renegotiate, remediate, or stop.
10. Apply exit controls to any partner that leaves the pilot.

**Branch:** a pilot can be technically successful but commercially unacceptable.
If the carrier loses money while the buyer benefits, renegotiate the allocation before scaling.
If the decision value is not supported by the data, stop even if event volume is high.

### Step 10 — Operate governance continuously

**Input:** live policy inventory, access logs, quality reports, and partner feedback.
**Output:** periodic decisions and evidence of remediation.

1. Review active recipients and roles at a defined cadence and after a material change.
2. Reconcile the shipment population with commercial relationships and source manifests.
3. Inspect failed deliveries, missing events, disputed corrections, and exceptions to retention.
4. Review subcontractors and destinations before adding them.
5. Reassess privacy and security risks when granularity, geography, or purpose changes.
6. Test removal of a partner's access and downstream notification, using a controlled scenario.
7. Compare measured value and costs with the initial business case.
8. Record decisions, owners, due dates, and any approved deviations.

Governance is not a quarterly slide deck disconnected from the operating platform.
A control is effective only if the owner can show that it ran, investigate exceptions, and enforce consequences.

## 6. Worked example

### Scenario and scope

A fictional manufacturer, Northstar Components, receives components from a supplier using carrier Harbor Freightways.
A third-party tower operator, Atlas Visibility, presents shipment events to Northstar's receiving planner.
All names and figures in this example are illustrative; they are not claims about real organizations.

The pilot covers 800 inbound shipments over 13 weeks.
The operational objective is to reduce missed dock appointments by identifying at-risk shipments before the rescheduling cutoff.
The carrier sends shipment ID, standardized event type, event time with UTC offset, an ETA interval, and a coarse exception code.
The buyer sends the confirmed appointment and any approved revised slot back to the carrier.
No continuous GPS history, driver name, freight rate, or unrelated customer loads are included in the buyer feed.

A **forecast** is the carrier's estimated arrival interval.
A **plan** is the receiving team's proposed alternative slot.
A **target** is the pilot's desired missed-appointment rate.
A **commitment** is a slot accepted by the relevant parties, not the ETA forecast.
The parties set a target of at most 8% missed appointments; the target is not a promised outcome.

### Before-and-after outcomes

The following table is a scenario assumption, not a measured real-world effect.
The baseline and pilot cohorts are assumed comparable in shipment mix and measurement.
A real pilot should test that assumption and use a comparison group if feasible.

| Measure | Baseline | Pilot |
|---|---:|---:|
| Shipments in the comparison period | 800 shipments | 800 shipments |
| Missed appointments | 104 shipments | 64 shipments |
| Missed-appointment rate | 13% | 8% |
| Avoided missed appointments | — | 40 shipments |
| Rescheduling alerts acted on before cutoff | Not measured | 88 alerts |
| Valid in-scope events received within 15 minutes | Not measured | 1,840 of 2,000 events |

The baseline rate is 104 / 800 = 0.13, or 13%.
The pilot rate is 64 / 800 = 0.08, or 8%.
The absolute reduction is 5 percentage points.
The relative reduction is 40 / 104 = 38.46%, rounded to two decimal places.
The freshness measure is 1,840 / 2,000 = 92%.
The freshness denominator includes valid in-scope events reconciled against the source, not only the timely events received by Atlas.

The number of acted-on alerts does not prove that all 40 avoided misses were caused by those alerts.
The example attributes the 40 avoided misses to the pilot only for the incremental business-case calculation.
A real evaluation should assess seasonality, mix changes, and other interventions.

### One shipment's decision trail

Shipment `S-204` has a confirmed delivery appointment at 14:00 local time.
At 08:45 the carrier reports an observed departure event at 08:30, with its offset.
At 09:10 the carrier sends an ETA interval of 14:30–15:10, computed at 09:05.
The buyer must request a replacement slot by 10:00.
The planner proposes 15:30; the carrier accepts at 09:35.
The updated appointment is the commitment.
The ETA interval remains a forecast and can change.

At 11:20 the source corrects the departure event time from 08:30 to 08:35.
Atlas links the correction to the original event, records source and receipt times, and notifies the authorized buyer view.
The correction does not automatically undo the slot decision.
The planner reviews it in context, while the audit record preserves what was known when the decision was made.

### Partner economics

Assume each avoided missed appointment saves Northstar USD 180 in avoidable receiving labor and disruption.
Assume each avoided miss saves Harbor USD 65 in waiting and dispatch costs.
The costs and savings cover only the 13-week pilot; they are incremental and exclude costs that occur regardless of the pilot.
Atlas's fee is a transfer from Northstar to Atlas, not a saving to the combined group.

| Pilot item | Northstar | Harbor | Atlas |
|---|---:|---:|---:|
| Operational savings | USD 7,200 | USD 2,600 | USD 0 |
| Integration and operating cost | USD 1,800 | USD 3,100 | USD 1,400 |
| Tower fee paid or received | −USD 2,000 | USD 0 | +USD 2,000 |
| Net before incentive payment | USD 3,400 | −USD 500 | USD 600 |

Northstar's operational saving is 40 × USD 180 = USD 7,200.
Harbor's operational saving is 40 × USD 65 = USD 2,600.
Northstar's net is USD 7,200 − USD 1,800 − USD 2,000 = USD 3,400.
Harbor's net is USD 2,600 − USD 3,100 = −USD 500.
Atlas's net is USD 2,000 − USD 1,400 = USD 600.
The combined net is USD 3,400 − USD 500 + USD 600 = USD 3,500.
Equivalently, external savings of USD 9,800 less external costs of USD 6,300 equal USD 3,500.
The tower fee cancels when adding all three parties.

The positive combined result does not make the proposal acceptable to Harbor.
Suppose Northstar agrees to a USD 900 pilot participation payment to Harbor.
The resulting nets are Northstar USD 2,500, Harbor USD 400, and Atlas USD 600.
The combined net remains USD 3,500 because the payment is another internal transfer.
This is one negotiated allocation, not a mathematically unique fair price.
It does not compensate for an unpriced confidentiality risk or prove the parties will agree.

### Sensitivity and decision

If only 25 missed appointments are truly avoided, combined external savings are 25 × (USD 180 + USD 65) = USD 6,125.
With the same USD 6,300 of pilot costs, combined net becomes −USD 175.
The break-even number is USD 6,300 / USD 245 = 25.71 avoided misses, so at least 26 whole misses are needed under these assumptions.
The benefit estimate is therefore sensitive to attribution.

The parties proceed only if the pilot agreement includes the data boundaries, the participation payment, the correction workflow, and a 13-week review.
If the carrier cannot deliver sufficiently timely events, the buyer may instead retain the manual process or renegotiate the pilot.
If a breach or unauthorized use occurs, the incident and suspension provisions govern; economic value does not override them.

### Rights matrix for the pilot

| Item | Northstar | Harbor | Atlas |
|---|---|---|---|
| In-scope shipment status | View its own covered shipments. | Submit and view its own covered shipments. | Process to provide the agreed tower service. |
| Confirmed appointment | Create or revise through approved workflow. | View the relevant confirmed appointment. | Route and display the appointment. |
| Raw location trace | Not included in buyer feed. | Retained in carrier systems under its own controls. | Not received for this pilot. |
| Cross-customer benchmark | Not authorized. | Not authorized. | Not authorized without separate approval. |
| Model training | No separate grant in this pilot. | No separate grant in this pilot. | Not authorized under these illustrative terms. |
| Correction | Request investigation and view outcome. | Investigate and submit corrections for its events. | Link, distribute, and log corrections. |
| Exit | Lose access at agreed termination, subject to retained exceptions. | Stop future feed as agreed and follow exit procedure. | Disable access, track downstream copies, and document completion. |

The matrix states the negotiated scenario, not a universal rule about ownership or model training.
It must be supported by actual contracts, security controls, and legal review before implementation.

## 7. Code, query, or calculation

The following Python 3 example illustrates an authorization decision and the pilot economics.
It uses only the standard library and synthetic data.
It is intentionally an in-memory teaching example, not a production authorization service.
It does not authenticate callers, enforce a database policy, secure logs, or implement deletion.

Assumptions:

- Identities and shipment relationships have already been authenticated and verified by a trusted system.
- The entitlement interval is start-inclusive and end-exclusive in UTC.
- Each request declares a purpose, which must match an approved purpose exactly.
- Only fields explicitly listed for the recipient are returned.
- Revoked entitlements deny subsequent access; this example does not erase previously delivered copies.
- Money is represented in integer US dollars for this calculation.

```python
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Grant:
    recipient: str
    shipment: str
    purpose: str
    start: datetime
    end: datetime
    fields: frozenset[str]
    active: bool = True


def view_event(event, recipient, purpose, now, grants):
    """Return an allowed projection, or None for any missing entitlement."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must include a UTC offset")
    matches = [
        grant for grant in grants
        if grant.active
        and grant.recipient == recipient
        and grant.shipment == event.get("shipment_id")
        and grant.purpose == purpose
        and grant.start <= now < grant.end
    ]
    if not matches:
        return None
    allowed = set().union(*(grant.fields for grant in matches))
    return {key: value for key, value in event.items() if key in allowed}


def pilot_net(avoided, buyer_payment=900):
    buyer_savings = avoided * 180
    carrier_savings = avoided * 65
    buyer = buyer_savings - 1800 - 2000 - buyer_payment
    carrier = carrier_savings - 3100 + buyer_payment
    tower = 2000 - 1400
    return buyer, carrier, tower, buyer + carrier + tower


utc = timezone.utc
start = datetime(2026, 1, 1, tzinfo=utc)
end = datetime(2026, 4, 2, tzinfo=utc)
now = datetime(2026, 2, 1, 12, tzinfo=utc)
event = {
    "shipment_id": "S-204",
    "event_type": "DELAYED",
    "eta_window": "14:30-15:10 local",
    "driver_name": "Example Person",
    "freight_rate_usd": 850,
}
grant = Grant(
    recipient="Northstar",
    shipment="S-204",
    purpose="appointment_management",
    start=start,
    end=end,
    fields=frozenset({"shipment_id", "event_type", "eta_window"}),
)
print(view_event(event, "Northstar", "appointment_management", now, [grant]))
print(view_event(event, "OtherBuyer", "appointment_management", now, [grant]))
print(view_event(event, "Northstar", "benchmarking", now, [grant]))
print("buyer, carrier, tower, total:", pilot_net(40))
print("combined net with 25 avoided:", pilot_net(25)[-1])
```

Expected output:

```text
{'shipment_id': 'S-204', 'event_type': 'DELAYED', 'eta_window': '14:30-15:10 local'}
None
None
buyer, carrier, tower, total: (2500, 400, 600, 3500)
combined net with 25 avoided: -175
```

The first request returns only the three allowed fields.
The second request fails because the recipient does not match.
The third request fails because benchmarking is not the permitted purpose.
At or after the grant's end time, the same request also returns `None`.
Setting `active=False` on a replacement grant illustrates future denial, not retrospective deletion.

Production implementations need centrally managed identities, a trustworthy purpose and entitlement source, policy versioning, secure storage, monitoring, and tests for every access path.
A client-supplied purpose string alone is not proof of an authorized purpose.
The example also does not decide which overlapping grants should be combined under a real contract; that policy must be explicitly designed and reviewed.

## 8. Data, standards, and regulatory context

### Data structure and interoperability

GS1 EPCIS 2.0 provides a standard way to express visibility events, including what happened, when, where, and in what business context.
Its vocabulary and interfaces support interoperable exchange; they do not verify an event's truth or grant a recipient permission to use it.
A party implementing EPCIS still needs identifiers, entitlement rules, correction procedures, and an agreement on which event fields are shared.

EDI standards can carry structured transactions between trading partners.
The specific message, version, code set, and implementation guide must be agreed by those partners.
An EDI document may support a shipment workflow, but its successful transmission does not prove the recipient was entitled to every contained field.
An API can expose the same operational information with different technical mechanics; the rights questions remain.

Time and identifiers require explicit conventions.
A timestamp without an offset is ambiguous across locations and daylight-saving changes.
A local shipment ID may collide across carriers, so use an agreed namespace or composite identifier.
A correction should identify the original assertion instead of making a historical record disappear.

### Privacy and commercial context

The EU GDPR applies when its material and territorial scope conditions are met; it does not apply automatically to every supply-chain record worldwide.
Article 5 sets principles including purpose limitation, data minimization, accuracy, storage limitation, and accountability for personal-data processing.
Article 6 addresses lawful bases, and Article 28 addresses processor obligations where the relevant controller–processor relationship exists.
International transfers require analysis under the applicable GDPR provisions, including Chapter V where relevant.
Consult the current regulation and qualified counsel for the actual parties, processing, and jurisdiction.

A business shipment record can contain personal data, for example a named driver, contact details, or a trace linkable to an identifiable person.
Removing the name does not automatically make the remaining location and time data anonymous.
Pseudonymized data can remain personal data when it relates to an identifiable person.
Commercial confidentiality and contractual restrictions can apply even where a record is not personal data.

The US legal landscape is not a single general equivalent of GDPR.
Applicable federal, state, sector-specific, contractual, and cross-border requirements depend on the facts.
Do not infer that a US partner can freely repurpose a dataset simply because the GDPR is inapplicable.
This lesson does not determine a lawful basis, controller role, transfer mechanism, or enforceability of any clause.

### Security and risk frameworks

The NIST Cybersecurity Framework 2.0 provides high-level cybersecurity outcomes for managing risk.
It is a framework, not a shipment-data license, a guarantee of security, or a certification issued by this lesson.
Use it to organize governance, identification, protection, detection, response, and recovery responsibilities.
Translate those outcomes into partner-specific controls and evidence.

The NIST Privacy Framework provides a voluntary risk-management approach for privacy.
It can structure discussion of processing, data flows, and individual impacts.
It does not replace an assessment of applicable legal obligations.

### Standards-versus-rights checklist

| Question | Standards can help with | Still requires agreement or assessment |
|---|---|---|
| What is the event? | Common event vocabulary and structure. | Whether the asserted event is accurate. |
| Who can see it? | Interfaces can carry authenticated requests. | Entitlement, purpose, and access policy. |
| How is it corrected? | Event representations can express updates. | Source responsibility and notification deadlines. |
| How long is it retained? | Systems can implement expiration. | Contractual and legal retention requirements. |
| Can it train a model? | Technical formats can make data usable. | Separate permission, rights, and legal basis where applicable. |
| Can it cross borders? | Systems can locate processing environments. | Applicable transfer rules and contractual safeguards. |

## 9. Industry practice and case evidence

### GS1 visibility practice

GS1 describes EPCIS as a standard for sharing visibility event data across and within enterprises.
The public specification supports a common event model; an implementer must still negotiate what is shared and with whom.
The standard is useful evidence for interoperability design, not evidence that a specific partner achieved a particular savings figure.

**Inference:** a shared event vocabulary can lower mapping friction among participants, but adoption cost and governance can still prevent a pilot from scaling.
That inference should be tested against the actual partner integration and operating cost.

### FDA food traceability context

The US Food and Drug Administration's Food Traceability Rule establishes additional recordkeeping requirements for certain foods on the Food Traceability List, subject to the rule's scope and exemptions.
Its critical tracking events and key data elements show how a regulatory traceability need can require records across supply-chain participants.
Consult the FDA's current implementation and compliance information before making a deadline or applicability decision; timelines and guidance may change.

**Inference:** a regulated traceability obligation can make participation more compelling, but does not remove the need to protect unrelated commercial and personal information.
The FDA rule is not a general mandate to publish all shipment events to every trading partner.

### IBM Food Trust public case material

IBM publicly describes Food Trust as a network for sharing food supply-chain data among authorized participants.
This is vendor material and demonstrates a marketed implementation pattern, not independent proof of a particular return on investment or a universal governance model.
Access and commercial arrangements must be assessed for the specific deployment.

**Inference:** a shared network can reduce bilateral integration work when many parties adopt common interfaces, but it may also concentrate operational dependency on the network operator.
Neither the presence of a shared ledger nor a named platform proves that an input is correct or that every downstream use is permitted.

### Choosing a sharing architecture

| Pattern | Benefit | Cost or limitation | Appropriate question |
|---|---|---|---|
| Bilateral feed | Narrow relationship and direct accountability. | Many mappings as partner count grows. | Are there few partners and stable purposes? |
| Central tower | Common operational view and centralized policy. | Concentrated custody, availability, and operator trust. | Can the operator enforce each partner's restrictions? |
| Federated query | Some data remains at source. | Query latency, source availability, and distributed enforcement. | Must raw records stay with each provider? |
| Controlled clean room | Restricted joint analysis and output review. | Setup cost and residual inference risk. | Is a jointly calculated aggregate sufficient? |
| Shared event network | Reusable connectivity and common conventions. | Network governance, membership, and portability issues. | Can all required partners agree on rules and exit? |

Architecture is not a substitute for a business agreement.
Federation can reduce copying but does not automatically prevent an authorized query from revealing sensitive information.
Centralization can simplify enforcement but increases the consequence of a broad access error.
A clean room can limit outputs but needs a threat model, query controls, and contractual authority.

### Negotiating incentives in practice

A participant contributes when expected value exceeds expected cost and risk, subject to strategic and legal constraints.
Value may be direct payment, faster receiving, fewer phone calls, lower detention, better dispute resolution, or access to reciprocal data.
Cost includes integration, operations, security review, correction, and opportunity cost.
Risk includes competitive disclosure, liability for incorrect data, regulatory exposure, and dependence on a platform.

A carrier may prefer a payment per covered shipment to an uncertain share of buyer savings.
A buyer may prefer a fixed pilot payment with a review rather than a permanent fee before causality is established.
A platform may seek permission to improve a model; the parties should assess that request separately from access needed for the operational service.
These are negotiation options, not claims about the terms of any named company's contract.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Purpose creep | Requests for unrelated analytics appear after launch. | Original operational access is reused without review. | Breach of trust or terms. | Separate purpose approval and enforce environment boundaries. |
| Over-collection | Feed includes names and continuous coordinates without a decision need. | Source exports its entire schema. | Unnecessary sensitivity and exposure. | Field inventory and minimum projection. |
| Wrong recipient | A buyer sees another buyer's shipment. | Entitlement is based on a broad tenant role. | Commercial disclosure. | Shipment-level authorization and negative tests. |
| Stale entitlement | Departed partner still has working credentials. | No link between contract end and identity lifecycle. | Unauthorized continuing access. | Time-bounded grants and exit reconciliation. |
| Uncontrolled export | Dashboard filters correctly but raw download is unrestricted. | Alternative path bypasses the policy. | Bulk disclosure. | Apply identical policy to APIs, exports, and support tools. |
| Misleading ETA | Planner treats a forecast as an accepted slot. | Estimate and commitment share a label. | Missed delivery or false promise. | Separate fields, ownership, and workflow states. |
| Silent correction | An event changes without notification. | Source overwrites the old record. | Decisions cannot be reconstructed. | Linked corrections and recipient notification. |
| Hidden downstream copies | Exit inventory omits analyst extracts. | No destination registry or export controls. | Deletion claims are unsupported. | Register destinations and test exit workflow. |
| Small-group leakage | A benchmark contains one carrier on a lane. | Aggregation leaves the provider identifiable. | Commercial information exposed. | Suppress unsafe outputs and evaluate repeated queries. |
| Unequal economics | Provider bears costs while buyer claims all savings. | Business case is calculated only in total. | Partner refuses to participate. | Participant-level costs and negotiated allocation. |
| Source clock error | Events appear to arrive before they occurred. | Time zone or clock is incorrect. | False freshness results and incorrect sequence. | Record offsets, monitor clocks, and investigate negatives. |
| Misplaced standards trust | Team claims EPCIS makes sharing compliant. | Interoperability is confused with legal authority. | Unauthorized use. | Separate standards mapping from rights review. |
| Unauthorized subcontractor | A new support vendor can view records without assessment. | Vendor onboarding bypasses the agreement. | Unapproved onward disclosure. | Subprocessor review, approval, and access minimization. |
| Free-text leakage | Exception notes contain personal or sensitive details. | Users enter narrative text into a shared field. | Privacy or confidentiality incident. | Controlled reason codes and restricted free text. |
| Weak denominator | Freshness is reported only for received events. | Missing events are absent from the calculation. | Performance appears better than reality. | Reconcile to a source manifest and report exclusions. |
| Log overexposure | Audit logs contain complete payloads indefinitely. | Logging is treated as exempt from minimization. | Secondary sensitive dataset. | Log necessary metadata, restrict and expire it. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Shipment IDs match across parties. | Events must attach to the right load. | Reused or mistyped IDs. | Reconcile against manifests and investigate collisions. |
| Each grant reflects a valid business relationship. | The access filter relies on entitlement data. | Contract termination not recorded. | Compare grant registry with current partner records. |
| An ETA interval supports the decision. | The pilot avoids sharing raw traces. | ETA is unavailable or too uncertain. | Measure interval quality and decision lead time. |
| Carrier events are sufficiently timely. | Rescheduling has a cutoff. | Batch delays or source outages. | Reconcile event and receipt times by cohort. |
| Baseline and pilot shipments are comparable. | Differences support an impact estimate. | Seasonality, mix, or another intervention changes. | Use matched comparisons and sensitivity analysis. |
| Unit savings are incremental. | Prevents claiming unavoidable fixed cost as benefit. | Labor cannot actually be redeployed. | Validate avoided costs with finance and operations. |
| All material pilot costs are included. | Partner-level net value informs negotiation. | Security or support work is omitted. | Review time sheets, invoices, and expected support load. |
| The provider can authorize the stated use. | A signed agreement must cover relevant rights. | Upstream carrier or individual rights constrain sharing. | Map upstream terms and obtain legal review. |
| Role and legal classifications are fact-specific. | Privacy obligations depend on actual processing. | A party independently determines a new purpose. | Reassess roles when processing changes. |
| Aggregates may still be sensitive. | Small groups can reveal participants. | An analyst assumes totals are always anonymous. | Assess group sizes, linkage, and repeated queries. |
| Revocation can be propagated. | Access removal must cover all paths. | An unmanaged export persists. | Inventory and periodically test downstream exit. |
| No major incident occurs in the pilot calculation. | The arithmetic omits uncertain loss events. | A disclosure or outage occurs. | Include risk assessment and incident provisions separately. |

The numerical example is deterministic and does not quantify uncertainty in missed appointments, unit savings, or risk.
A robust investment decision should use plausible ranges and measure outcomes over time.
The example's proposed controls are illustrative and not a representation that any specific law or contract has been satisfied.

## 12. Exercises

### Exercise 1 — Map incentives

**Time:** 15 minutes.
A retailer asks its carrier for continuous vehicle coordinates to reduce receiving delays.
The carrier subcontracts half of the loads and has no direct integration with those subcontractors.

**Deliverable:** a four-party map naming the retailer, carrier, subcontractor, and tower operator; list one benefit, one cost, and one concern for each.
**Acceptance criteria:** identify the subcontractor as a source or affected party; explain the retailer's decision deadline; distinguish the carrier's authority from the subcontractor's rights.

### Exercise 2 — Decide whether a new use is permitted

**Time:** 15 minutes.
The signed pilot permits appointment management for covered shipments.
A data scientist wants to use the same events to create an industry benchmark sold to other buyers.

**Deliverable:** a short decision and an approval checklist.
**Acceptance criteria:** do not infer permission from technical possession; identify a new purpose, intended audience, output risk, and need for contractual and legal review.

### Exercise 3 — Evaluate access

**Time:** 20 minutes.
A buyer has a valid grant for shipment `A-1`, appointment management, until 1 July 00:00 UTC.
A request at 30 June 23:59 UTC seeks the ETA and driver name.
The grant allows only shipment ID and ETA.
A second request at 1 July 00:00 UTC seeks the ETA.

**Deliverable:** state the result of each request under a start-inclusive, end-exclusive policy.
**Acceptance criteria:** filter the first request to allowed fields; deny the second; distinguish field projection from row entitlement.

### Exercise 4 — Plan a partner exit

**Time:** 20 minutes.
A carrier terminates the agreement at the end of the month.
Data exists in the tower database, a support export, a backup, and a buyer's local extract.

**Deliverable:** a sequence of at least six actions, with owner and evidence for each.
**Acceptance criteria:** disable future access and exports; track each destination; distinguish deletion from documented retention exceptions; notify affected recipients.

### Exercise 5 — Calculate pilot value

**Time:** 15 minutes.
Assume 30 avoided missed appointments, with USD 180 saved by the buyer and USD 65 saved by the carrier per miss.
The buyer incurs USD 1,800 cost and pays a USD 2,000 tower fee.
The carrier incurs USD 3,100 cost; the tower incurs USD 1,400 cost.
The buyer also pays the carrier USD 900.

**Deliverable:** calculate each party's net and combined net in USD.
**Acceptance criteria:** treat fees and the participation payment as transfers; state whether the carrier is better off and whether the total is positive.

### Exercise 6 — Separate standard and authorization

**Time:** 10 minutes.
A supplier says, “Our event uses EPCIS, so you may give it to all your customers.”

**Deliverable:** a three-sentence response.
**Acceptance criteria:** explain what the standard helps express; identify the unresolved sharing rights; propose a concrete next review step.

### Exercise 7 — Design operating controls

**Time:** 25 minutes.
A three-month pilot will share arrival exceptions for 500 shipments with one buyer.
Events can arrive late and the source sometimes corrects them.

**Deliverable:** a one-page control design covering roles, purposes, data fields, service objectives, corrections, access, and exit.
**Acceptance criteria:** include a denominator for freshness, a source owner for corrections, a deny case, and a documented exit exception process.

### Exercise 8 — Critique a benchmark

**Time:** 15 minutes.
An operator publishes average late minutes for each carrier and lane.
One lane has only one carrier and one customer, and a user can request overlapping weekly and monthly summaries.

**Deliverable:** identify two disclosure mechanisms and propose controls.
**Acceptance criteria:** discuss small-group identification and differencing across queries; do not describe ordinary averaging as a privacy guarantee.

## 13. Self-check answers

### Exercise 1

A reasonable map gives the retailer a receiving benefit and an integration and disclosure-management cost.
The carrier may reduce waiting but must support dispatch integration and may fear route disclosure.
The subcontractor may benefit from smoother appointments but may bear integration work and have concerns about vehicle or driver tracking.
The tower operator receives a service fee but bears security, support, and availability costs.
The retailer should specify when a slot can still be moved and whether an ETA interval would suffice instead of continuous coordinates.
The carrier's commercial agreement does not automatically establish all necessary rights over subcontractor or driver information.
Actual authority requires examining the relevant contracts and law.

### Exercise 2

The proposed benchmark is a new use and new audience, not automatically covered by appointment management.
Pause the benchmark and obtain explicit review and any required approvals before processing for that purpose.
Review underlying and derived data rights, small-group leakage, permitted recipients, retention, compensation, and applicable privacy requirements.
A transformation or aggregation does not itself grant permission.

### Exercise 3

The first request is within the grant's time window and may return shipment ID and ETA, assuming the other identity and purpose conditions hold.
It must not return the driver name because that field is not allowed.
The second request is exactly at the end of the interval and is denied because the end is exclusive.
A field mask alone would not prevent access to another shipment's otherwise permitted ETA.

### Exercise 4

A strong answer names the contract or partner owner who confirms termination and the effective time.
The identity administrator disables grants and credentials, and the export owner stops scheduled jobs.
The data steward inventories the database, support export, backup, and buyer extract and sends the agreed downstream notices.
Each destination owner deletes or restricts the copy according to the agreement and applicable obligations.
For backups or legal holds, the owner records the reason, restricted access, review date, and eventual disposition rather than promising immediate physical erasure.
The tower owner reconciles acknowledgments, resolves missing responses, and records completion evidence.

### Exercise 5

Buyer savings are 30 × USD 180 = USD 5,400.
Carrier savings are 30 × USD 65 = USD 1,950.
Buyer net is USD 5,400 − USD 1,800 − USD 2,000 − USD 900 = USD 700.
Carrier net is USD 1,950 − USD 3,100 + USD 900 = −USD 250.
Tower net is USD 2,000 − USD 1,400 = USD 600.
Combined net is USD 700 − USD 250 + USD 600 = USD 1,050.
The combined result is positive, but the carrier remains USD 250 worse off under these assumptions.
The transfers cancel in the combined calculation.

### Exercise 6

EPCIS provides a common way to represent and exchange visibility events.
It does not itself authorize disclosure to every customer or establish a legal basis for personal-data processing.
Review the agreement, recipient entitlement, purpose, and applicable legal requirements before onward sharing.

### Exercise 7 rubric

A complete design names the buyer, source carrier, tower operator, and accountable data and decision owners.
It restricts the purpose and fields to the appointment decision and defines eligible shipments and recipients.
It measures timely events against an independently reconciled set of valid in-scope events, reporting missing and excluded records.
It preserves event time and receipt time, links corrections to prior assertions, and notifies affected recipients.
It tests that an unauthorized buyer and an expired grant are denied.
It specifies exit, downstream copies, retained exceptions, and evidence of execution.
Partial credit requires recognizing that a delivery uptime metric alone does not establish event completeness or decision value.

### Exercise 8 rubric

The single-carrier lane reveals whose performance the average describes.
Comparing overlapping weekly and monthly results can reveal a suppressed or unreported period through subtraction.
Controls can include minimum group sizes, suppression, restricted query permissions, limits on overlapping queries, and output review.
These reduce risk but do not establish anonymity or differential privacy without the relevant analysis and mechanism.
Review the contract and legal basis separately from the technical safeguards.

## 14. Interview questions

1. **How would you decide whether to share raw GPS or an ETA interval?**
   - Strong: starts with the decision and cutoff, compares utility and exposure, and tests whether the interval suffices.
   - Weak: assumes more granular data is always better or automatically prohibited.
2. **What is the difference between authentication and authorization?**
   - Strong: identity is verified first; policy then checks recipient, shipment, purpose, and time.
   - Weak: treats a valid API key as permission to read every record.
3. **Why can a positive joint business case fail?**
   - Strong: allocates benefits, costs, and risks by participant and proposes a negotiable transfer.
   - Weak: assumes the total automatically persuades the data provider.
4. **How would you handle a corrected shipment event?**
   - Strong: identifies the source owner, links the correction, preserves provenance, and notifies recipients.
   - Weak: silently overwrites the event or treats all prior decisions as invalid.
5. **What does an event standard provide?**
   - Strong: explains interoperable vocabulary and structure while separating permission and truth.
   - Weak: claims using EPCIS automatically makes a feed compliant.
6. **What happens when a partner leaves?**
   - Strong: ends future access, stops exports, inventories copies, handles exceptions, and records evidence.
   - Weak: deletes one live table and asserts that every copy is gone.
7. **Can an aggregate leak commercial information?**
   - Strong: gives a small-group or differencing example and identifies output controls.
   - Weak: asserts that averaging always anonymizes data.
8. **How would you measure a data feed's usefulness?**
   - Strong: reconciles coverage, freshness, accuracy, and decision timing against a defined population.
   - Weak: reports only API uptime or number of messages.
9. **Who owns an ETA model trained from shared events?**
   - Strong: refuses a universal answer; examines agreement, input rights, model use, and applicable law.
   - Weak: assumes either the platform or the original provider always owns everything.
10. **What changes if a driver is identifiable?**
    - Strong: flags possible personal-data obligations, minimization, and fact-specific legal review.
    - Weak: assumes a business shipment can never contain personal data.

## 15. Key terms

- [Access control](#step-6--implement-policy-at-every-access-path) — enforcement of who may use which information.
- [Aggregation](#step-7--control-copies-derivatives-and-secondary-uses) — combining records into a summary.
- [Audit log](#step-6--implement-policy-at-every-access-path) — protected evidence of access and policy decisions.
- [Business incentive](#step-2--map-the-parties-and-their-incentives) — expected value motivating a participant.
- [Commitment](#one-shipments-decision-trail) — accepted operational promise, distinct from a forecast.
- [Correction](#step-8--design-correction-disputes-and-revocation) — linked amendment of an asserted event.
- [Data contract](#step-5--agree-the-event-and-quality-contract) — versioned operational specification for a data product.
- [Data minimization](#step-3--inventory-data-and-classify-sensitivity) — limiting data to what is needed.
- [Data provider](#step-2--map-the-parties-and-their-incentives) — participant making information available.
- [Data recipient](#step-4--define-the-permitted-purpose-and-party-boundary) — participant authorized to receive information.
- [Decision cutoff](#step-1--name-the-decision-and-its-deadline) — latest useful time for a specified action.
- [Derived data](#step-7--control-copies-derivatives-and-secondary-uses) — result computed from input data.
- [Entitlement](#step-6--implement-policy-at-every-access-path) — a scoped grant to access specified records.
- [EPCIS](#data-structure-and-interoperability) — GS1 visibility event standard.
- [ETA](#one-shipments-decision-trail) — estimated time of arrival, a forecast.
- [Event time](#step-5--agree-the-event-and-quality-contract) — time of the underlying occurrence.
- [Exit](#step-8--design-correction-disputes-and-revocation) — controlled termination and disposition workflow.
- [Field projection](#step-6--implement-policy-at-every-access-path) — limiting the fields returned to a recipient.
- [Freshness](#step-5--agree-the-event-and-quality-contract) — timeliness of data relative to a defined event.
- [Lineage](#step-7--control-copies-derivatives-and-secondary-uses) — trace of sources, transformations, and destinations.
- [Onward sharing](#step-4--define-the-permitted-purpose-and-party-boundary) — disclosure to an additional party.
- [Personal data](#privacy-and-commercial-context) — information relating to an identified or identifiable person under relevant law.
- [Processing time](#step-5--agree-the-event-and-quality-contract) — time the system handled an event.
- [Purpose limitation](#step-4--define-the-permitted-purpose-and-party-boundary) — restricting uses to defined purposes.
- [Retention](#step-7--control-copies-derivatives-and-secondary-uses) — how long and under what conditions data is kept.
- [Revocation](#step-8--design-correction-disputes-and-revocation) — ending or narrowing future access.
- [Rights matrix](#rights-matrix-for-the-pilot) — concise record of participant-specific permitted uses.
- [Row-level security](#step-6--implement-policy-at-every-access-path) — filtering records by entitlement.
- [Sharing agreement](#step-4--define-the-permitted-purpose-and-party-boundary) — documented terms governing data use.
- [Small-group leakage](#exercise-8--critique-a-benchmark) — disclosure through an identifiable aggregate population.
- [System of record](#step-8--design-correction-disputes-and-revocation) — agreed authority for a defined fact.
- [Transfer payment](#partner-economics) — payment between participants that cancels in combined net value.

## 16. Sources

1. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis), including the EPCIS 2.0 specification. Primary source for the visibility-event model and interfaces; does not grant data-use rights.
2. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5, 6, 28 and Chapter V. Primary legal text; applicability and interpretation depend on the facts and current law.
3. National Institute of Standards and Technology, [Cybersecurity Framework 2.0](https://www.nist.gov/cyberframework). Official voluntary risk-management framework; not a data-sharing permission.
4. National Institute of Standards and Technology, [Privacy Framework](https://www.nist.gov/privacy-framework). Official voluntary privacy risk-management resource.
5. US Food and Drug Administration, [Food Traceability Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods). Official explanation of scope, key data elements, critical tracking events, exemptions, and current implementation information.
6. IBM, [IBM Food Trust](https://www.ibm.com/products/supply-chain-intelligence-suite/food-trust). Company description of a food-data sharing network; vendor source, not independent evidence of savings.
7. National Institute of Standards and Technology, [De-Identifying Government Datasets: Techniques and Governance](https://doi.org/10.6028/NIST.SP.800-188), NIST Special Publication 800-188, 2023. Technical discussion of de-identification risks and governance; not a guarantee that a particular output is anonymous.

Sources are cited for the limited claims stated above.
The Northstar, Harbor, and Atlas scenario, figures, rights, and outcomes are invented for instruction.
No case result or regulatory approval is implied by the example.

## 17. Links

- [Module 19 topic map](_index.md)
- [Module 19 survey and its lab sections](../../../course/19_control_towers_visibility_events.md)
- [Event streams and EPCIS](02_event-streams-epcis.md)
- [Exception management and queue design](07_exception-queue.md)
- [Decision rights and escalation](09_decision-rights.md)
- [Course expansion specification](../../../course/MODULE_EXPANSION_SPEC.md)
