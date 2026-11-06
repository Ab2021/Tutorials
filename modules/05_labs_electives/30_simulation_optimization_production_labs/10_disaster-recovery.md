# Observability and Disaster Recovery

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Python 3.9+, basic files and JSON, and familiarity with the model service and deployment topics.  
> Version 2026.5

## 1. Why this topic exists

A replenishment planner cannot use a forecast service that has lost its model or cannot explain which input snapshot produced a recommendation.
A dispatcher cannot safely replay events if the recovered consumer has forgotten which messages it already committed.
A warehouse supervisor may continue executing an obsolete plan while a dashboard reports a healthy web server.
The technical incident is therefore also an operational decision problem.

Observability helps a team notice a deviation and understand its scope.
Recovery makes a usable, trustworthy service available again after failure.
Neither is a synonym for the other.
A service can emit excellent logs and have no restorable backup.
It can also have a valid backup while no one notices that the live forecast is stale.

The operations team must decide whether to wait, fail over, restore, or switch to a manual process.
The business owner must decide how much unavailable time and committed data loss are tolerable.
The data owner must decide which recovered records can be trusted and which require reconciliation.
The incident commander coordinates these decisions without treating a successful container restart as proof of business recovery.

In this topic, you design and rehearse a small recovery path for an order-processing dataset.
The dataset is fictional and deliberately small enough to inspect.
Its orders stand in for state used by a supply-chain analytics service; they are not evidence about a real company.
The lab creates a verified backup, simulates an incident, restores into a separate local directory, and checks the recovered rows.
It also records structured logs and simple metrics so that the drill leaves inspectable evidence.

The lab does not deploy a live service, connect to a production database, or prove an organization can meet its recovery objectives.
You will identify what a production-equivalent drill would additionally require.
You should finish with both an executable artifact and a runbook that a different operator can follow.
The key decision is not merely whether bytes can be copied.
It is whether the restored state is recent enough, internally consistent, secure, and suitable for the next business action.

## 2. Learning outcomes

After working through the topic, you can:

1. Distinguish an event log, a metric, a backup, and an independently verified restore.
   Exercise 1 asks you to classify evidence.
2. Calculate recovery time objective (RTO), recovery point objective (RPO), and observed gaps from explicit timestamps.
   Exercise 2 tests the arithmetic.
3. Identify a consistent recovery boundary across application data and dependent artifacts.
   Exercise 3 asks for a dependency map.
4. Create and integrity-check a backup and restore it into an isolated location.
   Exercise 4 uses the runnable lab in section 7.
5. Design an incident runbook with decision points, roles, communications, and rollback criteria.
   Exercise 5 asks you to produce one.
6. Explain why successful byte restoration is not sufficient evidence of business correctness.
   Exercise 6 asks you to design reconciliation checks.
7. Critique observability and restore evidence for missing controls and misleading metrics.
   Exercises 7 and 8 ask you to investigate unsafe scenarios.
8. Translate drill observations into corrective actions without claiming production readiness.
   Exercise 9 asks for a short recovery report.

## 3. The core mental model

```text
Business process: demand -> order -> forecast / decision -> operator action
                                |
                         application state
                                |
             +------------------+-------------------+
             |                                      |
        observe continuously                    protect state
      logs / metrics / alerts          consistent backup + catalog + keys
             |                                      |
        incident detected                 isolated restore environment
             |                                      |
   assess and choose strategy ---------> restore -> technical checks
                                                    |
                                      business reconciliation and approval
                                                    |
                                          controlled return to service
                                                    |
                                       evidence -> improve and rehearse
```

The left branch makes a failure visible and supplies context.
A metric summarizes a changing numerical quantity, such as the number of failed forecast requests in a time window.
A log records a discrete event, such as a failed restore check with a backup identifier.
An alert converts a measured condition into a request for a human or automation to act.
An alert should point to a decision and an owner; otherwise it is only noise.

The right branch makes recovery possible.
A backup is a retained copy of data and the metadata needed to recover it.
An inventory must also account for code, configuration, secrets, permissions, schema, model artifacts, and upstream or downstream positions.
A backup copy is not enough if an operator cannot find it, decrypt it, interpret it, and validate it under incident conditions.

The branches meet during the incident.
Metrics and logs identify the start, scope, and potential recovery point.
The restoration process supplies recovered data and evidence of its age and correctness.
The business owner determines whether the remaining gap is acceptable and how missing transactions will be reconstructed.

### Three different clocks

The first clock measures the interval from the agreed disruption start to restored service.
The second measures the age of the newest recoverable committed data at the disruption boundary.
The third measures how long an operator actually spends executing a drill.
These clocks answer different questions and may have different start and end points.
A fast restore after a late detection can still miss the service-level RTO.
A very recent backup can still have an unacceptable RTO if restoring it takes too long.

### Evidence travels with the decision

Record the incident identifier, event timestamps and their time zone, backup identifier, integrity result, recovered boundary, validation results, and approver.
Record unsuccessful attempts too; a successful retry does not erase the time spent finding the right backup.
Protect the evidence because logs and backup inventories may disclose sensitive operational details.
The lab uses public-looking synthetic values and local files only.

### A backup is not a replica

A replica can improve availability, but corruption or a malicious deletion may propagate to it.
A separately retained backup provides a point in time to which the team can return, subject to its retention and isolation properties.
A backup does not automatically provide a ready-to-serve replica.
Both may be needed, but this exercise focuses on backup and isolated restore.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Incident | An event requiring coordinated response to actual or suspected impact. | Calling every alert an incident without triage. |
| Detection time | Time at which the team or monitoring notices the condition; record in UTC. | Assuming detection equals failure start. |
| Disruption start | Agreed time at which the service ceased meeting its recovery scope. | Starting the clock only when restore begins. |
| Recovery time objective (RTO) | Maximum acceptable time from disruption to agreed recovery state, in minutes or hours. | Calling a measured restore duration an objective. |
| Recovery point objective (RPO) | Maximum acceptable age of recoverable data at disruption, in minutes or hours. | Treating backup schedule alone as proof of achieved RPO. |
| Recovery time actual | Observed elapsed disruption-to-acceptance interval for an incident or drill. | Omitting detection, approval, or verification. |
| Recovery point actual | Observed difference between disruption and latest valid recoverable committed state. | Using the file modification time without a verified data boundary. |
| Backup | A retained copy enabling recovery of a defined state. | Assuming a live replica or a copied file is automatically a recoverable backup. |
| Snapshot | A representation of state at a defined boundary, whose consistency depends on the mechanism. | Assuming independently copied changing files are mutually consistent. |
| Restore | Reconstructing usable state from retained recovery material. | Equating a completed copy with a validated service. |
| Restore drill | A controlled rehearsal of recovery and verification. | Calling backup creation a restore drill. |
| Isolation | Separation that prevents a drill from affecting real systems or recipients. | Assuming a different folder blocks network connections. |
| Integrity digest | A calculated value used to detect changes to bytes, such as SHA-256. | Treating a colocated digest as proof of authenticity against an attacker. |
| Manifest | Metadata describing backup files, identifiers, boundaries, and checks. | Assuming a manifest proves that its own contents are trustworthy. |
| Consistency | A property that recovered records satisfy the relevant transactional and application relationships. | Concluding that a matching checksum proves business consistency. |
| Reconciliation | Comparison of recovered state with an independent authoritative source or agreed control. | Comparing a file only with its own copied metadata. |
| Log | A discrete event record with contextual fields. | Logging secrets or assuming logs replace metrics. |
| Metric | A numerical measurement with an aggregation and unit. | Interpreting a cumulative counter as a rate without a time interval. |
| Counter | A metric that increases for events during a process lifetime or reporting interval. | Assuming a counter never resets after restart. |
| Gauge | A metric representing a value that can rise or fall, such as backup age. | Summing unrelated gauge values as though they were events. |
| Alert | A rule that notifies or invokes action when a condition is met. | Alerting on an arbitrary metric without an owner or response. |
| Runbook | A versioned set of actions and decisions for an operational task. | A list of shell commands with no authorization or rollback. |
| Recovery boundary | The latest included committed state, expressed using a timestamp, sequence, or other domain marker. | Using a backup completion timestamp as the data boundary. |
| Point-in-time recovery | Restoring a base backup and required transaction history to a selected boundary. | Assuming a periodic full backup alone supports arbitrary points in time. |
| Failover | Moving service to an alternative operational instance. | Assuming failover includes recovery from replicated corruption. |
| Failback | Returning service to the intended primary after validation. | Switching back merely because the original system responds. |
| Immutable retention | Controls that prevent modification or deletion for an enforced period. | Calling a file immutable because the team intends not to edit it. |
| Least privilege | Granting only the permissions required for a role and task. | Giving a backup writer unrestricted ability to delete all retained backups. |
| Recovery acceptance | Explicit authorization that technical and business checks meet agreed criteria. | Treating a process exit code as business sign-off. |
| Synthetic fixture | Artificial data with known contents used to make a test repeatable. | Inferring production data quality or performance from it. |
| UTC | Coordinated Universal Time, used here for comparable timestamps. | Comparing local times without handling time zones and clock skew. |

## 5. The mechanics, step by step

The sequence below is a production-oriented method, not a command to change production.
The executable exercise later implements only a small subset.

### Step 1: Define the business service

**Input:** A specific service, business owner, and operating calendar.
List who uses the service and what decision depends on it.
Name the minimum usable outcome, such as accepting and reconciling orders rather than merely returning an HTTP response.
Document business hours, seasonal peaks, and consequences of a stale recommendation.

**Output:** A short service description and recovery acceptance statement.
For an order-processing service, acceptance might require a checked order ledger and an approved reconciliation plan for missing orders.
It might also require a safe manual intake channel during the outage.
The appropriate criteria come from actual owners, not the lab author.

**Branch:** If there is no owner empowered to accept residual data loss, escalate before setting a recovery target.
The engineering team can estimate options but should not silently choose the business risk.

### Step 2: Map dependencies and consistency boundaries

**Input:** Architecture, data-flow diagrams, configuration inventory, and accountable owners.
List persistent databases, object storage, model files, feature definitions, container images, infrastructure configuration, certificates, and keys.
List message queues, offsets, idempotency records, and external systems that may have observed actions.
Identify services needed to authenticate, resolve names, decrypt backups, and communicate during recovery.

For each item record its owner, source of truth, retention, restore procedure, and dependency on other items.
For each data movement record whether delivery is at-most-once, at-least-once, or otherwise specified by the actual system.
A backup of a database without the corresponding queue position may replay orders or omit them.
A model artifact without its feature transformation or version metadata may load but give misleading results.

A consistent boundary is not necessarily a single wall-clock timestamp across every component.
It can be a transaction identifier, an event offset paired with a database checkpoint, or a documented application-level checkpoint.
Choose a boundary that allows relationships to be reconstructed and verified.
If a distributed snapshot cannot be made atomically, record the reconciliation method and the uncertainty.

**Output:** A dependency table and an agreed consistency boundary.
**Branch:** If the boundary cannot be established, treat the backup as incomplete for that recovery scenario.

### Step 3: Set objectives and scope

**Input:** Business impact assessment, dependency map, and realistic recovery options.
Set RTO as a maximum elapsed time for reaching a named acceptance state.
Set RPO as a maximum acceptable data-loss interval at the disruption boundary.
State the operating window, severity, responsible decision maker, and scope of each target.
Separate an internal target from a customer commitment or contractual service level.

For example, a 60-minute RTO does not mean that restoring files in 60 minutes is sufficient.
It may include detection, authorization, environment provisioning, restoring, validation, and communications.
Likewise, a 15-minute RPO is not assured simply by scheduling backups every 15 minutes.
A failed job, long-running snapshot, or missing transaction log can leave the newest valid recovery point older.

**Output:** A signed target sheet with assumptions and exception procedures.
**Branch:** If the measured or estimated recovery path exceeds a target, either improve the path or explicitly accept and document the risk.
Do not redefine the start time merely to make the metric pass.

### Step 4: Define useful observability

**Input:** Service objectives, process flow, and dependency map.
Measure user-visible availability and correctness signals, not only process health.
Examples include request failures, request latency, age of the latest accepted forecast, unprocessed order age, and backup age.
Attach an owner and response to each actionable alert.
Use bounded labels such as service and environment; avoid order identifiers as metric labels because cardinality can grow without bound.

Use structured logs for incident-relevant events.
Include timestamp, severity, service, environment, incident or operation identifier, result, and a safe reason code.
A correlation identifier can connect an API request to a job without including customer names or credentials.
Define retention and access based on security and legal requirements.
Avoid writing access tokens, passwords, or raw sensitive payloads into logs.

**Output:** A metric and log catalog with units, thresholds, owners, and retention.
**Branch:** If the metric pipeline itself is down, use an independent heartbeat or external observation where practical.
An absent backup-success metric must not be interpreted as success.

### Step 5: Design backup capture and retention

**Input:** Data inventory, recovery boundary, RPO, retention requirement, and threat model.
Choose an application-consistent or database-supported mechanism where concurrent writes exist.
For databases, follow the database vendor's method for base backups and transaction logs rather than copying active files blindly.
Record the true recovery boundary, backup start and completion times, schema version, software version, and dependencies.

Choose retention for operational errors, accidental deletion, corruption discovered late, and applicable policy.
Separate the authority that creates backups from the authority that deletes or changes retained copies where practical.
Consider a separate account or domain, restricted administrative access, encryption, and enforceable retention controls.
Verify that keys and recovery credentials themselves have a secure, tested recovery path.
A backup that cannot be decrypted is not useful.
A key stored only inside the failed environment creates a circular dependency.

**Output:** A backup policy and documented catalog.
**Branch:** If the expected recovery point is missing or a chain is broken, choose an earlier verified point and escalate its greater data-loss exposure.

### Step 6: Validate capture, not just scheduling

**Input:** Backup job output and catalog entry.
Check that the job completed, all expected components exist, metadata is plausible, and the integrity verification succeeds.
For incremental or transaction-log chains, validate continuity from the base backup to the intended point.
Check backup age against the intended boundary, not merely job start or completion time.
Alert on missing jobs and on missing telemetry as well as reported failure.

A checksum compares bytes against an expected digest.
A digest stored alongside the data can detect accidental damage when the manifest remains trustworthy.
An attacker able to replace both file and digest can defeat that check.
Use a trusted catalog, signature, authenticated storage, or equivalent controls appropriate to the threat model.

**Output:** A catalog entry marked available only after required checks.
**Branch:** If verification fails, quarantine the copy, notify the owner, and find another recovery point.

### Step 7: Prepare an isolated restore environment

**Input:** An approved drill plan, isolated destination, sanitized or authorized data, and a named operator.
Create a separate account, project, namespace, or network as appropriate to the actual system.
Block unintended outbound communication and production writes before loading state.
Replace real endpoints, email recipients, callbacks, scheduler triggers, and credentials with approved test equivalents.
Prevent restored jobs from automatically placing orders or replaying messages into live queues.

A different directory is useful for a file exercise but is not a security boundary.
A realistic drill should enforce separation at the identity, network, storage, and configuration layers.
Verify isolation with approved negative tests, such as denying access to a production endpoint.
Do not use a live supplier integration to see whether the sandbox is isolated.

**Output:** An approved environment with isolation evidence and a cleanup owner.
**Branch:** If isolation cannot be established, stop before loading sensitive data or starting application processes.

### Step 8: Select and restore a recovery point

**Input:** Incident facts, authorized recovery boundary, backup catalog, keys, and the restore procedure.
Identify the latest valid recovery point that is not contaminated by the incident.
For destructive or malicious changes, the newest copy may already contain the damage.
Estimate the data gap relative to the incident boundary before proceeding.
Record the identifier and reasons for selecting the point.

Restore to the isolated target using the platform's supported procedure.
Verify the backup before use, then verify the restored content independently.
Do not overwrite the only evidence of the failed system without an authorized preservation decision.
Record start and finish timestamps and any manual intervention.

**Output:** A restored candidate state and an execution record.
**Branch:** On a missing dependency, bad integrity check, or failed restore, stop using that candidate and escalate or select an earlier verified point.

### Step 9: Run technical validation

**Input:** Restored candidate, expected schema, integrity records, and software configuration.
Validate file and database integrity with supported tools.
Check schema versions, migration state, indexes or constraints where relevant, and expected application startup behavior.
Check that data can be read by the application version selected for recovery.
Inspect logs for errors and check that observability itself is reporting from the recovered environment.

Run representative queries against restored data.
A simple row count catches some gross failures but cannot establish completeness or uniqueness by itself.
Check keys, allowed states, date ranges, and relationships appropriate to the domain.
Run read-only probes before enabling any side effects.

**Output:** A pass/fail report with the check, observed result, and reviewer.
**Branch:** Any required technical check failure blocks acceptance until addressed or explicitly handled under the incident authority's policy.

### Step 10: Reconcile business state

**Input:** The recovered boundary, independent source records, operational logs, and business control totals.
Compare recovered orders, inventory movements, and decisions with independent sources that were not affected by the same failure.
Determine which committed transactions fall after the recovery boundary.
Classify each as recoverable by replay, recoverable manually, already executed externally, or unresolved.

For each replay candidate, confirm the idempotency key and the downstream action status.
A replayed event may already have caused a purchase order, carrier booking, or warehouse pick.
Deduplicate using documented business identifiers and confirm outcomes with the relevant system of record.
Maintain an exception queue for cases that cannot be resolved automatically.

**Output:** A reconciliation ledger and a business-owner decision.
**Branch:** If independent evidence is unavailable, document the uncertainty and use an authorized manual process rather than claiming completeness.

### Step 11: Accept, communicate, and cut over

**Input:** Technical checks, reconciliation ledger, known gaps, security assessment, and decision authority.
Have the designated owner accept the candidate or reject it with reasons.
Communicate the impact, affected interval, current workaround, and next update time without disclosing sensitive data.
Before routing traffic, verify integrations, capacity, access controls, alerting, and rollback conditions.
A staged cutover can limit impact, but its actual form must match the architecture and approved plan.

If returning to the former primary, establish how state accumulated during recovery will be transferred without losing newer writes.
Do not assume that switching a DNS record reconciles divergent databases.
Record who approved cutover and when the service met the agreed acceptance criteria.

**Output:** An accepted service state, communications log, and measured elapsed time.
**Branch:** If acceptance fails, continue the workaround or select another recovery strategy; do not report recovery based on process health alone.

### Step 12: Close the drill and improve

**Input:** Timeline, log and metric evidence, validation report, reconciliation, and feedback.
Calculate observed elapsed recovery time and observed recovery point age from the declared clocks.
Compare each observation with its target and record any uncertainty.
Assign corrective actions to owners with due dates, then schedule a repeat drill after changes.
Destroy or retain drill data according to the approved policy and verify cleanup.

**Output:** A dated report with evidence links and action owners.
**Branch:** If the drill excluded key dependencies, report only the tested scope; do not extrapolate a production RTO.

### Operational artifact A: recovery dependency register

Use a register rather than relying on an architecture diagram alone.
The diagram explains relationships; the register assigns responsibility and records concrete recovery evidence.
The following is a fictional example, not a complete inventory of any deployed system.

| Component | Required recovery material | Boundary or version | Validation and owner |
|---|---|---|---|
| Order ledger | Consistent database backup and required transaction history | Last committed transaction included | Data owner checks keys and totals. |
| Event publisher | Configuration and publication checkpoint | Publisher position corresponding to ledger | Platform owner compares published event IDs. |
| Event consumer | Consumer state and offset | Last event whose effect is committed | Warehouse owner checks side effects. |
| Forecast model | Model bytes and signed release metadata | Approved model version | Model owner checks load and sample predictions. |
| Feature logic | Code and feature schema | Version compatible with model | Analytics owner checks representative features. |
| Credentials | Authorized recovery access and key references | Current approved credential set | Security owner tests access without exposing values. |
| Network routing | Infrastructure configuration and endpoint inventory | Approved recovery environment | Platform owner checks deny rules and routes. |
| Monitoring | Alert rules, dashboards, and log destinations | Version compatible with service | On-call owner tests signal and missing-data behavior. |
| External communications | Contact list and approved templates | Current supplier and customer contacts | Incident commander checks recipients and cadence. |
| Manual order intake | Forms, identifiers, and exception register | Sequence covering incident interval | Business owner checks completeness and uniqueness. |

Record the location of the recovery instructions for each row.
Record a substitute owner if the primary owner is unavailable.
Do not place secret values in the register; store references to an approved secret-management process.
For each item, note whether its recovery source shares a failure domain with the primary system.
If the backup catalog is hosted only in the failed environment, the catalog itself is a recovery dependency.
If a software artifact is pulled from a public registry during an outage, network access and registry availability are dependencies too.
Preserving a version identifier is useful, but it does not guarantee the artifact can still be retrieved.
Retain and verify the artifacts needed for the actual recovery policy.

### Operational artifact B: alert design worksheet

Start with a question an operator can answer.
For example: “Is the newest verified recovery boundary too old for the agreed RPO?”
Define `backup_age_minutes` as the current UTC time minus the most recent verified data boundary, expressed in minutes.
The source of the boundary must be the catalog entry that passed the organization's capture checks.
It must not be the time at which a scheduler last started.

Choose a warning threshold that leaves time to investigate before the RPO is breached.
Choose a critical threshold based on the business objective and the expected delay in receiving telemetry.
If the RPO is 20 minutes and detection plus response can take 8 minutes, an alert first firing at minute 20 offers no advance margin.
This arithmetic helps choose a threshold but does not substitute for testing the whole pipeline.

Define the alert evaluation interval and time window.
A short transient delay may justify a warning rather than immediately paging an incident team.
A sustained breach, a verified backup failure, or a missing heartbeat may require escalation.
Specify behavior when no valid boundary exists: report an unknown or failure state, not a reassuring zero.
Specify the owner, escalation route, and link to the approved runbook.

An example alert record could include:

1. Service and environment, using bounded metric labels.
2. Observed verified backup age in minutes.
3. Applicable target and warning threshold in minutes.
4. Most recent verified backup identifier, included in controlled logs rather than as an unbounded metric label.
5. Time of the last telemetry update in UTC.
6. On-call role and incident escalation destination.
7. Instructions to inspect job state and catalog validity.
8. A response expectation appropriate to the service's operating hours.

Test an alert with synthetic input in a sandbox.
Test a stale boundary, a reported capture failure, and total absence of telemetry.
Confirm that the right person receives each signal and can access the runbook.
Record notification latency separately from the time at which the data first became stale.
An alert that fires correctly but reaches an unattended channel does not provide the intended operational control.

### Operational artifact C: structured event design

A useful restore event answers what was attempted, when, where, and with what result.
It also allows an incident reviewer to correlate events without revealing sensitive content.
For a production event, consider the following fields:

| Field | Purpose | Example constraint |
|---|---|---|
| `timestamp` | Place the event in the timeline. | UTC with an explicit offset. |
| `event` | Name a stable event type. | A bounded vocabulary such as `restore_started`. |
| `severity` | Support routing and prioritization. | A documented severity scale. |
| `service` | Identify the affected service. | A stable service identifier. |
| `environment` | Distinguish drill from production. | An approved environment identifier. |
| `operation_id` | Correlate steps of one restore. | A generated non-secret identifier. |
| `backup_id` | Identify the selected recovery material. | An opaque catalog identifier. |
| `result` | State success, failure, or interruption. | A bounded value. |
| `reason_code` | Identify a known failure class. | No raw credentials or customer payloads. |
| `actor_role` | Identify responsible operational role. | Avoid unnecessary personal details. |

A single event does not establish the full incident timeline.
Events may arrive late, be duplicated, or be missing during a pipeline failure.
Retain the original event time and, where supported, the ingestion time so reviewers can distinguish these cases.
Validate log timestamps against the organization's time synchronization assumptions.
Record uncertain boundaries as intervals or estimates instead of inventing exact values.

If a restore step fails, log a safe reason and the operation identifier.
Do not embed raw backup contents or a decryption key in an exception message.
If an operator must inspect sensitive evidence, use the approved access path and retain an access record.
Separate operational logs from a protected audit trail where the threat model requires it.
The lab's printed JSON demonstrates structure only; it does not implement these production controls.

### Operational artifact D: recovery-point selection

During an incident, list candidate recovery points from newest to oldest.
For each point ask whether the backup completed, whether integrity checks pass, and whether all required chain segments exist.
Ask whether its recovery boundary precedes the earliest plausible corruption.
If the corruption start is uncertain, record the uncertainty rather than choosing the latest point automatically.

Suppose a fictional operator has these candidates:

| Candidate | Data boundary | Integrity | Corruption assessment | Initial decision |
|---|---|---|---|---|
| A | 09:56 UTC | Pass | Known bad change present | Reject for clean recovery. |
| B | 09:43 UTC | Pass | No known bad change; needs validation | Consider with business approval. |
| C | 09:10 UTC | Pass | Earlier than suspected change | Fallback if B fails validation. |
| D | 08:40 UTC | Missing required log segment | Unknown | Reject as a complete chain. |

If the disruption is at 10:00 UTC, candidate B implies a 17-minute recovery-point age.
Candidate C implies a 50-minute age.
Candidate A is more recent but is not a clean candidate under the stated facts.
Candidate D is not usable to its claimed boundary because the chain is incomplete.
Neither a matching checksum nor a recent boundary establishes that candidate B is free of all semantic corruption.
Use independent evidence and application checks to investigate that question.

Record the selection, the evidence available at the time, and the approver.
If a later candidate is rejected during validation, record the new selection and the additional gap.
The business owner must know when selecting an older point exceeds the target.
Do not delete rejected copies unless authorized; they may be useful for forensic analysis.

### Operational artifact E: isolation checklist

Before a drill, the operator and security owner should agree on the scope of isolation.
The following checklist is a design aid, not proof that a particular cloud or local environment is secure.

1. Name the target environment and the account or administrative domain that owns it.
2. Identify every network path to production, suppliers, customers, and shared services.
3. Deny unintended egress before starting restored application processes.
4. Deny writes to production databases, buckets, queues, and APIs.
5. Confirm that the drill identity cannot assume a production write role.
6. Use approved test credentials or disable integrations that cannot be safely simulated.
7. Replace email and messaging recipients with approved test destinations.
8. Disable automatic schedulers, webhooks, purchase-order submission, and outbound dispatch.
9. Review DNS and service discovery so production names cannot bypass intended restrictions.
10. Review mounted volumes and shared storage for unintended production access.
11. Check that recovery keys are accessible only to authorized drill operators.
12. Decide whether the restored data is synthetic, sanitized, or authorized production data.
13. Define who may view the restored data and how access is logged.
14. Run approved negative connectivity tests without sending live business transactions.
15. Record the test method, result, timestamp, and reviewer.
16. Stop if a prohibited path remains available.
17. Start only the minimum components required for the test.
18. Recheck isolation after configuration or infrastructure changes.
19. Collect restore and validation evidence without exporting unnecessary sensitive data.
20. Destroy or retain the environment according to the approved cleanup plan.

Different technologies enforce isolation differently.
A filesystem directory can prevent accidental filename collisions but does not prevent network access.
A network policy can restrict packets but does not by itself prevent an overprivileged identity from accessing a shared service through another path.
A separate account can reduce shared permissions but still requires appropriate routing and role configuration.
Use multiple controls appropriate to the actual architecture and verify the controls that matter.

### Operational artifact F: incident communications

Agree on a communications cadence before an incident.
The incident commander should identify who can declare an incident, who can approve a recovery point, and who can authorize customer or supplier updates.
A technical operator can report a measured fact without promising an unapproved restoration time.
The business owner can describe the operational workaround and decide whether the data gap is tolerable.

A concise internal status update contains:

1. Incident identifier and severity under the organization's policy.
2. Affected service and business processes.
3. Known start time or an explicitly stated uncertainty range.
4. Current customer or operational impact.
5. Current containment and manual workaround.
6. Selected recovery strategy and its known limitations.
7. Latest verified recovery boundary and estimated data gap.
8. Technical and business validation still outstanding.
9. Decision owner and any decision required.
10. Time of the next update, even if no new facts are available.

Do not write “all orders recovered” until the independent reconciliation supports it.
Do not write “no data loss” merely because a restore process succeeded.
If an estimate changes, explain the new evidence and preserve the prior update.
Send external communications through the organization's approved channel.
Protect incident details that could expose credentials, customer information, or exploitable system weaknesses.

### Operational artifact G: recovery acceptance record

The acceptance record connects recovery actions to the agreed RTO endpoint.
It should be filled in during a real incident or drill, not retroactively invented.
For each required check, record a result, evidence location, reviewer, and time.

| Check | Passing evidence | Failure response |
|---|---|---|
| Candidate identified | Catalog ID, boundary, and selection decision | Stop and obtain an approved candidate. |
| Integrity verified | Backup and restored data checks appropriate to the platform | Quarantine and investigate. |
| Application compatibility | Version and schema checks pass | Use a compatible release or an approved migration. |
| Access and isolation | Required read/write and deny tests pass | Block startup or cutover. |
| Monitoring restored | Required probes and alert delivery work | Escalate before accepting unobservable service. |
| Order uniqueness | No duplicate business identifiers under the defined rule | Investigate and reconcile exceptions. |
| Recovery gap | Missing interval and affected IDs identified or bounded | Record uncertainty and use approved workaround. |
| External side effects | Existing picks, orders, and messages checked | Hold replay until owner resolves risk. |
| Residual exceptions | Assigned owners and agreed handling | Defer acceptance if criteria require closure. |
| Cutover approval | Authorized approver and timestamp recorded | Continue safe workaround. |

The acceptance record is not a substitute for the underlying evidence.
It points to query results, logs, change records, and reconciliation artifacts with suitable access restrictions.
If a check is not applicable, record why and who approved that interpretation.
If a check is waived, record the authority and risk rather than marking it as passed.

### Operational artifact H: timing and uncertainty

For an incident or drill, retain at least these timestamps:

1. Last committed data represented by the selected backup.
2. Earliest known sign of disruption.
3. Agreed disruption start used for the objective.
4. Detection by monitoring or an operator.
5. Incident declaration.
6. Approval to restore.
7. Restore start.
8. Restore completion.
9. Technical validation completion.
10. Business reconciliation completion or approved exception handling.
11. Acceptance and traffic cutover, if those are separate events.
12. Final external communication and incident closure.

Each timestamp should have a source, such as a monitored event, audit record, or operator note.
If sources disagree, preserve both and document why one is used for reporting.
An incident may start gradually, so the agreed disruption boundary can be an estimate.
For an uncertain boundary between 10:00 and 10:05 UTC with acceptance at 10:52 UTC, elapsed recovery is between 47 and 52 minutes.
State the interval rather than reporting an unjustified precision of 47 minutes.
Likewise, if the last known data boundary is a range, calculate the resulting range of possible recovery-point ages.

### Operational artifact I: tabletop-to-technical progression

A tabletop exercise asks participants to make decisions from an injected scenario.
It can reveal unclear ownership, missing contact details, or conflicting acceptance criteria without touching systems.
It cannot demonstrate that a backup can be decrypted or that a restore command works.

A technical restore exercise recovers a selected dataset into an isolated environment.
It can reveal broken backup chains, incompatible versions, missing keys, and invalid assumptions about duration.
It may still exclude supplier systems, realistic data volume, production permissions, and real incident pressure.

An integrated exercise combines technical recovery with business reconciliation and communications.
It should have explicit safety boundaries, authorized participants, and a stop procedure.
Record which components were simulated and which were actually exercised.
Increase scope only after the smaller exercise has exposed and resolved safety gaps.
In every format, assign corrective actions and repeat the relevant test after remediation.

### Operational artifact J: after-action analysis

After a drill, separate observations from hypotheses.
“The backup catalog lookup took 12 minutes” is an observation if supported by recorded timestamps.
“The lookup took 12 minutes because the catalog interface is confusing” is a hypothesis until evidence supports that cause.
“Add a printed catalog location to the runbook” is a proposed corrective action, not a measured outcome.

For every issue record:

1. The expected behavior or agreed target.
2. The observed behavior and supporting evidence.
3. The business or security consequence.
4. The scope of the evidence and any uncertainty.
5. A root-cause hypothesis if one is warranted.
6. A named action owner and due date.
7. A test that will demonstrate whether the action worked.
8. A review date and status under the organization's normal process.

Prioritize issues that invalidate the recovery path, such as missing keys or corrupted backups, before cosmetic reporting changes.
Do not infer that an action is effective merely because it has been assigned.
A subsequent controlled exercise should test the changed path.
Do not erase missed targets from the report after remediation; retain the historical result and link the follow-up evidence.

## 6. Worked example

Consider a fictional distributor's order intake and forecast service.
Its business owner sets a target RTO of 60 minutes and a target RPO of 20 minutes for committed order data.
These are illustrative internal targets, not a customer commitment or measured production capability.
The minimum recovered state is a validated order ledger plus an approved plan for reconciling orders after the recovery point.

The latest verified backup contains committed data through 09:43 UTC.
The backup completed at 09:47 UTC.
A storage incident begins at 10:00 UTC and the team detects it at 10:06 UTC.
All times in this example are on the same day.
The gap from data boundary to disruption is `10:00 - 09:43 = 17 minutes`.
The observed recovery point age is therefore 17 minutes, within the 20-minute target for this incident.
Using backup completion at 09:47 would incorrectly report 13 minutes and conceal 4 minutes of data age.

The team authorizes a restore at 10:12 UTC.
It completes the file and database restore at 10:34 UTC.
Technical checks finish at 10:41 UTC.
Reconciliation and business approval finish at 10:52 UTC.
The elapsed recovery time is `10:52 - 10:00 = 52 minutes`.
The observed result is within the 60-minute RTO by 8 minutes.
The restore operation alone took `10:34 - 10:12 = 22 minutes`, which is not the end-to-end recovery time.
Detection alone consumed 6 minutes; authorization consumed another 6 minutes.
The example is a calculation, not a claim of observed performance.

Suppose an independent intake ledger records 1,284 committed orders by 10:00 UTC.
The restored state contains 1,251 orders through 09:43 UTC.
The difference is `1,284 - 1,251 = 33 orders`.
The team matches 21 orders to source events that can be replayed safely.
It identifies 8 orders already transmitted downstream and marks them for controlled reconciliation rather than blind replay.
Four orders remain unresolved and require manual handling.
The classification totals `21 + 8 + 4 = 33 orders`.
The arithmetic establishes that the categories cover the count difference, not that each matched order is correct.
The team must compare identifiers and status, not just totals.

The owner accepts the service at 10:52 UTC only if the four unresolved orders are assigned to a documented manual queue and acceptance criteria permit that residual risk.
If those four orders cannot be safely handled manually, approval must wait and the measured RTO may be missed.
A time target is not permission to bypass correctness checks.
Likewise, a 17-minute data gap does not imply that the business impact is small: high-value orders could be concentrated in that interval.

### Alternative incident boundary

If the same backup were used for a disruption at 10:09 UTC, the data age would be 26 minutes.
The observed RPO would exceed the 20-minute target by 6 minutes.
The team should report the miss even if the restore completes quickly.
Possible causes include a failed backup, an invalid later backup, or a schedule that was never sufficient under actual timing.
The timeline alone does not identify which cause occurred.

### Why counts are necessary but insufficient

Suppose the recovered ledger had 1,251 rows but one order was duplicated and a different order was missing.
The row count would still match the expected recovered count.
A uniqueness check plus a set comparison of order identifiers catches this particular pair of errors.
Amount, status, and downstream confirmation checks may reveal further mismatches.
No single simple check proves all business semantics.

### Evidence ledger

| Evidence | Example value | Interpretation |
|---|---|---|
| Agreed disruption start | 10:00 UTC | Starts the elapsed RTO clock. |
| Verified data boundary | 09:43 UTC | Starts the data-age calculation. |
| Backup completion | 09:47 UTC | Operational fact, not the data boundary. |
| Technical checks complete | 10:41 UTC | Candidate state is technically checked. |
| Business acceptance | 10:52 UTC | Ends the stated RTO clock in this example. |
| Latest valid data age | 17 minutes | Within the illustrative 20-minute RPO. |
| End-to-end elapsed time | 52 minutes | Within the illustrative 60-minute RTO. |
| Orders after recovery boundary | 33 orders | Must be reconciled individually. |

## 7. Code, query, or calculation

### Lab contract and safety

This standalone lab uses only the Python standard library and a temporary directory.
It does not require Docker, Kubernetes, cloud credentials, a database server, or network access.
Use Python 3.9 or later and permission to create temporary files.
Copy the code below into a new local file named `recovery_lab.py`, then run `python recovery_lab.py`.
On Windows, `py recovery_lab.py` is an alternative if the Python launcher is installed.
The temporary directory is removed automatically when the process exits.
No production data, services, or credentials are involved.

The fixture simulates three valid orders in a quiescent JSON file.
The lab backs up the exact bytes, records a manifest with a data boundary and SHA-256 digest, and restores a copy to a distinct directory.
It rejects a deliberately corrupted backup before restoring the intact copy.
It checks the restored digest, schema shape, unique order identifiers, total quantity, and expected identifiers.
It prints JSON event logs and one summary of metrics and target calculations.
The fixed timestamps make results reproducible; they are not measured runtime performance.

A local directory separation is **not** network, identity, or credential isolation.
This fixture does not provide transactional snapshots for a live database, authenticated manifests, encryption, immutable retention, or point-in-time replay.
The backup and manifest share a local trust boundary; a malicious actor able to change both could forge a matching digest.
The test of corruption detection is not a test of attacker resistance.
The lab simulates a disruption using timestamps and a separate restore directory; it does not destroy the source.

```python
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def event(name, **fields):
    print(json.dumps({"event": name, **fields}, sort_keys=True))


def validate_orders(data):
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError("orders must be a list")
    ids = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"order_id", "quantity"}:
            raise ValueError("invalid order shape")
        order_id, quantity = row["order_id"], row["quantity"]
        if not isinstance(order_id, str) or not order_id:
            raise ValueError("invalid order identifier")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("invalid quantity")
        if order_id in ids:
            raise ValueError("duplicate order identifier")
        ids.add(order_id)
        total += quantity
    return ids, total


def restore(backup, manifest, destination):
    data = backup.read_bytes()
    if sha256(data) != manifest["sha256"]:
        raise ValueError("backup checksum mismatch")
    validate_orders(data)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "orders.json"
    target.write_bytes(data)
    restored = target.read_bytes()
    if sha256(restored) != manifest["sha256"]:
        raise ValueError("restored checksum mismatch")
    return validate_orders(restored)


def main():
    fixture = [
        {"order_id": "A-101", "quantity": 7},
        {"order_id": "A-102", "quantity": 3},
        {"order_id": "A-103", "quantity": 5},
    ]
    boundary = utc("2026-05-01T09:43:00Z")
    disruption = utc("2026-05-01T10:00:00Z")
    accepted = utc("2026-05-01T10:52:00Z")
    assert boundary.tzinfo == timezone.utc
    rpo_minutes = (disruption - boundary).total_seconds() / 60
    rto_minutes = (accepted - disruption).total_seconds() / 60
    metrics = {"backup_verified": 0, "restore_success": 0,
               "restore_rejected": 0}

    with tempfile.TemporaryDirectory(prefix="recovery-lab-") as temp:
        root = Path(temp)
        source, backup_dir, target = (root / "source", root / "backup",
                                      root / "isolated-restore")
        source.mkdir()
        backup_dir.mkdir()
        source_file = source / "orders.json"
        source_file.write_text(json.dumps(fixture, sort_keys=True), encoding="utf-8")
        backup_file = backup_dir / "orders.json"
        shutil.copyfile(source_file, backup_file)
        manifest = {
            "backup_id": "fixture-001",
            "data_boundary_utc": boundary.isoformat(),
            "sha256": sha256(source_file.read_bytes()),
        }
        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert sha256(backup_file.read_bytes()) == loaded_manifest["sha256"]
        metrics["backup_verified"] += 1
        event("backup_verified", backup_id=manifest["backup_id"],
              data_boundary_utc=manifest["data_boundary_utc"])

        corrupt = backup_dir / "corrupt.json"
        corrupt.write_bytes(backup_file.read_bytes() + b"corrupt")
        try:
            restore(corrupt, loaded_manifest, root / "rejected-restore")
        except ValueError as exc:
            assert str(exc) == "backup checksum mismatch"
            metrics["restore_rejected"] += 1
            event("restore_rejected", reason="checksum_mismatch")
        else:
            raise AssertionError("corrupted backup was accepted")
        assert not (root / "rejected-restore" / "orders.json").exists()

        ids, total = restore(backup_file, loaded_manifest, target)
        assert ids == {"A-101", "A-102", "A-103"}
        assert total == 15
        assert source_file.read_bytes() == (target / "orders.json").read_bytes()
        metrics["restore_success"] += 1
        event("restore_validated", backup_id=manifest["backup_id"],
              order_count=len(ids), quantity_total=total)

        assert rpo_minutes == 17 and rto_minutes == 52
        assert rpo_minutes <= 20 and rto_minutes <= 60
        event("drill_summary", metrics=metrics, recovery_point_minutes=rpo_minutes,
              recovery_time_minutes=rto_minutes, rpo_target_minutes=20,
              rto_target_minutes=60)
        print("PASS: backup, corruption rejection, restore, and fixture checks")


if __name__ == "__main__":
    main()
```

### Expected observations

The program prints four JSON event lines, followed by a `PASS` line.
The first reports `backup_verified` for `fixture-001` and the data boundary `2026-05-01T09:43:00+00:00`.
The second reports `restore_rejected` with reason `checksum_mismatch`.
The third reports `restore_validated` with 3 orders and a quantity total of 15 units.
The summary reports 1 verified backup, 1 rejected restore, and 1 successful restore.
It reports a 17-minute recovery-point age and a 52-minute recovery time against 20-minute and 60-minute targets.
A failed assertion or exception means the corresponding fixture check did not pass.
The output does not contain the temporary path and the temporary files are removed on exit.

### Guided inspection and extension

1. Read the fixture and predict the order count and quantity total before running the program.
2. Run the program and compare each printed event with the expected observations.
3. Explain why the corrupted file is rejected before a destination file is created.
4. Change a fixture quantity to another positive integer and predict which explicit assertion will fail.
5. Restore the original fixture; then change the boundary to `09:39` UTC and predict the RPO target assertion result.
6. Explain why a matching manifest alone cannot establish that the fixture represents all real orders.
7. Describe how a database-native backup and transaction log chain would replace the simple file copy for a live database.
8. Describe a network and identity isolation test that the local-directory exercise cannot perform.

Do not interpret these changes as a production failure injection procedure.
Use disposable local files only, and restore the original script before sharing its output.

## 8. Data, standards, and regulatory context

### Data classification and minimization

Recovery material can contain the same sensitive information as the primary system, sometimes for longer.
An order ledger may include personal data, contact information, pricing, supplier contracts, or commercially sensitive demand.
Classify both data and metadata, including logs, manifests, exception queues, and drill reports.
Use synthetic or appropriately authorized and de-identified data when the exercise does not require real records.
A manifest containing only a digest and a recovery boundary may still reveal operational patterns.

Data minimization does not mean removing fields needed for recovery without analysis.
It means selecting the smallest authorized dataset and the shortest appropriate retention consistent with the recovery and legal obligations.
A retention schedule for live records should be reconciled with backup expiration, legal holds, and deletion handling.
Consult the organization's privacy and legal teams for the applicable rules.
The local fixture is synthetic and does not establish legal compliance.

### NIST guidance

The National Institute of Standards and Technology (NIST) Computer Security Incident Handling Guide, Special Publication 800-61 Revision 2, describes incident handling activities including preparation, detection and analysis, containment, eradication, recovery, and post-incident activity.
It is guidance rather than a universal law for every organization.
Organizations should check the current status and newer guidance before adopting a particular revision.
The NIST Cybersecurity Framework 2.0 includes a Recover function and uses outcomes rather than prescribing this lab's implementation.
The framework does not certify that a recovery target is met.

NIST SP 800-34 Revision 1 discusses contingency planning for information systems, including recovery strategies and testing.
Its business-impact concepts help connect technical dependencies to service priorities.
It does not make a local file copy an application-consistent database backup.
The appropriate operational design depends on the platform and threat model.

### ISO and related standards

ISO 22301 specifies requirements for a business continuity management system.
It concerns a management system, not a guarantee that any particular backup is restorable.
ISO/IEC 27001 addresses an information security management system and requires controls to be selected and managed in context.
Possession of a standard or a product feature alone is not evidence of certification or an effective drill.
The official ISO descriptions linked in section 16 are the starting point; access to full standards may require purchase.

### Privacy law and contractual duties

The European Union General Data Protection Regulation (GDPR) includes provisions relevant to security of processing, including restoration of availability and access in a timely manner after an incident, as stated in Article 32(1)(c).
Applicability depends on the organization, processing, and circumstances.
The regulation does not specify the 20-minute RPO or 60-minute RTO used in this fictional example.
Retention and erasure requests involving backups require a context-specific assessment; this topic does not supply legal advice.
Contracts or sector-specific rules may impose additional obligations.
Do not infer compliance from completion of a toy lab.

### Metrics and logs as data

Define each metric's unit, time window, population, aggregation, and missing-data behavior.
A counter of restore successes should be accompanied by a count of attempts and failures.
A backup-age gauge should reflect the latest verified recovery boundary, not just the last job invocation.
A timestamp can be affected by clock skew, delayed delivery, or an untrusted source.
Use independent or synchronized time sources as appropriate and state uncertainty in incident timelines.

Logs should contain enough context to reconstruct actions without storing unnecessary secrets.
A structured event can include a backup identifier, operation result, and reason code.
Access-control, retention, and tamper-evidence requirements apply to the log pipeline as well as to backups.
An attacker who can erase both production data and its audit trail may obscure the incident boundary.
For this reason, assess independent logging and backup administrative boundaries.
The lab prints logs to standard output and deliberately makes no tamper-resistance claim.

### Recovery boundary metadata

Record the boundary in a stable format with a time zone or a domain-specific sequence.
A UTC timestamp is helpful but may not be sufficient for a distributed transaction.
For a streaming consumer, store the offset or equivalent cursor with the state it represents.
For a model service, preserve the model version, feature logic, configuration, and compatible runtime along with the data needed for validation.
For each dependency state how it is recovered and how compatibility is verified.

### Interpreting an RPO for periodic capture

An RPO is a bound on the data-loss interval the business is prepared to tolerate.
It is not the same thing as a timer configured on a scheduler.
For periodic capture, the age of the most recent valid boundary grows as time passes.
The age falls only when a newer valid boundary becomes available and is verified.

Consider a policy that starts one capture every 15 minutes.
Suppose each capture takes 4 minutes and is verified 2 minutes later.
If each capture represents state at its start, a newly verified backup is already 6 minutes old when it becomes available.
Immediately before the next successful verification, the prior boundary can be nearly 21 minutes old.
That assumes every capture succeeds and the schedule has no drift.
A 20-minute RPO therefore cannot be guaranteed by this simplified schedule even without a failed job.

The example does not model database-specific transaction logs or overlapping captures.
Continuous log shipping, coordinated snapshots, and different verification strategies have different timing behavior.
For any chosen mechanism, derive the worst-case age from its actual data boundary and availability semantics.
Include job failures, retry time, detection delay, and chain validity in the operational risk assessment.
Measure the boundary continuously and alert before its age becomes unacceptable.

### Retention and recovery point selection

Recovery point age is not the only reason to retain multiple copies.
A corruption may remain undetected for days while new backups continue to capture the damaged state.
A business error, malicious change, or faulty transformation can therefore affect several consecutive copies.
Retaining multiple verified recovery points provides options, but the retained points still need to be protected and usable.

An organization may need different retention intervals for operational recovery, investigation, contract, and legal requirements.
Those intervals can conflict with minimization and deletion obligations.
Do not invent a universal number of days from this educational example.
Document the applicable policy, decision authority, and exception process.
Verify that expiration actually occurs when authorized and that legal holds are handled through the approved process.

If an old point is selected because it precedes corruption, the data gap can be much larger than the normal RPO.
Estimate the recovery point age using the actual incident boundary and the selected backup's verified boundary.
Communicate the exposure before replaying transactions.
The reconciliation workload may increase with the size of the gap, but elapsed time alone does not establish the number or value of affected orders.
Use source records to determine the actual business impact.

### Supply-chain-specific verification controls

Recovery controls should reflect the consequences of each state transition.
An order can exist in an analytics ledger while a supplier has already accepted or rejected a purchase order.
A shipment can have left a facility while the recovered transport system still shows it as planned.
An inventory balance can be mathematically consistent yet stale relative to a physical count.
An allocation recommendation can be internally valid yet depend on an outdated capacity constraint.

For an order workflow, compare business identifiers, quantities, status, and downstream acknowledgments.
For a warehouse workflow, compare pick tasks with scan events and physical exception records.
For transport, compare shipment status with carrier confirmations and proof-of-delivery records where available.
For a forecast service, verify model version, feature definitions, input freshness, and representative output ranges.
For an optimization service, preserve the input snapshot, objective parameters, constraints, solver version, and decision approval.
These examples are controls to design and test, not guarantees that every required business condition has been captured.

Where independent records disagree, make the disagreement visible.
Do not resolve it by choosing whichever count agrees with the restored database.
Assign each exception an owner, evidence, decision, and final disposition.
Separate “missing from restored state” from “never executed in the physical world.”
The former can often be established from records; the latter may require external confirmation.

### Choosing a metric for recovery drills

Report both target and observation, with the target's scope and measurement endpoints.
Useful drill measures include time to detect, time to authorize, restore execution time, validation time, and end-to-end recovery time.
Each duration should have a start and end event, unit, and evidence source.
Do not sum stage durations unless the stages are non-overlapping and use compatible boundaries.
If validation runs in parallel with restoration, adding both elapsed durations double-counts the overlap.

Report data age at disruption separately from time to recover.
Also report the number of transactions reconciled, the number unresolved, and the source of each count.
A lower unresolved count is not automatically better if the team closed exceptions without adequate evidence.
Record whether the workload, environment, staffing, and integrations were representative.
Small synthetic fixtures are suitable for correctness demonstrations, not capacity or throughput claims.

## 9. Industry practice and case evidence

### What documented guidance supports

The NIST contingency planning guide describes testing recovery arrangements rather than relying exclusively on plans.
The NIST incident handling guide treats recovery as part of a broader lifecycle, with follow-up after the immediate event.
These sources support the practice of assigning roles, maintaining procedures, and evaluating recovery evidence.
They do not validate the numbers or timings in this fictional lab.

The U.S. Cybersecurity and Infrastructure Security Agency's ransomware guidance recommends maintaining offline, encrypted backups and regularly testing their availability and integrity in a disaster-recovery scenario.
This is security guidance, not a promise that every offline backup will recover a particular application.
The distinction matters when a backup is readable but a required key, software dependency, or business reconciliation source is missing.

### Publicly documented incident: Maersk and NotPetya

A. P. Moller - Maersk publicly described the significant impact of the June 2017 cyberattack in its 2017 annual report.
The report discusses disruption and recovery costs in the company's own reporting context.
Use the report for the documented company-level account rather than treating a classroom reconstruction as an audited technical timeline.
**Inference:** the case illustrates why dependency mapping and recovery exercises matter to global logistics operations.
The inference does not establish that the specific local Python procedure in this topic was used by Maersk.
It also does not establish a general RTO for other organizations.

### Publicly documented incident: GitLab database recovery

GitLab published a detailed account of its January 2017 database incident, including the consequences of an accidental deletion and difficulties with multiple backup and replication mechanisms.
The postmortem is a primary account from the affected operator.
Its value is the documented distinction between believing backups exist and proving that the selected recovery path works.
Technology, controls, and organizational practices may have changed since that incident.
Do not generalize its particular recovery time or data loss to a different service.

### Translating evidence into practice

A team can use these cases to ask whether a backup can be found under pressure, whether the credentials and dependencies are available, and whether the restored state is independently checked.
Those questions are applicable to supply-chain analytics because the output may trigger procurement, transport, or warehouse activity.
The public cases do not establish that every supply-chain system has the same architecture or regulatory obligations.
A company case should inform scenario design, not replace a local business-impact analysis.

### Evidence quality and boundaries

Primary incident reports are useful but may omit details for security, legal, or commercial reasons.
Vendor documentation describes intended behavior for a named product and version; test it in the actual configuration.
A tabletop exercise tests decisions and communication, while a technical restore tests a narrower execution path.
Neither alone proves that people, infrastructure, external integrations, and business reconciliation will work under a real incident.
Combine different forms of evidence and record the scope of each.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Backup job never ran | Missing heartbeat or stale boundary | Scheduler or credentials fail silently | RPO gap grows | Alert on missing success and telemetry. |
| Backup completes but data is inconsistent | Restore queries fail or dependencies disagree | Files copied during concurrent writes | Unusable or misleading state | Use supported consistent capture and restore drills. |
| Digest and backup both altered | Unexpected catalog changes | Same principal can replace both | Integrity check gives false confidence | Protect metadata independently or authenticate it. |
| Encryption key unavailable | Key recovery has never been exercised | Key exists only in failed account | Backup cannot be read | Maintain and test authorized independent key recovery. |
| Replica contains corruption | Same bad records appear on both nodes | Change propagates immediately | Failover preserves the error | Retain independently protected historical points. |
| Restore writes to live integration | Unexpected outbound calls in a drill | Production endpoint remains in configuration | Duplicate orders or notifications | Enforce network and identity isolation before startup. |
| Queue position mismatched | Duplicate or missing events after replay | State and offset captured at different boundaries | Repeated or omitted business actions | Capture coordinated checkpoints and reconcile. |
| Metrics pipeline unavailable | No data rather than explicit failure | Monitoring shares failed dependency | Incident is detected late | Monitor telemetry absence independently. |
| Backup age uses completion time | Dashboard looks fresher than data | Long capture delays the boundary | RPO breach hidden | Record and alert on verified data boundary. |
| Restored row count matches but identifiers differ | Count passes while exceptions rise | Duplicate and missing rows offset | Incomplete order ledger | Compare keys and independent source records. |
| RTO clock starts too late | Reports exclude detection or approval | Team measures only restore execution | Management sees a false pass | Agree start and acceptance criteria before incidents. |
| Backup chain has a missing segment | Latest incremental cannot be applied | A transaction-log interval is absent | Recovery point moves backward | Validate chain continuity and retain earlier valid points. |
| Old schema incompatible | Application fails after restore | Code and data versions diverge | Extra recovery time or data error | Preserve compatibility metadata and test migrations. |
| Logs expose credentials | Tokens appear in incident output | Unfiltered payload logging | Secondary security incident | Redact, restrict access, and audit logging policy. |
| Cardinality overwhelms monitoring | Metrics series count grows rapidly | Per-order labels create many series | High cost and missing signals | Use bounded labels and put identifiers in controlled logs. |
| Backup deletion shares production privileges | Same administrator can remove all copies | Inadequate separation | Malicious or accidental total loss | Separate roles and enforce retention where available. |
| Drill ends at process startup | Dashboard says healthy but ledger is wrong | No business acceptance criteria | Unsafe return to service | Require reconciliation and owner sign-off. |
| Failback loses recovered writes | Primary and recovered state diverge | Cutover ignores new transactions | Newly accepted orders disappear | Plan synchronization, verification, and rollback. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or address it |
|---|---|---|---|
| Fixture is quiescent | File copying represents one coherent state. | Concurrent writes during copy. | Use a database-supported consistent snapshot for real workloads. |
| Manifest is trusted | Digest comparison needs an expected value. | Attacker changes both file and manifest. | Protect or authenticate catalog separately. |
| Python standard library is available | Lab has no external dependencies. | Missing or incompatible interpreter. | Confirm Python 3.9+ before the exercise. |
| Temporary directory is writable | Lab creates source, backup, and restore files. | Permissions or storage exhaustion. | Use an approved writable local environment. |
| Local output is non-sensitive | Logs and fixture are printed without redaction. | Replacing fixture with real customer data. | Keep synthetic data; apply policy for real data. |
| Timestamps use one UTC timeline | Arithmetic compares consistent instants. | Clock skew or ambiguous local time. | Synchronize clocks and record uncertainty. |
| Data boundary is accurate | RPO calculation depends on it. | Metadata records completion instead of last committed data. | Verify boundary from the capture mechanism. |
| Restore destination is separate | Exercise avoids overwriting its source. | Path configuration points to the source. | Use explicit isolated destinations and production controls. |
| Business owner defines acceptance | RTO endpoint must be meaningful. | No authority for residual risk. | Assign owner and criteria before the drill. |
| External reference is independent | Reconciliation needs another source of evidence. | Both sources share the same corruption. | Map common failure domains and seek independent records. |
| Fixture checks cover only stated properties | Assertions are finite and known. | Unknown domain constraints or side effects. | Add application-specific invariants and acceptance tests. |
| Fixed times are illustrative | Lab is reproducible without waiting for an incident. | Mistaking simulated times for performance measurements. | Label them synthetic and measure real drills separately. |
| Backup contains necessary state | Restore can reconstruct a minimal ledger. | Missing schema, keys, code, or downstream status. | Inventory and test all real dependencies. |
| A directory is adequate for this demonstration | Only local files are touched. | Real processes access networks or shared credentials. | Enforce and test actual isolation boundaries. |

The code intentionally does not set file permissions, encrypt backups, sign manifests, retain historical copies, or enforce access controls.
Those are production design obligations and depend on the operating environment.
It also does not simulate a concurrent database, queues, an application server, a supplier endpoint, or multi-region failover.
Passing its assertions demonstrates only the listed local fixture properties.

## 12. Exercises

Complete the first four exercises individually, then use a partner for the runbook and reconciliation exercises if possible.
All data and times below are fictional.

### Exercise 1: Classify evidence — 10 minutes

A team records `restore_failed` with a backup identifier, a gauge showing the newest valid backup is 42 minutes old, and a file containing a snapshot of the order ledger.
Classify each item as a log, metric, or backup.
Explain which item demonstrates that a restore is usable.

**Deliverable:** Three classifications and one sentence about restore evidence.
**Acceptance:** Each classification is correct and the answer does not equate backup existence with a successful restore.

### Exercise 2: Calculate objectives — 15 minutes

The data boundary is 08:22 UTC, the disruption starts at 08:40 UTC, detection occurs at 08:47 UTC, and business acceptance occurs at 09:26 UTC.
The targets are an RPO of 15 minutes and an RTO of 50 minutes.
Calculate observed recovery point age, end-to-end recovery time, and target results.
Explain why measuring from detection would change the apparent RTO result.

**Deliverable:** Two calculations in minutes and two pass/fail judgments.
**Acceptance:** Use the disruption boundary for both calculations and include the full elapsed interval.

### Exercise 3: Map dependencies — 20 minutes

An order API stores orders in a database and publishes events to a broker.
A warehouse service consumes events and creates pick tasks.
A model service reads the same database and a versioned model artifact.

**Deliverable:** A dependency diagram or table, recovery boundaries, and three checks.
**Acceptance:** Include database, broker cursor, warehouse side effects, model artifact, credentials, and a reconciliation owner.
Identify at least one duplicate-action risk.

### Exercise 4: Run and inspect the lab — 25 minutes

Save and run the section 7 Python code in a disposable environment.
Record the event names, counters, recovered order count, quantity total, and synthetic RTO/RPO calculations.
Explain the corrupted-copy test and the limitation of a colocated manifest.

**Deliverable:** A short execution record and two limitations.
**Acceptance:** Four events and the final pass line are present; values match the stated fixture.
Do not submit real credentials or customer data.

### Exercise 5: Draft a runbook — 30 minutes

Design an incident runbook for the order API in exercise 3 after suspected database corruption.
Include incident declaration, preservation, isolation, recovery-point selection, validation, reconciliation, authorization, and communications.

**Deliverable:** A numbered runbook with role assignments, decision branches, and stop conditions.
**Acceptance:** A second operator can distinguish actions requiring approval from read-only checks.
Include a rollback or continued-manual-operation path and a plan for next updates.

### Exercise 6: Reconcile a recovery gap — 20 minutes

The independent intake ledger lists IDs `O1`, `O2`, `O3`, `O4`, and `O5`.
The recovered ledger lists `O1`, `O2`, and `O3`.
The warehouse confirms that `O4` was already picked; no downstream confirmation exists yet for `O5`.

**Deliverable:** A two-row exception ledger and an action for each missing ID.
**Acceptance:** Do not blindly replay `O4`; treat `O5` as unresolved until its downstream state is investigated.
Identify the owner who decides whether replay is safe.

### Exercise 7: Critique an alert — 15 minutes

A dashboard displays `last_backup_job_success = 1` indefinitely after one successful job.
It does not report the data boundary or missing job executions.

**Deliverable:** Two failure scenarios the dashboard misses and a revised alert specification.
**Acceptance:** Address freshness, missing telemetry, unit and window, owner, and response.

### Exercise 8: Evaluate a restore drill — 15 minutes

A team restored an encrypted database into a different directory on the production server.
The restored application retained live supplier credentials and scheduled jobs.
The operator did not start it, but reported that isolation was proven.

**Deliverable:** A short critique and a safer sequence.
**Acceptance:** Distinguish file separation from identity and network isolation; prohibit starting with live endpoints.
Include authorized key recovery and post-restore validation.

### Exercise 9: Write a drill report — 20 minutes

Using the fictional example in section 6, prepare a report for the business owner.
Include scope, timestamps, calculations, evidence, residual risk, and next action.

**Deliverable:** A one-page report.
**Acceptance:** Label all observations fictional, distinguish 22-minute restore execution from 52-minute end-to-end recovery, and list the four unresolved orders.
Do not claim production readiness.

## 13. Self-check answers

### Exercise 1

`restore_failed` is a structured log event.
The 42-minute backup-age value is a gauge metric, assuming its definition uses the latest verified data boundary.
The snapshot file is a backup artifact, subject to its consistency and retention properties.
None of those items alone demonstrates a usable restore.
That requires an actual restore and technical plus business validation appropriate to the service.

### Exercise 2

The observed recovery point age is `08:40 - 08:22 = 18 minutes`.
It misses the 15-minute RPO by 3 minutes.
The observed recovery time is `09:26 - 08:40 = 46 minutes`.
It meets the 50-minute RTO by 4 minutes.
Measuring from detection produces `09:26 - 08:47 = 39 minutes`, omitting 7 minutes of outage.
That 39-minute number may describe detection-to-acceptance, but it is not the specified end-to-end RTO measurement.

### Exercise 3 rubric

A strong diagram connects the order database to both event publishing and model inference.
It includes broker retention and the consumer cursor, the warehouse's already-executed actions, model artifact and feature version, and recovery credentials.
A consistency boundary ties the database commit and event state together or explicitly specifies how to reconcile them.
A strong check compares order IDs to source records, validates event replay and idempotency, and confirms model compatibility.
The business owner resolves discrepancies that automation cannot safely decide.
A weak answer lists only a database and a web server.

### Exercise 4

Expected event names are `backup_verified`, `restore_rejected`, `restore_validated`, and `drill_summary`.
The counters are 1 verified backup, 1 rejected corrupted restore, and 1 successful restore.
The fixture contains 3 orders and 15 units.
The synthetic calculations are 17 minutes for recovery point age and 52 minutes for elapsed recovery time.
The corrupt bytes no longer match the digest, so restore stops before writing a destination file.
The manifest is not authenticated and shares a trust boundary with the backup.
The exercise does not establish isolation against live systems or production recovery performance.

### Exercise 5 rubric

A strong runbook names an incident commander, technical operator, security contact, data owner, and business approver or explicitly combines roles for a small team.
It records the incident boundary and preserves evidence before destructive actions.
It selects a valid recovery point, isolates the target, runs technical checks, reconciles external side effects, and records a decision.
It contains stop conditions for failed integrity, missing keys, and unsafe connectivity.
It states how to operate manually and when to communicate again if recovery is not accepted.
A weak runbook consists only of commands that overwrite the production database.

### Exercise 6

`O4` is absent from the recovered ledger but already has a confirmed warehouse action.
Record it as an exception, preserve the confirmation, and reconcile the order state without generating a second pick.
`O5` is absent and its downstream state is unknown.
Investigate the warehouse and source logs before deciding whether replay is safe.
The order or warehouse business owner should approve the resolution according to the incident plan.
The absence of a confirmation is not proof that no side effect occurred.

### Exercise 7 rubric

A stale success value can remain 1 even if the next backup never runs.
A successful job can also contain an old recovery boundary or become invalid before restore.
An improved alert measures `current time - latest verified data boundary` in minutes, with a documented threshold and evaluation window.
A separate missing-telemetry condition detects when the measurement itself stops arriving.
The alert names an on-call owner and directs them to inspect the catalog, job status, and last valid recovery point.

### Exercise 8 rubric

A directory on the production server does not isolate network, identity, storage administration, or scheduled tasks.
Live credentials could permit supplier actions as soon as the application starts.
Prepare a separate authorized environment, deny production egress and writes, disable schedules, and replace credentials before starting anything.
Recover the key through an approved path, verify the chosen backup, restore, and perform technical and business checks.
Document what was tested rather than declaring isolation proven from a file path.

### Exercise 9 rubric

The report should state that all numbers are fictional and identify the system and recovery scope.
It should show a 17-minute recovery-point age, a 52-minute end-to-end recovery time, and a 22-minute restore execution interval.
It should record the 33-order gap, its 21/8/4 classification, and the unresolved four-order queue.
It should describe acceptance as conditional on the stated manual handling criteria.
It should list an owner for the next action and avoid extrapolating a production capability.

## 14. Interview questions

1. **How do RTO and RPO differ?**
   A strong answer distinguishes elapsed service recovery from data age at disruption and names the endpoints.
   A weak answer treats both as backup frequency.
2. **What ends the RTO clock?**
   A strong answer uses a pre-agreed service acceptance state including required business checks.
   A weak answer assumes a process start or file copy always ends it.
3. **Why can a backup job succeed while recovery fails?**
   A strong answer mentions inconsistent capture, missing keys, incompatible schema, broken chains, or unavailable dependencies.
   A weak answer assumes a successful scheduler status guarantees usability.
4. **How would you verify recovered orders?**
   A strong answer combines integrity, schema, uniqueness, independent ID-level reconciliation, and side-effect checks.
   A weak answer relies only on row count.
5. **What does a checksum establish?**
   A strong answer explains byte comparison against a trusted digest and the limitations of a colocated manifest.
   A weak answer claims it proves authenticity and business correctness.
6. **How would you isolate a restore drill?**
   A strong answer covers identity, network, storage, endpoints, scheduler behavior, and negative tests.
   A weak answer relies only on a different filename.
7. **What should an alert for backup freshness measure?**
   A strong answer uses the latest verified data boundary, units, threshold, missing data, owner, and action.
   A weak answer checks only that the scheduler is running.
8. **How can replay create a duplicate warehouse action?**
   A strong answer explains the mismatch between restored state and already-completed external effects and proposes idempotency plus reconciliation.
   A weak answer assumes replay is always safe.
9. **What changes when the newest backup is contaminated?**
   A strong answer selects an earlier verified point, quantifies the larger gap, and obtains business approval.
   A weak answer restores the newest file without checking the incident boundary.
10. **How should a team report a local drill?**
    A strong answer states scope, timings, evidence, exclusions, and unresolved risk without extrapolation.
    A weak answer declares the entire organization production-ready from one synthetic test.

## 15. Key terms

- [Alert](#4-definitions-and-vocabulary): a condition that prompts a defined response.
- [Backup](#4-definitions-and-vocabulary): retained recovery material for a defined state.
- [Business acceptance](#step-11-accept-communicate-and-cut-over): owner approval of checked recovery and residual risk.
- [Consistency boundary](#step-2-map-dependencies-and-consistency-boundaries): state relationship needed for coherent recovery.
- [Counter](#4-definitions-and-vocabulary): count of events over a defined process lifetime or interval.
- [Data boundary](#step-6-validate-capture-not-just-scheduling): newest committed state included in a backup.
- [Detection time](#4-definitions-and-vocabulary): time a failure is noticed.
- [Disruption start](#4-definitions-and-vocabulary): agreed beginning of impact.
- [Failback](#4-definitions-and-vocabulary): controlled return to the intended primary.
- [Failover](#4-definitions-and-vocabulary): move to an alternative operational instance.
- [Gauge](#4-definitions-and-vocabulary): measurement that can increase or decrease.
- [Idempotency](#step-10-reconcile-business-state): property supporting safe repetition under specified conditions.
- [Incident](#4-definitions-and-vocabulary): event requiring coordinated response.
- [Integrity digest](#4-definitions-and-vocabulary): value for detecting changed bytes against a trusted expectation.
- [Isolation](#step-7-prepare-an-isolated-restore-environment): separation preventing unintended interactions.
- [Log](#4-definitions-and-vocabulary): record of an individual event.
- [Manifest](#4-definitions-and-vocabulary): metadata describing a backup and its boundary.
- [Metric](#4-definitions-and-vocabulary): numerical measurement with defined meaning and unit.
- [Point-in-time recovery](#4-definitions-and-vocabulary): recovery to a selected state using suitable retained history.
- [Reconciliation](#step-10-reconcile-business-state): comparison and resolution against independent evidence.
- [Recovery point objective](#4-definitions-and-vocabulary): maximum acceptable recovery-point age.
- [Recovery time objective](#4-definitions-and-vocabulary): maximum acceptable elapsed disruption-to-recovery time.
- [Restore](#4-definitions-and-vocabulary): reconstruction of state from recovery material.
- [Restore drill](#4-definitions-and-vocabulary): controlled rehearsal of restore and verification.
- [Runbook](#4-definitions-and-vocabulary): documented operational actions and decision points.
- [Snapshot](#4-definitions-and-vocabulary): representation of state at a stated boundary.
- [Synthetic fixture](#7-code-query-or-calculation): artificial data with known properties for a repeatable test.
- [UTC](#4-definitions-and-vocabulary): time standard used for comparable timestamps.

## 16. Sources

1. NIST, [Computer Security Incident Handling Guide, SP 800-61 Revision 2](https://csrc.nist.gov/pubs/sp/800/61/r2/final), 2012. Incident-handling lifecycle; consult NIST for current guidance.
2. NIST, [Contingency Planning Guide for Federal Information Systems, SP 800-34 Revision 1](https://csrc.nist.gov/pubs/sp/800/34/r1/upd1/final), 2010. Contingency planning, recovery strategies, and testing.
3. NIST, [The NIST Cybersecurity Framework 2.0](https://www.nist.gov/cyberframework), 2024. Risk-management outcomes, including recovery.
4. CISA, [#StopRansomware Guide](https://www.cisa.gov/stopransomware/ransomware-guide). Backup and incident-preparation guidance; check the published revision date when applying it.
5. ISO, [ISO 22301 — Security and resilience — Business continuity management systems](https://www.iso.org/standard/75106.html). Official standard description; full requirements may require purchase.
6. ISO, [ISO/IEC 27001 — Information security management systems](https://www.iso.org/standard/27001). Official standard overview.
7. European Union, [Regulation (EU) 2016/679, GDPR, Article 32](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Security of processing; applicability requires legal assessment.
8. GitLab, [GitLab.com database incident](https://about.gitlab.com/blog/2017/02/01/gitlab-dot-com-database-incident/), 2017. Operator's incident account.
9. A. P. Moller - Maersk, [Annual reports](https://investor.maersk.com/financials/annual-reports), 2017 report. Company reporting on the cyberattack and its impact.
10. Python Software Foundation, [Python standard library documentation](https://docs.python.org/3/library/). Reference for `hashlib`, `json`, `tempfile`, `pathlib`, and `datetime` used in the lab.

## 17. Links

- [Module 30 index](_index.md).
- [Module 30 survey and labs](../../../course/30_simulation_optimization_production_labs.md).
- [Environment and repository lab](01_environment-repo.md).
- [Streaming events lab](07_streaming-events.md).
- [Model service and container lab](08_model-service-container.md).
- [Kubernetes and Terraform lab](09_kubernetes-terraform.md).
- [Course expansion specification](../../../course/MODULE_EXPANSION_SPEC.md).
