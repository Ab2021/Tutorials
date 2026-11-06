# Reproducibility and Audit Trail

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic file handling, arithmetic, forecasting vocabulary, and decision framing; Modules 07 and 25 are helpful.  
> Version 2026.5

## 1. Why this topic exists

A purchasing manager disputes an order recommendation made six months ago.
The current dashboard shows a different forecast for the same item and week.
A data engineer explains that late sales were corrected, the promotion calendar was revised, and the model was retrained.
The model owner knows the current code but cannot identify the container or parameter values used at the earlier decision.
The planner remembers making an override but the application retained only the final order.
The auditor asks for the original recommendation, the information available at the cutoff, the authorized action, and the evidence that links them.
Without a preserved run and decision trail, none of those can be established reliably from today's dashboard.

Reproducibility is the ability to reconstruct a specified result from an adequately identified set of inputs, instructions, software, and execution conditions.
An audit trail records the events and evidence needed to understand who did what, when, under which authority, and with what effect.
The two support each other but are not interchangeable.
A rerun may reproduce a number while leaving an unauthorized override undetected.
An event log may document an approval while lacking the historical data needed to rerun the model.

The business consequence depends on the decision.
A wrong weekly demand forecast can increase spoilage or stockouts.
A reconstructed forecast that silently uses today's corrected data can hide a feed outage that was visible at the original cutoff.
An order might have been reasonable based on available evidence even when later demand differed.
Conversely, a numerically correct prediction can have been deployed outside its approved scope.
Audit therefore asks both what the system calculated and how that calculation became a business action.

The immediate customer of the evidence may be a planner, model owner, validator, operations investigator, internal auditor, external examiner, or affected supplier.
Their questions and access rights differ.
Evidence design must make the relevant result discoverable without making all sensitive underlying data available to everyone.
The model owner is accountable for the package; the data custodian, platform operator, planner, approver, and independent reviewer each contribute distinct records.

Reconstruction is not an attempt to travel back in time by querying the current database.
It is a controlled replay of what was available and executable at a specific boundary.
If that boundary cannot be reconstructed, say so and quantify the consequences instead of labeling a current rerun as historical proof.
A partial reconstruction can still inform an investigation, but it must be marked partial.

### Prerequisite bridge

Module 06 is deferred in the expanded edition; this topic does not require its expanded files.
A **model** maps inputs into an estimate, score, or decision recommendation.
A **feature** is an input supplied to the model; a **parameter** is a learned or configured value used by it.
A **training set** is the information used to fit learned parameters; an **inference run** applies an existing fitted model to inputs.
A **snapshot** is a preserved representation of data at a defined time or version.
A **hash** is a deterministic digest of bytes, useful for checking that an artifact has not changed relative to a trusted reference.
A **dependency** is software or external functionality needed to execute the run.

A forecast estimates uncertain future demand.
A plan proposes actions based on that estimate and other constraints.
A target expresses a desired outcome, not a prediction.
A commitment is an authorized promise to a customer or supplier.
The example preserves all four separately.

## 2. Learning outcomes

After studying this topic you can:

1. **Define** a historical reconstruction question with a decision identifier, time boundary, and acceptance standard; see Exercise 1.
2. **Map** code, data, parameters, environment, outputs, and human decisions into an evidence graph; see Exercise 2.
3. **Select** a point-in-time dataset using availability rather than only business-effective dates; see Exercise 3.
4. **Calculate** a forecast, override, order, and financial exposure from preserved values; see Exercise 4.
5. **Run** a small replay and distinguish a digest mismatch from a numerical result mismatch; see Exercise 5.
6. **Design** role-based access, retention, and independent review without exposing unnecessary sensitive records; see Exercise 6.
7. **Assess** a missing or nondeterministic dependency and issue a bounded audit conclusion; see Exercise 7.
8. **Propose** an evidence-preserving remediation and retest for a failed reconstruction; see Exercise 8.

A successful answer identifies the evidence that would disprove the claim, not just the evidence that supports it.
A learner should be able to redraw the flow in the next section and explain why each arrow matters.

## 3. The core mental model

```text
Business question + decision ID + knowledge cutoff
                    |
                    v
     approved version / scope / authority
                    |
                    v
historical data --> transformations --> model + configuration
     |                    |                     |
     |                    |                     v
source versions       code + environment    model output
     |                    |                     |
     +---------> run manifest <-----------------+
                     |                         |
                     v                         v
             evidence integrity       planner override + approval
                     |                         |
                     +------------+------------+
                                  v
                     plan -> commitment -> actuals
                                  |
                                  v
                independent replay + review conclusion
```

Read the diagram from left to right as a dependency graph.
The model output is not self-explanatory: it depends on exact data, transformation logic, fitted artifacts, configuration, and execution conditions.
The same code can produce different results with a different cutoff or calendar.
The same input data can produce different results with a different model binary or rounding rule.
A planning action can differ legitimately from the model output after a documented human judgment.

The manifest binds these dependencies to a run identifier.
It records immutable identifiers and digests where feasible, and points to access-controlled stores rather than copying every sensitive record into a report.
A digest verifies bytes against a trusted recorded digest; it does not make the bytes available, prove their provenance, prove their correctness, or establish that the original record was authorized.
The manifest itself must be protected from undetected alteration.

The decision timeline and the data timeline are related but distinct.
A transaction may be effective on Monday, arrive on Wednesday, and be corrected the following Friday.
For a Tuesday decision, a Wednesday-arriving transaction was not available even if it refers to Monday.
A query using only the transaction date can leak later knowledge into an apparently historical replay.
Time zone, cutoff, latency, and backfill behavior are therefore part of the model's information boundary.

There are several levels of replay.
**Bitwise replay** requires identical output bytes.
**Numerical replay** permits a stated tolerance for floating-point calculations.
**Decision replay** asks whether the same decision would be made under a specified rule, while also exposing numerical differences that do not cross the decision boundary.
These are different claims, not an automatic ranking of business usefulness.
Choose and approve the criterion before looking at the replay result.

Evidence has a lifecycle.
It is captured at run time, sealed or otherwise protected, indexed, retained, periodically checked for recoverability, reviewed under controlled access, and eventually disposed of under an approved policy.
Retaining a checksum without the underlying artifact will not allow a replay.
Retaining an artifact without a trusted reference makes later integrity comparison weaker.
Keeping everything forever is not automatically appropriate: privacy, contractual, security, and legal obligations may require limits or deletion.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Run identifier | Unique reference to one execution | Treating model name as a unique run |
| Decision identifier | Reference to a business decision or decision line | Assuming a run always maps to exactly one order |
| Model version | Identified implementation and fitted artifact | Treating a descriptive label as proof of bytes |
| Source revision | Version or commit of executable logic | Recording a branch name that later moves |
| Dirty working tree | Local changes not included in the recorded revision | Assuming the commit alone identifies executed code |
| Dependency lock | Resolved package versions and, where available, hashes | Assuming a version range is a lock |
| Environment image | Identified operating system and runtime dependencies | Treating a mutable image tag as a digest |
| Model artifact | Serialized coefficients, weights, rules, or solver state | Recording only the training script |
| Parameter | Learned or configured model value, with units where applicable | Confusing run settings with trained weights |
| Configuration | Operational settings for a particular run | Assuming defaults never change |
| Seed | Initial state supplied to a pseudorandom generator | Assuming a seed guarantees deterministic execution |
| Snapshot | Immutable or recoverable data state at a named version | Taking a current extract and calling it historical |
| Event time | When an event occurred in the business world | Treating event time as ingestion time |
| Effective time | When a fact applies to the business | Assuming effective time means known time |
| Availability time | When a fact was available to this decision process | Using an unverified upstream publication time |
| Transaction time | When a version was recorded by a system | Assuming all downstream systems received it immediately |
| Cutoff | Latest permitted knowledge time, including time zone | Using a date with no time or timezone |
| Bitemporal record | Versioned fact with effective and system-time dimensions | Keeping only the latest effective value |
| Late-arriving data | An event received after its business occurrence | Silently including it in an earlier run |
| Backfill | Later insertion or correction of historical records | Treating backfilled history as original knowledge |
| Lineage | Relationships among source, transformation, artifact, and output | Assuming a data catalog alone verifies execution |
| Provenance | Evidence of origin and history of an artifact | Treating a filename as provenance |
| Digest | Fixed-length hash of bytes, such as SHA-256 | Treating a digest as encryption or a signature |
| Signature | Cryptographic authentication of data by a key | Assuming unsigned hashes identify an author |
| Manifest | Structured record of run artifacts and metadata | Listing files without versions or owners |
| Immutable storage | Storage protected against alteration under defined controls | Assuming a writable bucket is immutable |
| Retention | Period and rule for keeping records | Setting a duration without legal or business basis |
| Legal hold | Suspension of normal disposal for identified evidence | Deleting held material under ordinary retention |
| Least privilege | Restricting access to what a role needs | Giving auditors unrestricted production access |
| Segregation of duties | Separating conflicting responsibilities | Letting the developer self-approve a high-risk exception |
| Bitwise equality | Exactly equal bytes | Assuming equal numbers imply equal file bytes |
| Numerical tolerance | Approved bound on numerical differences, in output units | Choosing a tolerance after seeing a failure |
| Decision equivalence | Same decision under a specified policy | Assuming it proves identical model results |
| Override | Authorized human or system alteration to recommendation | Recording only the final plan |
| Audit event | Attributable record of an action, time, and object | Treating an editable application note as complete evidence |
| Reconstruction | Replay or supported rebuilding of a historical result | Running today's production model on today's data |
| Independent validation | Challenge by someone with appropriate separation and authority | Assuming independence guarantees competence |
| Exception | Documented deviation with owner, risk, expiry, and disposition | Hiding an unrecoverable artifact in a footnote |
| Retest | Evidence-based check after corrective action | Closing a ticket because an owner says it is fixed |

## 5. The mechanics, step by step

### 5.1 Establish the question and scope

1. Identify the business decision and the individual decision line.
   Record the model inventory identifier, run identifier, item, location, horizon, and output field.
   If a run produced thousands of recommendations, specify whether the audit covers one line, the full batch, or a representative sample.
   Record the requesting party, reason for review, and date of request.
2. State the exact claim.
   For example: "Using information available by 2026-04-07 12:00 UTC, the released model recommended 118 units for item A at site North for the following week."
   A claim about an order placed with a supplier is not answered by replaying the forecast alone.
3. Identify the decision cutoff and applicable calendar.
   Record whether the endpoint is inclusive, how timestamps were assigned, and whether the source time or downstream availability time controls eligibility.
   Record the business week definition and time zone, including daylight-saving transitions.
4. Specify the output comparison before replay.
   Choose byte equality, numerical tolerance, or a decision-equivalence rule.
   Identify rounding mode, units, reference output, null treatment, and handling of duplicate keys.
   A tolerance should be justified by the numerical method and the decision's sensitivity, not by the observed discrepancy.
5. Assign accountability.
   The model owner identifies the release and explains intended use.
   The data owner establishes source versions and time semantics.
   The platform custodian retrieves protected artifacts.
   The business owner supplies override and commitment evidence.
   The validator challenges the reconstruction, and an appropriately separate approver resolves material exceptions.

**Input:** decision request and model inventory.
**Output:** a signed or otherwise controlled reconstruction charter.
**Branch:** If the decision cannot be uniquely identified, stop short of asserting that the replay refers to the disputed decision; resolve identifiers or report the uncertainty.

### 5.2 Inventory and freeze the evidence

6. Retrieve the original manifest rather than creating a new manifest from today's environment.
   Record its creation time, owner, storage location, and integrity protection.
   If it is missing, preserve available evidence and mark the reconstruction as assembled after the fact.
7. Identify executable logic.
   Capture the source repository, immutable commit, relevant submodule revisions, and any uncommitted patch.
   Record notebooks, SQL, orchestration definitions, shell scripts, spreadsheet formulas, and rule configuration if they affected the run.
   A commit hash does not include untracked files, external services, runtime flags, or the interpreter.
8. Identify the model artifact.
   Record fitted weights or coefficients, training run identifier, feature schema, preprocessing state, and digest.
   A model may depend on encoders, scalers, vocabulary, or a lookup table separate from the primary weights.
   Capture each component needed to transform the historical input into the historical prediction.
9. Identify runtime dependencies.
   Record interpreter and package lock, container image digest, operating system and relevant architecture.
   For numerical libraries record versions and settings that affect arithmetic or parallelism.
   Record solver version, license-dependent features, thread count, and optimization tolerances when applicable.
   Record random generator algorithm and state where needed, not only an integer seed.
10. Identify input versions.
    Capture snapshots, immutable object versions, table versions or transaction logs, extraction queries, schema versions, and source system identifiers.
    Record reference calendars, item mappings, unit conversions, exchange rates, promotions, constraints, and business rules.
    A data warehouse query is not a snapshot if the underlying tables can be revised without recoverable history.
11. Identify output and action evidence.
    Preserve raw prediction, postprocessing output, planner-visible recommendation, overrides, approvals, submitted order, acknowledgments, and later outcomes.
    Link each by run and decision identifiers.
    Record whether the output was merely proposed, approved, transmitted, or accepted.
12. Seal the evidence package in a protected repository.
    Calculate cryptographic digests over exact bytes, store them in a protected manifest, and record object versions.
    Sign or separately protect the manifest according to policy.
    Keep copies and backup keys recoverable; test restoration rather than assuming a backup is usable.

**Input:** original execution and business systems.
**Output:** a registered artifact inventory with owners, locations, digests, and missing items.
**Branch:** If data retention permits only an aggregate or redacted copy, document precisely which claims can and cannot be independently replayed.

### 5.3 Reconstruct point-in-time inputs

13. Define a row's identity and versions.
    For sales, a key might combine source system, order line, and item; corrections must be distinguishable from duplicates.
    For a slowly changing product hierarchy, record which category assignment was available at the cutoff.
14. Apply the availability rule.
    For an inclusive cutoff, retain versions with availability timestamp less than or equal to the cutoff.
    From those, choose the latest valid version per key according to the source's version ordering.
    Exclude later arrivals even when their business-effective date is earlier.
    Do not choose a later correction just because it is now considered more accurate.
15. Preserve evidence of exclusions.
    Count eligible rows, later arrivals, corrections, missing timestamps, and unresolved duplicate keys.
    A count is a diagnostic, not a substitute for row-level lineage where that lineage is required.
    If timestamps are missing, state the chosen treatment and its effect; do not infer availability silently.
16. Reconstruct derived features using the original transformation and its historical reference tables.
    Respect aggregation windows, business calendars, missing-value rules, and unit conversions.
    A 7-day demand feature may use local operating days rather than seven UTC calendar dates; the convention must be explicit.
17. Compare reconstructed input with the retained original input extract, if one exists.
    Compare keys, row counts, schema, nulls, and normalized values; check the original byte digest when exact bytes are expected.
    A regenerated file can have different bytes because of row order or serialization even if its logical table is equivalent.
    Record both the byte-level result and the declared logical comparison.

**Input:** historical versions, time contract, and transformation logic.
**Output:** point-in-time input dataset and reconciliation report.
**Branch:** If historical versions are unavailable, use a clearly labeled sensitivity analysis only if it answers a narrower question; do not call it an exact replay.

### 5.4 Restore and run in isolation

18. Provision an isolated environment with no ability to place live orders.
    Restore the image and dependencies by immutable identifiers where possible.
    Restrict outbound network access, or provide preserved responses for external dependencies.
    Protect secrets: use a replay identity and retrieve credentials from an approved secret manager rather than putting them in the manifest.
19. Verify each artifact against the protected manifest before execution.
    A mismatch means the artifact differs from the reference; it does not identify which copy is correct.
    Quarantine the mismatch, preserve both versions, and investigate access logs and provenance.
    Do not simply update the expected digest to make the check pass.
20. Restore the exact model artifact and configuration.
    Capture feature order, categorical handling, threshold, rounding, currency, unit of measure, forecast horizon, and any optimization constraints.
    Check whether configuration values were injected by environment variables or remote feature flags.
21. Execute the replay with its own identifier and capture the command, start and end times, identity, logs, output location, and any warnings.
    Preserve the original as read-only evidence; write replay outputs to a separate location.
    A replay must not mutate source snapshots or trigger downstream business actions.
22. Evaluate determinism.
    Fixed seeds help but may not control parallel reductions, hardware differences, remote services, nondeterministic kernels, or external solver behavior.
    If bitwise replay is not technically possible, use a previously approved numerical or decision criterion and explain residual uncertainty.
    Record the actual platform and settings used for the replay.
23. Compare outputs at each material stage.
    Reconcile source-to-feature, feature-to-raw-output, raw-to-rounded-recommendation, recommendation-to-override, and override-to-commitment.
    Differences should be localized before a root cause is claimed.
    If a downstream order differs while the raw forecast matches, investigate policy, inventory, lot sizing, and overrides rather than retraining the model.

**Input:** verified package and authorized replay environment.
**Output:** replay artifacts and stage-by-stage comparison.
**Branch:** If an external service is unavailable and no response was retained, declare the dependency missing; a current response is not historical evidence.

### 5.5 Investigate differences and conclude

24. Classify each discrepancy.
    Separate missing evidence, integrity failure, semantic mismatch, numeric variance, unauthorized action, and changed business outcome.
    One incident can involve more than one category.
25. Trace a discrepancy to the earliest diverging stage.
    Compare the source snapshot and transformation inputs before comparing model coefficients.
    If the input matches but raw output differs, inspect model artifact, library version, randomness, and arithmetic order.
    If raw output matches but order differs, inspect policy and human action history.
26. Assess materiality in decision units and business exposure.
    A 0.01-unit numerical difference may be immaterial before rounding, but a threshold crossing can change an order by an entire case pack.
    State the proposed materiality criterion and whether it was approved before the replay.
    Avoid turning an indicative exposure into an asserted realized loss without outcome and causal evidence.
27. Record the outcome using explicit categories.
    **Reproduced** means all declared checks passed under the stated standard.
    **Partially reproduced** means specified stages passed while named claims remain unresolved.
    **Not reproduced** means one or more required comparisons failed.
    **Not assessable** means missing evidence prevents the specified test; it is not a pass.
28. Issue an accountable report.
    List the question, period, sources, tools, access restrictions, criterion, comparison, exceptions, owner, and conclusion.
    Include evidence references and distinguish facts from inference.
    Escalate integrity or authorization concerns through the incident process rather than changing evidence in place.
29. Remediate prospectively.
    Assign an owner and due date for the broken control, affected population, interim mitigation, and independent closure evidence.
    Preserve the original failure and subsequent corrective history.
    Rerunning a later period after installing a new control demonstrates the new control, not retroactive recovery of missing historical data.
30. Exercise the control periodically.
    Sample releases and operational runs according to risk; include high-impact and changed systems, not only convenient successful cases.
    Test restoration, access revocation, manifest integrity, and replay comparisons.
    Escalate overdue exceptions and review retention against current obligations.

**Input:** differences, decision context, and control standards.
**Output:** bounded conclusion, issue record, and independently reviewed remediation.

### 5.6 Minimum evidence matrix

| Evidence | Minimum identity | Custodian | Review question |
|---|---|---|---|
| Decision | Business key, horizon, cutoff | Planning owner | Which decision is in scope? |
| Release | Approval ID, scope, date | Model owner | Was this version approved for this use? |
| Source code | Commit plus local patch digest | Engineering | Was this executable logic used? |
| Dependencies | Lock and image digest | Platform | Can the runtime be restored? |
| Training artifact | Weights and preprocessing digests | Model owner | Is the fitted model identical? |
| Input data | Snapshot IDs and availability rule | Data owner | Was this information available then? |
| Configuration | Complete resolved values and units | Operator | Did defaults or flags change? |
| Output | Raw and presented predictions | Platform | Are transformations reconciled? |
| Intervention | Actor, time, reason, before and after | Planner | Was the change authorized? |
| Commitment | Transmitted order and acknowledgment | Procurement | What was actually promised? |
| Review | Criteria, findings, and sign-off | Independent reviewer | Is the conclusion supported? |
| Retention | Policy, expiry, legal-hold status | Records owner | Will evidence remain recoverable? |

### 5.7 Capture a manifest that supports an actual investigation

A manifest is a control record, not merely a convenient list of filenames.
Design it around questions an independent reviewer will need to answer.
Separate the identifiers used to locate artifacts from the attributes used to verify them.
An object path may change when storage is reorganized.
A content digest can help identify retained bytes after the move, but only if the reference is trusted.
Record both the original path and the new custody event when material is migrated.
Do not silently replace an old manifest with a new copy.

The following fields are a practical starting point.
The exact serialization is a local design choice.
The values should be resolved at execution time rather than copied from an intended deployment.

| Field group | Example fields | Control purpose |
|---|---|---|
| Identity | Run ID, parent job ID, model ID, decision batch ID | Locate one execution and its business scope |
| Timing | Start, finish, cutoff, timezone, schedule | Distinguish execution from information boundary |
| Code | Repository, commit, dirty patch, script entry point | Identify actual executable logic |
| Runtime | Image digest, operating system, interpreter, architecture | Restore the software environment |
| Packages | Lockfile digest, numerical libraries, solver version | Explain dependency-driven changes |
| Model | Weights digest, preprocessing digest, schema version | Identify fitted model and input contract |
| Inputs | Source ID, snapshot ID, query, timestamp semantics | Recover the historical input population |
| Configuration | Resolved values, feature flags, threshold, units | Avoid invisible defaults |
| Randomness | Generator, seed or state, relevant deterministic settings | Bound stochastic replay behavior |
| Output | Destination, schema, digest, row count | Compare results and detect incomplete batches |
| Decision | Recommendations, interventions, approval references | Link computations to actions |
| Custody | Owner, storage policy, protection, hold status | Maintain recoverability and accountability |
| Exceptions | Missing artifacts, warnings, compensating controls | Prevent a partial package being mistaken for complete |

Some inputs may contain secrets or sensitive commercial values.
The manifest should identify their controlled location and version, not expose secret material in a widely readable file.
Where the secret affects the result, retain an approved recoverable version or a controlled means of accessing it for the authorized retention period.
For a secret used only to authenticate to an external data service, preserve the service response or snapshot where lawful.
Possession of an old API key does not guarantee the remote service can reproduce its old response.

Distinguish a file digest from a logical-table digest.
A file digest changes if compression, row ordering, or newline encoding changes.
A logical-table digest requires a defined schema, ordering, null convention, encoding, and representation for decimals and timestamps.
For example, sorting rows by a nonunique business key is insufficient when ties can be emitted in arbitrary order.
Choose a total order or define a deterministic multiset comparison.
Retain the original file when byte-level provenance is important even if a normalized comparison is also used.

For a large distributed dataset, record partition identifiers and individual digests or another documented integrity scheme.
Record how the collection-level digest was constructed and which partitions it covers.
Do not assume a storage provider's generic entity tag is necessarily a SHA-256 digest of the complete uncompressed file.
The coverage must include late-arriving partitions and separately stored lookup tables.
A summary digest without the partition list may conceal missing data.

The capture process should distinguish a run that never started from a run that failed before producing output.
Create an execution event when the job starts.
Write a completion or failure event when the job ends.
Associate partial outputs with the failure and prevent their use as an approved complete batch.
Record retries as separate attempts with a shared parent job identifier.
If a retry uses a different snapshot or configuration, it is a different execution for audit purposes.

### 5.8 Establish time semantics across systems

A supply chain commonly has several clocks.
The warehouse management system records an event.
An integration layer transports it.
The analytics platform ingests it.
A feature job materializes it.
The forecast service reads it.
The planner sees the resulting recommendation.
Each time can differ.
The relevant availability time is when the particular decision process could actually use the information under the operational workflow.

Suppose a shipment event occurs at 08:00 local time.
The source system publishes it at 08:05.
An integration outage delays platform ingestion until 11:00.
The feature table refreshes at 11:30.
The model runs at 10:00.
Although the business event existed and the source published it before the run, the feature was not available to the model at 10:00.
A historical query that filters only on the event timestamp would give the replay knowledge the model did not have.
If an operator manually supplied the event through an approved separate channel, capture that intervention as another input with its own timestamp.

Document how the pipeline handles an event arriving exactly at the boundary.
An inclusive rule uses `available <= cutoff`.
An exclusive rule uses `available < cutoff`.
Either can be valid if consistently implemented and disclosed.
Distributed clocks can be skewed, so precision finer than the available clock accuracy can imply certainty that does not exist.
Where the order of two near-simultaneous events matters, use reliable ordering metadata or flag the ambiguity.

Represent timestamps with an offset and retain the business timezone when local calendars matter.
UTC supports unambiguous ordering, while the named local timezone supports reconstruction of local operating hours.
An offset alone is not a complete future or historical daylight-saving rule.
Also retain the relevant calendar version: holidays, fiscal weeks, and site operating days may be corrected.
If the calendar changed after the decision, a current date-to-week mapping can alter historical aggregates.

A deletion requires explicit time treatment too.
If an original transaction was later voided, the replay must know whether the void was available at the cutoff.
Using only the latest table state may remove a record that the original run saw.
An append-only change log, temporal table, or immutable snapshot can preserve the relevant history.
The design must still handle privacy obligations and lawful deletion requirements.

The availability of a supplier forecast can be contractual rather than purely technical.
A file may exist in a shared location before a run but be marked preliminary.
If policy says preliminary files must not be used, a timestamp alone does not make the file an eligible input.
Record publication status, approval state, and any access or use restrictions alongside time.
The same discipline applies to promotion plans and management overrides.

### 5.9 Secure custody and independent access

Identify a custodian for each class of evidence.
The custodian controls access and storage, but does not automatically determine the audit outcome.
Assign ownership for source data, model artifacts, execution logs, business decisions, and the manifest.
Avoid relying on a single person's personal account for a critical evidence archive.
When staff leave or change roles, transfer custody and review access.

Define at least three operational roles.
A producer writes run evidence under a restricted service identity.
A custodian administers retention and controlled retrieval.
A reviewer receives read-only access to a scoped package.
For high-risk cases, require independent authorization for changes to retention policy or protection settings.
An emergency access mechanism should be possible but must record who used it and why.

Access logs themselves need protection.
If the person who can alter the evidence can also erase its retrieval and modification history, the logs provide weaker assurance.
Send material security events to a separately protected logging service where feasible.
Record clock synchronization and the retention period for those logs.
The log need not expose a secret's value to establish that it was retrieved.
Periodic review should include failed access attempts and unusual bulk retrievals.

Encryption protects confidentiality only while key access is controlled and keys remain recoverable.
Record the key identifier and the owner responsible for approved restoration, not the key bytes.
Test whether an authorized reviewer can decrypt a sample after key rotation.
Plan for compromised keys, algorithm migration, and the departure of administrators.
When a key is revoked, distinguish prevention of future access from lawful destruction of retained evidence.

Retention should be set by a documented decision, not by a universal number in a course.
Consider the period in which business disputes arise, audit cycles, contracts, privacy obligations, regulatory duties, and technical restore cost.
Different evidence classes may require different schedules.
For example, a model artifact might outlive a raw transaction extract, requiring a narrower future reconstruction claim.
If a legal hold is applied, identify the covered records and suspend their normal disposal as directed by the responsible legal function.
Record the hold's release and resulting disposition.

Redaction requires care.
Removing a supplier name from a review report can be appropriate.
Removing the only stable supplier key from the replay dataset can prevent a join or make an exception impossible to investigate.
Use scoped identifiers or an authorized mapping service where the business question requires linkage.
The reviewer should know whether a redaction changes the computation or only the presentation.
Record both the original protected artifact and the redacted derivative when retention and law permit.

### 5.10 Design comparison criteria before execution

Start with the business question.
If the dispute concerns the exact file transmitted to a supplier, compare the original message bytes or a formally defined canonical message.
If it concerns a forecast computed by different processor architectures, a justified numerical tolerance may be necessary.
If it concerns an allocation decision, compare feasible actions, constraints, and materially affected customers as well as objective value.
The criterion must match the claim being made.

For numerical comparison, specify an absolute tolerance in output units.
If relative tolerance is used, define how it behaves near zero and for negative values.
For example, a rule can require `abs(replay - reference) <= a + r * abs(reference)`, with approved nonnegative `a` and `r`.
Specify whether the rule applies to each line, aggregate totals, or both.
An aggregate can match while large line-level errors cancel.
Conversely, a small line-level difference may be operationally important if it crosses a minimum-order threshold.

Rounding belongs in the comparison contract.
Distinguish the raw model value from the displayed rounded value and the policy action.
When two raw values round to the same integer, the displayed recommendation may match even though the raw comparison fails.
When two close raw values fall on opposite sides of a threshold, the action may differ materially.
Record the rounding mode explicitly; different systems handle half values differently.
Currency calculations also need the relevant precision and conversion time.

Compare sets as well as values.
A batch replay that returns 99 of 100 item-location rows cannot pass merely because the 99 common values match.
Record the expected key set, missing keys, extra keys, duplicate keys, and value discrepancies.
Specify whether ordering matters for the output's intended use.
For an ordered dispatch list, sequence can change the operational decision.
For an unordered table, order may be irrelevant after a canonical comparison.

Document exception handling in advance.
A missing artifact is not a numerical mismatch, and a numerical mismatch is not necessarily evidence of tampering.
An access restriction can make a test not assessable even when the artifact exists.
The issue severity should consider decision materiality, affected population, confidence, and compensating evidence.
Where the evidence is contradictory, report the contradiction and preserve both sources.

### 5.11 Reconstruct training separately from inference

An inference replay applies a particular fitted model to a particular input.
It does not prove that training the model again would produce the same fitted artifact.
When the claim concerns how a model was developed, the training run needs its own evidence package.
That package includes training and validation snapshots, split definitions, code, hyperparameter search space, selection rule, and resulting artifact.
It also includes the identities and times of approvals and any manual selection among candidates.

Training data may be substantially larger and more sensitive than inference data.
The retention decision should therefore address the training population explicitly.
A snapshot identifier is useful only if the referenced version remains recoverable for the required period.
If deletion or contractual restrictions make that impossible, preserve lawful derived evidence and explain which training claims cannot later be independently repeated.
Do not assert full training reproducibility based solely on a saved model file.

For stochastic training, record the random generator and all known seed settings.
Capture dependency versions, hardware characteristics, parallelism, and determinism settings.
Some kernels and distributed training operations can remain nondeterministic.
A reproducible process may be defined in terms of an approved distribution or metric range, but that is not an exact artifact replay.
If the original fitted artifact is retained, inference may still be exactly reproducible even when training is not.

Feature selection and hyperparameter tuning can leak validation information.
Preserve which data were used for each choice and the sequence in which candidates were evaluated.
An audit that restores the final model without this history cannot establish that the holdout was independent.
Link to the backtesting topic for the temporal evaluation design.
This topic supplies the evidence chain rather than asserting the model's predictive quality.

### 5.12 Handle external and interactive dependencies

An external weather service, exchange-rate feed, route engine, or vendor model can change without a local code change.
Record the request, response, response timestamp, version identifier where offered, and relevant terms of use.
Capture the actual response used rather than assuming a later request will return the same result.
Where a response cannot be retained, document that constraint and the narrower replay claim.
Contract negotiations can seek historical replay access, notification of changes, and evidence retention.

An interactive notebook may contain outputs created by cells executed out of order.
The notebook's displayed source is not enough to establish the in-memory state that produced an output.
For material decisions, capture an executable pipeline or a controlled notebook execution record with a clean kernel and pinned dependencies.
Record the input versions and final exported artifact.
A screenshot can support what a user saw but does not substitute for executable logic and input provenance.

A spreadsheet may depend on external workbooks, manual edits, hidden sheets, macros, volatile functions, and cached values.
Retain the workbook version, formulas, external references, calculation settings, and applicable approval history.
Distinguish a displayed cached value from a recalculated value.
If a macro or external feed is unavailable, the workbook alone may not reproduce the original result.
As with code, a hash verifies the retained workbook's bytes against a trusted reference but not the correctness of its formulas.

For an AI service that generates text, retain the relevant prompt template, supplied context, tool responses, model identifier, generation settings, and output when policy and law allow.
Do not assume a provider's model label uniquely identifies immutable weights or that temperature zero guarantees identical output.
Document where the service may have changed and how sensitive data were handled.
An exact output record may support what was shown even if the provider cannot rerun the same generation.
Any human acceptance or editing of the output remains a separate action to evidence.

### 5.13 Translate control failures into operational decisions

The first response to an integrity concern is to preserve evidence.
Restrict further modification through the established incident process.
Record the discovery time, observer, affected identifiers, and current storage locations.
Do not overwrite a suspicious artifact with a clean copy.
Do not accuse an individual on the basis of a digest mismatch alone.
Clock differences, serialization changes, corruption, and unauthorized modification are distinct hypotheses.

An incident owner should determine whether the affected model or decision is still in use.
If a high-impact model cannot be traced to its approved release, a business owner may need a temporary manual review or controlled fallback.
The appropriate action depends on the materiality and operational risks of both continuing and stopping.
Record the decision and its authority.
An interim control should have an expiry or review date, not become an untracked permanent exception.

Define the affected population before reporting remediation.
If one scheduled job used a mutable image tag, identify the releases and runs that used that tag.
Investigate whether registry history or deployment logs can recover the original digest.
Where the history is incomplete, distinguish recovered, unrecovered, and unaffected runs.
The presence of one good replay does not prove every run used the same image.

An issue record should name the control objective, failure, evidence, owner, risk, due date, and interim treatment.
Corrective action might change manifest capture, retention, access approval, or deployment policy.
Independent retesting should inspect actual executions after the change and attempt restoration.
If the historical evidence cannot be restored, the closure record should retain that residual limitation.
Closure of a prospective control issue is not retroactive proof that past decisions were reproducible.

### 5.14 A compact reviewer checklist

Before accepting a replay, ask the following questions.
Each answer should link to evidence or be marked unknown.

1. Is the disputed business decision uniquely identified?
2. Are the model output, plan, and commitment separate records?
3. Is the original cutoff explicit down to timezone and boundary rule?
4. Does the snapshot reflect what the decision process actually knew?
5. Are corrections and deleted records represented as historical versions?
6. Does the manifest identify actual code, not merely a moving branch?
7. Are uncommitted changes and external scripts accounted for?
8. Are fitted model and preprocessing artifacts recoverable?
9. Are dependencies and runtime settings pinned and retrievable?
10. Are all resolved configuration values and units preserved?
11. Are random state and known nondeterminism documented?
12. Was the reference manifest protected before the replay?
13. Were artifact integrity checks performed before use?
14. Was the replay isolated from operational ordering systems?
15. Were full key sets, counts, and intermediate stages reconciled?
16. Was the comparison threshold established before the outcome was known?
17. Are human overrides attributable and authorized?
18. Is the supplier acknowledgment distinct from order transmission?
19. Are missing items and access limitations visible in the conclusion?
20. Can another authorized reviewer follow the evidence references?
21. Is there an owner for each control failure and its remediation?
22. Was closure independently retested without altering original evidence?

The checklist is a prompt, not a substitute for professional judgment.
For a small low-impact report, some controls may reasonably be lighter.
For a high-autonomy or high-impact decision, a one-line assertion of reproducibility is insufficient.
The model's risk tier should determine the depth of preservation, sampling, and independent challenge.
Any exception should state the residual risk and authorized owner.

## 6. Worked example

### 6.1 Decision and assumptions

A distributor plans weekly orders for item A at its North distribution center.
The decision is made on 7 April 2026 at 12:00 UTC for the next business week.
The forecast concerns units demanded in that week, not revenue and not a promise to ship.
The planning target is 125 units of available stock.
The supplier commitment is the order actually acknowledged, which can differ from both target and forecast.
The figures are illustrative, not claims about a named company.

The released model uses a simple weighted average of the latest three eligible weekly demand observations.
The weights are 0.2, 0.3, and 0.5 from oldest to newest.
Assume the training and transformation choices have already been approved for this bounded example.
A separate inventory policy subtracts usable stock from the rounded forecast.
There are no open orders or minimum order quantities in this example.
The unit purchase cost is USD 8 per unit; this is cost exposure, not proven economic loss.

### 6.2 Historical source versions

| Week | Demand version | Available UTC | Units | Eligible at cutoff? |
|---|---|---|---:|---|
| W1 | Original | 2026-03-24 08:00 | 96 | Yes |
| W2 | Original | 2026-03-31 08:00 | 112 | Yes |
| W3 | Original | 2026-04-07 09:00 | 126 | Yes |
| W3 | Correction | 2026-04-08 09:00 | 146 | No |

The correction was entered after the decision.
It may be the best current estimate of the actual demand in W3, but it was not available to the earlier decision process.
The relevant historical snapshot contains 96, 112, and 126 units.
Assume the downstream system received the original W3 record at the availability time shown, not merely that the source published it then.

### 6.3 Calculate original recommendation

The weighted calculation is:

```text
0.2 × 96 units  = 19.2 units
0.3 × 112 units = 33.6 units
0.5 × 126 units = 63.0 units
Total            = 115.8 units
```

The policy rounds to the nearest whole unit, with half units rounded upward.
The model forecast presented to the planner is therefore 116 units.
Usable stock is 39 units; the proposed replenishment before intervention is 116 − 39 = 77 units.
The planner adds 8 units to cover an approved local event not in the model inputs.
The final plan is 77 + 8 = 85 units.
The buyer transmits 85 units and the supplier acknowledges 85 units.
The supplier commitment is 85 units, not the 116-unit forecast or the 125-unit stock target.

The 85-unit order has an illustrative purchase value of 85 × USD 8 = USD 680.
That is a cash commitment under the assumed price, not the realized benefit or loss of the model.
If later observed demand is 121 units, forecast error defined as actual minus forecast is 121 − 116 = 5 units.
That later outcome does not change what was known on 7 April.

### 6.4 What goes wrong with current history

An analyst reruns the same weights after the W3 correction and gets:

```text
0.2 × 96 + 0.3 × 112 + 0.5 × 146
= 19.2 + 33.6 + 73.0
= 125.8 units
```

The current-history forecast rounds to 126 units.
Holding the same 39 units of stock and the 8-unit intervention fixed, the hypothetical order is 126 − 39 + 8 = 95 units.
That is 10 units above the historical order, or USD 80 at the assumed unit cost.
The USD 80 is a difference in hypothetical order value, not a demonstrated loss.
This comparison isolates the corrected W3 record only because every other input and rule is assumed unchanged.

### 6.5 Trace and audit conclusion

| Stage | Historical evidence | Comparison |
|---|---|---|
| Cutoff | 2026-04-07 12:00 UTC, inclusive | W3 correction is ineligible |
| Inputs | 96, 112, 126 units | Original three values recovered |
| Weights | 0.2, 0.3, 0.5 | Resolved configuration retained |
| Raw output | 115.8 units | Replay must match within declared tolerance |
| Presented forecast | 116 units | Whole-unit rounding matches |
| Stock | 39 units | Historical stock snapshot required |
| Initial plan | 77 units | 116 − 39 |
| Override | +8 units | Actor, reason, time, and authorization required |
| Commitment | 85 units | Supplier acknowledgment required |
| Outcome | 121 units | Later information, not a model input at cutoff |

Suppose all historical artifacts and authorization records are recovered and checks pass.
The appropriate conclusion is that the specified forecast, plan, and acknowledged commitment were reconstructed for this one decision under the stated assumptions.
It is not evidence that the model is accurate for all items or that the order caused a particular financial result.
If the stock snapshot is missing, the forecast may still be reproduced, but the 77-unit initial plan cannot be independently confirmed.
If the signed supplier acknowledgment is missing, the record supports a submitted order, not necessarily an accepted commitment.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as `replay.py` and run `python replay.py`.
It constructs a small illustrative manifest in memory, selects the most recent version available at the cutoff, checks a protected-reference digest for a canonical input, calculates the result, and reconciles the planner's action.
The expected digest in this self-contained teaching example is calculated from the separately declared reference values; production systems must obtain it from an independently protected, previously captured manifest.
It demonstrates a comparison, not digital signatures, durable storage, source authentication, or production access controls.

```python
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json

CUTOFF = datetime.fromisoformat("2026-04-07T12:00:00+00:00")
RECORDS = [
    {"week": "W1", "available": "2026-03-24T08:00:00+00:00", "units": 96},
    {"week": "W2", "available": "2026-03-31T08:00:00+00:00", "units": 112},
    {"week": "W3", "available": "2026-04-07T09:00:00+00:00", "units": 126},
    {"week": "W3", "available": "2026-04-08T09:00:00+00:00", "units": 146},
]
WEIGHTS = (Decimal("0.2"), Decimal("0.3"), Decimal("0.5"))


def canonical_bytes(values):
    # Canonicalization is part of the comparison contract.
    return json.dumps(values, separators=(",", ":")).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def available_versions(records, cutoff):
    selected = {}
    for row in records:
        available = datetime.fromisoformat(row["available"])
        if available > cutoff:
            continue
        week = row["week"]
        previous = selected.get(week)
        if previous is not None and available == previous[0]:
            raise ValueError("Ambiguous versions at the same availability time")
        if previous is None or available > previous[0]:
            selected[week] = (available, row["units"])
    return [selected[week][1] for week in ("W1", "W2", "W3")]


def main():
    values = available_versions(RECORDS, CUTOFF)
    reference_digest = sha256(canonical_bytes([96, 112, 126]))
    actual_digest = sha256(canonical_bytes(values))
    if actual_digest != reference_digest:
        raise ValueError("Historical input digest mismatch")

    raw = sum((weight * value for weight, value in zip(WEIGHTS, values)), Decimal("0"))
    forecast = int(raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    stock = 39
    initial_plan = forecast - stock
    override = 8
    acknowledged_order = initial_plan + override
    print("input:", values)
    print("digest matches:", actual_digest == reference_digest)
    print("raw forecast:", raw, "units")
    print("presented forecast:", forecast, "units")
    print("initial plan:", initial_plan, "units")
    print("acknowledged order:", acknowledged_order, "units")
    print("order value: USD", acknowledged_order * 8)


if __name__ == "__main__":
    main()
```

Expected output:

```text
input: [96, 112, 126]
digest matches: True
raw forecast: 115.8 units
presented forecast: 116 units
initial plan: 77 units
acknowledged order: 85 units
order value: USD 680
```

The synthetic variable named `acknowledged_order` assumes an external acknowledgment exists.
In a real audit, the arithmetic cannot prove transmission or acknowledgment; retrieve and verify the supplier record separately.
The code uses a fully specified ordered week list and decimal arithmetic to avoid a floating-point tolerance in this particular calculation.
The digest is over a canonical JSON array, not the full source records; it does not verify timestamps, source identity, authorization, or the model configuration.
A production manifest must separately bind those artifacts and the canonicalization method to a protected reference.
The selection logic assumes one authoritative version per week at any given availability time and fails on equal-time ambiguity.
It also assumes timestamps are complete, timezone-aware, and represent actual downstream availability.
For real data, document corrections, deletions, ties, source keys, and late changes rather than extending these assumptions silently.

## 8. Data, standards, and regulatory context

### Data and records

The primary sources are operational transactions, historical master data, data pipelines, model artifacts, decision logs, approval records, and supplier messages.
Each is subject to a different owner and retention rule.
The auditor should request the minimum data necessary for the scoped question and preserve the ability to trace material transformations.
A pseudonymized report can be suitable for broad circulation while a restricted evidence store retains the authorized detail.
Pseudonymization is not equivalent to anonymization, and a digest of a low-entropy identifier may be guessable.
Do not place credentials, personal data, or sensitive supplier pricing in an unrestricted manifest.

Historical access is a separate control from retention.
An encrypted archive is useless for replay if the decryption key is lost or the service account has been retired without a recovery process.
Conversely, indefinitely retained access to plaintext evidence creates avoidable risk.
Use role-based access, time-limited approvals, logged retrievals, key rotation and recovery, and periodic access review.
A legal hold, if applicable, may suspend normal deletion; records and legal teams decide its scope and release.

### Standards and guidance

The US National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 describes voluntary functions for governing, mapping, measuring, and managing AI risk.
Its supporting playbook offers suggested actions, not a certification of any particular replay procedure.
A run manifest and audit trail can support these activities, but compliance with a voluntary framework cannot be inferred from a single manifest.

NIST SP 800-53 Revision 5 describes security and privacy controls including audit and accountability, configuration management, access control, and system integrity.
The control catalog is a source for designing and assessing controls; whether a particular organization must implement a particular baseline depends on its applicable obligations and adoption.
This teaching example is not a control assessment or authorization to operate.

The Federal Reserve and Office of the Comptroller of the Currency issued revised model risk management guidance in April 2026, replacing SR 11-7 for institutions covered by that guidance.
Applicability and supervisory expectations depend on the institution and relevant agency guidance; a supply-chain organization should not assume that bank supervisory guidance directly binds it.
Consult the current primary guidance and counsel rather than treating any course text as a legal conclusion.

The EU General Data Protection Regulation (GDPR) may apply where its territorial and material scope is met.
Article 5 sets principles including purpose limitation, data minimization, storage limitation, integrity and confidentiality; Article 32 addresses security of processing, subject to its terms.
The appropriate retention and access arrangement requires a contextual legal assessment, not an automatic indefinite retention period for audit purposes.

The EU AI Act's applicability depends on system classification, role, and phased dates; not every supply-chain model is a high-risk AI system.
Where applicable, relevant documentation, record-keeping, and logging duties depend on the precise provision and actor.
Consult the current legal text and amendments before setting a compliance deadline or claiming a particular system is covered.
A replay package is useful evidence but does not itself establish compliance.

### Interoperability and semantics

The W3C PROV family describes entities, activities, and agents and relationships among them.
It can structure lineage, but a standards-shaped graph is only as reliable as the observations and controls feeding it.
A run can be an activity, a snapshot an entity, and an operator an agent.
Record the relation between actual execution and the observed source, not merely a diagram of the intended pipeline.

GS1 identifiers can help identify traded items and locations when used under the organization's identification rules.
An identifier by itself does not preserve the historical description, unit conversion, or hierarchy attached to it.
EDI transaction identifiers can help link a purchase order to a supplier acknowledgment, but retain the message version, timestamp, sender, and any later amendments.
Do not infer that a transmitted order was accepted without an acknowledgment or other reliable evidence.

## 9. Industry practice and case evidence

### Documented guidance and bounded inference

The Federal Reserve's historical SR 11-7 guidance described model development, implementation, use, validation, governance, and controls.
The 2026 replacement guidance should be consulted for current applicability and expectations.
A practical inference for supply-chain models is that documented inputs, limitations, and independent challenge help owners examine whether a model's use remains appropriate.
That inference is not a claim that the historical bank guidance legally governed the distributor in the worked example.

The NIST AI Risk Management Framework organizes AI risk work across governance, context, measurement, and management.
Its public playbook gives organizations suggested actions and is not a mandatory checklist for every model.
An evidence manifest can help connect those functions to an actual execution.
The framework does not prescribe the exact JSON structure or numerical tolerance used in this topic.

The UK government's Post Office Horizon IT Inquiry documents a long-running public investigation involving an operational IT system and disputed accounting evidence.
The inquiry's reports and evidence should be consulted directly for findings and chronology.
The general lesson for this topic is not that every unexplained discrepancy proves a system defect or that every logging design would have prevented the events.
It is that disputes involving system records require scrutiny of the entire evidence chain, access, assumptions, and ability to challenge a system's outputs.
Do not treat this educational example as a reconstruction of the Horizon system.

### Operational patterns

A retailer running a daily forecast can capture a run manifest automatically after the pipeline resolves its data versions and software environment.
The manifest should be written only after the execution identity and actual inputs are known, not inferred from a deployment plan.
A failed run should also leave an event record; otherwise the absence of an output may look like a run that never occurred.
For a batch of many items, preserve batch-level artifacts and item-level links without duplicating a large model for every line.

A manufacturer using an optimizer must retain constraint data and solver configuration as well as code and demand forecasts.
If multiple optimal solutions exist, exact decision bytes may differ even when the objective is identical.
Define whether the control requires the exact plan, feasible plans within a cost tolerance, or both, and retain the original selected plan regardless.
A small objective difference may conceal a materially different allocation to a critical customer.

A procurement team using a vendor service may not receive model weights or the vendor's runtime image.
The buyer can still retain the request, timestamp, contractually available version identifier, response, configuration, and subsequent human action.
Ask for a replay or attestation mechanism and disclosure of version changes.
If the vendor cannot reproduce the historical response, the internal conclusion must explicitly exclude full independent model replay.
Contractual assurances can supplement, not replace, a technically possible reconstruction.

A forecasting team may retain a model card summarizing intended use, evaluation, and limitations.
The card is an index into evidence, not a substitute for the actual data versions and execution artifacts.
Keep the card's version synchronized with release approval and distinguish a retrospective correction to the card from the original approval.
A reviewer should be able to follow references to the original run without receiving blanket access to all production data.

### Review of business impact

Audit success is not measured solely by a high percentage of archived runs.
A useful control also measures whether a risk-weighted sample can be retrieved, verified, and replayed within the required investigation time.
The denominator should include failed and abandoned runs where relevant, not only completed runs that were easy to archive.
Track missing-artifact age, critical exceptions, restoration success, access-review completion, and unresolved lineage breaks.
Set targets locally based on materiality and regulatory or contractual requirements; no universal threshold is implied here.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Current data substituted | Historical query returns newly corrected records | Availability time ignored | False past forecast | Version sources and enforce cutoff |
| Mutable source branch | Manifest records `main` only | Branch moves after run | Wrong code replayed | Record commit and dirty patch |
| Missing weights | Training script exists but serialized model does not | Retraining changes parameters | Historical prediction unrecoverable | Retain fitted artifacts and preprocessing |
| Unlocked dependencies | Version ranges only | Library upgrades change behavior | Numerical or execution drift | Lock packages and image digest |
| Hidden environment flag | Local replay differs from scheduled run | Runtime default was injected | Incorrect policy or threshold | Capture resolved configuration |
| Mutable image tag | Same tag resolves to a new image | Registry tag overwritten | False environment identity | Pin and retain image digest |
| Hash without artifact | Digest exists but file expired | Retention mismatch | Integrity check cannot restore data | Align retention and restoration tests |
| Hash without trusted reference | Digest can be edited by artifact owner | Both artifact and digest replaced | Tampering not detectable by comparison | Protect or sign manifest independently |
| Missing human override | Only final plan is stored | Application overwrites recommendation | No action attribution | Append before/after event with reason |
| Unsupported commitment | Sent order treated as accepted | Acknowledgment absent | Overstated supplier obligation | Preserve response and amendments |
| Local clock confusion | Timestamps lack timezone | Daylight-saving or offset ambiguity | Wrong eligibility boundary | Store UTC and business timezone |
| Late-arriving correction | Backfill appears under old event date | Effective date mistaken for knowledge time | Look-ahead leakage | Retain transaction and availability history |
| Random seed overclaim | Seed present, hardware unspecified | Nondeterministic kernels or parallel order | Unexplained replay mismatch | Capture runtime and approved tolerances |
| Excessive privilege | Reviewer has permanent production write access | No separation or access expiry | Evidence alteration or data exposure | Read-only scoped time-limited access |
| Lost encryption key | Restore exercise fails despite backups | Keys expired or owner left | Evidence inaccessible | Controlled key escrow and restore drills |
| Invalid numerical tolerance | Tolerance chosen after replay | Failed result reclassified | Misleading pass | Preapprove units and threshold |
| Unsafe replay | Replay sends orders to live endpoint | Production integration retained | Duplicate business action | Isolate environment and block writes |
| Incomplete lineage | Catalog shows planned rather than executed path | Dynamic branch not captured | Wrong source assumed | Capture runtime lineage and reconcile |
| Premature deletion | Retention job ignores legal hold | Disposal controls not coordinated | Loss of required evidence | Apply documented hold and release process |
| Vendor version drift | Endpoint returns a different answer later | Remote model updated silently | Historical response not reproducible | Save request/response and demand version notice |
| Silent error suppression | Logs show success despite skipped rows | Exception swallowed | Incomplete recommendation batch | Record rejects and reconcile counts |
| Overbroad conclusion | One item replayed and model declared sound | Sample not representative | False assurance | State scope and unresolved risks |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Availability timestamp is accurate | Determines eligible knowledge | Ingestion timestamp delayed or fabricated | Compare source and downstream logs |
| Versions are retained | Allows recovery of original values | Compaction or retention deletion | Restore sampled historical versions |
| Manifest is trustworthy | Provides reference for integrity check | Owner can edit artifact and digest | Verify access controls and signature |
| Input keys are stable | Identifies version history | Item mergers or reused identifiers | Reconcile historical master-data mappings |
| Model artifact is complete | Replays fitted transformation | External encoder or lookup omitted | Run in isolated environment and inventory dependencies |
| Environment can be restored | Reproduces execution behavior | Image deleted or hardware unsupported | Perform periodic recovery exercises |
| Outputs are attributable | Links model and decision | Shared account or missing run ID | Reconcile identity and event logs |
| Rounding rule is known | Converts raw forecast to units | Changed defaults or hidden postprocessing | Compare raw and displayed outputs |
| Action history is complete | Distinguishes forecast from order | UI overwrites prior value | Test append-only intervention history |
| Tolerance is appropriate | Enables a fair comparison | Threshold crossing or arbitrary bound | Analyze decision sensitivity before replay |
| Data access is authorized | Allows lawful review | Expired approval or privacy restriction | Obtain scoped authorization and log access |
| Supplier message is authentic | Supports commitment claim | Forged, superseded, or unacknowledged order | Verify channel, version, and acknowledgment |
| Retention is lawful and sufficient | Supports later investigation | Deletion duty or unexpected hold | Coordinate records, privacy, and legal owners |
| Example's linear formula is representative | Supports worked arithmetic | Production model is nonlinear or stochastic | Tailor artifacts and acceptance criteria to method |

Reproducing a historical number does not establish predictive validity, fairness, conceptual soundness, or causal business value.
A replay establishes only the defined relationship between identified evidence and a result under a stated comparison rule.
The model may have been wrong, approved in error, or used outside scope even if all bytes match.
Conversely, a legitimate nondeterministic implementation can meet an approved numerical or decision standard without bitwise equality.
Neither claim should be extrapolated beyond the audited scope.

## 12. Exercises

### Exercise 1 — Define the claim

A buyer disputes the purchase quantity for item B at site East on 4 May 2026.
The dashboard now displays a revised forecast.
**Deliverable:** a one-paragraph reconstruction charter and a list of five required identifiers.
**Acceptance criteria:** distinguish forecast from order; define decision line, cutoff with time zone, comparison standard, owner, and scope.
**Time:** 15 minutes.

### Exercise 2 — Draw lineage

A source sales feed feeds a daily feature job and a released forecasting service.
A planner edits the recommended quantity, and a buyer transmits an order to a supplier.
**Deliverable:** an evidence graph with at least eight nodes and labeled edges.
**Acceptance criteria:** include time-specific data, transformation code, model artifact, configuration, raw output, override, order, and acknowledgment; identify the owner of each.
**Time:** 20 minutes.

### Exercise 3 — Select the historical version

At a cutoff of 10:00 UTC on 10 June, a record has value 40 units available at 09:00 UTC and a correction to 47 units available at 11:00 UTC.
A separate record has value 5 units available exactly at 10:00 UTC.
The rule is inclusive.
**Deliverable:** eligible values, their sum, and a sentence on why the correction is excluded.
**Acceptance criteria:** apply availability time, not current truth, and include the boundary value.
**Time:** 10 minutes.

### Exercise 4 — Reconcile a decision

A forecast is 118 units; usable stock is 43 units; the planner increases the resulting recommendation by 6 units.
The supplier accepts the transmitted order at USD 9 per unit.
**Deliverable:** initial order, acknowledged quantity, and purchase value with units.
**Acceptance criteria:** show arithmetic and distinguish purchase value from realized loss.
**Time:** 10 minutes.

### Exercise 5 — Examine the replay program

Use the section 7 program and the printed expected output.
Change the W3 original value from 126 to 127 in a private copy without changing the reference array.
**Deliverable:** identify the expected program behavior and the exact control being exercised.
**Acceptance criteria:** explain why the program stops before printing a forecast and why the digest check alone cannot prove provenance.
**Time:** 15 minutes.

### Exercise 6 — Design access and retention

A model uses supplier prices and employee identifiers.
An independent reviewer needs to reproduce one decision, while a broad management audience needs only the result.
**Deliverable:** a role-by-resource access matrix and retention decision process.
**Acceptance criteria:** least privilege, authorization, logging, key recovery, deletion or hold treatment, and an appropriately redacted report.
**Time:** 25 minutes.

### Exercise 7 — Assess missing evidence

The original raw forecast and final order exist, but the historical inventory snapshot and planner override log were deleted.
The retained forecast matches a replay.
**Deliverable:** a bounded audit conclusion and two candidate follow-up sources.
**Acceptance criteria:** do not assert that the order calculation or authorization was reproduced; distinguish missing evidence from proof of unauthorized behavior.
**Time:** 15 minutes.

### Exercise 8 — Remediate a failed control

Three of ten sampled high-impact runs lack the dependency image digest; all have source commits and output files.
**Deliverable:** issue statement, risk, interim mitigation, owner, due date, and independent retest design.
**Acceptance criteria:** distinguish historical recovery from prospective capture; retain original findings; specify sample and closure evidence.
**Time:** 25 minutes.

## 13. Self-check answers

### Answer 1

A suitable charter identifies the item and site, decision line, relevant run, business horizon, and the original decision cutoff with timezone and inclusion rule.
It states whether the question concerns raw forecast, approved plan, transmitted order, or supplier acknowledgment.
It names a model owner and an independent reviewer, a reference output, a predetermined equality or tolerance criterion, and the specific period and population.
Five identifiers might be model ID, release ID, run ID, decision-line ID, and supplier message ID.
A date alone is not an adequate cutoff where the feed updates intraday.

### Answer 2

A strong graph includes source snapshot, feature transformation code, feature dataset, model and preprocessing artifact, resolved configuration, execution environment, raw forecast, presented recommendation, override, transmitted order, and supplier acknowledgment.
Edges should describe actual derivation or authorization, not just arrows with no meaning.
The data custodian owns the snapshot; engineering owns the transformation and runtime; the model owner owns the model; planning owns the override; procurement owns the order record.
Independent review checks the graph and tests selected edges against evidence.
A graph without object versions cannot distinguish a historical state from today's state.

### Answer 3

The eligible values are 40 and 5 units.
Their sum is 40 + 5 = 45 units.
The 47-unit correction was unavailable until 11:00 UTC and is therefore excluded at the 10:00 UTC cutoff.
The 5-unit record is included because the rule explicitly includes the boundary.
If its actual downstream availability differed from its source timestamp, the analysis would need that downstream evidence.

### Answer 4

The initial order is 118 − 43 = 75 units.
The acknowledged order is 75 + 6 = 81 units.
The purchase value is 81 × USD 9 = USD 729.
The value is a commitment under the assumed price, not an estimate of realized loss.
A loss calculation would require additional evidence, including what happened to the purchased units and a defensible comparator.

### Answer 5

The changed array fails the digest comparison and raises `ValueError` before a forecast is printed.
The check establishes only that the selected canonical values differ from the declared reference digest.
Because the reference digest is created in the same script from a hard-coded array, it is not independently protected evidence.
A malicious or mistaken change to both the reference and the input could still produce a match.
It also does not verify source timestamps, output authorization, or the human override.

### Answer 6

A strong design gives the reviewer approved, time-limited read access to only the required snapshots and artifacts, with logged retrieval and no production write permission.
The broad report excludes unnecessary employee identifiers and sensitive price detail.
The custodian maintains encryption keys and tests recovery under separation of duties.
A records owner sets a documented schedule with privacy, legal, contract, and business input, applies holds where relevant, and approves eventual disposal.
A weak design shares all data indefinitely or deletes the only recoverable evidence before obligations are assessed.

### Answer 7

The raw forecast is reproduced under the declared model and input comparison, assuming those artifacts were verified.
The initial inventory-based recommendation, the override's authorization, and the causal chain to the final order are not independently established from the listed records.
Potential sources include an inventory ledger or warehouse snapshot and an approval ticket, planner application event stream, or buyer correspondence.
A candidate source is not proof until its identity, time semantics, and integrity have been checked.
The absence of a log is a control deficiency, not by itself evidence that a planner acted improperly.

### Answer 8

The observed exception rate is 3 ÷ 10 = 30% of this sample; it is not automatically the population rate.
The missing image digest weakens the ability to prove or restore the original runtime for those three runs.
An engineering owner can investigate registry, deployment, and execution logs for recoverable historical identifiers while the model-risk owner records the exposure.
An interim control can require a second-person release check and preserved dependencies for new high-impact runs.
Set a due date according to local severity policy rather than inventing a universal deadline.
The independent retest should sample new runs, inspect the protected manifests and image availability, and perform at least one restore and replay.
Close only when evidence meets the agreed criteria; do not erase the original three exceptions or claim missing past artifacts were recovered without proof.

## 14. Interview questions

1. **What is the difference between a replay and an audit trail?**  
   Strong: distinguishes computation from attributable events, authorization, and evidence integrity.  
   Weak: assumes a reproduced number proves the action was authorized.
2. **Why is an event date insufficient for a historical forecast?**  
   Strong: explains downstream availability, late arrivals, corrections, and cutoff.  
   Weak: selects today's latest record solely by its old business date.
3. **What does a SHA-256 digest prove?**  
   Strong: compares bytes to a trusted reference and names provenance and availability limits.  
   Weak: treats hashing as encryption, signature, or independent proof of truth.
4. **How would you handle a missing dependency image?**  
   Strong: seeks recoverable evidence, bounds the conclusion, and records a control issue.  
   Weak: replaces it with the current image and calls the replay exact.
5. **Can a seed guarantee identical results?**  
   Strong: discusses library, hardware, parallelism, external calls, and comparison criteria.  
   Weak: assumes a seed controls every nondeterministic source.
6. **How do you reconstruct a planner intervention?**  
   Strong: requests before/after values, actor, time, reason, approval, and linked order.  
   Weak: infers the override from the final order alone.
7. **What would you retain for an optimizer?**  
   Strong: includes constraints, objective, input versions, solver version and settings, and original solution.  
   Weak: retains only the optimization script.
8. **How do you balance reproducibility and privacy?**  
   Strong: uses scoped access, retention governance, protected identifiers, recovery, and deletion or hold procedures.  
   Weak: either grants permanent full access or deletes all evidence without analysis.
9. **When is decision equivalence insufficient?**  
   Strong: notes material numeric discrepancies, threshold sensitivity, and the separate question of model correctness.  
   Weak: declares all equal actions identical evidence.
10. **How should a vendor black box be evaluated?**  
    Strong: retains request and response, asks for immutable version and replay rights, and states limits.  
    Weak: claims full independent reproducibility from a current API call.

## 15. Key terms

- **[Audit event](#4-definitions-and-vocabulary):** attributable record of an action and its context.
- **[Audit trail](#1-why-this-topic-exists):** linked evidence of activity, authority, and effects.
- **[Availability time](#4-definitions-and-vocabulary):** time information was accessible to the decision process.
- **[Backfill](#4-definitions-and-vocabulary):** later insertion or correction of historical data.
- **[Bitemporal record](#4-definitions-and-vocabulary):** record with effective and system-time dimensions.
- **[Commitment](#1-why-this-topic-exists):** authorized promise, distinct from a forecast or proposed plan.
- **[Configuration](#4-definitions-and-vocabulary):** resolved settings used by a run.
- **[Cutoff](#4-definitions-and-vocabulary):** boundary on information eligibility.
- **[Decision equivalence](#4-definitions-and-vocabulary):** equality of actions under a stated decision rule.
- **[Dependency lock](#4-definitions-and-vocabulary):** resolved versions of software dependencies.
- **[Digest](#4-definitions-and-vocabulary):** fixed-length hash used to compare bytes against a reference.
- **[Effective time](#4-definitions-and-vocabulary):** time a fact applies in the business domain.
- **[Environment image](#4-definitions-and-vocabulary):** identified runtime and its packaged dependencies.
- **[Forecast](#1-why-this-topic-exists):** estimate of a future uncertain quantity.
- **[Independent validation](#4-definitions-and-vocabulary):** appropriately separated challenge of model evidence.
- **[Least privilege](#4-definitions-and-vocabulary):** limiting a role's access to its task needs.
- **[Lineage](#4-definitions-and-vocabulary):** relationships among data, transformations, outputs, and actors.
- **[Manifest](#4-definitions-and-vocabulary):** structured inventory binding a run to its artifacts.
- **[Model artifact](#4-definitions-and-vocabulary):** saved fitted parameters and associated components.
- **[Numerical tolerance](#4-definitions-and-vocabulary):** approved bound on numerical replay differences.
- **[Override](#4-definitions-and-vocabulary):** documented change to a recommendation.
- **[Plan](#1-why-this-topic-exists):** proposed action, which may change before commitment.
- **[Provenance](#4-definitions-and-vocabulary):** documented origin and history of evidence.
- **[Reconstruction](#4-definitions-and-vocabulary):** controlled rebuilding of a historical result.
- **[Retention](#4-definitions-and-vocabulary):** governed duration and conditions for preserving records.
- **[Run identifier](#4-definitions-and-vocabulary):** unique reference to a specific execution.
- **[Seed](#4-definitions-and-vocabulary):** initial pseudorandom-generator state or input.
- **[Snapshot](#4-definitions-and-vocabulary):** preserved data representation at a defined version.
- **[Target](#1-why-this-topic-exists):** desired result, not necessarily a prediction.
- **[Transaction time](#4-definitions-and-vocabulary):** time a record version was registered in a system.

## 16. Sources

1. NIST, **Artificial Intelligence Risk Management Framework (AI RMF 1.0)**, 2023. [Framework and resources](https://www.nist.gov/itl/ai-risk-management-framework). Voluntary risk-management framework, not a legal certification.
2. NIST, **AI RMF Playbook**. [Playbook](https://airc.nist.gov/airmf-resources/playbook/). Suggested practices for the framework's functions; consult current version.
3. NIST, **Security and Privacy Controls for Information Systems and Organizations**, SP 800-53 Rev. 5. [Official publication](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final). Control catalog; applicability depends on adoption and obligations.
4. Board of Governors of the Federal Reserve System, **Supervisory Guidance on Model Risk Management**, SR 11-7, 2011. [Historical guidance](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm). Consult current replacement guidance and relevant supervisory jurisdiction before use.
5. Board of Governors of the Federal Reserve System, **Revised Guidance on Model Risk Management**, SR 26-2, 17 April 2026. [Official guidance and scope](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm). Check the applicable agency and institution scope.
6. European Union, **Regulation (EU) 2016/679**, General Data Protection Regulation. [EUR-Lex official text](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Articles 5 and 32 address principles and security respectively, subject to the regulation's scope and terms.
7. European Union, **Regulation (EU) 2024/1689**, Artificial Intelligence Act. [EUR-Lex official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). Check current amendments and phased application.
8. W3C, **PROV Overview**, W3C Working Group Note, 2013. [Specification](https://www.w3.org/TR/prov-overview/). Conceptual vocabulary for provenance, not an assurance that captured lineage is correct.
9. Post Office Horizon IT Inquiry. [Official inquiry site](https://www.postofficehorizoninquiry.org.uk/). Primary source for reports, evidence, and inquiry chronology.
10. NIST, **Secure Hash Standard**, FIPS 180-4. [Official publication](https://csrc.nist.gov/pubs/fips/180-4/upd1/final). Defines approved hash algorithms; hashing alone does not establish authenticity.
11. Python Software Foundation, **Python 3 Standard Library**. [Documentation](https://docs.python.org/3/library/). Reference for `datetime`, `decimal`, `hashlib`, and `json` used in the example.

Sources are educational, not legal advice; no example's controls imply regulatory compliance.

## 17. Links

- [Module 26 topic index](_index.md): sequence and scope.
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md): overall model-risk context.
- [Model inventory and tiering](01_model-inventory-tiering.md): identify ownership, use, and control burden.
- [Data quality and representativeness](03_data-representativeness.md): assess the population and historical input quality.
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md): use point-in-time data in historical comparisons.
- [Planning model metrics](05_planning-model-metrics.md): evaluate errors and decision-relevant performance.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): apply evidence and replay ideas to operational modeling.
