---
title: "Data Observability and Incident Triage"
topic: "Data observability and incident triage"
collection: "data-quality-observability-and-reliability-engineering"
level: "core"
evidence: "DERIVED"
created: "2026-11-11"
last_reviewed: null
content_status: "draft"
---

# Data Observability and Incident Triage

This sheet is a DERIVED operating reference. All thresholds, timelines, volumes, and cost figures marked ILLUSTRATIVE are constructed teaching inputs, not observed production results.

## 1. File Header

- **Decision supported:** Decide whether a data product is safe to publish, quarantine, replay, or serve in a declared degraded mode.
- **Primary reader:** The on-call data owner who must bound an incident before changing data or infrastructure.
- **Product boundary:** One named dataset or stream, its declared grain, its contract version, and its downstream consumers.
- **Time boundary:** Every check states event time, processing time, observation time, and the evaluation window explicitly.
- **Safety boundary:** A passing transport job never by itself proves that records are complete, correct, timely, or fit for a decision.
- **Evidence boundary:** Operational examples are ILLUSTRATIVE; do not interpret them as claims about a particular organisation.
- **Minimum contract:** Owner, backup owner, consumers, key, grain, freshness objective, quality checks, lineage, and containment action.
- **Escalation trigger:** Page only when a detected breach needs a human decision within the stated response objective.
- **Publication gate:** A product is publishable only when the relevant contract checks pass or an approved exception is recorded.
- **Recovery definition:** Recovery requires corrected output, completed reconciliation, and an evidence record that affected consumers were handled.
- **Non-goal:** This sheet does not select a vendor, establish legal policy, or replace domain validation by the owning team.
- **Notation convention:** Times are UTC; rates are fractions unless shown as percent; counts are records or partitions as named.
- **Operating rule — Name the decision before measuring:** A seven-day volume baseline is meaningless unless the reader knows which delivery or action it gates.
- **Operating rule — Separate detection from diagnosis:** An alert can establish that a symptom exists; it rarely establishes the component that caused it.
- **Operating rule — Prefer bounded comparisons:** Compare like-for-like partitions, source populations, contract versions, and time semantics before declaring drift.
- **Operating rule — Preserve evidence before repair:** Capture query identifiers, run versions, sampled keys, and observed timestamps before a rerun overwrites clues.
- **Operating rule — Use reversible containment:** Quarantine, pause publication, or pin a known-good snapshot before choosing a destructive correction.
- **Operating rule — Record accepted debt:** If an SLO exception is granted, record scope, expiry, owner, consumer impact, and compensating control.
- **Operating rule — Treat lineage as a hypothesis:** Static lineage can be stale; verify the executed run, actual input partitions, and deployed transformation version.
- **Operating rule — Route by ownership:** A monitor without a responsible recipient creates an acknowledgement loop rather than incident response.
- **Operating rule — Distinguish absence from lateness:** No records, delayed records, and records beyond watermark require different recovery and reconciliation paths.
- **Operating rule — Make degradation explicit:** Serving a dated snapshot may be safer than serving a mixed or partially corrected current partition.
- **Operating rule — Measure alert usefulness:** Count pages that lead to an action, not merely checks that emit a notification.
- **Operating rule — Close against the contract:** Close only after the stated SLO, schema, completeness, and consumer reconciliation conditions are evaluated.

### Scope checklist
- **Identity check:** Assign a stable product name and a contract version so historical incidents can be compared without ambiguous aliases.
- **Grain check:** State whether one row represents an order, event, account-day, model prediction, or another unit of observation.
- **Key check:** State the uniqueness key and whether duplicates are invalid, expected, or merged by a deterministic rule.
- **Time semantics check:** State which timestamp drives freshness, which drives partitioning, and how late arrivals are handled.
- **Population check:** State the inclusion predicate, source systems, exclusion rules, and whether deleted records remain visible.
- **Consumers check:** List decision makers, dashboards, models, exports, and materialised products that read the output.
- **Quality dimensions check:** Name the checks for completeness, validity, uniqueness, consistency, distribution, and referential integrity.
- **SLO check:** Write the objective as a measurable target over a named window rather than a statement that data should be reliable.
- **Runbook check:** Give the first five diagnostic queries or observations and the owner to contact for each failure domain.
- **Containment check:** Pre-author whether to quarantine, suppress an export, serve a snapshot, backfill, or halt a dependent action.
- **Reconciliation check:** Define the comparison that proves repaired output matches the authoritative source or approved correction set.
- **Retention check:** Keep enough raw and processed evidence to replay the incident window within the recovery commitment.

## 2. Concept Map / Mental Model

Observability is an inference system around a data product. It converts emitted signals into a claim about contract health, then routes an action that limits incorrect downstream decisions.

### 1. Source emission
- **Role:** A producer emits records, files, or change events with identifiers, event timestamps, and source-version evidence.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the source emission boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the source emission boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 1. Source emission.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 1. Source emission.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 1. Source emission.
### 2. Ingestion boundary
- **Role:** A collector records arrival time, offsets, checksums, rejected records, and the source partition range accepted.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the ingestion boundary boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the ingestion boundary boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 2. Ingestion boundary.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 2. Ingestion boundary.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 2. Ingestion boundary.
### 3. Raw preservation
- **Role:** Immutable landing storage keeps the received bytes or canonical records so diagnosis does not depend on transformed output.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the raw preservation boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the raw preservation boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 3. Raw preservation.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 3. Raw preservation.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 3. Raw preservation.
### 4. Transformation execution
- **Role:** A job applies a versioned rule to declared inputs and produces a run record with parameters and output partition identifiers.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the transformation execution boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the transformation execution boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 4. Transformation execution.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 4. Transformation execution.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 4. Transformation execution.
### 5. Quality evaluation
- **Role:** Tests compute evidence about required fields, keys, referential relationships, distributions, and semantic invariants.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the quality evaluation boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the quality evaluation boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 5. Quality evaluation.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 5. Quality evaluation.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 5. Quality evaluation.
### 6. Publication decision
- **Role:** A gate exposes, quarantines, delays, or labels the output according to the contract and approved exceptions.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the publication decision boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the publication decision boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 6. Publication decision.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 6. Publication decision.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 6. Publication decision.
### 7. Consumption
- **Role:** Dashboards, features, models, applications, and exports read the released product under known freshness and completeness assumptions.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the consumption boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the consumption boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 7. Consumption.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 7. Consumption.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 7. Consumption.
### 8. Detection
- **Role:** Monitors compare observed evidence with baselines, thresholds, or change-point rules and attach a severity estimate.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the detection boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the detection boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 8. Detection.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 8. Detection.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 8. Detection.
### 9. Triage
- **Role:** The responder narrows time, scope, lineage, and symptom class before selecting containment and a repair path.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the triage boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the triage boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 9. Triage.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 9. Triage.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 9. Triage.
### 10. Learning loop
- **Role:** A post-incident change improves the contract, instrumentation, runbook, test, or ownership model and is tested on replay.
- **Observable evidence:** Capture a timestamped, versioned record proving what entered or left the learning loop boundary.
- **Question answered:** Is the current symptom introduced before, within, or after the learning loop boundary?
- **Common misleading signal:** A green status at this stage can coexist with a semantic error established by another stage. In this context, apply it to 10. Learning loop.
- **Decision implication:** Use this boundary to remove whole branches of the incident hypothesis tree, not to assign blame. In this context, apply it to 10. Learning loop.
- **Recovery linkage:** Any repair that crosses this boundary must preserve a before-and-after comparison keyed by run and partition. In this context, apply it to 10. Learning loop.

### Signal-to-decision path
- **Signal:** A measurable fact such as latest accepted event time, row count, null fraction, or schema fingerprint.
- **Expectation:** A contract limit, baseline interval, or distributional reference that makes the signal interpretable.
- **Breach:** A comparison result outside tolerance, accompanied by window, denominator, and time-semantics metadata.
- **Severity:** An estimate of decision harm based on affected consumers, materiality, reversibility, and elapsed exposure.
- **Owner:** The accountable role that can decide containment or coordinate with a component owner.
- **Action:** A reversible containment, diagnostic query, repair, replay, reconciliation, or exception approval.
- **Evidence:** Artifacts retained to support closure: values, query/run versions, sample keys, affected partitions, and consumer status.

### Mental-model distinctions
- **Telemetry versus observability:** Telemetry is emitted data; observability is the ability to answer a new operational question from available evidence.
- **Data quality versus data usefulness:** Values can pass syntactic tests yet be unfit for a pricing, safety, eligibility, or model-training decision.
- **Freshness versus completeness:** A recent partition can omit source records; a complete historical partition can be too old for a real-time consumer.
- **Lineage versus impact:** A dependency graph says what can be reached; actual blast radius depends on which partitions and consumers used the bad output.
- **Anomaly versus incident:** An anomaly is a surprise in a metric; an incident is a breach requiring coordinated action or threatening a decision.
- **Repair versus recovery:** Repair changes data or code; recovery also restores safe consumer state and reconciles actions taken while incorrect data was visible.
- **SLO versus threshold:** An SLO defines expected service over time; a threshold is one detector setting that may be tuned without changing the objective.
- **Suppression versus containment:** Suppressing an alert removes a signal; containment reduces exposure to the potentially incorrect product.
- **Backfill versus replay:** A backfill fills a historical range; a replay reproduces processing from preserved inputs, often to validate a changed transformation.
- **Root cause versus contributing condition:** A root cause explains the initiating defect; contributing conditions explain why detection, containment, or recovery was ineffective.

## 3. Foundational Definitions

- **Data product:** A versioned, owned dataset or stream delivered for named consumers with a declared contract.
  - **Operational use:** Record the data product with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from data product than the stated contract, denominator, and observed evidence support.
- **Contract:** The testable agreement covering grain, schema, semantics, quality limits, freshness, ownership, and response behavior.
  - **Operational use:** Record the contract with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from contract than the stated contract, denominator, and observed evidence support.
- **Grain:** The real-world unit represented by one record, such as one shipment event or one customer-day observation.
  - **Operational use:** Record the grain with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from grain than the stated contract, denominator, and observed evidence support.
- **Primary key:** The field or tuple intended to identify one grain instance; it is not automatically unique until checked.
  - **Operational use:** Record the primary key with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from primary key than the stated contract, denominator, and observed evidence support.
- **Natural key:** A business identifier supplied by a source, which can be reused or corrected and therefore needs temporal interpretation.
  - **Operational use:** Record the natural key with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from natural key than the stated contract, denominator, and observed evidence support.
- **Surrogate key:** A technical identifier assigned by a system, useful for joins but not proof that business duplicates are absent.
  - **Operational use:** Record the surrogate key with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from surrogate key than the stated contract, denominator, and observed evidence support.
- **Event time:** The time when the described business event occurred, carried by the record and potentially corrected later.
  - **Operational use:** Record the event time with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from event time than the stated contract, denominator, and observed evidence support.
- **Processing time:** The time when a pipeline processed a record, which depends on system load and retry behavior.
  - **Operational use:** Record the processing time with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from processing time than the stated contract, denominator, and observed evidence support.
- **Observation time:** The time when a monitor evaluated a signal; it is the anchor for alert and SLO calculations.
  - **Operational use:** Record the observation time with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from observation time than the stated contract, denominator, and observed evidence support.
- **Watermark:** A stated bound used to decide when a window is sufficiently complete, not a guarantee that later records cannot arrive.
  - **Operational use:** Record the watermark with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from watermark than the stated contract, denominator, and observed evidence support.
- **Freshness lag:** Observation time minus the latest valid time under a declared clock; units are seconds or minutes.
  - **Operational use:** Record the freshness lag with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from freshness lag than the stated contract, denominator, and observed evidence support.
- **Completeness:** The fraction of expected eligible records that are present, requiring a credible denominator or reconciliation source.
  - **Operational use:** Record the completeness with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from completeness than the stated contract, denominator, and observed evidence support.
- **Validity:** The fraction of evaluated values satisfying a field-level domain, format, range, or predicate rule.
  - **Operational use:** Record the validity with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from validity than the stated contract, denominator, and observed evidence support.
- **Uniqueness:** The absence or controlled rate of duplicate keys within a stated population and time window.
  - **Operational use:** Record the uniqueness with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from uniqueness than the stated contract, denominator, and observed evidence support.
- **Consistency:** Agreement between related fields, tables, systems, or successive transformations under an explicit join and timing rule.
  - **Operational use:** Record the consistency with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from consistency than the stated contract, denominator, and observed evidence support.
- **Referential integrity:** The property that a foreign key resolves to an allowed parent record or a documented exception class.
  - **Operational use:** Record the referential integrity with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from referential integrity than the stated contract, denominator, and observed evidence support.
- **Schema fingerprint:** A canonical representation of field names, types, nullability, and ordering rules used to identify interface changes.
  - **Operational use:** Record the schema fingerprint with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from schema fingerprint than the stated contract, denominator, and observed evidence support.
- **Semantic version:** A contract identifier whose change signals a meaning, population, calculation, or compatibility difference.
  - **Operational use:** Record the semantic version with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from semantic version than the stated contract, denominator, and observed evidence support.
- **Baseline:** A historical or modelled reference distribution used to judge normal variation for a comparable operating context.
  - **Operational use:** Record the baseline with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from baseline than the stated contract, denominator, and observed evidence support.
- **Control limit:** A detector boundary selected for a specified false-alert and missed-detection trade-off; it is not a universal truth.
  - **Operational use:** Record the control limit with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from control limit than the stated contract, denominator, and observed evidence support.
- **SLO:** A target for a measurable service indicator over a defined window, population, and exclusion policy.
  - **Operational use:** Record the slo with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from slo than the stated contract, denominator, and observed evidence support.
- **SLI:** The service-level indicator measured against an objective, for example on-time valid partitions divided by due partitions.
  - **Operational use:** Record the sli with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from sli than the stated contract, denominator, and observed evidence support.
- **Error budget:** The allowed amount of SLO noncompliance within a window, expressed in failures, duration, or proportion.
  - **Operational use:** Record the error budget with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from error budget than the stated contract, denominator, and observed evidence support.
- **Incident:** A confirmed or credible contract breach that needs coordinated response to prevent or limit consumer harm.
  - **Operational use:** Record the incident with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from incident than the stated contract, denominator, and observed evidence support.
- **Severity:** A prioritisation label derived from scope, materiality, urgency, reversibility, and regulated or safety consequence.
  - **Operational use:** Record the severity with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from severity than the stated contract, denominator, and observed evidence support.
- **Blast radius:** The set of actual or plausible consumers, decisions, partitions, and actions affected by a defective input or output.
  - **Operational use:** Record the blast radius with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from blast radius than the stated contract, denominator, and observed evidence support.
- **Lineage edge:** A recorded dependency from a source artifact or transformation run to a downstream artifact or run.
  - **Operational use:** Record the lineage edge with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from lineage edge than the stated contract, denominator, and observed evidence support.
- **Critical path:** The dependency sequence whose delay or defect prevents a defined product objective from being met.
  - **Operational use:** Record the critical path with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from critical path than the stated contract, denominator, and observed evidence support.
- **Quarantine:** A controlled state that retains output for analysis while preventing its ordinary publication or consumption.
  - **Operational use:** Record the quarantine with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from quarantine than the stated contract, denominator, and observed evidence support.
- **Known-good snapshot:** A versioned output previously verified against the contract and eligible for explicitly declared degraded serving.
  - **Operational use:** Record the known-good snapshot with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from known-good snapshot than the stated contract, denominator, and observed evidence support.
- **Reconciliation:** A comparison proving that corrected output and affected downstream state match authoritative or approved expected results.
  - **Operational use:** Record the reconciliation with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from reconciliation than the stated contract, denominator, and observed evidence support.
- **Idempotency:** The property that repeating an operation with the same identity and input produces no additional unintended effect.
  - **Operational use:** Record the idempotency with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from idempotency than the stated contract, denominator, and observed evidence support.
- **Deduplication:** The process of identifying multiple representations of one grain instance using documented keys and precedence rules.
  - **Operational use:** Record the deduplication with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from deduplication than the stated contract, denominator, and observed evidence support.
- **Backfill:** A controlled production of missing or corrected historical partitions over a bounded time range.
  - **Operational use:** Record the backfill with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from backfill than the stated contract, denominator, and observed evidence support.
- **Replay:** Re-execution from retained inputs, offsets, or snapshots to reproduce or validate a processing result.
  - **Operational use:** Record the replay with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from replay than the stated contract, denominator, and observed evidence support.
- **Run manifest:** The immutable record of code version, inputs, parameters, timestamps, outputs, checks, and outcome for one execution.
  - **Operational use:** Record the run manifest with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from run manifest than the stated contract, denominator, and observed evidence support.
- **Data drift:** A material change in observed data distribution, relationship, or meaning relative to a relevant reference population.
  - **Operational use:** Record the data drift with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from data drift than the stated contract, denominator, and observed evidence support.
- **Distribution shift:** A change in value frequencies or quantiles that can be benign, expected, or symptomatic of a defect.
  - **Operational use:** Record the distribution shift with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from distribution shift than the stated contract, denominator, and observed evidence support.
- **Cardinality:** The number of distinct values in a field or key population; abrupt movement can signal enumeration or join changes.
  - **Operational use:** Record the cardinality with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from cardinality than the stated contract, denominator, and observed evidence support.
- **Null rate:** The number of null values divided by evaluated records for a named field and window.
  - **Operational use:** Record the null rate with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from null rate than the stated contract, denominator, and observed evidence support.
- **False positive:** A detector alert when the contract is not materially breached under the declared evaluation rule.
  - **Operational use:** Record the false positive with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from false positive than the stated contract, denominator, and observed evidence support.
- **False negative:** A material contract breach that the detector fails to alert on within the required response interval.
  - **Operational use:** Record the false negative with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from false negative than the stated contract, denominator, and observed evidence support.
- **Mean time to detect:** Average elapsed time from incident start to credible detection, defined with a consistent start event.
  - **Operational use:** Record the mean time to detect with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from mean time to detect than the stated contract, denominator, and observed evidence support.
- **Mean time to contain:** Average elapsed time from credible detection to an action that prevents further ordinary exposure.
  - **Operational use:** Record the mean time to contain with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from mean time to contain than the stated contract, denominator, and observed evidence support.
- **Mean time to recover:** Average elapsed time from incident start to restored and reconciled service under the closure definition.
  - **Operational use:** Record the mean time to recover with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from mean time to recover than the stated contract, denominator, and observed evidence support.
- **Runbook:** An executable decision aid that maps symptoms to discriminating checks, owners, containment actions, and escalation rules.
  - **Operational use:** Record the runbook with its population, time window, and producing system so it can be compared during triage.
  - **Boundary:** Do not infer more from runbook than the stated contract, denominator, and observed evidence support.

## 4. Core Formulas with Full Derivation

Each result below is DERIVED. A monitor must carry the selected clock, population, and exclusion policy; otherwise arithmetic can be correct while the operational conclusion is wrong.

### 4.1 Freshness lag and on-time test
- **Purpose:** Freshness answers whether the newest valid record or completed partition is recent enough for a consumer.
```text
FORMULA → L_f = t_o - t_v
WHERE → L_f is freshness lag in minutes; t_o is observation time in UTC minutes; t_v is the latest valid event or partition time in the declared population.
CONSTRAINT → The subtraction is dimensional because both times use the same UTC clock. A product is on time when L_f <= L_max, where L_max is the contract limit.
NUMERICAL EXAMPLE → ILLUSTRATIVE: at 10:17 UTC, the latest valid event time is 10:02 UTC, so L_f = 10:17 - 10:02 = 15 minutes.
INTERPRETATION → If L_max = 20 minutes, 15 <= 20 and the evaluated freshness condition passes at 10:17 UTC.
```
- **Derivation note:** The subtraction is dimensional because both times use the same UTC clock. A product is on time when L_f <= L_max, where L_max is the contract limit.
- **Worked value:** ILLUSTRATIVE: at 10:17 UTC, the latest valid event time is 10:02 UTC, so L_f = 10:17 - 10:02 = 15 minutes.
- **Decision reading:** If L_max = 20 minutes, 15 <= 20 and the evaluated freshness condition passes at 10:17 UTC.
- **Explicit validity condition:** Validity condition: t_v must exclude malformed, duplicate, future-dated, or quarantined records; otherwise one bad timestamp can create a false freshness pass.
- **Triage use:** Record the raw numerator, denominator or timestamps beside the result so a responder can test whether the apparent breach is observational rather than real. In this context, apply it to 4.1 Freshness lag and on-time test.
- **Counterexample:** A mathematically valid result cannot justify a release decision when the input population excludes the records whose safety or materiality is under review. In this context, apply it to 4.1 Freshness lag and on-time test.

### 4.2 Availability SLI and error budget
- **Purpose:** Availability measures the share of due evaluation opportunities that satisfy all required release conditions.
```text
FORMULA → A = G / D;  B = D - G
WHERE → A is availability as a unitless fraction, G is the count of good due partitions, D is the count of due partitions, and B is the count of failed due partitions.
CONSTRAINT → For an SLO target S over the same window, the permitted budget is B_max = D(1-S); a breach occurs when B > B_max.
NUMERICAL EXAMPLE → ILLUSTRATIVE: 286 of 288 hourly partitions were good, so A = 286/288 = 0.993055..., or 99.306%.
INTERPRETATION → For S = 0.99, B_max = 288(1-0.99) = 2.88 partitions; integer operational policy must state whether two or three failures are allowed.
```
- **Derivation note:** For an SLO target S over the same window, the permitted budget is B_max = D(1-S); a breach occurs when B > B_max.
- **Worked value:** ILLUSTRATIVE: 286 of 288 hourly partitions were good, so A = 286/288 = 0.993055..., or 99.306%.
- **Decision reading:** For S = 0.99, B_max = 288(1-0.99) = 2.88 partitions; integer operational policy must state whether two or three failures are allowed.
- **Explicit validity condition:** Validity condition: D must include every partition that was actually due, including weekends or planned exclusions only when the contract says so.
- **Triage use:** Record the raw numerator, denominator or timestamps beside the result so a responder can test whether the apparent breach is observational rather than real. In this context, apply it to 4.2 Availability SLI and error budget.
- **Counterexample:** A mathematically valid result cannot justify a release decision when the input population excludes the records whose safety or materiality is under review. In this context, apply it to 4.2 Availability SLI and error budget.

### 4.3 Null-rate test
- **Purpose:** Null rate detects missing values in a required or conditionally required field.
```text
FORMULA → r_null = n_null / n_eval
WHERE → r_null is the unitless null fraction, n_null is the number of null field values, and n_eval is the number of records for which the requirement applies.
CONSTRAINT → The numerator must be a subset of the denominator; if conditional applicability is ignored, optional records can dilute a required-field breach.
NUMERICAL EXAMPLE → ILLUSTRATIVE: 54 addressable records have a null settlement_currency among 18,000 evaluated records, so r_null = 54/18,000 = 0.003.
INTERPRETATION → Expressed as a percentage, 0.003 × 100 = 0.30%; compare 0.003 to the contract threshold, for example 0.001.
```
- **Derivation note:** The numerator must be a subset of the denominator; if conditional applicability is ignored, optional records can dilute a required-field breach.
- **Worked value:** ILLUSTRATIVE: 54 addressable records have a null settlement_currency among 18,000 evaluated records, so r_null = 54/18,000 = 0.003.
- **Decision reading:** Expressed as a percentage, 0.003 × 100 = 0.30%; compare 0.003 to the contract threshold, for example 0.001.
- **Explicit validity condition:** Validity condition: count explicit null, blank, sentinel, and parse-failure representations according to the field contract rather than treating them as interchangeable by default.
- **Triage use:** Record the raw numerator, denominator or timestamps beside the result so a responder can test whether the apparent breach is observational rather than real. In this context, apply it to 4.3 Null-rate test.
- **Counterexample:** A mathematically valid result cannot justify a release decision when the input population excludes the records whose safety or materiality is under review. In this context, apply it to 4.3 Null-rate test.

### 4.4 Volume deviation score
- **Purpose:** A volume detector compares an observed count with a context-matched historical baseline.
```text
FORMULA → z = (x - mu) / sigma
WHERE → x is the current count, mu is the baseline mean count, sigma is the baseline standard deviation in counts, and z is measured in standard deviations.
CONSTRAINT → Subtracting mu centers the observation; dividing by sigma scales ordinary variation to one unit so different-volume products can be compared.
NUMERICAL EXAMPLE → ILLUSTRATIVE: x = 82,000 records, mu = 100,000 records, and sigma = 6,000 records give z = (82,000 - 100,000)/6,000 = -3.0.
INTERPRETATION → A two-sided rule such as |z| >= 3 flags this observation, but the alert must retain the comparable weekday, season, and source-state assumptions.
```
- **Derivation note:** Subtracting mu centers the observation; dividing by sigma scales ordinary variation to one unit so different-volume products can be compared.
- **Worked value:** ILLUSTRATIVE: x = 82,000 records, mu = 100,000 records, and sigma = 6,000 records give z = (82,000 - 100,000)/6,000 = -3.0.
- **Decision reading:** A two-sided rule such as |z| >= 3 flags this observation, but the alert must retain the comparable weekday, season, and source-state assumptions.
- **Explicit validity condition:** Validity condition: the baseline must be stable enough for mean and standard deviation to summarize it; use quantiles or a count model for skewed, seasonal, or low-count data.
- **Triage use:** Record the raw numerator, denominator or timestamps beside the result so a responder can test whether the apparent breach is observational rather than real. In this context, apply it to 4.4 Volume deviation score.
- **Counterexample:** A mathematically valid result cannot justify a release decision when the input population excludes the records whose safety or materiality is under review. In this context, apply it to 4.4 Volume deviation score.

### 4.5 Detection precision and recall
- **Purpose:** Detector quality is separate from data quality: it describes whether alerting identifies material breaches.
```text
FORMULA → precision = TP/(TP+FP); recall = TP/(TP+FN)
WHERE → TP, FP, and FN are counts of true alerts, false alerts, and missed material incidents over the labelled evaluation period.
CONSTRAINT → Precision conditions on a page and estimates actionability; recall conditions on an incident and estimates detection coverage.
NUMERICAL EXAMPLE → ILLUSTRATIVE: 18 pages included 12 confirmed material breaches, while four other material breaches had no page. Precision = 12/18 = 0.667.
INTERPRETATION → Recall = 12/(12+4) = 0.75; tuning a threshold can improve precision while lowering recall, so choose against explicit harm and response capacity.
```
- **Derivation note:** Precision conditions on a page and estimates actionability; recall conditions on an incident and estimates detection coverage.
- **Worked value:** ILLUSTRATIVE: 18 pages included 12 confirmed material breaches, while four other material breaches had no page. Precision = 12/18 = 0.667.
- **Decision reading:** Recall = 12/(12+4) = 0.75; tuning a threshold can improve precision while lowering recall, so choose against explicit harm and response capacity.
- **Explicit validity condition:** Validity condition: labels require an independent closure review; counting acknowledgements as true positives biases the measure toward noisy detectors.
- **Triage use:** Record the raw numerator, denominator or timestamps beside the result so a responder can test whether the apparent breach is observational rather than real. In this context, apply it to 4.5 Detection precision and recall.
- **Counterexample:** A mathematically valid result cannot justify a release decision when the input population excludes the records whose safety or materiality is under review. In this context, apply it to 4.5 Detection precision and recall.

### 4.6 Blast-radius priority score
- **Purpose:** A triage score can order investigation work when several products are affected, but it does not replace safety or policy escalation.
```text
FORMULA → P = I × U × C × (1-R)
WHERE → I is estimated decision impact on a 0–1 scale, U is urgency on a 0–1 scale, C is consumer exposure on a 0–1 scale, and R is reversibility on a 0–1 scale.
CONSTRAINT → Multiplication makes a near-zero dimension reduce the priority, while (1-R) assigns greater urgency to effects that are hard to undo.
NUMERICAL EXAMPLE → ILLUSTRATIVE: I = 0.8, U = 0.9, C = 0.7, and R = 0.25 produce P = 0.8×0.9×0.7×0.75 = 0.378.
INTERPRETATION → A score of 0.378 is meaningful only relative to other products using the same rubric; it should be stored with its component judgments.
```
- **Derivation note:** Multiplication makes a near-zero dimension reduce the priority, while (1-R) assigns greater urgency to effects that are hard to undo.
- **Worked value:** ILLUSTRATIVE: I = 0.8, U = 0.9, C = 0.7, and R = 0.25 produce P = 0.8×0.9×0.7×0.75 = 0.378.
- **Decision reading:** A score of 0.378 is meaningful only relative to other products using the same rubric; it should be stored with its component judgments.
- **Explicit validity condition:** Validity condition: any safety, regulatory, security, or irreversible customer-harm trigger overrides this heuristic score and follows its mandated escalation path.
- **Triage use:** Record the raw numerator, denominator or timestamps beside the result so a responder can test whether the apparent breach is observational rather than real. In this context, apply it to 4.6 Blast-radius priority score.
- **Counterexample:** A mathematically valid result cannot justify a release decision when the input population excludes the records whose safety or materiality is under review. In this context, apply it to 4.6 Blast-radius priority score.

### 4.7 Mechanism invariant: safe publication gate
- **Preconditions:** A run manifest identifies immutable inputs, transformation version, contract version, quality results, and the output partition candidate.
- **State transition:** The gate moves a candidate from pending to published only after every blocking check passes or an authorised exception is attached.
- **Invariant and justification:** No ordinary consumer can read a quarantined candidate because the published pointer references only approved output identifiers.
- **Worked trace:** ILLUSTRATIVE run R-204 passes schema and freshness, fails required-key uniqueness, and remains quarantined while the prior published pointer remains unchanged.
- **Failure boundary:** If a consumer bypasses the published pointer and reads a physical storage prefix directly, the invariant does not protect that consumer; access design must remove the bypass.
- **Audit evidence:** Retain the state transition actor, timestamp, check results, exception identifier, and before-and-after pointer values.

## 5. Worked Numerical Example

The following incident is ILLUSTRATIVE. It demonstrates a payment-settlement fact table that is due every 15 minutes and feeds an operations dashboard plus a downstream reconciliation export.

### 5.1 Declared contract before the incident
- **Product:** settlement_event_fact, one row per accepted settlement event version.
- **Key:** settlement_event_id plus source_revision; the current view selects the highest accepted revision per event.
- **Event clock:** source_event_at identifies when the settlement event occurred.
- **Processing clock:** warehouse_loaded_at identifies when the row became available to the transformation.
- **Freshness objective:** At least 99% of due 15-minute partitions have valid data with event-time lag at most 30 minutes over 30 days.
- **Completeness objective:** Every source control-total record is represented or an approved exclusion is attached before the partition is released.
- **Validity objective:** settlement_currency is non-null for 99.9% or more of applicable accepted events in each released partition.
- **Uniqueness objective:** No duplicate current-view settlement_event_id may be published within a partition.
- **Containment:** Quarantine the affected partition, hold the reconciliation export, and serve the last verified dashboard snapshot with a stale label.
- **Recovery proof:** Reconcile source counts and amounts, rerun affected transforms, verify downstream export status, and record the exposure interval.

### 5.2 Observations and initial detector results
- **10:00 UTC:** Partition P10:00 is due; expected source control total is not yet final because the agreed watermark remains open.
- **10:15 UTC:** P10:00 receives 24,800 accepted source events; the historical context band for this interval is 24,000 to 27,000 events.
- **10:30 UTC:** P10:00 is released after schema, key, and amount checks; its latest valid event time is 10:26 UTC.
- **10:45 UTC:** P10:15 is due; ingestion reports a connection retry surge from one regional producer.
- **11:00 UTC:** P10:15 lands with 18,200 records while its comparable baseline mean is 25,000 and standard deviation is 2,000.
- **11:01 UTC:** The volume detector reports z = (18,200 - 25,000)/2,000 = -3.4, crossing the configured absolute-three control boundary.
- **11:02 UTC:** The null detector finds 91 null settlement_currency values among 18,200 applicable records.
- **11:03 UTC:** The freshness monitor sees newest valid event time 10:19 UTC at observation time 11:03 UTC.
- **11:04 UTC:** The publication gate quarantines P10:15 because both volume and required-field checks are blocking.
- **11:05 UTC:** The dashboard remains on P10:00 and displays a stale-data annotation; the reconciliation export is not emitted.
- **11:08 UTC:** The incident owner captures run manifests, source offset ranges, detector values, sample keys, and the consumer list.
- **11:12 UTC:** Lineage shows the missing records originate upstream of the regional normalisation transform, not in the final fact-table merge.

### 5.3 Arithmetic check one: freshness
```text
FORMULA → L_f = t_o - t_v
WHERE → t_o = 11:03 UTC and t_v = 10:19 UTC, both measured on the event-time freshness clock.
CONSTRAINT → The contract limits event-time lag to L_max = 30 minutes; future-dated or invalid events are excluded.
NUMERICAL EXAMPLE → L_f = 11:03 - 10:19 = 44 minutes.
INTERPRETATION → 44 minutes exceeds 30 minutes by 14 minutes, so freshness fails at the observation time.
```
- **Independent re-check:** From 10:19 to 11:00 is 41 minutes, then three additional minutes reach 11:03; therefore 41 + 3 = 44 minutes.
- **Sensitivity:** If the monitor had evaluated at 10:49 UTC with the same newest valid event, lag would be 30 minutes and would pass only under an inclusive `<= 30` policy.
- **Validity reminder:** The result says nothing about completeness; a current but partial partition can have a low freshness lag.

### 5.4 Arithmetic check two: null rate
```text
FORMULA → r_null = n_null / n_eval
WHERE → n_null = 91 applicable events with a null required currency and n_eval = 18,200 applicable accepted events.
CONSTRAINT → The partition must meet r_null <= 0.001, equivalent to a 0.10% maximum null rate.
NUMERICAL EXAMPLE → r_null = 91 / 18,200 = 0.005 exactly.
INTERPRETATION → 0.005 × 100 = 0.50%, which is five times the 0.10% maximum rate.
```
- **Independent re-check:** A 0.10% limit over 18,200 records permits at most 18.2 nulls; whole-record policy must choose 18 or 19, and observed 91 exceeds either choice.
- **Sensitivity:** If 73 records were reclassified as non-applicable, the denominator would become 18,127 and the rate would be 91/18,127 ≈ 0.502%, still a breach.
- **Validity reminder:** Verify whether null encodes source absence, mapping loss, an unrecognised currency code, or a legitimate conditional exception before selecting repair.

### 5.5 Arithmetic check three: volume deviation
```text
FORMULA → z = (x - mu) / sigma
WHERE → x = 18,200 records, mu = 25,000 records, and sigma = 2,000 records for comparable Wednesday 10:15 partitions.
CONSTRAINT → The reference excludes maintenance windows and uses the same source-region population; the detector flags |z| >= 3.
NUMERICAL EXAMPLE → z = (18,200 - 25,000)/2,000 = -6,800/2,000 = -3.4.
INTERPRETATION → |−3.4| = 3.4 exceeds 3, indicating an unusually low count relative to this baseline.
```
- **Independent re-check:** Expected minus observed equals 25,000 - 18,200 = 6,800 records; 6,800 divided by 2,000 standard-deviation units equals 3.4.
- **Sensitivity:** If the standard deviation were 2,500 because a campaign day was included, z would be -6,800/2,500 = -2.72 and would not cross this particular detector boundary.
- **Validity reminder:** A volume anomaly supports investigation, not proof of loss; a genuine demand change or changed inclusion rule can produce the same statistic.

### 5.6 Arithmetic check four: error-budget exposure
```text
FORMULA → A = G/D; B = D-G; B_max = D(1-S)
WHERE → D = 2,880 due 15-minute partitions in a 30-day window, S = 0.99, and two prior bad partitions are already recorded.
CONSTRAINT → Each quarantined or late-invalid partition counts as one failed due opportunity; approved maintenance exclusions are removed from D before calculation.
NUMERICAL EXAMPLE → B_max = 2,880(1-0.99) = 28.8; the operational ceiling is 28 whole failed partitions if fractional failures are not allowed.
INTERPRETATION → This incident would be the third failure, leaving 28 - 3 = 25 whole-partition failures of remaining allowance under the stated rounding policy.
```
- **Independent re-check:** 1% of 2,880 is 28.8; because a fraction of a 15-minute partition cannot fail, the contract must explicitly choose floor, ceiling, or duration-weighted accounting.
- **Sensitivity:** A 99.5% target would allow 2,880×0.005 = 14.4, or 14 whole failures under a floor policy, materially changing escalation urgency.
- **Validity reminder:** Error budget describes historical allowance, not permission to knowingly publish a currently harmful partition.

### 5.7 Triage trace and containment rationale
- **Bound the symptom:** The responder fixes the scope at P10:15, the regional producer path, and consumers of settlement_event_fact; no claim is made about earlier partitions without evidence.
- **Validate clocks:** The responder confirms the 44-minute value uses event time and that ingestion retries do not merely delay processing-time telemetry.
- **Test source arrival:** Source offset and control-total evidence shows fewer events arriving from one region, making a final-table-only defect less likely.
- **Test transformation:** The normalisation transform log identifies an unrecognised currency mapping for a newly emitted source code, explaining the null symptom for received records.
- **Separate mechanisms:** The missing count and null expansion share a producer deployment window but may be distinct defects; the runbook does not collapse them into one cause prematurely.
- **Contain safely:** Quarantine preserves P10:15 for inspection, while a labelled P10:00 snapshot avoids mixing current and partial figures in the dashboard.
- **Repair source path:** Restore the producer connection and verify offsets cover the missing interval before replaying; otherwise a replay would faithfully reproduce incompleteness.
- **Repair mapping:** Add the approved mapping under a reviewed contract version, then test the mapping on retained examples including the new code.
- **Replay bounded range:** Reprocess P10:15 from immutable raw input using recorded code and configuration versions; do not overwrite the quarantined artifact before comparison.
- **Reconcile output:** Compare event count, amount total, duplicate keys, null rate, and source control total for the corrected partition.
- **Release deliberately:** Publish only after the gate records passing evidence and downstream export eligibility is re-evaluated.
- **Close exposure:** Identify whether dashboard viewers or manual operators acted during 10:45–release; reconcile any decisions that used the stale labelled snapshot or withheld export.

### 5.8 Closure evidence
- **Input identity:** The replay input manifest lists raw object checksums, source offsets, and the precise P10:15 event-time boundary.
- **Transformation identity:** The repair records code revision, mapping-table revision, configuration hash, and execution parameters.
- **Quality identity:** The release record includes schema result, 0 duplicate current-view keys, null numerator and denominator, and amount reconciliation result.
- **Freshness identity:** The release record states observation time, latest valid event time, computed lag, and whether the SLO window was updated.
- **Consumer identity:** The dashboard owner confirms snapshot labelling removal only after release; export status records whether a delayed file replaced a suppressed file.
- **Incident identity:** The closure names the distinct initiating defects, contributing monitoring gaps, containment time, recovery time, and preventive tests.

### 5.9 Decision checkpoints during the example
- **Checkpoint — Is the page actionable?**
  - **Answer:** Yes: the event-time lag, low count, and null-rate breach each correspond to a release decision and named owners.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Is the page actionable?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Is the page actionable?.
- **Checkpoint — Can the output be safely published with a warning?**
  - **Answer:** No: the contract designates completeness and required currency as blocking because reconciliation consumers need full, interpretable settlement events.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can the output be safely published with a warning?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can the output be safely published with a warning?.
- **Checkpoint — Can a prior snapshot be served?**
  - **Answer:** Yes, only with its as-of time displayed and only to consumers whose declared degraded mode permits a dated view.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can a prior snapshot be served?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can a prior snapshot be served?.
- **Checkpoint — Can the export be regenerated later?**
  - **Answer:** Yes, if downstream recipients receive a replacement protocol and the export identifier prevents double processing.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can the export be regenerated later?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can the export be regenerated later?.
- **Checkpoint — Does the volume anomaly prove data loss?**
  - **Answer:** No: it proves an unusual count; source offsets and control totals are needed to distinguish delayed arrival from a changed population.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Does the volume anomaly prove data loss?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Does the volume anomaly prove data loss?.
- **Checkpoint — Does the null-rate breach prove a mapping defect?**
  - **Answer:** No: it proves absent required values; sample payloads, mapping revision, and source code frequency discriminate producer absence from transform loss.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Does the null-rate breach prove a mapping defect?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Does the null-rate breach prove a mapping defect?.
- **Checkpoint — Should an error budget be consumed during quarantine?**
  - **Answer:** Under the declared SLO, yes: a due partition that is not valid and released on time is a failed opportunity even when quarantine prevents harm.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Should an error budget be consumed during quarantine?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Should an error budget be consumed during quarantine?.
- **Checkpoint — Can a replay use current source state?**
  - **Answer:** Not safely without recording it: late source corrections or mutable lookup tables can make the replay non-reproducible.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can a replay use current source state?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: Can a replay use current source state?.
- **Checkpoint — What makes the final count credible?**
  - **Answer:** Agreement between replayed source offsets, source control total, and deduplicated fact-table count under the documented key rule.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: What makes the final count credible?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: What makes the final count credible?.
- **Checkpoint — What makes the amount total credible?**
  - **Answer:** A currency-aware reconciliation that separates true zero amount, missing currency, conversion policy, and any approved rounding convention.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: What makes the amount total credible?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: What makes the amount total credible?.
- **Checkpoint — When may the dashboard annotation be removed?**
  - **Answer:** After the corrected partition is published, the visible aggregate is recomputed, and the dashboard owner verifies the displayed as-of time.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: When may the dashboard annotation be removed?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: When may the dashboard annotation be removed?.
- **Checkpoint — When is the incident not yet closed?**
  - **Answer:** When code is repaired but affected consumer actions, delivered files, model features, or audit records have not been reconciled.
  - **Evidence to retain:** Link the decision to the incident timeline, the contract clause, and the exact run or query result used at that moment. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: When is the incident not yet closed?.
  - **Unsafe shortcut:** Do not substitute a verbal assurance for a recorded comparison when the release or containment decision has downstream consequences. In this context, apply it to 5.9 Decision checkpoints during the example. Specific checkpoint: When is the incident not yet closed?.

## 6. Visual / Geometric Intuition

Think of each monitor as projecting a high-dimensional data product onto a small number of views. No single projection proves health; complementary projections shrink the plausible failure region.

### Freshness as a horizontal distance
- **Picture:** Draw a time axis with the newest valid event at one point and observation time at another. The gap is freshness lag; moving the monitor later increases the gap even when the data does not change.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Freshness as a horizontal distance.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Freshness as a horizontal distance.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Freshness as a horizontal distance.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Freshness as a horizontal distance.
### Watermark as a moving frontier
- **Picture:** Place received events to the left of a watermark and possible late events to the right. Advancing the watermark improves timeliness but increases the risk that later corrections require reprocessing.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Watermark as a moving frontier.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Watermark as a moving frontier.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Watermark as a moving frontier.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Watermark as a moving frontier.
### Completeness as coverage
- **Picture:** Imagine expected source keys as a reference set and published keys as a second set. Missing keys are the part of the reference set outside the published set, while unexpected keys deserve a separate investigation.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Completeness as coverage.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Completeness as coverage.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Completeness as coverage.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Completeness as coverage.
### Uniqueness as overlap
- **Picture:** Represent records with the same business key as points stacked at one coordinate. A duplicate detector finds excess occupancy; it does not decide which record is authoritative without precedence rules.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Uniqueness as overlap.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Uniqueness as overlap.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Uniqueness as overlap.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Uniqueness as overlap.
### Schema as a coordinate system
- **Picture:** A schema change can rename, remove, or reinterpret an axis. Values may still parse while downstream calculations point along the wrong semantic direction.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Schema as a coordinate system.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Schema as a coordinate system.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Schema as a coordinate system.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Schema as a coordinate system.
### Distribution drift as shifted mass
- **Picture:** A histogram whose mass moves right may reflect a genuine price change, a unit conversion, a population shift, or a parser defect. The picture identifies movement, not cause.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Distribution drift as shifted mass.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Distribution drift as shifted mass.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Distribution drift as shifted mass.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Distribution drift as shifted mass.
### Control limits as bands
- **Picture:** A baseline curve with upper and lower bands distinguishes ordinary variation from unusual observations. Wide bands reduce pages but can miss harmful changes; narrow bands do the reverse.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Control limits as bands.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Control limits as bands.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Control limits as bands.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Control limits as bands.
### Lineage as a directed graph
- **Picture:** Nodes represent artifacts or runs and edges represent actual dependencies. A path shows potential propagation, while observed reads and partition overlap determine realised impact.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Lineage as a directed graph.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Lineage as a directed graph.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Lineage as a directed graph.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Lineage as a directed graph.
### Blast radius as a cone
- **Picture:** From a defective partition, expand downstream through consumers, but intersect each branch with the time and partition it actually read. The cone narrows when a consumer uses a fixed snapshot.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Blast radius as a cone.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Blast radius as a cone.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Blast radius as a cone.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Blast radius as a cone.
### Quarantine as a side lane
- **Picture:** A gate sends suspect output to an inspectable lane rather than the published lane. The lane is useful only if consumers cannot silently route around it.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Quarantine as a side lane.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Quarantine as a side lane.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Quarantine as a side lane.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Quarantine as a side lane.
### Error budget as a finite area
- **Picture:** Plot failures over an SLO window; each failed partition or failed minute consumes part of an allowed area. A quiet recent period cannot erase an active harmful breach.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Error budget as a finite area.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Error budget as a finite area.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Error budget as a finite area.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Error budget as a finite area.
### Triage as hypothesis pruning
- **Picture:** Start with source, transport, transform, publish, and consumer hypotheses. Each discriminating check removes branches, making the next check cheaper and more informative.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Triage as hypothesis pruning.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Triage as hypothesis pruning.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Triage as hypothesis pruning.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Triage as hypothesis pruning.
### Reconciliation as a join
- **Picture:** Place authoritative records and repaired records in two columns keyed by identity and time. Matched rows support recovery; left-only, right-only, and unequal measures expose unresolved differences.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Reconciliation as a join.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Reconciliation as a join.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Reconciliation as a join.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Reconciliation as a join.
### Cost as a frontier
- **Picture:** Plot detector cost against expected harm avoided. The useful operating region is not necessarily the most sensitive point because human response capacity and false-page cost are real constraints.
- **What the picture preserves:** It keeps the relevant boundary, direction, or comparison population visible while a responder reasons under time pressure. In this context, apply it to Cost as a frontier.
- **What the picture hides:** It does not encode ownership, business materiality, or semantic intent unless those labels are added explicitly. In this context, apply it to Cost as a frontier.
- **Triage application:** Use the picture to choose the next discriminating observation, then record the measured result rather than relying on the sketch. In this context, apply it to Cost as a frontier.
- **Misread to avoid:** A visually large anomaly is not automatically high severity; exposure and reversibility can be more important than magnitude. In this context, apply it to Cost as a frontier.

### Coordinate choices that prevent false conclusions
- **Event time versus processing time:** Plot both when retries or backlog exist; a healthy processing-time curve can hide an event-time freshness breach.
- **Absolute count versus rate:** Use rates when denominator movement is expected; use absolute counts when a small percentage can still represent many material records.
- **Global aggregate versus segment:** Slice by region, producer version, tenant, source type, or schema version when cancellation in the global aggregate can hide opposing defects.
- **Current partition versus historical window:** Inspect the current partition for containment but compare it with a seasonally and operationally comparable reference for diagnosis.
- **Potential lineage versus executed lineage:** Use static design lineage for preparedness and run-level lineage for incident claims about what actually consumed the defective artifact.
- **Detector event versus business impact:** Plot alert time and consumer-read time separately; early detection is valuable only if it precedes material use or enables reconciliation.
- **Raw value versus canonical value:** Compare both when normalisation, currency conversion, tokenisation, or unit conversion could be the failure domain.
- **Known-good snapshot versus current output:** Show as-of timestamps beside both; a fallback is safe only when readers can see which state they are receiving.

## 7. Comparison / Selection Table

| Monitoring approach | Best signal | Strength | Weakness | Select when | Avoid when |
|---|---|---|---|---|---|
| Freshness threshold | latest valid timestamp | Directly maps to lateness objective | Cannot prove record completeness | Consumers require a recent partition | Time clock or late-arrival policy is unspecified |
| Volume z-score | record count | Flags unusual deviation at comparable scale | Assumes a useful baseline shape | Stable context has enough history | Seasonality, campaigns, or regime shifts dominate |
| Control-total reconciliation | count or amount against source | Tests end-to-end coverage with an external anchor | Requires trusted source total | An authoritative control record exists | Control total shares the same failure domain |
| Schema fingerprint | field/type/nullability signature | Detects interface changes before semantic use | May miss meaning changes under same type | Producer contracts evolve frequently | Consumers need semantic validation only |
| Row-level validity rule | predicate failures | Localises invalid values with samples | Can be costly on high-volume streams | A field domain is explicit | Business rule is ambiguous or context-dependent |
| Uniqueness check | duplicate key rate | Protects grain assumptions | Needs deterministic key and precedence | Double counting would harm decisions | Natural keys are intentionally reused without time scope |
| Referential-integrity test | unmatched child keys | Finds broken joins and late dimensions | Can page during expected arrival skew | Parent-child timing is contracted | Eventual consistency has no stated tolerance |
| Distribution distance | histogram or quantile shift | Detects subtler shape changes | Does not identify cause | Feature population is stable enough to compare | A known seasonal or policy shift is unlabelled |
| Rule-based semantic invariant | cross-field relation | Tests business logic beyond types | Needs maintained domain knowledge | The relation is stable and material | The rule changes faster than version control |
| Lineage impact query | executed downstream reads | Targets containment to real consumers | Depends on complete run metadata | Many downstream products exist | Lineage is static, partial, or stale |
| Synthetic canary record | known test identity | Provides an end-to-end trace | Can diverge from real traffic patterns | Pipeline supports isolated test data | Canary would contaminate regulated or aggregate outputs |
| Consumer-side reconciliation | reported total versus product total | Detects post-publication transformation loss | May detect too late for prevention | Consumer applies material transforms | Consumer has no authoritative comparison |
| Change-point detector | sustained metric shift | Adapts to non-stationary baselines | Harder to explain and tune | Gradual or unknown shifts matter | Simple contract limits are sufficient |
| Sampling audit | manually reviewed records | Examines meaning and exceptions | Low coverage and human cost | Semantic correctness dominates | Fast automated containment is required |

### Selection rules
- **Start from harm:** Choose a detector because it prevents a named incorrect decision, not because the metric is convenient to collect.
- **Pair leading and lagging evidence:** Use upstream ingestion signals for early warning and consumer reconciliation for proof that no loss escaped the publication boundary.
- **Use an independent anchor where possible:** A source control total or independently emitted checksum is stronger than two checks calculated from the same defective transformation.
- **Align response time:** A detector that needs a human within five minutes should have a clear owner, low noise, and an action that can be taken in five minutes.
- **Budget attention explicitly:** High-sensitivity monitors should route to a queue or ticket when their expected urgency does not justify interrupting an on-call responder.
- **Version detector assumptions:** Baseline population, threshold, key rule, and watermark are part of the detector contract and need change review.
- **Test the safe failure path:** Inject or replay known breaches to verify alert routing, quarantine, fallback labels, and reconciliation without waiting for an incident.
- **Retire misleading monitors:** A detector repeatedly acknowledged without action is a signal-design defect; replace it with a decision-linked control rather than merely widening a threshold.

## 8. Failure Modes & Edge Cases

### FM-01 Future timestamp masks lateness
- **Symptom:** A producer clock sends an event dated hours ahead, making latest-event freshness look excellent.
- **Competing causes:** Clock skew, timezone parsing, or a deliberately invalid source value are competing causes.
- **Discriminating check:** Compare event time with ingestion time and reject values beyond the contract’s future-skew tolerance.
- **Correction:** Quarantine the offending records from freshness aggregation and correct the producer clock or parser.
- **Prevention:** Add a bounded-future-time validity rule and monitor rejected future-dated counts.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-01 Future timestamp masks lateness.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-01 Future timestamp masks lateness.

### FM-02 Duplicate retry inflates volume
- **Symptom:** A retry re-emits accepted records after an acknowledgement timeout, causing count growth without new business events.
- **Competing causes:** Producer retry semantics, consumer offset reset, or a changed key extraction are competing causes.
- **Discriminating check:** Compare duplicate-key rate and message identifiers across the retry window; count distinct keys as well as rows.
- **Correction:** Apply idempotent merge or deduplication using documented precedence, then replay only the affected keys.
- **Prevention:** Require stable event identifiers and test retry behavior with deliberate acknowledgement loss.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-02 Duplicate retry inflates volume.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-02 Duplicate retry inflates volume.

### FM-03 Late arrival appears as loss
- **Symptom:** Events arrive after a partition is declared complete, making a count check fail at first observation.
- **Competing causes:** Network delay, source batching, or an incorrect watermark are competing causes.
- **Discriminating check:** Inspect arrival-time histogram after the event window and compare with the agreed lateness distribution.
- **Correction:** Hold release until watermark or publish with declared incompleteness and schedule a bounded correction.
- **Prevention:** Set watermark from measured late-arrival policy and record correction expectations for consumers.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-03 Late arrival appears as loss.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-03 Late arrival appears as loss.

### FM-04 Semantic change passes schema
- **Symptom:** A field remains a string but switches from gross amount to net amount, so parsing succeeds while decisions change.
- **Competing causes:** Producer contract change, undocumented feature flag, or downstream interpretation drift are competing causes.
- **Discriminating check:** Compare semantic version, distribution, cross-field invariants, and producer release notes.
- **Correction:** Quarantine new-version records or branch processing until consumers approve the changed meaning.
- **Prevention:** Require contract-version review for meaning and unit changes, not only type changes.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-04 Semantic change passes schema.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-04 Semantic change passes schema.

### FM-05 Global aggregate cancels segment errors
- **Symptom:** One region drops records while another duplicates them, leaving the global total near normal.
- **Competing causes:** Regional outage, shard rebalance, or segment-label mapping error are competing causes.
- **Discriminating check:** Compute volume and validity checks by region, source version, and other material strata.
- **Correction:** Contain only impacted segments when isolation is reliable; otherwise hold the aggregate product.
- **Prevention:** Choose contract slices that match independent failure domains and consumer materiality.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-05 Global aggregate cancels segment errors.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-05 Global aggregate cancels segment errors.

### FM-06 Baseline absorbs recurring defect
- **Symptom:** A rolling baseline learns a persistent bad level and stops alerting after several periods.
- **Competing causes:** True demand shift, persistent source fault, or baseline window contamination are competing causes.
- **Discriminating check:** Compare the adaptive baseline with a fixed contractual control total and inspect baseline updates around incident start.
- **Correction:** Freeze or reset the contaminated baseline after verified repair rather than accepting the new level as normal.
- **Prevention:** Protect baselines with incident labels and use independent hard limits for critical metrics.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-06 Baseline absorbs recurring defect.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-06 Baseline absorbs recurring defect.

### FM-07 Small denominator exaggerates rate
- **Symptom:** One invalid record among two records yields a 50% rate that may page despite negligible impact.
- **Competing causes:** Low traffic, partition-key error, or genuine critical-record failure are competing causes.
- **Discriminating check:** Inspect numerator and denominator together, plus identity and materiality of the invalid record.
- **Correction:** Use a combined absolute-count and rate rule or route low-volume anomalies to review.
- **Prevention:** Declare minimum denominator policy and escalation override for individually critical records.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-07 Small denominator exaggerates rate.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-07 Small denominator exaggerates rate.

### FM-08 Large denominator hides harm
- **Symptom:** A 0.01% invalid rate may represent thousands of financially material records.
- **Competing causes:** High-volume legitimate nulls, a concentrated customer segment, or systemic mapping loss are competing causes.
- **Discriminating check:** Calculate absolute count, monetary exposure, and segment concentration alongside the fraction.
- **Correction:** Quarantine or prioritise affected segments based on materiality rather than percentage alone.
- **Prevention:** Set both rate and absolute-impact limits for high-volume products.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-08 Large denominator hides harm.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-08 Large denominator hides harm.

### FM-09 Backfill overwrites newer correction
- **Symptom:** A historical replay writes an older source version over a corrected current record.
- **Competing causes:** Non-idempotent merge, missing version fence, or wrong partition predicate are competing causes.
- **Discriminating check:** Compare source revision and update timestamp before and after a dry-run merge.
- **Correction:** Use version-aware upsert with a monotonic revision rule and validate on a shadow target first.
- **Prevention:** Require backfill manifests, write fences, and rollback snapshots for mutable products.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-09 Backfill overwrites newer correction.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-09 Backfill overwrites newer correction.

### FM-10 Alert storm hides primary incident
- **Symptom:** Many downstream monitors page after one upstream source outage, overwhelming responders.
- **Competing causes:** True independent failures, lineage fan-out, or a monitoring configuration loop are competing causes.
- **Discriminating check:** Trace page timestamps and lineage to identify a shared earliest failing dependency.
- **Correction:** Group derivative alerts under an incident while keeping evidence for impacted consumers.
- **Prevention:** Model dependency-aware suppression that preserves the root signal and safety-critical exceptions.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-10 Alert storm hides primary incident.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-10 Alert storm hides primary incident.

### FM-11 Dashboard cache shows stale current label
- **Symptom:** A dashboard displays cached values but labels them as current after data publication stops.
- **Competing causes:** Cache invalidation failure, missing as-of field, or delayed dashboard query are competing causes.
- **Discriminating check:** Compare displayed as-of timestamp with dataset release manifest and browser/API cache metadata.
- **Correction:** Show the last verified as-of time and stale state; prevent automatic label refresh without new data.
- **Prevention:** Make freshness metadata a first-class field propagated to the presentation layer.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-11 Dashboard cache shows stale current label.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-11 Dashboard cache shows stale current label.

### FM-12 Lineage misses dynamic read
- **Symptom:** A transformation chooses an input table from runtime configuration, leaving static lineage incomplete.
- **Competing causes:** Dynamic SQL, configuration drift, or uninstrumented external read are competing causes.
- **Discriminating check:** Inspect executed query plans, runtime parameters, and storage access logs for actual dependencies.
- **Correction:** Expand containment to plausible hidden consumers until executed lineage is captured.
- **Prevention:** Emit run-level input and output identifiers from orchestration and query layers.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-12 Lineage misses dynamic read.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-12 Lineage misses dynamic read.

### FM-13 Check uses same corrupted source
- **Symptom:** Both numerator and denominator derive from the same truncated stream, so reconciliation passes falsely.
- **Competing causes:** Shared ingestion fault, common transformation, or incorrect control-total join are competing causes.
- **Discriminating check:** Establish whether the anchor is independently produced before the suspected failure boundary.
- **Correction:** Escalate to a source-side audit or independent ledger rather than accepting self-referential agreement.
- **Prevention:** Document independence assumptions for each control and test shared-failure scenarios.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-13 Check uses same corrupted source.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-13 Check uses same corrupted source.

### FM-14 Quarantine path is bypassed
- **Symptom:** A direct storage reader continues consuming a quarantined partition outside the publication gate.
- **Competing causes:** Legacy integration, broad permissions, or undocumented ad hoc query are competing causes.
- **Discriminating check:** Audit read logs and permissions for physical paths versus approved product views.
- **Correction:** Revoke bypass access or redirect consumers to a stable published view before declaring containment.
- **Prevention:** Use access controls and product interfaces that make the safe path the easiest path.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-14 Quarantine path is bypassed.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-14 Quarantine path is bypassed.

### FM-15 Unit conversion shifts values
- **Symptom:** A producer changes milliseconds to seconds or cents to currency units, creating plausible but wrong numbers.
- **Competing causes:** Unit change, parser regression, or source documentation error are competing causes.
- **Discriminating check:** Check dimensional invariants, quantile ratios, and cross-field relationships against source version.
- **Correction:** Quarantine the new unit version and replay after an explicit conversion rule is approved.
- **Prevention:** Store units in contracts and test boundary values that distinguish common scale factors.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-15 Unit conversion shifts values.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-15 Unit conversion shifts values.

### FM-16 PII redaction removes join key
- **Symptom:** A privacy transform hashes or drops an identifier needed by a downstream integrity check.
- **Competing causes:** Intended policy change, redaction bug, or consumer dependency violation are competing causes.
- **Discriminating check:** Compare approved privacy contract, join success rate, and transformation version; do not expose raw PII to diagnose.
- **Correction:** Use an approved token or secure reconciliation service rather than restoring unrestricted identifiers.
- **Prevention:** Review downstream key requirements during privacy-transform change control.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-16 PII redaction removes join key.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-16 PII redaction removes join key.

### FM-17 Partial schema rollout splits consumers
- **Symptom:** Some producers emit a new optional field while old consumers reject or ignore it inconsistently.
- **Competing causes:** Staged rollout, incompatible parser, or undocumented default semantics are competing causes.
- **Discriminating check:** Segment by producer and consumer version; inspect schema fingerprints and rejection logs.
- **Correction:** Use backward-compatible dual-read or hold the new producer version until compatibility is verified.
- **Prevention:** Adopt explicit compatibility tests and expiry dates for transitional contract versions.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-17 Partial schema rollout splits consumers.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-17 Partial schema rollout splits consumers.

### FM-18 Monitor outage creates false confidence
- **Symptom:** The monitor stops evaluating but the dashboard remains green because last status is retained.
- **Competing causes:** Scheduler failure, credential expiry, or metrics pipeline outage are competing causes.
- **Discriminating check:** Check monitor heartbeat, evaluation timestamp, and freshness of the status artifact itself.
- **Correction:** Mark health unknown and page the monitoring owner when observation exceeds its own freshness objective.
- **Prevention:** Instrument monitors with independent heartbeat and stale-status detection.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-18 Monitor outage creates false confidence.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-18 Monitor outage creates false confidence.

### FM-19 Reconciliation total matches but identities differ
- **Symptom:** Counts and amounts agree while individual records are swapped or duplicated and offsetting.
- **Competing causes:** Key collision, wrong join, or compensating duplicate/delete are competing causes.
- **Discriminating check:** Perform key-level set difference and sampled attribute comparison, not aggregate-only reconciliation.
- **Correction:** Repair using identity-aware merge and re-evaluate affected downstream actions.
- **Prevention:** Use layered reconciliation: totals, key sets, and critical attributes.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-19 Reconciliation total matches but identities differ.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-19 Reconciliation total matches but identities differ.

### FM-20 Suppressed alert becomes permanent blind spot
- **Symptom:** A temporary mute remains after maintenance and masks later material breaches.
- **Competing causes:** Forgotten expiry, broad suppression scope, or misrouted schedule are competing causes.
- **Discriminating check:** Inspect active suppression records, expiry, matcher scope, and test alert delivery with a safe synthetic event.
- **Correction:** Remove or narrow the mute, then evaluate whether the incident window needs retrospective review.
- **Prevention:** Require expiry, owner, reason, and periodic audit for every suppression.
- **Escalation boundary:** Escalate immediately when the affected decision is safety-critical, legally controlled, irreversible, or cannot be bounded from available evidence. In this context, apply it to FM-20 Suppressed alert becomes permanent blind spot.
- **Evidence to preserve:** Keep the failing values or hashes, time window, contract version, run manifest, and identities of products that could have consumed the output. In this context, apply it to FM-20 Suppressed alert becomes permanent blind spot.

## 9. Complexity & Cost Analysis

Cost analysis is ILLUSTRATIVE and uses symbolic variables unless a numerical teaching example is marked. Measure cost across compute, storage, network, human attention, and false-decision exposure; cheap metrics can be expensive when they trigger poor response.

### 9.1 Computational cost by monitor
- **Freshness maximum:** Maintain the maximum valid timestamp per partition or stream key. Streaming state is O(k) for k active keys; each accepted event updates one comparison in O(1) expected time.
- **Row-count volume:** Increment counters per partition and segment. Storage is O(k) counters, while a batch count scan is O(n) over n rows unless metadata provides a trustworthy count.
- **Null-rate validity:** Maintain null and evaluated counters for each field/segment pair. Streaming update is O(f) per row for f checked fields; batch validation is O(nf).
- **Duplicate-key detection:** Exact detection requires a set or sort keyed by identity. Hash-state memory is O(u) for u unique keys; sort-based batch detection is O(n log n) time and may spill.
- **Referential integrity:** A hash join costs O(n+m) expected time and O(min(n,m)) memory for parent and child populations; indexed lookups change practical constants but not semantic requirements.
- **Schema fingerprint:** Canonicalise field metadata and hash it. Cost is O(f) for f fields, usually negligible compared with payload scanning, but semantic review is human-cost dominated.
- **Quantile drift:** Approximate sketches use bounded memory per segment and merge efficiently; exact quantiles generally require O(n log n) sorting or equivalent order-statistics work.
- **Distribution distance:** Histogram comparison costs O(b) after O(n) assignment to b bins. Bin definitions must be versioned because changing them changes the detector meaning.
- **Control-total reconciliation:** Aggregate source and target by matching partition and key. Cost is O(n+m) for scans plus join or aggregation overhead; an independent source can add transfer cost.
- **Lineage traversal:** A breadth-first impact search is O(V+E) across V nodes and E edges, but run-level partition filtering can greatly reduce the graph explored.
- **Sample audit:** Selecting s random or stratified records is O(s) after an index or reservoir is maintained, while human review time often dominates technical cost.
- **Canary trace:** One synthetic record per route is computationally small, but isolation, cleanup, and non-contamination controls add operational design cost.

### 9.2 Storage, retention, and replay trade-offs
- **Raw retention:** Retaining immutable raw inputs increases storage roughly with arrival volume times retention duration, but it enables evidence-preserving replay and reduces dependence on mutable sources.
- **Manifest retention:** Run manifests are small relative to data but high value; store input identities, output identities, versions, parameters, checks, and state transitions long enough to audit recovery.
- **Metric retention:** High-cardinality labels can make monitoring storage grow faster than source volume; retain actionable dimensions and aggregate or sample exploratory dimensions.
- **Sample retention:** Keep privacy-approved failing examples or hashes with enough context to reproduce a check, but minimise sensitive payload retention under applicable policy.
- **Snapshot retention:** Known-good snapshots consume storage and governance effort; their value is proportional to how many consumers have declared a safe degraded mode.
- **Replay retention:** To replay deterministically, retain not only inputs but lookup versions, code, configuration, random seeds when applicable, and reference-data snapshots.
- **Partition granularity:** Finer partitions reduce blast radius and replay scope but increase metadata, scheduling, small-file, and alert-management overhead.
- **Compression choice:** Compression lowers storage and network cost but can raise query latency and make partial inspection harder; choose per recovery objective rather than by a blanket rule.

### 9.3 Human-response economics
- **Page cost:** A page consumes attention, context-switch time, and sleep or coverage capacity; its cost grows when responders cannot take a safe action from the alert evidence.
- **False-negative cost:** A missed breach can create incorrect decisions, remediation work, contractual breach, customer harm, or regulatory exposure; quantify the decision pathway rather than assuming every miss is equal.
- **Triage cost:** Broad or unpartitioned signals force expensive searches. Segment-level evidence and run manifests reduce time spent asking which data, version, and consumer are involved.
- **Containment cost:** A conservative quarantine may delay value, while permissive publication may create reconciliation debt. Compare both against the reversibility and materiality of the downstream action.
- **Recovery cost:** Replay compute, source load, backfill coordination, and consumer reconciliation can exceed the original processing cost; design idempotence and partitions to bound this cost.
- **Maintenance cost:** Every monitor needs ownership, threshold review, dependency updates, and incident learning. A broad set of unowned checks becomes a hidden reliability liability.
- **Opportunity cost:** A stale but clearly labelled snapshot may permit safe work to continue; a silent stale result can create larger downstream opportunity and correction costs.

### 9.4 ILLUSTRATIVE cost calculation
```text
FORMULA → C_total = C_compute + C_storage + C_network + C_human + C_incident
WHERE → Monthly monitoring cost components are expressed in USD for a constructed example; C_incident is expected monthly remediation cost, not a guaranteed loss.
CONSTRAINT → Add only costs measured over the same month and avoid treating avoided harm as realised savings without evidence.
NUMERICAL EXAMPLE → C_compute = $420, C_storage = $180, C_network = $60, C_human = $900, and C_incident = $640.
INTERPRETATION → C_total = 420 + 180 + 60 + 900 + 640 = $2,200 per illustrative month.
```
- **Independent re-check:** $420 + $180 = $600; $600 + $60 = $660; $660 + $900 = $1,560; $1,560 + $640 = $2,200.
- **Sensitivity:** If better alert routing reduces human attention by 30%, the human component becomes $900×0.70 = $630 and total becomes $1,930, holding every other constructed input fixed.
- **Validity condition:** Do not use this arithmetic to justify a production budget without measured workload, cloud pricing, labour policy, and incident-frequency evidence.

### 9.5 Cost-aware design choices
- **Aggregate before escalating:** Compute cheap coarse signals broadly, then use targeted expensive checks only when a coarse signal or material consumer warrants them.
- **Use sketches for exploratory drift:** Approximate quantiles or cardinality sketches can control memory when exact values are unnecessary for containment, but validate approximation error for thresholds.
- **Shard state by meaningful keys:** Avoid one global state store when regions or tenants fail independently; do not over-shard into labels that nobody can triage.
- **Retain replay-critical artifacts:** Spend storage on inputs and manifests that convert an uncertain repair into a reproducible one; delete duplicated diagnostic copies without recovery value.
- **Make pages information-rich:** Including partition, clock, baseline, owner, run identifier, and next check can cost little telemetry but save substantial human time.
- **Prefer deterministic repairs:** Idempotent operations reduce the cost of safe retry, testing, and rollback, especially under partial failure.
- **Measure monitor health:** A cheap heartbeat prevents the costly false confidence created by silently stale monitoring state.
- **Review cardinality budgets:** Every new metric dimension should name the question it answers and the retention cost it adds.

## 10. Key Theorems / Results Reference

### Result 1 — A detector is not a diagnosis
- **Statement:** A threshold crossing establishes a discrepancy between observation and expectation. It does not identify the cause because multiple mechanisms can produce the same signal.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 1 — A detector is not a diagnosis.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 1 — A detector is not a diagnosis.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 1 — A detector is not a diagnosis.

### Result 2 — Correlated checks do not create independence
- **Statement:** Two monitors computed from the same truncated input can agree while both are wrong. Independent anchors must cross the suspected failure boundary.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 2 — Correlated checks do not create independence.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 2 — Correlated checks do not create independence.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 2 — Correlated checks do not create independence.

### Result 3 — Precision-recall trade-off
- **Statement:** For a fixed incident-labeling process, tightening an alert often reduces false pages and may increase missed incidents. Tune against harm, response capacity, and stated SLO needs.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 3 — Precision-recall trade-off.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 3 — Precision-recall trade-off.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 3 — Precision-recall trade-off.

### Result 4 — Aggregation can hide cancellation
- **Statement:** An aggregate total can remain normal while segment defects offset. Slicing must follow plausible failure domains and material consumer segments.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 4 — Aggregation can hide cancellation.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 4 — Aggregation can hide cancellation.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 4 — Aggregation can hide cancellation.

### Result 5 — Watermarks are policy statements
- **Statement:** A watermark determines when to treat an event-time window as complete enough to release. It encodes a latency-versus-correction trade-off rather than a fact about the world.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 5 — Watermarks are policy statements.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 5 — Watermarks are policy statements.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 5 — Watermarks are policy statements.

### Result 6 — Idempotence bounds retry harm
- **Statement:** If an operation keyed by a stable identity has the same result when repeated, retries need not create duplicate effects; idempotence does not itself solve missing input.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 6 — Idempotence bounds retry harm.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 6 — Idempotence bounds retry harm.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 6 — Idempotence bounds retry harm.

### Result 7 — Error budgets need an accounting rule
- **Statement:** A percentage target becomes actionable only after defining due opportunities, exclusions, rounding, and whether partial duration failures consume proportional budget.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 7 — Error budgets need an accounting rule.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 7 — Error budgets need an accounting rule.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 7 — Error budgets need an accounting rule.

### Result 8 — Blast radius is conditional
- **Statement:** A lineage path establishes possible propagation. Actual impact requires evidence that the downstream consumer read the defective version or overlapping partition.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 8 — Blast radius is conditional.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 8 — Blast radius is conditional.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 8 — Blast radius is conditional.

### Result 9 — Reconciliation requires a trustworthy comparator
- **Statement:** Matching a product against itself through a different query does not prove correctness. The comparator needs independent provenance or a separately validated invariant.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 9 — Reconciliation requires a trustworthy comparator.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 9 — Reconciliation requires a trustworthy comparator.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 9 — Reconciliation requires a trustworthy comparator.

### Result 10 — Semantic validity exceeds syntactic validity
- **Statement:** A value can have the expected type, range, and schema while representing a changed unit, population, or business meaning.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 10 — Semantic validity exceeds syntactic validity.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 10 — Semantic validity exceeds syntactic validity.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 10 — Semantic validity exceeds syntactic validity.

### Result 11 — Observability has an observability problem
- **Statement:** A green health panel is meaningful only if the monitor evaluated recently and its own data path, scheduler, and routing are healthy.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 11 — Observability has an observability problem.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 11 — Observability has an observability problem.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 11 — Observability has an observability problem.

### Result 12 — Containment should precede broad repair
- **Statement:** When scope is uncertain, stop new exposure with a reversible action before executing an expansive backfill or changing multiple components.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 12 — Containment should precede broad repair.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 12 — Containment should precede broad repair.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 12 — Containment should precede broad repair.

### Result 13 — Deterministic evidence shortens incidents
- **Statement:** Run manifests, immutable inputs, and versioned lookups make it possible to reproduce a result instead of reconstructing it from memory and mutable systems.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 13 — Deterministic evidence shortens incidents.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 13 — Deterministic evidence shortens incidents.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 13 — Deterministic evidence shortens incidents.

### Result 14 — Availability and correctness are different axes
- **Statement:** A punctual partition can be wrong, and a delayed partition can be correct. Release policies need both timeliness and quality conditions.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 14 — Availability and correctness are different axes.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 14 — Availability and correctness are different axes.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 14 — Availability and correctness are different axes.

### Result 15 — Degraded mode is a product contract
- **Statement:** A known-good snapshot is safe only when its age, scope, and allowed consumer uses are stated; a silent fallback converts reliability work into hidden semantic risk.
- **Operational consequence:** Encode the result in a monitor, runbook branch, contract clause, or review checklist rather than relying on oral memory. In this context, apply it to Result 15 — Degraded mode is a product contract.
- **Boundary:** Re-evaluate when source semantics, consumer actions, dependency topology, or policy requirements change. In this context, apply it to Result 15 — Degraded mode is a product contract.
- **Evidence question:** What observation would falsify the current interpretation and make a different branch of the runbook more likely? In this context, apply it to Result 15 — Degraded mode is a product contract.

## 11. Interview / Exam Questions

### Q1 — Define freshness lag and state why clock choice matters.
- **Answer:** Freshness lag is observation time minus the latest valid time under a declared clock. Event time reveals business recency, while processing time can look current during delayed or backfilled event arrival.
- **Reasoning:** A complete answer names the selected clock, excludes invalid future timestamps, and states the maximum allowed lag.
- **Assessment focus:** Diagnostic category: conceptual.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q1 — Define freshness lag and state why clock choice matters..

### Q2 — At 14:40 UTC the latest valid event is 14:12 UTC and the limit is 25 minutes. Does freshness pass?
- **Answer:** Lag is 14:40 − 14:12 = 28 minutes. Because 28 > 25, freshness fails by 3 minutes under an inclusive at-or-below limit.
- **Reasoning:** Re-check: 12 to 40 is 28 elapsed minutes. The result assumes both timestamps use the declared same timezone and event clock.
- **Assessment focus:** Diagnostic category: calculation.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q2 — At 14:40 UTC the latest valid event is 14:12 UTC and the limit is 25 minutes. Does freshness pass?.

### Q3 — A required field has 36 nulls among 24,000 applicable records. Its maximum null rate is 0.2%. Is there a breach?
- **Answer:** The null fraction is 36/24,000 = 0.0015 = 0.15%. Since 0.15% is below 0.2%, the rule passes numerically.
- **Reasoning:** A strong answer also asks whether the 36 records are concentrated in a critical segment, because rate-only passing may not imply acceptable harm.
- **Assessment focus:** Diagnostic category: calculation and judgment.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q3 — A required field has 36 nulls among 24,000 applicable records. Its maximum null rate is 0.2%. Is there a breach?.

### Q4 — A count falls from baseline mean 10,000 with standard deviation 500 to 8,400. Compute the z-score.
- **Answer:** z = (8,400 − 10,000)/500 = −1,600/500 = −3.2. Its absolute value crosses a two-sided 3-sigma detector.
- **Reasoning:** This is an anomaly, not proof of a pipeline fault; inspect seasonality, source status, and inclusion-rule changes.
- **Assessment focus:** Diagnostic category: calculation and diagnosis.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q4 — A count falls from baseline mean 10,000 with standard deviation 500 to 8,400. Compute the z-score..

### Q5 — Why can a schema check pass during a material data incident?
- **Answer:** Schema checks capture representation such as field names and types, not necessarily unit, population, business definition, or aggregation change. A string can remain a string while its meaning changes.
- **Reasoning:** Use contract versions, cross-field invariants, source-release review, and distribution checks to add semantic evidence.
- **Assessment focus:** Diagnostic category: conceptual.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q5 — Why can a schema check pass during a material data incident?.

### Q6 — A 99% SLO covers 720 hourly due partitions. How many whole failed partitions are allowed under a floor policy?
- **Answer:** Budget equals 720×(1−0.99) = 7.2 failed partitions. A floor policy allows 7 whole failures; an eighth exceeds the target.
- **Reasoning:** The rounding policy must be part of the SLO because a ceiling policy would produce a different operational threshold.
- **Assessment focus:** Diagnostic category: calculation.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q6 — A 99% SLO covers 720 hourly due partitions. How many whole failed partitions are allowed under a floor policy?.

### Q7 — A monitor pages for low global volume, but region A is down 50% and region B is up 50%. What design weakness is exposed?
- **Answer:** Aggregate cancellation hides opposing segment defects. The monitor population is too coarse for the independent failure domains that matter.
- **Reasoning:** Add regional slices and define which segments are material enough to page, while controlling cardinality and response burden.
- **Assessment focus:** Diagnostic category: diagnosis.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q7 — A monitor pages for low global volume, but region A is down 50% and region B is up 50%. What design weakness is exposed?.

### Q8 — What is the first containment action when a partition may be incorrect but lineage is incomplete?
- **Answer:** Prevent new ordinary publication of the suspect partition and preserve its evidence, usually by quarantine or pinning a known-good labelled snapshot where permitted.
- **Reasoning:** Do not begin an unbounded backfill before recording scope, input identities, and consumer paths; repair can otherwise destroy evidence or widen impact.
- **Assessment focus:** Diagnostic category: judgment.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q8 — What is the first containment action when a partition may be incorrect but lineage is incomplete?.

### Q9 — A detector produced 9 pages: 6 were confirmed material incidents, and 2 other incidents had no page. Find precision and recall.
- **Answer:** Precision is 6/9 = 0.666..., or 66.7%. Recall is 6/(6+2) = 0.75, or 75%.
- **Reasoning:** The ninth page is a false positive under the stated labels; label quality and incident definition must be reviewed before retuning.
- **Assessment focus:** Diagnostic category: calculation.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q9 — A detector produced 9 pages: 6 were confirmed material incidents, and 2 other incidents had no page. Find precision and recall..

### Q10 — How do you distinguish a late-arrival problem from a data-loss problem?
- **Answer:** Compare source offsets, control totals, and arrival-time distribution after the agreed watermark. Late records should appear eventually; loss leaves an unreconciled source-to-target gap.
- **Reasoning:** The answer must state the authoritative source and the lateness policy; an arbitrary wait does not establish completeness.
- **Assessment focus:** Diagnostic category: diagnosis.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q10 — How do you distinguish a late-arrival problem from a data-loss problem?.

### Q11 — Why is static lineage insufficient for incident blast radius?
- **Answer:** Static lineage describes possible dependencies, but a downstream product may not have run, may have read a prior snapshot, or may have consumed a non-overlapping partition.
- **Reasoning:** Use run-level input/output manifests and consumer read evidence to bound actual exposure.
- **Assessment focus:** Diagnostic category: conceptual.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q11 — Why is static lineage insufficient for incident blast radius?.

### Q12 — Choose between serving a stale snapshot and publishing a partial current partition.
- **Answer:** Choose according to declared consumer tolerance, but a labelled known-good snapshot is often safer when partial current output could be interpreted as complete.
- **Reasoning:** The decision requires as-of time, scope, action reversibility, and a plan to reconcile any decisions made while degraded.
- **Assessment focus:** Diagnostic category: judgment.
- **Common weak response:** Naming a tool or generic dashboard without defining the population, clock, comparison, and action it enables. In this context, apply it to Q12 — Choose between serving a stale snapshot and publishing a partial current partition..

### Q13 — A partition has 2 duplicates among 100,000 rows, but both duplicates feed a high-value payout. Should a rate threshold alone decide severity?
- **Answer:** No. The duplicate rate is 2/100,000 = 0.002%, but exposure depends on the duplicated payout values and reversibility, not only row fraction.
- **Reasoning:** Compute monetary and action impact, quarantine the affected identities, and reconcile downstream payout state before relying on a global rate.
- **Assessment focus:** calculation and judgment.
- **Common weak response:** Treating a single rate, log message, or mute action as sufficient evidence without checking downstream exposure and recovery obligations. In this context, apply it to Q13 — A partition has 2 duplicates among 100,000 rows, but both duplicates feed a high-value payout. Should a rate threshold alone decide severity?.

### Q14 — What evidence makes a backfill reproducible?
- **Answer:** A reproducible backfill identifies immutable inputs or offsets, time boundaries, code revision, configuration, lookup versions, write mode, output target, and validation results.
- **Reasoning:** Without lookup and configuration identity, the same raw input can produce a different output, making comparison and rollback ambiguous.
- **Assessment focus:** diagnosis.
- **Common weak response:** Treating a single rate, log message, or mute action as sufficient evidence without checking downstream exposure and recovery obligations. In this context, apply it to Q14 — What evidence makes a backfill reproducible?.

### Q15 — When should an alert be suppressed rather than routed?
- **Answer:** Suppress only a bounded, owned, expiring signal during a documented condition when another control covers material harm; suppression is not a substitute for containment.
- **Reasoning:** A correct answer specifies scope, expiry, reason, owner, and a post-change test showing the alert resumes.
- **Assessment focus:** judgment.
- **Common weak response:** Treating a single rate, log message, or mute action as sufficient evidence without checking downstream exposure and recovery obligations. In this context, apply it to Q15 — When should an alert be suppressed rather than routed?.

## 12. Quick Reference Card

| Situation | First evidence | Immediate safe action | Next discriminating check | Owner route |
|---|---|---|---|---|
| Freshness breach | Observation time, newest valid event time, selected clock, and allowed lag | Quarantine current output or serve labelled known-good snapshot if contracted | Compare source arrival, watermark, and processing backlog | Data-product on-call plus source owner |
| Low volume | Observed count, baseline context, numerator population, and segment split | Hold affected partition when completeness is blocking | Check control total, offsets, and segment-level counts | Ingestion owner and producer owner |
| High null rate | Null numerator, applicable denominator, field contract, and sample identities | Prevent release when field is required for consumer decision | Compare source payload, mapping version, and conditional applicability | Transformation owner and domain steward |
| Duplicate keys | Duplicate key samples, key rule, source revision, and merge behavior | Quarantine or block double-counting publication | Trace retries, offset resets, and precedence rule | Producer owner and warehouse owner |
| Schema change | Old/new fingerprint, contract version, compatible consumers, and release timing | Hold incompatible version or route through approved compatibility path | Test semantic invariants and consumer parser behavior | Producer contract owner and consumer owners |
| Distribution shift | Reference window, current window, segment view, and detector statistic | Route to triage queue unless material rule also fails | Check seasonality, policy changes, units, and source version | Domain owner with data owner |
| Monitor stale | Last evaluation time, heartbeat, scheduler status, and alert route | Mark health unknown; do not trust cached green state | Test independent heartbeat and credential path | Monitoring platform owner |
| Suspected lineage fan-out | Suspect run/partition, executed reads, and consumer list | Group derivative alerts and stop new exposure at gate | Traverse run-level lineage with partition overlap | Incident commander and downstream owners |
| Backfill request | Affected range, raw input retention, code/config versions, and rollback point | Dry-run and fence writes before production replay | Verify idempotence, source revisions, and consumer correction protocol | Data-product owner and storage owner |
| Closure decision | Passing checks, reconciliation output, exposure interval, and consumer status | Release only after evidence is recorded | Verify no bypass consumer or pending corrective action remains | Incident owner with accountable product owner |
