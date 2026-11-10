---
case_id: "CASE-2026-STR001"
case_title: "PayPal observability: choosing a recoverable streaming path for telemetry"
case_version: "2.0"
created: "2026-11-07"
last_reviewed: null
status: "draft"
content_status: "draft"
review_status: "OPEN"
implementation_stage: "production-migration reported by the source; not independently verified"
case_type: "streaming-systems"
domain: "Observability platform engineering; streaming analytics"
primary_pattern: "recoverable-stream-processing"
secondary_patterns: ["partition-aware-scaling", "migration-reconciliation"]
evidence_level: "published"
source_basis: "published"
as_of_date: "2024-12-03"
owner_role: "Observability platform team"
source_index: "INDEX.md"
review_trigger: "New primary evidence, a material change in broker or runner delivery semantics, or a completed editorial review"
---

# PayPal observability: choosing a recoverable streaming path for telemetry

## 1. The One-Line Framing

Given a continuous flow of application telemetry — log-derived metric points and trace spans produced by services running on several language runtimes — decide which streaming execution platform and which ingestion design should turn those records into timely, complete, replayable observability outputs for developers, on-call engineers, and command-centre staff, subject to an operational constraint that the previous self-managed deployment could not hold (most pipelines lacked checkpointing and lost data across restarts), relative to that previous Pulsar-plus-self-managed-Flink baseline.

The unit of analysis is one telemetry record at the moment it crosses the ingestion boundary, not a payment authorisation, a merchant account, or a fraud score. That distinction matters because the arithmetic of a telemetry path is dominated by volume, cardinality, and completeness rather than by monetary value per record: a dropped telemetry record does not move money, but a dropped telemetry record during an incident removes the evidence that operators need in order to stop money from moving incorrectly elsewhere.

Decision frequency is continuous rather than episodic. The platform choice was made once (the source reports a proof of concept, a Pulsar-to-Kafka ingestion change, and a Dataflow pipeline), but the same team re-makes a smaller version of the decision at every capacity change, every schema revision, every runner image update, and every retention adjustment. A case that only teaches "pick the managed runner" therefore teaches the wrong recurring decision.

The primary success condition is a telemetry path whose outputs are trustworthy enough to diagnose an incident while the incident is still in progress. "Trustworthy" decomposes into three conditions that must hold together: records that were accepted at the source appear at the sink (completeness), they appear soon enough to be useful (freshness), and they appear the expected number of times, or the excess is disclosed (effect multiplicity). Any one of these conditions alone can be satisfied by a path that fails the other two, which is the central engineering point of this case.

The reusable pattern is `recoverable-stream-processing`: source replay, processor state, and sink effects must be aligned to one explicit recovery contract, and the contract must be tested as a whole rather than inferred from a product label. Two supporting patterns appear throughout: `partition-aware-scaling`, because every global average in this domain hides a key distribution, and `migration-reconciliation`, because a migration that changes the broker and the runner simultaneously creates two independent opportunities for the same logical record to be lost or duplicated.

Evidence declaration, stated before any number appears: the qualitative situation and the reported intervention come from a vendor blog co-authored by a PayPal observability architect and a Google Cloud engineering manager, cited as [S1] in Section 5. The reported outcomes are the authors' narrative claims; they are not an independent measurement, not a controlled comparison, and not an audited cost statement. Every numeric quantity in this case — volumes, rates, retention windows, partition counts, completeness fractions, unit prices, latency budgets — is a constructed teaching value labelled ILLUSTRATIVE. No number below should be read as a PayPal measurement, and no percentage improvement is inferred from the source.

The decisive tradeoff a reader should carry away is this: freshness is not bought with faster processing once a pipeline performs event-time windowing. The dominant term in end-to-end freshness is the deliberate waiting for late data to arrive, and shortening that wait converts freshness into late-arrival corrections. Section 6 quantifies the exchange rate — at the illustrative workload, cutting allowed lateness from 120 s to 60 s removes 60 seconds from p95 finalisation and triples the daily correction volume from 0.40 million events to 1.20 million, which is 0.40% of the day's records reprocessed. That trade is a product decision about how much correction downstream consumers can absorb, and no amount of additional worker capacity changes it.

The obvious simpler framing — "move to a managed service so the operations team stops babysitting clusters" — misses the part of the decision that actually determines whether the migration is safe. A managed runner removes cluster provisioning and image maintenance; it does not remove the need to define source retention, deduplication identity, sink write semantics, late-data policy, or the cutover boundary between two brokers. Those five artefacts are what fail in production, and all five are owned by the migrating team regardless of who operates the compute.

## 2. Context & Background

The domain is observability platform engineering for a large payments organisation. Application services emit structured logs, metric points, and distributed traces; a collector layer receives them; a streaming layer transforms raw emissions into the aggregates and span records that dashboards, alerting rules, and incident responders consume. The observable product is not a single screen but a contract: a named metric series with stable labels, a bounded lag from event occurrence to queryability, and a trace that can be followed across service boundaries.

The workflow before the decision runs roughly as follows. An application writes a log line or a span through an instrumentation SDK. An OpenTelemetry-style collector batches and forwards records to a broker. A stream processor parses each record, extracts dimensions, maintains aggregation state per key, and emits derived metric points and assembled spans. A storage backend indexes those outputs, and query surfaces and alert evaluators read from it. The user of the pipeline is, at the point of use, usually a human under time pressure: an engineer asking why latency rose, a support engineer reconciling a merchant complaint, or a command-centre operator deciding whether a degradation is regional or global.

Stakeholders and their distinct interests are worth separating because the case is partly a governance story. The platform team owns the pipeline and its cost. Application teams own emission volume, label choice, and schema changes — they can break the platform without touching it. Site reliability engineers and incident commanders are consumers whose work becomes slower when telemetry is incomplete, and whose trust in the platform is the real availability metric. Security and privacy owners constrain what may be retained and for how long. Finance constrains unit cost per ingested byte or per retained series. The decision owner for the platform choice is the observability organisation with cloud and security review; the source does not name an individual approver.

The baseline is the interesting part, and it must be stated as the source states it. Until the reported 2023 transition, the platform ran self-managed Flink-based infrastructure for log-derived metrics and spans. The authors report that most of those pipelines lacked checkpointing and that data was lost during restarts. They also describe provisioning for peak, manual cluster creation and maintenance, security alignment work, a proprietary solution with limited community support, an unsupported customised binary, and end-of-sale exposure. Each of those is a claim about one deployment. None of them is a property of Apache Flink as a system, and a reader who converts "our deployment lacked checkpoints" into "Flink loses data" has learned the wrong lesson from the case.

Scale, as reported: applications on Java, Go, and Node.js were said to produce approximately three petabytes of logs per day. That figure is an aggregate of application log generation, not a measured Dataflow input rate, not a peak, and not a per-partition distribution. The article gives no petabyte convention (decimal or binary), no measurement window finer than "per day", no statement of what fraction of that log estate becomes streaming input, and no compression or replication accounting. A teaching case therefore has to construct its own workload to make the arithmetic legible, and must say so every time it does.

Timing matters in two separate senses. First, calendar time: the source is dated 2024-12-03, so it describes a migration believed complete before that date; the publication date is not a cutover date and the case does not invent one. Second, event time versus processing time, which is the technical timing distinction that governs correctness. A record carries the time at which the activity happened (event time) and arrives at the broker at some later wall-clock moment (ingestion time). Windows are assigned by event time, watermarks estimate how complete event time has become, and allowed lateness decides how long an already-finalised window may still be corrected. Confusing the two produces the single most common silent wrong answer in streaming analytics: charts that look healthy because they describe a different population than the one the question asked about.

Geographic and organisational boundaries are not disclosed. The source does not state regions, replication topology, cell structure, or which PayPal business units consume which outputs. Any architecture drawing in Section 8 is therefore explicitly a teaching construction, not a reconstruction.

Prior attempts and why the previous process persisted are only partly visible. The baseline existed because self-managed Flink is a legitimate choice: it offers control over state backends, upgrade timing, and cost shape, and it avoids dependence on a single vendor's quota and pricing. The reported failure was not that the technology choice was absurd; it was that operating it without checkpoint discipline accumulated a correctness debt that surfaced at the worst moment — a restart. That is a recognisable organisational pattern: the cost of a missing recovery contract is paid during incidents, when nobody has time to add one.

Applicable requirements are stated only at the level of intent. The source mentions the need for security alignment and the importance of stable, timely telemetry. It does not cite a regulatory obligation, a contractual service level, a data-residency rule, or an internal control standard. This case therefore does not assert any specific compliance regime for PayPal; it treats privacy, retention, and access control as requirements that the reader's own organisation must source from its own policy owners.

The consequence of the context is a decision with an asymmetric cost profile. A platform that is slightly more expensive per byte is a budget conversation. A platform that silently loses the telemetry of the service that is failing is an incident-amplification event, because the responders lose the instrument while using it. That asymmetry — cheap visible cost versus expensive invisible failure — is why the case is organised around completeness and recovery rather than around throughput.

A useful way to hold the domain in mind is that an observability pipeline is a market with a peculiar demand curve. Demand is generated automatically by other systems, so it grows without anyone deciding to consume more; it is inelastic in the short run, because a service cannot stop emitting telemetry during an incident without blinding its own responders; and it is highly skewed, because a small number of services and a small number of labels generate most of the volume. The consequence is that cost control in observability is almost never about the mean. It is about the tail of producers and the tail of cardinality, which is why Section 6.10 computes state size as a function of key count rather than of event count.

The domain also has an unusual property that shapes every design decision: the consumer and the operator are often the same person. A platform team that degrades telemetry to save cost during a busy period is degrading the tool it will use to diagnose the consequences of that busy period. That self-referential relationship is the origin of the "who watches the watchers" requirement in Section 12.14, and it is why the health signal for the telemetry path must live outside the telemetry path. It is also why arguments for telemetry reliability are structurally different from arguments for, say, batch-reporting reliability: the person who must approve the cost is the same person who will be paged when the cost saving removes their instrument.

Three quantities describe the operating environment well enough for a design conversation, and all three are usually unmeasured. The first is records per second at peak, which sets the capacity requirement. The second is distinct series, which sets state, index, and query cost. The third is the ratio of intentionally excluded records to eligible records, which sets the credibility of every completeness statement the platform makes. A team that can state all three can size a runner; a team that cannot is choosing between vendors on the basis of marketing, which is the situation the published case does not let us rule out for the baseline.

The reporting gap is itself part of the context rather than an accident of the source. Observability platforms are rarely measured with the rigour they apply to the systems they observe: the platform's own completeness has no owner outside the platform team, its freshness target is negotiated informally with consumers, and its cost is compared year over year against a number that grew with traffic. This is a recognised structural asymmetry in the discipline, and it explains why the case's most transferable content is a measurement design rather than an architecture. A team that adopts the measurement design will discover its own architecture's weaknesses; a team that adopts only an architecture will not.

Finally, the organisational timing matters. A migration like this is opportunistic in the sense that it becomes possible when a cloud relationship, a support ending, and an operational pain coincide. The source describes the pain (restart loss, manual operations) and the destination (a managed runner), and the coincidence is what allows a two-variable change to be approved at all. A reader planning a similar move should expect the window to be narrow and should arrive at it with the two cliff measurements already in hand, because a decision window that opens on schedule pressure is a window in which unmeasured risks get accepted.

## 3. The Business Question

The reconstructed decision question, labelled a paraphrase because the source reports an outcome rather than a request: "Which ingestion and stream-processing design should carry our application telemetry so that metrics and traces remain timely and complete through restarts, without the peak-provisioned manual operations that the current deployment requires?"

A second, subordinate question is embedded in the first and is easy to overlook: "How will we know, at the moment of cutover, that the replacement path is at least as complete as the path it replaces?" The first question selects a platform. The second question selects a measurement design. This case argues, and Section 6 demonstrates with arithmetic, that the second question is the harder one and the one whose neglect produces the failure modes catalogued in Section 12.

| Decision contract field | Content for this case |
|---|---|
| Question | Which ingestion design and stream-processing platform should generate metrics and spans from application telemetry, with defined recovery behaviour and controlled operational burden? |
| Decision owner | Observability platform organisation, with cloud platform and security review; individual approver not disclosed by the source |
| Trigger | Reported restart-time data loss in most pipelines, manual cluster operations, unsupported customised binary, and end-of-sale risk |
| Deadline | Not disclosed; the source describes a proof of concept followed by a migration, with no published schedule |
| Available actions | (a) retain the self-managed deployment and add checkpointing, automation, and a supported build; (b) migrate to a managed runner with the existing broker; (c) migrate both broker and runner; (d) split the estate so that only part of the telemetry moves |
| Value at stake | Incident-diagnosis capability, on-call workload, infrastructure and labour cost, and the credibility of dashboards used in merchant-facing support |
| Objective quantity | Maximise the fraction of eligible emitted records that become queryable outputs within the freshness budget, at the lowest total cost, while never publishing an output that is silently incomplete |
| Hard constraints | Telemetry must satisfy internal security and access requirements; the path must survive single-worker, single-broker-partition, and sink-degradation events without unbounded loss |
| Scope | Log-derived metrics and trace spans produced by instrumented application services; the streaming ingestion, processing, and export path |
| Out of scope | Payment authorisation, fraud scoring, merchant risk models, the tracing UI product itself, and any statement about PayPal's current production configuration |

The available actions deserve more than a list, because the choice among them is the case's real content. Action (a) — repair in place — is the strongest alternative and is routinely dismissed too fast. Adding checkpointing to an existing Flink deployment is a configuration and testing exercise with a known blast radius: the topology, the state backend, and the schema contracts stay fixed, so the only new variable is recovery. Automating cluster lifecycle is an engineering project with a clear deliverable. Replacing an unsupported customised binary is a supply-chain task. Each of those is bounded, and together they could plausibly satisfy the stated problem without changing broker, runner, or partitioner at once. The countervailing argument in the source is end-of-sale exposure and limited community support: repairing a platform whose vendor path is closing buys time but not a durable destination. A rigorous decision record would score (a) explicitly and state which of its deliverables was judged infeasible, and the source does not publish that scoring.

Action (b) — new runner, existing broker — isolates the execution-engine change. That has a real methodological advantage for evaluation: the broker retention model, the partition key distribution, and the offset semantics stay constant, so any change observed in completeness or freshness is attributable to the processor. It is the design a careful team would prefer for a first migration wave, precisely because Section 6 shows that most silent loss originates at boundaries rather than inside a transform.

Action (c) — change both — is what the source reports: Pulsar ingestion replaced by Kafka ingestion, and self-managed Flink replaced by Dataflow. It delivers the largest operational change in one step, and it also destroys the ability to attribute a post-migration discrepancy to one cause. If, after cutover, three merchant-dashboard aggregates disagree with the old path, the candidates include a different partition key, a different commit protocol, a different late-data policy, and a different parser version. Four candidates, one observation. That is not an argument that the migration was wrong; it is an argument that the migration needs an evaluation design that can separate the candidates, and Section 9 specifies one.

Action (d) — split the estate — is the option a risk-averse organisation often picks and the source does not mention. Move the highest-volume, lowest-value log-derived aggregates first, leave trace assembly on the repaired legacy path, and migrate traces once the aggregate path has demonstrated recovery under drill. It reduces the simultaneous-variable count at the cost of running two platforms for longer, which is a real and quantifiable operational burden. Section 7 discusses why the honest conclusion is that the case does not establish which of these four was optimal, only which one was reported and which conditions would have made it unsafe.

The relationship between the business objective and the measurable proxy is where most observability migrations quietly go wrong. The business objective is "responders can diagnose and act". The available proxy is "outputs arrive within the freshness budget with completeness above a threshold". These can disagree in both directions. A path can hit a p95 freshness target of ten seconds while dropping the records of one saturated partition, in which case responders see a fast and confidently wrong picture — the disagreement that matters most, because latency dashboards look excellent. Conversely, a path can be slower than target while being perfectly complete, which delays diagnosis but does not mislead it. The guardrail that resolves the conflict is a completeness floor reported per partition alongside every freshness percentile, plus an explicit "absent coverage" alarm that fires when a partition emits nothing at all rather than reporting a lag of zero.

An expensive false decision looks like this: the team declares the migration successful because cost fell and the aggregate p95 freshness improved, removes the legacy path, and then discovers during the next major incident that one high-volume service's telemetry was being filtered by a predicate introduced in the new transform. Cost fell partly because a fraction of the data was never processed. That is the archetype this case is written to prevent, and it is why Section 12 devotes separate entries to incorrect filter deployment, false healthy freshness, sampling policy mismatch, and identity collisions rather than folding them into a single "data loss" heading.

An expensive false negative also exists and is worth naming so the case is not read as counsel for paralysis. A team that never cuts over keeps paying the operational tax of an unrecoverable pipeline, and the tax is paid in incidents: each restart during an incident destroys the evidence the incident needed. The decision is therefore not "migrate or be safe" but "which risk is bounded and measurable, and which is unbounded and invisible". Migration risk is bounded by drills and reconciliation. The risk of an unrecoverable pipeline is bounded only by luck, because it fires precisely when the platform is already degraded.

The question can be answered by an action at two levels, and a good interview answer names both. At the platform level: adopt a runner whose recovery behaviour is a documented, testable contract, and accept the migration cost. At the operational level: refuse to declare the migration complete until a deliberately injected worker failure, a deliberately injected sink failure, and a deliberately expired retention window have each been survived with reconciled identities. The second answer is the one that transfers to any organisation; the first is specific to a vendor relationship the reader may not have.

## 4. Assumptions & Constraints

The assumption register below is deliberately explicit about testability, because four of the five load-bearing assumptions in a telemetry migration cannot be verified by reading documentation. They are verified by injecting a failure and inspecting what the sink received. Every row carries a status; none is marked verified, because no test was executed for this case.

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | A replayable source retains every required record for longer than the worst-case recovery interval | Data / infrastructure | Compare configured retention against measured worst-case drain time plus margin; run an expiry drill | Unverified | Replay cannot reach records that expired; the gap is permanent | Extend retention, or retain an independent authoritative copy of raw emissions | Retention or recovery-objective change / platform team |
| A-02 | Every telemetry record carries an identifier that is stable across producer retries and unique per logical event | Data | Sample the same logical event at producer, broker, and sink; count distinct identities per emitted event | Unverified | Reconciliation cannot distinguish loss from duplication; both errors hide | Introduce a producer-side event identity contract, or define a documented composite key | Producer SDK or schema release / instrumentation owners |
| A-03 | Sink mutations tolerate retry, or the sink can deduplicate on a supplied key | Data / model | Inject failure between sink write and progress commit; count duplicate outputs | Unverified | At-least-once recovery publishes duplicate metric points and double-counted spans | Idempotent write keyed on output identity, or a transactional sink protocol | Sink contract change / sink owner |
| A-04 | Sink and runner capacity exceed combined live-plus-replay demand during recovery | Infrastructure | Test the bottleneck with concurrent live traffic plus a controlled replay | Unverified | Recovery never converges; the backlog grows while the team believes it is draining | Reserve recovery headroom, throttle replay explicitly, or add capacity at the true bottleneck | Load or topology change / platform team |
| A-05 | Downstream consumers can distinguish a provisional window from a final one | Data / business | Review consumer contracts and dashboard query semantics | Unverified | Provisional aggregates are treated as final and drive bad decisions | Publish completeness metadata, or hold outputs until the lateness horizon closes | Consumer contract or dashboard change / consuming teams |
| A-06 | The record schema is forward-compatible across the migration window | Data | Deploy a reader against an unknown future version; verify explicit failure rather than silent coercion | Unverified | Silent field coercion corrupts values without raising errors | Schema registry with compatibility checks; quarantine unknown versions | Schema release / producer and platform teams |
| A-07 | Cardinality of the emitted metric dimensions is bounded by a known policy | Data / business | Measure distinct series per day and per service; identify top contributors | Unverified | State and index growth outruns capacity; cost rises without volume rising | Cardinality budget per service, with a review path for exceptions | New label or dimension rollout / service owners |
| C-01 | Telemetry handling must satisfy internal security, access, and retention requirements | Security / compliance | Obtain the actual control requirements from the responsible owner; do not infer from geography | Review needed | A migration that changes data location or access path may violate an unstated control | Security review before cutover; minimisation at the collector | Policy change / security owner |
| C-02 | Diagnosis requires outputs to be both timely and complete | Business | Negotiate an explicit service objective with consumers; record it as a contract | Qualitative only | "Improved stability" becomes unfalsifiable, and no acceptance test can fail | Define freshness percentile, completeness floor, and coverage alarm | Consumer expectation change / incident review |
| C-03 | Broker and runner integration must work under actual network and identity policy | Infrastructure | Test connectivity, authorization, and audit logging before cutover | Unverified | Cutover stalls on permissions discovered at the last moment | Pre-provision principals and test with a controlled record | Identity policy change / cloud platform team |
| C-04 | Cost comparison is possible over comparable workload windows | Business | Collect baseline and target bills normalised by ingested bytes, retained series, and replay volume | Unverified, and not published by the source | Cost claims become directional at best; a cheaper bill may reflect less processed data | Normalise by workload; separate replay and incident-driven cost | Pricing or workload change / finance partner |
| C-05 | An operator can declare an interval of telemetry incomplete and have that declaration respected | Business / process | Exercise the declaration in an incident drill and verify downstream display | Unverified | Consumers keep treating a gap as truth, and responders chase phantom health | Completeness metadata surfaced in the query layer; documented escalation | Incident-process change / incident command |
| C-06 | Rollback to the legacy path remains technically possible for the duration of the migration | Infrastructure | Verify the legacy path can still read current source records; test a rollback with in-flight data | Unverified | A failed cutover becomes an unrecoverable outage rather than a delay | Keep a compatible dual-read capability, or accept a forward-fix-only cutover explicitly | Schema or broker-retention change / platform team |

The four assumptions most likely to reverse the recommendation are A-01, A-02, A-03, and A-04, and they share a structure: each one is about a boundary rather than a component. A-01 governs whether replay is possible at all. A-02 governs whether the team can tell what happened. A-03 governs whether repeating work is safe. A-04 governs whether any of it converges in time. A team that verifies all four has verified the recovery contract. A team that verifies only runner throughput has verified almost nothing about the decision.

Consider the consequence chain when A-01 fails, because it is the most instructive. Suppose the worst-case drain time for a plausible backlog is 150 minutes and the broker retention is configured at 72 hours. Superficially there are 42 hours of headroom, so A-01 looks safe. But retention configured by time is not retention delivered by time: a size-based eviction can shorten the effective window, and a partition that stops being consumed while others advance will have its own expiry clock running independently. The measured quantity must therefore be "oldest retained offset age per partition", not "configured retention period". A platform that reports a configured retention of 72 hours while one partition's oldest record is 40 minutes old is a platform whose real recovery window is 40 minutes. Section 12.48 treats the configuration error; Section 9 defines the measurement.

Now consider A-02 failing in a subtler way. Suppose the producer assigns an identifier per emission, not per logical event. A retry after a network timeout emits the same log line again with a new identifier. Deduplication downstream now cannot collapse the pair, so the metric counts the event twice; meanwhile the reconciliation job, which compares distinct identifier sets, reports zero loss. Both the completeness measurement and the output values are wrong, and they are wrong in the same direction: everything looks complete and volumes look slightly high, which is routinely attributed to organic growth. The discriminating test is to compare the count of logical events observed by the producer against distinct identifiers at the sink for the same window and the same producer version — an exercise that requires cooperation from the emitting teams and is therefore usually skipped.

Now A-03, which is where streaming systems most often claim more than they deliver. A runner can guarantee that its own internal state is exactly once, and the same runner can still publish duplicate outputs, because the durable commit of a sink write is outside the runner's state. The failure window is small and precise: the sink acknowledges the write, the runner dies before recording that acknowledgement, and the replay repeats the write. Small windows are the ones teams convince themselves do not matter, and they matter exactly in proportion to the sink's cost per duplicate. For an aggregated metric point, a duplicate may be invisible behind an averaging query; for a count of failed requests, a duplicate inflates the number that a responder uses to size an incident.

C-03 deserves an explicit note because it is the constraint most often discovered late and least often discussed. Identity and network policy in a large organisation is not a detail of the cloud provider; it is an organisational artefact with its own approval latency. A migration plan that assumes a service principal can read a topic and write to a sink by default will spend its final week in a permissions queue. The correct treatment is to treat access provisioning as a workstream with its own lead time, tested with a controlled record end to end before any volume moves.

Finally, the constraints that are genuinely outside the team's control should be stated plainly so that the reader can distinguish them from choices. The security and retention requirements (C-01) are imposed. The requirement that diagnosis be timely (C-02) is imposed by the nature of incident response, even though its numeric threshold is negotiated. The compatibility of broker and runner under network policy (C-03) is imposed by the organisation's platform standards. Everything else in the table — retention duration, identity scheme, sink protocol, headroom, schema policy, cardinality budget, rollback capability — is a design choice the team owns, and each one has a cost that must be paid somewhere. The case's practical advice is to name the owner of each choice in the design document, because an unowned choice becomes a default, and defaults in streaming pipelines are the ones that lose data.

## 5. Data, Tools & Method

### 5a. Data Specification

The unit of analysis is a single telemetry record at ingestion: one log-derived metric observation, or one span, as emitted by an instrumented service and admitted to the broker. Records are not independent in the statistical sense — they cluster by service, by host, by trace, and by time — and every metric in this case must therefore be reported with the clustering dimension that matters for the question (usually service and partition) rather than as a global average.

| Field | Specification for this case | Status |
|---|---|---|
| Unit of analysis | One telemetry record at the ingestion boundary; one derived output record at the sink | Defined |
| Primary key | Producer-assigned event identifier; must be stable across retries | Not disclosed by the source; treated as a required contract (A-02) |
| Partition key | Not disclosed; in the teaching model a hash of service plus a coarse key, giving uneven load | Illustrative |
| Time fields | `event_time` (when the activity occurred), `ingest_time` (broker admission), `sink_visible_time` (queryable) | Three distinct fields, never interchangeable |
| Volume and window | Source reports ≈3 PB of application logs per day, aggregate, convention unstated; teaching model uses 300 million records/day | Reported aggregate; teaching volume ILLUSTRATIVE |
| Payload size | Teaching model uses 900 bytes/record mean, with a heavy tail from multi-line stack traces | Illustrative |
| Schema | Structured record with service name, environment, schema version, severity, dimensions, payload | Teaching envelope; real schema not published |
| Outcome or label | No model label exists; the correctness oracle is identity-level reconciliation and aggregate comparison | N/A — this is a systems case, not a predictive one |
| Population | Records admitted to the broker in a declared event-time interval, from declared producer versions | Must be declared per measurement |
| Exclusions | Records filtered by an intentional sampling policy; records rejected by an intentional, documented validation rule | Must be enumerated per measurement, never silent |
| Join semantics | Source identities joined to sink identities within a bounded lateness horizon; duplicates counted separately | Defined in Section 6.6 |
| Splits | No train/validation/test split; the analogue is a frozen comparison cohort plus a separate drill window | Defined in Section 9 |

The three time fields carry the case's central measurement distinction, so they are worth separating carefully. `event_time` is assigned by the producer and describes the world. `ingest_time` is assigned at broker admission and describes the platform's exposure to the world. `sink_visible_time` is assigned when the output becomes queryable and describes when a human could have seen it. Freshness is `sink_visible_time − event_time` for the population that eventually arrives. Ingestion delay is `ingest_time − event_time` and belongs to the producers and the collector, not to the streaming job. Processing delay is `sink_visible_time − ingest_time` and belongs to the job. When an operator says "the pipeline is lagging", the first diagnostic question is which of those two differences grew, because they have different owners, different fixes, and different failure modes.

The absence of a label is not a gap in this case; it is a definitional property. There is no ground truth column that says whether a metric point was correct. Correctness has to be constructed from identity reconciliation (did this specific record reach the sink, once, with the right values?) and from aggregate comparison against an independent implementation over a bounded window. The consequence is that evaluation effort concentrates on building the oracle, not on fitting a model, and a case study that omits the oracle construction has omitted the method.

A note on the reported petabyte figure, since readers reach for it constantly. Three petabytes per day of application logs is a statement about what applications write. The streaming pipeline sees a subset: after collector sampling, after dropping debug-level lines under policy, after the parser discards unparseable records, and after compression affects bytes on the wire but not bytes in memory. Converting the reported figure into an input rate therefore requires at least four ratios that the source does not supply. In this case we do not attempt the conversion. Instead we choose a teaching input rate explicitly (300 million records per day at 900 bytes mean) so that the arithmetic in Section 6 is checkable, and we label every derived number as a property of the teaching model.

The teaching model's workload shape matters more than its level, so it is specified with a diurnal profile rather than as a constant. The model uses a daily mean of 3,472 records per second, a peak hour at 2.6× the mean, and a nightly trough at 0.35× the mean. This shape is the reason Section 6.2 computes both a mean and a peak-derived capacity requirement, and it is the reason a capacity plan validated only at the mean fails on the first Monday. The profile is invented; the phenomenon is not.

Partition-level structure is the model's second essential property. The teaching model has 24 partitions with load shares ranging from 1.1% to 12.5% of records. A uniform distribution would give each partition 4.167%. The busiest partition therefore carries exactly three times its proportional share, which is the number Section 6.2 uses to show why aggregate capacity and partition capacity are different budgets. Again: the shares are constructed to make a real structural problem visible, not reported by any source.

Completeness accounting requires an explicit registry of intentional exclusions, because an honest completeness metric is meaningless without one. The teaching model declares three exclusion classes: `SAMPLE` (collector sampling under a stated policy), `INVALID` (records failing a documented validation rule, routed to quarantine with their identity preserved), and `OUT_OF_SCOPE` (record types deliberately not processed by this pipeline). Any record not in one of those classes and not present at the sink counts against completeness. Without this registry, a completeness number of 99.7% cannot be interpreted: it might mean the pipeline is excellent, or it might mean the pipeline is quietly discarding a class of records that the metric never counted.

### 5b. Tools & Stack

The reported baseline is Pulsar ingestion with a self-managed Flink-based streaming deployment. The reported target is Kafka ingestion with Google Cloud Dataflow, which executes pipelines written against the Apache Beam programming model. The source states that partitioning was redesigned and shuffling optimised with vendor engineering support, and that BigQuery integration supported real-time analytics initiatives for failed custom interactions and merchant monitoring.

| Layer | Baseline as reported | Target as reported | What remains the team's responsibility either way |
|---|---|---|---|
| Producer instrumentation | SDKs for Java, Go, Node.js | Same | Emission volume, label cardinality, schema compatibility, event identity |
| Collector | OpenTelemetry-style collectors | Same | Sampling policy, backpressure behaviour, rejection accounting |
| Broker | Pulsar | Kafka | Retention, partition count, key distribution, access control |
| Execution | Self-managed Flink, mostly without checkpointing | Managed Dataflow on Beam | Transform semantics, state design, late-data policy, watermark policy |
| Sink | Observability backend | Observability backend plus BigQuery initiatives | Output identity, duplicate handling, visibility semantics, query contracts |
| Operations | Manual cluster creation and maintenance, peak provisioning | Managed service with autoscaling controls | Quota management, cost guardrails, drill execution, incident ownership |

The right way to read that table is by column, not by row. The third column is the case's argument: the migration changes who provisions compute and who patches images, and it changes nothing about the five artefacts that determine correctness. A team that reads only the first two columns concludes the migration is an operational simplification. A team that reads the third column concludes the migration is an operational simplification *plus* a re-implementation of five contracts, and plans accordingly.

The source does not disclose worker sizing, fanout, Kafka topic configuration, connector choice, sink commit protocol, state backend parameters, or the Beam version in use. Those omissions are consequential rather than cosmetic: fanout and topic configuration determine per-partition capacity, and the commit protocol determines whether recovery can be exactly-once in effect or merely at-least-once. This case therefore never asserts them. Where Section 6 needs a parameter, it is named as a teaching parameter and given a plausible value.

One tooling choice that the source implies but does not detail deserves explicit treatment: the reconciliation path. Comparing two brokers requires reading from both, and the legacy path is being decommissioned. A team that wants an offline reconciliation job must therefore keep a legacy read path alive for the comparison window, or stage a raw copy of source records into storage during the comparison. This is a schedule constraint hidden inside a tooling decision: the reconciliation capability has to exist before the legacy path is removed, which means the migration plan must reserve capacity and time for a job whose only product is evidence.

### 5c. Method Summary

The method the case teaches has five stages. First, characterise the workload: records per second at mean and peak, bytes per record, partition shares, and cardinality. Second, size the path against the strictest bottleneck rather than the average, which yields a capacity number and a headroom number. Third, define the recovery contract as a set of three coupled requirements — retention at least as long as worst-case drain, state recoverable to a consistent point, and sink effects repeatable or deduplicable — and test each with an injected failure. Fourth, define the measurement contract: freshness percentiles per partition, completeness against a declared population, duplicate rate, and a coverage alarm for silence. Fifth, define the cutover contract: per-partition boundaries, single-writer ownership, and a reconciliation window whose results are required before the legacy path is retired.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Repair the existing self-managed deployment: add checkpointing, automate lifecycle, replace the unsupported build | Source describes the failure as missing checkpoints and manual operations | Smallest number of simultaneously changed variables; keeps broker and keys fixed; known blast radius | Does not address end-of-sale or community-support exposure; leaves the team owning cluster operations permanently | Considered, not reported as tested | A supported build with a viable support horizon becomes available |
| Keep the broker, change only the execution engine | Methodological advantage: isolates the runner as the single changed variable | Attribution of any post-migration discrepancy stays possible | Leaves the ingestion layer and its operational characteristics untouched | Considered as an evaluation design, not reported | Broker end-of-life forces a move anyway |
| Change broker and execution engine together | Source reports this: Pulsar to Kafka, Flink to Dataflow | Largest operational change in one step; aligns ingestion with the chosen runner's ecosystem | Destroys single-cause attribution; requires reconciliation across two broker semantics | Reported as chosen | Any observed divergence that cannot be attributed |
| Split the estate: migrate aggregates first, traces later | Not mentioned by the source | Reduces simultaneous variables; lets the team learn on the cheaper workload | Two platforms to operate for an extended period; duplicated skills and on-call | Not tested; presented as an alternative | Evidence that trace assembly is the riskier workload |
| Retain the legacy path indefinitely and invest nothing | The status quo | Zero migration cost | Loses telemetry on every restart, during incidents, which is exactly when it is needed | Rejected in the source by implication | Only if telemetry loss is formally accepted for this estate |

The strongest alternative is the first row, and a fair reading of the case requires saying why it loses. Repairing in place fixes the reported symptom (restart loss) at the lowest coordination cost, and it can be validated with exactly the same drills this case prescribes. What it does not fix is the supply-chain problem: an unsupported customised binary and a product at end of sale mean the repair has a shelf life measured in the vendor's remaining support window. So the honest formulation is not "repair was wrong" but "repair buys recovery now and repeats the migration later, at a moment chosen by the vendor rather than by the team". An organisation with a low tolerance for repeated migrations may still prefer the reported path; an organisation with a low tolerance for a two-variable cutover may prefer repair-then-migrate as two sequential steps.

The evaluation design the case recommends is therefore staged regardless of which candidate is chosen. Stage one: verify the recovery contract on the current workload, with injected failures (this is where the interesting bugs live and where they are cheapest to find). Stage two: run a shadow comparison over a frozen cohort with the new path's writes isolated from production side effects. Stage three: cut over partition by partition, with a single declared writer and a reconciliation gate per partition. Stage four: retire the legacy path only after the reconciliation results for the full cohort are accepted by a named owner. The source reports a proof of concept and a migration but not these stages, so this staging is a teaching recommendation, and Section 10 labels it as such.

### 5d. Provenance, Quality, Exclusions & Leakage

**Source.** [S1] Varun Raju (PayPal architect) and Avi Baruch (Google Cloud engineering manager), "PayPal's Real-Time Revolution: Migrating to Google Cloud for Streaming Analytics", Google Cloud Blog, published 2024-12-03, accessed 2026-11-07, at `https://cloud.google.com/blog/products/data-analytics/paypals-dataflow-migration-real-time-streaming-analytics`. No other source was obtained for this case.

**Provenance assessment.** The article is a primary account in the narrow sense that its authors participated in the work, and it is simultaneously a vendor publication whose publisher sells the platform being recommended and whose second author is employed by that publisher. Both facts are relevant to how the qualitative outcome claims should be read, and neither makes the architectural description worthless: a vendor blog that names a specific migration from Pulsar to Kafka and describes partition redesign is reporting an event, and events can be true even when the framing is promotional.

**What the source supplies.** The problem framing (most pipelines lacked checkpointing, data lost on restart), the operational complaints (manual clusters, peak provisioning, security alignment, unsupported binary, end-of-sale), the intervention shape (proof of concept, Pulsar to Kafka, Dataflow, partition and shuffle optimisation), the reported benefit categories (lower infrastructure and operational cost, improved stability and uptime, development agility), one scale figure (≈3 PB of logs per day), and two downstream initiatives (BigQuery-backed failed-custom-interaction analytics and merchant monitoring).

**What the source withholds, and why it matters.** No baseline measurements, no post-migration measurements with denominators, no workload distribution, no cost accounting, no availability definition, no incident artefact, no cutover date, no topology, no configuration, no code, and no mention of any independent audit. Each omission blocks a specific inference. Without a baseline, "reduced cost" cannot be sized. Without a denominator, "improved uptime" cannot be distinguished from a change in measurement. Without a cutover date, before/after comparisons cannot be aligned with external events such as seasonal volume, and this is the single largest threat to any causal reading of the reported benefits.

**Quality caveats specific to this domain.** Telemetry pipelines have a measurement trap that most published case studies fall into: the population being measured is defined by the pipeline itself. If the new path drops a class of records, the measured freshness of the surviving records may improve, because the surviving population is easier to process. A before/after freshness comparison is therefore confounded by any change in the population reaching the sink. The remedy is to compute both metrics over a fixed eligible population defined at the source, not at the sink, which requires the source-side identity registry that A-02 describes and which published narratives almost never have.

**Leakage and temporal validity, in the systems sense.** This case has no train/test split, but it has three leakage analogues that must be controlled or explicitly accepted. (i) Late-arrival leakage: a shadow comparison that freezes its window too early will score the new path as incomplete because records that would have arrived late have not yet arrived; the comparison horizon must exceed the lateness allowance. (ii) Replay-induced distortion: if reconciliation replay runs against the same sink that serves production queries, the comparison work changes the system it is measuring, and both freshness and cost are inflated by the measurement itself. (iii) Configuration drift: if the legacy and new paths are observed at different wall-clock periods, any diurnal or seasonal difference is attributed to the migration. The first is controlled by a horizon, the second by isolation, the third by matched windows or by normalising against the workload curve.

**Exclusions.** This case excludes payments authorisation architecture, fraud detection, merchant risk, the internal design of the observability query engine, and all statements about PayPal's present configuration. It also excludes any inference from the company's line of business to its technology: being a payments company tells us that telemetry loss during an incident is expensive, and it tells us nothing about topic counts or worker sizes.

**Ethical and privacy handling.** Telemetry from a payments environment can contain personal or sensitive data even when it is intended for debugging. This case asserts no specific control for PayPal; it states the requirement (C-01) and treats minimisation at the collector, retention limits, and access logging as design obligations that the responsible owner must source and verify. Any reader importing the design into their own organisation must obtain their own determination.

### 5e. Reproducibility & Method-Selection Ledger

| ID | Claim or decision | Evidence type | Location | Supports | Does not establish | Recheck trigger |
|---|---|---|---|---|---|---|
| SRC-01 | The article exists with the stated authors, date, and content | SOURCED | [S1] | The case's factual base | That the described system is currently operating as described | Publication update or retraction |
| CLM-01 | Most prior pipelines lacked checkpointing and lost data during restarts | SOURCED (author claim) | [S1], challenge section | The problem framing in Sections 2–3 | A loss rate, a loss mechanism, or a statement about Flink generally | Any first-party postmortem |
| CLM-02 | The team ran a proof of concept, moved ingestion from Pulsar to Kafka, and adopted Dataflow | SOURCED (author claim) | [S1], migration section | The intervention described in Sections 8 and 10 | Scope, sequencing, configuration, or completion date | A first-party architecture disclosure |
| CLM-03 | Partitioning was redesigned and shuffling optimised | SOURCED (author claim) | [S1], optimisation section | The scaling discussion in 6.2 | The algorithm, parameters, or effect size | A published engineering detail |
| CLM-04 | Cost fell and stability and uptime improved | SOURCED (author claim, qualitative) | [S1], benefits section | The bounded outcome statement in Section 11 | Any magnitude, denominator, window, or causal attribution | A quantitative disclosure |
| CLM-05 | BigQuery-backed analytics for failed custom interactions and merchant monitoring were initiatives | SOURCED (author claim) | [S1], innovation section | The downstream-use discussion in Section 8 | Deployment completeness or measured value | A first-party delivery note |
| DER-01 | Mean ingest rate, partition capacity, backlog drain, freshness composition, state size, and cost figures in Section 6 | DERIVED (teaching model) | Section 6 | The quantitative walkthrough | Anything about PayPal's actual workload | Any real measurement superseding the model |
| OPN-01 | Whether repair-in-place was formally evaluated and rejected | OPEN | Section 5c | An honest statement of the decision gap | That it was not considered | A first-party decision record |
| OPN-02 | The cutover boundary method (shadow, dual-write, per-partition) | OPEN | Sections 8 and 10 | The cutover design discussion | That any specific method was used | A first-party migration note |
| OPN-03 | Whether the reported availability improvement survives a change of measurement definition | OPEN | Sections 9 and 11 | The caution in Section 11 | That the claim is false | A definition of the uptime metric |

To reproduce the reported results, a third party would need authorised workload traces, an equivalent baseline configuration, both pipeline implementations, recovery drill artefacts, the underlying bills, and the definitions behind each reported benefit. None of that is available. What *is* reproducible in this case is the arithmetic: every formula in Section 6 can be recomputed from its stated inputs with a calculator, and every one is checked by an independent route in the text. That reproducibility is a property of the teaching model, and it is stated as such wherever a number appears.

The method-selection ledger is separate from the evidence ledger because a decision can be reasonable on evidence that is later shown to be weak. The decision to change broker and runner together (CLM-02) is recorded as a reported choice, not as a validated optimum, and the case's own assessment is that its defensibility depends on an evaluation design the source does not describe. A reader who needs to defend a similar decision in their own organisation should be able to point to a scored comparison of the five candidates in 5c, and to a stated reason why the two-variable change was preferred to a sequenced one. That is the artefact this case cannot supply for PayPal and can supply as a template.

## 6. Analysis Walkthrough

### 6.0 The teaching model, stated once

Every quantity in this section belongs to one constructed model, so the model is fixed here and never silently changed. **Evidence label: ILLUSTRATIVE throughout.** The model describes a telemetry estate of 12 billion records per day at a mean payload of 900 bytes, spread across 96 broker partitions, with a diurnal peak at 2.6× the daily mean and a nightly trough at 0.35×. The busiest partition carries 3.125% of records, which is exactly three times the 1.0417% a uniform distribution would give. Per-partition processing capacity is 15,000 records/s and per-partition sink write capacity is 20,000 records/s; both are treated as hard ceilings in the arithmetic and as approximate in reality. Unit prices are illustrative and are stated at the point of use.

The reason for fixing the model before any calculation is methodological. Published observability narratives mix workloads, and a reader who carries an unstated volume in their head will silently mis-scale every derived number. By declaring the volume, the partitions, and the peak factor up front, each result below can be recomputed and each sensitivity can be attributed to a named parameter.

### 6.1 Mean and peak ingest rate

The first calculation converts a daily total into a rate, and then separates mean from peak. Both are needed: the mean sizes the monthly bill, the peak sizes the capacity.

```text
FORMULA [R-01]: lambda_mean = N_day / 86400
FORMULA [R-02]: lambda_peak = k_peak * lambda_mean
WHERE: N_day = records per day [records/day]; 86400 = seconds per day [s/day];
       lambda_mean = mean arrival rate [records/s];
       k_peak = peak-to-mean ratio [dimensionless], 2.6 in this model.
CONSTRAINT: decimal time units; the ratio is a workload property, not a service
            guarantee; the mean is an arithmetic daily average, not a typical second.
NUMERICAL EXAMPLE: N_day = 12e9 records/day.
                   lambda_mean = 12e9 / 86400 = 138888.889 records/s.
                   lambda_peak = 2.6 * 138888.889 = 361111.111 records/s.
INDEPENDENT CHECK: 138888.889 * 86400 = 12,000,000,000 records exactly;
                   and 361111.111 / 138888.889 = 2.600.
INTERPRETATION: capacity sized for the mean sees 260% of design load at peak.
BOUNDARY: at k_peak = 1 (a perfectly flat workload) the distinction vanishes;
          real telemetry is never flat because the services it observes are not.
```

The byte view is the same calculation with a different unit, and it is the view that determines network, broker, and per-GB billing:

```text
FORMULA [R-03]: B_mean = N_day * b / 86400
WHERE: b = mean payload [bytes/record], 900 here;
       B_mean = mean byte rate [bytes/s].
CONSTRAINT: mean payload only; the 900-byte mean hides a long tail caused by
            multi-line stack traces, which inflate p99 payload by roughly 6x.
NUMERICAL EXAMPLE: B_mean = 12e9 * 900 / 86400 = 1.25e8 bytes/s = 125 MB/s.
                   Daily bytes = 1.25e8 * 86400 = 1.08e13 B = 10.8 TB (decimal).
                   At peak: k_peak * 125 MB/s = 325 MB/s.
INDEPENDENT CHECK: 10.8e12 / 900 = 1.2e10 records = 12e9 records, consistent
                   with N_day. Decimal TB used throughout; 10.8 TB = 9.82 TiB.
INTERPRETATION: a plausible-looking 125 MB/s mean becomes 325 MB/s at peak,
                and the p99 record is roughly 5.4 kB rather than 900 B.
BOUNDARY: compression changes bytes on the wire, not records in memory; a 4:1
          compressed wire rate does not reduce parser CPU or state size.
```

### 6.2 Why the aggregate capacity number is a lie

The natural capacity plan divides peak load by aggregate service capacity. That plan is wrong in a specific, predictable way, and the arithmetic shows exactly how wrong.

```text
FORMULA [R-04]: U_agg = lambda_peak / (P * mu_part)
FORMULA [R-05]: U_hot = (lambda_peak * s_hot) / mu_part
WHERE: P = partition count [partitions], 96 here;
       mu_part = per-partition processing capacity [records/s], 15000 here;
       s_hot = share of records on the hottest partition [fraction], 0.03125 here;
       U_agg, U_hot = utilisation [fraction of capacity].
CONSTRAINT: every partition is assumed to receive its key share of the peak load;
            a resharding or a key hot-spot invalidates s_hot immediately.
NUMERICAL EXAMPLE: U_agg = 361111.111 / (96 * 15000) = 361111.111 / 1440000
                   = 0.2508, i.e. 25.1% utilisation.
                   U_hot = (361111.111 * 0.03125) / 15000
                   = 11284.722 / 15000 = 0.7523, i.e. 75.2%.
INDEPENDENT CHECK: uniform share per partition = 1/96 = 0.010417;
                   0.03125 / 0.010417 = 3.000, confirming the 3x statement.
                   Hot-partition load 11284.722 * 15000^-1 = 0.7523 ✓.
INTERPRETATION: a dashboard showing aggregate utilisation reports comfortable
                headroom while the partition that sets user-visible freshness
                is three quarters saturated.
BOUNDARY: if s_hot reaches 0.04167 (i.e. 4x uniform) the hot partition exceeds
          capacity at peak and its backlog grows for the duration of the peak.
SENSITIVITY: at k_peak = 3.0 the hot partition would sit at 86.8%; at
             s_hot = 0.05 it would exceed 100% at the 2.6x peak.
```

This is the single most transferable result in the case. It explains why a team can add workers, watch aggregate utilisation fall, and observe no improvement in the freshness of the noisiest service: the constraint was never aggregate capacity. It also explains why `partition-aware-scaling` is a named pattern rather than a footnote.

### 6.3 Backlog accumulation and drain

Recovery arithmetic has two halves: how much accumulates while a dependency is unavailable, and how long the surplus capacity takes to remove it. Both are conservation statements, and both are routinely misapplied by dividing backlog by service rate without subtracting concurrent arrivals.

```text
FORMULA [R-06]: Q = lambda_incident * t_out
WHERE: Q = accumulated backlog [records];
       lambda_incident = arrival rate during the incident [records/s];
       t_out = outage duration [s].
CONSTRAINT: the processor cannot complete work while the sink is unavailable, so
            arrivals accumulate at full rate; no producer-side shedding assumed.
NUMERICAL EXAMPLE: incident at 2.0x mean, so lambda_incident = 277777.778 rec/s;
                   t_out = 45 min = 2700 s.
                   Q = 277777.778 * 2700 = 750,000,000 records (= 750 million).
                   Bytes: 750e6 * 900 = 6.75e11 B = 675 GB.
INDEPENDENT CHECK: 750e6 / 12e9 = 0.0625 of a day's volume; and
                   0.0625 * 86400 = 5400 s of mean-equivalent traffic, while
                   2700 s at 2x mean is also 5400 mean-equivalent seconds ✓.
INTERPRETATION: 45 minutes of sink unavailability manufactures 750 million
                queued records, which is 675 GB of retention pressure.
BOUNDARY: if producers apply no shedding, Q scales linearly with t_out; a 6-hour
          outage would queue 6.0 billion records and 5.4 TB.
```

```text
FORMULA [R-07]: T_drain = Q / (mu_sink - lambda_r)
WHERE: mu_sink = aggregate sink write capacity [records/s], 96 * 20000 = 1920000;
       lambda_r = arrival rate during recovery [records/s], 277777.778;
       T_drain = drain duration [s].
CONSTRAINT: mu_sink > lambda_r; constant rates; the sink is the only bottleneck
            and does not degrade further under replay pressure.
NUMERICAL EXAMPLE: spare = 1920000 - 277777.778 = 1642222.222 rec/s.
                   T_drain = 750e6 / 1642222.222 = 456.7 s = 7.6 min.
                   End-to-end recovery = 2700 + 456.7 = 3156.7 s = 52.6 min.
INDEPENDENT CHECK: 1642222.222 * 456.7 = 749,999,995 ≈ 750e6 ✓;
                   and 750e6 / 456.7 = 1,642,238, within rounding of the spare rate.
INTERPRETATION: a 52.6-minute recovery for a 45-minute outage is a 16.9% overhead,
                which looks excellent and is entirely a consequence of generous
                spare sink capacity.
BOUNDARY: at mu_sink = 1,000,000 the spare is 722,222 and drain becomes
          1038 s, nearly tripling total recovery to 62.3 min.
```

The recovery overhead ratio is worth isolating because it is the number to quote in a design review:

```text
FORMULA [R-08]: r_overhead = lambda_r / (mu_sink - lambda_r)
WHERE: r_overhead = excess recovery time per unit of outage [dimensionless].
NUMERICAL EXAMPLE: 277777.778 / 1642222.222 = 0.16913, so each outage minute
                   costs 1.169 minutes of recovery.
INDEPENDENT CHECK: 2700 * 1.16913 = 3156.7 s ✓, matching [R-07]'s total.
INTERPRETATION: recovery overhead is a property of the spare-capacity ratio, not
                of the queue length; the queue length only sets the multiplier.
BOUNDARY: as mu_sink approaches lambda_r the ratio diverges; the model stops
          being meaningful before it reaches infinity, because retention expires.
```

### 6.4 The retention test that actually matters

Retention is the assumption that makes replay possible, and it is almost always specified in days and delivered in bytes. The two must be reconciled explicitly.

```text
FORMULA [R-09]: T_eff = C_topic / (lambda_mean * b)
WHERE: C_topic = configured topic capacity [bytes], 20e12 here (20 TB);
       T_eff = effective time retention [s].
CONSTRAINT: size-based retention with a byte cap; time-based retention may be
            shorter, in which case the minimum of the two governs.
NUMERICAL EXAMPLE: T_eff = 20e12 / (138888.889 * 900) = 20e12 / 1.25e8
                   = 160000 s = 44.44 h = 1.852 days.
INDEPENDENT CHECK: 1.852 days * 10.8 TB/day = 20.0 TB ✓.
INTERPRETATION: a topic whose operators believe they configured "72 hours"
                delivers 44.4 hours, because the byte cap binds first.
BOUNDARY: at the peak rate the same 20 TB holds only 38.5% of that window
          (17.09 h), so retention shrinks whenever the workload is busy — precisely
          when an incident is likely.
```

```text
FORMULA [R-10]: t_out_max = (f_margin * T_eff) / (1 + r_overhead)
WHERE: f_margin = fraction of retention reserved for safety [fraction], 0.60 here;
       t_out_max = longest survivable outage [s].
CONSTRAINT: the outage is a complete sink stall at the stated arrival rate;
            partial degradation gives a different, less pessimistic answer.
NUMERICAL EXAMPLE: usable window = 0.60 * 160000 = 96,000 s.
                   t_out_max = 96000 / 1.16913 = 82,113 s = 22.8 h.
INDEPENDENT CHECK: 82113 * 1.16913 = 96,000 s ✓, i.e. outage plus drain exactly
                   consumes the reserved window.
INTERPRETATION: a 45-minute outage is nowhere near the limit; the retention
                design is comfortable for outages and nowhere near adequate for
                a multi-day replay.
BOUNDARY: raising f_margin to 0.90 permits 34.2 h, at the cost of accepting that
          the last 10% of retention is consumed under normal replay demand.
```

### 6.5 The scenario that actually breaks retention: discovered transform defects

Outages are the scenario teams plan for. Defects discovered late are the scenario that exceeds retention, because the required replay window is measured in days rather than minutes and is set by detection latency rather than by outage length.

```text
FORMULA [R-11]: f_recoverable = T_eff / t_replay_required
WHERE: t_replay_required = source history needed to repair the defect [s];
       f_recoverable = fraction of that history still present in the source.
CONSTRAINT: the defect affects every record in the window; no record was
            preserved in an independent archive; T_eff as computed in [R-09].
NUMERICAL EXAMPLE: defect detected 5 days after deployment, so
                   t_replay_required = 5 * 86400 = 432,000 s.
                   f_recoverable = 160000 / 432000 = 0.37037.
                   Unrecoverable fraction = 1 - 0.37037 = 0.62963.
                   Records in window = 5 * 12e9 = 60e9;
                   unrecoverable records = 60e9 * 0.62963 = 37.78e9.
INDEPENDENT CHECK: recoverable records = 60e9 * 0.37037 = 22.22e9, and
                   22.22e9 + 37.78e9 = 60e9 ✓.
INTERPRETATION: 63% of a five-day defect window is permanently unrepairable;
                three of the five days have already been evicted.
BOUNDARY: at a 12-hour detection latency the window is 43,200 s and
          f_recoverable = 160000/43200 = 3.70 > 1, so the replay is fully
          possible — the cliff between safe and unsafe sits near 44 hours.
```

The drain time for the recoverable portion completes the picture, because it determines whether the repair itself becomes a second incident:

```text
FORMULA [R-12]: T_repair = (f_recoverable * N_window) / (mu_sink - lambda_r)
NUMERICAL EXAMPLE: (0.37037 * 60e9) / 1642222.222 = 22.22e9 / 1642222.222
                   = 13,530 s = 3.76 h.
INDEPENDENT CHECK: 13,530 * 1642222.222 = 2.222e10 ✓.
INTERPRETATION: the repair consumes 3.76 hours of the entire sink surplus, during
                which live freshness has no headroom and any new incident has none.
BOUNDARY: reserving only 25% of sink capacity for repair (480,000 rec/s) stretches
          the same repair to 22.22e9 / 480,000 = 46,292 s = 12.9 h.
```

That last boundary is the practical design rule: repair capacity should be a reserved, rate-limited fraction, and its duration should be estimated before the repair is scheduled, not discovered while it runs.

### 6.6 Freshness decomposition and the watermark tax

End-to-end freshness for windowed output decomposes into four additive terms, and the largest one is deliberate.

```text
FORMULA [R-13]: F_p95 = d_ingest + d_proc + L_allowed + d_sink
WHERE: d_ingest = p95 ingest delay [s], 1.8 here;
       d_proc = p95 processing time before emission [s], 0.9 here;
       L_allowed = allowed lateness / watermark delay [s], 120 here;
       d_sink = p95 sink visibility delay [s], 2.4 here.
CONSTRAINT: all terms are p95 of the same eligible population; a sum of p95s is
            an upper bound on the p95 of the sum, not an equality.
NUMERICAL EXAMPLE: F_p95 = 1.8 + 0.9 + 120 + 2.4 = 125.1 s.
                   Watermark share = 120 / 125.1 = 0.9592, i.e. 95.9%.
INDEPENDENT CHECK: non-watermark terms sum to 5.1 s, which is also the freshness
                   of the un-windowed per-record path; 5.1 + 120 = 125.1 ✓.
INTERPRETATION: 96% of windowed-output freshness is a policy choice, not an
                engineering result; optimising the pipeline cannot beat 120 s.
BOUNDARY: for raw record-level output (no window) freshness is 5.1 s, a 24.5x
          difference produced entirely by the aggregation contract.
```

The distinction between the sum of p95s and the p95 of the sum is not pedantry. If the four delay components are positively correlated — which they are during an incident, because the same saturated worker inflates all of them — then the true p95 of the sum exceeds the sum of p95s. The formula is therefore a lower bound on the true tail in the correlated case, which is the case that matters.

### 6.7 The cost of lowering allowed lateness

Shortening the watermark delay is the only lever that materially improves windowed freshness, and it is paid for in corrections. Both sides of that trade are computable.

Given an illustrative arrival-delay distribution in which 0.400% of records arrive more than 60 s after their event time and 0.1333% arrive more than 120 s after it:

```text
FORMULA [R-14]: V_corr = N_day * P(delay > L_allowed)
WHERE: V_corr = records that will trigger a window correction [records/day];
       P(delay > L) = tail mass of the arrival-delay distribution beyond L.
CONSTRAINT: the correction cost is the count of windows that must be re-emitted,
            not the count of late records; the two coincide only when each late
            record lands in a distinct window, which is false in practice.
NUMERICAL EXAMPLE: at L = 120 s, V_corr = 12e9 * 0.001333 = 15,996,000 records
                   affecting an estimated 0.40e6 windows.
                   At L = 60 s, V_corr = 12e9 * 0.00400 = 48,000,000 records
                   affecting an estimated 1.20e6 windows.
INDEPENDENT CHECK: 48.0e6 / 16.0e6 = 3.00, matching the 0.400%/0.1333% ratio;
                   and 1.20e6 / 0.40e6 = 3.00 ✓.
INTERPRETATION: halving allowed lateness removes 60 s from p95 freshness and
                triples the daily correction workload from 0.40 to 1.20 million
                window re-emissions, i.e. from 0.40% to 1.20% of the day's records.
BOUNDARY: at L = 15 s the tail mass rises to roughly 2.1%, giving 6.3 million
          window corrections per day and a downstream load that many consumers
          cannot absorb without versioned reads.
```

The exchange rate is therefore roughly 60 seconds of freshness for 0.80 million additional window corrections per day, or 75 microseconds of freshness per correction. Stated that way, the decision becomes a product conversation: whether a responder's diagnosis is improved more by one minute less latency or harmed more by 800,000 extra correction events per day that consumers must reconcile.

### 6.8 Completeness and duplicate arithmetic

Completeness is a set operation, and row counts are not a substitute for it. A worked window makes the difference concrete.

```text
FORMULA [R-15]: C = |I_src ∩ I_sink| / |I_src|
FORMULA [R-16]: D = (R_sink - |I_sink|) / |I_sink|
WHERE: I_src, I_sink = distinct eligible identities at source and sink;
       R_sink = total sink rows; C = completeness [fraction];
       D = duplicate rate [fraction of sink identities].
CONSTRAINT: identical event-time range, identity scope, and exclusion registry on
            both sides; identities must be stable across producer retries (A-02).
NUMERICAL EXAMPLE: |I_src| = 1,000,000; source rows = 1,000,000.
                   Sink: 998,900 distinct identities, 2,300 extra rows,
                   R_sink = 1,001,200.
                   C = 998,900 / 1,000,000 = 0.99890 (99.890%).
                   D = 2,300 / 998,900 = 0.002302 (0.230%).
INDEPENDENT CHECK: 1,001,200 - 998,900 = 2,300 ✓; missing identities
                   = 1,000,000 - 998,900 = 1,100 ✓; and the row-count ratio
                   1,001,200 / 1,000,000 = 1.0012 would be reported as
                   "100.12% delivered", concealing both errors.
INTERPRETATION: 1,100 records were lost and 2,300 were duplicated, while the
                crude row ratio suggests over-delivery.
BOUNDARY: if producer retries mint new identifiers the duplicate rate is
          understated and the completeness rate is overstated simultaneously;
          the measurement degrades silently in the flattering direction.
```

### 6.9 Why the mean freshness is the wrong number

Averaging freshness across a heavy-tailed population produces a number that describes no user's experience. A constructed distribution shows the scale of the distortion.

```text
FORMULA [R-17]: F_mean = sum_i(n_i * f_i) / sum_i(n_i)
WHERE: n_i = record count in freshness bucket i; f_i = freshness of that bucket [s].
NUMERICAL EXAMPLE: 99,000 records at 3 s; 900 at 12 s; 100 at 900 s.
                   F_mean = (99000*3 + 900*12 + 100*900) / 100000
                          = (297000 + 10800 + 90000) / 100000
                          = 397800 / 100000 = 3.978 s.
                   Cumulative: 99.0% at <=3 s; 99.9% at <=12 s; 100% at <=900 s,
                   so p95 = 12 s and p99.9 = 900 s.
INDEPENDENT CHECK: 99,000+900+100 = 100,000 ✓; and 900/3.978 = 226.2, so the
                   p99.9 experience is 226x worse than the reported mean.
INTERPRETATION: a 3.98 s mean coexists with a 15-minute worst case for exactly
                the 100 records most likely to matter — the ones queued behind a
                saturated partition during the incident under investigation.
BOUNDARY: removing the tail entirely (100 records) changes the mean to 3.08 s,
          a 22% improvement in a metric that no consumer experiences.
```

### 6.10 State size and the cardinality multiplier

Aggregation state is a resource with a size that can be computed, and cardinality is the parameter that makes it explode without any change in event volume.

```text
FORMULA [R-18]: S_state = K * s_entry * W_inflight
WHERE: K = distinct aggregation keys (metric series) [series], 1.2e6 here;
       s_entry = bytes per key per window [bytes], 512 here;
       W_inflight = concurrently open windows per key [windows], 2 here.
CONSTRAINT: excludes timers, index overhead, and backend replication; the true
            resident size is typically 1.5x-2.0x this figure.
NUMERICAL EXAMPLE: S_state = 1.2e6 * 512 * 2 = 1.2288e9 B = 1.229 GB
                   = 1.144 GiB (1 GiB = 1,073,741,824 B).
INDEPENDENT CHECK: 1.2e6 * 1024 = 1.2288e9 ✓; and 1.2288e9/1073741824 = 1.1444 ✓.
INTERPRETATION: modest, and easily held in memory — which is exactly why the next
                calculation surprises teams.
BOUNDARY: at 10x replication or a 4x larger state entry the same key count needs
          49.2 GB, which changes the state backend choice.
```

```text
FORMULA [R-19]: K_new = K * v_new
WHERE: v_new = new distinct values added by a label [values], 12 here (a region label).
NUMERICAL EXAMPLE: K_new = 1.2e6 * 12 = 14.4e6 series.
                   S_new = 14.4e6 * 512 * 2 = 1.47456e10 B = 14.75 GB, a 12x rise.
INDEPENDENT CHECK: 14.75 / 1.229 = 12.00 ✓.
INTERPRETATION: one innocuous-looking label multiplies state, index size, and
                query cardinality by 12, with no change in event volume and no
                change in the number of things being observed.
BOUNDARY: a label that is genuinely unbounded (a request identifier, a payload
          hash, an unnormalised merchant identifier) does not multiply K by a
          constant, it makes K grow with traffic, and no capacity plan survives.
```

### 6.11 Cost arithmetic, with its own limitation stated

Cost is computed per unit because a total is meaningless without the units that produced it. All prices are illustrative and are not vendor quotes.

```text
FORMULA [R-20]: Cost_month = (B_day * 30 / 1e9) * p_compute
                            + (C_topic / 1e9) * p_retain
                            + (S_out_day * 30 / 1e9) * p_store
WHERE: p_compute = $0.055 per GB processed [$/GB];
       p_retain = $0.10 per GB-month of broker retention [$/GB-month];
       p_store = $0.10 per GB-month of sink storage [$/GB-month];
       S_out_day = daily sink storage growth [GB/day].
NUMERICAL EXAMPLE: compute = (10.8e12 * 30 / 1e9) * 0.055
                           = 324000 * 0.055 = $17,820.
                   broker retention = (20e12 / 1e9) * 0.10 = 20000 * 0.10 = $2,000.
                   Sink: 1.2e6 series * (86400/15) samples/day = 6.912e9 samples/day;
                   at 2 B/sample compressed = 13.824e9 B/day = 13.824 GB/day;
                   30 days = 414.72 GB; 414.72 * 0.10 = $41.47.
                   Total = 17820 + 2000 + 41.47 = $19,861.47 per month
                   = $238,337.64 per year.
INDEPENDENT CHECK: 324000 GB * 0.055 = 17,820 ✓ (324000 * 0.05 = 16,200 and
                   324000 * 0.005 = 1,620, summing to 17,820); and
                   6.912e9 / 15-s resolution = 1.2e6 series * 5760 intervals/day ✓.
INTERPRETATION: in this model compute dominates at 89.7% of the bill, retention is
                10.1%, and sink storage is 0.2%.
BOUNDARY: this cost structure is a property of the model, not of observability in
          general. Real observability estates are usually dominated by storage and
          query cost because series are retained for many months at high query
          rates; a model that reports compute dominance is telling the reader about
          its own assumptions.
```

```text
FORMULA [R-21]: Cost_replay = (t_replay_h / 24) * B_day / 1e9 * p_compute
NUMERICAL EXAMPLE: a 6-hour full replay = (6/24) * 10.8e12 = 2.7e12 B = 2700 GB;
                   cost = 2700 * 0.055 = $148.50.
                   A 5-day replay of the recoverable 37.04% = 22.22e9 records
                   = 2.0e13 B = 20000 GB; cost = 20000 * 0.055 = $1,100.00.
INDEPENDENT CHECK: $148.50 / $19,861.47 = 0.00748, i.e. 0.75% of a monthly bill.
INTERPRETATION: replay is cheap in money and expensive in capacity contention and
                in correctness risk; pricing it is the least important part of
                planning it.
BOUNDARY: replay cost is negligible only while the sink has spare capacity. If
          replay must run concurrently with a peak, the true cost is the live
          freshness degradation, which is not a line item.
```

### 6.12 What the team tries first, and why it does not work

The first instinct in every lag investigation is to add processing capacity, and the arithmetic explains precisely why that instinct fails in most of the failure modes catalogued in this case.

If lag is caused by a hot partition ([R-05]), adding workers until aggregate utilisation falls from 25% to 8% changes nothing, because the hot key remains on one partition and one worker owns it. The measured symptom is stable: aggregate utilisation improves, per-partition lag on the hot partition does not. If lag is caused by sink throttling ([R-07] with a reduced mu_sink), adding workers increases the pressure on the constrained sink and can worsen queueing. If lag is caused by an expired retention window ([R-09]), no amount of capacity recovers records that no longer exist. If lag is caused by a watermark held open for late data ([R-13]), capacity is irrelevant because the pipeline is deliberately waiting.

This is presented as reasoning, **not** as a record of what PayPal attempted. The source reports a proof of concept and partition and shuffle optimisation, which is consistent with a team that understood the distinction, but the source does not describe the sequence of attempts or any failed trial.

### 6.13 Sensitivity summary

| Stress | Original condition | Alternative | Effect on conclusion | Evidence | Decision consequence |
|---|---|---|---|---|---|
| Peak factor | 2.6× mean | 3.0× mean | Hot partition rises from 75.2% to 86.8% utilisation; still under capacity | DERIVED [R-05] | Headroom for one more label without resharding |
| Peak factor, extreme | 2.6× mean | 3.5× mean | Hot partition reaches 101.3%: backlog grows during every peak | DERIVED [R-05] | Must reshard or split the hot key before adding dimensions |
| Hot-share | 3.125% | 4.167% | Hot partition at 100.3% at peak even at 2.6× | DERIVED [R-05] | The 4×-uniform threshold is the practical design ceiling |
| Retention | 72 h configured | 44.4 h delivered by byte cap | Survivable outage falls from ~37 h to 22.8 h | DERIVED [R-09], [R-10] | Retention must be measured, not read from configuration |
| Detection latency | 12 h | 5 days | Recoverable fraction falls from 100% to 37.0% | DERIVED [R-11] | Invest in output-side correctness monitoring, not just lag |
| Allowed lateness | 120 s | 60 s | p95 improves 60 s; corrections triple to 1.20 M/day | DERIVED [R-13], [R-14] | Product decision requiring consumer agreement |
| Mean payload | 900 B | 2,000 B | Retention window falls to 20.0 h at the same byte cap | DERIVED [R-09] | Payload growth silently consumes retention |
| Cardinality | 1.2 M series | 14.4 M series | State rises 12× to 14.75 GB | DERIVED [R-18], [R-19] | Label governance is a capacity control |
| Unit price | $0.055/GB | $0.11/GB | Monthly compute doubles to $35,640 | DERIVED [R-20] | Price is the least stable input to any cost claim |

The two rows that change a decision most are the hot-share row and the detection-latency row, because both sit on cliffs rather than slopes. Utilisation near 100% on one partition does not degrade gracefully; it produces unbounded lag for one service while the platform reports health. Detection latency beyond the retention window does not degrade gracefully either; it converts a repairable defect into a permanent data gap. Every other parameter in the table moves a number by a factor of two or three, which is a budget conversation.

### 6.14 Highest-leverage step and the branch that reverses the answer

The highest-leverage step in the whole analysis is measuring per-partition load and per-partition retention rather than their aggregates. It costs one query against broker metadata and one configuration review, and it invalidates or confirms the two cliff-shaped risks in the sensitivity table. A team that performs only that step has captured most of the decision's value.

The branch that reverses the answer is the detection-latency one. If the organisation can detect output defects within a few hours — because it compares aggregates against an independent reference implementation continuously, rather than waiting for a human to notice a dashboard anomaly — then [R-11] never binds, retention sizing becomes an outage-recovery question only, and a shorter retention period is defensible. If detection takes days, then retention must be sized in days and the byte cap must be raised accordingly, or an independent archive of raw records must exist. That single organisational capability, not the choice of runner, determines whether the migration's recovery story holds.

## 7. The Finding

The supported finding is narrower than the headline and more useful: the coauthors report that before the migration most pipelines lacked checkpointing and lost data on restart, that they ran a proof of concept, moved ingestion from Pulsar to Kafka, adopted Dataflow, and redesigned partitioning and shuffling, and that they observed lower infrastructure and operational cost together with improved stability, uptime, and development agility [CLM-01 to CLM-05]. Those are reported claims by interested participants, published without a comparable baseline, without denominators, and without an independent audit, so the case establishes the shape of an intervention and the existence of a problem, not the magnitude of a gain.

The teaching inference that follows is about where correctness lives, and it is supported by the arithmetic rather than by the source. In the teaching model, 95.9% of windowed-output freshness is allowed lateness, not processing speed [R-13]; 99.89% completeness coexists with a 0.23% duplicate rate under a row count that reads as 100.12% delivered [R-15, R-16]; a topic configured for 72 hours of retention delivers 44.4 hours because the byte cap binds first [R-09]; and a defect detected five days late is 63% unrepairable at that retention [R-11]. None of these results depend on the choice of runner. All of them are consequences of contracts the migrating team owns.

The uncertainty is concentrated in three places. First, evidence: no measurement exists for PayPal, so every magnitude claim about the reported benefits is unsupported, and the case does not supply one. Second, attribution: because the broker and the runner changed together, any observed post-migration difference has at least four plausible causes — partition key, commit protocol, late-data policy, parser version — and the published narrative does not describe the comparison design that would separate them [OPN-02]. Third, portability: an observability estate's cost and correctness profile depends on series cardinality, retention months, and query patterns, none of which are disclosed, so the case cannot be transferred to another organisation as a sizing template.

The most plausible competing explanation for the reported improvement is not that Dataflow is inherently better but that the migration forced the team to write down contracts that had never been written down: checkpoint intervals, identity schemes, retention policy, and partition keys. On that explanation, most of the gain would have been available from repairing the existing deployment, and the managed runner contributed operational relief rather than correctness. This explanation cannot be excluded from the published evidence, and a reader who has heard the case summarised as "managed service fixed our telemetry" should treat that summary as unsupported. What can be said is that the migration made the contracts explicit, and that explicitness is where the durable value lies.

The finding therefore ends with an asymmetry rather than a recommendation. Committing to the recovery contract is unconditional: retention measured rather than configured, identity verified rather than assumed, sink effects tested rather than documented, and completeness reported per partition rather than in aggregate. Choosing a particular runner is conditional on operational context, support horizon, and cost shape. An organisation that adopts only the first half of that sentence has captured the value of this case; an organisation that adopts only the second half has bought a vendor relationship and a migration project.

Three observations would change this finding, and each is checkable by a first-party disclosure rather than by argument. First, if the team published a completeness measurement defined at the source, over a matched workload window, showing that the legacy path's completeness was already above the target and that the new path held it, then the migration's correctness value would be established and the competing explanation in the previous paragraph would be refuted. Second, if the reported availability improvement came with a definition — for example, the fraction of eligible event-time windows delivered complete within a stated latency — and a baseline computed the same way, then the claim would become a measurement rather than a direction, and the case would shift from a teaching model to a documented result. Third, if the migration had changed only the execution engine while holding the broker constant, then any observed difference would be attributable to the runner, and the attribution uncertainty that limits this case would disappear. None of those three disclosures exists, and the absence of the third is the most consequential, because it was a choice available before the project started rather than a reporting decision made afterwards.

## 8. System/Solution Design

### 8.1 Reported boundary and what is not claimed

The source describes the path from instrumented applications through a collector layer to a streaming processor and onward to observability storage, with a broker change from Pulsar to Kafka and an execution change to Dataflow, plus BigQuery-backed analytics for failed custom interactions and merchant monitoring. It does not disclose the complete topology, the number of clusters or jobs, the delivery semantics, the region layout, the commit protocol, the state backend, or the schema registry. Nothing in the diagrams below is a reconstruction of PayPal's architecture; they are labelled teaching constructs and their purpose is to make ownership and failure paths visible.

### 8.2 Ownership map

The single most useful design artefact for a pipeline of this kind is not a component diagram but an ownership table, because every failure mode in Section 12 resolves to "nobody owned this boundary".

| Boundary | Artefact owned | Owner | The failure this ownership prevents |
|---|---|---|---|
| Producer to collector | Emission volume, label set, schema version, event identity | Instrumentation / service teams | Identity churn, cardinality explosion, undeclared schema changes |
| Collector to broker | Sampling policy, rejection accounting, backpressure behaviour | Platform team | Silent drops presented as broker health |
| Broker | Partition count, key distribution, retention (measured), access control | Platform team with broker operations | Retention shorter than recovery, hot partitions |
| Broker to processor | Offset semantics, checkpoint interval, watermark policy | Stream processing team | Replay that cannot reach records; windows that finalise early |
| Processor state | State schema, TTL, compatibility across upgrades | Stream processing team | Unrecoverable state after a version change |
| Processor to sink | Output identity, write protocol, duplicate policy, visibility semantics | Platform team with sink owner | Duplicated or partially visible outputs |
| Sink to consumer | Query contracts, completeness metadata, correction and retraction policy | Consuming teams with platform team | Provisional aggregates treated as final |
| Cutover | Per-partition boundaries, single-writer declaration, reconciliation gate | Migration lead, named | Gaps and overlaps at the boundary |

The point of the table is that only two rows change hands when a managed runner is adopted — the checkpoint/watermark row gains a vendor-provided mechanism, and the state row gains a vendor-provided backend — while the other six rows stay exactly where they were. A migration plan that budgets effort only for the two changing rows systematically under-resources the six that decide correctness.

### 8.3 Teaching architecture, labelled illustrative

```text
  applications (Java, Go, Node.js)
        |  emit structured logs, metric points, spans
        v
  instrumented SDK  --->  collector fleet  --->  validation + redaction
        |                     |                        |
        |                     | reject -> quarantine  | (identity preserved)
        |                     v                        v
        |              broker topic (Kafka), 96 partitions, keyed by service
        |                     |
        |            +--------+---------+
        |            |                  |
        |            v                  v
        |      stream processor    bounded reconciler
        |      (event-time windows, (source vs sink identity
        |       watermark, state)    sets, frozen cohort)
        |            |                  |
        |            v                  ^
        |      sink writes              |
        |      (idempotent on output id)|
        |            |                  |
        |            v                  |
        |      observability store -----+
        |            |
        |            v
        |      query surfaces, alert evaluators, dashboards
        |
        +--> raw archive (independent copy, retention >= 2x broker)
```

Four design decisions in that diagram carry most of the correctness, and each deserves its rationale rather than its presence.

The **independent raw archive** exists because broker retention is a byte cap that shrinks under load [R-09]. An archive whose retention is expressed in time and enforced by its own capacity policy is the only defence against a defect discovered after the broker window closed. It is also the cheapest component in the diagram, which makes its frequent absence a governance failure rather than an engineering trade.

The **quarantine path with identity preserved** exists because the alternative to preserving a bad record is discarding it, and discarding it converts a visible bug into an invisible gap. Quarantine must not become a silent graveyard, so its depth and age are production metrics with alarms attached, and its records must be replayable after a parser fix.

**Idempotent sink writes keyed on output identity** exist because the runner's state commit and the sink's write commit are two separate durable events, and the failure window between them is small, permanent, and invisible in aggregate metrics [R-16]. The output identity must be derived from the window and the aggregation key, not from processing time, so that a replay of the same window produces the same identity and therefore overwrites rather than appends.

The **bounded reconciler** exists because a comparison performed against the same sink that serves production queries changes the system it measures. It runs a frozen cohort, writes to an isolated location, and its output is evidence rather than data.

### 8.4 Online path: successful execution

A record is admitted by the collector with a producer-assigned identity, a schema version, and an event time. The collector applies the declared sampling policy and records a sample decision so that the record's absence from the unsampled stream is explainable. The broker assigns it to a partition by key and appends it at an offset. The processor reads from a committed offset, parses under a specific transform version, and advances watermark as event time progresses. When a window's watermark passes its end plus the allowed lateness, the window emits an output record whose identity is derived from the window bounds and aggregation key, and the sink upserts on that identity. The consumer's query joins the aggregate with the completeness metadata for the window.

The invariant that makes this chain trustworthy is stated once and tested everywhere: for every record in the declared eligible population, exactly one output effect exists per declared output identity, or the discrepancy is quantified in metadata. Every component above exists to preserve that invariant, and every component can break it independently.

### 8.5 Online path: failed execution and recovery

The interesting design work is the failure path, so it is traced explicitly rather than summarised.

A processor worker dies after the sink acknowledged a write but before the state checkpoint recorded that acknowledgement. On restart, the processor restores state from the last checkpoint and replays from the corresponding source offset. It re-derives the same window output, produces the same output identity, and the sink upserts. The duplicate is absorbed. Note the three preconditions: the checkpoint must be recent enough that the replay is short, the replay must be possible at all (retention), and the output identity must be deterministic from window plus key (idempotence). Remove any one and the same recovery produces either data loss, an unrecoverable stall, or duplicated effects.

A broker partition becomes unavailable while other partitions advance. The processor's watermark is computed across partitions, so a stalled partition holds the watermark back for windows that span its keys, even though those windows have nothing to do with the stalled keys. This is the mechanism behind the counter-intuitive observation that one unhealthy partition degrades freshness for apparently unrelated dashboards when the watermark is global rather than per-key. The design response is per-key watermarking where the semantics allow it, and a declared partial-coverage state where they do not, so that consumers learn the window is provisional rather than waiting silently.

Source retention expires during a long recovery. No checkpoint can help, because the payload no longer exists at the source. The only recovery is from an independent archive, and if that does not exist the interval is permanently incomplete. The design response is [R-09] and [R-10] used as a hard gate before cutover, plus an alarm on "oldest retained offset age" that fires well before the limit rather than at it.

The sink throttles under replay pressure. Spare capacity disappears, drain time diverges [R-08], and the replay competes with live traffic for the same quota. The design response is an explicit replay rate limit that reserves live capacity, plus a schedule that prefers low-traffic windows rather than assuming replay is free.

### 8.6 Offline path

The offline path has exactly one job: produce evidence that the online path is correct, over a cohort where the answer can be established independently.

It reads a frozen list of identities from the source for an event-time interval whose lateness horizon has closed, reads the corresponding outputs from the sink, and computes [R-15] and [R-16] at partition and key granularity. It recomputes a sample of aggregates with an independent implementation — deliberately a different code path, ideally a different engine, because two invocations of the same defective transform agree perfectly. It records the input snapshot, the transform version, the exclusion registry version, and the cohort definition, so that its result is attributable to a configuration rather than to a date.

The offline path must also compare *values*, not only presence. A pipeline that delivers every identity with a wrong unit will pass an identity reconciliation perfectly. The cheapest effective guard is a dimensional oracle: for each metric family, declare the expected unit and a plausible range, and flag values outside it. Unit and timezone errors survive every structural check and fail this one immediately, which is why the dimensional oracle is worth declaring in a table rather than relying on review.

### 8.7 Operational readiness gates

Before any traffic moves, six gates should be closed, and each has an observable pass condition rather than a documentation artefact:

| Gate | Pass condition | How it fails silently |
|---|---|---|
| Retention sufficiency | Measured oldest-offset age exceeds [R-10] survivable outage with margin, at peak rate | Configured retention is quoted instead of measured |
| Identity contract | A producer retry produces the same identity; a distinct event produces a different one | Identity is minted per emission, so duplicates look like new events |
| Sink idempotence | Injected failure between write and commit leaves no duplicate effect | Timeout is treated as proof of failed write |
| Recovery capacity | Live-plus-replay drain converges with the reserved replay rate | Headroom is computed at mean load |
| Completeness reporting | A deliberately silenced partition raises an alarm within one window | Freshness for zero records reports as zero lag |
| Rollback viability | The legacy path can read current source records end to end | Schema changed, so rollback is technically impossible |

The right-hand column is the case's real warning: none of these gates fail loudly. Each fails by producing a plausible dashboard.

## 9. Evaluation & Validation

### 9.1 Measurement definitions

Every metric below states numerator, denominator, unit, population, window, comparator, and guardrail. None has been measured for PayPal; all are proposed.

**Freshness.** Numerator: for each eligible output, `sink_visible_time − event_time`. Reported as p50, p95, and p99.9 over a fixed window, per partition and per service. Population: records whose event time falls in the window and which appear at the sink. Comparator: the same statistic from the legacy path over a matched workload period. Guardrail: completeness floor, because freshness measured over a population that excludes the hard records is a measure of the pipeline's selectivity rather than its speed.

**Completeness.** Numerator: distinct eligible source identities also present at the sink after the declared lateness horizon. Denominator: distinct eligible source identities. Unit: fraction in parts per million, reported per partition. Population: records not in the exclusion registry. Comparator: target threshold agreed with consumers plus the legacy path's measured baseline. Guardrail: an absent-coverage alarm for partitions emitting nothing.

**Duplicate rate.** Numerator: sink rows minus distinct sink identities. Denominator: distinct sink identities. Unit: fraction. Guardrail: separately report legitimate multiple outputs (for example one span per service hop), because a naive duplicate metric counts them as errors.

**Recovery time.** Incident start to verified drain *and* reconciliation, not job restart to "running". The distinction matters because restart-to-running is measurable automatically and is therefore the number that gets reported, while drain-and-reconcile is the number that describes recovery.

**Recovery completeness.** Completeness computed specifically over the incident's event-time interval. This is the metric that catches a recovery which restores throughput but not data.

**Cost.** Total of compute, broker retention, sink storage, network, support, and labour, normalised per ingested GB and per retained series-month, over comparable workload windows, with replay and incident-driven work itemised separately so that a migration cannot look cheaper by doing less work.

### 9.2 Validation protocol

Stage one, single-component injection, in a non-production environment with production-shaped load: kill a worker mid-window; kill a worker between sink write and commit; revoke sink credentials mid-write; throttle the sink to 50% of nominal; expire a partition's retention artificially and confirm the alarm fires before the data is needed.

Stage two, shadow comparison: run the new path against the same source for a frozen cohort, with writes isolated. Compare identities, values, and completeness at partition granularity. The cohort must be chosen to include the noisiest service, the largest-cardinality series, and at least one service with heavy late arrival, because a mean-shaped cohort hides exactly the failures that matter.

Stage three, staged cutover: move one partition group, reconcile, then the next. Each group has a declared writer. The reconciliation gate is a pass condition, not a report.

Stage four, retirement: remove the legacy path only after the full cohort reconciles and the archive retention is confirmed independently of the broker.

### 9.3 What the reported outcomes can and cannot support

The source's claims are qualitative and directional. "Reduced infrastructure and operational costs" cannot be sized without a baseline bill and a workload normalisation; "improved stability and uptime" cannot be sized without a definition of uptime for a telemetry platform, which is itself an open question (is a pipeline "up" when it processes records, or when it delivers complete windows?). The case therefore treats the claims as evidence that the team believed the migration succeeded, and as evidence that the described operational problems were real enough to motivate a migration. It treats them as no evidence about magnitude.

An observational before/after design would be confounded by at least four factors in this setting: workload growth (a payments platform's telemetry volume is not stationary), seasonal incident load, changes in the measurement surface (new dashboards, new series), and the migration team's own attention, which temporarily improves everything it touches. A rigorous design would need a matched cohort, a stable metric definition, and a comparison window that does not straddle a major incident.

### 9.4 Guardrails

Privacy: minimisation at the collector, redaction before persistence, access logging on quarantine, and a retention limit that the archive also honours. Correctness: the reconciliation gate cannot be waived by schedule pressure, because the schedule pressure is precisely what makes waiving it attractive. Operational: a declared incomplete-coverage state that responders can see, so that a gap degrades gracefully into "we know less" rather than silently into "all is well". Cost: a guardrail on cardinality growth per release, since [R-19] shows that a single label can multiply the bill by twelve without multiplying the volume.

### 9.5 Acceptance rule with numbers, and why the thresholds are not arbitrary

An evaluation that ends in "metrics look good" is not an evaluation. The rule below converts the measurements in 9.1 into a pass or fail, and every threshold is derived rather than chosen.

The completeness floor comes from the retention arithmetic. If the organisation can detect an output defect within 12 hours and the delivered retention window is 44.4 hours, then a completeness floor of 99.9% per partition over a rolling hour is sufficient, because the worst case is bounded: a partition that fails completely loses at most 12 hours of data, which is repairable from the 44.4-hour window. If detection takes five days, no completeness floor over a rolling hour helps, because the loss is not detectable within the repair window; the floor must be paired with an independent archive, and the archive's retention becomes the binding constraint. The floor is therefore always the pair (completeness threshold, detection latency, archive retention), and quoting any one of the three alone is meaningless.

The freshness target comes from the decomposition in [R-13]. For windowed output, a target below `L_allowed + 5.1 s` is unattainable by construction; the engineering terms contribute 5.1 s and no optimisation can remove the policy wait. A team that adopts a 30-second p95 target while running a 120-second watermark has set an impossible target and will spend a quarter discovering that, unless the decomposition is written down first. The defensible target for the teaching model is 130 s (125.1 s plus a 4% margin), and the defensible target for the un-windowed per-record path is 8 s.

The duplicate ceiling comes from the sink's cost per duplicate. For an averaged gauge, a duplicate of 0.1% changes the value by less than the noise; for a count used to size an incident, a duplicate of 1% changes the number a responder quotes. The ceiling is therefore per output family, not per pipeline, and a single pipeline-wide ceiling is a symptom of a team that has not asked what the outputs are used for.

The recovery-time budget comes from [R-08]. If the spare-capacity ratio during recovery is 0.169, then a 45-minute outage recovers in 52.6 minutes, and any budget below that figure is a request for more sink capacity rather than a statement of intent. Writing the ratio into the budget makes the dependency visible: the team is not promising 52.6 minutes, it is promising 1.169 times the outage duration, which is a statement it can defend under questioning.

| Measurement | Threshold in the teaching model | Where the number comes from | What must be true for it to hold |
|---|---|---|---|
| Completeness per partition, rolling 1 h | ≥ 99.90% | Detection latency (≤12 h) inside delivered retention (44.4 h) | Source-side denominator exists; exclusions registered |
| Freshness p95, windowed output | ≤ 130 s | [R-13] decomposition plus 4% margin | Allowed lateness stays at 120 s |
| Freshness p95, per-record output | ≤ 8 s | Engineering terms 5.1 s plus 57% margin | No windowing on that path |
| Duplicate rate, count outputs | ≤ 0.10% | Cost of a duplicate to an incident-size decision | Output identity is deterministic |
| Duplicate rate, gauge outputs | ≤ 1.00% | Below the aggregation noise floor | Aggregation is an average, not a sum |
| Recovery time | ≤ 1.169 × outage duration | [R-08] with measured spare capacity | Sink capacity unchanged during recovery |
| Effective retention | ≥ 2 × detection latency | [R-11] at the 50% recoverable point | Byte cap sized from measured rate |
| Cardinality growth per release | ≤ 10% of series inventory | [R-19] shows unbounded growth is fatal | Budget applies to new labels only |

### 9.6 Measurement bias, and the three checks that catch it

Every measurement in 9.1 can be gamed by the system it measures, and the gaming is usually not deliberate. Three checks catch most of it.

The first is the **population check**. Recompute each metric with the population defined at the source rather than at the sink. If the two definitions produce materially different numbers, the sink-defined version is flattering the pipeline by excluding exactly the records that were hard to process. This check is cheap — one extra join — and it is the single highest-value item in the evaluation design.

The second is the **failing-metric check**. Deliberately break one small part of the pipeline and confirm the metric moves by the amount predicted. Remove one producer's records from a test stream and confirm completeness falls by that producer's share; add a one-hour lag to one partition and confirm freshness p95 for that partition moves by an hour. A metric that cannot be made to fail by a known fault is not measuring what the team thinks it is.

The third is the **owner check**. Ask the person who would be paged by the metric to explain it. If a completeness alarm is routed to a team that does not know what the denominator is, the alarm will be silenced within a quarter, and the metric will be replaced by one that never fires. Measurement design is as much an organisational artefact as a technical one, and a metric without a consumer who understands it is a metric waiting to be muted.

### 9.7 What would constitute strong evidence, ranked

Evidence quality in this domain has a clear ordering, and it is worth stating so that a reader can calibrate any future case study they encounter.

Strongest: a completeness and freshness measurement defined at the source, over a matched workload window, with the same definitions applied to the legacy and new paths, plus the drill artefacts. Second: measured throughput and cost with a workload normalisation, even without the completeness measurement, because it establishes at least that the same work was done. Third: a defined uptime or availability metric applied consistently before and after, which is directionally informative if the definition did not change. Fourth: the qualitative claim that cost fell and stability improved, which is what the source provides. Fifth and weakest: an aggregate volume or throughput figure with no denominator, which is where the three-petabyte figure sits and why it is not used here.

The ranking is not a criticism of the source; it is a tool for the reader. A case study at level four can still be valuable — it establishes that a problem was real and that a migration happened — provided the reader does not upgrade it to level one in their memory. Most of the damage in this domain is done by that upgrade, which happens naturally because a headline is easier to remember than its qualifications.

## 10. Action Taken

**Labelled teaching recommendation, not a report of PayPal's actions.** The source reports the intervention at the level of "proof of concept, Pulsar to Kafka, Dataflow, partition and shuffle optimisation". The sequence below is what the analysis in Sections 6 and 9 implies a team should do, and it is presented as a plan a reader can execute.

Step one, before touching anything: measure the two cliff-shaped quantities. Query the broker for oldest retained offset age per partition, and compute [R-09] against the measured mean and peak byte rate. Query per-partition load and compute [R-05]. These two queries decide whether the migration can safely proceed at all, and they cost minutes.

Step two, write the contracts that currently exist only in behaviour: the event identity rule, the schema compatibility rule, the output identity rule, the late-data policy with its numeric allowance, and the exclusion registry. Each is a short document with a named owner. This step is unglamorous and is the one that most distinguishes a migration that holds from one that does not.

Step three, build the reconciler before the cutover, not after. It needs a frozen cohort definition, an isolated output location, and an independent recomputation path. Its first useful act is to measure the *legacy* path's completeness, because a migration cannot be validated against an unmeasured baseline, and the legacy path's true completeness is usually worse than the team believes.

Step four, run the injection drills in 9.2 stage one against the new path at production-shaped load. Each drill has a pass condition and a recorded artefact. A drill that fails is a finding, not a delay.

Step five, cut over in partition groups with a single declared writer per group and a reconciliation gate between groups.

Step six, retire the legacy path only when the full cohort reconciles, the archive is confirmed, and the completeness dashboard has been reviewed by a consumer who would notice a gap.

The action the case explicitly does **not** recommend is declaring success on aggregate freshness and cost. Both metrics improve under population loss, which is the failure the case is written to prevent. A team that reports only those two numbers has built a dashboard that cannot fail, which is a different and worse problem than a dashboard that is wrong.

### 10.1 Action ledger with owners and exit criteria

A plan without an owner per step is a wish, and a migration has enough parallel work that an unowned step quietly becomes nobody's. The ledger below is the operational form of the six steps.

| # | Action | Owner | Exit criterion | Cost if skipped |
|---|---|---|---|---|
| 1 | Measure per-partition load and per-partition oldest retained offset age | Platform, with broker operations | Two numbers per partition, at mean and at peak, recorded in the migration document | The two cliff risks stay invisible until they fire |
| 2 | Publish the event identity and output identity rules | Instrumentation owners with platform | A producer retry produces one identity; a replay produces one output effect | Duplicates look like new events; completeness measures the wrong thing |
| 3 | Fix the offset-commit ordering relative to sink acknowledgement | Stream processing | Injected commit-gap drill produces no duplicate effect | Silent loss or silent duplication depending on the ordering chosen |
| 4 | Size retention from detection latency, not from a default | Platform with incident command | Archive retention ≥ 2× detection latency, verified by an expiry drill | A five-day-old defect is 63% unrepairable |
| 5 | Build the reconciler against the legacy path before cutover | Migration lead | Legacy completeness measured at partition granularity | No baseline exists, so no post-migration comparison is meaningful |
| 6 | Run the seven drills with recorded artefacts | Platform with incident command | Each drill's pass condition satisfied and stored | Recovery is asserted rather than demonstrated |
| 7 | Declare per-group single-writer boundaries with timestamps | Migration lead | Contiguous identity coverage across every boundary | Gaps and overlaps at the boundary, invisible in aggregate metrics |
| 8 | Review the completeness dashboard with a consumer | Platform with a named consumer | A consumer can explain the denominator and the alarm route | The metric is muted within a quarter |

The ordering is deliberate and it is not the ordering most teams use. The instinct is to provision the new environment and move traffic, and to treat measurement and contracts as documentation that follows. That ordering fails because steps 1 through 5 exist to produce the numbers that make the cutover decision defensible, and once traffic has moved, the legacy comparison becomes progressively harder to construct: schemas drift, the old cluster's retention is repurposed, and the team that knew its quirks is reassigned. The cheapest moment to measure the baseline is before anyone has a reason to change it.

### 10.2 What the team should be prepared to abandon

Three commitments should be explicitly reversible, and naming them in advance is what makes the cutover decision honest rather than a point of no return.

First, the cutover schedule. If step four's drills fail, the correct action is to delay, and a schedule that cannot absorb a delay converts a technical finding into a pressure to waive the gate. Second, the two-variable scope. If step five's reconciliation shows a discrepancy that cannot be attributed within the comparison window, the correct action is to fall back to the staged design — new runner on the legacy broker, or new broker with the legacy runner — rather than to keep two unexplained variables in play. Third, the estate coverage. If one class of telemetry (for example high-cardinality trace assembly) proves materially riskier than the aggregate path, the correct action is to leave it on the repaired legacy path for a later wave, which is the "split the estate" candidate from Section 5c.

What should **not** be abandonable is the reconciliation gate. Every other element of the plan can be resized, resequenced, or descoped without changing the case's central claim; removing the gate removes the only mechanism that can distinguish a successful migration from a migration that merely looked successful.

## 11. Outcome & Results

**What the source reports.** Dataflow reduced infrastructure and operational costs; critical-pipeline stability and uptime improved; development productivity and operational agility improved; the migration followed a proof of concept, a Pulsar-to-Kafka ingest change, and partition and shuffle optimisation; BigQuery-backed analytics for failed custom interactions and merchant monitoring were pursued as follow-on initiatives [CLM-02 to CLM-05]. The reported log estate is approximately three petabytes per day across Java, Go, and Node.js applications.

**What that establishes.** A migration of this shape was executed and was regarded as successful by its participants. The operational problems that motivated it — restart-time loss, manual clusters, peak provisioning, unsupported binary, end-of-sale exposure — were real enough to justify a multi-step platform change. The described architecture is consistent with a team that understands the difference between aggregate and per-partition capacity, since partition redesign is explicitly named.

**What it does not establish, item by item.** No cost figure exists, and no cost denominator; a lower bill with a smaller processed population is not a saving. No uptime definition exists, so "improved uptime" cannot be compared across paths. No baseline window is published, so the comparison cannot be placed on a workload curve. No uncertainty is reported, and no independent audit. No incident or drill artefact is described, so recovery is asserted rather than demonstrated. The three-petabyte figure describes application log generation, not pipeline input, not peak, and not per-partition load, and it is not used anywhere in this case's arithmetic for that reason.

**A bounded statement a reader can defend.** "The reported migration addressed a documented recovery weakness in a self-managed deployment by consolidating ingestion and execution onto a managed runner, and the participants report operational and cost benefits without publishing the measurements that would size them. The correctness properties that matter — retention sufficiency, identity stability, sink idempotence, per-partition completeness — are owned by the migrating team regardless of runner choice, and the published narrative does not describe how they were verified."

**Why the difference between those two statements matters professionally.** The first is a headline and is unfalsifiable. The second is a claim that survives a hostile reviewer, and it is also the claim that transfers to another organisation: it names the properties to test rather than the product to buy. Interviewers who have run observability platforms ask the second kind of question, because they have seen a migration declared successful and then watched a partition silently drop for a week.

**The counterfactual the case cannot resolve.** If the team had repaired the existing deployment — added checkpointing, automated cluster lifecycle, replaced the unsupported build — would the reported benefits have appeared anyway? The most plausible reading of the evidence is that a substantial part would have, because the reported problem is a missing recovery contract rather than insufficient execution capacity, and a recovery contract is portable across runners. The part that would not have appeared is the reduction in operational surface area, which is real and is a legitimate reason to migrate. The case therefore supports "the migration was reasonable" without supporting "the migration was necessary for correctness".

### 11.1 Outcome claims decomposed against what each would require

The table below takes each reportable element of the source's outcome narrative and states what a defensible version would need. Its purpose is not to demand that a vendor blog become a paper; it is to give a reader a checklist that transfers to any internal migration review they will ever attend.

| Reported element | What it asserts | What a defensible version needs | Available in [S1] | How the case uses it |
|---|---|---|---|---|
| "Reduced infrastructure cost" | Total cost of the path fell | Baseline bill, workload normalisation, replay and incident cost itemised | No | Not sized; treated as directional only |
| "Reduced operational cost" | Labour per unit of telemetry fell | Labour measurement or a proxy such as on-call hours and manual interventions | No | Not sized; the mechanism (managed provisioning) is credible and portable |
| "Improved stability" | Fewer failures or better behaviour under failure | Failure-rate definition, incident count, and equal observation windows | No | Not sized; the checkpoint mechanism is consistent with the claim |
| "Improved uptime" | Availability rose | An uptime definition for a telemetry platform and a matched comparison | No | Not sized; the definition is itself an open question [OPN-03] |
| "Improved development productivity" | Delivery throughput rose | A delivery measure such as lead time, with a pre-migration baseline | No | Not sized; noted as a secondary claim |
| "~3 PB of logs per day" | Emission volume | Petabyte convention, window, and the fraction entering the pipeline | Partially — the number exists, the qualifications do not | Not used in any calculation |
| Partitioning redesigned, shuffling optimised | A specific technical intervention | Partition key policy, counts, and the measured effect on skew | No | Used as evidence the team understood per-partition capacity |
| BigQuery FCI analytics | Downstream innovation | Deployment completeness and measured value | No | Treated as an initiative, not a result |

The table's most useful row is the fourth. "Improved uptime" for a telemetry platform cannot be evaluated without first answering a question the industry has not settled: is the platform up when it is processing records, when it is delivering complete windows, or when its queries return a correct answer within an acceptable latency? Each definition produces a different number for the same system, and a migration can improve one while degrading another. A pipeline that processes every record promptly but publishes windows with a two-minute watermark delay has excellent availability under the first definition and poorer availability under the second, with no change in behaviour. That is why the case insists on the completeness-and-freshness pair rather than on availability as a single figure, and why [OPN-03] is recorded as open rather than resolved by inference.

### 11.2 The one number that would have settled the case

If a single measurement could have been published, the most valuable would be completeness defined at the source, at partition granularity, for a matched window, on both paths. It would settle the question the case actually cares about — whether the new path delivered the same population — and it would simultaneously validate or invalidate the reported cost and stability improvements, because both are conditional on the same work having been done. Freshness percentiles would be second, and a bill normalised by ingested bytes third.

The reason this one number dominates is structural: in a telemetry migration, every other metric is downstream of the population. A cheaper path that processes less is not cheaper. A faster path that processes easier records is not faster. A more stable path that fails by silently filtering is not more stable. The population is the unit of account, and any evaluation that does not fix it first is measuring the pipeline's preferences rather than its performance.

## 12. Failure Modes & What Went Wrong

The failure the source actually reports is loss during restarts when most pipelines lacked checkpointing. Everything else in this section is a constructed diagnostic, labelled as such, drawn from the mechanisms in Section 6 rather than from any reported incident. Each entry states competing causes, a discriminating check that separates them, a bounded fix, and a verification that the fix worked — because a fix without verification is a hypothesis.

### 12.1 Restart loss with no checkpoint

**Symptom.** After a worker or job restart, output volume for a window is lower than the corresponding source volume, with no error and no alarm.

**Competing causes.** (i) No checkpoint at all, so state is rebuilt empty and in-flight windows are lost. (ii) A checkpoint exists but the source offset is committed independently of it, so replay starts after the lost records. (iii) The checkpoint restores correctly but the window's trigger policy discards partial windows on restore. Three different fixes, one symptom.

**Discriminating check.** Compare the restore point's source offset with the last offset whose outputs reached the sink. If the offset is ahead of the last emitted output, cause (ii) applies. If they align but the window is empty, cause (iii). If no restore point exists, cause (i).

**Bounded fix.** Configure interval checkpoints with a retention policy that survives the expected restart count; bind the offset commit to the checkpoint rather than to processing; and set window triggers so that a restored partial window can still fire. Bound the work by fixing only the checkpoint/offset binding for one job first, then rolling to the estate.

**Verification.** Kill the job mid-window in a controlled test, restore, and reconcile identities across the killed window [R-15]. The pass condition is completeness above the agreed floor for that window, not "the job is running".

### 12.2 Offset committed before sink effect

**Symptom.** No lag, healthy pipeline, but consumers report missing aggregates for a short interval around a deployment.

**Competing causes.** (i) The commit precedes the sink write, so a crash between them loses the record permanently while lag stays at zero afterward. (ii) The sink write succeeded but the output identity collided with an existing one and was overwritten by an older value. (iii) The sink write landed in a partition the consumer does not query.

**Discriminating check.** For the affected interval, compare committed offsets with sink row versions: cause (i) shows no sink rows for identities whose offsets were committed; cause (ii) shows rows whose version regressed; cause (iii) shows rows present but absent from the queried partition's results.

**Bounded fix.** Move the commit after the sink acknowledgement and make the write idempotent on output identity so that the replayed record upserts rather than duplicates. This is a two-line ordering change with a large semantic consequence, which is why it deserves a drill rather than a code review.

**Verification.** Inject failure precisely in the window between write and commit, then confirm that replay produces exactly one effect per output identity and that no identity in the interval is absent.

### 12.3 Hot partition masked by aggregate utilisation

**Symptom.** Aggregate capacity looks comfortable; one service's dashboard is consistently ten minutes behind during peaks; adding workers changes nothing.

**Competing causes.** (i) A hot key concentrates load on one partition [R-05]. (ii) A single upstream service produces a disproportionate share because it emits a metric per request. (iii) The sink partitions on the same key and is slow for that key's volume.

**Discriminating check.** Plot per-partition ingress and per-partition processing rate. In case (i) one partition's ingress is near capacity while peers idle; case (ii) shows the same pattern with a single dominant producer identity; case (iii) shows the processor keeping up while sink write latency for that key rises.

**Bounded fix.** If the key is genuinely required for ordering, the fix is to split the key by a subordinate dimension with an explicit ordering contract per sub-key, and to document that cross-sub-key order is not preserved. If ordering is not required, change the key. Adding partitions helps only if the key distribution is re-derived, since a hot key remains hot across a larger partition space.

**Verification.** After the change, recompute [R-05] from measured shares and confirm the busiest partition's utilisation at peak is below the design ceiling, then confirm that the affected service's p95 freshness returns to the population's p95.

### 12.4 Retention shorter than the defect-detection latency

**Symptom.** A transform defect is found days after deployment; replay is attempted; the earliest affected records are gone.

**Competing causes.** (i) The byte cap binds before the time limit [R-09]. (ii) A partition's retention clock advanced while it was not consumed, so its oldest record is much younger than the configured period. (iii) The topic was recreated during a deployment, resetting retention entirely.

**Discriminating check.** Query the oldest retained record's age per partition and compare it with the configured period. All three causes produce a gap between configured and actual, and the per-partition distribution distinguishes them: (i) affects all partitions uniformly at high load, (ii) affects specific partitions, (iii) shows a discontinuity at the deployment timestamp.

**Bounded fix.** Size retention from the measured detection latency, not from a default; add the independent raw archive with time-based retention; and alarm on oldest-offset age with a threshold well below the limit.

**Verification.** Run an expiry drill: deliberately let one partition's oldest record age past the alarm threshold and confirm the alarm fires, then confirm the archive can supply the same identities.

### 12.5 Late data finalised early

**Symptom.** A window is finalised and then a consumer sees a different value for the same window later, or sees the value change after the dashboard was screenshotted during an incident.

**Competing causes.** (i) Allowed lateness is shorter than the actual arrival-delay tail [R-14]. (ii) A watermark advanced on ingestion time rather than event time, so a producer with a delayed clock appears late. (iii) A producer's retries arrive after the horizon and are counted as fresh events rather than as the same event.

**Discriminating check.** Compare the arrival-delay distribution against the configured allowance; compare event time against ingestion time for the affected producers to separate (ii); compare identities between the original and the late arrival to separate (iii).

**Bounded fix.** Set the allowance from the measured tail at a declared quantile, publish windows as provisional until the horizon closes, and make the correction mechanism versioned so consumers can distinguish a correction from a new fact.

**Verification.** Replay a deterministic out-of-order trace containing records at 1.01× and 0.99× the allowance and confirm exactly the expected set of windows is corrected and no others.

### 12.6 Duplicates hidden by an averaging query

**Symptom.** Nobody notices anything. A count-based alert fires late, once, and the postmortem reveals that a metric had been double-counted for a week.

**Competing causes.** (i) Replay after an unacknowledged write duplicated the output. (ii) Two writers were active during a cutover overlap. (iii) The output identity is not deterministic — it includes processing time — so a replay creates a new identity rather than replacing the old one.

**Discriminating check.** Query sink rows per output identity for the affected window. Cause (i) shows two rows with different write timestamps and the same value; cause (ii) shows two writers distinguished in metadata; cause (iii) shows two identities that differ only in a timestamp field.

**Bounded fix.** Derive the output identity from window bounds plus aggregation key only, and make the sink upsert. Where the sink cannot upsert, write to an append-only table with the identity as a deduplication key at query time and document that the raw table contains duplicates.

**Verification.** Inject the write-commit failure, then assert that the count of distinct output identities equals the count of logical windows and that no identity has more than one effective value.

### 12.7 Freshness reports zero while a partition is silent

**Symptom.** A dashboard shows excellent latency for a service whose telemetry stopped entirely.

**Competing causes.** (i) Freshness is computed over arriving records only, so an empty population produces no samples and the last computed value is retained. (ii) A producer was decommissioned and its silence is expected. (iii) The partition's records are arriving but failing validation and being quarantined, so the sink sees nothing while the broker sees traffic.

**Discriminating check.** Compare broker ingress per partition with sink ingress per partition. Case (i) shows zero on both; case (iii) shows broker ingress with a rising quarantine count; case (ii) is resolved by consulting the producer registry rather than by any metric.

**Bounded fix.** Add an explicit coverage metric — expected sources versus reporting sources — with an alarm on absence, and make freshness queries return "no data" rather than the last value. This is a query-layer change, not a pipeline change, and it is the highest-value fix in this section.

**Verification.** Deliberately stop one producer in a test environment and confirm the coverage alarm fires within one window while the freshness panel shows an explicit no-data state.

### 12.8 Schema change breaks parsing silently

**Symptom.** Output volume for one metric family drops by a fraction that looks like organic variation; no error rate moves.

**Competing causes.** (i) A new optional field was added and the parser's strict validation rejects records that omit it. (ii) A field's type changed from string to number and the parser coerces it to a default. (iii) A renamed field leaves the old name absent, so a dimension becomes null and series collapse.

**Discriminating check.** Group rejection and null rates by schema version. Cause (i) shows rejections concentrated on the new version; cause (ii) shows a value distribution shifted to a suspicious default (zero or empty); cause (iii) shows series count falling while record count holds.

**Bounded fix.** Require backward-compatible schema evolution with an explicit registry check, make parsers fail loudly on unknown required fields and tolerate unknown optional ones, and treat a null dimension as a validation failure rather than a value.

**Verification.** Deploy a reader against a synthetic future version containing an added optional field and a removed field, and confirm the reader either parses correctly or quarantines with a specific reason — never silently coerces.

### 12.9 Cardinality explosion from one label

**Symptom.** State size, index size, and cost rise sharply with flat event volume; query latency degrades; some dashboards time out.

**Competing causes.** (i) A new label with bounded but large cardinality [R-19]. (ii) A label whose values are unbounded, such as a request identifier or a payload hash. (iii) A label whose values are high-cardinality in practice because an upstream system encodes an identifier into a "region" field.

**Discriminating check.** Measure distinct series per label per day and fit the growth: (i) plateaus after the rollout, (ii) grows linearly with traffic forever, (iii) grows in steps that correlate with an upstream release.

**Bounded fix.** Impose a cardinality budget per service with a review path for exceptions; require aggregation at the producer for unbounded fields; and keep a small, queryable sample of high-cardinality detail outside the metric path.

**Verification.** Recompute K and S_state [R-18] after the fix and confirm the growth curve flattens, while confirming that the diagnostic capability the label was added for is still available through the sampled detail path.

### 12.10 Migration overlap creates a double-write interval

**Symptom.** Post-cutover totals for a period exceed both paths' individual totals.

**Competing causes.** (i) Both paths wrote during the overlap and the sink accepted both. (ii) The new path replayed a range the legacy path had already delivered. (iii) The sink's upsert key differs between paths, so identical logical outputs have different physical identities.

**Discriminating check.** Compare output identities from both paths for the overlap interval. Cause (i) shows two distinct identities with the same logical value; cause (ii) shows one identity twice with different write times; cause (iii) shows identity fields whose composition differs, which is visible only if both identities are logged.

**Bounded fix.** Declare a single writer per partition group at all times, and treat the declaration as a controlled artefact with a timestamp rather than an operational understanding. Reconcile the overlap interval explicitly before removing either path.

**Verification.** Recompute [R-16] over the overlap and confirm the duplicate rate is zero for the declared single-writer interval, and that the interval boundaries are contiguous with no gap.

### 12.11 Replay saturates the sink and degrades live freshness

**Symptom.** A repair is running; live dashboards degrade; responders cannot tell whether the degradation is the original incident or the repair.

**Competing causes.** (i) Replay shares the sink's capacity without a reservation [R-12]. (ii) Replay triggers alert evaluation on historical data, consuming evaluator capacity and producing false pages. (iii) Replay writes to the same series that live queries read, so query cost rises alongside write cost.

**Discriminating check.** Compare live throughput, replay throughput, and sink queue depth in a three-way plot. Cause (i) shows live throughput falling as replay rises; cause (ii) shows alert evaluations spiking with event times in the past; cause (iii) shows query latency rising with no change in write latency.

**Bounded fix.** Reserve a fixed replay rate and cap it; tag replay writes so alert evaluation can exclude them by event time; write replays to a separate series namespace and merge at the query layer with an explicit precedence rule.

**Verification.** Run the repair at the reserved rate against production-shaped load and confirm live p95 freshness stays within its normal band, replay drains monotonically, and no historical alert fires.

### 12.12 Completeness measured against a population defined by the pipeline

**Symptom.** The completeness metric reads 99.99% during a week when a whole service's telemetry was missing.

**Competing causes.** (i) The denominator is sink identities rather than source identities, so absent records are not counted [R-15]. (ii) The service was excluded from the eligible population by an undocumented rule. (iii) The service's records were sampled to zero by a policy that reports sampling as a success.

**Discriminating check.** Recompute completeness with the denominator taken from a source-side registry that the pipeline cannot influence. Any material difference identifies (i). Then diff the exclusion registry against the producer registry: cause (ii) shows an entry that no one remembers adding, and cause (iii) shows a sampling rate of zero with no alarm.

**Bounded fix.** Define completeness at the source, maintain the exclusion registry as a reviewed artefact with owners and expiry dates, and alarm on any sampled-to-zero population.

**Verification.** Remove one producer's records from a test stream while leaving its registry entry intact, and confirm the reported completeness falls by that producer's share — the metric must be able to fail, or it is decoration.

### 12.13 Rollback that cannot read the new data

**Symptom.** Cutover encounters problems; the team decides to roll back; the legacy path cannot parse records produced since the new schema was deployed.

**Competing causes.** (i) The new producer emits a schema version the legacy reader predates. (ii) The legacy path reads from the retired broker. (iii) The legacy path is technically able to read but its state was purged during the transition.

**Discriminating check.** Attempt a controlled read of a current record by the legacy path before cutover. All three causes fail this test, and the failure message distinguishes them: a decode error indicates (i), a connection error indicates (ii), and a successful read with wrong state indicates (iii).

**Bounded fix.** Either maintain dual-format emission for the migration window or declare explicitly that the cutover is forward-fix-only and resource the team accordingly. The second option is legitimate; the illegitimate version is assuming rollback works without testing it.

**Verification.** Execute an actual rollback in a staging environment with records produced under the new schema, and confirm the legacy path's outputs reconcile for the interval.

### 12.14 Telemetry path loses its own telemetry

**Symptom.** During an incident, the platform cannot report its own health; lag dashboards are themselves stale.

**Competing causes.** (i) The platform's self-metrics share the pipeline they measure, so a failure removes the measurement. (ii) The self-metric series exceeded cardinality limits and were dropped. (iii) The self-metrics are emitted but the query surface depends on the same degraded sink.

**Discriminating check.** Compare an out-of-band health check — a synthetic record injected and followed end to end by an independent observer — with the platform's self-reported health. Divergence identifies (i) or (iii); cause (ii) shows self-metric series absent from the catalogue.

**Bounded fix.** Maintain a minimal, low-cardinality, independently stored health signal and a synthetic end-to-end probe. This is the "who watches the watchers" requirement, and it is the one most often deferred because it has no user-visible feature.

**Verification.** Deliberately degrade the main pipeline and confirm the synthetic probe reports failure while the platform's own dashboards are unavailable — the probe must not depend on the system it monitors.

### 12.15 Cost falls because the population shrank

**Symptom.** The monthly bill drops after migration and the team reports success; months later, a dashboard is found to have been missing a service since cutover.

**Competing causes.** (i) A filter predicate introduced in the new transform excludes a class of records. (ii) A producer's sampling rate was raised during the migration and never documented. (iii) The new path's cardinality policy aggregates away a set of series that were previously distinct.

**Discriminating check.** Compare ingested bytes and emitted series counts before and after, normalised by producer registry size. All three causes reduce volume, and the pattern separates them: (i) reduces records but not series; (ii) reduces records uniformly across one producer; (iii) reduces series while records hold.

**Bounded fix.** Make the exclusion registry the only sanctioned mechanism for reducing volume, require a change record per entry with an expiry, and add a volume-per-producer guardrail that alarms on unexplained drops.

**Verification.** Recompute per-producer record and series counts for a matched window and confirm that every delta is explained by a registry entry; confirm that deleting the entry restores the population.

### 12.16 Recovery validated only by restart

**Symptom.** Recovery drills pass. The next real incident does not.

**Competing causes.** (i) The drill tested restart, not drain-and-reconcile [Section 9]. (ii) The drill ran at low load, so spare capacity was abundant and [R-08] looked favourable. (iii) The drill restored state but never verified that outputs after restore matched the pre-failure population.

**Discriminating check.** Compare the drill's pass criterion with the metric definitions in 9.1. If the criterion is "job running", cause (i) applies. If the drill's traffic was below peak, cause (ii). If no reconciliation step exists in the drill script, cause (iii).

**Bounded fix.** Redefine the drill's pass condition to include completeness over the affected interval at peak-shaped load, with a reconciliation step that runs as part of the drill and produces an artefact.

**Verification.** Re-run the drill with an artificial bottleneck that halves sink capacity, and require the reconciliation to pass. A drill that passes only with generous headroom has tested the headroom.

### 12.17 Failure-mode register summary

| Failure | Primary signal | The check that discriminates | Owner |
|---|---|---|---|
| Restart loss | Window output below source volume, no error | Restore offset versus last emitted output | Stream processing |
| Commit-before-write | Zero lag with missing aggregates | Committed offsets versus sink row versions | Stream processing |
| Hot partition | One service persistently behind | Per-partition ingress and rates | Platform |
| Retention shortfall | Replay cannot reach the defect window | Oldest retained age per partition | Platform with broker ops |
| Early finalisation | Window values change after publication | Arrival-delay tail versus allowance | Stream processing with consumers |
| Hidden duplicates | Count-based alert fires late | Rows per output identity | Sink owner |
| Silent partition | Freshness looks excellent | Broker versus sink ingress, coverage metric | Platform |
| Schema drift | Volume falls, no error rate moves | Rejections and nulls by schema version | Instrumentation |
| Cardinality explosion | Cost and state rise, volume flat | Distinct series per label per day | Service owners |
| Overlap double-write | Totals exceed both paths | Output identities across the overlap | Migration lead |
| Replay contention | Live degradation during repair | Live, replay, and queue depth together | Platform |
| Population-defined completeness | Completeness near perfect during a real gap | Source-side denominator | Platform |
| Unusable rollback | Rollback cannot parse current records | Controlled legacy read before cutover | Migration lead |
| Blind platform | Self-health unavailable during incident | Out-of-band synthetic probe | Platform |
| Shrinking-population saving | Bill falls, consumption falls with it | Per-producer volume normalisation | Finance with platform |
| Hollow drill | Drill passes, incident does not | Drill pass criterion and load shape | Incident command |

The register's value is the middle column. Every row has a signal that is ambiguous alone and a check that becomes decisive when paired with it. A team that builds only the signals builds a set of dashboards; a team that builds the checks builds a capability.

## 13. Transferable Lesson

The transferable lesson is that in streaming systems the unit of reliability is the entire path — source acceptance, durable retention, recoverable state, repeatable sink effect, and consumer interpretation — and that no component in that path can be certified by its product label. A managed runner can guarantee its own state consistency and still permit duplicated outputs, because the sink commit is outside its scope. A broker can guarantee durable append and still deliver a retention window shorter than the recovery it is expected to support, because the byte cap binds before the time limit. A watermark can guarantee a bounded wait and still finalise a window before its late records arrive, because the bound is a policy parameter rather than a property of the data.

The second transferable element is the arithmetic discipline that follows from taking the first seriously. Three quantities decide most streaming designs and all three are cheap to compute: per-partition utilisation at peak [R-05], effective retention measured rather than configured [R-09], and the spare-capacity ratio during recovery [R-08]. A design review that produces those three numbers has covered more risk than one that produces a component diagram, and it takes an afternoon.

The third element is the honesty requirement that comes from measurement design. A completeness metric whose denominator is defined by the pipeline is a metric that cannot fail, and a freshness metric computed over surviving records improves when the pipeline drops records. The general form of the rule is: any metric whose population is defined downstream of the system being measured will flatter that system as it degrades. The remedy is always the same shape — define the population at the source, report per partition, and add an explicit absence signal.

**When the lesson does not apply.** A stateless, disposable diagnostic feed with formally accepted loss needs no recovery contract, and the investment this case recommends would be waste. A pipeline whose sink effects are trivially reversible — a cache refresh, a preview render — can tolerate at-least-once semantics without deduplication, because the cost of a duplicate is bounded by a recomputation. A pipeline whose consumers only ever ask "is anything arriving" does not need [R-15] at all. The lesson's precondition is that a missing or doubled output would change a human decision, which is true of incident telemetry and false of many other streams.

**What changes in a different domain.** In order-processing, the same five-artefact contract appears but the cost asymmetry inverts: a duplicate effect is a double charge, so idempotence moves from a correctness nicety to a hard requirement, and reconciliation must be per-transaction rather than statistical. In sensor fusion, event time is dictated by device clocks, so clock synchronisation becomes a first-class design component rather than a diagnostic footnote. In financial reporting, the correction mechanism becomes a governed process with an audit trail, because a restated window is a disclosure event rather than a dashboard refresh. In each case the mechanism is the same and the constraint that dominates is different — which is why the case teaches a contract rather than a configuration.

**The organisational lesson, which is the one that survives job changes.** The reported baseline persisted with missing checkpoints not because nobody knew that checkpointing exists, but because the cost of adding it was visible in a roadmap and the cost of not having it was invisible until an incident. That asymmetry is the reason recovery work loses prioritisation arguments and the reason the argument has to be reframed: retention, identity, idempotence, and completeness reporting are not reliability improvements to be scheduled after features, they are the preconditions that make the platform's numbers interpretable at all. A platform whose data may be incomplete cannot tell you whether anything else is working.

### 13.1 A portability test a reader can run in an hour

The lesson is only useful if it can be applied before a migration rather than after. The test below is designed for that: nine questions, each answerable from configuration and metadata in under ten minutes, each with a specific consequence if the answer is unsatisfactory.

| # | Question | Where the answer lives | Consequence of a bad answer |
|---|---|---|---|
| 1 | What is the oldest retained record's age, per partition, at peak load? | Broker metadata, not configuration files | Recovery window is shorter than believed; replay may be impossible |
| 2 | What is the busiest partition's share of peak load? | Broker ingress metrics by partition | Aggregate capacity is misleading; one service will lag first |
| 3 | Does a producer retry mint a new identity? | Producer SDK source or a controlled retry test | Duplicates are indistinguishable from new events; completeness is overstated |
| 4 | Is the offset commit after the sink acknowledgement? | Connector or transform code, not documentation | Silent loss in the commit window, invisible in lag metrics |
| 5 | Is the output identity derivable from window bounds and key only? | Transform code | Replay appends rather than replaces; the sink accumulates duplicates |
| 6 | How long after an output error would anyone notice? | Monitoring design and on-call runbooks | Retention must be sized in days, not hours |
| 7 | Is completeness computed against a source-side denominator? | Metric definitions | The metric cannot fail, so it certifies nothing |
| 8 | Does an empty partition produce an alarm or a zero? | Query and alert configuration | A silent partition reads as perfect health |
| 9 | Can the legacy path read a record produced under the current schema? | A controlled read test | Rollback is not available, whatever the plan says |

The test's design principle is that none of the nine questions asks about a product. They ask about retained state, key distribution, identity stability, commit ordering, determinism, detection latency, denominator provenance, absence signals, and reverse compatibility — nine properties that any streaming design has or lacks regardless of who operates it. That is what makes the lesson portable, and it is also why a reader can run the test against a design they have not built and still get a useful answer.

### 13.2 The two failure archetypes this case exists to prevent

The first archetype is **the migration that succeeds at the wrong thing**. Every gate passes, cost falls, freshness improves, and the reason is that a filter quietly excluded a class of records. The tell is that the improvement is larger than the mechanism can explain: a managed runner does not make a pipeline three times cheaper by being better at compute, so a threefold cost reduction is evidence of a population change rather than an efficiency gain. Section 12.15 gives the discriminating check, and the check is cheap enough that there is no defensible reason to skip it.

The second archetype is **the drill that tests the environment rather than the failure**. Recovery drills run at low load, with generous headroom, against a warm cache, and they pass because the failure they inject is small relative to the surplus. The tell is that the drill's pass criterion is a state ("the job is running") rather than an outcome ("the affected window reconciled"). Section 12.16 gives the discriminating check: run the drill with an artificial bottleneck that halves sink capacity, and require reconciliation to pass. A drill that only survives generous headroom has tested the headroom.

Both archetypes share a structure worth naming: they are failures of measurement rather than failures of engineering. Nothing in either scenario is built wrong; the wrongness is in what the team chose to look at. That is why this case spends more of its length on denominators, populations, and discriminating checks than on component selection, and why the transferable lesson is stated as a measurement discipline rather than as an architecture recommendation.

## 14. Connections & Cross-References

### 14.1 Related material in this collection

The mapped teaching home is this module's Case_Studies collection, with crosswalk anchors for the streaming source family. Within the collection, [CASE-2026-STR002](02_Flink_CEP_Streaming_Fraud_Detection.md) treats the same recovery contract under a different constraint: its event-time sequence matching requires versioned rule consistency, so that a rule effective at the event's time is applied even after recovery, and its external case effects require idempotence at the case-management boundary rather than at a metric sink. Reading the two together separates the invariant from the domain: both refuse to let a replay re-apply a decision under a changed rule, and both discover that the sink — not the processor — is where exactly-once finally succeeds or fails.

The [collection index](INDEX.md) records this case's evidence limitations and stage. The [course source crosswalk](../../../00_SOURCE_TOPIC_CROSSWALK.md) maps the streaming sources to this module. The authoring contract is [00_AUTHORING_AND_NUMERICAL_STANDARDS.md](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md), which governs the derivation, evidence and question requirements applied here.

### 14.2 Method lineages a reader should be able to name

Event-time windowing with watermarks and allowed lateness comes from the Dataflow model lineage, which is also the programming model the target runner executes; the important inheritance is that a watermark is a heuristic completeness estimate, not a guarantee, so every windowed result is either provisional or accompanied by a lateness assumption. Checkpointed state with replayable sources is the Chandy–Lamport lineage applied to stream processing: consistent global snapshots taken without stopping the world, which is why a checkpoint is a position from which a consistent replay can begin rather than a copy of the world.

The delivery-semantics vocabulary — at-most-once, at-least-once, effectively-once — belongs to a third lineage that is frequently misquoted. Effectively-once is a property of an end-to-end path with idempotent or transactional effects; a processor that advertises exactly-once state does not thereby deliver exactly-once outputs. This case uses "effect multiplicity" rather than "exactly-once" wherever the claim is about outputs, because the latter has been stretched to the point of uselessness in marketing material.

### 14.3 Open questions this case does not resolve

Whether repair-in-place was formally evaluated and rejected [OPN-01]; the cutover method and its per-partition boundary definition [OPN-02]; the definition behind each reported benefit [OPN-03]; the actual delivery semantics of the target path; the topic and partition configuration; and the state backend parameters. Each is resolvable only by first-party artefacts, and none of them changes the arithmetic in Section 6, which is why the case can be complete as a lesson while remaining incomplete as a record.

### 14.4 Interview pack

Fifteen questions follow, each with a checked answer, the reasoning that earns credit, a follow-up that tests depth, an acceptable alternative answer, a red flag, and the section that supports the answer.

**Question 1. What is the bounded decision in this case, and what is the baseline?**

Answer: choose an ingestion and stream-execution design for log-derived metrics and trace spans, against a baseline of Pulsar ingestion with self-managed Flink whose pipelines mostly lacked checkpointing.

Reasoning: the baseline matters as much as the choice, because the reported failure — restart-time loss — is a property of that deployment's recovery contract, not of its execution engine. Naming the baseline correctly is what prevents the case from being misremembered as a product comparison.

Follow-up: would repairing the baseline have been defensible? Acceptable answer: yes, and it is the strongest alternative — smallest blast radius, one changed variable, same drills apply — with the counterargument that an unsupported binary and an end-of-sale product give the repair a shelf life set by the vendor rather than the team.

Red flag: describing the decision as "move from Flink to Dataflow because Dataflow is better", which is both unsupported and not the decision.

Support: Sections 2, 3, 5c.

**Question 2. The article says most pipelines lacked checkpointing. What exactly does that establish?**

Answer: that the described deployment had no consistent recovery point for in-flight state, so a restart discarded in-flight windows; it establishes a deployment weakness, not a property of Flink.

Reasoning: a checkpoint is a position from which consistent replay can begin. Without one, a restart has no defined starting point, so the choice is between losing state and losing offset alignment. Both are losses, and neither is inherent to the engine.

Follow-up: does adding checkpoints alone guarantee end-to-end correctness? Acceptable answer: no — the offset commit must be bound to the checkpoint, the source must retain the replay range, and the sink must tolerate repeated effects. Three separate conditions, one of which the checkpoint addresses.

Red flag: "Flink loses data on restart." That generalisation is contradicted by the existence of checkpointing and reveals that the candidate has memorised the headline rather than the mechanism.

Support: CLM-01, Section 8.5.

**Question 3. Compute per-partition utilisation at peak for 12 billion records per day, a 2.6× peak factor, 96 partitions, 15,000 records/s per partition, with the busiest partition carrying 3.125% of records. Is the platform healthy?**

Answer: mean is 12e9/86400 = 138,888.9 records/s; peak is 2.6 × 138,888.9 = 361,111.1 records/s. Aggregate utilisation is 361,111.1/1,440,000 = 25.1%. Hot-partition load is 361,111.1 × 0.03125 = 11,284.7 records/s, which is 75.2% of 15,000. Healthy at the aggregate level, three-quarters saturated on the partition that governs freshness.

Reasoning: the aggregate number answers "do we have enough total capacity", and the per-partition number answers "will the noisiest service be on time". They differ by a factor of three because the share is three times uniform. A design review needs the second number.

Follow-up: what peak factor makes the hot partition exceed capacity? Acceptable answer: 15,000/11,284.7 = 1.329, so 2.6 × 1.329 = 3.45×; and at a 4.167% hot share (4× uniform) the threshold falls to 2.59×, essentially the current peak.

Red flag: dividing by the number of partitions and declaring capacity sufficient, which is the exact error [R-04] is written to expose.

Support: [R-01], [R-02], [R-04], [R-05].

**Question 4. A sink is unavailable for 45 minutes at 2.0× mean load. How much backlog accumulates, and how long does it take to drain with 1,920,000 records/s of sink capacity?**

Answer: backlog = 277,777.8 × 2,700 = 750 million records, or 675 GB at 900 bytes. Spare capacity during recovery = 1,920,000 − 277,777.8 = 1,642,222.2 records/s. Drain = 750e6/1,642,222.2 = 456.7 s, so total recovery is 3,156.7 s = 52.6 minutes.

Reasoning: the drain divides by spare capacity, not capacity. Dividing 750e6 by 1,920,000 gives 390.6 s and understates recovery by 66 seconds, because arrivals continue during the drain. The overhead ratio is 0.169, so each outage minute costs 1.169 minutes.

Follow-up: what if only half the sink capacity is available because one region is degraded? Acceptable answer: spare becomes 960,000 − 277,777.8 = 682,222.2, drain becomes 1,099.4 s, total 63.3 minutes, and the overhead ratio rises to 0.407.

Red flag: using a backlog figure without subtracting concurrent arrivals, or ignoring retention while computing drain.

Support: [R-06], [R-07], [R-08].

**Question 5. A topic is configured for 72 hours of retention with a 20 TB byte cap, at 12 billion records/day and 900 bytes mean. What retention is actually delivered, and what outage can the team survive?**

Answer: daily bytes = 12e9 × 900 = 1.08e13 B = 10.8 TB, so 20 TB holds 20/10.8 = 1.852 days = 44.4 hours. With 60% reserved for safety and a 0.169 recovery overhead, the longest survivable complete stall is 0.60 × 160,000/1.169 = 82,113 s = 22.8 hours.

Reasoning: the byte cap binds before the time limit, so the configured number is a ceiling that the workload lowers. At peak rate the same cap holds only 17.1 hours, so retention shrinks exactly when incidents are likely. And the survivable outage is not the retention period — it is the retention period discounted for safety and divided by the recovery overhead.

Follow-up: the team claims 72 hours of headroom for a three-day replay. Acceptable answer: that claim is wrong on two counts — the delivered window is 44.4 hours at mean and 17.1 at peak, and a three-day replay is impossible at mean load, let alone at peak.

Red flag: quoting configured retention as though it were delivered retention. This is the single most common mistake in production review.

Support: [R-09], [R-10].

**Question 6. A defect is found five days after deployment. How much of the affected window can be repaired?**

Answer: the required replay window is 432,000 s; the delivered retention is 160,000 s, so the recoverable fraction is 0.370, and 62.96% of the window — roughly 37.8 billion of 60 billion records — is permanently gone unless an independent archive exists.

Reasoning: repair latency, not outage length, is the binding constraint on retention sizing, and it is set by detection capability rather than by infrastructure. The drain for the recoverable portion is 22.22e9/1,642,222.2 = 13,530 s = 3.76 hours at full spare capacity, which is why repairs need a reserved rate rather than the whole surplus.

Follow-up: at what detection latency does retention stop binding? Acceptable answer: 160,000/86,400 = 1.852 days; beyond that, replay is partial. So the safe detection target is under about 44 hours with a five-day requirement needing an archive.

Red flag: assuming a checkpoint or a replay request can recreate data the broker has evicted.

Support: [R-11], [R-12].

**Question 7. Freshness p95 for windowed output is 125.1 s, composed of 1.8 s ingest, 0.9 s processing, 120 s allowed lateness, and 2.4 s sink. Where should the team optimise?**

Answer: nowhere in the pipeline. 95.9% of the budget is allowed lateness, which is a policy choice; the engineering terms total 5.1 s. Reducing allowed lateness is the only lever with material effect, and it is a product decision with a correction cost attached.

Reasoning: a watermark is a heuristic completeness estimate, so the wait is buying late-data coverage, not throughput. Cutting from 120 s to 60 s removes 60 s and triples correction volume from 0.40 to 1.20 million window re-emissions per day.

Follow-up: why is a sum of four p95s not the p95 of the sum? Acceptable answer: the components correlate during degradation — one saturated worker inflates all of them — so the true tail exceeds the sum, making the formula a lower bound in the case that matters.

Red flag: proposing more workers for a latency budget dominated by a deliberate wait.

Support: [R-13], [R-14].

**Question 8. Source shows 1,000,000 records. Sink shows 998,900 distinct identities across 1,001,200 rows. What is the completeness and duplicate rate, and what does a naive count say?**

Answer: completeness is 998,900/1,000,000 = 99.890%; duplicates are 2,300/998,900 = 0.230%; missing identities are 1,100. The naive row ratio is 1,001,200/1,000,000 = 100.12%, which reads as over-delivery and conceals both errors.

Reasoning: completeness is a set intersection, and a row count is not a set intersection. Simultaneous loss and duplication are common because they have independent causes — a poisoned record and a replay — so the errors do not cancel, they superpose in the count while remaining visible in the identity sets.

Follow-up: what single assumption would invalidate this measurement? Acceptable answer: identity stability. If producer retries mint new identifiers, duplicates understate and completeness overstates at the same time, and both errors point the flattering direction.

Red flag: reporting a completeness number without stating the population, the exclusions, and the lateness horizon. A number without those three is not a measurement.

Support: [R-15], [R-16].

**Question 9. A dashboard says p95 freshness is 12 s and the mean is 3.98 s. Why might both be misleading?**

Answer: the distribution is heavy-tailed — 99,000 records at 3 s, 900 at 12 s, 100 at 900 s — so the mean is 3.978 s while p99.9 is 900 s, which is 226× the mean. The 100 worst records are typically the ones queued behind the saturated partition during the incident under investigation.

Reasoning: freshness distributions are mixtures, because a healthy majority on time and a stuck minority produce a mean that describes nobody. Percentiles over the whole population help, but only if reported per partition, since a global p95 can hide a partition whose records are all in the tail.

Follow-up: is the mean ever the right number? Acceptable answer: for capacity and cost planning it is — bytes and records aggregate legitimately. It is the wrong number for any question a human asks about their own experience.

Red flag: quoting a single freshness number without stating the population, the percentile, and the partition scope.

Support: [R-17].

**Question 10. A team adds a region label with 12 values to 1.2 million series. What happens, and what if the label is a request identifier instead?**

Answer: state goes from 1.2e6 × 512 × 2 = 1.229 GB to 14.4e6 × 512 × 2 = 14.75 GB, a 12× rise with no change in event volume. If the label is a request identifier, K grows with traffic rather than by a constant, so no capacity plan converges — the state is a function of load.

Reasoning: cardinality is the product of dimension values, and it is the parameter that makes observability costs behave unlike compute costs. A bounded label is a budget decision; an unbounded label is a design error, because it converts a monitoring system into a per-event database without the storage economics of one.

Follow-up: how do you keep the diagnostic capability while removing the dimension? Acceptable answer: sample the high-cardinality detail into a separate store with its own retention, and keep the metric path aggregate — two systems, two cost curves, one query surface.

Red flag: capping cardinality globally and thereby erasing merchant-specific signals that are the reason the labels existed.

Support: [R-18], [R-19].

**Question 11. What is the right cutover design when both the broker and the runner change?**

Answer: per-partition boundaries with a single declared writer per group, a frozen reconciliation cohort compared on identities and values, and a gate between groups. The declared writer is a timestamped artefact, not an understanding.

Reasoning: a simultaneous broker and runner change creates at least four competing explanations for any divergence — partition key, commit protocol, late-data policy, and parser version — and only a staged comparison can separate them. the ownership map in Section 8.2 shows that six of eight boundaries do not change hands, so most of the risk is in contracts the team already owned.

Follow-up: what makes rollback viable? Acceptable answer: a tested read of current-schema records by the legacy path. If that test fails, the cutover is forward-fix-only and must be resourced as such rather than assumed reversible.

Red flag: deleting the legacy path as soon as the new path shows healthy lag, which removes the only comparison that could reveal a population loss.

Support: Sections 8.2, 9.2, 10.

**Question 12. Where does the failure to publish exactly-once effects usually sit, and how do you test it?**

Answer: at the sink boundary. A runner can commit state exactly once internally and still duplicate an output, because the durable sink write is outside the runner's state. The test is to inject a failure precisely between the sink acknowledgement and the progress commit, then count effects per output identity.

Reasoning: the failure window is small, which is exactly why teams assume it does not matter, and its cost scales with the sink's cost per duplicate — negligible for an averaged gauge, serious for a count that a responder uses to size an incident.

Follow-up: what if the sink has no upsert? Acceptable answer: write append-only with the identity as a deduplication key at query time, and document that the raw table holds duplicates — the alternative is pretending the problem does not exist.

Red flag: "the runner supports exactly-once, so we are fine", which confuses a state guarantee with an effect guarantee.

Support: Sections 8.5, 12.2, 12.6.

**Question 13. How do you measure completeness without a population defined by the pipeline?**

Answer: define the eligible population at the source, from a producer registry and a versioned exclusion registry, and compute the intersection at the sink after the lateness horizon. Then verify the metric can fail by removing one producer's records in a test and checking that the reported completeness drops by that producer's share.

Reasoning: any metric whose denominator is defined downstream of the system it measures improves as the system degrades, because the hard records are precisely the ones that go missing from the denominator. The test that the metric can fail is the only defence, and it takes an hour.

Follow-up: what about intentional sampling? Acceptable answer: sampling is an exclusion with a declared rate and an owner, and it must alarm if the rate goes to zero for a population that should exist.

Red flag: a completeness dashboard that has never shown a value below 99.9%, which usually means the denominator is wrong rather than the pipeline is excellent.

Support: Section 9.1, 12.12, 12.15.

**Question 14. The migration reports lower cost and better uptime. What would you require before accepting that?**

Answer: a comparable baseline with a workload normalisation, a definition of uptime for a telemetry platform, matched comparison windows, and separately itemised replay and incident-driven cost. Absent those, the claims are directional and should be reported as such.

Reasoning: a cheaper bill can mean less work was done, and better uptime can mean a changed definition — is the platform up when it processes records, or when it delivers complete windows? Observational before/after designs in this setting are confounded by volume growth, seasonal incident load, and a changed measurement surface.

Follow-up: is the source therefore useless? Acceptable answer: no — it is credible evidence that a migration happened and that specific operational problems were real, and it is not evidence about magnitude. Distinguishing those two uses is the skill being tested.

Red flag: inventing a percentage, or conversely dismissing the account entirely because it is vendor-published. Both are failures of calibration in opposite directions.

Support: Sections 5d, 9.3, 11.

**Question 15. Change one constraint: telemetry may be retained for only 24 hours for privacy reasons. How does the design change?**

Answer: the recovery contract must be re-sourced from an independent archive with its own retention and access controls, or from a redacted projection that preserves identity and dimensions while discarding payload. Replay windows become a function of the archive rather than the broker, and the defect-detection target must fall inside 24 hours or the recoverable fraction falls below 1.0 immediately.

Reasoning: the retention constraint is not negotiable, so the design must move the durability obligation rather than shrink it. A redacted projection is often sufficient because most replay needs are structural — was this identity present, was this aggregate right — rather than payload-level.

Follow-up: what breaks in [R-11]? Acceptable answer: T_eff becomes 86,400 s, so a five-day defect window has f_recoverable = 0.20, and detection latency becomes the dominant design parameter rather than a monitoring convenience.

Red flag: retaining raw payloads anyway because replay is convenient, which converts a privacy constraint into a compliance incident.

Support: Sections 4 (C-01), 6.5, 9.4, 13.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Observability platform engineering; streaming analytics over log-derived metrics and trace spans |
| Problem type | Systems decision: choose an ingestion and stream-execution design whose recovery behaviour is defined and testable, under an operational burden the previous deployment could not sustain |
| Input / output grain | Input: one telemetry record at broker admission (identity, schema version, event time, payload). Output: one derived metric point or assembled span per declared output identity, per window and key |
| Baseline | Reported: Pulsar ingestion with self-managed Flink, most pipelines without checkpointing, resulting in restart-time loss; manual clusters, peak provisioning, unsupported customised binary, end-of-sale exposure |
| Reported intervention | Proof of concept; ingest moved from Pulsar to Kafka; Dataflow adopted on the Beam model; partitioning redesigned and shuffling optimised with vendor support; BigQuery-backed analytics for failed custom interactions and merchant monitoring |
| Primary pattern | `recoverable-stream-processing` — align source replay, processor state, and sink effects to one explicit, drill-verified contract |
| Supporting patterns | `partition-aware-scaling` (compute per-partition, never aggregate); `migration-reconciliation` (frozen cohort, identity and value comparison, gate per cutover group) |
| Reported result | Qualitative and directional: lower infrastructure and operational cost, improved stability and uptime, improved development agility. No magnitudes, denominators, windows, or independent audit |
| Key lesson | Correctness lives at the boundaries — retention, identity, idempotence, completeness reporting — and those boundaries stay with the migrating team regardless of who operates the compute |
| Failure to prevent | Declaring success on aggregate freshness and cost while the processed population silently shrank |
| Evidence / stage | Published vendor narrative [S1], co-authored by an interested party; production migration reported; every number in this case is an ILLUSTRATIVE teaching value |
| Review | Draft; `review_status: OPEN`; `last_reviewed: null`; revisit on new primary evidence or a change in broker or runner delivery semantics |

### Numbers worth carrying

```text
[R-05] Hot-partition utilisation  = (peak_rate * s_hot) / mu_partition
       Teaching model: (361111.1 * 0.03125) / 15000 = 0.752, versus aggregate
       0.251. Same load, two very different answers.

[R-07] Drain                       = B / (mu - lambda)
       Teaching model: 750e6 / (1920000 - 277777.8) = 456.7 s for a 2700 s outage.

[R-08] Recovery overhead           = lambda / (mu - lambda) = 0.169
       Each outage minute costs 1.169 minutes.

[R-09] Effective retention         = C_topic / (lambda_mean * b)
       Teaching model: 20 TB / (138888.9 * 900 B/s) = 44.4 h, not the 72 h configured.

[R-11] Recoverable fraction        = T_eff / t_replay_required
       Teaching model: 160000 / 432000 = 0.370 for a five-day-old defect.

[R-13] Windowed p95 freshness      = d_ingest + d_proc + L_allowed + d_sink
       Teaching model: 1.8 + 0.9 + 120 + 2.4 = 125.1 s, of which 95.9% is policy.

[R-15] Completeness                = |I_src INTERSECT I_sink| / |I_src|
       Teaching model: 998900 / 1000000 = 0.9989 while row ratio reads 1.0012.
```

### The five contracts that decide the outcome

| Contract | Question it answers | Fails silently when |
|---|---|---|
| Retention | Can replay reach every record the recovery needs? | Configured period is quoted instead of measured oldest-offset age |
| Identity | Can two observations of the same logical event be recognised as one? | Identity is minted per emission, so retries look like new events |
| Idempotence | Is repeating an effect safe? | A timeout is taken as proof the write failed |
| Completeness reporting | Would a gap be visible? | The denominator is defined downstream of the pipeline |
| Cutover boundary | Which path owns which records, from when? | "Single writer" is an understanding rather than a timestamped artefact |

### Drills with pass conditions

| Drill | Injected fault | Pass condition |
|---|---|---|
| Restart mid-window | Kill a worker with a window in flight | Identity reconciliation passes for the killed window |
| Commit-gap | Fail between sink write and progress commit | One effect per output identity after replay |
| Partition stall | Stop one broker partition | Coverage alarm fires within one window; other partitions' freshness unaffected |
| Retention expiry | Let one partition's oldest record age past the alarm threshold | Alarm fires before the record is needed; archive supplies the identities |
| Replay contention | Repair at the reserved rate under peak load | Live p95 freshness within its normal band; replay backlog falls monotonically |
| Population loss | Remove one producer's records, leave its registry entry | Reported completeness falls by that producer's share |
| Rollback | Read current-schema records on the legacy path | Successful decode and reconciled output, or an explicit forward-fix-only decision |

### Supplemental record

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-2026-STR001 / 2.0 / 2024-12-03 |
| Case type | streaming-systems |
| Source basis | published (vendor blog, interested publisher and co-author) |
| Claim ledger | SRC-01 (sourced); CLM-01 to CLM-05 (sourced); DER-01 (derived teaching model); OPN-01 to OPN-03 (open) |
| Implementation stage | Production migration reported by the source; configuration and semantics unverified |
| Evidence level | Published narrative for the situation and intervention; ILLUSTRATIVE for every quantity |
| Primary metric proposed | Completeness per partition against a source-defined population, paired with freshness percentiles, never reported alone |
| Known limitation | Workload, metric definitions, cutover method, and quantified outcomes are not disclosed; repair-in-place was not shown to have been evaluated |
| Review trigger / owner | New primary evidence, delivery-semantics change, or completed editorial review / observability platform team |