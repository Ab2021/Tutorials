---
case_id: "CASE-02-001"
case_title: "Build a Data Quality CLI"
case_version: "2.1"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Data engineering quality controls"
primary_pattern: "data-quality-gate"
secondary_patterns: ["contract-first", "idempotent-operation"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "data platform lead"
source_index: "02_python_sql_linux_and_engineering_basics/Case_Studies/INDEX.md"
review_trigger: "Schema, source SLA, threshold, runtime, or release-policy change"
---

# Build a Data Quality CLI

**Evidence record:** this manuscript is an ILLUSTRATIVE teaching design. Batch sizes, defect counts, run times, alert volumes, and prices are constructed inputs chosen to expose trade-offs; the arithmetic over them is recomputed by hand and is VERIFIED, but nothing here is a production CLI run, an observed incident, or a measured company outcome. **Implementation stage:** proposed. **Editorial status:** draft; source verification and publication review remain OPEN.

The case exists because "is the data clean" is not a decidable business question. "Should this batch become the default input for downstream jobs, under contract C, with this evidence, at this time" is decidable, auditable, and reversible. Everything below is scaffolding for that second question.

## 1. The One-Line Framing
Given one immutable batch, one pinned contract, and a bounded run window, the data platform lead must choose among **publish, publish-with-warning, quarantine, block, and retain-prior**, and must produce a signed decision with reproducible evidence rather than a quality score.

The single sentence version: a quality gate is a *release* decision about a contract, not a verdict about truth.

## 2. Context & Background
A daily ingestion pipeline accepts customer-event records from an upstream producer and writes one partition per calendar day. Analysts, a nightly finance aggregate, and a product funnel dashboard all read whichever partition is currently pointed to as "latest". The current baseline is an analyst opening a sample of the file in a notebook, running several ad hoc checks by hand, and posting a message in a channel that says the partition looks fine.

That baseline is not stupid. It is cheap to change, it adapts to anomalies nobody anticipated, and its failures are visible to a human within minutes. Its weaknesses are equally concrete: the checks are not written down anywhere executable, the evidence disappears when the notebook kernel closes, two analysts can reach opposite conclusions from the same file, and at 03:00 nobody runs anything at all.

The proposed replacement is a command-line tool that reads a manifest and a versioned contract, runs declared checks under resource budgets, writes append-only findings, and asks a policy engine which release action is permitted. The tool never repairs source data. Repair is a separate, approved transformation with its own version, because a gate that silently fixes rows cannot be audited against the rows it received.

Scale is fixed at 10,000,000 rows per day and roughly 1.2 KB per row, giving about 12 GB per daily partition. Those two numbers are the only reason any cost or runtime figure in this manuscript has meaning; they are planning inputs, not measurements.

Timing matters more than most teams expect. The gate runs after the producer declares the partition complete and before the release pointer moves. Freshness is therefore measured against the producer's declared completion time, not against the clock at review time, because a reviewer who arrives four hours late should not manufacture a freshness failure.

Ownership is split deliberately. The data platform lead records the release recommendation and signs overrides. The source owner explains anomalies and supplies remediation evidence. The downstream owner decides whether a warned dataset is fit for their specific use. No single person holds all three roles, and the policy engine is not allowed to merge them.

The tool is explicitly not a privacy review, a security control, or a compliance certification. Those questions are asked elsewhere and cannot be answered by counting nulls.

## 3. The Business Question
| Field | Contract |
|---|---|
| Question | For batch `B` under contract hash `C`, which of publish / warn / quarantine / block / retain-prior is the permitted action? |
| Decision owner | Data platform lead records the action; source owner supplies anomaly explanations; downstream owner accepts or refuses warned data. |
| Trigger | Producer declares the partition complete, or the declared completion deadline passes without a completion signal. |
| Inputs at decision time | Immutable manifest, pinned contract, observed counts, samples, parse-failure counts, prior comparable partition, runner revision. |
| Actions | Publish (move pointer), warn (move pointer with acknowledged findings), quarantine (retain candidate, no pointer move), block (no pointer move, page owner), retain-prior (explicit no-op with reason). |
| Objective | Stop known contract violations from silently becoming default downstream inputs, while keeping every action reversible and explained. |
| Hard constraints | Stable batch identity, pinned contract hash, no silent repair, bounded run time, one active action per batch/contract pair. |
| Scope | One partition, one contract version, one release decision window. |
| Out of scope | Semantic truth beyond the declared contract, privacy determination, permanent source remediation, dashboard correctness. |

The averaging trap is the reason the objective is phrased as a constraint rather than a score. If a batch has 999 passing checks and one failed identity check, an average of 99.9% reads as excellent while the join that consumes that key is broken. Severity is therefore a *precedence relation*, not a weight.

A second trap is the reverse: treating every deviation as fatal. A batch with a 0.3% rise in an optional free-text field's null rate is not a reason to halt the finance aggregate. The policy must distinguish "violates a declared invariant" from "moved against an expectation", and only the first is allowed to block.

### 3a Release policy table
| Condition observed | Declared severity | Permitted action | Who must act | Reversible by |
|---|---|---|---|---|
| Identity, schema-required, or referential invariant violated | critical | block | platform lead pages source owner | no action needed; pointer never moved |
| Freshness outside declared SLA by more than one interval | critical | quarantine with timer | source owner supplies reprocessing ETA | publishing a later corrected batch |
| Volume outside the seasonal band | major | quarantine or warn, owner's choice with rationale | source owner | pointer swap to prior partition |
| Schema addition not in contract | major | review-required, then publish or block | contract owner signs amendment | new contract version |
| Non-critical null rate above expectation | minor | warn | downstream owner acknowledges | pointer swap |
| Distribution shift with no invariant broken | minor | warn or publish | downstream owner | pointer swap |
| Run incomplete (timeout, crash, out-of-memory) | none, but unproven | block publication of critical rules | operations re-runs within window | re-run produces new attempt record |
| Contract hash changed after planning | procedural | review-required | contract owner | discard attempt, re-pin contract |

**Takeaway:** the policy is a total function from findings to actions. Any finding class without a row in this table is a contract defect, not an operator judgement call.

## 4. Assumptions & Constraints
| ID | Assumption or constraint | How it is checked | If it is wrong | Fallback | Owner |
|---|---|---|---|---|---|
| A-01 | Batch identity is immutable once manifest is written | re-hash manifest and object prefix before and after scan | findings attach to data that changed underneath them | quarantine, discard attempt | ingestion |
| A-02 | Contract is pinned by hash before execution | compare stored hash with loaded hash at start and at policy time | thresholds drift mid-run | review-required, new attempt | platform |
| A-03 | Row count is readable from manifest without a full scan | compare manifest count to an independent bounded count | cost estimate and volume rule are wrong | run bounded count first, then decide | platform |
| A-04 | Reference partition is comparable | compare calendar key, producer version, schema version, backfill flag | drift rule fires on incomparable baselines | downgrade drift to informational | source owner |
| A-05 | Event timestamps carry an explicit timezone or offset | reject naive timestamps at parse | freshness misread by whole hours | quarantine the time rule only | producer |
| A-06 | Quarantine storage has room for one partition plus retention window | measure bytes and retention policy before publish | evidence deleted under pressure | stop intake rather than delete evidence | operations |
| A-07 | Producer declares completion at or before the SLA deadline | compare completion signal time with deadline | gate runs on partial data and calls it complete | run in provisional mode, mark findings provisional | producer |
| A-08 | Sampling is deterministic given a stored seed and fraction | re-select sample twice, compare row identity sets | sample results are not reproducible | force full scan for critical rules | platform |
| A-09 | Every rule has an owner and a severity in the contract | contract schema validation rejects unowned rules | nobody acts on a blocking finding | block by default until owner assigned | contract owner |
| A-10 | Runner revision is recorded with each attempt | embed build hash in attempt record | results cannot be attributed to code | mark attempt unattributable, do not publish | platform |
| A-11 | Duplicate detection fits in the runner's memory budget | compare estimated hash table size with limit | runner killed mid-check, partial result looks like a pass | partition-level dedupe or external sort | platform |
| A-12 | Review queue capacity is known in alerts per day | compare alert count with declared reviewer minutes | warnings accumulate unreviewed | raise severity threshold only via contract version | operations |
| A-13 | Prior published partition remains addressable | pointer history retains at least one prior target | rollback impossible | refuse to move pointer at all | platform |
| A-14 | Timeout is treated as "unproven", never as "pass" | state machine test with a cancelled scan | silent acceptance of unaudited data | block publication of critical rules | platform |
| C-01 | The CLI cannot write to the source prefix | read-only credential test on a scratch object | irreversible mutation of evidence | abort the attempt | platform |
| C-02 | Overrides require named owner, rationale, and expiry | policy engine rejects override without all three | open-ended bypass | treat as no override | platform lead |
| C-03 | Runtime fits inside the declared window | timed rehearsal at production row count | backlog grows without bound | degrade only non-critical rules, never critical | operations |

Two of these assumptions are load-bearing beyond the others. A-01 and A-14 each have the property that violating them produces a *plausible* result rather than an error, which is the worst failure shape for a gate. A gate that crashes is annoying; a gate that reports a pass over data it never read is dangerous.

**Takeaway:** the assumption table is a test plan, not a disclaimer. Each row names the check that would falsify it.

## 5. Data, Tools & Method
### 5a. Data Specification
| Element | Specification |
|---|---|
| Decision unit | one `(batch_id, contract_hash)` pair; the release action attaches to this pair and nothing else |
| Row grain | one source event, keyed by `event_id`, with `event_time`, `ingest_time`, and `producer_version` |
| Cut-off | gate decision uses only evidence with collection time at or before the declared completion deadline plus grace seconds |
| Population | all physical rows present in the declared partition prefix, including rows that fail to parse |
| Counts | `rows_declared` (manifest), `rows_seen` (parser), `rows_parsed`, `parse_failures`, per-column non-null counts |
| Units | counts in rows; rates as dimensionless fractions; durations in seconds; storage in bytes; money in USD |
| Keys | `event_id` is the identity key; `(partition_date, batch_id)` is the release key; `finding_id` is unique per rule and attempt |
| Time semantics | event time is the source's claim; processing time is when the platform wrote the row; freshness uses the difference |
| Sampling frame | deterministic hash of `event_id` bucket modulo 10,000 compared against a stored threshold, so the sample is stable across reruns |
| Reference partition | same weekday, same producer version, same schema version, and flagged non-backfill |
| Exclusions | unreadable objects, missing manifest, unsupported encoding, and rows outside the declared prefix; each exclusion carries a reason code |
| Not available at decision time | downstream usage, corrected late arrivals, future restatements, and the eventual business impact of a bad partition |
| Label | the release action is a policy outcome, not a supervised-learning target; it is never used as a training label |
| Severity source | contract field `severity` per rule, validated against the policy table at load time |
| Retention | findings retained for the audit window; raw candidate retained until the quarantine timer expires |
| Row count source | manifest metadata primary, bounded recount as a cross-check |
| Distinct estimate source | exact distinct on the identity key for critical rules, approximate sketch only for informational rules |
| Freshness definition | 95th percentile of `decision_time - event_time` over parsed rows, compared against declared SLA |
| Quarantine location | separate prefix with the same read permissions as published data, plus a pointer record |
| Approval record | owner identity, timestamp, rationale text, expiry timestamp, and the finding set it covers |
| Metric export | per-batch finding counts by severity, run duration, and action, exported to the platform's own metrics store |

The population definition is the most contested line in this table. Teams routinely compare non-null counts to *parsed* rows, which quietly removes unparseable rows from the denominator and makes the data look better precisely when it is worse. The specification says all physical rows, and Section 6 quantifies why that choice changes the release action.

### 5b. Tools & Stack
- Python 3.11 with a single entry point invoked as `dq check --batch <id> --contract <path>`; no daemon, so the tool cannot drift out of version sync with a long-running process.
- Contract stored as YAML under version control, hashed at load, and pinned in the attempt record.
- Manifest read from object storage metadata; content hash computed by the storage layer where available, otherwise by a streaming digest.
- Execution engine is a thread pool over partitions of the object list; each worker owns one row range and returns counters, never row payloads.
- Findings appended to a ledger keyed by `(batch_id, contract_hash, rule_id, attempt)`.
- Policy engine is a pure function from the finding set plus contract to an action; it has no I/O, which is what makes it testable.
- Report renderer emits one JSON document for automation and one short terminal summary for the human who has ninety seconds.
- Release adapter performs a compare-and-swap on a pointer record; it does not copy or delete data objects.
- Clock source is injected, so tests can simulate a deadline passing without waiting.

The deliberate omission is a scheduler inside the tool. Scheduling belongs to the orchestration layer that already knows about dependencies; a CLI that also schedules itself becomes impossible to run twice for debugging.

### 5c. Method Summary
1. Resolve the batch identity and load the pinned contract; refuse to proceed if either is ambiguous.
2. Validate that every rule has an owner, a severity, and a cost class; a rule without a severity is a contract error, not a default-warn.
3. Plan execution: metadata-only rules first, then deterministic samples, then full scans, ordered by severity so a timeout still yields the most decision-relevant findings.
4. Execute within per-rule and global budgets, recording partial counts when a rule is cut short.
5. Aggregate findings by precedence, never by averaging, and compute the permitted action set.
6. Require owner acknowledgement for warn, and owner rationale plus expiry for any override.
7. Move the release pointer with compare-and-swap, keep the previous pointer, and write the attempt record before returning the exit code.

Step 7 is ordered that way on purpose. If the process dies between the pointer move and the attempt record, the audit trail loses the action; if it dies between the attempt record and the pointer move, the audit trail shows an intent that did not happen, which an operator can reconcile. The second failure is recoverable, the first is not.

### 5d. Data provenance, quality, exclusions and leakage controls
- Every finding carries batch ID, object prefix, content hash, contract hash, runner revision, rule revision, and collection timestamp.
- Each rule declares its scope: `metadata`, `sample`, or `full_scan`. A sample finding never satisfies a full-scan rule, and the report prints scope beside every number.
- Parse failures are counted before any completeness calculation and stay in the denominator; dropping them is the single most common way a failing batch is reported as passing.
- Freshness is computed from the original `event_time`, never from a later corrected or restated timestamp, because the decision happened before the correction existed.
- A comparison against a prior partition is only allowed when calendar key, producer version, schema version, and backfill flag all match; otherwise the rule degrades to informational.
- Excluded rows are enumerated with reason codes and counts, and an exclusion count above a declared ceiling is itself a finding.
- Downstream outcomes — dashboard errors, finance adjustments, support tickets — are never inputs to the release decision. Using them would leak the answer into the gate and make the gate look accurate after the fact.
- A contract edited after planning invalidates the attempt; the run is marked review-required rather than re-evaluated under new thresholds, because the finding set was produced under the old ones.
- Sample seeds and fractions are stored so a reviewer can reconstruct exactly which rows were inspected.
- A manually amended contract is a new version with a new hash and therefore a new decision; silent reruns under altered thresholds are prohibited.

**Takeaway:** provenance here means a reviewer can answer "which bytes, under which rules, at which time, by which code" without asking the person who ran the tool.

### 5e. Reproducibility and method-selection ledger
| ID | Artifact | Purpose | Access | Limitation |
|---|---|---|---|---|
| CON-01 | quality contract with severities | defines rules and precedence | proposed | not an approved organizational policy |
| MAN-01 | batch manifest | identity, declared count, declared schema | illustrative | constructed, not a real object listing |
| ATT-01 | attempt record | replays inputs and configuration | proposed | no executable run attached |
| SMP-01 | stored seed and fraction | sample reproducibility | proposed | sampling error remains regardless |
| POL-01 | policy table | action selection | proposed | requires owner approval to be binding |
| ORC-01 | independent counter | arithmetic cross-check | proposed | checks declared properties only |
| BUD-01 | runtime and memory budgets | bounded work | proposed | derived from planning rates, not benchmarks |

Method-selection rationale, recorded because the choice is not obvious: a rule-based gate was chosen over a statistical anomaly detector because the contract must be explainable to the source owner who will contest it. A detector that says "this partition is unusual" cannot be argued with; a rule that says "`event_id` non-null must be 100.0% and it is 97.5%" can be.

## 6. Analysis Walkthrough
All inputs below are ILLUSTRATIVE. All arithmetic is VERIFIED by hand and cross-checked twice; no number is copied from a prior draft.

### 6a The denominator trap in completeness
```text
FORMULA [DQ-01]: completeness = non_null_required / rows_present
WHERE: non_null_required [rows] counts required-column non-null values among parsed rows;
       rows_present [rows] counts every physical row in the declared prefix, parsed or not.
CONSTRAINT: parse failures must be counted and kept in the denominator; a sample denominator
            estimates only the sampled property, never the full partition.
NUMERICAL EXAMPLE:
  rows_present = 10,000 ; parse_failures = 200 ; rows_parsed = 9,800
  non-null event_id among parsed rows = 9,750
  naive:     9,750 / 9,800  = 0.994898 -> 99.49%
  corrected: 9,750 / 10,000 = 0.975000 -> 97.50%
  gap = 99.4898 - 97.5000 = 1.99 percentage points
INTERPRETATION: the naive figure looks like a rounding-level defect; the corrected figure is a
                2.5% identity gap that breaks joins. The denominator, not the data, decided the story.
```
Independent check of the corrected figure: 10,000 − 9,750 = 250 missing identity values, and 250/10,000 = 0.025, so 1 − 0.025 = 0.975. Both routes agree.

The boundary where the formula misleads: when parse failures are concentrated in a single column while the required column parses everywhere, keeping failures in a shared denominator punishes an unrelated rule. The contract therefore computes completeness per required column, with parse failures attributed to the columns that caused them, and a separate whole-row parse-failure rule. Conflating the two produces either false blocks or hidden gaps depending on which way the attribution falls.

A second boundary: if the partition prefix contains a legitimate companion file (a checksum sidecar, a schema descriptor), counting it as a row makes every completeness figure slightly wrong and, worse, makes the row count non-reproducible as new sidecars appear. Population must be defined by object pattern, not by listing the prefix.

### 6b Volume drift against a seasonal expectation
```text
FORMULA [DQ-02]: drift_raw      = (N_obs - N_ref) / N_ref
                 drift_adjusted = (N_obs - N_ref * s) / (N_ref * s)
WHERE: N_obs, N_ref [rows]; s is a dimensionless seasonal factor declared by the source owner
       for the calendar key, with 0 < s, and s = 1 when no seasonal effect is declared.
CONSTRAINT: N_ref must match producer version, schema version, weekday pattern, and be non-backfill.
NUMERICAL EXAMPLE:
  N_ref = 10,000,000 ; N_obs = 8,150,000 ; declared holiday factor s = 0.86
  expected = 10,000,000 * 0.86 = 8,600,000
  drift_raw      = (8,150,000 - 10,000,000)/10,000,000 = -0.185000 -> -18.50%
  drift_adjusted = (8,150,000 - 8,600,000)/8,600,000   = -0.052326 -> -5.23%
  shortfall vs expectation = 8,600,000 - 8,150,000 = 450,000 rows
INTERPRETATION: raw drift triggers the critical band; adjusted drift sits 0.23 points outside a
                +-5% band, so the action is warn-with-rationale, not block. The seasonal claim is
                itself evidence that must be versioned, or it becomes an excuse applied after the fact.
```
Verify the adjusted value independently: 450,000/8,600,000 = 0.0523256, so the shortfall is 5.23% and the band edge is 5.00%; the excess is 0.23 points, which is 0.0023 × 8,600,000 ≈ 20,000 rows. That row figure is the useful one operationally, because it converts a percentage argument into "about twenty thousand events are missing relative to a normal holiday".

Counterexample where the formula breaks: if the source had a partial outage and then a backfill inside the same partition, `N_obs` can match expectation while the *arrival distribution over time* is wrong. Volume is a first moment; it cannot see a hole followed by a refill. The discriminating check is a per-hour arrival histogram compared with the same calendar key, not a total.

Also note the asymmetry: drift is signed, and a large positive drift is usually a duplicate or a fan-out bug rather than healthy growth. A rule that only alarms on shortfall misses the more common integrity failure.

### 6c Runtime, queueing, and the degenerate backlog
```text
FORMULA [DQ-03]: S = N / r ;  rho = lambda / mu ;  W_q = rho / (mu - lambda)   [M/M/1, planning only]
WHERE: N [rows]; r [rows/s]; S [s] service time; lambda [batches/min] arrival rate;
       mu = 60/S [batches/min] service rate; rho dimensionless; W_q [min] expected queue wait.
CONSTRAINT: valid only for lambda < mu, exponential-ish service, single server, no retries, steady state.
NUMERICAL EXAMPLE:
  N = 10,000,000 rows ; r = 50,000 rows/s -> S = 200 s = 3.3333 min
  mu = 60/3.3333 = 18 batches/hour = 0.3000 batches/min
  batches arrive every 10 min -> lambda = 0.1000 batches/min
  rho = 0.1000/0.3000 = 0.3333
  W_q = 0.3333/(0.3000 - 0.1000) = 1.6667 min
INTERPRETATION: with the nominal scan rate the queue waits under two minutes, so a 10-minute
                cadence has slack. The number is a planning figure, not a latency promise.
```
The boundary case is the important one. If a policy change forces a full rescan because a critical rule must be re-evaluated, or the runner degrades to 15 minutes per batch, then mu = 1/15 = 0.0667 batches/min while lambda is still 0.1000, so rho = 1.5 > 1 and no steady state exists. Backlog grows at (0.1000 − 0.0667) × 60 = 2.0 batches per hour, which is 48 batches per day against 24 arrivals. In that regime the queue model's own assumptions fail: wait time is not finite, and the only real options are raising throughput, lowering arrival rate by skipping non-critical checks, or accepting that releases lag by hours.

A second boundary: `r` measured on a warm cache is not `r` for a cold read of a new partition. Substituting the warm figure into the model understates `S` and hides exactly the degradation that causes backlog. The honest input is a cold-read rate measured on a partition of the declared size.

Counterexample to the model's usefulness: when each batch is independent and no queue exists — a scheduled run that starts whether or not the previous finished — the backlog shows up as overlapping runs and memory pressure rather than as waiting time. Then the discriminating check is concurrent-run count, not `W_q`.

### 6d Sampling power and the rule of three
```text
FORMULA [DQ-04]: P(detect) = 1 - (1 - p)^n ;  n_95 ~= 3 / p
WHERE: p is the per-row defect probability (dimensionless, small); n [rows] sample size;
       n_95 is the sample needed so that P(detect) >= 0.95 when p is the true rate.
CONSTRAINT: independent rows, uniform sampling, defect rate homogeneous in the population;
            the approximation degrades when p is not small or defects cluster.
NUMERICAL EXAMPLE:
  p = 1e-4 (one defect per 10,000 rows)
  n = 1,000  -> P = 1 - (0.9999)^1000  = 0.0952 -> 9.5%
  n = 30,000 -> P = 1 - (0.9999)^30000 = 0.9502 -> 95.0%
  3/p = 3/1e-4 = 30,000 rows, matching the computed 95% point
INTERPRETATION: a 1,000-row sample is blind to a 1-in-10,000 defect nine times out of ten, so a
                "sample passed" statement cannot discharge an identity rule. It can only discharge
                rules whose tolerable defect rate is far above the sample resolution.
```
Independent check: with n = 30,000 and expected defects np = 3, a Poisson view gives P(at least one) = 1 − e^(−3) = 0.9502, matching the binomial figure to four decimals. Two derivations agreeing is the reason the rule of three is trustworthy at this scale.

Boundary where it breaks: clustered defects. If one producer shard emits malformed rows for twenty minutes, defects are not independent, and a uniform sample can land entirely outside the affected shard. The discriminating check is a per-shard, per-hour defect count, which is cheap because it is a count, not a scan. Homogeneity is an assumption the gate must test rather than assume.

Second boundary: `p` is unknown before the check, so the rule cannot size a sample; it can only price one. Sizing therefore runs backwards from the contract: what defect rate would we tolerate in a critical field, and is that tolerance above or below the sample resolution? If the tolerable rate is below 1/n, the rule must be a full scan, and no amount of sampling cleverness changes that.

### 6e Cost per decision: scan versus review
```text
FORMULA [DQ-05]: C_scan = (S/60) * c_c ;  C_review = A * m * c_m ;  ratio = C_review / C_scan
WHERE: S [s] scan time; c_c [USD/compute-minute]; A [alerts/day]; m [minutes/alert];
       c_m [USD/reviewer-minute]; ratio dimensionless.
CONSTRAINT: excludes storage, egress, engineering time, and the cost of a wrong release.
NUMERICAL EXAMPLE:
  S = 200 s = 3.3333 min ; c_c = 0.20 USD/min -> C_scan = 0.6667 USD/batch
  A = 40 alerts/day ; m = 6 min ; c_m = 0.75 USD/min -> C_review = 180.00 USD/day
  daily scan spend at 10 batches = 6.667 USD/day ; ratio = 180.00/6.667 = 27.0
INTERPRETATION: reviewer attention costs 27 times the compute, so optimising scan speed is the
                wrong lever; reducing alert volume or alert handling time is the lever.
```
Annualising both sides: 180 × 365 = 65,700 USD/year of review against 6.667 × 365 ≈ 2,433 USD/year of compute. Verify the compute annual figure: 0.6667 × 10 × 365 = 2,433.5. The comparison is robust to a factor-of-three error in either input, which is what makes it usable for a decision rather than merely interesting.

Boundary where the arithmetic misleads: the formula charges nothing for a wrong release. If a blocked-but-correct batch delays a finance close, the true cost can exceed a year of review; if an accepted-but-wrong batch corrupts a dashboard used for a pricing decision, the cost is unbounded in principle. The formula compares the two *visible* costs and must be labelled as such.

A cheap-looking optimisation fails for the same reason. Cutting the full scan to a 5% sample reduces C_scan to 0.0333 USD and saves 6.33 USD/day, but Section 6d shows the sample cannot discharge a 1-in-10,000 identity rule. The saving is real and irrelevant.

### 6f Duplicate detection under a memory bound
```text
FORMULA [DQ-06]: M_hat = n_e * b ;  P_collision ~= n_e^2 / 2^65
WHERE: n_e [distinct keys] ; b [bytes per hash entry] ; M_hat [bytes];
       P_collision is for 64-bit hashes over n_e keys (birthday approximation).
CONSTRAINT: hash table resident in memory; no spill; 64-bit hashes uniform; n_e >= 1.
NUMERICAL EXAMPLE:
  n_e = 10,000,000 ; b = 16 B (8-byte hash + 8-byte accounting) -> M_hat = 160,000,000 B = 152.6 MiB
  P_collision ~= (1e7)^2 / 2^65 = 1e14 / 3.689e19 = 2.71e-6
INTERPRETATION: a 128 MiB runner limit cannot hold the identity set, and the hash is safe against
                accidental collision, so the constraint is memory, not hash width.
```
Check the collision figure another way: the number of unordered pairs is n(n−1)/2 ≈ 5e13, and each pair collides with probability 2^−64 ≈ 5.42e−20, giving 5e13 × 5.42e−20 ≈ 2.7e−6. Agrees.

The immediate engineering consequence: at 152.6 MiB against a 128 MiB cap, the check dies at roughly 84% of the batch. A naive runner reports "no duplicates found" for the fraction it processed, which is the A-14 failure shape in its purest form. The remedies are partition-level dedupe with a cross-partition reconciliation step, or an external sort with a bounded merge, both of which cost more I/O and both of which are honest about their scope.

Boundary where the collision approximation fails: adversarial or non-uniform keys. If `event_id` is a truncated timestamp with low entropy, the birthday approximation is invalid and duplicates are real rather than accidental. The discriminating check is the distinct-count-to-row-count ratio per time bucket; a ratio near 1.0 with visible repeats means the key is not unique by construction, and the fix belongs upstream.

### 6g Referential gaps amplify into orphan volume
```text
FORMULA [DQ-07]: g = |missing_keys| / |dimension_keys| ;  o = |orphan_rows| / |fact_rows|
                 amplification = o / g
WHERE: g and o are dimensionless fractions; missing_keys [keys] are fact-side keys absent from the
       dimension; dimension_keys [keys] the dimension population; orphan_rows [rows] the fact rows
       whose key is missing; fact_rows [rows] the fact population.
CONSTRAINT: the missing keys must be a random subset of the dimension, and each key must carry a
            similar number of fact rows; both assumptions fail under skew, which is the point.
NUMERICAL EXAMPLE:
  missing_keys = 500 ; dimension_keys = 250,000 -> g = 500/250,000 = 0.002000 -> 0.20%
  orphan_rows  = 18,000 ; fact_rows = 1,000,000 -> o = 18,000/1,000,000 = 0.018000 -> 1.80%
  amplification = 0.018000/0.002000 = 9.0 ; orphan rows per missing key = 18,000/500 = 36
INTERPRETATION: a referential defect that reads as two-tenths of one percent on the dimension side
                surfaces as nearly two percent of the fact table, so a threshold set on the
                dimension measurement understates the downstream damage by a factor of nine.
```
Independent check: 500 keys × 36 rows each = 18,000 rows, and 18,000/1,000,000 = 1.8%. The amplification factor is just the average fact rows per key, which is a useful identity: at this grain, a rate-based rule and a row-based rule differ by exactly the fan-out.

Where the identity breaks: if the 500 missing keys are the heavy hitters — the top customers, the busiest sensors — each may carry hundreds of rows, and the amplification is far above 9. Then a 0.2% dimension gap can orphan 20% of the fact table, which is why the orphan rule must be measured in rows and given its own severity rather than derived from the dimension rate. If the missing keys are inactive accounts carrying four rows each, the same 0.2% gap orphans only 500 × 4 = 2,000 rows, a 0.2% fact rate that a 1% threshold would wave through. Same dimension number, two opposite decisions.

The practical rule: never set a threshold on the upstream count and assume the downstream count follows. Measure the rows.

### 6h Freshness as a percentile, not a maximum
```text
FORMULA [DQ-08]: p95 = L_k + (0.95N - C_{k-1}) / n_k * w_k
WHERE: buckets indexed by k with lower bound L_k [h], width w_k [h], count n_k [rows];
       C_{k-1} [rows] the cumulative count before bucket k; N [rows] total rows;
       p95 [h] is the interpolated 95th percentile of the freshness distribution.
CONSTRAINT: buckets are contiguous and non-overlapping; interpolation inside a bucket assumes the
             freshness values are uniform there, which is a modelling choice, not a fact.
NUMERICAL EXAMPLE, baseline:
  buckets 0-1h:7,000,000  1-2h:1,200,000  2-3h:800,000  3-4h:500,000  4-6h:400,000  >6h:100,000
  N = 10,000,000 ; 0.95N = 9,500,000 ; cumulative through 3-4h = 7,000,000+1,200,000+800,000+500,000
  = 9,500,000 exactly -> p95 = 4.00 h, landing on a bucket boundary
NUMERICAL EXAMPLE, perturbed (same total, 30,000 rows moved out of 3-4h into later buckets):
  cumulative through 3-4h = 9,470,000 ; remaining = 9,500,000 - 9,470,000 = 30,000
  fraction into 4-6h bucket = 30,000/420,000 = 0.0714 -> p95 = 4.00 + 0.0714*2 = 4.14 h
INTERPRETATION: 30,000 rows out of ten million moved the headline freshness figure by 0.14 hours,
                while the maximum would have jumped by whatever the worst row did.
```
Why percentile rather than maximum: a maximum is a statistic about one row, and one pathological row — a retried producer request, a test event with a 2019 timestamp — can move it arbitrarily. A p95 is a statistic about 500,000 rows at this scale, so no single record controls it. Why percentile rather than mean: the mean is dragged by the same tail in the same direction, and it also hides a bimodal distribution where 10% of rows arrive six hours late while 90% arrive in twenty minutes.

The boundary where interpolation misleads: if the 4-6h bucket is genuinely bimodal because two producer instances have different clocks, the uniform-within-bucket assumption places p95 at 4.14h when the truth is closer to 6.00h for one instance. The discriminating check is a two-dimensional histogram by producer instance and hour; a single instance showing a spike is an instance problem, not a partition problem.

A second boundary: p95 computed over parsed rows only is not the same measure as p95 over physical rows, for exactly the denominator reason in 6a. Unparseable rows have no trustworthy `event_time`, so they must be excluded *and* counted in a separate parse-failure rule, never merged silently.

### 6i Partition skew makes the timeout a data-dependent cliff
```text
FORMULA [DQ-09]: t_max = (N * s_max) / r ;  skew = s_max / (1/P)
WHERE: N [rows]; s_max is the largest single-partition share of rows (dimensionless);
       r [rows/s] per-worker scan rate; P the worker count (dimensionless);
       t_max [s] the slowest worker's time; skew is the ratio of the heaviest share to a uniform share.
CONSTRAINT: workers are assigned whole partitions; every worker runs concurrently; r is constant
            across partitions, which fails when a partition holds wide rows or nested fields.
NUMERICAL EXAMPLE:
  N = 10,000,000 ; P = 64 ; uniform share = 1/64 = 0.015625 -> 156,250 rows -> 3.125 s at 50,000 rows/s
  observed heaviest share s_max = 0.048 -> 480,000 rows -> t_max = 480,000/50,000 = 9.6 s
  skew = 0.048/0.015625 = 3.072
INTERPRETATION: the mean work per worker is 3.125 s but the batch takes 9.6 s, so any budget set from
                the mean is 3.07 times too optimistic. A per-worker timeout of 12.5 s leaves the
                heaviest worker only 23% headroom, and a 0.08 share would need 16.0 s and time out.
```
The consequence is that the timeout rule and the partition layout are the same rule. A budget of 240 s for the whole scan implies 240/200 = 1.2 separate full passes at the nominal rate, so the contract cannot afford two passes over the data; if a second pass is needed for correctness it must be fused into the first. Meanwhile a per-worker timeout derived from the mean will cancel the heaviest partitions routinely, and a cancelled partition produces an unproven rule for those rows only — a partially scoped failure that is much harder to reason about than a whole-batch failure.

The remediation is a two-phase assignment: sort partitions by known byte size, then assign greedily to the least-loaded worker. That reduces `t_max` toward the mean plus one large partition, and it makes `t_max` predictable enough to budget. It does not fix wide-row variance, which requires sizing by bytes rather than rows.

Counterexample: when the storage layer splits a large partition into row-group reads, worker load stops tracking partition share and starts tracking byte share; then `s_max` on rows is the wrong input and the model over- or under-estimates depending on compression. The discriminating check is bytes scanned per worker against wall time per worker.

### 6j Fusing checks changes the cost curve, not the results
```text
FORMULA [DQ-10]: T_k = S + c*(k-1) ;  marginal_k = T_k - T_{k-1} = c
WHERE: k the number of rules evaluated in one pass (dimensionless, k >= 1);
       S [s] the single-pass scan time; c [s] the marginal per-rule evaluation cost;
       T_k [s] total wall time for k fused rules; marginal_k [s] the cost of adding rule k.
CONSTRAINT: the rules read the same columns in the same pass; a rule needing a different projection,
            a join, or a sort is not fusable and costs a separate pass.
NUMERICAL EXAMPLE:
  S = 200 s ; c = 4 s per additional rule
  T_1 = 200 s ; T_2 = 204 s ; T_3 = 208 s ; T_5 = 216 s
  marginal per added rule = 4 s, against 200 s for a separate pass
  fused budget check: 208 s scan + 2 s metadata + 10 s sampling = 220 s, inside a 240 s budget
INTERPRETATION: the second and later rules are 2% of the cost of the first, so the expensive decision
                is how many passes to make, not how many rules to write.
```
This is the arithmetic behind a design rule that looks like premature optimisation but is not: rules must be declared with a projection, and the planner groups rules by projection into passes. Five rules over the same projection cost 216 s; the same five written as separate queries cost 200 × 5 = 1,000 s, which exceeds four consecutive batch windows and guarantees backlog even at a 10-minute cadence.

Where the model is wrong: `c` is not constant. A regex rule costs more per row than a null check, and a rule that maintains a distinct set grows superlinearly as `n_e` rises. Substituting a single `c = 4 s` for a regex over free text will underestimate by an order of magnitude, and the budget will be blown by one expensive rule added late. The honest input is a per-rule measured `c`, with the planner budgeting the sum.

A second boundary: fusing changes failure scope. One pass with five rules means one crash loses all five results, whereas five passes lose one. The mitigation is checkpointing per rule within the pass, so a crash mid-pass still leaves the completed rules' counters.

### 6k Reviewer capacity is a queue with its own utilisation
```text
FORMULA [DQ-11]: rho_rev = (A * m) / M_day ;  backlog_rate = max(0, A - M_day/m)
WHERE: A [alerts/day]; m [hours/alert]; M_day [reviewer-hours/day]; rho_rev dimensionless;
       backlog_rate [alerts/day] the rate at which the queue grows when rho_rev > 1.
CONSTRAINT: one reviewer, constant arrival rate, no prioritisation, no batching of similar alerts;
            the rate form assumes alerts are interchangeable in handling time, which they are not.
NUMERICAL EXAMPLE:
  M_day = 8 reviewer-hours/day ; m = 0.1 h/alert (6 min) -> capacity = 8/0.1 = 80 alerts/day
  A = 40 -> rho_rev = 40/80 = 0.5000, backlog_rate = 0
  A = 120 -> rho_rev = 120/80 = 1.5000, backlog_rate = 120 - 80 = 40 alerts/day, 200 after 5 days
INTERPRETATION: the same 8 reviewer-hours absorb 40 alerts comfortably and lose ground at 120, so an
                alert-volume increase of 3x turns a working process into a five-day backlog in one week.
```
Cross-check with the cost figure: 40 alerts × 6 min = 240 min = 4.0 hours of the 8 available, i.e. 50% utilisation, matching rho_rev = 0.5. At 120 alerts the requirement is 720 min = 12 hours against 8 available, a 4-hour daily deficit that matches the 40-alert/day backlog at 6 min each (40 × 6 = 240 min = 4 h).

The reason this belongs in a data quality case rather than an operations case is that alert volume is a *contract* consequence, not a workload the team must absorb. Every rule promoted from minor to major multiplies its alert rate. A single rule that fires on 30% of batches contributes 0.3 alerts/day per batch, so at three batches a day it contributes 0.9 alerts/day; promoting six such rules consumes 5.4 alerts/day of an 80-alert capacity. Volume is designed, and the design is the severity column.

Boundary where the model fails: alerts are not interchangeable. One identity failure may take ninety minutes and fifty minor warnings may take two minutes each in a batch action. Then the arrival-rate model understates utilisation badly, and the discriminating check is per-severity handling time rather than a single `m`.

### 6l Detecting slow drift with a cumulative sum instead of a threshold
```text
FORMULA [DQ-12]: S_i = max(0, S_{i-1} + (x_i - mu_0)/sigma_0 - k) ;  alarm when S_i > h
WHERE: x_i the batch statistic [same unit as mu_0]; mu_0 and sigma_0 the declared baseline mean and
       standard deviation; k [sigma units] the slack, typically 0.5; h [sigma units] the decision
       threshold, typically 4 to 5; S_i is dimensionless.
CONSTRAINT: baseline parameters are stable and known before monitoring; the statistic is roughly
            normal; one-sided detection of an upward shift.
NUMERICAL EXAMPLE:
  k = 0.5, h = 5, shift delta = 1 sigma per batch
  each batch adds (1 - 0.5) = 0.5 to S ; alarms after 5/0.5 = 10 batches
  at delta = 2 sigma: adds 1.5 per batch -> 5/1.5 = 3.33 -> alarms on the 4th batch
  at delta = 3 sigma: adds 2.5 per batch -> alarms on the 2nd batch
INTERPRETATION: a 1-sigma shift is invisible to a 3-sigma threshold rule forever, and a CUSUM finds
                it in ten batches by accumulating a small persistent excess.
```
The comparison that justifies the extra machinery: a fixed threshold at 3 sigma triggers on a 1-sigma shift with probability about 0.0013 per batch, so at 365 batches a year you expect 0.47 false alarms and no detections. The CUSUM at the same false-alarm budget detects the shift in ten batches. The cost is that a CUSUM has memory, so one excursion keeps `S` elevated for several batches and an operator who resets it wrongly delays the next detection.

Boundary where this breaks: the baseline itself drifts. If `mu_0` was set in January and the business grew 20% by June, every batch contributes a positive increment and the CUSUM alarms continuously — a permanent alarm that teams learn to ignore. The correction is to re-estimate `mu_0` on a rolling, declared window, and to record the window in the contract so the baseline is as auditable as the threshold.

Second boundary: a single large excursion followed by a return to normal. The CUSUM absorbs it into `S` and may or may not alarm, whereas a threshold rule catches it immediately. Use both: a threshold for spikes, a CUSUM for creep, and a rule that says which one owns the decision.

### 6m Quarantine storage is a standing cost with a retention cliff
```text
FORMULA [DQ-13]: B_day = N * b_row ;  C_ret = B_day * D * p_storage
WHERE: N [rows/day]; b_row [bytes/row]; B_day [bytes/day]; D [days] retention;
       p_storage [USD per byte per day], commonly quoted per TB-month and converted.
CONSTRAINT: quarantined partitions are retained in full; no compression is claimed; the copy does not
            duplicate already-published bytes in this estimate.
NUMERICAL EXAMPLE:
  N = 10,000,000 ; b_row = 1,200 B -> B_day = 12,000,000,000 B = 12.0 GB/day
  D = 30 days -> 360 GB = 0.3516 TB ; p_storage = 23 USD/TB-month
  monthly retention cost = 0.3516 * 23 = 8.09 USD
INTERPRETATION: retention of a month of quarantined candidates costs about eight dollars, which is
                twelve times the daily compute figure but small against the 180 USD/day review cost.
```
Convert carefully, because this is where unit errors live. 12.0 GB/day × 30 = 360 GB; 360/1024 = 0.3516 TB; 0.3516 × 23 = 8.09 USD per month. A common mistake is multiplying GB by a TB price without dividing by 1024, which overstates cost by three orders of magnitude and leads teams to shorten retention for a reason that does not exist.

The cliff is not cost, it is capacity. Quarantine grows whenever releases fail, and releases fail in clusters — a producer change can quarantine fourteen consecutive batches. At 12 GB/day, fourteen days is 168 GB, which is fine; but a batch size increase to 40 GB/day makes the same event 560 GB and, more importantly, makes the *reprocessing* window fourteen times longer than the team's ability to reprocess. Retention policy should be derived from the reprocessing capability, not from the storage bill.

Boundary: if quarantine holds a copy rather than a pointer and the source object is also retained, the real footprint doubles to 16.17 USD/month — still small, but the duplication is the thing to notice, because two copies can diverge and then nobody knows which was examined.

### 6n Rate resolution limits what a threshold can mean
```text
FORMULA [DQ-14]: resolution = 1/n ;  halfwidth_95 ~= 1.96*sqrt(p(1-p)/n)
WHERE: n [rows observed]; p the observed rate (dimensionless); resolution is the smallest nonzero
       rate expressible as one row in n; halfwidth_95 is the normal-approximation interval half-width.
CONSTRAINT: rows independent and identically distributed with respect to the property; the normal
            approximation is unreliable when np < 5, which is exactly the rare-defect regime.
NUMERICAL EXAMPLE (p = 0.998, i.e. a 0.2% defect rate):
  n = 200     -> resolution 0.50 pp ; halfwidth 0.62 pp -> interval wider than the quantity
  n = 1,000   -> resolution 0.10 pp ; halfwidth 0.28 pp
  n = 10,000  -> resolution 0.01 pp ; halfwidth 0.088 pp
  n = 100,000 -> resolution 0.001 pp ; halfwidth 0.028 pp
INTERPRETATION: at n = 200 the interval half-width exceeds the 0.2% effect being measured, so a
                threshold of 0.2% cannot be adjudicated; the rule must either scan more rows or
                declare that it is measuring nothing at this scale.
```
A consequence that surprises teams: two batches at n = 10,000 whose rates differ by 0.02 percentage points cannot be distinguished, because one row in 10,000 is 0.01 pp and two batches differ detectably only above roughly twice the resolution. Thresholds finer than the measurement resolution are not strict; they are coin flips with a policy attached.

The rare-defect boundary: at p = 0.0001 with n = 10,000 the expected count is 1, so the normal interval is meaningless and the correct statement is a Poisson-based one. That is the same territory as the rule of three in 6d, approached from the measurement side rather than the sampling side.

### 6o Budget allocation as an explicit knapsack, not a queue
```text
FORMULA [DQ-15]: maximise  sum(value_i * x_i)  subject to  sum(cost_i * x_i) <= B_rem ;  x_i in {0,1}
WHERE: value_i is a declared priority weight per rule (dimensionless, set by severity);
       cost_i [s] the marginal evaluation cost of rule i in the fused pass; B_rem [s] the budget left
       after the mandatory first pass; x_i indicates whether the rule runs.
CONSTRAINT: costs are additive within one pass, rules are independent, and a rule not run leaves its
            property unproven for that batch — it is never "passed".
NUMERICAL EXAMPLE:
  S = 200 s scan ; total budget 240 s -> B_rem = 40 s
  weights: critical 100, major 10, minor 1
  rule            severity  cost[s]  value density (weight/cost)
  range_check     critical      4      25.00
  freshness       critical      6      16.67
  cardinality     minor         2       0.50
  referential     major        22       0.45
  enum            minor         3       0.33
  regex_text      minor        12       0.08
  greedy fill by density: 4 + 6 + 2 + 22 + 3 = 37 s <= 40 s, leaving 3 s unused
  total runtime = 200 + 37 = 237 s, inside 240 s
INTERPRETATION: value density, not severity alone, orders execution; a critical rule at 6 s beats a
                major at 22 s because the marginal budget is the scarce resource.
```
The uncomfortable part of this result is that `referential` at 22 s is only admitted because it is major and there is slack; a 30 s budget would drop it and report its property unproven. That is the correct behaviour and it is also the behaviour operators dislike, which is why the report must print the dropped rules explicitly rather than silently producing a shorter finding list.

Where the knapsack model is wrong: costs are not independent when two rules share a distinct set, and the value weights are ordinal rather than cardinal. Treating them as cardinal invites a team to "solve" policy by reweighting. The weights should be derived mechanically from the severity order — critical above major above minor — with no free parameters, so the only way to change the outcome is to change a severity in the contract.

A second boundary: a greedy fill is not optimal in general. If the budget were 22 s, greedy takes range (4), freshness (6), cardinality (2), and enum (3) for 15 s and then cannot fit referential (22), yielding value 100+100+1+1 = 202; taking referential alone yields 10. Greedy wins here, but the classic counterexample exists whenever one high-value item is slightly too large to combine. For six rules the difference is irrelevant; for sixty, an exact solve is cheap and should be used.

### 6q Late arrivals force a provisional decision, not a delayed one
```text
FORMULA [DQ-16]: visible(t) = N * (1 - u(t)) ;  L = N * u_final ;  gap = N * (u_final - u(t_gate))
WHERE: N [rows] final partition size; u(t) the fraction of rows not yet arrived at age t
       (dimensionless, decreasing in t); t_gate the decision time; L [rows] rows that will arrive
       after the gate; gap [rows] rows missing at gate time relative to the final population.
CONSTRAINT: u(t) is a declared arrival curve for the producer, not an assumption of punctuality;
            the gate cannot see rows that do not exist yet, and must not pretend it did.
NUMERICAL EXAMPLE:
  N = 40,000,000 ; u(t_gate) = 0.28 -> visible = 40,000,000 * 0.72 = 28,800,000 rows
  u_final = 0.28 held constant (all lateness unresolved) -> gap = 40,000,000 * 0.28 = 11,200,000 rows
  reprocessing cost of one corrective pass = 200 s and 0.6667 USD at the Section 6c rate
INTERPRETATION: a gate that runs at completion sees 28.8 million of an eventual 40 million rows, so
                every volume and completeness figure is scoped to 72% of the population and must say so.
```
The design response is a provisional state rather than a longer wait. Waiting has a cost too: holding the release until `u(t)` is negligible delays every consumer, and the delay is paid daily. A provisional publish with a declared re-evaluation trigger — the arrival curve reaching its plateau, or a fixed age such as six hours — bounds both the error and the delay.

The invariant that keeps this honest: a provisional decision names its population fraction in the attempt record, and any downstream aggregate that reads provisional data must be able to see that flag. A provisional batch that looks identical to a final one in the pointer record is a silent correctness bug affecting every consumer at once.

Boundary where the formula fails: if lateness is correlated with content — late rows are systematically the large orders or the failing devices — then the 72% visible subset is not a random 72%, and every rate computed on it is biased rather than merely noisy. The discriminating check is to compare the distribution of a few stable columns between early and late rows for a batch where both are eventually available.

### 6r Severity composition without averaging
| Observed finding set | Counted pass rate if averaged | Permitted action under policy | Why the average is irrelevant |
|---|---|---|---|
| 1 critical identity failure, 0 others | 0.0% | block | the single failure is a declared invariant, so the rate never enters the decision |
| 1 critical identity failure, 999 minor passes | 99.9% | block | the average is at its most flattering exactly when the block is required |
| 0 critical, 3 major warnings | not meaningful | warn with acknowledgement | major findings need an owner, not a threshold |
| 0 critical, 1 major, 40 minor | not meaningful | warn; review-required if the major is unreviewed | volume of minor findings does not upgrade or downgrade a major |
| 0 findings, run timed out before critical rules | not meaningful | block publication of critical rules | unproven is a distinct state from clean |
| 0 findings, contract hash changed mid-run | not meaningful | review-required | the evidence set no longer matches the contract |

The table's purpose is to make the precedence order explicit enough to implement as a fold. The policy is evaluated as: if any critical finding is present and unresolved, the action is block; else if any run-incompleteness affects a critical rule, block; else if any major finding lacks an acknowledgement, warn or quarantine per the rule's declared action; else publish. Warnings can never soften a block, and minor findings can never harden one.

A useful invariant to test the implementation: adding a passing minor rule to the contract must not change any action for any batch. If it does, severities are being weighted somewhere, and the weighting will eventually be gamed by whoever can add checks.

### 6h State trace for one decision
| State | Entry condition | Evidence written | Pointer effect | Exits to |
|---|---|---|---|---|
| RECEIVED | manifest located and hashed | manifest hash, object prefix, arrival time | none | VALIDATED or REJECTED-MANIFEST |
| VALIDATED | identity, schema, and contract hash consistent | contract hash, rule count, planned cost | none | RUNNING |
| RUNNING | planner produced a rule order | per-rule start times and budgets | none | AGGREGATED or INCOMPLETE |
| INCOMPLETE | global budget exhausted or worker died | partial counts, which rules are unproven | none | BLOCKED-CRITICAL or REVIEW_REQUIRED |
| AGGREGATED | all declared rules returned a status | finding set with scopes | none | POLICY |
| POLICY | finding set folded by precedence | permitted action set and reason codes | none | WARN, QUARANTINED, BLOCKED, PUBLISHED |
| WARN | only non-critical findings remain | acknowledgement request with expiry | none | PUBLISHED or QUARANTINED |
| QUARANTINED | critical non-identity finding or owner choice | reason, timer, candidate location | none | PUBLISHED (later corrected batch) or EXPIRED |
| BLOCKED | critical invariant violated | blocking finding and owner page record | none | REVIEW_REQUIRED after source fix |
| PUBLISHED | permitted action plus required acknowledgements | attempt record, approver, prior pointer | compare-and-swap to candidate | ROLLED_BACK if needed |
| ROLLED_BACK | owner action after publish | both pointers, reason, impact note | swap back to prior | terminal |
| REVIEW_REQUIRED | contract, runner, or source version changed | diff summary of what changed | none | RUNNING under new versions |

Two state-machine properties are worth testing directly. First, no edge writes the pointer except PUBLISHED and ROLLED_BACK, which means a crash in any other state cannot publish anything. Second, `REVIEW_REQUIRED` is reachable from `INCOMPLETE` as well as from a version change, because an interrupted run and a changed contract both destroy the correspondence between evidence and decision.

**Sensitivity:** halving the scan rate doubles `S` from 200 s to 400 s and doubles per-batch compute to 1.333 USD, but leaves the review cost untouched at 180 USD/day, so the review-to-compute ratio falls from 27.0 to 13.5 and the ranking of levers does not change. Raising the identity rule from full-scan to sample raises the tolerated defect rate from effectively zero to 1/1,000, which is a policy statement about acceptable broken joins, not a performance tuning knob.

## 7. The Finding
The analysis supports a contract-driven gate with five named actions and no aggregate quality score. Three results do the work. First, the denominator choice moved completeness from 99.49% to 97.50% on identical data, which is larger than any threshold argument the team had been having. Second, the seasonal adjustment moved volume drift from −18.50% to −5.23%, converting an apparent catastrophe into a marginal warning that still required an owner note. Third, reviewer attention at 180 USD/day against 6.67 USD/day of compute means the operational bottleneck is the alert queue, not the scanner.

The gate is eligible for a bounded offline rehearsal only after the contract is approved, a prior pointer exists to roll back to, and per-rule budgets are declared. Nothing in this manuscript establishes that any batch was published under it. The strongest claim available is narrower and still useful: the decision procedure is specified well enough to be implemented twice by different people and compared.

## 8. System/Solution Design
```text
[manifest + objects] --> [identity validator] --> [rule planner] --> [bounded runners]
                                |                                        |
                       reject on ambiguity                     counters + samples
                                |                                        v
[contract store, hashed] -----------------------------> [finding ledger, append-only]
                                                                         |
                                                                         v
                                                            [policy fold: precedence]
                                                                         |
        [owner console: acknowledge / override + rationale + expiry] <---+
                                                                         |
                                          +------------------------------+------------------+
                                          v                                                 v
                               [pointer CAS: publish]                         [quarantine / block / retain]
                                          |                                                 |
                                          +--> [prior pointer retained] <-------------------+
                                                       |
                                              [rollback = pointer swap]
```

Ownership in the design is assigned to components, and the assignment is the security model. The manifest validator owns batch identity, so no other component can assert what data it read. The planner owns resource selection but cannot downgrade a critical full-scan rule to a sample without a new contract hash. The ledger owns append-only evidence and refuses updates, so a re-run adds an attempt instead of overwriting a finding. The policy fold is a pure function and owns the action set, which means the action can be recomputed from the ledger by anyone, including a sceptical source owner. The owner console owns acknowledgements and overrides with expiry. The release adapter owns the only pointer mutation and uses compare-and-swap so a concurrent release cannot interleave.

The central invariant: for any `(batch_id, contract_hash)` there is at most one active release action, and every action is derivable from the ledger plus the pinned contract. A retry either reproduces the same action from the same evidence or moves the record to `REVIEW_REQUIRED` because an input version changed. There is no third outcome.

Two design decisions deserve their reasoning stated. The CLI is stateless between invocations, because a daemon that caches contract state will eventually serve a decision under a contract nobody approved. And findings are stored as data rather than log lines, because a grep over logs cannot answer "how many identity failures did we publish last quarter" with any confidence.

Failure containment is structural. Any unhandled exception in a rule marks that rule `unproven`, not `passed`. Any exception in the policy fold aborts the release with no pointer move. Any exception in the release adapter after the compare-and-swap is reconciled by comparing the pointer record against the attempt record at next startup, and the reconciliation is recorded as its own finding.

```text
[attempt start] --> read pointer --> run rules --> fold policy --> CAS pointer --> write attempt
       ^                                                                  |
       |                                                                  v
       +-------------- reconcile on startup: pointer==attempt ? <--------+
                       mismatch -> emit RECONCILIATION finding, no auto-fix
```

### 8a The CLI's own interface is part of the contract
A quality tool that only a human can interpret is not a gate, because the automation around it will inevitably treat "process exited" as "data is fine". The process interface therefore needs as much specification as the rules do.

| Exit code | Meaning | Orchestration behaviour | Pointer effect |
|---|---|---|---|
| 0 | published, all declared rules ran | proceed to downstream dependencies | moved |
| 10 | blocked on a critical invariant | fail the pipeline, page the source owner | not moved |
| 11 | quarantined with a timer | keep the prior target, schedule re-evaluation | not moved |
| 12 | unproven: a critical-scope rule did not complete | fail the pipeline, page platform on-call | not moved |
| 20 | contract invalid (missing owner, severity, or duplicate rule ID) | stop before any scan | not moved |
| 21 | runner or environment error | retry once, then page platform on-call | not moved |
| 22 | publish permitted with unresolved warnings | proceed only if the consumer declares tolerance | moved |

Seven codes, and the distinction between 10, 11, 12, and 21 is the whole value. At 03:00 the on-call engineer reads the code and knows whether to wake the source owner, the platform owner, or nobody. Collapsing them into one nonzero code forces a log dive, which is exactly the moment nobody has time for one.

Two flags carry disproportionate weight. `--dry-run` executes the plan and the rules but never touches the pointer, and it exists so a cautious operator can compare a proposed action against the recorded one without creating a second decision. `--explain <rule_id>` prints the rule's definition, its evidence, and its arithmetic for one rule, and it exists because "why did this block" is the most common question after an incident and the least commonly supported one.

The output document has a fixed shape: `decision`, `action_set`, `attempt`, `findings[]`, `exclusions[]`, `rules_not_run[]`, and `limits`. The `rules_not_run` array is mandatory even when empty, because its absence is what allows a partial run to be read as a complete one. `limits` records the declared budgets and the achieved values, so a reviewer can see that the run finished in 214 s of a 240 s budget rather than inferring headroom that did not exist.

The idempotency key is `(batch_id, contract_hash)`, the same pair as the decision unit, and the tool refuses to run if an active action already exists for that key while the `--force-new-attempt` flag is absent. The flag exists for investigating a failure; it records the reason in the attempt and does not overwrite the prior action.


## 9. Evaluation & Validation
Every test below states its numerator, denominator, population, window, comparator, guardrails, uncertainty, and execution status. Status is `not-executed` throughout, because no run of this design exists; claiming otherwise would be the exact failure this case warns about.

### T01 — critical identity null at 250 rows
**Numerator:** batches whose action is block when one or more `event_id` values are null.
**Denominator:** batches constructed with at least one null `event_id`.
**Population / window:** 200 synthetic batches, one release window each.
**Comparator:** policy fold result against the hand-computed expected action from the release policy table.
**Guardrail:** zero batches with a null identity key may reach PUBLISHED.
**Uncertainty:** construction is exact, so the only uncertainty is harness error; any mismatch is a defect.
**Status:** not-executed (proposed).

### T02 — completeness denominator regression
**Numerator:** runs where reported completeness differs from the contract-defined value.
**Denominator:** runs containing at least one unparseable row.
**Population / window:** 100 batches with injected parse failures at 0.5%, 2%, and 10%.
**Comparator:** independent counter implemented separately from the runner.
**Guardrail:** reported value must never exceed the corrected value; a higher report is a fail.
**Uncertainty:** none in construction; the counter and runner share a spec, which is a known weakness.
**Status:** not-executed (proposed).

### T03 — seasonal adjustment applied only with a versioned factor
**Numerator:** runs where drift uses a factor not present in the pinned contract.
**Denominator:** runs where the source owner declared a seasonal factor.
**Population / window:** 50 batches across two calendar keys, one factor per key.
**Comparator:** stored contract hash plus factor table hash recorded in the attempt.
**Guardrail:** an undeclared factor must produce an unadjusted drift figure and a review-required state.
**Uncertainty:** the factor's own accuracy is not tested here, only its provenance.
**Status:** not-executed (proposed).

### T04 — timeout must not read as pass
**Numerator:** runs where a cancelled rule is reported with status `passed`.
**Denominator:** runs with a forced cancellation inside a critical rule.
**Population / window:** 40 runs, cancellation injected at 25%, 50%, and 90% of expected scan time.
**Comparator:** state machine trace plus final action.
**Guardrail:** final action must be block or review-required, never publish, in all 40 runs.
**Uncertainty:** cancellation timing granularity is milliseconds; a rule finishing just before cancellation is a legitimate pass and is excluded by construction.
**Status:** not-executed (proposed).

### T05 — duplicate detection beyond the memory cap
**Numerator:** runs where duplicate counts are reported for only part of the batch without a scope flag.
**Denominator:** runs sized above 128 MiB of estimated identity set.
**Population / window:** 20 batches at 8M, 10M, and 14M rows.
**Comparator:** external-sort reference implementation.
**Guardrail:** any partial result must carry `scope=partial` and must not satisfy a full-scan rule.
**Uncertainty:** the 16-byte-per-entry estimate is a model; real footprint varies with allocator behaviour.
**Status:** not-executed (proposed).

### T06 — sample reproducibility
**Numerator:** pairs of runs selecting identical row sets under the same seed and fraction.
**Denominator:** all run pairs executed.
**Population / window:** 30 pairs at fractions 1%, 5%, and 20%.
**Comparator:** hashed row-identity set per run.
**Guardrail:** any divergence must be explainable by a source mutation recorded in the manifest.
**Uncertainty:** none for a static batch; the test is invalid if the object changes mid-test.
**Status:** not-executed (proposed).

### T07 — schema addition without a contract amendment
**Numerator:** runs classifying an undocumented column as acceptable.
**Denominator:** runs where an extra column is injected.
**Population / window:** 60 batches with one, three, and ten extra columns.
**Comparator:** contract field list.
**Guardrail:** action must be review-required, never silent publish.
**Uncertainty:** some producers add debug columns legitimately; the test asserts the process, not the intent.
**Status:** not-executed (proposed).

### T08 — schema removal of a required column
**Numerator:** runs that publish after a required column disappears.
**Denominator:** runs with one required column removed.
**Population / window:** 30 batches, each missing a different required column.
**Comparator:** contract required-field set.
**Guardrail:** block with the missing column named in the finding.
**Uncertainty:** nested or renamed fields may look like removal; the comparator must resolve paths explicitly.
**Status:** not-executed (proposed).

### T09 — freshness with a naive timestamp
**Numerator:** runs where a timestamp without an offset is accepted as fresh.
**Denominator:** runs containing naive timestamps.
**Population / window:** 25 batches across three producer versions.
**Comparator:** contract timezone declaration.
**Guardrail:** quarantine the time rule only; other findings still reported.
**Uncertainty:** a producer with a consistently wrong offset is indistinguishable from a wrong clock without an external source.
**Status:** not-executed (proposed).

### T10 — retry idempotency
**Numerator:** distinct active actions per `(batch_id, contract_hash)` after N retries.
**Denominator:** retried runs.
**Population / window:** 50 runs retried 1, 2, and 5 times, with the same and with a changed contract.
**Comparator:** ledger cardinality.
**Guardrail:** exactly one active action for unchanged inputs; review-required when a contract hash differs.
**Uncertainty:** compute spend may still repeat even when actions do not; that is expected and recorded.
**Status:** not-executed (proposed).

### T11 — rollback restores the prior pointer
**Numerator:** rollbacks where the pointer target equals the recorded prior target.
**Denominator:** rollbacks executed.
**Population / window:** 20 publish-then-rollback sequences including one with an intervening release.
**Comparator:** pointer history records.
**Guardrail:** evidence for the rolled-back batch remains readable.
**Uncertainty:** downstream caches may serve stale data after a correct rollback; the test does not measure cache lag.
**Status:** not-executed (proposed).

### T12 — override without expiry
**Numerator:** overrides accepted with missing owner, rationale, or expiry.
**Denominator:** override attempts with at least one field omitted.
**Population / window:** 18 attempts covering each single omission and each pair.
**Comparator:** policy engine validation.
**Guardrail:** all 18 rejected with a specific reason code.
**Uncertainty:** none; the validation is deterministic.
**Status:** not-executed (proposed).

### T13 — contract hash change mid-run
**Numerator:** runs that continue after the contract hash changes.
**Denominator:** runs with a simulated mid-run contract edit.
**Population / window:** 15 runs, edit injected before planning, during scanning, and after aggregation.
**Comparator:** hash captured at start versus hash at fold time.
**Guardrail:** review-required in all 15, with the diff listed.
**Uncertainty:** a change that only reformats YAML should not alter behaviour, so the test compares parsed-rule diffs as well as raw bytes.
**Status:** not-executed (proposed).

### T14 — alert volume against reviewer capacity
**Numerator:** warnings generated per day.
**Denominator:** declared reviewer minutes per day.
**Population / window:** 30 simulated days at the illustrative 40 alerts/day and at 120 alerts/day.
**Comparator:** acknowledged-versus-generated ratio at day end.
**Guardrail:** unreviewed warnings must appear as an explicit backlog finding rather than disappearing.
**Uncertainty:** simulated acknowledgement behaviour is a model of people, and models of people are the weakest input here.
**Status:** not-executed (proposed).

Acceptance for the rehearsal is a conjunction, not a score: every test above produces its expected state transition, every finding carries scope and provenance, the prior pointer is restored on demand, and no critical rule is ever satisfied by a sample. Failure of any single test blocks the rehearsal rather than being averaged into a pass rate.

## 10. Action Taken
No CLI was deployed and no partition was published by this manuscript. The proposed next actions are sequenced so that each one is falsifiable: obtain an approved contract with severities and owners; build the independent counter before the runner, so the comparator is not written by the same person who wrote the code under test; construct synthetic batches that exercise each severity class; execute the state trace offline; and have the data platform lead decide whether a limited rehearsal on one non-critical feed is justified.

An explicit non-action is also recorded: the team did not lower the identity rule from full scan to a 5% sample, despite the 6.33 USD/day saving it would produce, because Section 6d shows the sample cannot see a 1-in-10,000 defect and the rule exists precisely for that rate.

## 11. Outcome & Results
**Outcome status: not measured.** The only verified results are the recomputed arithmetic in Section 6: 9,750/10,000 = 97.50% against a naive 99.49%; (8,150,000 − 8,600,000)/8,600,000 = −5.23%; 10,000,000/50,000 = 200 s and ρ = 0.3333 with W_q = 1.667 min; a 30,000-row sample detecting a 1e−4 rate with probability 0.950; 180 USD/day of review against 6.667 USD/day of compute, a ratio of 27.0; and 152.6 MiB of identity-hash memory against a 128 MiB cap.

No claim is made about real defect rates, real run times, storage costs, reviewer throughput, or adoption. The figures are planning inputs with verified arithmetic, and the distinction between those two things is the whole point of the case.

## 12. Failure Modes & What Went Wrong
### F-01 — the scanner reports a pass over rows it never read
**Symptom:** a batch publishes with a clean finding set, and a downstream join loses rows the next morning.
**Competing causes:** (a) the parser skipped unreadable rows and the denominator shrank; (b) a rule timed out and its status was recorded as passed; (c) the identity rule ran on a sample rather than a full scan.
**Discriminating check:** compare `rows_seen` with `rows_declared` and inspect each rule's `scope` and `status` fields. Only cause (a) shows a denominator smaller than the manifest count; cause (b) shows `elapsed ≈ budget` with a pass; cause (c) shows `scope=sample` on a rule whose contract says `full_scan`.
**Immediate mitigation:** roll the pointer back to the prior partition, which is cheap because publish was a pointer swap.
**Correction:** make unreadable rows a first-class finding with their own rule and severity, and make `unproven` an unrepresentable-as-pass status in the result schema.
**Verification:** re-run the same batch and confirm that the finding set now contains a parse-failure finding whose count equals the injected count.

### F-02 — the seasonal excuse is written after the miss
**Symptom:** volume drift of −18.5% passes as a −5.2% adjusted figure, and nobody can say when the 0.86 factor was agreed.
**Competing causes:** (a) the factor was genuinely declared before the batch; (b) the factor was added after the shortfall was observed; (c) the factor belongs to a different calendar key.
**Discriminating check:** the factor's contract hash and commit timestamp against the batch's arrival time. A factor introduced after the batch arrived is cause (b); a factor whose calendar key differs is cause (c).
**Immediate mitigation:** downgrade the action to warn-with-rationale and record that the adjustment is contested.
**Correction:** move seasonal factors into the contract with effective-date ranges, and reject adjustments whose effective date is after the batch's own event date.
**Verification:** replay the batch with the earlier contract version and confirm the unadjusted −18.50% blocks or quarantines as the policy dictates.

### F-03 — duplicate detection dies at the memory cap and looks clean
**Symptom:** the duplicate rule finishes quickly on a large batch and reports zero, while the same batch on a smaller partition reports thousands.
**Competing causes:** (a) the hash table was sized correctly and there really are duplicates only in the smaller partition; (b) the worker was killed and a supervisor reported the partial count; (c) the rule silently truncated its scan range.
**Discriminating check:** `M_hat` from Section 6f against the container limit, plus the process exit status and the rule's `rows_examined` count. A truncated range shows `rows_examined < rows_parsed`; an OOM kill shows a nonzero signal status.
**Immediate mitigation:** mark the identity rule unproven and block publication of that rule's scope.
**Correction:** partition the dedupe by a stable hash prefix and reconcile across partitions, or spill to an external sort with an explicit merge step.
**Verification:** run the external-sort reference implementation on the same batch and compare counts exactly, not approximately.

### F-04 — the contract changes while the run is in flight
**Symptom:** two reviewers recompute the same batch and get different actions from what the ledger shows.
**Competing causes:** (a) the contract was edited mid-run; (b) one reviewer loaded a different contract version; (c) the policy fold has an ordering bug that makes results depend on finding insertion order.
**Discriminating check:** hash the contract at planning, at fold time, and in the reviewer's environment. All three equal points to (c); a difference at fold time points to (a); a difference only in the reviewer's environment points to (b).
**Immediate mitigation:** freeze the attempt at review-required and stop recomputation until the versions agree.
**Correction:** pin the contract by content hash for the duration of the attempt, and make the policy fold an explicit reduce over a sorted finding list.
**Verification:** replay with three contracts differing only in key order and confirm identical actions.

### F-05 — warnings accumulate until nobody reads them
**Symptom:** the warn path is used for 40 alerts a day, the acknowledgement queue is 300 items deep, and warnings become a ritual.
**Competing causes:** (a) too many rules are major when they should be minor; (b) each warning takes longer than the six minutes budgeted; (c) the acknowledgement UI does not batch similar findings.
**Discriminating check:** per-rule acknowledgement latency and per-rule finding volume. Cause (a) shows a handful of rules dominating volume; cause (b) shows latency near or above budget across all rules; cause (c) shows high latency with low volume per rule.
**Immediate mitigation:** publish a backlog finding that states the unreviewed count, so the ritual is visible in the release record.
**Correction:** re-severity the dominant rules through a new contract version, and merge findings by rule and column rather than by row.
**Verification:** measure the acknowledged-versus-generated ratio for a declared window before and after the re-severity change.

### F-06 — the timezone assumption fails silently at a boundary
**Symptom:** freshness passes for a producer that writes local time without an offset, and fails for the same producer after a daylight-saving transition.
**Competing causes:** (a) the producer omits the offset; (b) the offset is present but wrong; (c) the gate compares event time to the wrong processing timestamp.
**Discriminating check:** inspect raw values around the transition hour. Cause (a) shows naive strings; cause (b) shows a constant offset that disagrees with the producer's declared zone; cause (c) shows a processing timestamp drawn from a later stage than declared.
**Immediate mitigation:** quarantine the time rule and report the affected hour range.
**Correction:** reject naive timestamps at parse and require an explicit zone identifier per producer version.
**Verification:** replay the transition day and confirm the gate either reports a one-hour band or refuses the timestamps, never a silent pass.

### F-07 — rollback works but downstream still reads the old data
**Symptom:** the pointer was rolled back, the release record is correct, and a dashboard still shows the bad partition six hours later.
**Competing causes:** (a) a downstream cache holds the previous read; (b) a downstream job materialized the bad partition into its own table; (c) the pointer was rolled back to the wrong target.
**Discriminating check:** query the pointer record and each consumer's own metadata. Cause (a) shows correct pointer plus stale cache key; cause (b) shows a derived table whose load time postdates the bad publish; cause (c) shows a pointer target that never appeared in the good history.
**Immediate mitigation:** publish the rollback as an event consumers must handle, rather than assuming they poll.
**Correction:** add a contract requirement that consumers record the pointer generation they read, so a stale read is detectable.
**Verification:** replay a publish-then-rollback with an instrumented consumer and confirm the generation recorded matches the final pointer.

### F-08 — the gate becomes the place where policy is decided
**Symptom:** an operator edits a threshold in the code during an incident, the batch publishes, and the next review cannot reconstruct why.
**Competing causes:** (a) thresholds live in code rather than the contract; (b) the override mechanism was bypassed; (c) the contract allows an unowned rule to exist with a default severity.
**Discriminating check:** compare the runner revision and contract hash in the attempt record with the values in version control at the incident time. A revision bump with no contract change points to (a); a matching revision with a publish despite a critical finding points to (b); a rule with no owner in the contract points to (c).
**Immediate mitigation:** treat the release as review-required and reconstruct the evidence from immutable inputs.
**Correction:** move every threshold into the contract, require owner and severity at load time, and make the runner refuse to start when either is absent.
**Verification:** attempt to start the runner with a contract missing an owner and confirm a startup failure, not a warning.

### F-09 — the manifest count and the recount disagree and nobody decides which is authoritative
**Symptom:** the volume rule uses 10,000,000 from the manifest while the parser saw 9,400,000 rows; the drift rule reports −6.0% and the completeness rule uses a different denominator, so the report contains two populations.
**Competing causes:** (a) the manifest count was written before the producer finished writing; (b) the parser dropped rows without counting them as failures; (c) the object prefix contains objects from a previous attempt that the manifest does not describe.
**Discriminating check:** compare `rows_declared`, `rows_parsed + parse_failures`, and an independent object-level count. (a) shows declared < parsed for a partial write; (b) shows parsed + failures < declared with no excluded objects; (c) shows more objects than the manifest lists, with earlier modification times.
**Immediate mitigation:** mark the run unproven and treat the release as review-required, because two populations in one report cannot both be right.
**Correction:** declare one authoritative count in the contract — the object-level recount — and make the manifest count a *finding* rather than a denominator input.
**Verification:** replay a batch where the manifest is deliberately understated by 5% and confirm the report contains one population and one manifest-mismatch finding.

### F-10 — a passing rule is deleted rather than fixed
**Symptom:** identity failures vanish from the report after a change, and the batch publishes; three weeks later a join fails on keys nobody checked.
**Competing causes:** (a) the rule was removed from the contract; (b) the rule's projection changed so it reads a column that is always populated; (c) the rule's severity was downgraded to informational and filtered out of the summary.
**Discriminating check:** diff the contract between the two attempts and inspect the finding set, not the summary. A missing `rule_id` points to (a); a rule present with `rows_examined` far below the batch size points to (b); a finding present with severity `informational` outside the summary points to (c).
**Immediate mitigation:** restore the prior contract hash for the next decision and treat the interval as unproven for identity.
**Correction:** require that removing or re-projecting a critical rule is a contract change with a stated replacement, and make the report enumerate rule IDs so absence is visible rather than silent.
**Verification:** compare the rule ID sets of two consecutive attempts and assert the set difference is empty for critical severity, or explained in the contract.

**Takeaway:** each failure above is a case where the system produced a plausible answer while its evidence was invalid. The countermeasure is never "monitor more"; it is to make invalid evidence unrepresentable in the result schema.

## 13. Transferable Lesson
- A quality gate is a release decision over a contract, so its unit is the `(batch, contract)` pair, not a score.
- Denominators decide stories. Compute every rate over physical rows and attribute parse failures explicitly.
- Seasonal and calendar adjustments are evidence with versions and effective dates, never verbal context applied after the fact.
- Sampling is a scope statement. If a rule's tolerable defect rate is below 1/n, sampling cannot discharge it and saying so is the honest output.
- Cost the reviewer, not only the scanner: at 180 USD/day against 6.67 USD/day the queue is the system.
- Memory limits are correctness limits when a check needs a resident set; a truncated scan that reports zero is a silent pass.
- Make unproven, partial, and superseded unrepresentable as passing states in the result schema.
- Point at data with a compare-and-swap pointer so rollback is a metadata operation rather than a re-run.

| Dimension | Manual notebook review | This CLI design | What transfers to another domain |
|---|---|---|---|
| Unit | "the file" as a whole | `(batch_id, contract_hash)` | the decision must be pinned to an immutable input pair |
| Evidence | cells executed in a kernel | append-only ledger with scopes and hashes | evidence must outlive the session that produced it |
| Denominator | whatever the query returned | every physical row, exclusions enumerated | the population definition is a policy statement |
| Comparison | last time it looked odd | declared seasonal factor with effective date | expectations are versioned artefacts, not memories |
| Cost model | analyst hours, unmeasured | 180 USD/day review against 6.67 USD/day compute | the human queue is usually the binding constraint |
| Failure shape | someone notices late | unproven is a distinct, non-passing state | make invalid evidence unrepresentable in the schema |
| Recovery | re-ingest and hope | pointer swap to a retained prior target | reversibility is a metadata design property |
| Change control | thresholds edited in code | contract hash pinned per attempt | configuration is evidence and must be hashed |

Read the table as a checklist rather than a menu: a domain that cannot name its immutable input pair, its denominator, and its reverse path is not ready for automation, and building the tool will only make the ambiguity faster.

## 14. Connections & Cross-References
This case depends on Python process and I/O control, SQL-style aggregation semantics for the counters, Linux-style file and permission boundaries for quarantine, and REST-style resource identity for the pointer. Those techniques constrain the implementation; they do not supply the ownership or evidence discipline, which is the actual subject of the case.

### Interview pack
#### Q1. What exactly is the decision unit, and why is a batch alone insufficient?
**Answer:** The unit is `(batch_id, contract_hash)`. A batch alone cannot be released or refused because "released under which rules" is undefined; the same rows can pass a lenient contract and block under a strict one.
**Reasoning:** every finding, threshold, and severity comes from the contract, so the evidence set is a function of both inputs. Attaching an action to the batch alone makes a later contract edit silently reinterpret a past decision.
**Follow-up:** ask which fields in the attempt record would let you reconstruct the decision a year later; the answer must include the contract hash and runner revision.
**Red flag:** describing the unit as "the dataset" or "the table", which cannot be pinned.
**Scoring:** full credit names both components, states the consequence of dropping either, and identifies the record that proves the pairing.

#### Q2. Completeness on a batch is 99.49% one way and 97.50% another. Compute both and say which is correct.
**Answer:** With 10,000 physical rows, 200 parse failures, and 9,750 non-null identity values, 9,750/9,800 = 99.49% hides the failures; 9,750/10,000 = 97.50% is the contract value, because the denominator is every physical row.
**Reasoning:** dropping 200 unparseable rows removes exactly the rows most likely to be broken, so the naive figure improves as the data worsens. The 1.99-point gap is the measurement of that self-flattery.
**Follow-up:** ask what happens if parse failures are concentrated in a different column; the correct answer is per-column attribution plus a separate whole-row rule.
**Red flag:** choosing the larger number because it is "more accurate to what parsed".
**Scoring:** full credit shows both divisions, the gap in percentage points, and the attribution caveat.

#### Q3. Volume is down 18.5% raw and 5.23% after a declared holiday factor of 0.86. What action, and what would change it?
**Answer:** Warn with owner rationale, since −5.23% sits 0.23 points outside a ±5% band. It becomes block if the factor was not declared before the batch arrived, and it becomes publish if the band is ±6% under an approved contract version.
**Reasoning:** the adjusted figure is only as good as the factor's provenance. A factor added after the shortfall was seen is a narrative, not an expectation.
**Follow-up:** ask for the row equivalent of the breach; 0.0023 × 8,600,000 ≈ 20,000 rows converts a percentage argument into an operational one.
**Red flag:** applying the factor without recording where it came from and when it took effect.
**Scoring:** full credit computes the expected count, both drifts, and at least one condition that flips the action.

#### Q4. A full scan takes 200 seconds at 50,000 rows/s and batches arrive every 10 minutes. Is there slack, and where does it vanish?
**Answer:** Utilisation is 0.3333 and expected queue wait is 1.667 minutes, so there is slack. It vanishes when per-batch service exceeds 10 minutes, for example a 15-minute degraded scan, where ρ = 1.5 and backlog grows 2 batches per hour.
**Reasoning:** ρ < 1 is necessary but not sufficient for comfort; the waiting time scales as ρ/(1−ρ), so a service time creeping toward the arrival interval degrades far faster than linearly.
**Follow-up:** ask what to do at ρ = 1.5. The honest options are raise capacity, lower arrival rate by dropping non-critical rules, or accept lag; silently skipping critical rules is not on the list.
**Red flag:** comparing average service time with the arrival interval and declaring safety, without checking the distribution or the cold-read rate.
**Scoring:** full credit computes S, μ, ρ, W_q, and names the regime where the model fails.

#### Q5. How large a sample do you need to detect a defect rate of one in ten thousand with 95% confidence?
**Answer:** About 30,000 rows, since n ≈ 3/p = 3/0.0001. A sample of 1,000 detects it with probability 1 − 0.9999^1000 ≈ 9.5%.
**Reasoning:** independent detection is a geometric-style waiting problem, so the complement scales exponentially in n. The Poisson cross-check with np = 3 gives 1 − e^−3 = 95.0%, agreeing with the binomial figure.
**Follow-up:** ask when the calculation is invalid. Clustered defects break independence, so a per-shard and per-hour count is the discriminating check.
**Red flag:** claiming a passed sample proves absence at any rate, or quoting 95% without the sample size that produced it.
**Scoring:** full credit derives n from p, computes the 1,000-row power, and states the homogeneity assumption.

#### Q6. Compute the daily cost of scanning and the daily cost of reviewing, then name the bottleneck.
**Answer:** Scanning is 3.333 minutes × 0.20 USD = 0.6667 USD per batch, 6.67 USD per day at ten batches. Review is 40 alerts × 6 minutes × 0.75 USD = 180 USD per day. The ratio is 27.0, so review is the bottleneck.
**Reasoning:** compute scales with rows and is nearly free at this size; attention scales with rule count and severity choices, which are policy decisions the team controls directly.
**Follow-up:** ask what a 5% sample would save (6.33 USD/day) and whether that saving is worth an undetectable identity defect; it is not.
**Red flag:** optimising the scanner because it is the part with a profiler.
**Scoring:** full credit computes both sides, the ratio, and explains which lever the ratio justifies.

#### Q7. Duplicate detection needs 152.6 MiB and the container allows 128 MiB. What actually happens, and how do you detect it?
**Answer:** The worker dies at roughly 84% of the batch, and a naive supervisor reports the partial count. You detect it by comparing `rows_examined` with `rows_parsed`, checking the exit signal, and comparing `M_hat` with the limit.
**Reasoning:** 10,000,000 × 16 B = 160,000,000 B = 152.6 MiB, so the failure is deterministic rather than occasional, which means it will reproduce if anyone looks.
**Follow-up:** ask for two remedies and their costs. Partition-level dedupe with cross-partition reconciliation, or external sort with bounded merge; both add I/O and both state their scope honestly.
**Red flag:** raising the container limit and calling the problem solved when the next batch is 20M rows.
**Scoring:** full credit computes the memory, names the truncated-result failure, and proposes a scope-honest remedy.

#### Q8. Two engineers recompute the same batch and get different actions. Diagnose it.
**Answer:** Three candidates: the contract changed mid-run, one engineer loaded a different contract version, or the policy fold depends on finding order. Contract hashes at planning, fold time, and in each environment separate them.
**Reasoning:** the fold must be a reduce over a sorted set; if insertion order matters, then two runs of the same evidence produce different decisions, which destroys reproducibility more thoroughly than a threshold dispute.
**Follow-up:** ask how to prove the fold is order-independent. Shuffle the finding list under a fixed seed and assert identical actions across many permutations.
**Red flag:** reconciling by asking which engineer is more senior.
**Scoring:** full credit gives three causes, the hash comparison that discriminates them, and the sorted-reduce correction.

#### Q9. What does a timeout mean, and which state must it never reach?
**Answer:** A timeout means the property is unproven. It must never reach the passing state, and publication of rules it covers must be blocked or sent to review.
**Reasoning:** the difference between "no violations found" and "no violations were looked for" is invisible in a report that prints a count of zero, which is exactly how a cancelled scan becomes a clean bill of health.
**Follow-up:** ask how to encode it. Make unproven a distinct status in the result schema and make the policy fold treat it as unresolved for critical scope.
**Red flag:** retrying until something returns quickly, which converts a capacity problem into an evidence problem.
**Scoring:** full credit distinguishes unproven from failed and from passed, and names the schema change that enforces it.

#### Q10. Why must the gate never repair source data?
**Answer:** Because a repair destroys the evidence the decision was made on, and the decision then cannot be audited against the rows as received. Repair is a separate transformation with its own version and approval.
**Reasoning:** auditing asks "what did we receive, what did we conclude, what did we do". A silent fix makes the first and third indistinguishable, so a later dispute has no ground truth.
**Follow-up:** ask where a fix belongs. A versioned transformation that writes a new batch with a link to the input hash, so both states remain addressable.
**Red flag:** framing repair as a convenience that "saves a whole re-ingest".
**Scoring:** full credit ties repair to evidence destruction and prescribes the versioned-transformation pattern.

#### Q11. The seasonal factor is 0.86 and the observed count is 8.15M against a 10M reference. Show that the adjustment is arithmetically consistent, then attack it.
**Answer:** Expected is 8.6M, and (8.15M − 8.6M)/8.6M = −5.23%, consistent with a 450,000-row shortfall. The attack is that volume is a first moment: a multi-hour gap followed by a backfill restores the total while leaving the arrival distribution wrong.
**Reasoning:** the formula constrains one number, so any defect that preserves the total is invisible to it. The per-hour histogram compared with the same calendar key is the check that sees it.
**Follow-up:** ask what a large positive drift usually means; typically duplication or fan-out, not growth.
**Red flag:** treating the adjusted figure as a validation of the factor itself.
**Scoring:** full credit recomputes the shortfall, states the moment limitation, and names the histogram check.

#### Q12. What is the central invariant of the release design, and how would you test it?
**Answer:** At most one active action per `(batch_id, contract_hash)`, derivable from the ledger plus the pinned contract. Test it by retrying the same batch one, two, and five times and asserting ledger cardinality of one, then changing the contract hash and asserting review-required.
**Reasoning:** idempotency is not about avoiding repeated work; it is about ensuring a retry cannot create a second, contradictory decision that later readers must arbitrate.
**Follow-up:** ask whether repeated compute spend is acceptable under the invariant. Yes, and it should be recorded, because hidden repeated spend is how a retry loop becomes a budget incident.
**Red flag:** implementing idempotency by checking "did we already publish" without recording which attempt published.
**Scoring:** full credit states the invariant, the derivation requirement, and the two-part test.

#### Q13. Rollback is a pointer swap and it worked, yet a dashboard is still wrong. What do you check?
**Answer:** Three things: pointer history shows the correct target, consumer caches hold stale reads, or a consumer materialised the bad partition into its own table before the rollback.
**Reasoning:** a pointer swap only changes what new readers see. Anything that copied bytes or cached a result has its own retention, so rollback is a notification problem as much as a metadata problem.
**Follow-up:** ask what contract requirement reduces the ambiguity. Consumers record the pointer generation they read, so a stale read becomes detectable rather than arguable.
**Red flag:** concluding rollback failed because a dashboard lags.
**Scoring:** full credit separates pointer correctness from consumer lag and prescribes generation recording.

#### Q14. You may not lower the identity rule to a sample. What would you tell a manager who points at the saving?
**Answer:** The saving is 6.33 USD/day against 180 USD/day of review, so it is economically irrelevant, and it removes the only check that can see a 1-in-10,000 identity defect. The correct response is to leave the rule and reduce alert volume elsewhere.
**Reasoning:** the decision trades a negligible cost reduction for an unquantified correctness loss on a join-critical key. Any argument for it has to quantify the downstream cost of broken joins, which nobody has.
**Follow-up:** ask which rules *can* be sampled; those whose tolerable defect rate exceeds 1/n, which is a per-rule question the contract must answer explicitly.
**Red flag:** accepting the sample because the batch "looked fine" at 1,000 rows.
**Scoring:** full credit compares the magnitudes and grounds the refusal in detection power rather than taste.

#### Q15. Design the exit codes and say why an operator should care.
**Answer:** Distinct nonzero codes for block, quarantine, unproven, and tool error, so orchestration can distinguish "the data is bad" from "the tool failed". All nonzero block publication; only zero permits a pointer move.
**Reasoning:** an operator triaging at 03:00 needs to know whether to call the source owner or the platform on-call, and a single nonzero code forces them to read logs to find out.
**Follow-up:** ask what a successful run with warnings returns; a distinct warning code that still permits publish only after acknowledgement.
**Red flag:** returning zero when the run was incomplete because "no findings were raised".
**Scoring:** full credit maps each terminal state to one code and ties the code to an on-call action.

#### Q16. Which single changed constraint would most likely force a redesign?
**Answer:** A requirement that the decision be available within one minute of partition completion. At 200 s per scan with 90% of a batch needing full scans, the bounded budget forces metadata-first ordering, incremental checks, and an explicit provisional decision state.
**Reasoning:** the design currently spends its budget scanning; compressing the window means spending it on ordering rules by decision value, so the report must be able to say "these three critical rules ran, these five minor rules are pending".
**Follow-up:** ask whether provisional publication is acceptable; usually only for a named consumer with a declared tolerance, never as a default.
**Red flag:** promising the window by shortening budgets uniformly, which converts critical checks into unproven ones.
**Scoring:** full credit names the constraint, computes why it bites, and proposes ordering rather than speed.

## 15. Quick Reference Card
| Field | Rule |
|---|---|
| Decision unit | `(batch_id, contract_hash)` pair; the pair is what an action attaches to |
| Actions | publish, warn, quarantine, block, retain-prior; exactly one active action per pair |
| Completeness | non-null required values / physical rows, parse failures retained and attributed per column |
| Volume drift | (observed − reference × seasonal) / (reference × seasonal); raw drift is diagnostic only |
| Runtime | S = N/r; ρ = λ/μ; W_q = ρ/(μ−λ) is meaningless once ρ ≥ 1 |
| Sampling | a rule needing a rate below 1/n must run as a full scan |
| Memory | identity set ≈ 16 B per distinct key; exceeding the cap makes the check unproven, not clean |
| Cost levers | review at 180 USD/day dominates compute at 6.67 USD/day by 27× |
| Invariants | no silent repair; no pointer move outside publish and rollback; unproven never equals passed |
| Evidence | illustrative arithmetic only; no batch was published under this design |
| Escalate | identity ambiguity, contract change, timeout, memory cap, or authority gap |
