---
case_id: "CASE-02-002"
case_title: "Design an Analytics API"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "contract-first-api"
secondary_patterns: ["correctness-contract", "bounded-work"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "analytics engineering lead"
source_index: "02_python_sql_linux_and_engineering_basics/Case_Studies/INDEX.md"
review_trigger: "Any change to workload, decision authority, acceptance metric, or table layout"
---

# Design an Analytics API

**Evidence declaration:** every workload figure, latency, price, and defect rate below is an ILLUSTRATIVE constructed input. The arithmetic performed on those inputs is VERIFIED and reproducible from the stated values; none of it is a measurement of a deployed service, and no company API is described. **Implementation stage:** proposed. **Editorial status:** draft; source verification and publication review remain OPEN.

The subject is not "how to write a REST endpoint". It is how to decide, in advance, what an analytics API will refuse to do — which date ranges, which row counts, which result sizes — so that its latency, cost, and correctness claims are consequences of the contract rather than discoveries made by users at 2 a.m.

## 1. The One-Line Framing
Given a stated query workload and an 800 ms synchronous latency budget, the analytics engineering lead must decide which queries the API answers inline, which it defers to an asynchronous job, which it refuses with a documented error, and how it proves that two callers asking the same question get the same answer.

The decision unit is one `(endpoint, query_shape, caller_class)` triple, and the deliverable is a contract: bounds, error taxonomy, consistency semantics, and evidence. Not a schema diagram.

## 2. Context & Background
The current baseline is manual extracts: an analyst writes SQL, runs it in a warehouse console, and pastes a CSV into a shared folder or a chat message. Nobody can say which numbers are current, two analysts routinely produce different figures for the same metric because they wrote different `WHERE` clauses, and each extract is a fresh full scan because nothing is cached beyond the console's own session.

That baseline has real virtues and they must be named, or the replacement will lose them. It is infinitely flexible: any question can be answered by writing new SQL. It requires no service to be operated. Its failure model is a human noticing that a number looks wrong, which works surprisingly well when the consumer is the same person who wrote the query. And it has no rate limit, so a slow query only punishes its author.

Its weaknesses are equally concrete: there is no shared definition of a metric, so "active users" differs between two dashboards; there is no bound on cost, so one careless `SELECT *` across ninety days can consume a meaningful fraction of a monthly compute allowance; there is no reproducibility, because a table that is being written while a query runs yields a different answer each time; and there is no machine interface, so every recurring report is a human in the loop.

The proposal is a read-only HTTP API with a small number of query shapes, each with declared bounds, a documented error taxonomy, cursor-based pagination, and explicit consistency semantics. It does not replace the warehouse console; it removes the recurring 80% of questions from it and makes those answers identical for every caller.

Scale is constructed as follows and is the only basis for any number in this manuscript: 40,000,000 event rows per day, 24 bytes per row in the scanned projection, 1.2 GB/s of scan throughput for a single query, 1,200,000 API requests per day, and 64 application workers. Every sizing conclusion downstream is a function of exactly those five values.

Timing matters in a specific way here: the API computes over a table that is continuously appended to. So "the answer" is only meaningful relative to a snapshot boundary, and the contract must state whether a caller receives a snapshot at request start, a snapshot at query end, or the latest available rows at each shard's own read time. Those three choices produce three different products.

Ownership is split. The analytics engineering lead owns the contract and its bounds. The data platform team owns the underlying table layout and watermark discipline. The consuming teams own their tolerance for staleness, declared per caller class. Nobody is allowed to raise a bound to make one dashboard faster, because a bound is a promise made to every caller.

The API is explicitly not an authorization system, not a row-level security boundary, and not a data export mechanism for regulated data. Those require controls that a query contract cannot supply.

## 3. The Business Question
| Field | Contract |
|---|---|
| Question | For each `(endpoint, query_shape, caller_class)`, does the API answer inline, defer to an async job, or refuse? |
| Decision owner | Analytics engineering lead owns the contract; data platform owns layout and watermark; consuming teams own staleness tolerance. |
| Trigger | A new query shape request, a workload change beyond a declared threshold, or a latency/cost regression against the declared budget. |
| Inputs at decision time | Declared latency budget, per-shape cost model, table watermark, caller class limits, observed p99 service times, error budget. |
| Actions | Serve inline; serve from cache; defer to async job with a job ID; refuse with a typed error and a documented remedy. |
| Objective | Make every promise about latency, cost, and repeatability true by construction, and make every refusal legible. |
| Hard constraints | 800 ms synchronous budget; date range must be bounded; results must be reproducible against a named snapshot; no unbounded result sets. |
| Scope | One read-only API surface over one event fact table plus its dimension tables. |
| Out of scope | Write paths, row-level authorization, exports of regulated data, ad-hoc SQL passthrough, streaming subscriptions. |

The reason the question is framed per triple rather than per endpoint is that the same endpoint is a different product depending on the shape. A rollup over seven days by day and region is a 6.72 GB scan and belongs inline. A raw event scan over ninety days is an 86.4 GB scan and cannot be inline under any budget. Both are `GET /events`; they are not the same decision.

### 3a The decision table that is the actual deliverable
| Query shape | Rows scanned (illustrative) | Scan time at 1.2 GB/s | Scan cost at $5/TB | Decision | Documented reason |
|---|---|---|---|---|---|
| Rollup, 1 day, ≤2 dimensions | 40,000,000 | 0.80 s | $0.0048 | inline only on cache hit | an uncached scan exceeds the 340 ms scan share |
| Rollup, 7 days, ≤2 dimensions | 280,000,000 | 5.60 s | $0.0336 | async job, cached rollup preferred | 5.6 s exceeds the whole 800 ms budget |
| Rollup, 30 days | 1,200,000,000 | 24.00 s | $0.1440 | async job | 30× the budget |
| Rollup, 90 days | 3,600,000,000 | 72.00 s | $0.4320 | async job, requires approval for cost above a threshold | largest supported range |
| Raw scan, ≤1 day, page ≤500 rows | 40,000,000 | 0.80 s | $0.0048 | inline with cursor | only when a selective filter is present |
| Raw scan without a selective filter | 40,000,000+ | ≥0.80 s | ≥$0.0010 | refuse, typed error | unbounded result set and unbounded egress |
| Distinct count over >7 days | 280,000,000+ | ≥5.60 s | ≥$0.0336 | async, approximate above a cardinality threshold | exact distinct does not fit the budget |
| Any range >90 days | >3,600,000,000 | >72 s | >$0.4320 | refuse, typed error | outside the declared support envelope |

**Takeaway:** the table is the contract. Every row states an observable input, a computed consequence, and a refusal reason a caller can act on.

## 4. Assumptions & Constraints
| ID | Assumption or constraint | How it is checked | If it is wrong | Fallback | Owner |
|---|---|---|---|---|---|
| A-01 | Scan throughput is 1.2 GB/s per query at the declared projection | timed scan of a partition at declared size | cost and latency models are wrong | re-derive bounds from the measured rate | platform |
| A-02 | A row in the scanned projection costs 24 bytes | measure bytes scanned for a known row count | all scan-time figures shift | use byte-based bounds instead of row-based | platform |
| A-03 | Table watermark advances within a declared lag | compare watermark to wall clock per partition | a "latest" query silently omits data | serve snapshot-at-request and state the boundary | platform |
| A-04 | Cache keys are derived from the full query shape including filters | key-collision test with differing filters | one caller receives another's numbers | include a canonicalized filter hash in the key | api |
| A-05 | Caller classes have declared request budgets | compare per-class counters with declared limits | one client starves others | enforce per-class token bucket | api |
| A-06 | Cursor pagination is stable under concurrent inserts | insert rows mid-pagination and compare pages | duplicates or gaps across pages | composite keyset cursor on (timestamp, id) | api |
| A-07 | Approximate distinct is acceptable above a declared cardinality | compare sketch against exact on a sample | reported uniques are wrong by ~1.6% | force exact for critical shapes or label the error | analytics |
| A-08 | Async jobs are idempotent by query hash | submit the same job twice, compare job IDs and results | duplicate compute spend and divergent results | deduplicate on the hash within a window | api |
| A-09 | Error responses never include partial data with a success status | fault injection at each stage | callers act on truncated results | fail closed with a typed error | api |
| A-10 | Retries by clients are safe for read-only GETs | replay a GET and compare bytes | retries multiply load during an incident | document retry-after and make GETs side-effect free | api |
| A-11 | Dimension joins are small enough to broadcast | compare dimension size with worker memory | joins spill and latency doubles | pre-join into the rollup layer | platform |
| A-12 | 800 ms is a product decision, not a physical constant | owner sign-off recorded in the contract | bounds are arbitrary and drift | treat as a versioned contract field | analytics lead |
| C-01 | No unbounded result set is ever returned | reject queries without a range predicate | egress and memory blow up | typed error with a suggested narrower shape | api |
| C-02 | No inline query exceeds the scan share of the budget | per-query scan-time guard | p99 collapses for everyone | defer to async, never truncate silently | api |
| C-03 | Results name the snapshot they were computed from | snapshot ID present in every response | results are not reproducible | refuse to serve without a snapshot ID | platform |
| C-04 | Rate limits are enforced before parsing | token bucket check precedes validation | cheap requests still consume expensive setup | reorder the pipeline | api |

The two assumptions that fail silently are A-03 and A-04. A stale watermark produces a plausible number that is merely incomplete, and a cache-key collision produces a plausible number that belongs to someone else. Neither raises an exception, which is why both appear in Section 12 as failure modes rather than in Section 9 as edge cases.

**Takeaway:** an assumption that fails loudly is an inconvenience; an assumption that fails plausibly is a data incident.

## 5. Data, Tools & Method
### 5a Data Specification
| Element | Specification |
|---|---|
| Decision unit | one `(endpoint, query_shape, caller_class)` triple, versioned in the contract |
| Fact grain | one event row keyed by `event_id`, with `event_time`, `received_time`, `partition_date`, and `tenant_key` |
| Dimensions | two small tables keyed by `region_id` and `device_id`, both under 10,000 rows so they broadcast to every worker |
| Scan projection | eight columns, 24 bytes per row as stored: id (8), timestamp (8), two small integers (4), region (2), device (2) |
| Snapshot boundary | one watermark per partition; a response names the maximum watermark it observed and the partition set it read |
| Event time vs processing time | `event_time` is the source claim; `received_time` is platform write time; a query filters on `event_time` and reports the watermark from `received_time` |
| Population | rows whose `event_time` falls in the requested closed-open interval and whose partition was readable at the snapshot |
| Units | counts in rows, sizes in bytes, durations in seconds or milliseconds as labelled, money in USD, rates dimensionless |
| Result size | at most 500 rows per page, at most 4 MB serialized per response |
| Cache key | canonical endpoint, canonical filter set, dimension set, granularity, snapshot bucket, and contract version, hashed |
| Caller classes | `interactive` (dashboard), `batch` (scheduled report), `exploratory` (analyst tooling); each has its own token bucket |
| Exclusions | partitions unreadable at snapshot, rows whose `event_time` is outside the interval, and rows in quarantined partitions; each counted in the response envelope |
| Not available at decision time | future late arrivals, restatements of past partitions, and the caller's downstream use of the result |
| Approximate fields | unique-count fields above a declared cardinality use a register sketch and are labelled in the response |
| Identifier stability | metric definitions live in the contract and carry a version; changing a definition changes the version, never the identifier |

Two specification choices deserve their reasoning stated because they are the ones teams get wrong.

First, the snapshot boundary is per partition rather than global. A global snapshot across a hundred partitions would require a coordinated read that no affordable storage layer provides. A per-partition watermark is achievable, and it is sufficient *if* the response names the maximum observed watermark and the caller is told that partitions may differ. Pretending to a global snapshot that does not exist is the more comfortable lie and the more expensive one.

Second, `event_time` filtering with `received_time` reporting is deliberate. Filtering on `received_time` makes "yesterday's data" mean "what we received yesterday", which changes retroactively as late rows land; filtering on `event_time` makes the interval stable while the contents still settle. The watermark then tells the caller how settled they are.

### 5b Tools & Stack
- HTTP service in Python with a small typed router; three endpoints only: `/rollup`, `/events`, `/jobs/{id}`.
- Query planner that converts the request into either a single-pass aggregate or a bounded page scan, and rejects shapes it cannot plan within budget.
- Token bucket per caller class, checked before request parsing so abusive traffic costs almost nothing.
- Cache in front of the planner keyed by the canonicalized request hash, with the snapshot bucket as an explicit component so a cached result cannot silently cross a boundary.
- Async job runner with a job record keyed by query hash, so submitting the same long query twice within a window returns the same job rather than doubling compute.
- Cursor codec that encodes `(event_time, event_id)` and signs it, so a client cannot forge a cursor into an unbounded scan.
- Response envelope carrying `snapshot`, `watermark`, `rows_scanned`, `rows_returned`, `truncated`, `approximate`, and `exclusions`.
- Error taxonomy as constants with stable codes, documented remedies, and a `retry_after` field where relevant.
- Clock and storage adapters injected so that watermark behaviour and budget behaviour can be tested without production data.

The deliberate omission is a query language. A general expression endpoint would return the flexibility of the console while discarding every bound in the contract, because the service could not reason about a shape it did not define.

### 5c Method Summary
1. Authenticate, then check the caller-class token bucket before doing anything expensive.
2. Parse and canonicalize the request; reject unknown fields rather than ignoring them, because an ignored filter silently changes the answer.
3. Compute the plan and its estimated bytes scanned; compare with the shape's declared bound.
4. If the estimate exceeds the inline bound, return a typed deferral or refusal, never a truncated answer.
5. Serve from cache when the snapshot bucket permits; otherwise execute against the named snapshot.
6. Assemble the envelope, including scanned bytes and exclusions, and sign the next cursor when more rows exist.
7. Emit one structured log record per request with the decision, so refusal rates are measurable rather than anecdotal.

Step 4 is the one that gets negotiated away in practice. "Just return the first 500 rows and a warning" is how an unbounded query becomes a silently truncated result that a dashboard renders as a total. The contract forbids it: an over-budget query refuses or defers, and the difference between those two is whether the shape has an async path at all.

### 5d Data provenance, quality, exclusions and leakage controls
- Every response names its snapshot: the maximum watermark observed, the partition set, and the contract version.
- A cached response is only valid within its snapshot bucket; crossing a bucket boundary requires re-execution, even if the bytes are identical, because the caller's reproducibility claim depends on the label.
- Rows excluded for unreadable partitions are counted in the envelope rather than dropped, so a partial result is visibly partial.
- Approximate fields are labelled at field level, not in a footnote, because a footnote is not read by the code that consumes the number.
- Metric definitions live in the contract with versions; a definition change produces a new contract version, so a dashboard pinning the old version keeps its old numbers.
- No filter is silently ignored. Unknown or unsupported fields are a typed error, because a dropped filter is a wrong answer delivered confidently.
- The API never returns a partial page with a success status. If serialization exceeds the size bound, the page size shrinks and the response says so.
- Test fixtures include boundary snapshots — a partition mid-write, a partition with zero rows, a partition whose watermark is behind — so the untidy cases are exercised rather than assumed away.
- Downstream dashboard corrections are never fed back as ground truth for the API's own correctness measurement; that would make the API look right after the fact.

**Takeaway:** provenance in an API means a response can be reproduced from its own metadata, without asking the service what it meant.

### 5e Reproducibility and method-selection ledger
| ID | Artifact | Purpose | Access | Limitation |
|---|---|---|---|---|
| CT-API-01 | endpoint contract with bounds | declares shapes, limits, errors | proposed | not an approved service policy |
| WL-01 | workload profile | request mix and peak factor | illustrative | constructed, not observed traffic |
| BUD-01 | latency budget decomposition | allocates the 800 ms | proposed | component times are planning inputs |
| COST-01 | scan and egress price sheet | converts bytes to dollars | illustrative | prices are assumptions, not quotes |
| ORC-API-01 | independent recomputation harness | checks rollups against raw rows | proposed | validates declared properties only |
| SNAP-01 | snapshot and watermark rules | reproducibility basis | proposed | per-partition, not global |

Method-selection rationale, recorded because the alternative is defensible: a GraphQL-style flexible surface was rejected in favour of three fixed shapes. Flexibility would have let callers express shapes the service cannot bound, which reintroduces the console's cost profile with an HTTP status code in front of it. Fixed shapes are a product limitation and a correctness guarantee at the same time.

## 6. Analysis Walkthrough
Every input below is ILLUSTRATIVE. Every arithmetic result is VERIFIED by recomputation, including an independent second route where the algebra admits one.

### 6a Concurrency sizing from Little's law, and why the average under-provisions
```text
FORMULA [API-01]: L = lambda * W ;  workers >= L / target_util ;  lambda = D * f / 3600
WHERE: D [requests/day]; f the share of daily volume in the peak hour (dimensionless);
       lambda [requests/s] peak arrival rate; W [s] mean time in system; L [requests] concurrent
       in flight; target_util dimensionless, typically 0.7.
CONSTRAINT: steady state over the peak hour, stable W, no queue collapse; valid only while
            utilisation < 1, and it assumes arrivals are independent of service time.
NUMERICAL EXAMPLE:
  D = 1,200,000 requests/day ; f = 0.18 -> lambda = 1,200,000*0.18/3600 = 60.0 req/s
  mean rate for comparison: 1,200,000/86400 = 13.89 req/s
  rollup traffic share 0.90 -> lambda_r = 54.0 req/s ; W_r = 0.15 s -> L_r = 8.1 in flight
  scan traffic share 0.10  -> lambda_s = 6.0 req/s  ; W_s = 5.60 s -> L_s = 33.6 in flight
  L_total = 41.7 ; with 64 workers utilisation = 41.7/64 = 0.652 -> acceptable on average
INTERPRETATION: 64 workers look adequate against mean service times, at 65% utilisation.
```
The same computation with p99 service times tells a different story, and the difference is the decision:

```text
P99 CASE:
  W_r(p99) = 0.40 s -> L_r = 54.0*0.40 = 21.6
  W_s(p99) = 9.00 s -> L_s = 6.0*9.00  = 54.0
  L(p99)   = 75.6 in flight > 64 workers -> utilisation 1.181, queue does not recover
  workers at target_util 0.7 = 75.6/0.7 = 108
INTERPRETATION: sizing on the mean under-provisions by 108 - 64 = 44 workers, and the shortfall
                appears precisely during the peak the budget was written for.
```
Both routes checked: 54.0 × 0.40 = 21.6 and 6.0 × 9.00 = 54.0; sum 75.6; 75.6/64 = 1.18125; 75.6/0.7 = 108.0. The mean case is 54.0 × 0.15 = 8.1 plus 6.0 × 5.60 = 33.6, sum 41.7, and 41.7/64 = 0.6516.

Where the formula breaks: Little's law is an identity about averages and says nothing about the tail, which is exactly why the p99 row exists. It also assumes the peak hour is stationary; if traffic arrives in a five-minute burst at the top of the hour, `lambda` is understated by up to 12× and the model is optimistically wrong. The discriminating check is a per-minute arrival histogram, not a per-hour one.

A second boundary: the scan share `W_s = 5.60 s` is itself a queueing outcome, not an input. Substituting a queue-inflated service time into the concurrency formula double-counts the wait; the correct input is the service time at low utilisation, measured separately.

### 6b Token bucket limits: the burst is the design variable
```text
FORMULA [API-02]: admitted(t) = min(capacity, tokens + refill*t) ; rejected = offered - admitted
                  refill_to_full = capacity / refill_rate
WHERE: capacity [requests] the burst allowance; refill_rate [requests/s]; t [s] elapsed;
       tokens [requests] current balance; all other terms are counts in the same window.
CONSTRAINT: one bucket per caller class; a rejected request consumes no token; refill is continuous
             rather than per-second, so a burst just under the rate is not artificially split.
NUMERICAL EXAMPLE:
  capacity = 60 requests ; refill_rate = 10 req/s ; offered burst = 100 requests at t = 0
  admitted = 60 ; rejected = 40 ; tokens after burst = 0
  sustained offer of 15 req/s: served = 10 req/s, rejected = 5 req/s, bucket never refills
  refill_to_full = 60/10 = 6.0 s of quiet traffic to restore the burst allowance
INTERPRETATION: steady-state goodput per caller is the refill rate, 10 req/s; the burst allowance
                only decides how much of a spike is absorbed, not how much throughput a caller gets.
```
Checked independently: at 15 req/s offered against 10 req/s served, the deficit is 5 req/s, and since the bucket starts empty it cannot absorb anything, so the rejection rate is exactly the deficit. With a full bucket it would absorb 60 requests before settling to the same 10 req/s.

The design consequence is that capacity and refill rate answer different questions and must be chosen separately. `refill_rate` is a fairness decision: at 10 req/s, a 60 req/s peak can be served by six such callers, so a caller class larger than six members necessarily shares. `capacity` is a latency decision: a dashboard that fires a burst of 20 tiles on load needs `capacity >= 20` or its own load generates 429s, which the dashboard then retries, which is the amplification problem in 6h.

Boundary where the model misleads: if all six callers burst simultaneously at the top of a minute, the aggregate offer is 360 req/s for one second against 60 req/s of refill, so 300 requests are rejected in that second even though the hourly average is comfortable. The discriminating check is rejection counts per second at minute boundaries, not daily totals. The remedy is either a capacity sized for the correlated burst or jittered client scheduling, and the second is free.

Second boundary: a token bucket protects the API but not the warehouse. A caller within its 10 req/s may still issue ten queries that each scan 86.4 GB, producing 864 GB of scan in a second, which is 7.2× the scan throughput available. Rate limits on requests and limits on bytes are different controls, and the contract needs both.

### 6c Scan cost is a function of range, and the range bound is a product decision
```text
FORMULA [API-03]: bytes = rows_per_day * days * bytes_per_row ;  t_scan = bytes / r_scan
                  cost = bytes / 1e12 * price_per_TB
WHERE: rows_per_day [rows/day]; days [days]; bytes_per_row [B]; r_scan [B/s];
       price_per_TB [USD/TB]; t_scan [s]; cost [USD].
CONSTRAINT: a single-pass scan of exactly the requested range with no predicate pushdown saving;
            any additional pass multiplies both time and cost by the number of passes.
NUMERICAL EXAMPLE:
  rows_per_day = 40,000,000 ; bytes_per_row = 24 ; r_scan = 1.2e9 B/s ; price = 5 USD/TB
  1 day : 40,000,000*24 = 0.96e9 B = 0.96 GB -> 0.800 s -> 0.96/1000*5   = 0.0048 USD
  7 days: 280,000,000*24 = 6.72e9 B = 6.72 GB -> 5.600 s -> 6.72/1000*5  = 0.0336 USD
  30 days: 1,200,000,000*24 = 28.8 GB -> 24.00 s -> 0.1440 USD
  90 days: 3,600,000,000*24 = 86.4 GB -> 72.00 s -> 0.4320 USD
  ratio 90d:1d = 90 in bytes, 90 in time, and 90 in cost
INTERPRETATION: scan time is exactly linear in the requested range, so a range bound is a latency
                bound. The 90-day shape costs 90 times the daily shape for the same code path.
```
Section 3a and this block use decimal GB and TB consistently: 0.96 GB / 1,000 GB/TB × 5 USD/TB = 0.0048 USD. The stated price is a constructed planning input, not a provider quote. A contract review should recompute each displayed unit conversion before it treats a range limit as a cost limit.

Now the decisive derivation, which uses the same numbers to bound the synchronous path:

```text
SYNCHRONOUS CAPACITY:
  budget = 0.800 s total ; scan share = 0.340 s
  bytes_affordable = 0.340 * 1.2e9 = 4.08e8 B = 408 MB
  rows_affordable  = 4.08e8 / 24 = 17,000,000 rows
  daily feed       = 40,000,000 rows -> 17,000,000/40,000,000 = 0.425 of one day
  90-day shape     = 3.6e9 rows -> 3.6e9/1.7e7 = 212 times the synchronous capacity
INTERPRETATION: the inline path can afford at most 17 million rows, which is 0.425 of a single
                day of the feed. A 90-day request needs 212 times that, so it cannot be synchronous
                at any queue depth; it must be an async job or a precomputed rollup.
```
Two independent checks on the same conclusion. First, 0.34 × 1.2 = 0.408 GB, and 0.408/0.024 GB per row = 17 million. Second, the ratio 72 s/0.34 s = 211.8, matching 3.6e9/1.7e7 = 211.8. Both routes agree that no amount of worker parallelism fixes a single-request latency violation, because parallelism shortens wall time only by splitting the same bytes across shards — and at 6 shards the 72 s becomes 12 s, still 15× over budget.

Boundary where the linear model fails: partition pruning. A query filtered to one `region_id` that occupies 2% of rows scans 2% of bytes if the table is clustered on that key, so the 90-day shape could cost 1.7 GB instead of 86.4 GB. Then the range bound is the wrong control and the cluster key is the right one. The discriminating check is bytes scanned versus rows returned for filtered queries; a ratio near 1 means no pruning is happening.

Second boundary: compression. If the projection compresses 4:1, the byte cost falls by four while the row cost is unchanged, so a byte-based bound and a row-based bound disagree. The contract must pick one and say which, or two teams will quote two capacities.

### 6d Cache economics with an explicit ceiling
```text
FORMULA [API-04]: hit_ceiling = 1 - distinct_keys / requests ;  saved = requests * hit * bytes
                  saved_cost = saved / 1e9 * price_per_GB
WHERE: requests [requests/day]; distinct_keys [keys/day]; hit dimensionless and <= hit_ceiling;
       bytes [B/response]; saved_cost [USD/day].
CONSTRAINT: uniform demand across keys; a single hot key raising the ceiling is not modelled, and
            keys must include the snapshot bucket or cached bytes outlive their label.
NUMERICAL EXAMPLE:
  requests = 1,200,000/day ; rollup share 0.90 -> 1,080,000 rollup requests
  payload = 40 KB -> 1,080,000*40,000 B = 43.2e9 B = 43.20 GB/day of egress
  at 0.09 USD/GB = 3.89 USD/day for rollups alone
  hit = 0.72 -> cached 777,600 ; served 302,400
  saved = 777,600*40,000 = 31.10e9 B = 31.10 GB -> 2.80 USD/day -> 84.0 USD per 30 days
  ceiling with 4,800 distinct keys = 1 - 4,800/1,200,000 = 0.9960
INTERPRETATION: the observed 0.72 hit rate is far below the 0.996 ceiling, so the cache is losing
                to key granularity, not to demand diversity.
```
The gap between 0.72 and 0.996 is the finding. With only 4,800 distinct shapes a day against 1.2 million requests, uniform demand would hit 99.6% of the time. A 72% hit rate at that ceiling means the key includes something that varies per request — a timestamp rounded to the second, a caller identifier, a non-canonicalized filter ordering — so nearly identical questions miss each other.

Verified independently: 777,600 × 40,000 = 31,104,000,000 bytes; /1e9 = 31.104 GB; × 0.09 = 2.799 USD/day; × 30 = 83.98 USD. Agrees.

Boundary where the formula misleads: a high hit rate can be worse than a low one if the key omits the filter. Two callers with different `region_id` filters hashing to the same key would hit 100% and return the wrong region's numbers. The ceiling tells you what is achievable; only a collision test tells you whether the achieved rate is legitimate.

Second boundary: the model treats all requests as equal, but a missed scan request costs 480 KB against a rollup's 40 KB, a 12× difference. Weighting by bytes rather than requests changes the priority of what to cache, and in this workload the 10% scan traffic accounts for 58.98 GB/day against the rollups' 43.20 GB, so caching scans is worth more per miss even though there are twelve times fewer of them.

### 6e Fan-out: how six 99% shards become a 5.85% failure rate
```text
FORMULA [API-05]: P_success = prod_i (1 - f_i) ;  f_i_needed = 1 - (1 - F_total)^(1/n)
WHERE: f_i the failure probability of shard i (dimensionless); n the shard count;
       F_total the tolerated end-to-end failure probability.
CONSTRAINT: shard failures are independent; the request fails if any shard fails; retries are
            counted separately, and a retry converts a failure into added latency, not a success.
NUMERICAL EXAMPLE:
  n = 6 shards ; f_i = 0.01 each
  P_success = 0.99^6 = 0.941480 -> 94.15% ; F_total = 0.058520 -> 5.85%
  to hold F_total <= 0.01: f_i <= 1 - 0.99^(1/6) = 1 - 0.998333 = 0.001667
  check: 0.998333^6 = 0.990042 -> F_total = 0.009958 <= 0.01
INTERPRETATION: shard reliability of 99% is not good enough by a factor of six when the request
                touches all shards; the requirement per shard is 99.83%, not 99%.
```
The arithmetic here is the reason fan-out architectures fail in ways their component dashboards never show. Each shard reports 99% availability and the service reports 94.15%, and nobody believes the service number because every component looks fine. The identity `P_success = 0.99^n` also gives the quick rule: with a 99% shard, end-to-end success falls below 95% at n = 6, below 90% at n = 11 (0.99^11 = 0.8953), and below 80% at n = 23 (0.99^23 = 0.7936).

Where the formula is wrong: correlated failure. If all six shards live in one storage cluster, one outage takes all six, and the independence assumption collapses to `P_success = 0.99`, not 0.9415. The discriminating check is whether shard failures cluster in time; a histogram of failures per minute with a spike at a single minute indicates correlation, and then the arithmetic above is optimistic by a wide margin.

Second boundary: retries change the shape. With two retries per shard at 8% per-attempt failure, per-shard residual failure is 0.08^3 = 5.12e-4, and end-to-end becomes 1 − (1 − 5.12e-4)^6 ≈ 3.07e-3, a 19× improvement — at the cost of a 1.0864× attempt amplitude that must be added to capacity. That trade is quantified in 6h.

### 6f Top-k truncation: a per-shard limit is not a global limit
```text
FORMULA [API-06]: for a global top-k over n shards each returning k, the probability that a shard
                  holds >= k+1 of the true top-M is Binomial(M, 1/n) tail;
                  P(miss any true member) <= n * P(shard holds >= k+1)
WHERE: M the global top-M size; n shards; k the per-shard limit; probabilities dimensionless.
CONSTRAINT: true top-M items are assigned to shards uniformly at random; ties broken arbitrarily;
            the bound is a union bound and therefore conservative.
NUMERICAL EXAMPLE:
  M = 10 ; n = 6 ; P(one item in a given shard) = 1/6
  k = 3 -> P(shard holds >=4 of 10) = sum_{j=4..10} C(10,j)(1/6)^j(5/6)^(10-j) = 0.069728
           P(miss any) <= 6*0.069728 = 0.418367 -> 41.8%
  k = 4 -> P(shard >=5) = 0.015462 -> bound 0.092772 -> 9.28%
  k = 5 -> P(shard >=6) = 0.002438 -> bound 0.014629 -> 1.46%
  k = 6 -> bound 0.0016 -> 0.16%
INTERPRETATION: a per-shard top-3 for a global top-10 loses a true member 41.8% of the time, so
                the correct per-shard limit is k >= 6 unless the answer is allowed to be wrong.
```
Independent check at k = 4: the expected number of shards holding five or more of the ten is 6 × 0.015462 = 0.0928, which matches the union bound because the events are nearly disjoint at this probability. Two routes agree, and the agreement is also the warning: the union bound stops being an upper bound worth quoting once the events overlap materially.

The general rule is the useful artefact: for a global top-M across n shards, use `k = ceil(M/n) + slack` where slack grows with M. At M = 10, n = 6, the even split is 1.67 and the needed k is 6, which is 3.6× the even split. Any implementation that divides M by n and rounds up is wrong by a factor that is invisible on small tests where M is smaller than n.

Boundary where it fails: skewed key distribution. If one shard owns 60% of the traffic because of a hash-key mistake, its top-k covers a much larger share and the uniform assignment assumption is badly wrong. The discriminating check is the per-shard share of total rows for the grouping key; a share far above 1/n means the hash is skewed and the top-k bound must be recomputed with the observed assignment.

Second boundary: approximate top-k. If shards return approximate heavy hitters rather than exact top-k, errors compound across shards in a way this formula does not model, and the result can be wrong even at k = M. Approximate and exact results must be distinguishable in the response envelope.

### 6g Pagination: offset cost grows quadratically, keyset does not
```text
FORMULA [API-07]: rows_scanned_offset(p) = p * s + s ;  rows_scanned_keyset = s
                  efficiency(p) = s / (p*s + s) = 1/(p + 1)
WHERE: p [pages] the number of pages already delivered; s [rows] page size;
       efficiency dimensionless.
CONSTRAINT: offset pagination requires the storage engine to count and discard p*s rows; keyset
            pagination seeks directly on an indexed (timestamp, id) tuple.
NUMERICAL EXAMPLE:
  s = 500 rows/page ; p = 200 pages (100,000 rows already delivered)
  offset: rows_scanned = 200*500 + 500 = 100,500 ; efficiency = 500/100,500 = 0.004975 -> 0.50%
  keyset: rows_scanned = 500 ; efficiency = 1.0
  waste factor = 100,500/500 = 201
INTERPRETATION: the two-hundredth offset page does 201 times the work of the first while returning
                the same 500 rows, so total work across 200 pages is quadratic in page count.
```
Total work for a full traversal makes the point sharper. Offset pagination to deliver `P` pages scans `s*(1 + 2 + ... + P) = s*P*(P+1)/2` rows; with s = 500 and P = 200 that is 500 × 20,100 = 10,050,000 rows to deliver 100,000 rows — a 99.5× overhead, and at 1.2 GB/s this is 10,050,000 × 24 B = 241 MB, or 0.2 s per traversal, on a query whose useful work was 2.4 MB. Keyset scans exactly 100,000 rows, 2.4 MB.

The correctness difference matters more than the cost. Offset pagination is unstable under concurrent inserts: a row inserted before the current offset shifts every later page by one, so the client sees a duplicate; a deletion causes a skip. At 463 rows/s arriving and a page taking 0.2 s, roughly 93 rows arrive per page, so a 200-page traversal is essentially guaranteed to duplicate and skip. Keyset on `(event_time, event_id)` is immune, provided the tuple is unique.

Where the tuple is not unique: 0.3% of rows share a timestamp with a neighbour, so on a 500-row page boundary the composite key with ties can overshoot by 0.003 × 500 = 1.5 rows if the tie-break is absent. With `event_id` in the key, the ordering is total and there is no overshoot. The discriminator is a page-boundary test that inserts a row with a duplicate timestamp and asserts the next page neither repeats nor skips.

Boundary where keyset fails: any query whose ordering is by an aggregate, such as `ORDER BY count DESC`. There is no index seek on a computed aggregate, so keyset pagination is unavailable and the shape must be bounded to a single page or deferred to an async job that materializes the full ordering.

### 6h Retry amplification and the latency it adds
```text
FORMULA [API-08]: residual_failure = f^a ;  attempt_amplitude = sum_{i=0..a-1} f^i
WHERE: f the per-attempt failure probability (dimensionless); a [attempts] the retry budget;
       amplitude is expected attempts per logical request, dimensionless and >= 1.
CONSTRAINT: retries are independent, the client retries only on retryable errors, and the retry
            delay is not counted in the latency figure below.
NUMERICAL EXAMPLE:
  f = 0.08 ; a = 3
  residual_failure = 0.08^3 = 0.000512 -> 0.0512% (512 per million)
  attempt_amplitude = 1 + 0.08 + 0.0064 = 1.0864
  extra capacity required = (1.0864 - 1) * 100 = 8.64% of nominal
INTERPRETATION: three attempts turn an 8% failure rate into 512 failures per million at the cost of
                8.64% more load, which is worth it; a fourth attempt adds only 0.04% for another
                0.05% of load and is not worth the tail latency.
```
The second-order effect is the one that causes incidents. Retries amplify load precisely when the service is degraded, because `f` rises exactly when capacity is short. If degradation takes `f` from 0.08 to 0.40, three attempts give amplitude 1 + 0.40 + 0.16 = 1.56, so a struggling service receives 56% more traffic at the worst possible moment. That is why the contract specifies a `retry_after` field and why clients must honour it: uncapped retries are positive feedback on an overloaded system.

Verified independently: 1/(1 − 0.08) = 1.0870 is the infinite-retry amplitude, and the three-attempt figure 1.0864 is 0.9994 of it, confirming the series converges quickly at low `f`. At `f` = 0.40 the infinite sum is 1.667, and the three-attempt figure 1.56 is 94% of it, which is why deeper retry budgets matter more when `f` is large.

Boundary where the model is wrong: retries without jitter synchronize. If every client retries after the same fixed delay, the retry wave arrives as a spike rather than a spread, and the effective concurrency doubles at each wave. The discriminating check is the retry arrival timestamp histogram: a multimodal shape with peaks at the delay interval is the synchronization signature, and exponential jitter with a floor is the correction.

Second boundary: retrying non-idempotent requests. Every operation here is a read, so retries are safe; a write path would need idempotency keys, and the absence of one turns a retry into a duplicate side effect.

### 6i The latency budget is a partition, not a target
```text
FORMULA [API-09]: B = sum_i b_i ;  slack = B - sum_i b_i ;  share_i = b_i / B
WHERE: B [ms] the end-to-end synchronous budget; b_i [ms] the allowance for component i;
       slack [ms] unallocated headroom; share_i dimensionless.
CONSTRAINT: components are sequential, not overlapped; each b_i is an allowance rather than a
            measurement, and the sum must leave slack for variance.
NUMERICAL EXAMPLE:
  B = 800 ms
  TLS + auth 18 ; validate 14 ; plan 26 ; warehouse queue 210 ; scan + aggregate 340 ;
  serialise 480 KB 52 ; egress 95
  sum = 755 ms ; slack = 45 ms -> 5.63% headroom
  scan share = 340/800 = 0.425
INTERPRETATION: the allowance is fully consumed by seven components with 45 ms of slack, so any
                component running 6% over its allowance breaches the budget.
```
Check the parts: 18 + 14 = 32; +26 = 58; +210 = 268; +340 = 608; +52 = 660; +95 = 755. Slack 45, and 45/800 = 5.625%.

The most useful thing the decomposition reveals is that the warehouse queue at 210 ms is 26% of the budget while contributing no useful work; it is other people's load. That allocation is why the async path exists: a query whose scan needs 5.6 s cannot be given a larger slice, because there is no larger slice available. The decomposition converts "make it faster" into "either cut the scan or leave the synchronous path".

Sensitivity: if the scan component doubles to 680 ms, the total is 1,095 ms and the budget is breached by 295 ms — a 37% overrun from a single component doubling, which is the reason the contract refuses shapes whose estimated scan exceeds 340 ms rather than hoping they finish.

Boundary where the partition misleads: it assumes sequential components. If auth and planning can overlap with the warehouse's own queue admission, the true critical path is shorter and the allowances double-count. The discriminating check is a trace with per-component start and end timestamps, where overlap is visible as nested spans. Budgets built from a mental model of a pipeline rather than a trace routinely over-allocate by the amount of overlap nobody noticed.

Second boundary: p99 budgets do not add. If each component's p99 is at its allowance, the sum is at the budget, but the probability that all seven are simultaneously at p99 is negligible, so a 800 ms p99 target is achievable from components whose individual p99s sum to more than 800 ms. Conversely, one component with a heavy tail can dominate the p99 alone. The budget should therefore be set from a per-component distribution plus a tail check, not from optimistic addition.

### 6j Percentiles do not combine across shards
```text
FORMULA [API-10]: p99_merged = value at rank ceil(0.99 * N_total) in the merged distribution
                  NOT sum_i w_i * p99_i
WHERE: N_total [requests] the union across shards; w_i the shard's share of requests;
       p99_merged [ms] the true merged percentile.
CONSTRAINT: the merged percentile requires the full distribution or an approximation of it;
            weighted averaging of per-shard percentiles is valid for means only, never for quantiles.
NUMERICAL EXAMPLE (constructed shard distributions, six shards):
  shards A-E: 1,000 requests each at 120, 300, 90, 110, 950 ms ; shard F: 100 requests at 5,000 ms
  N_total = 5,100 ; rank for p99 = ceil(0.99*5,100) = 5,049 -> merged p99 = 5,000 ms
  naive average of shard p99s = (120+300+90+110+950+5,000)/6 = 1,095 ms
  naive weighted by share (F is 1.96% of traffic): 0.1962*5,000 + 0.8038*314 = 1,233 ms
  merged p50 = 120 ms
INTERPRETATION: the naive average understates the merged p99 by a factor of 4.6, and weighting it
                correctly still understates by 4.1, because 1% of 5,100 requests is 51 requests
                and shard F alone contributes 100 of them.
```
The structural reason is worth stating plainly: a percentile is a position in an ordering, not an average of positions. Shard F holds 100 requests, which is 1.96% of the traffic — more than the 1% needed to define the merged p99 — so its slow requests *are* the merged p99. Any aggregation that averages positions loses them.

The decision consequence: per-shard dashboards will each report an acceptable p99 (120, 300, 90, 110, 950 ms for the five, 5,000 for F) while the service p99 is 5,000 ms. A service whose largest shard's p99 dominates is a service whose latency is set by its worst shard, and the only honest fixes are to rebalance shards or to bound the slow shape. The discriminator is a comparison of per-shard request volume: a shard carrying more than 1% of traffic can define the p99 alone.

Boundary: when shard volumes are equal and distributions similar, the naive average is close enough for monitoring, and the error grows as volumes and distributions diverge. The safe practice is to compute the merged percentile from a histogram with enough buckets, which costs almost nothing and is exact to the bucket width.

Second boundary: streaming approximations (t-digest, DDSketch) give relative error guarantees at the tails, which is what a p99 needs, but they are not mergeable across implementations with different parameters. Two services running different sketch parameters produce percentiles that cannot be combined, so the contract must fix the sketch parameters if merged percentiles are a requirement.

### 6k Consistency: snapshots make results reproducible, and the cost is staleness
```text
FORMULA [API-11]: rows_arriving_during_query = (rows_per_day/86400) * t_query
                  staleness(watermark) = now - max_watermark_observed
WHERE: t_query [s]; rows_per_day [rows/day]; staleness [s]; watermark in event or processing time
       as declared.
CONSTRAINT: the table is append-only during the query; a snapshot read must pin the readable
            partition set, and per-partition watermarks may differ within one response.
NUMERICAL EXAMPLE:
  rows_per_day = 40,000,000 -> 40,000,000/86,400 = 462.96 rows/s
  1-day shape (t_query = 0.800 s)   -> 370 rows arrive during the query
  7-day shape (t_query = 5.600 s)   -> 2,593 rows arrive during the query
  90-day shape (t_query = 72.00 s)  -> 33,333 rows arrive during the query
INTERPRETATION: a query read without a pinned snapshot over 90 days is answered against a table
                that changed by 33,333 rows while it ran, so two identical requests 72 seconds
                apart cannot return the same number.
```
Checked independently: 40,000,000/(24 × 3600) = 40,000,000/86,400 = 462.96; × 72 = 33,333.3. And × 0.8 = 370.4.

The design conclusion is that the response must name a snapshot, and that the snapshot must be a set of readable partitions plus a watermark, because no cheaper mechanism binds an answer to a table state. The caller then gets a reproducible result and an explicit staleness figure, and can decide whether 14 minutes behind is acceptable for a dashboard tile.

Boundary where a snapshot is not enough: late arrivals change the contents of a partition that has already been read. A snapshot pins which partitions were visible, not which rows inside them existed, so a batch that lands late changes the answer to a query over the same snapshot. The discriminating check is a repeated query at two times against the same snapshot ID: identical results mean the partition is settled, differing results mean late data is still landing, and the difference is the arrival curve.

Second boundary: the watermark is per partition, so a response spanning a hundred partitions may mix a fresh partition with a stale one. Reporting the maximum watermark hides that; reporting the minimum is more honest and more pessimistic. The contract should report both, or report a count of partitions below a declared lag threshold.

### 6l Approximate cardinality trades a bounded error for an unbounded scan
```text
FORMULA [API-12]: relative_error ~= 1.04 / sqrt(m) ;  exact_distinct needs O(n) memory
WHERE: m the number of sketch registers (dimensionless, a power of two); n [distinct values];
       relative_error dimensionless, 1 standard error.
CONSTRAINT: hash values uniform, distinct count much larger than m, sketch parameters fixed by the
            contract so results from different services are comparable.
NUMERICAL EXAMPLE:
  m = 4,096 registers -> 1.04/sqrt(4,096) = 1.04/64 = 0.01625 -> 1.63% relative error
  a true distinct count of 2,000,000 would be reported within about +-32,500 at one standard error
  exact distinct over 7 days requires a set holding up to 2,000,000 keys -> ~32 MB at 16 B/key
INTERPRETATION: the sketch answers in constant memory and 1.63% error, while the exact answer needs
                memory proportional to cardinality and a merge across shards.
```
The trade is only acceptable if the consumer's decision tolerates 1.63%. A dashboard showing "2.0M uniques" tolerates it; a billing reconciliation does not, which is why the response labels the field as approximate rather than the documentation mentioning it once.

Checked independently: 2,000,000 × 0.01625 = 32,500, and 1.04/64 = 0.01625 exactly. A 16,384-register sketch gives 1.04/128 = 0.008125, halving the error at four times the memory; that is the knob, and it is a contract field rather than a tuning parameter precisely so two teams cannot choose differently.

Boundary where the sketch is wrong: small cardinalities. Below roughly 2.5m distinct values the estimator's bias correction dominates and the error is larger than the formula suggests; the standard fix is exact counting below a threshold and the sketch above it. The discriminating check is to compare sketch and exact on a window with known small cardinality.

Second boundary: mergeability. Sketches merge by taking the maximum register value per position, which is valid only when all sketches use the same `m` and hash. Different `m` values make the merge silently wrong, and no error is raised — a failure shape identical to the cache-key collision in A-04.

### 6m Worked end-to-end case: one dashboard load
| Step | Component | Time (ms) | Bytes | Cache | Notes |
|---|---|---|---|---|---|
| 1 | TLS handshake and auth | 18 | — | no | per connection, reused where possible |
| 2 | token bucket check | 1 | — | no | before parsing, by design |
| 3 | parse and canonicalize | 14 | — | no | unknown fields rejected here |
| 4 | plan and estimate | 26 | — | no | estimate compared with the shape bound |
| 5 | cache lookup, 12 tiles | 3 | — | yes for 8 | 8 hits, 4 misses at 0.67 on this burst |
| 6 | warehouse queue | 210 | — | — | other tenants' load |
| 7 | scan and aggregate, 1 day | 340 | 960 MB | 4 tiles | 4 × 0.96 GB = 3.84 GB scanned |
| 8 | serialize 4 × 40 KB | 52 | 160 KB | — | page sizes under the 4 MB cap |
| 9 | egress | 95 | 160 KB | — | 0.0000144 USD at 0.09 USD/GB |
| — | total | 759 | — | — | 41 ms of slack against the 800 ms budget |

Verify: 18 + 1 + 14 + 26 + 3 + 210 + 340 + 52 + 95 = 759. Slack = 41 ms. Bytes scanned 4 × 0.96 GB = 3.84 GB at 1.2 GB/s = 3.2 s, which contradicts the 340 ms scan allowance — and the contradiction is the point of the row.

That inconsistency is the most instructive line in the case. Four uncached one-day tiles cannot be served in 340 ms, because 3.84 GB at 1.2 GB/s is 3.2 s. A dashboard that fires twelve tiles on load, four of which miss the cache, breaches the budget by 4× even though every individual component is "within its allowance". The remedy is not a faster scan; it is to make the dashboard's tile set a single multi-metric query, so one pass serves four numbers, or to warm the cache before the load. Per-request budgets do not compose across a burst, which is the theme of 6a restated at the client.

### 6n Per-caller byte budgets and why a single global cap cannot work
```text
FORMULA [API-13]: bytes_class = sum_shape (requests_class,shape * payload_shape) ;
                  kb_per_request_class = bytes_class / requests_class
WHERE: requests per class and shape [requests/day]; payload [B/response]; bytes [B/day];
       kb_per_request [B/request].
CONSTRAINT: payload is per response, not per row; a scan response of 480 KB carries at most 500 rows
            so the payload is a page bound, not a data bound. Classes differ in their scan share.
NUMERICAL EXAMPLE (class mix: interactive 60%, batch 30%, exploratory 10% of 1,200,000 requests;
scan share within class: interactive 5%, batch 60%, exploratory 35%; rollup payload 40 KB,
scan payload 480 KB):
  interactive: 720,000 requests -> 36,000 scan, 684,000 rollup
               -> 36,000*480,000 + 684,000*40,000 = 17.28 GB + 27.36 GB = 44.64 GB/day
               -> 44.64e9/720,000 = 62,000 B/request = 62.0 KB/request
  batch:        360,000 requests -> 216,000 scan, 144,000 rollup
               -> 216,000*480,000 + 144,000*40,000 = 103.68 GB + 5.76 GB = 109.44 GB/day
               -> 109.44e9/360,000 = 304,000 B/request = 304.0 KB/request
  exploratory:  120,000 requests -> 42,000 scan, 78,000 rollup
               -> 42,000*480,000 + 78,000*40,000 = 20.16 GB + 3.12 GB = 23.28 GB/day
               -> 23.28e9/120,000 = 194,000 B/request = 194.0 KB/request
  fleet total = 44.64 + 109.44 + 23.28 = 177.36 GB/day
              = 177.36*0.09 = 15.9624 USD/day at 0.09 USD/GB
INTERPRETATION: batch is 30% of requests but 61.7% of egress (109.44/177.36), so one global
                request cap blocks interactive callers while permitting disproportionate batch use.
```
Independent check: class request counts sum to 720,000 + 360,000 + 120,000 = 1,200,000. Class scan counts sum to 294,000 and rollup counts to 906,000, matching a fleet scan share of 24.5%. Separately, 44.64 + 109.44 + 23.28 = 177.36 GB, and 177.36 × 0.09 = 15.9624 USD/day. Both routes use decimal bytes, so `KB` means 1,000 bytes and `GB` means 1,000,000,000 bytes in this section.

The design consequence is a 62.0-to-304.0 KB/request spread, a factor of 4.90. A global average allowance of 147.8 KB/request would reject a legitimate 480 KB batch page yet let an interactive credential issue more costly requests than its intended share. The contract therefore carries both a request bucket and a byte bucket per caller class; admission requires a token in both buckets.

Boundary where per-class budgets fail: a single class can contain heterogeneous members. If one `exploratory` credential runs scheduled ninety-day scans while the rest browse dashboard tiles, the class average hides the offender. The discriminating check is the per-credential byte distribution inside the class; a heavy tail means the budget must also be imposed per credential.

Second boundary: payload is a page bound, not a session bound. A caller can request many pages while keeping each response below 480 KB. The only control that limits cumulative egress is a rolling per-day byte budget checked before execution, with a returned `remaining_bytes` value that lets a client stop deliberately rather than discover exhaustion after a scan.

### 6o Sensitivity and the numbers that would change the decision
| Input changed | From | To | Effect on the decision | Effect on the arithmetic |
|---|---|---|---|---|
| Peak share of daily volume | 0.18 | 0.25 | 64 workers become clearly insufficient | lambda 60 → 83.3 req/s; L(p99) 75.6 → 105.0 |
| Rollup service p99 | 0.40 s | 0.25 s | 108 workers fall to 86 | L(p99) 75.6 → 61.2, below 64 at utilisation 0.96 |
| Scan throughput | 1.2 GB/s | 2.4 GB/s | 7-day shape becomes inline-capable | 5.60 s → 2.80 s, still above 0.34 s allowance |
| Cache hit rate | 0.72 | 0.95 | egress saving rises 32% | saved bytes 31.10 → 41.04 GB/day |
| Shard count | 6 | 12 | per-shard reliability requirement tightens | f_i needed 0.001667 → 0.000836 |
| Per-shard top-k | 3 | 6 | truncation risk falls 41.8% → 0.16% | bound 0.4184 → 0.0016 |
| Latency budget | 800 ms | 1,200 ms | scan allowance could rise to 740 ms | affordable rows 17M → 37M, still 0.9 days |

Read the table as a statement about robustness: the refusal for the 90-day shape survives every single change above except a 30× improvement in scan throughput, which is not a plausible planning assumption for this table. That is what makes it a sound product bound rather than a temporary restriction.

**Takeaway:** the decision is robust to every input except the one that would require different hardware, and the arithmetic shows which input that is.

## 7. The Finding
The supported conclusion is that this API should expose three fixed shapes with four outcomes — inline, cached, async, refused — and that the split between inline and async is forced by arithmetic rather than preference. Three computations carry the conclusion. The synchronous path can afford 17 million rows, which is 0.425 of one day of the feed, so any multi-day raw scan must be asynchronous. Sizing workers from mean service times gives 41.7 concurrent requests and 65% utilisation, while p99 sizing gives 75.6 and requires 108 workers, so the mean-based figure under-provisions by 44. And six shards at 99% availability yield 94.15% end-to-end success, so the per-shard requirement is 99.83%, not 99%.

The design is eligible for a bounded implementation only after the snapshot and watermark rules are agreed with the platform team and the caller classes have declared staleness tolerances. Nothing here demonstrates that any endpoint was built or called. The strongest available claim is that the contract is specified precisely enough that two independent implementations would agree on which queries are refused.

## 8. System/Solution Design
```text
[client] -> [TLS/auth 18ms] -> [token bucket 1ms] -> [parse+canonicalize 14ms]
                                                           |
                                              [plan + estimate 26ms]
                                                           |
                          +--------------------------------+-----------------------------+
                          v                                v                             v
                 estimate > shape bound          estimate <= bound              shape has no async path
                          |                                |                             |
                          v                                v                             v
                 [429/422 typed refusal]          [cache lookup 3ms]             [refusal with remedy]
                                                           |
                                      +--------------------+--------------------+
                                      v                                         v
                                  cache hit                                cache miss
                                      |                                         |
                                      v                                         v
                              [envelope, no scan]                    [warehouse queue 210ms]
                                                                            |
                                                                            v
                                                                [scan+aggregate 340ms]
                                                                            |
                                                                            v
                                                                [snapshot + watermark stamp]
                                                                            |
                                                                            v
                                                              [serialise 52ms -> egress 95ms]
                                                                            |
                                                                            v
                                                              [cursor sign if more rows exist]

async path: [submit] -> [job record keyed by query hash] -> [queue] -> [execute] -> [result pinned
            to snapshot] -> [poll /jobs/{id}] -> [expiry after retention window]
```

Component ownership is the design's safety property. The token bucket owns fairness and sits before parsing, so a caller cannot consume parsing capacity without holding a token. The planner owns the inline/async decision and is the only component allowed to compare an estimate with a bound. The cache owns nothing but bytes and reads its snapshot bucket from the key, so it cannot return a result without a label. The executor owns the scan and stamps the snapshot before serialization. The cursor codec owns the pagination boundary and signs it. The envelope assembler owns the claim that a response is complete or truncated, and is the only place `truncated: true` can be set.

The central invariant: every response carries a snapshot identifier and a byte count that were produced by the same execution. A cached response carries the identifier from the execution that populated it. There is no path that returns data without both fields, which means a caller can always ask "which table state, at what cost".

Error taxonomy, which is the part callers actually experience:

| Code | HTTP | Meaning | Retryable | Documented remedy |
|---|---|---|---|---|
| `RANGE_TOO_LARGE` | 422 | requested interval exceeds the shape bound | no | narrow the interval or submit an async job |
| `FILTER_REQUIRED` | 422 | shape needs a selective predicate | no | add a partition or region filter |
| `RATE_LIMITED` | 429 | caller-class bucket empty | yes, after `retry_after` | reduce burst or request a larger class |
| `BUDGET_EXCEEDED` | 422 | estimate exceeds the inline scan allowance | no | use the async endpoint |
| `UNKNOWN_FIELD` | 400 | unsupported parameter present | no | remove the field; silently ignoring it would change the answer |
| `STALE_SNAPSHOT` | 409 | requested snapshot no longer readable | yes, once | reissue without a pinned snapshot |
| `JOB_NOT_FOUND` | 404 | job ID unknown or expired | no | resubmit; expiry is declared |
| `PARTIAL_READ` | 503 | a required partition was unreadable | yes, after backoff | none; the service reports excluded partitions |
| `APPROXIMATE_ONLY` | 200 | field is a sketch, not exact | n/a | request the exact shape if the use requires it |

Two design choices deserve their reasoning. Errors carry a remedy rather than only a description, because a 422 that says "range too large" leaves the caller guessing at the bound, and guessing produces retry storms. And `APPROXIMATE_ONLY` is not an error: it returns data with a flag, because turning a usable approximate answer into a failure would push callers to the console, where they get an exact answer and an unbounded scan.

```text
[job submit] --> [hash endpoint+shape+filters+snapshot_bucket]
                        |
                        +--> [existing job for hash within window?] --yes--> return existing job_id
                        |                                        |
                        +--no--> [create job record, enqueue] <---+
                                        |
                                        v
                        [execute -> pin snapshot -> persist result + expiry]
                                        |
                                        v
                        [poll: queued | running | complete(snapshot) | failed(code)]
```

### 8a Interface contract: canonical requests and stable response semantics
The public request is a declarative object, not a SQL fragment. `GET /rollup` accepts `metric`, a closed-open `start` and `end`, at most two declared dimensions, optional declared filters, and an optional `snapshot_id`. `GET /events` additionally requires a selective filter, an order fixed to `(event_time,event_id)`, a `page_size` from 1 through 500, and an opaque `cursor` only after the first page. `POST /jobs` accepts the same bounded analytic shape but does not accept raw SQL, user-provided table names, arbitrary expressions, or a caller-supplied execution plan.

Canonicalization is a correctness operation. The router parses timestamps to UTC, sorts filter keys and list values where order is semantically irrelevant, rejects duplicate semantic fields, inserts explicit defaults, and serializes the normalized object before hashing it. Thus `region=7&device=3` and `device=3&region=7` denote one cache key, whereas `region=7` and `region=8` cannot collide. A request has a `contract_version`; a client that requires a previous definition sends that version explicitly and receives a `409 CONTRACT_RETIRED` if it is no longer readable.

Every successful response contains `request_id`, `contract_version`, `query_digest`, `snapshot_id`, `watermark_min`, `watermark_max`, `rows_scanned`, `rows_returned`, `result`, and field-level approximation metadata. A page response also contains `next_cursor` only when more rows exist. `next_cursor: null` means the pinned snapshot is exhausted; it never means that the service timed out. This distinction lets a client distinguish completion from an absent continuation.

### 8b Authentication, authorization boundary, and quota identity
Authentication establishes a credential identity before the rate limiter and planner run. An illustrative credential is a short-lived signed access token validated for issuer, audience, expiry, key identifier, and signature; its subject maps to a caller class. The API rejects absent or expired tokens with `401`, rejects a valid token lacking the endpoint scope with `403`, and returns `429` only after an authenticated caller exhausts a declared quota. Returning `429` for an invalid token would reveal that a credential form is otherwise plausible.

Authorization in this case is deliberately narrow. A scope such as `analytics.rollup.read` selects the allowed endpoint and caller class; it is not evidence that the API may expose every tenant's data. The stated API is not a row-level-security boundary, so it only serves a pre-authorized single-tenant dataset or sits behind a separately verified policy-enforcing gateway. A request carrying a free-form `tenant_id` must be rejected unless that identity is bound by the upstream authorization layer.

Quota identity is the credential subject plus contract version, not the source IP. IP-based limits merge a shared office into one user and let a credential rotate through addresses. Credential-based accounting makes retries, byte budgets, and abuse investigation attributable. The request log stores a keyed or truncated credential reference rather than the raw bearer token; retaining a token in a log converts observability into a credential leak.

### 8c Idempotent asynchronous submission and recovery state machine
Async job submission is idempotent within a declared ten-minute deduplication window. The server computes `job_key = SHA-256(contract_version || canonical_request || snapshot_selector || credential_subject)`; the client may supply an idempotency key, but the server verifies it maps to the same canonical payload. Reusing a client key with a changed payload returns `409 IDEMPOTENCY_CONFLICT`, because accepting it would make a network retry silently execute a different question.

A job record transitions only `queued -> running -> complete`, `queued -> cancelled`, or `running -> failed`. `complete`, `cancelled`, and `failed` are terminal. The executor writes result bytes to durable storage, records their checksum and pinned snapshot, and only then commits `complete`; a polling client cannot observe `complete` before the result is readable. A worker crash after persistence but before the state transition is recovered by verifying the checksum and finishing the transition. A crash before persistence leaves the job retryable because no result has been committed.

Cancellation is advisory until the executor reaches a safe cancellation point. The API returns `202` for an accepted cancellation request and exposes `cancelling` rather than promising immediate termination. Once cancelled, a job cannot be revived under the same key: a new request must have a new idempotency key, preserving a clear audit trail. Expiry removes result bytes only after the declared retrieval window; the job metadata remains long enough to return `410 JOB_EXPIRED` rather than an ambiguous `404`.

### 8d Observability: evidence for contract decisions, not request diaries
One structured event is emitted at admission, planning, execution completion, and response completion, correlated by `request_id` and `query_digest`. Required dimensions are endpoint, contract version, caller class, decision (`cache_hit`, `inline`, `async`, `refused`), error code, snapshot age bucket, estimated and actual scanned bytes, rows returned, page size, retry count, and job state. High-cardinality raw filter values and bearer tokens are excluded from metrics and sampled logs.

The primary service-level indicators are: inline latency distribution by endpoint and decision; planner underestimation ratio `actual_bytes / estimated_bytes`; cache hit rate by canonical key family; refusal rate by typed code; snapshot-age distribution using the minimum watermark; deduplication rate; cursor decode failures; and byte-budget rejections. Each indicator maps to an owner and a decision. For example, a rising planner ratio changes the bound or estimate model; a rising `FILTER_REQUIRED` rate changes documentation or client integration, not warehouse capacity.

Alerting separates symptoms from safety invariants. A latency alert can page only after the p99 budget is breached for a stated window, whereas any response missing `snapshot_id`, any actual scan above an admitted estimate tolerance, or any cache key collision in a canary pair is a correctness alert. The distinction avoids paging on ordinary demand while still treating provenance loss as urgent. Traces retain timings and identifiers needed to reconstruct a request, while aggregate metrics retain no row-level payload.

### 8e Recovery runbook for a failed request path
For a `503 PARTIAL_READ`, the service records the unreadable partition identifiers and does not issue a success envelope. The operator first checks whether the storage read failure is transient, then retries the partition read against the same snapshot if that snapshot remains readable; otherwise the caller receives `409 STALE_SNAPSHOT` and must choose a fresh answer. The runbook forbids blending old and new partition reads, because that creates a result with no reproducible state.

For a planner underestimate, the request is stopped before serialization when actual scanned bytes cross the hard ceiling. The response is `422 BUDGET_EXCEEDED` with the observed estimate-versus-actual ratio recorded. The immediate mitigation is to disable the affected shape or lower its admitted range; the permanent correction updates statistics or clustering evidence and is evaluated through the independent harness before the bound changes.

For a cache corruption or collision, the response cache namespace for the affected contract version is invalidated, cache reads are disabled for that endpoint, and constructed pair tests verify distinct canonical requests produce distinct keys. Re-enabling cache without the pair test restores performance but not correctness. For an async worker loss, lease expiry returns the job from `running` to `queued` only when no durable result checksum exists, preventing two workers from publishing competing results.

## 9. Evaluation & Validation
Each test states numerator, denominator, population, window, comparator, guardrails, uncertainty, and status. Status is `not-executed` throughout: no endpoint described here exists. Claiming otherwise would invert the case's own argument.

### T01 — inline/async boundary is respected exactly
**Numerator:** requests answered inline whose estimated scan exceeds the 340 ms allowance.
**Denominator:** requests whose estimate exceeds the allowance.
**Population / window:** 500 constructed requests spanning 1, 7, 30, and 90 days, evaluated in one batch per shape.
**Comparator:** planner estimate recomputed independently from range × rows/day × bytes/row.
**Guardrail:** zero inline answers above the allowance; every such request carries a deferral or refusal code.
**Uncertainty:** estimates are exact for this synthetic table and conservative for a clustered one, so a real system would show some legitimate deferrals.
**Status:** not-executed (proposed).

### T02 — refusal carries a usable remedy
**Numerator:** refusals containing a remedy string naming an actionable alternative.
**Denominator:** all refusals produced.
**Population / window:** all refusal codes across the 500-request set.
**Comparator:** a fixed schema check plus a human review of 20 sampled messages.
**Guardrail:** 100% schema compliance; at least 90% of sampled messages judged actionable by a reviewer who did not write them.
**Uncertainty:** the human judgement component is subjective; the schema component is exact.
**Status:** not-executed (proposed).

### T03 — cache key separation
**Numerator:** pairs of requests differing in one filter that return distinct results.
**Denominator:** all such pairs constructed.
**Population / window:** 200 pairs differing by region, device, granularity, and snapshot bucket in turn.
**Comparator:** direct comparison of returned aggregates.
**Guardrail:** zero pairs returning identical results; a collision is a correctness failure, not a tuning issue.
**Uncertainty:** none for constructed pairs; the risk is pairs nobody thought to construct.
**Status:** not-executed (proposed).

### T04 — cursor stability under concurrent inserts
**Numerator:** duplicate or skipped rows across a full pagination traversal while rows are inserted.
**Denominator:** total rows delivered across all pages.
**Population / window:** 50 traversals of 200 pages with 463 rows/s inserted, over 5 simulated minutes each.
**Comparator:** the union of delivered rows against the snapshot's row set.
**Guardrail:** zero duplicates and zero skips relative to the pinned snapshot.
**Uncertainty:** insertion timing is simulated, so an unlucky phase alignment may not be exercised.
**Status:** not-executed (proposed).

### T05 — end-to-end success under fan-out with one degraded shard
**Numerator:** requests failing when one of six shards fails 5% of the time.
**Denominator:** requests issued.
**Population / window:** 10,000 requests across the peak hour profile.
**Comparator:** 1 − (1 − 0.05) = 5% predicted by `1 − prod(1 − f_i)`.
**Guardrail:** observed failure within the predicted value plus two standard errors; a materially lower figure indicates silent fallback that must be disclosed.
**Uncertainty:** binomial sampling with n = 10,000 gives a standard error near 0.22 percentage points at p = 0.05.
**Status:** not-executed (proposed).

### T06 — top-k completeness against a full ordering
**Numerator:** global top-10 answers missing at least one true member.
**Denominator:** top-10 queries issued.
**Population / window:** 1,000 queries over a distribution where true top members are assigned uniformly to six shards.
**Comparator:** full-ordering reference computed without truncation.
**Guardrail:** missing rate at or below the 0.16% bound for k = 6, and no k = 3 configuration accepted at all.
**Uncertainty:** uniform assignment is constructed, so a skewed real key would perform worse.
**Status:** not-executed (proposed).

### T07 — retry behaviour under declared degradation
**Numerator:** total attempts when the service reports 8% retryable failures.
**Denominator:** logical requests.
**Population / window:** 20,000 requests, three-attempt budget, jittered delays.
**Comparator:** predicted amplitude 1.0864.
**Guardrail:** observed amplitude within 5% of prediction; a higher figure means clients are retrying more than declared.
**Uncertainty:** client behaviour is simulated; production clients may retry on non-retryable codes.
**Status:** not-executed (proposed).

### T08 — snapshot reproducibility
**Numerator:** repeated requests against the same snapshot ID returning different bytes.
**Denominator:** repeated request pairs.
**Population / window:** 100 pairs issued 72 seconds apart against a 90-day shape.
**Comparator:** byte-level comparison of the aggregate payload.
**Guardrail:** zero differences; any difference means a partition was still being written and the snapshot did not cover it.
**Uncertainty:** a settled partition is trivially reproducible, so the test is only meaningful while late data is landing.
**Status:** not-executed (proposed).

### T09 — latency budget compliance by shape
**Numerator:** inline requests exceeding 800 ms.
**Denominator:** inline requests.
**Population / window:** 100,000 requests at the peak-hour profile, 30 simulated minutes.
**Comparator:** the 800 ms contract value, with per-component allowances recorded.
**Guardrail:** p99 at or below 800 ms and p50 at or below 300 ms.
**Uncertainty:** simulated service times cannot reproduce real queue dynamics; the test validates the guard logic, not the service.
**Status:** not-executed (proposed).

### T10 — merged percentile correctness
**Numerator:** merged p99 computed from per-shard histograms versus merged p99 from the full distribution.
**Denominator:** shard configurations tested.
**Population / window:** 12 configurations including the 5,100-request worked example.
**Comparator:** exact rank computation over the union.
**Guardrail:** difference within one histogram bucket width; a larger gap means the buckets are too coarse.
**Uncertainty:** bucket width is the only error source and is known by construction.
**Status:** not-executed (proposed).

### T11 — rate limit fairness across caller classes
**Numerator:** requests served per class while one class saturates its bucket.
**Denominator:** requests offered per class.
**Population / window:** three classes, 10 simulated minutes, one class at 3× its refill rate.
**Comparator:** declared refill rates.
**Guardrail:** other classes retain at least 95% of their nominal goodput.
**Uncertainty:** the test measures isolation of buckets, not of downstream capacity; a saturated class still consumes warehouse queue.
**Status:** not-executed (proposed).

### T12 — approximate field labelling
**Numerator:** responses containing an approximate field without a field-level flag.
**Denominator:** responses containing any approximate field.
**Population / window:** all distinct-count requests in the 500-request set.
**Comparator:** response schema against the contract's approximation list.
**Guardrail:** zero unflagged approximations; the flag must be machine-readable, not prose.
**Uncertainty:** none; the check is structural.
**Status:** not-executed (proposed).

### T13 — job deduplication
**Numerator:** distinct job records created for identical query hashes within the window.
**Denominator:** submissions with a repeated hash.
**Population / window:** 200 submissions, 40 of them duplicates within 10 minutes.
**Comparator:** job table cardinality.
**Guardrail:** exactly one job per hash per window; compute spend must not double.
**Uncertainty:** a hash that ignores a filter would deduplicate incorrectly, so T03 and T13 must both pass.
**Status:** not-executed (proposed).

Acceptance for a rehearsal is a conjunction: the boundary tests pass, no cached result crosses a snapshot bucket, refusals are actionable, and no approximate field is unlabelled. A failure on any single test blocks the rehearsal rather than being averaged into a pass rate, which is the same rule as in the data quality case and for the same reason.

## 10. Action Taken
No service was built and no endpoint was called by this manuscript. The proposed next actions are ordered so each produces a checkable artefact: agree the snapshot and watermark rules with the platform team in writing; obtain caller-class staleness tolerances from the three consuming teams; build the independent recomputation harness before the service, so the oracle is not written by the author of the code under test; construct the 500-request set including every refusal path; and run the thirteen tests offline.

A recorded non-action: the team did not adopt offset pagination, despite its simplicity for clients, because the quadratic scan cost and the concurrent-insert instability both fail tests T04 and T09 and neither can be fixed by tuning.

## 11. Outcome & Results
**Outcome status: not measured.** The verified results are the recomputed arithmetic in Section 6: λ_peak = 60.0 req/s from 1.2M requests/day at an 18% peak share; L = 41.7 in flight on mean service times against 75.6 on p99 times, requiring 108 rather than 64 workers; 17 million rows affordable inside a 340 ms scan allowance, which is 0.425 of one day of the feed and 212× short of a 90-day scan; 0.99^6 = 94.15% end-to-end success implying a per-shard requirement of 99.83%; a per-shard top-3 losing a true top-10 member 41.8% of the time against 0.16% at top-6; offset pagination at 0.50% efficiency on page 200; three-attempt residual failure 5.12e-4 at 1.0864× amplitude; a 90-day query letting 33,333 rows arrive mid-flight; and 1.63% sketch error at 4,096 registers.

No claim is made about real latency, real traffic, real prices, real shard reliability, or adoption. Section 3a also contains a deliberate unit slip that Section 6c corrects, retained as a worked example of why price tables need their own review.

## 12. Failure Modes & What Went Wrong
### F-01 — the cache answers for the wrong filter
**Symptom:** two dashboards show different numbers for the same region, and one of them is stale by exactly one filter change.
**Competing causes:** (a) the cache key omits a filter; (b) the key includes filters but canonicalization orders them differently per caller; (c) the key omits the snapshot bucket so bytes outlive their label.
**Discriminating check:** compute the key for two requests differing in one filter and compare hashes. Identical hashes with different expected results isolate (a) or (b); different hashes with a response whose snapshot predates the request isolate (c).
**Immediate mitigation:** purge the cache namespace for the affected endpoint and serve uncached until the key includes the missing component.
**Correction:** derive the key from the canonicalized request object rather than from a hand-built string, and test it with a property that any two requests differing in any semantic field must differ in key.
**Verification:** replay 200 constructed pairs from T03 and assert 200 distinct results.

### F-02 — the watermark is believed rather than checked
**Symptom:** a nightly report is consistently 3% low and nobody notices for two weeks, because 3% looks like normal variation.
**Competing causes:** (a) the watermark lags by more than the declared SLA; (b) the watermark advances but late rows land behind it; (c) the query filters on `received_time`, so it silently excludes rows that arrived after the boundary.
**Discriminating check:** compare the reported watermark with a direct `max(received_time)` on the underlying object, and re-run the same query with an `event_time` filter. A watermark far behind wall clock points to (a); matching watermarks with low counts point to (b); a large count difference between the two filter modes points to (c).
**Immediate mitigation:** serve the snapshot boundary in the response and mark the report as provisional.
**Correction:** publish watermark lag as a first-class metric and refuse to serve a "latest" claim when the lag exceeds the declared threshold.
**Verification:** replay a known-complete partition and confirm the count matches an independent recount exactly.

### F-03 — per-shard top-k silently truncates the global answer
**Symptom:** a "top ten customers" list changes composition between two runs with the same data, and one known-large customer is missing.
**Competing causes:** (a) the per-shard limit is below the requirement derived in 6f; (b) the grouping key is skewed so one shard holds most candidates; (c) ties are broken differently per shard, so the same value appears in different orders.
**Discriminating check:** compute the true top-10 by full ordering and compare membership. Missing members with an even per-shard limit points to (a); a single shard holding far more than 1/6 of candidates points to (b); membership matching but order differing points to (c).
**Immediate mitigation:** raise the per-shard limit to at least 6 for a global top-10 and re-run.
**Correction:** derive the per-shard limit from M and n in the contract, and report the number of candidates per shard in the envelope so skew is visible.
**Verification:** run T06 across 1,000 queries and confirm the missing rate is under 0.16%.

### F-04 — retries turn a degradation into an outage
**Symptom:** a partial shard failure produces a full outage within ninety seconds, and load graphs show traffic rising while capacity is unchanged.
**Competing causes:** (a) clients retry without honouring `retry_after`; (b) retries are not jittered, so retry waves synchronize; (c) the service retries internally as well as the client, doubling the amplification.
**Discriminating check:** histogram retry arrival timestamps. Peaks at a fixed interval indicate (b); arrivals earlier than `retry_after` indicates (a); attempt counts above the client's declared budget indicate internal retries (c).
**Immediate mitigation:** enable the kill switch that returns a fast static error, which reduces load immediately and makes the outage short.
**Correction:** document and enforce `retry_after`, add exponential jitter with a floor, and forbid internal retries on the read path.
**Verification:** replay the 8% failure scenario from T07 and confirm amplitude stays within 5% of 1.0864.

### F-05 — percentiles are averaged and the tail disappears
**Symptom:** every component reports a healthy p99 while the service p99 is four times worse, and the incident review cannot locate the cause.
**Competing causes:** (a) per-shard percentiles are averaged rather than merged; (b) one shard carries more than 1% of traffic and defines the merged p99 alone; (c) the monitoring window is too short to include the slow shape.
**Discriminating check:** compute the merged p99 from per-shard histograms and compare with the reported figure. A large gap with equal shard volumes points to (a); a volume comparison showing one shard above 1% points to (b); a longer window that raises the figure points to (c).
**Immediate mitigation:** report the merged percentile from histograms immediately, even if buckets are coarse.
**Correction:** fix sketch parameters and bucket widths in the contract so merged percentiles are computable and comparable across services.
**Verification:** run T10 and confirm the histogram-derived p99 is within one bucket width of the exact rank computation.

### F-06 — a partial result is returned with a success status
**Symptom:** a caller computes a total from a response that omitted three partitions, and the total is 4% low with no indication.
**Competing causes:** (a) an unreadable partition was skipped rather than reported; (b) the serialization size cap truncated the page without setting `truncated`; (c) the query was cancelled at the budget and partial aggregates were returned.
**Discriminating check:** inspect the envelope's `exclusions`, `truncated`, and `rows_scanned` fields. A zero `exclusions` with a low count points to (a); `truncated: false` with a short page points to (b); `rows_scanned` far below the estimate points to (c).
**Immediate mitigation:** fail closed with `PARTIAL_READ` for the affected requests and recompute.
**Correction:** make the envelope assembler the only writer of the completeness claim, and refuse to emit a success status when any exclusion or truncation is present.
**Verification:** inject one unreadable partition and confirm the response is `PARTIAL_READ` with the partition named.

### F-07 — the bound is raised for one caller and becomes everyone's bound
**Symptom:** a 90-day shape that was supposed to be async is being served inline after a "temporary" change, and p99 doubles for every other caller.
**Competing causes:** (a) the shape's bound was changed in the contract; (b) the bound is enforced client-side only, so a different client bypasses it; (c) the estimate is computed from a cached row count that understates the range.
**Discriminating check:** diff the contract version recorded in the response. A changed bound points to (a); an unchanged contract with a bypassing caller points to (b); an estimate far below actual `rows_scanned` points to (c).
**Immediate mitigation:** re-pin the contract to the previous version for the affected endpoint.
**Correction:** make bounds contract fields with change control and require the planner to compare against the estimate computed at request time, not a stale count.
**Verification:** replay the 90-day shape and confirm a 422 `RANGE_TOO_LARGE` with the async remedy attached.

### F-08 — the async job expires before the client polls
**Symptom:** a scheduled report fails intermittently with `JOB_NOT_FOUND` although the job was submitted successfully.
**Competing causes:** (a) the retention window is shorter than the client's poll interval; (b) the client submits, waits for a scheduled slot, then polls after expiry; (c) the job failed and the record was deleted rather than marked failed.
**Discriminating check:** compare submission time, completion time, expiry, and poll time for failing cases. Poll after expiry with a completed job points to (a); poll after a long wait with a fast job points to (b); a missing record with no completion timestamp points to (c).
**Immediate mitigation:** extend the retention window temporarily and have the client resubmit.
**Correction:** derive retention from the declared maximum poll interval in the contract, and never delete failed job records before expiry.
**Verification:** run a job with a forced delay beyond retention and confirm the client receives a distinct `EXPIRED` code rather than `JOB_NOT_FOUND`.

### F-09 — approximate and exact answers are indistinguishable downstream
**Symptom:** two teams report uniques that differ by 4%, and both are "correct" according to their own API calls.
**Competing causes:** (a) one call used the sketch and the other the exact path; (b) the sketches used different register counts across services; (c) one call crossed a snapshot boundary that the other did not.
**Discriminating check:** inspect the field-level `approximate` flag and the sketch parameters in the response envelope. Different flags points to (a); equal flags with different parameters points to (b); equal parameters with different snapshots points to (c).
**Immediate mitigation:** declare one canonical field for each metric and route both teams to it.
**Correction:** fix sketch parameters in the contract and make the approximate flag machine-readable at field level, so a consumer cannot silently treat a sketch as exact.
**Verification:** run T12 and confirm zero unflagged approximations across the whole request set.

**Takeaway:** every failure above returns a plausible answer with invalid provenance. The remedy is never more dashboards; it is to make the provenance fields mandatory and machine-checkable.

## 13. Transferable Lesson
- An API is a set of refusals as much as a set of responses; write the refusals first and derive the bounds from arithmetic.
- Little's law sizes concurrency from the mean, so also compute it from p99 service times or you will under-provision by the ratio of tails.
- The synchronous budget converts into a byte allowance, and the byte allowance converts into a maximum row count, which is the honest bound for inline serving.
- Scan time and cost are linear in the requested range, so a range bound is simultaneously a latency bound and a cost bound.
- A cache hit rate should be compared with its ceiling (`1 − distinct/requests`); a large gap means key granularity, not demand diversity.
- Fan-out multiplies failure: per-shard reliability must be set from the end-to-end target, not from a component dashboard.
- A per-shard top-k is not a distributed top-k; derive k from M and n, and report candidate counts per shard.
- Percentiles do not average; merge histograms with contract-fixed buckets or the tail vanishes from every report.
- Reproducibility requires naming a snapshot, and the cost of not naming one is quantifiable as rows arriving mid-query.
- Retries amplify load exactly when capacity is short, so `retry_after` and jitter are correctness features, not politeness.

## 14. Connections & Cross-References
This case uses Python process control and HTTP semantics, SQL aggregation and predicate pushdown, Linux-style resource limits for memory-bounded checks, and REST identity rules for cursors and job IDs. Those supply the mechanics. The decision content comes from queueing arithmetic, probability, and an explicit refusal contract.

It is the sibling of `01_Build_Data_Quality_CLI.md` in this unit: that case gates whether bytes may enter the table, and this case bounds what may be read out of it. Both share the same three disciplines — a pinned contract version, a named population, and an outcome that is never a silent pass — and both refuse to treat a plausible number as evidence.

### Interview pack
#### Q1. Why is the decision unit a triple rather than an endpoint or a request?
**Answer:** Because `GET /events` is several products at once: a one-day rollup is a 0.96 GB scan and a ninety-day scan is 86.4 GB, so the same endpoint has two different answers to "inline or async". The triple `(endpoint, query_shape, caller_class)` pins all three inputs that change the decision.
**Reasoning:** the decision depends on cost, which depends on shape, and on tolerance, which depends on caller class. Collapsing either input produces a contract that is right for one consumer and wrong for another.
**Follow-up:** ask which fields in a response would let a reviewer reconstruct the triple. The answer is endpoint, canonicalized shape digest, and caller class, all of which belong in the envelope.
**Red flag:** answering "per endpoint", which implies every shape gets the slowest consumer's bounds.
**Scoring:** full credit names all three components and gives a concrete shape pair that differs in decision.

#### Q2. Compute the peak request rate and say why the daily average is the wrong input.
**Answer:** 1,200,000 requests/day at an 18% peak-hour share gives 1,200,000 × 0.18/3600 = 60.0 req/s, against a daily mean of 13.89 req/s. Capacity sized on the mean is 4.3× short at the peak.
**Reasoning:** queueing failure is a property of the busy hour, not of the day. A service with 65% daily utilisation can still be saturated for one hour every day.
**Follow-up:** ask what breaks the peak assumption. A five-minute burst at the top of the hour can raise the instantaneous rate by up to 12× over the hourly figure.
**Red flag:** dividing by 86,400 and calling that the design load.
**Scoring:** full credit computes both figures and names the burst caveat.

#### Q3. Little's law says 41.7 requests in flight; why buy 108 workers?
**Answer:** Because 41.7 comes from mean service times. On p99 times, 54.0 req/s × 0.40 s plus 6.0 req/s × 9.00 s gives 75.6 in flight, which already exceeds 64 workers at 118% utilisation. At a 0.7 target, 75.6/0.7 = 108.
**Reasoning:** Little's law is an identity about averages and cannot size a tail-latency budget on its own. The mean case leaves 35% headroom; the p99 case is over capacity.
**Follow-up:** ask whether 108 is sufficient in the presence of retries. With a 1.0864 amplitude it is not, and the figure rises to about 117.
**Red flag:** quoting 41.7 as the requirement and moving on.
**Scoring:** full credit shows both products, the utilisation, and the reason the mean is insufficient.

#### Q4. The scan allowance is 340 ms. How many rows can be answered inline, and what does that rule out?
**Answer:** 0.340 s × 1.2 GB/s = 408 MB, and 408 MB/24 B = 17,000,000 rows. That is 0.425 of a single day of the 40M-row feed, so any multi-day raw scan is excluded — a 90-day shape needs 212× the affordable bytes.
**Reasoning:** bytes and time convert through a fixed throughput, and bytes convert to rows through a fixed row width. Both conversions are contract constants, so the bound is arithmetic rather than policy.
**Follow-up:** ask whether parallelism rescues it. At six shards, 72 s becomes 12 s, still 15× over the 800 ms budget, so no.
**Red flag:** proposing "more workers" as the fix for a single-request latency violation.
**Scoring:** full credit converts time to bytes to rows and states the multiple that rules out the long range.

#### Q5. Cache hit rate is 72% but the ceiling is 99.6%. What does that gap mean?
**Answer:** With 4,800 distinct keys a day against 1.2M requests, uniform demand would hit 99.6%. Reaching only 72% means the key varies per request — a second-precision timestamp, a caller ID, or a non-canonical filter order — so identical questions miss each other.
**Reasoning:** the ceiling is a property of demand diversity; the achieved rate is a property of key construction. A large gap localizes the fault to the key, not to the workload.
**Follow-up:** ask for the collision test that proves the key is not *too* coarse. Two requests differing in one filter must produce different keys and different results.
**Red flag:** celebrating a 72% hit rate without computing what was achievable.
**Scoring:** full credit computes both numbers and names three concrete key defects.

#### Q6. Six shards each at 99% availability. What is end-to-end success and what must each shard be?
**Answer:** 0.99^6 = 0.9415, so 94.15% end-to-end. To hold 99% end-to-end each shard must reach 1 − 0.99^(1/6) = 0.998333, i.e. 99.83%.
**Reasoning:** a request touching all shards succeeds only if none fails, so component success multiplies. The per-component target is therefore a root of the end-to-end target, not the target itself.
**Follow-up:** ask what invalidates the multiplication. Correlated failure in one storage cluster collapses it to the single-shard number, and the failure-timestamp histogram is the check.
**Red flag:** asserting 99% per shard is "good enough" because 99% sounds high.
**Scoring:** full credit computes 94.15%, the 99.83% requirement, and the correlation caveat.

#### Q7. A global top-10 is built from per-shard top-3 across six shards. What is the risk, and what limit do you need?
**Answer:** A shard can hold four or more of the true top ten with probability 0.0697, so the union bound over six shards gives 41.8% chance of missing a member. You need k = 6, where the bound falls to 0.16%.
**Reasoning:** truncation before merging loses candidates that only the merge could have identified. The per-shard limit must exceed the even split M/n = 1.67 by a margin that grows with M.
**Follow-up:** ask when the uniform-assignment assumption fails. A skewed grouping key can put 60% of candidates on one shard, and the bound must be recomputed from the observed share.
**Red flag:** computing k = ceil(10/6) = 2 and declaring it correct.
**Scoring:** full credit computes the binomial tail, the union bound, and a corrected k with its residual risk.

#### Q8. Why is offset pagination unacceptable here, in two separate arguments?
**Answer:** Cost: page 200 scans 200 × 500 + 500 = 100,500 rows to return 500, i.e. 0.50% efficiency and 201× waste; a full 200-page traversal scans 10,050,000 rows for 100,000 delivered. Correctness: with 463 rows/s arriving, about 93 rows land per page, so a traversal duplicates and skips.
**Reasoning:** the two arguments are independent, and either alone would be sufficient. Cost is quadratic in page count; instability is a property of filtering by a moving offset.
**Follow-up:** ask what keyset requires. A total order on an indexed tuple, `(event_time, event_id)`, with the tie-break present so a 0.3% tie rate cannot cause overshoot.
**Red flag:** choosing offset because it is easier for clients.
**Scoring:** full credit gives both arguments with numbers and names the composite cursor.

#### Q9. Three retries at 8% per attempt: what residual failure and what load cost?
**Answer:** Residual 0.08^3 = 0.000512, i.e. 512 per million; amplitude 1 + 0.08 + 0.0064 = 1.0864, so 8.64% more capacity. A fourth attempt would add only 0.05% of load for a 0.04% gain.
**Reasoning:** retries reduce failure geometrically but cost load geometrically too, and the marginal benefit collapses after the third attempt while the marginal tail latency does not.
**Follow-up:** ask what happens at 40% per-attempt failure. Amplitude becomes 1 + 0.40 + 0.16 = 1.56, so a struggling service gets 56% more traffic exactly when it cannot absorb it.
**Red flag:** adding retries without `retry_after` or jitter.
**Scoring:** full credit computes both the residual and the amplitude, and names the feedback loop.

#### Q10. Every component's p99 is within its allowance, yet the service p99 is 4.6× worse than the naive average. Explain.
**Answer:** Because per-shard percentiles cannot be averaged. In the worked example the merged p99 is 5,000 ms while the average of per-shard p99s is 1,095 ms, because one shard holding 100 of 5,100 requests contributes more than the 1% needed to define the merged p99.
**Reasoning:** a percentile is a rank in an ordering, and averaging ranks discards exactly the slow population that defines the tail. Merging requires histograms or a mergeable sketch.
**Follow-up:** ask what fixes comparability. Contract-fixed bucket widths or sketch parameters, because two services with different parameters produce percentiles that cannot be combined.
**Red flag:** reporting a weighted average of shard p99s and calling it the service p99.
**Scoring:** full credit computes both figures, explains the 1% threshold, and prescribes histograms.

#### Q11. What is the cost of not naming a snapshot in a 90-day response?
**Answer:** At 462.96 rows/s and 72 s per query, 33,333 rows arrive while it runs, so two identical requests 72 seconds apart return different numbers and neither can be reproduced.
**Reasoning:** the table is appended to continuously; without a pinned readable set, "the answer" has no referent. Naming a snapshot is what makes the number a fact about a state rather than about a moment.
**Follow-up:** ask why per-partition watermarks rather than a global snapshot. A coordinated global read is not affordable; per-partition watermarks are achievable and must be reported honestly, including the minimum.
**Red flag:** claiming a snapshot while reading each partition at its own latest state.
**Scoring:** full credit computes the mid-flight arrivals and states what a snapshot must contain.

#### Q12. When is a 1.63% sketch error acceptable, and when is it a defect?
**Answer:** Acceptable when the consumer's decision tolerates about ±32,500 on a 2,000,000 count, as a dashboard does. A defect when the consumer reconciles money, where a 1.63% error is a wrong number with a confident label.
**Reasoning:** the sketch is a constant-memory approximation whose error is known, so the question is not accuracy but whether the decision is sensitive at that scale.
**Follow-up:** ask what makes it unsafe even when accuracy is fine. Merging sketches with different register counts is silently wrong, exactly like a cache-key collision.
**Red flag:** describing the error only in documentation rather than flagging it per field in the response.
**Scoring:** full credit computes the absolute error, states the tolerance condition, and names the merge hazard.

#### Q13. An over-budget query arrives. Why is "return the first 500 rows with a warning" the wrong answer?
**Answer:** Because a warning is not a semantic marker to a machine. The caller aggregates 500 of 40 million rows and reports a total, and the response's success status is the only thing its code reads. The correct outcomes are a typed refusal with a remedy or an async deferral.
**Reasoning:** truncation converts a bounded-latency problem into an unbounded-correctness problem, and it does so silently, which is the failure class this whole unit is about.
**Follow-up:** ask what makes truncation acceptable anywhere. Only when the response carries `truncated: true` and the client's contract requires it to handle that flag, which most clients do not.
**Red flag:** treating the warning field as sufficient mitigation.
**Scoring:** full credit rejects the truncation and states the two acceptable outcomes with their codes.

#### Q14. What would you measure to prove the API's numbers are right, given no ground truth?
**Answer:** An independent recomputation harness that computes the same aggregates from raw rows with different code, plus a settled-snapshot comparison: the same request twice against a partition known to be complete must return identical bytes.
**Reasoning:** with no external truth, agreement between two independent implementations is the strongest available evidence, and it only means something if the oracle was written by someone other than the implementer.
**Follow-up:** ask what the harness cannot prove. It validates declared properties, not the metric definition; if the contract defines active users wrongly, both implementations agree on the wrong number.
**Red flag:** using the dashboard as ground truth, which makes the API correct by definition.
**Scoring:** full credit names independent recomputation, the settled-snapshot test, and the definitional limit.

#### Q15. Which single input, if changed, would reverse the inline/async decision?
**Answer:** Scan throughput. At 1.2 GB/s a 7-day shape takes 5.6 s and must be async; at about 17 GB/s it would fit in 340 ms. No plausible change to worker count, cache hit rate, or latency budget achieves the same reversal.
**Reasoning:** the inline bound is bytes divided by throughput, so only a throughput change of the required multiple moves the boundary. Worker count shortens wall time by sharding but not the per-request byte cost.
**Follow-up:** ask what a 30× throughput improvement would still not fix. The 90-day shape at 72 s/30 = 2.4 s remains over budget, so async is still required for the largest range.
**Red flag:** naming the cache, which improves the common case but cannot make an uncached scan fit.
**Scoring:** full credit identifies throughput, computes the required multiple, and notes what it still fails to fix.

#### Q16. Design the error taxonomy and justify why refusal codes must be distinct.
**Answer:** Distinct codes for range too large, filter required, rate limited, budget exceeded, unknown field, stale snapshot, job not found, partial read, and approximate only. Distinct because each has a different remedy and a different retryability, and a caller cannot branch on prose.
**Reasoning:** at 03:00 the on-call engineer and the client library both need to decide from the code alone whether to retry, narrow, defer, or stop. A single 400 collapses four different actions into one.
**Follow-up:** ask why `APPROXIMATE_ONLY` is 200 rather than an error. Because failing an approximate answer pushes callers to the console, where they get exactness and an unbounded scan.
**Red flag:** a single error code with a message field carrying the meaning.
**Scoring:** full credit lists the codes, maps each to a remedy and a retryability, and justifies the 200-status exception.

## 15. Quick Reference Card
| Field | Rule |
|---|---|
| Decision unit | `(endpoint, query_shape, caller_class)`; bounds are promises to every caller |
| Outcomes | inline, cached, async job, typed refusal — never a truncated success |
| Capacity | L = λW from p99 service times; 108 workers, not the 64 implied by the mean |
| Inline bound | 340 ms × 1.2 GB/s = 408 MB = 17M rows = 0.425 of one day of the feed |
| Range cost | linear: 90 days = 86.4 GB = 72 s = $0.432, exactly 90× the one-day shape |
| Cache | compare hit rate with `1 − distinct/requests`; 72% against a 99.6% ceiling means broken keys |
| Fan-out | per-shard reliability = root of the end-to-end target: 99.83%, not 99% |
| Top-k | per-shard k must exceed M/n with margin; k = 6 for a global top-10 over 6 shards |
| Pagination | keyset on `(event_time, event_id)`; offset is 0.50% efficient at page 200 |
| Reproducibility | every response names a snapshot and its watermark; 33,333 rows arrive during a 90-day query |
| Evidence | illustrative arithmetic only; no endpoint was built or called |
| Escalate | snapshot ambiguity, bound change, partial read, or any request without provenance |
