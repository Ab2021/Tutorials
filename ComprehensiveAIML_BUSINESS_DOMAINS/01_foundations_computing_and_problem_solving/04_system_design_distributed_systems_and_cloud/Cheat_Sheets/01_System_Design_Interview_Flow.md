---
collection: "system-design-distributed-systems-cloud"
file_id: "01"
topic: "System design interview flow"
level: 2
status: "draft; unreviewed"
version: "2.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Change to the interview time box, the capacity unit, the latency objective, the peak-factor assumption, or any formula block whose validity condition is relaxed"
notation_style: "plain-text"
prerequisites: ["Percentage and unit arithmetic", "Basic queueing vocabulary: arrival rate, service rate, utilisation"]
pattern_tags: ["requirements-framing", "capacity-sizing", "latency-budgeting", "failure-analysis"]
source_status: "OPEN — visible source titles consulted as inventory only; no source bodies available"
owner_role: "staff engineer running or scoring a system design loop"
source_index: "INDEX.md"
---

# System Design Interview Flow

## 1. File Header

> **Purpose:** Turn a 35–50 minute system design conversation into a repeatable engineering procedure that produces decisions, numbers with units, and failure policy — not a diagram.
> **Prerequisites:** Percentage arithmetic; unit conversion between bytes and bits; the words arrival rate, service rate, utilisation, percentile; ability to distinguish a measured value from a constructed one.
> **Key outcome:** Given a one-sentence product prompt, produce within the time box: a scoped requirement set, a source-of-truth and consistency contract, a sized capacity estimate with a stated peak factor and headroom target, an end-to-end latency budget, a failure/recovery policy, and a named validation plan.
> **This file does NOT cover:** individual system catalogues (feed ranking, chat, payments) as separate designs; interviewing psychology and salary negotiation; distributed-consensus proofs; vendor product comparisons.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED arithmetic and queueing results from stated premises; ILLUSTRATIVE workload and cost inputs; OPEN where a claim needs a measured deployment to settle it.
> **Retrieval cue:** If you are 15 minutes into a design and have not written a number with a unit, start at Section 4.2 and Section 9b.

The interview flow is a procedure for converting an ambiguous request into a defensible set of decisions.
It is not a script of phrases, and it is not a diagram-drawing contest.
Every phase of the flow exists because skipping it produces a specific, predictable class of engineering defect.
Skipping requirements produces a design for the wrong system.
Skipping the write contract produces duplicate charges, lost updates, or an "exactly once" claim nobody can honour.
Skipping sizing produces a diagram whose component count has no relationship to load.
Skipping the latency budget produces a design that passes every individual component check and fails end to end.
Skipping failure policy produces a design that is correct only while all dependencies are healthy.

The whole file is organised around one invariant, stated here and revisited in Section 4.4.
**Invariant (decision-readiness):** every claim in a finished design must be traceable to either a stated constraint, a computed quantity with units, or a named failure policy with an owner.
A sentence that cannot be traced to one of those three is decoration.
Decoration does not survive the follow-up question, which is why interviewers ask "why?" three levels deep.

**Time-box arithmetic.** A conventional loop allots about 45 minutes of discussion.
Let the phases be scope (S), non-functional contract (N), sizing (Z), high-level design (H), deep dive (D), and failure/validation (F).
Section 9a derives the allocation rule `t_i = T * w_i` and works an example where T = 45 min.
The rule matters because unallocated time is spent on whichever phase the candidate finds most comfortable, which is almost never the phase that decides the outcome.

**What a strong answer looks like from the outside.**
It begins by restating the action in one sentence and asking at most three questions that change the design.
It writes down a load estimate before drawing boxes, and it labels the estimate as illustrative.
It names the durable source of truth and the identity of the write before naming caches or queues.
It walks one successful request and one failed request through the same diagram.
It closes with what it would measure first if the design were built.

**What a weak answer looks like.**
It begins with a technology name.
It draws twelve boxes connected by unlabeled arrows.
It uses the word "scalable" as if it were a measurement.
It answers the capacity question with "we'll add more servers."
It answers the consistency question with "we'll use eventual consistency" without saying which read may be stale and for how long.

The difference between those two answers is not knowledge of distributed systems.
Both candidates may know the same facts about replication and partitioning.
The difference is that one of them runs a procedure that forces the numbers and the failure policy onto the page.
This sheet is that procedure, written out with its arithmetic.

**Evidence discipline for this file.**
All workload figures below describe a constructed teaching scenario: a 20-million-daily-active-user product with 25 reads and 2 writes per user per day.
Those inputs are ILLUSTRATIVE.
The arithmetic performed on them is DERIVED and has been recomputed independently.
No figure here is a benchmark of any real service, and no component count should be read as a recommendation for a real system of that shape.
Where a conclusion depends on a distributional assumption — exponential service times, independent shard failures — the assumption is named at the point of use and its removal is discussed in Section 4.4.

**How to use this sheet under time pressure.**
Read Section 2 once for orientation, Section 3 for vocabulary, then keep Section 4 and Section 12 open while practising.
Section 11 is the self-check: if you cannot answer Q-07 and Q-11 without notes, the sizing and failure phases are not yet automatic.
Section 8 is the diagnostic table to consult after a practice run that felt wrong but you cannot say why.

**Navigation within this sheet.**

| Need | Go to |
|---|---|
| Orient before a practice run | [Section 2](#2-concept-map--mental-model) |
| Check a definition mid-conversation | [Section 3](#3-foundational-definitions) |
| Derive or justify a capacity number | [Section 4.1](#41-simplest--base-case), [4.2](#42-general-case) |
| Find the boundary where a formula breaks | [Section 4.4](#44-asymptotic--limiting-behaviour-and-counterexample) |
| Follow one end-to-end calculation | [Section 5](#5-worked-numerical-example) |
| Choose a consistency contract or shard key | [Section 7](#7-comparison--selection-table) |
| Diagnose a symptom after practice | [Section 8](#8-failure-modes--edge-cases) |
| Size compute, storage, bandwidth, cost | [Section 9b](#9b-worked-sizing-compute-storage-bandwidth-cost) |
| Memorise the defensible results | [Section 10](#10-key-theorems--results-reference) |
| Self-test | [Section 11](#11-interview--exam-questions) |
| Scan for thirty seconds | [Section 12](#12-quick-reference-card) |

## 2. Concept Map / Mental Model

The flow is a sequence of narrowing commitments.
Each phase removes a class of designs and adds a class of constraints.
Later phases are allowed to send you back to earlier ones; the arrows below include two feedback edges because a sizing result routinely invalidates a requirement and a failure analysis routinely invalidates a component choice.

```text
[Prompt: "design X"]

        |
        v
[S1 SCOPE]  action -> eligible actors -> success/failure -> out of scope
        |                                        |
        |  cannot name the action                |  scope too wide for T
        v                                        v
[ask; do not design]                     [shrink to one read path
                                          + one write path]
        |
        v
[S2 CONTRACT]  read path, write path, source of truth,
               identity, idempotency key, freshness bound
        |                                        ^
        |  write effect not expressible once     |
        v                                        |
[change the effect model:                     [S3 numbers show
 command log / reconciliation /                  a different
 human approval]                                 bottleneck]
        |
        v
[S3 SIZING]  offered load -> peak -> units -> nodes / shards / bytes
        |
        v
[S4 SHAPE]  ingress -> compute -> durable state -> async -> cache -> obs
        |                                        ^
        |  latency budget overruns               |
        v                                        |
[S5 BUDGET]  per-hop p99 -> fanout tail -> retry amplification
        |
        v
[S6 FAILURE]  timeout, retry, duplicate, stale, partial write,
              overload, dependency outage, migration, rollback
        |
        v
[S7 VALIDATE]  trace one success + one failure; name metrics; name rollback
        |
        v
[Decision record: constraints, numbers, policy, open questions]
```

Read the diagram as a commitment funnel, not a waterfall.
The left column is what you know; the right column is what you do when the phase fails.
Every "fail" branch is a legitimate answer.
An interview answer that never takes a fail branch is either lucky or unexamined.

### 2.1 The four objects the flow produces

The flow produces four artefacts, and the interviewer is checking for all four even when asking about one.
The first is a **requirement set**: one user action, the actors allowed to perform it, the definition of success, and at least two things explicitly excluded.
The second is a **contract**: which component owns which fact, what a write means exactly once, and how stale a read is permitted to be.
The third is a **sized shape**: a component list where each component's count or capacity is justified by a computed quantity.
The fourth is a **policy set**: what happens at timeout, at overload, at dependency loss, and at rollback.

A design missing the fourth artefact is the most common strong-looking failure.
The diagram looks complete, the numbers are plausible, and then the follow-up "what happens when the cache is cold and the database is at 90%?" has no answer.
That question is not a trick; it is the whole point of building the system.

### 2.2 Why requirements come before components

A component choice is only meaningful relative to a requirement it satisfies.
"Add a cache" is not a decision; "reduce origin read load from 17,361 to 1,389 requests per second because origin capacity is 1,500 per second" is a decision.
The first formulation cannot be wrong, which is exactly why it is useless.
The second can be wrong, and therefore can be checked.

The same asymmetry applies to consistency.
"We'll use eventual consistency" is a slogan.
"An author sees their own edit within one second; other viewers may see it stale for up to five seconds; a deleted item must never be served after the deletion commit is acknowledged" is a contract.
Only the second can be tested, and only the second tells you whether you need read-your-writes routing, a version fence, or a tombstone with a grace period.

### 2.3 The peak factor is a modelling decision

Every capacity number in the flow is a rate multiplied by a peak factor.
The peak factor encodes a claim about the arrival process: that busy-hour load is some multiple of the daily mean.
Choosing 3× versus 5× changes the node count roughly proportionally, so the number is not cosmetic.
It should be stated as an assumption with a reason, not asserted as a fact.
Section 4.2 shows how the peak factor enters the server-count formula and Section 4.4 shows what happens when it is underestimated.
A defensible interview sentence is: "I am assuming a 3× busy-hour factor over the daily mean because the workload is consumer-facing with an evening peak; if the true factor were 5× the fleet would need 45 nodes instead of 28."
That sentence contains an assumption, a reason, a number, and the sensitivity result.
It is worth more than any diagram.

### 2.4 The flow's central invariant and its two corollaries

The decision-readiness invariant says every claim traces to a constraint, a computed quantity, or a failure policy.
Its first corollary is that an unlabeled number is worse than no number, because it invites false confidence.
Every quantity in this sheet therefore carries a unit and a label: requests per second, bytes, milliseconds, nodes, currency units.
Its second corollary is that the flow must produce a **stopping condition**.
Without one, sizing expands until the time box expires.
The stopping condition used here is: stop sizing when the dominant bottleneck is identified and its count is within a factor of two of a design decision that would change.
Identifying that a read path needs roughly 28 application nodes is decision-relevant.
Computing whether it needs 27 or 29 is not, and doing so consumes the deep-dive time.

### 2.5 What changes between variants of the flow

The flow's seven phases are stable; their weights are not.
A read-heavy consumer product spends most of its sizing time on egress bandwidth and cache working set.
A write-heavy event pipeline spends it on retention, partition count, and consumer lag.
A workflow with money movement spends it on the write contract, idempotency, and reconciliation.
A system with a hard real-time loop spends it on the tail latency budget and the degradation path.
Section 7 turns those differences into a selection table, and Section 4.3 gives the variant formulas that replace the base capacity formula when arrivals are bursty or when the read path fans out.

**Takeaway:** the interview is a funnel of commitments that ends in four artefacts — requirements, contracts, sized shape, policy — and any phase skipped reappears as a specific defect later.

## 3. Foundational Definitions

Sixteen terms carry the rest of the sheet.
Each is defined so that a reader can decide whether a given sentence is an instance of it or not.
Where a term is routinely confused with a neighbour, the neighbour is named.

| Term | Formal meaning | Units / type | Valid use | Nonexample |
|---|---|---|---|---|
| Interview flow | Ordered set of phases with a time budget and per-phase exit criteria | procedure | structuring a design conversation or a design review | a list of technologies to mention |
| User action | The single observable operation the design must support | verb + object | scoping the whole design | "build a platform" |
| Eligible actor | Principal permitted to invoke the action under a stated rule | principal set | defining authorization and abuse surface | "all users" |
| Success criterion | Observable postcondition that counts the action as complete | predicate | defining what to measure | "fast and reliable" |
| Non-functional requirement | Bound on latency, availability, durability, freshness, cost, or privacy | quantity + unit + percentile + window | constraining the design | an adjective such as "scalable" |
| Offered load | Arrival rate presented to a component before admission control | requests/s | capacity input | accepted load |
| Accepted load | Arrival rate the system admits after shedding or rate limiting | requests/s | overload analysis | offered load |
| Peak factor | Ratio of busy-hour rate to daily mean rate | dimensionless, >1 | converting mean to provisioned capacity | a safety margin on top of peak |
| Service rate | Completions per second by one server when saturated | requests/s per server | node count derivation | latency |
| Utilisation | Arrival rate divided by total service capacity | dimensionless, in [0,1) | queueing and headroom reasoning | a percentage of CPU used, which is a different measurement |
| Headroom | Unused fraction of capacity retained for bursts, failover, and deploys | dimensionless | sizing target | spare hardware in a warehouse |
| Latency budget | Allocation of an end-to-end percentile target across hops | milliseconds, per percentile | design constraint and diagnosis | an average latency figure |
| Fanout | Number of downstream calls required to answer one request | dimensionless, integer >=1 | tail-latency and availability composition | payload size |
| Amplification | Work performed per unit of user-visible work | ratio | retry, replication, and index cost accounting | parallelism |
| Idempotency key | Value that makes repeated delivery of the same command produce one effect | opaque token, unique per logical command | write contract | a database primary key on a different entity |
| Source of truth | Component whose state is authoritative for a fact when copies disagree | component + fact | resolving conflicts and rebuild paths | a cache, even a very large one |
| Consistency contract | Statement of which reads may observe which versions, and for how long | predicate over versions and time | read-path design and testing | "eventual consistency" |
| Reconciliation | Procedure that compares an intended effect log to observed state and repairs divergence | procedure | duplicate, partial-write, and outage recovery | a retry |

**Interview flow.** The procedure in Section 2, not the conversation's mood.
A flow has exit criteria per phase; a conversation has moods.
If you cannot state the exit criterion for the phase you are in, you are not in a phase.

**User action.** One verb applied to one object by one class of actor.
"Post a comment", "reserve a seat", "fetch a profile" are actions.
"Support social features" is a product area, and a design scoped to a product area cannot be sized because it has no single request to count.

**Success criterion.** A predicate evaluated after the action, not during it.
For a payment it is "the ledger contains exactly one debit of the stated amount for this command identifier".
For a feed read it is "the response contains posts from the last N minutes for the followed set, or a documented stale fallback with its age".
The criterion is what the validation phase in Section 5 tests.

**Non-functional requirement.** Any bound that changes the design and is not a functional behaviour.
It must carry a quantity, a unit, a percentile, and a window, because a bare number is not a requirement.
"p99 read latency under 200 ms at 3× mean load, measured over a trailing 5-minute window" is a requirement.
"Low latency" is a wish, and it will be satisfied by any design the author likes.

**Offered versus accepted load.** This distinction is the entire content of overload reasoning.
A queue does not reduce offered load; it relocates when the work happens.
Admission control is the only mechanism that changes accepted load, and it must be deliberate because it converts load into errors.
A design that claims a queue "handles spikes" has confused absorption with resolution.

**Peak factor.** A modelling input, not a safety factor.
Confusing the two produces double counting: an engineer applies 3× for the evening peak and then another 2× "just in case", and the resulting fleet is sized for 6× mean, which is expensive and still not guaranteed.
State the peak factor and the headroom target separately; Section 4.2 multiplies by one and divides by the complement of the other.

**Service rate and utilisation.** Service rate is measured at saturation, so it is a per-server property under a stated request mix.
Utilisation is a ratio of rates, and it is not the same as a CPU gauge.
A server at 40% CPU can be at 90% utilisation of its concurrency limit if requests block on I/O.
When you quote a utilisation number, say which resource it refers to.

**Headroom.** The deliberate gap between provisioned capacity and peak demand.
It pays for three distinct things: arrival burstiness within the peak hour, the loss of one node or zone during failure, and the rollout surge during a deploy.
Because it serves three purposes, a single small number is rarely sufficient; Section 9b works an example where 35% headroom covers the first two and the third is handled by deploy sequencing.

**Latency budget.** A per-percentile allocation across hops, not a single total.
The allocation is only meaningful if the percentile is stated, because p50 budgets and p99 budgets produce different architectures.
Section 4.2 derives the composition rule and Section 4.4 shows the counterexample where adding per-hop p99 values overestimates while multiplying them ignores correlation.

**Fanout.** The count of downstream operations needed for one response.
Fanout is the single most under-modelled quantity in interview designs, because it multiplies both latency tail and failure probability while leaving mean latency almost unchanged.
A request that reads three shards has three chances to be slow, and Section 4.3 quantifies how quickly that erodes a service-level objective.

**Amplification.** Any ratio of internal work to user-visible work.
Retries amplify request rate; replication amplifies write throughput; secondary indexes amplify storage and write cost; fanout amplifies read cost.
Amplification is where the difference between a plausible design and a working one usually lives, because it turns a comfortable utilisation into an overloaded one.

**Idempotency key.** An opaque value, generated once per logical command by the client or the command's origin, that the effect owner records alongside the effect.
It is not a request ID that changes on retry, and it is not a natural key such as an order number if the order can be legitimately modified by several commands.
The distinction matters because the whole duplicate-suppression mechanism depends on the key being stable across retries and unique across commands.

**Source of truth.** The component whose answer wins when copies disagree.
Naming it is a prerequisite for the rebuild path: if the truth is in the log, a lost index is rebuildable; if the truth is in the index, it is not.
Many designs implicitly treat the search index as a source of truth and then discover that they cannot reconstruct it after a mapping change.

**Consistency contract.** A predicate over versions and time, stated per read path.
It must answer three questions: may this read see a version older than the writer's own write, by how much wall-clock or logical time, and what must never be observed (for example, a deleted item).
Section 7 contrasts the contract shapes and their disqualifiers.

**Reconciliation.** A periodic or triggered procedure comparing an intended-effect log with observed state.
Retries reduce the probability of divergence; reconciliation bounds the duration of divergence.
A design that claims exactly-once effects without either an idempotent effect or a reconciliation procedure has claimed a property it cannot verify, and Section 8 treats that as a named failure mode.

**Takeaway:** the vocabulary is small but each term marks a decision boundary; the pairs offered/accepted, peak/headroom, and retry/reconciliation are the ones most often conflated, and each conflation produces a specific design defect.

## 4. Core Formulas with Full Derivation

This section carries the arithmetic of the flow.
Each result is derived from a definition before it is used, and each is followed by the five-part block.
Formula IDs are stable so that later sections can cross-reference without restating the derivation.
The base case assumes Poisson arrivals, exponential service, independent servers, and a stationary rate.
Section 4.2 relaxes stationarity by adding a peak factor and relaxes single-server service by adding a server pool.
Section 4.3 gives the variants that replace the base results when the workload fans out, when writes are replicated, or when the read path is cached.
Section 4.4 collects the boundaries at which the whole family fails.

### 4.1 Simplest / Base Case

**Validity before use.** The results in this subsection apply to a single component, with arrivals that are Poisson at constant rate, service times that are exponential and independent, an infinite waiting room, a work-conserving server, and a steady state that exists.
They do not apply to a batch job, to a component with admission control that rejects work, or to a system whose rate is changing faster than the queue can equilibrate.
Under those conditions the numbers are lower bounds at best and fiction at worst.

**Derivation of the arrival rate.** Take a population of D daily active users, each performing r independent actions per day.
By definition the daily action count is `A = D * r`.
Spreading A uniformly over the 86,400 seconds of a day gives the mean rate `lambda_mean = A / 86400`.
The uniformity assumption is what makes this a *mean* rather than a peak, and it is almost always false in a way that matters, which is why Section 4.2 introduces the peak factor immediately.

```text
FORMULA [SD-01]: lambda_mean = D * r / 86400
WHERE: lambda_mean mean arrival rate [actions/s];
       D daily active users [users];
       r actions per user per day [actions/(user*day)];
       86400 seconds per day [s/day].
CONSTRAINT: r is a daily average and actions are spread uniformly;
            D is the daily-active population, not the registered population;
            one action maps to one unit of work at the component being sized.
NUMERICAL EXAMPLE: D = 20,000,000; r = 25;
                   lambda_mean = 20,000,000 * 25 / 86,400
                               = 500,000,000 / 86,400
                               = 5,787.04 actions/s.
INTERPRETATION: about 5.8 thousand reads per second on average; this is a
                lower bound on provisioned capacity, never a provisioning target.
```

The unit check is worth doing explicitly: `users * actions/(user*day) / (s/day) = actions/s`.
If a candidate instead divides by 24 or by 1,000 the result is wrong by a known factor, and the arithmetic is trivially checkable, which is why interviewers use this question.
A wrong order of magnitude here invalidates every downstream number, so the check is cheap insurance.

**Derivation of the wait time.** For a single server with arrival rate `lambda` and service rate `mu`, stability requires `rho = lambda/mu < 1`.
The number in system for an M/M/1 queue is geometric with parameter `1 - rho`, giving mean `L = rho/(1-rho)`.
By Little's law, `W = L/lambda = 1/(mu - lambda)`.
Because the response time distribution in this model is exponential with that mean, the 99th percentile is `-ln(0.01) * W = 4.6052 * W`.
The exponential shape is a consequence of the memoryless assumption; if service times were deterministic the tail would be far shorter, and Section 4.4 uses that difference as a counterexample.

```text
FORMULA [SD-02]: W = 1/(mu - lambda);  W_p99 = ln(100) * W
WHERE: W mean response time [s]; mu service rate [req/s per server];
       lambda arrival rate at that server [req/s]; ln(100) = 4.6052 [-];
       W_p99 99th percentile response time [s].
CONSTRAINT: M/M/1, rho = lambda/mu < 1, infinite buffer, work-conserving,
            exponential service; percentile form requires the exponential law.
NUMERICAL EXAMPLE: mu = 1,200 req/s; lambda = 598.66 req/s; rho = 0.4989;
                   W = 1/(1200 - 598.66) = 1/601.34 s = 1.663 ms;
                   W_p99 = 4.6052 * 1.663 = 7.66 ms.
INTERPRETATION: at 50% utilisation a single node's mean is 1.7 ms and its p99
                is already 4.6x the mean; the multiplier is a property of the
                distribution, not of the hardware.
```

The ratio `W_p99/W = 4.6052` is independent of utilisation, which is a surprisingly useful fact.
It means that in the M/M/1 family, moving from 50% to 90% utilisation multiplies both mean and p99 by the same factor of five.
It also means that any claim that "p99 is fine because the mean is fine" is a claim about the distribution shape, not about the load.

**Derivation of server count.** Provision `N` identical servers each with service rate `mu`.
Total capacity is `N*mu`, so the utilisation is `lambda/(N*mu)`.
Requiring utilisation at or below a target `rho_max` gives `N >= lambda/(mu*rho_max)`, and integrality forces the ceiling.
The target exists because the queueing penalty is convex in utilisation: from SD-02, moving `rho` from 0.65 to 0.90 multiplies wait by `(1-0.65)/(1-0.90) = 3.5`.
A separate deploy headroom converts active servers into deployed servers, because during a rolling deploy and during a single-node failure the remaining servers must still meet the target.

```text
FORMULA [SD-03]: N_active = ceil(lambda_peak/(mu*rho_max));
                 N_deployed = ceil(N_active/(1-h))
WHERE: N_active servers needed to serve peak at target utilisation [count];
       N_deployed servers to provision including failure/deploy headroom [count];
       lambda_peak peak arrival rate [req/s]; mu per-server service rate [req/s];
       rho_max utilisation target [-]; h headroom fraction [-].
CONSTRAINT: homogeneous servers, independent work, uniform routing;
            rho_max < 1; h covers single-node loss and rollout surge but not
            a correlated zone failure unless stated.
NUMERICAL EXAMPLE: lambda_peak = 17,361.11; mu = 1,200; rho_max = 0.65; h = 0.20;
                   N_active = ceil(17361.11/780) = ceil(22.26) = 23;
                   N_deployed = ceil(23/0.80) = ceil(28.75) = 29.
INTERPRETATION: 23 servers carry peak at 65% utilisation, and 29 must exist so
                that losing one still leaves 28, i.e. above the 23 required.
```

The ceiling operations are not pedantry.
`ceil(22.26) = 23` rather than 22 because 22 servers would put utilisation at `17361.11/(22*1200) = 0.658`, above the target.
An answer of "about 22" is acceptable in an interview if the direction of the rounding is stated; an answer that rounds down and then claims the target is met is a defect.

**Derivation of storage.** If the component durably records `W_day` events per day with mean payload `b` bytes, the raw daily volume is `W_day*b`.
Over a retention window of `d` days the logical volume is `W_day*b*d`.
Physical volume adds replication factor `R` and a non-negative index/metadata overhead fraction `i`.
The product is a floor, not an estimate, because it excludes compaction slack, tombstone residue, and backup copies, which Section 4.4 quantifies.

```text
FORMULA [SD-04]: S_raw = W_day*b;  S_phys = S_raw*d*R*(1+i)
WHERE: S_raw raw bytes per day [bytes/day]; W_day write events per day [events/day];
       b mean payload [bytes/event]; d retention [days];
       R replication factor [-]; i index+metadata overhead fraction [-];
       S_phys physical bytes for the retention window [bytes].
CONSTRAINT: mean payload is representative of the distribution and does not
            hide a heavy tail that dominates the p99 object size;
            payload is stored uncompressed; no compaction slack included.
NUMERICAL EXAMPLE: W_day = 40,000,000; b = 4,096; d = 30; R = 3; i = 0.40;
                   S_raw = 40,000,000*4,096 = 163,840,000,000 B/day
                         = 163.84 GB/day = 152.59 GiB/day;
                   S_phys = 163.84e9*30*3*1.40 = 2.0644e13 B
                          = 20.64 TB = 18.78 TiB.
INTERPRETATION: a 30-day retention window costs about 20.6 TB of physical
                storage, which sets the shard count and the backup budget.
```

**Derivation of egress bandwidth.** Each read returns `s` bytes of payload.
Multiplying the peak read rate by the payload gives bytes per second; multiplying by 8 gives bits per second, which is the unit that network links are sold in.
Confusing bytes and bits is the classic factor-of-eight error and it is detectable by unit analysis alone.

```text
FORMULA [SD-05]: B = lambda_peak * s;  B_bits = 8*B
WHERE: B egress bytes per second [B/s]; lambda_peak read rate [reads/s];
       s mean response payload [bytes/read]; B_bits link capacity [bit/s].
CONSTRAINT: s is the mean payload actually crossing the network boundary,
            after compression if compression is enabled end to end;
            protocol overhead, TLS records, and retransmissions excluded.
NUMERICAL EXAMPLE: lambda_peak = 17,361.11; s = 12 KiB = 12,288 B;
                   B = 17,361.11*12,288 = 2.1333e8 B/s = 203.5 MiB/s;
                   B_bits = 8*2.1333e8 = 1.707e9 bit/s = 1.71 Gbit/s.
INTERPRETATION: the read path needs about 1.7 Gbit/s of sustained egress at
                peak, so a single 10 Gbit/s link is plausible but leaves little
                room for retransmission and cross-zone replication traffic.
```

**Derivation of Little's law and its use.** For any stable system, the long-run average number in the system equals the arrival rate times the average time in the system.
The identity follows from counting: each arriving item contributes one unit of occupancy for its residence time, so total occupancy-time per unit time equals rate times mean residence.
It requires only stability and consistent definitions of the boundary, which is why it is the most transferable formula in the sheet.

```text
FORMULA [SD-06]: L = lambda * W
WHERE: L mean items in system [items]; lambda arrival rate [items/s];
       W mean residence time [s].
CONSTRAINT: stable system (no unbounded growth), stationary rates, and the
            boundary of "system" identical on both sides of the equation.
NUMERICAL EXAMPLE: a connection pool serving 17,361.11 req/s with 30 ms mean
                   residence must hold L = 17361.11*0.030 = 520.8 in-flight
                   requests; a pool of 512 is therefore already binding.
INTERPRETATION: Little's law converts a latency target into a concurrency
                requirement, and a concurrency limit that is below L will queue
                or reject regardless of how fast the backend is.
```

That example is the reason connection-pool sizing belongs in the latency phase rather than the capacity phase.
A backend that can answer in 30 ms still fails if the front end permits only 256 concurrent calls.
The formula also runs backwards usefully: given a concurrency limit of 512 and a measured rate of 17,361 req/s, the implied mean residence is 29.5 ms, so a latency regression is arithmetically detectable from pool occupancy and rate alone.

### 4.2 General Case

**What changes.** Three things: arrivals are no longer stationary, so a peak factor enters; service is provided by a pool with a shared queue rather than one server, so the Erlang C result replaces M/M/1; and the end-to-end path consists of multiple hops, so latency composes.

**Derivation of the peak rate.** Let `P` be the ratio of the busy-hour rate to the daily mean.
Then `lambda_peak = P * lambda_mean` by definition of P.
The peak factor is an assumption about the arrival process, and its justification is usually a measured diurnal profile, which an interview candidate does not have.
Stating a plausible value and its sensitivity is the honest move; Section 4.4 shows the sensitivity result.

```text
FORMULA [SD-07]: lambda_peak = P * lambda_mean = P*D*r/86400
WHERE: P busy-hour peak factor over the daily mean [-], P > 1;
       other symbols as SD-01.
CONSTRAINT: the busy hour is the sizing interval; a shorter burst (a
            one-minute spike) may exceed this estimate and needs its own
            measurement or a burst-absorption mechanism.
NUMERICAL EXAMPLE: P = 3; lambda_peak = 3*5,787.04 = 17,361.11 req/s.
INTERPRETATION: the provisioned fleet is sized for roughly triple the daily
                average, which is a modelling assumption to be stated aloud.
```

**Derivation of the pooled-server wait (Erlang C).** With `m` servers, arrivals Poisson at `lambda`, and exponential service at `mu` each, define offered load `a = lambda/mu` (dimensionless erlangs) and `rho = a/m`.
The probability that an arrival must wait is the Erlang C function `C(m,a)`.
It is computed stably by first obtaining the Erlang B blocking probability from the recursion `B_0 = 1`, `B_k = a*B_{k-1}/(k + a*B_{k-1})`, then using `C = B_m/(1 - rho*(1 - B_m))`.
Conditional on waiting, the mean wait is `1/(m*mu - lambda)`, so the unconditional mean queueing delay is `W_q = C/(m*mu - lambda)`.
The recursion is the practical form because the closed form with factorials overflows, and the relation between B and C follows from the birth-death balance equations.

```text
FORMULA [SD-08]: a = lambda/mu; rho = a/m;
                 B_k = a*B_{k-1}/(k + a*B_{k-1}), B_0 = 1;
                 C = B_m/(1 - rho*(1-B_m));
                 W_q = C/(m*mu - lambda)
WHERE: m servers [count]; a offered load [erlangs];
       rho per-server utilisation [-]; C probability of queueing [-];
       W_q mean queueing delay [s]; lambda, mu as before.
CONSTRAINT: M/M/m with a single shared queue, infinite buffer, homogeneous
            servers, no priority classes; requires a < m.
NUMERICAL EXAMPLE: a node has m = 6 workers, mu = 200 req/s each;
                   lambda = 598.66 req/s; a = 2.993; rho = 0.499;
                   B_6 = 0.0518 (recursion); C = 0.0983;
                   W_q = 0.0983/(1200 - 598.66) = 1.63e-4 s = 0.163 ms.
INTERPRETATION: with six workers only 9.8% of requests queue at all, and the
                mean queueing component is a fraction of a millisecond; at
                four workers the same load gives C = 0.507 and a 2.5 ms mean
                queueing delay, a fifteenfold difference from one worker fewer.
```

The four-worker row in that interpretation is the useful teaching point, and it is recomputed in the table below.
Adding workers from four to six does not change the service rate per worker; it changes the probability that a request finds all workers busy.
In an interview, quoting the probability of queueing is often more persuasive than quoting the mean delay, because it explains why the p99 moves.

| Workers m | Offered load a | Utilisation rho | P(queue) C | Mean queue delay |
|---:|---:|---:|---:|---:|
| 4 | 2.993 | 0.748 | 0.507 | 2.52 ms |
| 6 | 2.993 | 0.499 | 0.098 | 0.16 ms |
| 8 | 2.993 | 0.374 | 0.013 | 0.013 ms |

**Derivation of end-to-end mean latency.** For a serial chain of hops, the mean of the sum is the sum of the means, with no independence requirement, because expectation is linear.
That is the only composition rule in this sheet that is exact without distributional assumptions.
The corresponding statement for percentiles is false in general and Section 4.4 gives both directions of the error.

```text
FORMULA [SD-09]: E[T_total] = sum_i E[T_i]
WHERE: E[T_i] mean latency of hop i [ms]; E[T_total] mean end-to-end [ms].
CONSTRAINT: serial dependency only; no parallelism; means exist and are finite.
            Exact for means by linearity of expectation; NOT valid for
            percentiles.
NUMERICAL EXAMPLE: hops of 15, 40, 60, 25, 20, 40 ms;
                   E[T_total] = 200 ms, leaving 100 ms of a 300 ms target.
INTERPRETATION: mean budgeting is easy and exact; it also hides the tail, so it
                is necessary but never sufficient.
```

**Derivation of the retry amplification factor.** Suppose each attempt fails independently with probability `f`, and the caller makes at most `R` attempts total.
The number of attempts is a truncated geometric variable.
Its expectation is `sum_{k=0}^{R-1} f^k = (1 - f^R)/(1 - f)`, since the probability of reaching attempt `k+1` is `f^k`.
That ratio is the attempt amplification per call.
When retries occur at several layers, the amplifications multiply, which is why layered retries are dangerous even when each layer's budget looks modest.

```text
FORMULA [SD-10]: E[attempts] = (1 - f^R)/(1 - f);
                 E_system = prod_j E_j
WHERE: f per-attempt failure probability [-]; R maximum attempts per layer [-];
       E_j amplification at layer j [-]; E_system total attempts per
       user-visible request [-].
CONSTRAINT: independent failures per attempt, fixed R per layer, retries
            issued immediately or with delays that do not change f;
            no retry budget or circuit breaker truncating the process.
NUMERICAL EXAMPLE: f = 0.20, R = 3:
                   E = (1 - 0.008)/0.80 = 1.24 attempts per call.
                   Three nested layers: E_system = 1.24^3 = 1.907.
INTERPRETATION: a 20% failure rate turns one logical request into almost two
                physical requests across three layers, which nearly doubles
                load on the failing dependency at the moment it can least
                afford it.
```

That result is the arithmetic behind the "retry storm" failure mode in Section 8.
Note that increasing `R` from 3 to 5 at `f = 0.2` changes E from 1.24 to 1.2496, an increase of under 1%.
The marginal benefit of extra attempts collapses geometrically while the marginal cost in load at the dependency does not, which is the quantitative case for bounded retries with jitter and a retry budget rather than "retry a few times."

**Derivation of fanout tail and availability.** If one request requires `k` independent downstream calls and succeeds only if all succeed, then with per-call success probability `q` the request success probability is `q^k`.
Inverting, to achieve an overall success probability `q` you need per-call `q^(1/k)`.
The same algebra applies to latency percentiles when the calls are independent: to have 99% of requests complete under a threshold, each of `k` independent calls must complete under it with probability `0.99^(1/k)`.
Independence is the load-bearing assumption and it is usually optimistic, because a shared dependency, a shared host, or a shared network path induces correlation.

```text
FORMULA [SD-11]: q_call = q_req^(1/k);  A_serial = prod_i A_i
WHERE: k fanout count [count]; q_req end-to-end success probability [-];
       q_call required per-call probability [-]; A_i availability of
       component i [-]; A_serial end-to-end availability [-].
CONSTRAINT: independent downstream outcomes; serial dependency (all must
            succeed); correlated failures make the required per-call value
            stricter than this formula suggests.
NUMERICAL EXAMPLE: k = 12, target q_req = 0.99:
                   q_call = 0.99^(1/12) = 0.999163, i.e. each shard may fail
                   8.4e-4 of the time, not 1e-2.
                   Availability: five serial components at A = 0.9995 give
                   A_serial = 0.9995^5 = 0.997503,
                   i.e. 1,314 minutes of unavailability per 365-day year.
INTERPRETATION: a 12-way fanout demands four nines from every shard merely to
                deliver two nines end to end, which is why fanout reduction
                and hedged requests appear in almost every read-path design.
```

The availability arithmetic uses 525,960 minutes per year (365 * 24 * 60), so `(1-0.9995^5)*525960 = 1,313.6` minutes.
That is about 21.9 hours, and it is larger than most engineers expect before they compute it.
The reason is that five components each of which is down 0.05% of the time are, in the independent model, down simultaneously for a nonzero fraction that compounds.

### 4.3 Key Special Cases / Variants

| Variant | What changes | Assumptions | Use when | Avoid when | Equation ID |
|---|---|---|---|---|---|
| Single-server M/M/1 | One work-conserving server, exponential service | Poisson arrivals, rho < 1 | teaching, single-threaded component, worst-case reference | multi-core pools, deterministic service, priority classes | SD-02 |
| Pooled Erlang C | `m` homogeneous workers with one queue | Poisson arrivals, exponential service, no abandonment | sizing worker pools and thread counts | batch queues, deadline-driven schedulers | SD-08 |
| Cached read path | Origin sees `(1-h)` of the rate | Hit ratio `h` stable over the sizing window | read-heavy content with reuse | per-user data with low reuse, write-then-read workloads | SD-04, SD-12 |
| Replicated write path | Write work multiplies by `R` and by index count | Quorum or synchronous replication | durability requirements above one node | write-heavy with cross-region latency budget | SD-04, SD-13 |
| Fanout read path | Latency and availability compose across `k` calls | Independent outcomes | aggregate views over partitions | hard single-partition reads | SD-11 |
| Bursty arrivals | Peak factor plus queue drain | Rate step change with finite surplus | launches, scheduled jobs, retry storms | smooth diurnal load | SD-07, SD-14 |

**Cached read path.** With hit ratio `h`, the origin rate is `lambda_origin = lambda_peak*(1-h)`.
The formula is trivial; its validity is not.
It requires that the hit ratio is measured on the same key distribution and the same time window as the load, that the cache is warm, and that hits are not themselves rate limited by cache throughput.
Section 4.4 shows the cold-start counterexample where `h = 0` for the first minutes.

```text
FORMULA [SD-12]: lambda_origin = lambda_peak*(1-h);  B_origin = lambda_origin*s
WHERE: h hit ratio [-] in [0,1]; lambda_origin origin request rate [req/s];
       B_origin origin egress [B/s]; s payload [B/read].
CONSTRAINT: h is measured on the same population and window; cache capacity
            covers the working set; cache itself is not the bottleneck.
NUMERICAL EXAMPLE: lambda_peak = 17,361.11; h = 0.92;
                   lambda_origin = 17361.11*0.08 = 1,388.89 req/s;
                   B_origin = 1388.89*12,288 = 1.7067e7 B/s = 16.3 MiB/s.
INTERPRETATION: the cache removes 92% of origin load, a 12.5x reduction, but
                the remaining 1,389 req/s still require a sized origin fleet.
```

**Replicated write path.** Each logical write becomes `R` physical writes, and each secondary index adds another physical write per replica in most storage engines.
The amplification factor is `R*(1 + n_index)`, so the peak write rate at the storage layer is `lambda_write_peak*R*(1+n_index)`.
This is the number to compare against per-node write capacity, not the logical rate.

```text
FORMULA [SD-13]: lambda_storage = lambda_write_peak*R*(1+n_index)
WHERE: lambda_write_peak logical write rate at peak [writes/s];
       R replication factor [-]; n_index secondary index count [-];
       lambda_storage physical writes at the storage layer [writes/s].
CONSTRAINT: synchronous replication, each index maintained on write,
            no write coalescing or batching across replicas.
NUMERICAL EXAMPLE: lambda_write_peak = 1,388.89; R = 3; n_index = 1;
                   lambda_storage = 1388.89*3*2 = 8,333.3 writes/s.
INTERPRETATION: a logical write rate of 1.4 thousand per second becomes 8.3
                thousand physical writes per second, so write capacity must be
                sized on the amplified figure.
```

**Bursty arrivals and queue drain.** When the arrival rate steps from `lambda` to `lambda_o > mu` for a duration `t_o`, backlog grows at `lambda_o - mu` and reaches `B = (lambda_o - mu)*t_o`.
After the step ends, if service rate `mu_r` exceeds the recovery arrival rate `lambda_r`, the backlog drains in `T = B/(mu_r - lambda_r)`.
The second formula is the one candidates forget: absorbing a burst is only useful if the system can subsequently drain it before the next burst.

```text
FORMULA [SD-14]: B = (lambda_o - mu)*t_o;  T = B/(mu_r - lambda_r)
WHERE: B backlog accumulated [requests]; lambda_o overload arrival rate
       [req/s]; mu service capacity [req/s]; t_o overload duration [s];
       mu_r recovery service rate [req/s]; lambda_r recovery arrival rate
       [req/s]; T drain time [s].
CONSTRAINT: fluid approximation, constant rates within each interval,
            initially empty queue, no losses, and mu_r > lambda_r.
NUMERICAL EXAMPLE: lambda_o = 34,722.2; mu = 30,000; t_o = 300;
                   B = (34722.2-30000)*300 = 1,416,667 requests;
                   mu_r = 30,000; lambda_r = 25,000;
                   T = 1,416,667/5,000 = 283.3 s.
INTERPRETATION: a five-minute overload episode leaves a backlog that needs
                almost five more minutes of surplus capacity to clear; if
                bursts arrive more often than every 283 s the queue diverges.
```

The fluid approximation ignores the queueing tail, so it understates the latency experienced during the episode.
It is still the right first-order tool because it answers the question the interviewer is asking: does this system return to steady state, and how long does it take.

**Cost per request.** For `N` servers at monthly cost `c` each and `R_month` requests per month, the infrastructure cost per million requests is `N*c*1e6/R_month`.
The formula is accounting, not engineering, but it forces the candidate to attach a price to a design choice, which is the point of a cost question.

```text
FORMULA [SD-15]: C_per_M = N*c*1e6/R_month
WHERE: C_per_M cost per million requests [currency]; N server count [count];
       c monthly cost per server [currency/(server*month)];
       R_month requests per month [requests/month].
CONSTRAINT: c includes only compute for the fleet; storage, egress, and
            managed-service fees are separate line items and must be added.
NUMERICAL EXAMPLE: N = 29; c = 450; R_month = 5787.04*86400*30 = 1.500e10;
                   C_per_M = 29*450*1e6/1.500e10 = 13,050/15,000 = 0.87.
INTERPRETATION: compute costs well under one currency unit per million
                requests, so at this scale the design should optimise for
                correctness and operability rather than for compute price.
```

**Time allocation.** Let `T` be the available discussion time and `w_i` the intended share of phase `i`, with `sum_i w_i = 1`.
Then `t_i = T*w_i`.
The equation is trivial; its purpose is to make the allocation explicit and checkable, because the common failure is not a bad allocation but an implicit one.

```text
FORMULA [SD-16]: t_i = T*w_i;  sum_i w_i = 1
WHERE: t_i minutes for phase i [min]; T total time [min]; w_i share [-].
CONSTRAINT: phases are the seven in Section 2; w_i > 0 for scope, contract,
            sizing, shape, failure, and validation.
NUMERICAL EXAMPLE: T = 45; w = (0.11, 0.13, 0.18, 0.20, 0.20, 0.11, 0.07);
                   t = (5.0, 6.0, 8.0, 9.0, 9.0, 5.0, 3.0) minutes, sum 45.
INTERPRETATION: sizing and deep dive get nine minutes each; if you have spent
                20 minutes on the diagram you have already overspent shape.
```

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

| Result/invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| SD-02 p99 multiplier 4.605 | Exponential service | Measure the service-time distribution; compare p99/mean | Deterministic service gives p99 close to the mean | Use an empirical quantile or a bound for the actual distribution |
| SD-03 node count linear in peak | Homogeneous servers, uniform load | Per-node rate distribution; max/mean ratio | One hot key sends 5% of traffic to one node | Split hot keys or isolate the hot partition |
| SD-04 storage with 1.4x index | No compaction slack or tombstones | Compare logical bytes to physical bytes used | LSM store with 30% garbage inflates 1.4x to about 2x | Measure physical amplification; add it explicitly |
| SD-09 mean latencies add | Serial hops only | Identify parallel branches; count critical path | Fanout of 12 reduces in parallel, not serial | Compose along the critical path only |
| SD-10 retry amplification | Independent failures, no budget | Count attempts at the dependency during an incident | Circuit breaker tripped, amplification dropped to 1 | Model the breaker state explicitly |
| SD-11 fanout tail | Independent downstream outcomes | Correlate errors across shards | Shared network partition fails all shards at once | Assume full correlation in the worst case |
| SD-12 cache hit ratio | Stable working set, warm cache | Hit ratio during deploy and after a flush | Cold start with h = 0 for 90 s | Size origin for the cold-start rate or protect it with admission control |
| SD-14 queue drain | `mu_r > lambda_r` | Compare recovery capacity to residual arrivals | Recovery arrivals equal capacity | Backlog never drains; shed or add capacity |

**The percentile-composition counterexample.** Adding per-hop p99 values and calling the sum the end-to-end p99 is wrong in both directions.
Suppose two serial hops, each with median 10 ms and p99 100 ms.
If the slow events are perfectly correlated — both hops are slow together — the end-to-end p99 is about 200 ms, and the sum of p99 values happens to be right.
If the slow events are independent, the probability that both exceed 100 ms is `0.01*0.01 = 1e-4`, so the end-to-end p99 is well below 200 ms; summing overestimates.
If the hops are anti-correlated, summing can underestimate.
The mean adds exactly; percentiles require the joint distribution, which a candidate almost never has.
The defensible interview move is to budget on the critical path with explicit per-hop targets and then say which correlation assumption you are making.

**The utilisation asymptote.** From SD-02, `W = 1/(mu - lambda) = (1/mu)/(1-rho)`.
As `rho -> 1`, `W -> infinity`, and the growth is hyperbolic rather than linear.
Quantitatively, a change from `rho = 0.90` to `rho = 0.95` doubles the wait, as the table below shows.
This is why "we are at 90% utilisation, we have room" is a claim that requires a specific tail objective, not a general reassurance.

| Utilisation rho | Mean wait multiplier 1/(1-rho) | p99 for exponential service, s = 0.5 ms |
|---:|---:|---:|
| 0.50 | 2.0 | 4.61 ms |
| 0.65 | 2.86 | 6.58 ms |
| 0.80 | 5.0 | 11.51 ms |
| 0.90 | 10.0 | 23.03 ms |
| 0.95 | 20.0 | 46.05 ms |

**The cold-cache counterexample.** SD-12 is correct but conditional on a warm cache with a measured `h`.
After a deploy that changes the value format, a cache node loss, or a regional failover, `h` starts at zero and rises.
During that window the origin sees `lambda_peak`, not `lambda_peak*(1-h)`.
For the worked numbers, that is 17,361 req/s instead of 1,389 req/s, a 12.5x increase against a fleet sized for the lower figure.
The correct design response is either to size the origin for cold start while protecting it with a concurrency limit that sheds load, or to warm the cache before shifting traffic, and to say which you chose.
Section 8 lists this as failure F-03 with the discriminating check.

**The peak-factor sensitivity.** Because `N` is roughly linear in `P`, an underestimated peak factor is not a rounding error.
The table below recomputes the node count for a range of peak factors at the same daily mean and the same utilisation target.

| Peak factor P | Peak rate req/s | Active nodes | Deployed nodes | Utilisation achieved |
|---:|---:|---:|---:|---:|
| 2 | 11,574 | 15 | 19 | 0.508 |
| 3 | 17,361 | 23 | 29 | 0.499 |
| 4 | 23,148 | 30 | 38 | 0.508 |
| 5 | 28,935 | 38 | 48 | 0.502 |
| 6 | 34,722 | 45 | 57 | 0.508 |

A candidate who states P = 3 without comment and a candidate who states P = 3 and then says "at P = 5 this design needs 48 deployed nodes rather than 29, so the peak assumption is the largest single source of sizing risk" have produced the same design and very different answers.
The second has demonstrated that they know which assumption is load-bearing.

**The "exactly once" limit.** No arrangement of retries provides exactly-once delivery across an unreliable channel.
The finite version is: exactly-once *effects* are achievable when the effect owner records the idempotency key within the same atomic operation that applies the effect.
If the record and the effect are in different systems, a crash between them produces either a duplicate or a lost effect, and the resolution is a reconciliation procedure over a durable log.
This is a limit, not an approximation: the result is independent of how many retries you permit.
The correct interview sentence names the effect owner, the atomic boundary, and the reconciliation procedure, and declines to claim more.

**Takeaway:** the base formulas are exact within narrow models, and the value of Section 4 lies as much in its boundary table as in its derivations — every result here has a named condition whose violation changes the recommendation.

## 5. Worked Numerical Example

**Example title and evidence label.** "Profile read path for a 20M-DAU consumer product" — ILLUSTRATIVE workload, DERIVED arithmetic, independently recomputed.
No part of this is a measurement of any real service.

**Workflow and decision.** A user opens a profile screen.
The design must decide: the number of read-path application servers, whether a cache is required and how much memory it needs, the number of storage shards for a 30-day hot window, the per-hop latency budget at p99, and whether the write path can tolerate three replicas at peak.
The decision consequence is a sized component list with a stated bottleneck.

**Input data.** Every row is a constructed teaching input.

| Input | Symbol | Value | Unit | Provenance |
|---|---|---|---|---|
| Daily active users | D | 20,000,000 | users | ILLUSTRATIVE |
| Reads per user per day | r_read | 25 | reads/(user·day) | ILLUSTRATIVE |
| Writes per user per day | r_write | 2 | writes/(user·day) | ILLUSTRATIVE |
| Busy-hour peak factor | P | 3 | dimensionless | ILLUSTRATIVE assumption |
| Mean read payload | s | 12 | KiB/read | ILLUSTRATIVE |
| Mean write payload | b | 4 | KiB/write | ILLUSTRATIVE |
| Per-server read service rate | mu | 1,200 | req/s | ILLUSTRATIVE |
| Utilisation target | rho_max | 0.65 | dimensionless | design choice |
| Failure/deploy headroom | h | 0.20 | dimensionless | design choice |
| Cache hit ratio (warm) | h_cache | 0.92 | dimensionless | ILLUSTRATIVE |
| Replication factor | R | 3 | copies | design choice |
| Secondary index count | n_idx | 1 | indexes | design choice |
| Index + metadata overhead | i | 0.40 | fraction | ILLUSTRATIVE |
| Retention window | d | 30 | days | product requirement |
| End-to-end p99 target | — | 300 | ms | product requirement |
| Distinct profile objects | O | 60,000,000 | objects | ILLUSTRATIVE |

**Valid assumptions and what is intentionally simplified.**
Arrivals are treated as Poisson at a stationary peak rate, service times as exponential, and servers as homogeneous.
The cache hit ratio is assumed stable across the peak hour instead of varying with the diurnal pattern.
Replication is synchronous and each secondary index is maintained on the write path.
The model omits TLS handshake cost, connection setup, cross-zone replication traffic, and background compaction.
These omissions make the estimate a floor for bandwidth and a rough central estimate for compute.

### Step 1 — Mean and peak read rate

```text
lambda_mean_read = D*r_read/86400
                 = 20,000,000*25/86,400
                 = 500,000,000/86,400
                 = 5,787.037 req/s

lambda_peak_read = P*lambda_mean_read
                 = 3*5,787.037
                 = 17,361.111 req/s
```

### Step 2 — Peak write rate and storage amplification

```text
lambda_mean_write = 20,000,000*2/86,400 = 40,000,000/86,400 = 462.963 req/s
lambda_peak_write = 3*462.963 = 1,388.889 req/s

writes_per_day      = 20,000,000*2 = 40,000,000 writes/day
daily_bytes         = 40,000,000*4,096 = 163,840,000,000 B
                    = 163.84 GB/day = 152.5879 GiB/day
30_day_raw_bytes    = 163,840,000,000*30 = 4,915,200,000,000 B
                    = 4.9152 TB = 4.4704 TiB
30_day_physical     = 4.9152e12*3*1.40 = 2.0644e13 B
                    = 20.644 TB = 18.775 TiB
storage_write_rate  = 1,388.889*3*2 = 8,333.334 writes/s
```

### Step 3 — Read-path server count

```text
N_active   = ceil(lambda_peak_read/(mu*rho_max))
           = ceil(17,361.111/(1,200*0.65))
           = ceil(17,361.111/780)
           = ceil(22.2578) = 23 servers

N_deployed = ceil(N_active/(1-h))
           = ceil(23/0.80)
           = ceil(28.75) = 29 servers

utilisation achieved = 17,361.111/(29*1,200)
                     = 17,361.111/34,800
                     = 0.4989
```

The achieved utilisation is 0.4989 rather than 0.65 because the headroom step over-provisions slightly.
That is the correct direction: the fleet serves peak at 50% utilisation, leaving room for the loss of a node and for the rollout surge.
The per-node rate is `17,361.111/29 = 598.66 req/s`, which is the figure used in the Section 4.2 Erlang C block.

A candidate may reasonably ask whether 29 nodes at 50% utilisation is wasteful.
The answer depends on the cost of a missed latency objective, and Section 9b attaches a number: the compute bill is about 0.87 currency units per million requests, so the difference between 23 and 29 nodes is a small fraction of that.
At larger scale the same arithmetic can invert, and Section 7's selection guide says to revisit the headroom target when compute cost dominates.

### Step 4 — Per-node wait and tail

```text
per-node rate       lam = 17,361.111/29 = 598.66 req/s
per-node wait       W   = 1/(1,200 - 598.66) s
                        = 1/601.34 s = 1.6629e-3 s = 1.663 ms
p99 (exponential)   W99 = 4.6052*1.663 ms = 7.658 ms
mean in system      L   = lam*W = 598.66*1.6629e-3 = 0.9955 requests
```

An in-system occupancy of about one request per node looks almost empty, and it is.
That is what a 50% utilisation target buys: the node is never more than a request deep on average, so the queueing component of latency is negligible and the remaining latency is dominated by downstream hops rather than by the node's own scheduling.

The same computation at the target utilisation of 0.65 rather than the achieved 0.4989 gives `W = 1/(1200-780) = 2.381 ms` and `W99 = 10.96 ms`.
The difference between those two tails is the concrete value of the headroom, which is a useful sentence for the interview.

### Step 5 — Egress and origin load with cache

```text
egress peak   = 17,361.111*12,288 B = 2.1333e8 B/s = 203.5 MiB/s
                                    = 1.707 Gbit/s
origin rate   = 17,361.111*(1-0.92) = 1,388.889 req/s
origin egress = 1,388.889*12,288 B = 1.7067e7 B/s = 16.3 MiB/s

cache memory for the hot set:
  hot fraction 20% of 60,000,000 objects = 12,000,000 objects
  bytes/object  = 12,288 B + 200 B overhead = 12,488 B
  hot set      = 12,000,000*12,488 = 1.49856e11 B = 149.9 GB
  full set     = 60,000,000*12,488 = 7.4928e11 B = 749.3 GB
```

The cache decision is now numeric rather than stylistic.
Serving 92% of reads from cache requires about 150 GB of memory for the hot set, and if the whole object set must be resident the requirement is about 749 GB.
That distinction — hot-set residency versus full-set residency — is the difference between a plausible cache tier and an unbounded memory bill, and it must be stated.

### Step 6 — Cold-start stress case

```text
cold cache:      h = 0 for the warming window
origin rate   = 17,361.111 req/s  (12.5x the warm-cache 1,388.889)
origin egress = 17,361.111*12,288 = 2.1333e8 B/s = 203.5 MiB/s
```

Against an origin fleet sized from Section 4.1's `N` formula at 1,389 req/s, this is fatal: `ceil(1388.889/780) = 2` nodes would be provisioned, and the cold-start rate would overload them by a factor of about twelve.
The design response is to size the origin with a concurrency limit set for the warm case and to admit cold-start traffic gradually, or to warm the cache before shifting traffic.
Either way the interview answer must contain the number, because the number is what makes the hazard visible.

### Step 7 — Fanout latency budget

Assume the read path fans out to 3 storage shards in parallel and the response requires all three.
The end-to-end target is 300 ms at p99.

```text
hop                     mean      p99 budget
client -> edge           5 ms      20 ms
edge -> app (routing)   10 ms      35 ms
app -> cache (hit path) 15 ms      40 ms
app -> origin (miss)    25 ms      90 ms
app -> 3 shards parallel 25 ms     90 ms
response serialization   8 ms      25 ms
                                              sum of p99 = 300 ms
```

The sums are 20+35+40+90+90+25 = 300 ms exactly, which leaves no margin and therefore fails the budget on any real day.
A defensible budget reserves at least 30% for the unexpected, so the p99 allocations should sum to no more than about 210 ms.
Recomputing with that constraint:

```text
allocations: 15, 25, 30, 60, 60, 20  ->  sum 210 ms, headroom 90 ms
```

The cache-hit path and the origin-miss path are alternatives, not a serial chain, so the budget must be taken on the worst of the two, which is the miss path.
Treating them as serial double-counts latency and produces an over-constrained budget; treating them as parallel without noting that the miss path is the slow one produces an optimistic one.
This is exactly the kind of modelling error Section 8 failure F-05 describes.

### Step 8 — Fanout tail consequence

```text
required per-shard success for overall 0.99 with k = 3:
  q_call = 0.99^(1/3) = 0.996655  (tail 3.34e-3 per shard)
serial availability if each shard is up 0.9995 of the time:
  A = 0.9995^3 = 0.998501
  downtime = (1-0.998501)*525,960 min/yr = 788.5 min/yr
```

A 3-way fanout against shards with five nines of availability delivers less than three nines end to end.
That is the arithmetic behind the claim in Section 2.4 that fanout is the most under-modelled quantity in the flow.

### Independent checks

The following checks were each recomputed from the inputs rather than from the intermediate results.

| Check | Method | Result |
|---|---|---|
| Rate conversion | 500,000,000/86,400 by long division | 5,787.037, consistent |
| Peak multiplication | 5,787.037*3 | 17,361.111, consistent |
| Byte arithmetic | 40,000,000*4,096 | 163,840,000,000 exactly |
| TB conversion | 4.9152e12/1e12 | 4.9152 TB |
| TiB conversion | 4.9152e12/2^40 | 4.4704 TiB |
| Node ceiling | 17,361.111/780 | 22.2578, ceiling 23 |
| Headroom ceiling | 23/0.80 | 28.75, ceiling 29 |
| Achieved utilisation | 17,361.111/34,800 | 0.49888 |
| Little's law | 598.66*1.6629e-3 | 0.9955 requests |
| Bandwidth | 17,361.111*12,288 | 2.1333e8 B/s |
| Bit conversion | 8*2.1333e8 | 1.7067e9 bit/s |
| Cache memory | 12,000,000*12,488 | 1.49856e11 B = 149.9 GB |
| Cold-start ratio | 17,361.111/1,388.889 | 12.5 exactly |
| Availability | 0.9995^3 | 0.9985010 |

The ratio in the cold-start check is exactly 12.5 because `1/(1-0.92) = 12.5`; that identity is a useful sanity check whenever a hit-ratio claim is quoted.
If a candidate reports that a 92% hit ratio gives a "twelve times reduction", they are right; if they report "an order of magnitude", they are approximately right; if they report "a 92% reduction in cost", they have conflated request reduction with cost, since the origin fleet is not the only cost line.

### Wrong path

A plausible mistake is to compute the server count from the mean rate and then apply the peak factor to the node count as a rounding step:

```text
wrong:  N = ceil(5,787.037/780) = ceil(7.419) = 8 nodes
        then "add 3x for peak"     -> 24 nodes
correct: N = ceil(3*5,787.037/780) = 23 active, 29 deployed
```

The wrong path happens to land near the right answer here, which makes it dangerous: it teaches a procedure that is right by accident and wrong by up to `P-1` nodes in general.
The two quantities are `ceil(P*x)` and `P*ceil(x)`, and since `ceil` is superadditive only trivially, the wrong form is never smaller.
The table below recomputes both across a plausible range of daily means.

| Mean rate | Raw ratio x | Correct `ceil(3x)` | Wrong `3*ceil(x)` | Error |
|---:|---:|---:|---:|---:|
| 5,467.8 | 7.010 | 22 | 24 | +2 |
| 5,725.2 | 7.340 | 23 | 24 | +1 |
| 5,787.0 | 7.419 | 23 | 24 | +1 |
| 5,850.0 | 7.500 | 23 | 24 | +1 |
| 6,162.0 | 7.900 | 24 | 24 | 0 |

The wrong path overcounts in every row and by two nodes in the worst case, and it can never undercount, because rounding up before multiplying adds a whole server before the peak factor is applied.
An overcount of two nodes is not a disaster; the real hazard is that the same instinct, applied to the headroom step, can undercount.
Applying headroom before the utilisation target gives `ceil(17,361.11/780/0.8) = ceil(27.82) = 28` rather than 29, because headroom must act on the integer active count: servers are not divisible, and `ceil(23/0.8) = 29` is the only ordering that guarantees 23 servers survive the loss of one.
So the correct procedure is fixed: rate, then utilisation target, then ceiling, then headroom, then ceiling again, and the intermediate integers are never rounded in a different order.

A second plausible mistake in the same example is to size storage from the logical write rate alone.
That gives \`163.84 GB/day\` and a 30-day figure of \`4.92 TB\`, which is a factor of \`3*1.40 = 4.2\` below the physical requirement of \`20.64 TB\`.
The error is invisible until the cluster is half full, because capacity dashboards usually report logical or post-compression bytes.
The correction is to multiply by the replication factor and by a measured physical overhead, and to compare the result against the storage layer's own reported bytes on a small pilot rather than against an application-side counter.

### Production-scale change

At ten times this workload the read path needs roughly 290 deployed nodes and the egress rises to about 17 Gbit/s.
Beyond that point the design changes shape rather than merely scaling:
the cache tier becomes multi-region and the hit ratio becomes a per-region quantity, because a single-region cache would require cross-region fetches that consume the latency budget.
The storage shard count grows to about 100 for the 30-day window at 2 TB per shard, and the fanout per request becomes the dominant tail risk rather than the per-node queue.
None of those transitions is visible in the formulas alone; they are visible in the sensitivity of the formulas to a parameter, which is why Section 4.4 tabulates sensitivity rather than just boundary failure.

### Remaining uncertainty

The example does not establish what the real hit ratio would be, whether the payload distribution has a tail that dominates egress, or whether the storage engine's physical amplification matches the assumed 1.40.
Those are the three quantities a real project would measure first, in that order, because they carry the largest multiplicative leverage on the design.

**Takeaway:** the worked example converts six assumptions into 29 servers, 20.6 TB, 150 GB of cache, 1.7 Gbit/s, and a 300 ms budget that does not actually fit — and the last result is the most valuable, because a budget that fits on paper and fails in practice is the defect this procedure exists to catch.

## 6. Visual / Geometric Intuition

The three diagrams below are schematic and not to scale.
Each carries labelled axes or states and a named key point, because an unlabelled diagram teaches nothing that a paragraph would not.

### 6.1 The utilisation cliff

```text
wait W (multiples of service time s)
 20 |                                                    *
    |                                                  *
 15 |                                               *
    |                                            *
 10 |                                     *
    |                              *
  5 |                    *
    |        *
  2 |*
  1 +--+-----+-----+-----+-----+-----+-----+-----+-----+--> rho
    0.0  0.15  0.3  0.45  0.6  0.7  0.8  0.9  0.95  0.99

curve: W/s = 1/(1-rho).  Key point: rho = 0.95 gives 20x s, rho = 0.99
gives 100x s.  The curve is hyperbolic, so equal increments in rho are not
equal increments in wait.
```

Reading the diagram: the vertical axis is wait measured in service times, so a server with a 1 ms service time at `rho = 0.95` waits 20 ms.
The horizontal axis is utilisation.
The shape is the reason utilisation targets exist and the reason a 5-percentage-point move from 0.90 to 0.95 is not a small change.
The diagram also shows why the mean is a poor summary near the cliff: the variance grows with the mean in this family, so the tail moves faster than the eye expects.

### 6.2 The time-box as a fixed-width bar

```text
T = 45 min
|<--5-->|<--6-->|<---8--->|<---9--->|<---9--->|<--5-->|<3>|
 scope  contract  sizing    shape     deep     fail   valid
 11%    13%       18%       20%       20%      11%    7%

cumulative checkpoints:
  at t=5   : one action + two exclusions written
  at t=11  : write contract + freshness bound written
  at t=19  : peak rate, node count, bytes all written
  at t=28  : boxes drawn, each labelled with a component count
  at t=37  : one success trace + one failure trace complete
  at t=42  : three metrics + rollback named
  at t=45  : decision record closed
```

Reading the diagram: each segment is a phase, and the checkpoints below the bar are the observable evidence that the phase happened.
A phase without its checkpoint has not happened, regardless of how long was spent in it.
The checkpoints are deliberately written as artefacts (written numbers, named contracts) rather than as activities, because activities cannot be verified from the outside.

### 6.3 The read path with cache branch and fanout

```text
client
  |  (1) 17,361 req/s peak
  v
edge / TLS terminate
  |
  v
app fleet (29 nodes, 599 req/s each)
  |
  +--> cache lookup  --hit 92%--> return  (no origin traffic)
  |
  +--> origin / storage router
           |
           +--> shard A            +--> shard B  > (3-way fanout, all must succeed)
           +--> shard C /
                    |
                    v
              response assembly

bandwidth annotations:
  client<->edge      203.5 MiB/s
  app->cache         203.5 MiB/s (hit payload also crosses this link)
  app->origin         16.3 MiB/s
```

Reading the diagram: the branch at the cache is a fork, not a sequence, and the 92% label is the hit ratio whose complement sizes the origin.
The fanout bracket on the right is where the tail risk lives: three shards must all answer, so the per-request p99 is the maximum of three shard latencies, not the average.
The bandwidth annotations show that the cache does not reduce the app-to-cache link, only the app-to-origin link, which is why a cache is not a bandwidth fix for the front end.

### 6.4 Queue behaviour under a burst

```text
backlog B
    |
1.4M|            /|
    |           / |
    |          /  |
    |         /   |
    |        /    |
   0+-------+-----+-----+----------> time
         t_o=300s  |  T=283s
                  burst ends        drained
        lambda_o = 34,722 > mu = 30,000   -> growth 4,722/s
        after burst: lambda_r = 25,000 < mu_r = 30,000 -> drain 5,000/s
```

Reading the diagram: the rising edge is the overload interval with slope `lambda_o - mu`, and the falling edge is the drain with slope `mu_r - lambda_r`.
The key point is that the descending edge is shallower than the ascending edge in this example, so recovery takes almost as long as the incident.
If the two slopes were equal in magnitude the triangle would be symmetric and the area — the total queueing work — would be easy to estimate.
The area under the curve is the total excess work, which is `0.5*1,416,667*300 = 2.13e8` request-seconds, a quantity that directly determines whether the drain is feasible in the available quiet period.

**Takeaway:** the four sketches reduce to three relationships — a hyperbolic cliff in utilisation, a fixed budget that must be spent deliberately, and a branching path whose tail is set by the slowest branch — and each of them is checkable by arithmetic rather than by intuition.

## 7. Comparison / Selection Table

The flow's phases are fixed, but three decisions inside the flow genuinely vary by workload: the consistency contract for the read path, the mechanism for absorbing peak load, and the shard key.
Each is compared below with its baseline, its disqualifier, and its cost.

### 7.1 Consistency contract shapes

| Shape | Preconditions | Guarantee delivered | Cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Single-writer strong consistency | All writes for a key route to one owner | Linearizable reads and writes for that key | Cross-region write latency; owner outage blocks writes | Ledger balances, inventory reservation, uniqueness constraints | Write latency budget below round-trip to the owner region; multi-region write availability required | DERIVED from consistency model definitions |
| Read-your-writes routing | Session stickiness or version token | A client sees at least its own prior writes | Sticky routing, token plumbing | Profile edits, comment posting, settings | Anonymous readers share no session identity | DERIVED |
| Bounded staleness | Replicas lag within a measured bound | Reads may trail the writer by at most `Delta` time | Replica lag monitoring and a serving gate | Feeds, catalogues, search indexes | A stale read causes a wrong financial or safety action | DERIVED, requires measured lag |
| Eventual consistency, unbounded | None stated | Copies converge when replication runs | Cheapest in the happy path | Analytics copies, caches | Any read path that cannot tolerate an unbounded stale window | ILLUSTRATIVE as a slogan; OPEN unless lag is measured |
| Quorum read/write | `R + W > N` | Read set intersects write set | Latency of the slowest of `R` or `W` replicas | Multi-node key-value stores | Latency budget cannot afford the quorum tail | DERIVED |

The quorum row deserves its derivation because it is often quoted as a slogan.
With `N` replicas, a write acknowledged by `W` and a read served by `R` have intersecting sets whenever `R + W > N`.
For `N = 3`, `W = 2` gives intersection with any `R >= 2`; `R = 1, W = 3` also intersects because the read's single replica must be in the write's complete set.
The guarantee is about set intersection, not about ordering, so it does not by itself give linearizability; it gives read-your-write at the version granularity the store exposes.
That distinction is the difference between a correct and an incorrect interview answer, and it is the reason the row's "guarantee delivered" column says "read set intersects write set" rather than "strong consistency".

Selection order for consistency:

```text
IF a stale read causes an irreversible wrong action
   (money movement, safety interlock, quota enforcement)
   -> single-writer strong consistency on that key, accept the latency.
ELSE IF the writer must observe its own effect (edit, post, settings)
   -> read-your-writes routing or a version token; do not require global strong.
ELSE IF a bounded staleness figure is measurable and the product accepts it
   -> bounded staleness with a serving gate that rejects reads beyond the bound.
ELSE IF the read is purely analytical or a cache
   -> eventual consistency, but state the lag bound you will monitor.
If two conditions conflict, the irreversible-action condition wins; escalate
the latency budget rather than the consistency level.
```

### 7.2 Peak-load absorption mechanisms

| Mechanism | Preconditions | What it actually does | Cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Vertical scaling | Single-node limit not reached | Raises per-server service rate | Step-function cost, blast radius on failure | Small systems, quick mitigation | Per-server rate already at the technology ceiling | DERIVED |
| Horizontal fan-out | Work is partitionable and stateless per request | Raises total service rate linearly | Coordination, cache locality loss, cost linear in load | Read paths, stateless APIs | Requests need cross-partition state on the hot path | DERIVED |
| Cache | Working set fits and reuse is high | Removes `h` of read load from origin | Memory, invalidation complexity, cold-start hazard | Read-heavy with skew | Per-user unique data with no reuse | DERIVED, requires measured `h` |
| Queue with bounded depth | Downstream can drain faster than arrivals on average | Shifts work in time, not in total | Added latency, unbounded growth if depth is not enforced | Bursty but drainable workloads | Sustained overload; the queue becomes a latency sink | DERIVED |
| Admission control / shedding | A defensible priority order exists | Reduces accepted load to protect the served set | Explicit user-visible errors | Overload protection, dependency failure | Every request is equally critical and none may fail | DERIVED |
| Precompute / materialised view | Update rate is below read rate by a wide margin | Converts read work into write work | Staleness, storage, rebuild cost | Feeds, aggregates, leaderboards | Read and write rates are comparable | DERIVED |

The distinction that decides between the cache row and the queue row is worth stating plainly.
A cache reduces the *rate* of work presented downstream, and it does so only for the hit fraction.
A queue reduces the *instantaneous* rate by moving work later, and it does so for the whole admitted set.
A queue therefore cannot fix a sustained rate deficit, because moving work later does not change the total.
The drain arithmetic in SD-14 is the check: if `mu_r <= lambda_r` the backlog never clears, and no amount of queue depth changes that.

### 7.3 Shard-key choices

| Key choice | Preconditions | Load distribution | Operational cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Hash of entity id | Entity id is high-cardinality and stable | Uniform, but range scans scatter | Rebalancing on membership change | Point reads and writes by id | Range queries are a primary access pattern | DERIVED |
| Range on a sortable field | Queries cluster on that field | Skewed by recency and by hot ranges | Split/merge logic; hot tail | Time-series, feeds by recency | The leading range receives most writes | DERIVED |
| Consistent hashing ring | Node membership changes frequently | Uniform in expectation; per-node variance remains | Virtual nodes, ring metadata propagation | Elastic cache and storage tiers | Key popularity is extremely skewed | DERIVED |
| Directory / lookup table | Key-to-shard mapping may be reassigned | Whatever the directory says | Extra hop and a highly available directory | Tenants with growth-based placement | Directory becomes the bottleneck | DERIVED |
| Composite (tenant, id) | Tenant is the isolation boundary | Uniform within tenant, uneven across tenants | Per-tenant hot partitions | Multi-tenant systems | One tenant dominates total volume | DERIVED |

Consistent hashing is often justified with the wrong number.
Adding or removing one node in a ring with `n` nodes moves roughly `1/n` of keys, whereas modulo hashing moves about `(n-1)/n`.
At `n = 10` that is 10% versus 90%; at `n = 100` it is 1% versus 99%.
The expected fraction moved is a statement about key count, not about request miss rate, and the two differ when key popularity is skewed — a point that Section 8 failure F-04 develops.

### 7.4 Phase weighting by workload type

| Workload type | Heaviest phases | Lightest phases | Why | Evidence |
|---|---|---|---|---|
| Read-heavy consumer | Sizing (egress, cache working set), budget (fanout tail) | Contract (reads are simple) | Egress bandwidth and fanout dominate cost and tail | DERIVED from Sections 4–5 |
| Write-heavy event pipeline | Contract (idempotency, ordering), sizing (retention, partitions) | Budget (loose per-event latency) | Duplicate and late events are the correctness risk | DERIVED |
| Money movement | Contract (effect-once, reconciliation) | Shape (few components) | Correctness dominates; performance is secondary | DERIVED |
| Search / retrieval | Sizing (index size), shape (index vs source of truth) | Contract (index is derivable) | Index rebuild and freshness define operability | DERIVED |
| Real-time control loop | Budget (worst case, not p99) | Sizing (small fleet) | Deadline miss is a safety event, not a latency regression | DERIVED |
| Analytics / reporting | Sizing (scan volume), isolation | Budget (batch latency) | Read amplification and isolation from production load | DERIVED |

**Takeaway:** the three selections that most change a design are the consistency contract, the peak-absorption mechanism, and the shard key; each has a baseline, a disqualifier, and a cost, and the disqualifier is what makes the choice defensible rather than stylistic.

## 8. Failure Modes & Edge Cases

Each entry below is a distinct mechanism, not a restatement of another entry's symptom.
Every entry names competing causes, because the same observable — elevated latency, growing backlog, duplicated records — almost never has a single possible origin.
The discriminating check is the part that makes the entry useful; a check that cannot distinguish between hypotheses is not a diagnostic.

### F-01 — Duplicate side effect after a client retry

**Symptom.** The ledger contains two debits for one user action, or the order service contains two orders with identical contents and timestamps one second apart.
**Competing causes.** (a) The client retried after a timeout that occurred after the server committed, and the server has no idempotency record. (b) A message broker redelivered because the consumer acknowledged after processing rather than atomically with it. (c) Two distinct user actions were legitimately issued, and the duplicate is a product bug rather than a transport bug. (d) An upstream load balancer retried a non-idempotent POST on a connection reset.
**Discriminating check.** Join the two records on the client-supplied command identifier. If they share one, it is a transport duplicate (a, b, or d). If they have different identifiers, it is (c) and the defect is in the client's button handling or in a legitimate double submission. To separate (a) from (b), check whether the second record arrived from the broker's delivery path or from the HTTP ingress: the ingress access log and the consumer offset log are different evidence sources.
**Bounded fix.** Move the idempotency record into the same transaction as the effect, keyed on the client's command identifier, with a retained window at least as long as the maximum retry horizon. If the store cannot co-locate them, write the identifier to a unique index on the effect record itself.
**Verification.** Replay the recorded command identifier twice against a test environment and confirm one effect and one stored identifier. Then re-run with the process killed between the effect write and the identifier write, and confirm the reconciliation job repairs the divergence rather than duplicating.
**Provenance.** HYPOTHETICAL, constructed from the standard retry-after-commit mechanism; not an observed incident.

### F-02 — Retry amplification during a dependency slowdown

**Symptom.** Dependency request rate rises above the pre-incident baseline while user-visible request rate is flat or falling, and the dependency's own latency increases further as it receives more work.
**Competing causes.** (a) Layered retries multiplying as in SD-10. (b) A load generator or crawler adding genuine load. (c) A traffic shift from a failing region. (d) A cache tier failing to fill, pushing its misses onto the dependency.
**Discriminating check.** Tag every outbound request with its attempt number and its originating layer. If the attempt-number distribution has mass above 1 while user-visible rate is flat, it is (a). If attempt number 1 dominates and total rate is up, it is (b) or (c) — then compare ingress by source and by region. For (d), compare the cache hit ratio across the same window; a drop in `h` with flat key volume identifies the fill failure.
**Bounded fix.** Cap total attempts per user-visible request rather than per layer, add jittered backoff, and install a retry budget expressed as a ratio of retried to first-attempt requests. During the incident, disable retries at the innermost layer first, because that is where the multiplication compounds.
**Verification.** Re-run a load test with the dependency artificially delayed to the 99th-percentile service time, and confirm that total attempts per user request stay below the budget and that the dependency's received rate does not exceed its pre-incident capacity.
**Provenance.** HYPOTHETICAL; the amplification arithmetic is DERIVED in SD-10.

### F-03 — Cold-cache stampede on origin

**Symptom.** Origin request rate spikes by roughly a factor of `1/(1-h)` for a few minutes after a deploy, a cache node replacement, or a regional failover, and origin latency or error rate rises with it.
**Competing causes.** (a) Cache flush or restart lost the working set. (b) A value-format change invalidated keys. (c) The origin fleet was scaled down and the cache was masking a sizing error. (d) A hot key expired and every concurrent request for it missed simultaneously.
**Discriminating check.** Compare the cache's resident key count and hit ratio over time against the deploy or restart timestamp. A step to zero in resident keys is (a). A hit-ratio drop with resident keys intact is (b). If hit ratio is normal but origin rate is high, it is (c). For (d), inspect miss counts by key: a single key accounting for a large share of misses identifies it.
**Bounded fix.** Bound origin concurrency with an explicit limit and shed beyond it, so the cold path degrades rather than collapses; pre-warm before shifting traffic; add request coalescing so that `n` concurrent misses for one key produce one origin fetch.
**Verification.** Simulate a cache flush under peak load and confirm origin rate stays below its sized capacity and that the shed fraction is visible as an explicit error class rather than as a timeout.
**Provenance.** HYPOTHETICAL; quantified in Section 5 step 6 at a 12.5x ratio.

### F-04 — Hot shard from skewed key popularity

**Symptom.** Per-node utilisation varies from near zero to above the target, mean latency is fine, but p99 latency is dominated by one node and the aggregate error rate is elevated.
**Competing causes.** (a) A single key carries a large share of traffic. (b) Hash function or key construction produces poor distribution for a subset of keys. (c) Uneven shard sizes from an earlier range split. (d) One node has degraded hardware or a noisy neighbour.
**Discriminating check.** Plot request rate per shard and per key. If one key explains most of the variance, it is (a). If the distribution is broad but not uniform, it is (b). If rate is uniform but stored bytes are not, it is (c). If rate and bytes are uniform while latency is not, it is (d) — compare CPU steal, disk queue depth, and network errors for that node against its peers.
**Bounded fix.** For (a), split the key's value across sub-keys with a fan-in read, or replicate the hot key with a read-randomising wrapper. For (b), change the key hash or salt the key. For (c), rebalance with a size-aware split. For (d), replace the node and keep the rate-per-shard evidence.
**Verification.** Confirm that the maximum per-shard rate divided by the mean per-shard rate falls below the target's design tolerance, and that the p99 latency of the slowest shard is within the per-shard budget derived in SD-11.
**Provenance.** HYPOTHETICAL; the fanout requirement is DERIVED in SD-11.

### F-05 — Latency budget that sums paths that are actually alternatives

**Symptom.** The per-hop p99 allocations sum to more than the end-to-end objective, so the design is declared infeasible, or the allocations are trimmed until they fit and the result is a budget nobody can meet.
**Competing causes.** (a) Cache-hit and origin-miss paths were treated as serial. (b) Retries were included in the base budget. (c) Parallel fanout branches were summed instead of maximised. (d) The percentile of the end-to-end objective differs from the percentiles of the components and the mismatch was ignored.
**Discriminating check.** Draw the request as a directed acyclic graph and mark which nodes lie on the critical path. Any allocation on a branch that is not on the critical path is double counting. Then check whether the objective's percentile matches the component percentiles: a p99 total cannot be built from p50 components without a distributional model.
**Bounded fix.** Recompute the budget on the critical path only, take the maximum across parallel branches, hold retries in a separate contingency line, and state the correlation assumption. Then reserve at least 30% of the objective as unallocated headroom.
**Verification.** Instrument one synthetic request per path variant (hit, miss, partial failure) and confirm that the measured sum along the critical path fits the allocation, and that the unallocated headroom remains.
**Provenance.** HYPOTHETICAL; the arithmetic is worked in Section 5 step 7.

### F-06 — Unbounded queue converting overload into latency

**Symptom.** Error rate stays near zero while p99 latency rises without bound, and requests eventually time out at the client even though the service reports success.
**Competing causes.** (a) A queue with no depth limit absorbing a sustained deficit. (b) Consumer concurrency too low for the arrival rate. (c) A poison message causing repeated delivery and blocking a partition. (d) Head-of-line blocking from a slow dependency inside the consumer.
**Discriminating check.** Compare arrival rate to completion rate over a window. A monotonically increasing depth with sustained deficit is (a) or (b); separate them by checking consumer capacity against arrival rate directly. For (c), inspect per-partition offsets: a single partition's lag growing while others keep pace identifies it. For (d), measure the dependency's latency from inside the consumer rather than at the dependency's own metrics.
**Bounded fix.** Set a maximum depth and a maximum message age; reject or shed on breach rather than accepting work that cannot be served within the deadline. Add dead-letter routing for poison messages. Address the deficit by capacity or by admission, not by depth.
**Verification.** Run an overload test at `lambda = 1.2*mu` and confirm that latency stays bounded, that the shed fraction is reported, and that depth returns to its baseline after the load is removed within the drain time computed by SD-14.
**Provenance.** HYPOTHETICAL; drain arithmetic DERIVED in SD-14.

### F-07 — Split-brain write acceptance after a partition

**Symptom.** Two nodes report successful writes for the same key with different values, and after the partition heals one value is silently lost or one writer's acknowledgment is retroactively invalidated.
**Competing causes.** (a) Both sides of a partition retained quorum because the cluster was misconfigured to allow it. (b) A failover promoted a replica that had not received the latest writes. (c) A client wrote to a stale leader address after a leadership change. (d) A multi-region active-active design applied last-writer-wins by wall clock and the clocks disagreed.
**Discriminating check.** Inspect the configuration's quorum requirement and the partition's node counts on each side. If both sides met quorum, it is (a). If only one did and the other still accepted writes, it is (b) or a bypass. For (c), check whether the write's target address matches the leader at the time of the write. For (d), compare the physical clocks of the two writers against the value timestamps.
**Bounded fix.** Require a strict majority for writes, make leadership changes fenced with a monotonically increasing epoch that is validated on write, and if multi-region active-active is required, replace wall-clock ordering with a logical ordering plus a documented conflict-resolution rule that the product can explain to a user.
**Verification.** Partition the cluster in a test environment with the minority side unable to reach a majority, and confirm that minority-side writes fail rather than diverge; then heal and confirm that no acknowledged write is lost.
**Provenance.** HYPOTHETICAL; consistency reasoning DERIVED from quorum intersection in Section 7.1.

### F-08 — Migration that breaks the old and new schema simultaneously

**Symptom.** A deploy of a schema or index change causes errors in one direction of traffic, or a rollback restores the old code against a new schema and fails.
**Competing causes.** (a) The change was not expand/contract: the old code cannot read the new shape and the new code cannot read the old. (b) A backfill ran concurrently with live writes and overwrote newer values. (c) An index build blocked writes. (d) A cache still holds values in the old format after the deploy.
**Discriminating check.** Read the deployment order. If any step removes a field or tightens a constraint before all readers tolerate the change, it is (a). For (b), compare the backfill's read timestamp against the version of the value it wrote; a backfill writing a version older than the current one identifies it. For (c), look for write stalls aligned with index-build start. For (d), check the cache's value version tag.
**Bounded fix.** Split into expand, dual-write/read-new-fallback-old, backfill with version comparison, then contract in a later deploy. Version cache values so a mixed cache is readable. Build indexes with an online path or a shadow table.
**Verification.** Run the migration against a snapshot with concurrent writes injected, and confirm: rollback works at every step, no write is lost, and the dual-read path is exercised by a test that only the new format satisfies.
**Provenance.** HYPOTHETICAL; the phase ordering is DERIVED from the requirement that every intermediate state be readable by both versions.

### F-09 — Timeout chosen without reference to the downstream tail

**Symptom.** A service reports timeouts at a rate that does not match the downstream's error rate, or a retry storm begins when the downstream slows slightly.
**Competing causes.** (a) The timeout equals or is below the downstream's p99, so normal tail events become errors. (b) The timeout exceeds the caller's own deadline, so work continues after the caller has given up. (c) Timeouts are uniform across operations with very different service-time distributions. (d) Connection-pool exhaustion makes the timeout fire while the request is queued, not while it is executing.
**Discriminating check.** Compare the timeout value against the downstream's measured latency distribution at the same load. A timeout near the p99 is (a). Compare the caller's deadline chain; a child timeout exceeding the parent's remaining budget is (b). For (d), measure time-in-pool separately from time-in-service: time spent waiting for a connection is not service time and produces timeouts whose count matches pool occupancy rather than backend latency.
**Bounded fix.** Set the timeout from the caller's remaining budget minus downstream allowance, require child deadlines to be strictly smaller than parent deadlines, and derive per-operation timeouts from measured distributions rather than a global constant. Bound pool wait separately and fail fast on exhaustion.
**Verification.** Inject latency at the 99.5th percentile of the downstream distribution and confirm that errors appear only above the intended threshold, and that no request executes after its caller's deadline.
**Provenance.** HYPOTHETICAL; the deadline-chain rule is DERIVED from the budget composition in SD-09.

### F-10 — Sizing from mean payload when the tail dominates egress

**Symptom.** Provisioned bandwidth is adequate at the mean but links saturate during peak, and the saturation correlates with a subset of large responses.
**Competing causes.** (a) Payload distribution is heavy-tailed and the mean understates the busy-hour byte rate. (b) Compression is enabled for some content types and not others. (c) A response shape changed, for example embedding a larger object. (d) Retransmissions and protocol overhead consume a fixed fraction that was excluded from the estimate.
**Discriminating check.** Compute egress as the sum of bytes over a real time window rather than as rate times mean payload, and compare the two. A large gap is (a). Break egress down by content type to test (b) and (c). Compare link byte counters against application-reported payload bytes to test (d).
**Bounded fix.** Size on measured bytes per second at peak, keep the mean-payload formula only as a floor, and state the tail ratio. Add per-request size limits so a single response cannot consume a disproportionate share of a link.
**Verification.** Re-run the peak-hour replay and confirm that measured egress is below the provisioned link capacity with the stated headroom, and that the largest single response is a bounded fraction of link capacity.
**Provenance.** HYPOTHETICAL; the floor nature of SD-05 is stated in its constraint block.

**Takeaway:** ten distinct mechanisms, each with at least three competing causes and a check that separates them; the recurring theme is that no symptom in this table has one cause, so the diagnostic must be chosen to discriminate rather than to confirm the first hypothesis.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic and operational complexity

The "algorithm" here is the flow itself plus the components it specifies.
Sizes are defined once: `D` daily active users, `r` actions per user per day, `P` peak factor, `k` fanout, `m` workers per node, `n` nodes, `d` retention days.

| Operation | Input sizes | Time | Space | I/O or communication | Assumptions / dominant bottleneck |
|---|---|---|---|---|---|
| Flow phase: scope | 1 prompt | O(1) human time, ~5 min | O(1) | none | bounded by the candidate's discipline |
| Flow phase: sizing | `D, r, P, k` | O(number of components), ~8 min | O(1) | none | arithmetic is trivial; the cost is choosing assumptions |
| Request handling at one node | `m` workers | O(1) service time `1/mu` plus queueing | O(`L`) in-flight = O(`lambda*W`) | one downstream call per hop | queueing and downstream latency, not CPU |
| Read with fanout | `k` shards | O(max of `k` parallel latencies) | O(`k`) concurrent descriptors | `k` network calls | slowest shard; tail not mean |
| Cache lookup + miss fill | `O` objects | O(1) lookup, O(1) fill | O(`h_cache*window*bytes`) | one origin fetch per miss | hit ratio and origin capacity during cold start |
| Storage write with replication | `R` replicas, `n_idx` indexes | O(1) logical, O(`R*(1+n_idx)`) physical | O(`S_phys`) | `R` network writes | write amplification, not logical rate |
| Range scan for reporting | rows in range | O(rows) | O(output) | sequential I/O | scan volume and isolation from production |
| Rebuild of a derived index | all source records | O(total records) | O(index size) | full read of source | source-of-truth availability and rebuild throttling |

The table's most important entry is the second: the request-handling cost is not CPU time but queueing plus downstream latency.
That is why adding CPU to a node whose workers block on I/O does not raise throughput, and why `mu` must be measured for the actual request mix rather than inferred from a microbenchmark.

Two complexity claims deserve explicit boundaries.
The fanout cost `O(max)` is true in parallel and false in a serial chain, where it is `O(sum)`; the difference between a 12-shard parallel read and a 12-shard serial read is a factor of twelve in the latency contribution.
The rebuild cost is linear in records but quadratic in wall-clock cost if rebuild throttling is absent, because a rebuild competing with production I/O degrades production and slows itself, which is a feedback loop rather than a complexity class.

### 9b. Worked sizing: compute, storage, bandwidth, cost

**Workload.** The Section 5 scenario: `D = 20,000,000`, `r_read = 25`, `r_write = 2`, `P = 3`, `s = 12 KiB`, `b = 4 KiB`.
**Environment.** Homogeneous application nodes with `mu = 1,200` read requests/s each and `m = 6` workers per node; a cache tier; a replicated store with `R = 3` and one secondary index. These are illustrative parameters, not a hardware specification.
**Budget.** p99 read latency 300 ms end to end; 30-day hot retention; 35% headroom target for burst and failover.

**Compute.**

```text
lambda_peak_read = 17,361.11 req/s
N_active         = ceil(17,361.11/(1,200*0.65)) = 23
N_deployed       = ceil(23/0.80)                 = 29
per-node rate    = 17,361.11/29 = 598.66 req/s
utilisation      = 598.66/1,200 = 0.4989
in-flight per node = 598.66*1.6629e-3 = 0.996 requests
```

**Storage.**

```text
daily raw      = 163.84 GB/day
30-day raw     = 4.9152 TB
30-day physical= 4.9152*3*1.40 = 20.644 TB = 18.775 TiB
shards at 2 TB = ceil(18.775 TiB / 2 TiB equivalent)
               = ceil(18.775/2) = 10 shards
rows in 30 days = 40,000,000*30 = 1.2e9 rows
rows per shard  = 1.2e9/10 = 1.2e8
```

**Bandwidth.**

```text
peak egress  = 17,361.11*12,288 B = 213.33 MB/s = 203.5 MiB/s = 1.707 Gbit/s
peak ingress = 1,388.89*4,096 B   = 5.69 MB/s  = 5.43 MiB/s  = 0.0455 Gbit/s
```

The read-to-write bandwidth asymmetry is about 37 to 1 in bytes per second and about 12.5 to 1 in request rate.
That asymmetry is the reason a read-heavy design optimises the read path and treats the write path as a correctness problem rather than a throughput problem.

**Cost.** Using an illustrative 450 currency units per node per month and an illustrative 0.021 per GB-month for replicated storage:

```text
compute monthly     = 29*450 = 13,050
requests monthly    = 5,787.04*86,400*30 = 1.500e10
cost per million    = 13,050*1e6/1.500e10 = 0.870
storage monthly     = 20,644 GB*0.021 = 433.5
cache monthly       = 149.9 GB resident at an illustrative 0.10/GB-month = 15.0
egress monthly      = 203.5 MiB/s*2.592e6 s/month = 5.275e11 B = 527.5 GB
                    * illustrative 0.09/GB = 47.5
total illustrative  = 13,050 + 433.5 + 15.0 + 47.5 = 13,546
```

**Bottleneck.** At this scale compute dominates the bill at about 96% of the illustrative total, and the dominant compute cost is read handling.
That conclusion flips at a different workload: a write-heavy pipeline with large retention would have storage and egress dominating, and a design decision that halves retention would then be worth more than a decision that halves node count.
The sensitivity is the point, not the absolute figures.

**Sensitivity to the peak factor.** Recomputed in Section 4.4: 19 deployed nodes at `P = 2` and 57 at `P = 6`, a factor of three.
**Sensitivity to the hit ratio.** At `h = 0.80` the origin rate is 3,472 req/s, requiring `ceil(3472/780) = 5` nodes; at `h = 0.95` it is 868 req/s, requiring 2 nodes before headroom.
**Production difference.** Real deployments add connection setup, TLS handshakes, cross-zone replication traffic, background compaction, metrics export, and retry overhead; each of those appears as additional bytes or additional requests and none is in the model above.

**Economics per successful outcome.** If 0.5% of reads are served from a degraded path that is still a success, and 0.05% fail, the cost per *successful* read differs from the cost per attempted read by a factor of about 1.0005.
That difference is negligible here, but it is not negligible in systems where failure rates are percent-level; quoting cost per attempt when the failure rate is 3% understates the true cost per success by the same 3%.

**Takeaway:** sizing is a sequence of multiplications where the dominant term changes with the workload, so the useful output of Section 9 is not a total but the identification of which single assumption moves the total most.

## 10. Key Theorems / Results Reference

The results below are the ones that make the flow's arithmetic defensible.
Each is stated with its hypotheses, because the hypotheses are the part candidates omit and the part that decides whether a claim is valid.

| Theorem/result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Source |
|---|---|---|---|---|---|
| Little's law | For a stable system in steady state, `L = lambda*W` | stability; stationary rates; consistent system boundary; long-run averages | converts a latency target into a concurrency requirement | no statement about the distribution of `W` | Standard queueing theory (Little, 1961) |
| M/M/1 mean response time | `W = 1/(mu-lambda)` for `lambda < mu` | Poisson arrivals, exponential service, one server, infinite buffer, FCFS | establishes the hyperbolic utilisation penalty | not valid for deterministic or heavy-tailed service | Standard queueing theory |
| M/M/1 response quantile | `P(W <= t) = 1 - exp(-(mu-lambda)t)` | as above, exponential response law | gives `W_p = -ln(1-p)/(mu-lambda)`, so p99 is 4.605x the mean | says nothing about p99.9 without the same model | Standard queueing theory |
| Erlang C | Wait probability `C(m,a)` with `a = lambda/mu`; mean queue delay `C/(m*mu-lambda)` | Poisson arrivals, exponential service, `m` homogeneous servers, one queue, infinite buffer | sizes worker pools and predicts queue probability | assumes no abandonment and no priority classes | Standard teletraffic theory (Erlang) |
| Erlang B recursion | `B_k = a*B_{k-1}/(k+a*B_{k-1})`, `B_0 = 1` | loss system, no queueing | numerically stable route to Erlang C | not the wait probability itself | Standard teletraffic theory |
| CAP / PACELC framing | During a partition a distributed store must forgo either availability or consistency; absent a partition it still trades latency against consistency | a partition is possible; the store replicates | forces an explicit statement of the partition-time behaviour | does not say which choice is correct; does not forbid all consistency levels | Widely taught distributed-systems result; use as framing, not as a theorem with a proof here |
| Quorum intersection | With `N` replicas, read set `R` and write set `W` intersect when `R+W>N` | fixed replica set; each operation contacts distinct replicas; no reconfiguration during the operation | gives read-your-write at the granularity of the stored version | not linearizability; not ordering; not durability without a durable write | DERIVED from set intersection |
| Availability of a serial chain | `A_total = prod_i A_i` for independent components | serial dependency; independence | end-to-end availability is below every component's | independence is usually false; correlation makes it worse | DERIVED from the product rule |
| Availability of a replicated group | `A = 1-(1-A_i)^r` for `r` independent replicas that can each serve | read from any replica; independent failures | replication raises read availability quickly | does not raise write availability; does not cover correlated failure | DERIVED |
| Amdahl-style bound on parallel fanout | Speedup is limited by the serial fraction and by the slowest parallel branch | fixed problem size; independent branches | adding shards past the slowest branch's latency yields nothing | not a statement about throughput | DERIVED from the max-of-`k` latency |
| Consistent hashing movement | Removing one of `n` ring nodes moves about `1/n` of keys; modulo moves about `(n-1)/n` | uniform key hash; ring with balanced virtual nodes | explains why ring membership changes are cheaper | expected key movement is not request miss rate under skew | DERIVED |
| Exactly-once effects | Exactly-once *delivery* is unattainable over an unreliable channel; exactly-once *effects* require the identifier and the effect in one atomic boundary | unreliable channel; retries permitted; a durable store exists | forces the design to name the effect owner and the reconciliation path | does not promise the absence of divergence, only bounded detection | DERIVED |

Three of these deserve a caution that the table cannot carry.

The CAP framing is frequently misused as "pick two of three". The precise statement is about behaviour during a partition: a replicated store that must remain available will serve possibly stale or divergent data, and a store that must remain consistent will refuse or delay some operations.
Outside a partition the tradeoff continues as latency versus consistency, which is why the PACELC extension exists.
An interview answer that says "we choose AP" without saying what a user sees during a partition has used the acronym without making the decision.

The quorum result is about set intersection, so it gives a guarantee at the granularity of the version stored on a replica.
Two writes concurrent with each other may still both succeed and require conflict resolution, because intersection ensures that a read sees *some* recent write, not that writes are serialised.
That is why the row says "read-your-write at the granularity of the stored version" and not "strong consistency".

The exactly-once result is the one most often overstated.
It is not a theorem about a particular product; it is an argument from the atomicity requirement: if the record of having performed an effect and the effect itself can be separated by a crash, then some execution produces either two effects or none, and no retry policy removes that execution.
The constructive response is co-location plus reconciliation, and the honest phrasing is "at-most-once without duplicates for the retained window, with reconciliation for the window boundary".

**Takeaway:** four results — Little's law, the M/M/1 family, the Erlang recursion, and quorum intersection — carry almost all of the flow's quantitative weight, and each is only as good as the hypotheses printed beside it.

## 11. Interview / Exam Questions

Fifteen questions across recall, application, and judgement.
Every answer is checked; where a quantity is computed, the arithmetic is shown.
Each question carries a red flag, because the red flag is what an interviewer is actually listening for.

### Level 1 — Definition / Recall

**Q-01 — What must exist before a component diagram is drawn?**
*Answer.* A named user action, the eligible actors, the success predicate, at least two exclusions, and the write/read contract. The diagram is the consequence of those, not the input to them.
*Reasoning.* A component is only justified relative to a requirement it satisfies; without the requirement, the component list cannot be evaluated and the follow-up question has no answer.
*Acceptable alternative.* Some candidates begin with a back-of-envelope load estimate before writing the contract, provided they state the action first. That order is fine because sizing is also a requirement check.
*Follow-up.* "Which of those five would you drop if you had five minutes?" — the contract, never the action.
*Red flag.* Beginning with "I'd use Kafka and Redis" and no action statement.
*Scoring.* 0 no artefact named; 1 action only; 2 action plus contract; 3 all five plus a stated exclusion.
*Support.* Section 1 and Section 2.1.

**Q-02 — Distinguish offered load from accepted load, and say which one a queue changes.**
*Answer.* Offered load is the arrival rate presented to the system; accepted load is the rate actually admitted. A queue changes neither — it changes *when* the work is done. Only admission control changes accepted load.
*Reasoning.* Conservation of work: a queue accumulates backlog at `lambda - mu` and holds it, but total work is unchanged. Admission converts some offered work into rejections, which is the only mechanism that reduces accepted load.
*Acceptable alternative.* A candidate may say a bounded queue with shedding changes accepted load, which is correct because the shed is admission control at the queue's mouth.
*Follow-up.* "What does an unbounded queue change, then?" — the latency distribution and the failure mode, from latency to memory exhaustion.
*Red flag.* "The queue absorbs the spike, so we don't need to scale."
*Scoring.* 0 conflates the two; 1 defines one; 2 defines both and says the queue changes timing; 3 adds that shedding is the only load reduction.
*Support.* Section 3, F-06.

**Q-03 — Why is "eventual consistency" not a consistency contract?**
*Answer.* Because it names no bound. A contract must state which reads may see which versions, for how long, and what must never be observed. "Eventual" permits an unbounded stale window, which is untestable and can violate a product promise.
*Reasoning.* A predicate over versions and time is testable; a convergence claim without a time bound is not. The deleted-item case is the sharp example: a read must never observe a deleted item after the deletion is acknowledged, and "eventual" does not express that.
*Acceptable alternative.* A candidate may defend "eventual" for an analytics copy, where no read path has a freshness promise. That is defensible if the copy is explicitly excluded from user-facing reads.
*Follow-up.* "Write the contract for an author editing their own profile." — author sees own write immediately; other viewers may trail by at most a stated bound; deleted fields never reappear.
*Red flag.* "Eventual consistency means it's eventually right, which is fine for most things."
*Scoring.* 0 slogan; 1 says it names no bound; 2 gives a three-part contract; 3 adds a concrete never-observe condition and a test.
*Support.* Section 3, Section 7.1.

### Level 2 — Derivation / Apply

**Q-04 — Size the read fleet for `D = 20M`, `r = 25` reads/day, `P = 3`, `mu = 1,200` req/s, `rho_max = 0.65`, headroom 20%.**
*Answer.* `lambda_mean = 20e6*25/86,400 = 5,787.04`; `lambda_peak = 17,361.11`; `N_active = ceil(17,361.11/780) = 23`; `N_deployed = ceil(23/0.8) = 29`; achieved utilisation `17,361.11/34,800 = 0.4989`.
*Reasoning.* Rate first, then capacity, then integrality, then headroom. Applying headroom before the utilisation target gives 28 and is wrong because headroom must apply to an integer server count.
*Acceptable alternative.* Rounding to "about 23 active, about 29 deployed" is fine if the direction of rounding is stated.
*Follow-up.* "What if P is actually 5?" — `N_active = ceil(28,935/780) = 38`, `N_deployed = ceil(38/0.8) = 48`; the peak factor is the largest single source of sizing risk.
*Red flag.* Computing from the mean and multiplying the node count by 3 at the end.
*Scoring.* 0 wrong rate; 1 rate right, count wrong; 2 all four numbers; 3 plus the `P = 5` sensitivity and why ceilings do not commute.
*Support.* SD-01, SD-03, SD-07, Section 4.4.

**Q-05 — A node has 6 workers at 200 req/s each and receives 598.66 req/s. What fraction of requests queue, and what is the mean queueing delay?**
*Answer.* `a = 598.66/200 = 2.993`; `rho = 0.499`; the Erlang B recursion gives `B_6 = 0.0518`; `C = 0.0518/(1-0.499*(1-0.0518)) = 0.0983`; `W_q = 0.0983/(1,200-598.66) = 0.163` ms.
*Reasoning.* Erlang C is the wait probability for a pooled queue; the conditional wait is `1/(m*mu-lambda)`, so multiplying gives the unconditional mean. Check the reduction: with `m = 4` the same load gives `C = 0.507` and `W_q = 2.52` ms, a fifteenfold change from two fewer workers.
*Acceptable alternative.* Approximating with `C ~= 0.1` and concluding "about a tenth of requests queue" is acceptable if the reduction check is shown.
*Follow-up.* "Why does one worker fewer matter so much?" — because `rho` rises from 0.499 to 0.748, and the queue probability grows faster than utilisation.
*Red flag.* Using `W = 1/(mu-lambda)` with `mu = 1,200` and treating the pool as one server; that gives 1.66 ms for the whole time, which conflates service with queueing.
*Scoring.* 0 wrong model; 1 `a` and `rho` only; 2 `C` and `W_q`; 3 plus the `m = 4` comparison.
*Support.* SD-08.

**Q-06 — With a 0.92 cache hit ratio, by what factor does origin read load fall, and what is the cold-start risk?**
*Answer.* Origin load falls by `1/(1-0.92) = 12.5`, so 17,361.11 becomes 1,388.89 req/s. At cold start `h = 0` and the origin receives the full 17,361.11 req/s, so a fleet sized for 1,389 is overloaded by 12.5x.
*Reasoning.* The reduction factor is the reciprocal of the miss ratio, an identity worth memorising. The cold-start case is the same formula with `h = 0`, and it is the case that decides whether the cache is a capacity plan or a latent outage.
*Acceptable alternative.* Sizing the origin for cold start and never benefiting from the cache is also defensible but expensive; it should be stated as a deliberate choice.
*Follow-up.* "How do you serve during the warming window?" — bound origin concurrency, shed beyond it, coalesce concurrent misses for the same key, pre-warm before traffic shift.
*Red flag.* "The cache gives us a 12x cost reduction" without qualifying warm versus cold.
*Scoring.* 0 wrong factor; 1 factor only; 2 factor and cold-start number; 3 plus a mitigation with a shed policy.
*Support.* SD-12, F-03, Section 5 step 6.

**Q-07 — A read fans out to 12 shards, each up 99.95% of the time, independently. What is end-to-end availability, and what would you change?**
*Answer.* `0.9995^12 = 0.994016`, which is about 3,147 minutes of unavailability per year. To reach 99.9% end to end each shard would need `0.999^(1/12) = 0.999917`. The response is to reduce fanout, add hedged or redundant requests on the slowest shards, or make the aggregation tolerate partial results.
*Reasoning.* Serial composition multiplies; the per-component requirement is the end-to-end requirement raised to `1/k`. Partial-result tolerance changes the composition from "all must succeed" to "enough must succeed", which is a different and much cheaper requirement.
*Acceptable alternative.* Assuming full correlation instead of independence gives `0.9995` rather than `0.9995^12`; that is the optimistic extreme and should be labelled as such.
*Follow-up.* "What is the latency consequence of the same fanout at p99?" — each shard must hit the threshold with probability `0.99^(1/12) = 0.999163`, so the per-shard tail target is far stricter than the end-to-end one.
*Red flag.* "Replication makes it highly available" without computing the composition.
*Scoring.* 0 wrong composition; 1 product stated; 2 number; 3 plus per-shard requirement and a fanout-reduction remedy.
*Support.* SD-11, Section 4.4.

**Q-08 — Three nested retry layers each allow 3 attempts at a 20% per-attempt failure rate. How much extra load reaches the dependency?**
*Answer.* Per layer `E = (1-0.2^3)/(1-0.2) = 0.992/0.8 = 1.24`. Three layers give `1.24^3 = 1.907`, so one logical request becomes about 1.9 physical requests.
*Reasoning.* The truncated geometric expectation gives attempts per call; nesting multiplies expectations because each layer independently retries the layer below. Raising `R` from 3 to 5 changes `E` from 1.24 to 1.2496, so extra attempts buy almost nothing while multiplying load.
*Acceptable alternative.* A candidate may argue the layers are not independent because failures are correlated; that makes the true amplification lower in some incidents and higher in others, and the honest answer states the assumption.
*Follow-up.* "What single change gives the most protection?" — a retry budget at the innermost layer plus a circuit breaker, because that is where the multiplication compounds.
*Red flag.* "Retries improve reliability, so more retries are better."
*Scoring.* 0 no arithmetic; 1 per-layer factor; 2 nested factor; 3 plus the marginal-benefit observation and a budget remedy.
*Support.* SD-10, F-02.

### Level 3 — Judgement / Design / Debug

**Q-09 — Latency rose from 120 ms to 900 ms at p99. Throughput is flat, error rate is flat, CPU is flat, and the queue depth metric is not instrumented. Diagnose.**
*Answer.* The flat CPU with rising latency rules out a compute saturation story. The candidates are: queueing behind a downstream dependency, connection-pool exhaustion, head-of-line blocking on a shared resource, or a retry layer that is not visible in error rate. The discriminating check is to measure time-in-pool separately from time-in-service, then to compare downstream latency at the 99th percentile in the same window.
*Reasoning.* If time-in-pool is a large share of the total, the fix is pool sizing or admission, not backend capacity. If downstream p99 rose while its throughput is flat, the dependency is queueing, and the correct response is to reduce concurrency against it rather than add it.
*Acceptable alternative.* Starting with a trace of one slow request is also correct and often faster; distributed tracing answers the same question with less inference.
*Follow-up.* "The dependency says its own latency is unchanged. What now?" — check whether its measurement point is after queueing; a dependency measuring inside the handler reports service time and hides wait.
*Red flag.* "Add more servers" as the first response to a latency increase with flat CPU.
*Scoring.* 0 single guess; 1 two hypotheses; 2 discriminating check; 3 check plus what a negative result means and an escalation.
*Support.* F-09, Section 8 method.

**Q-10 — The same key is written by two regions during a partition and both acknowledge. What do you do, in the interview and in the design?**
*Answer.* In the interview, state that the partition-time behaviour was under-specified and that the design must choose: refuse writes on the minority side, or accept both and define a conflict-resolution rule the product can explain. In the design, require majority quorum for writes, fence leadership with a monotonic epoch, and if active-active is a hard requirement, use a logical ordering with an explicit merge rule rather than wall-clock last-writer-wins.
*Reasoning.* Both sides acknowledging means the quorum requirement was either misconfigured or bypassed. Acknowledged writes cannot be silently dropped, so the choice is between not acknowledging and defining a merge; wall-clock ordering is not a merge rule because clocks disagree.
*Acceptable alternative.* A candidate may choose availability and define a last-writer-wins with a documented clock-skew bound. That is defensible only if the skew bound is stated and the product tolerates the resulting lost update.
*Follow-up.* "Which choice would you make for a balance?" — consistency; the latency cost is acceptable and the irreversibility is not.
*Red flag.* "We'd use vector clocks" without saying what the application does with a conflict.
*Scoring.* 0 no choice; 1 names CAP; 2 states partition behaviour and a fence; 3 adds a merge rule the product can explain and a quorum correction.
*Support.* F-07, Section 7.1, Section 10.

**Q-11 — Your design meets the p99 target on the critical path but the sum of per-hop p99 budgets exceeds the objective. What is wrong and what do you do?**
*Answer.* Either the budgets are allocated across branches that are alternatives rather than serial (the cache-hit and origin-miss paths), or they double count retries, or parallel branches were summed instead of maximised. Recompute on the critical path, take the maximum across parallel branches, move retries to a contingency line, and reserve at least 30% of the objective unallocated.
*Reasoning.* Summing p99 values across alternative paths is a category error; it produces an over-constrained budget that is then trimmed into something unmeetable. The correct composition for means is linear, and for percentiles requires the joint distribution, so a budget must be built on explicit per-hop targets plus headroom rather than on arithmetic that pretends percentiles add.
*Acceptable alternative.* Building the budget bottom-up from measured component distributions and simulating the composition is more rigorous and defensible if the distributions are available.
*Follow-up.* "Where does the retry allowance go?" — into the contingency, because retries are a response to failure and a failure budget is not a latency allocation.
*Red flag.* Trimming each hop until the sum fits.
*Scoring.* 0 no diagnosis; 1 recognises double counting; 2 recomputes on critical path with a max across branches; 3 adds headroom and the percentile-composition caveat.
*Support.* Section 4.4, F-05, Section 5 step 7.

**Q-12 — A stakeholder asks for "five nines". What must be true before you can promise it, and what would you offer instead?**
*Answer.* Five nines is 5.26 minutes of unavailability per year. To promise it you need: every serial dependency at or above that level (five serial components at 99.999% each compose to 99.995%, i.e. 26 minutes), an error budget with a burn-rate alerting policy, a single-request definition of "available", a deployment mechanism that does not remove capacity, and a tested failover. Offer a tiered objective instead: a per-request target, an error budget, and an explicit degraded mode.
*Reasoning.* Composition makes a flat target almost impossible; the honest alternative is to specify what the system does when a dependency is unavailable, because a degraded but successful response is a success under most user-facing definitions.
*Acceptable alternative.* Promising it for a narrow, single-dependency critical path with a fallback is sometimes feasible and should be stated with the dependency list.
*Follow-up.* "What is the error budget per month at 99.95%?" — `0.0005*30*24*60 = 21.6` minutes.
*Red flag.* "We'll add redundancy" with no composition arithmetic.
*Scoring.* 0 accepts the number; 1 states the minutes; 2 composition arithmetic; 3 plus an error budget, a definition, and a degraded mode.
*Support.* SD-11, Section 10, Section 4.4.

**Q-13 — You must ship a schema change with zero downtime and a working rollback at every step. What is the sequence and how do you verify it?**
*Answer.* Expand, dual-write with read-new-fallback-old, backfill with version comparison so the backfill cannot overwrite a newer value, verify, then contract in a later deploy. Verify by running the migration against a snapshot with concurrent writes, and by executing a rollback at each intermediate step.
*Reasoning.* Every intermediate state must be readable by both the old and new code, because during a rolling deploy both are running. A backfill that writes without comparing versions can resurrect an old value over a newer one, which is a data-loss bug that no amount of testing after the fact repairs.
*Acceptable alternative.* A shadow table with a switch-over read path is a valid variant and makes the rollback a routing change, at the cost of double storage during the transition.
*Follow-up.* "What about a cache holding the old format?" — version the cached values and read-fallback on version mismatch; otherwise the mixed cache is a third incompatible reader.
*Red flag.* "Deploy the migration and the code together."
*Scoring.* 0 single-step migration; 1 expand/contract named; 2 all four phases; 3 plus the version-comparison backfill and rollback verification at each step.
*Support.* F-08.

**Q-14 — A candidate's design uses a queue to absorb a 2x peak that lasts 10 minutes every hour. Is that defensible?**
*Answer.* No, not as stated. A 10-minute overload every hour presents 10 minutes of deficit and 50 minutes of surplus per cycle. With `lambda_o = 2*mu` for 600 s, backlog reaches `(2*mu-mu)*600 = 600*mu` requests; draining needs `600*mu/(mu_r-lambda_r)` and only works if recovery arrivals fall well below capacity. The mean rate over the hour is still above capacity in the deficit sense if the deficit is not recovered, and the queue grows monotonically cycle over cycle.
*Reasoning.* Conservation: the queue returns to its baseline only if surplus work per cycle exceeds deficit work. Deficit per cycle is `600*mu` request-seconds; surplus is `3000*(mu_r-lambda_r)`. With `mu_r = mu` and `lambda_r = 0.83*mu`, surplus is `3000*0.17*mu = 510*mu`, which is less than `600*mu`, so the backlog grows each cycle.
*Acceptable alternative.* If the hourly peak is absorbed by *provisioning* rather than by a queue, the design is fine; the defect is the absorption mechanism, not the peak shape.
*Follow-up.* "What would make the queue work?" — recovery capacity above the residual arrival rate with enough surplus per cycle, or a shed policy that caps the backlog.
*Red flag.* "The queue smooths it out."
*Scoring.* 0 accepts; 1 says the queue will grow; 2 backlog arithmetic; 3 plus the per-cycle surplus comparison and a bounded alternative.
*Support.* SD-14, Section 7.2.

**Q-15 — You have 45 minutes and the interviewer asks for "a design for ride tracking". What are your first three sentences?**
*Answer.* First: name the action and its eligible actors, for example "a rider's live position is read by the matched driver and by a support agent, within two seconds of capture". Second: name what is out of scope, for example payment and matching. Third: name the load and the peak assumption you will size on, or say you will estimate it next.
*Reasoning.* Those three sentences convert an ambiguous prompt into a sized problem and make the time allocation explicit. "Ride tracking" alone could mean ingest of GPS points, fan-out to riders, historical replay, or an analytics pipeline, and each has a different bottleneck.
*Acceptable alternative.* A candidate may open with two clarifying questions instead of assertions, provided the answers are then used; questions whose answers would not change the design are worse than assertions.
*Follow-up.* "Which of those is the bottleneck?" — depending on the answers, either the ingest rate (writes with high cardinality) or the fan-out (reads with a small audience and strict freshness).
*Red flag.* Beginning with "I'd use WebSockets, Kafka, and Redis."
*Scoring.* 0 technology first; 1 action only; 2 action plus exclusions; 3 plus load framing and a stated bottleneck hypothesis.
*Support.* Section 1, SD-16, Section 2.

**Takeaway:** the question set spans recall, arithmetic, and diagnosis, and every answer reduces to one of three moves — compute a quantity, name a contract, or discriminate between competing causes — which is what the flow trains.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | System design interview flow: converting an ambiguous prompt into requirements, contracts, a sized shape, and a failure policy within a fixed time box |
| Core formula | `lambda_peak = P*D*r/86400`, then `N_active = ceil(lambda_peak/(mu*rho_max))` and `N_deployed = ceil(N_active/(1-h))`; valid only for homogeneous servers with uniform routing at a stationary peak, and always a floor rather than a measurement |
| Key assumption | The busy-hour peak factor `P` and the cache hit ratio: both are stated as assumptions and both move the node count roughly proportionally, so an unstated value is the largest single sizing risk |
| Complexity | Flow cost is dominated by choosing assumptions, not arithmetic; per-request cost is queueing plus downstream latency (max across parallel fanout, sum along the serial critical path); storage cost is `writes/day*bytes*retention*replicas*index_overhead` |
| Use when | You must produce a defensible design under a time box, or review one for missing contracts, numbers, or failure policy |
| Avoid when | The task is a single well-specified algorithm, or the decision is purely a product-scope question with no system behaviour to size |
| Diagnostic | Flat CPU with rising p99 means queueing or pooled-wait, not compute; a rising dependency rate with flat user rate means retry amplification; a step change in origin rate with a flat cache key count means a cold or invalidated cache |
| Guardrail | Sum the p99 budget along the critical path only, take the maximum across parallel branches, and leave at least 30% of the objective unallocated |
| Evidence limit | All workloads and costs here are ILLUSTRATIVE; the arithmetic is DERIVED and recomputed, but no figure is a benchmark of any real service, and the file is unreviewed |
| Go deeper | [Module 04 cheat-sheet index](INDEX.md) for the planned capacity-formula and distributed-tradeoff sheets; the capacity formulas here are the prerequisite for both |
