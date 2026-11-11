## 9. Complexity & Cost Analysis

Cost in this domain is dominated by three quantities that are easy to mismeasure: the dedupe sort, the out-of-order state buffer, and the heartbeat volume. Each is computed below rather than asserted, and each has a stated regime in which a cheaper structure is adequate.

### 9.1 Dedupe and transition application

Applying a transition log idempotently requires knowing, per `(unit_id, transition, attempt)`, whether the transition has already been applied. Two structures are viable.

```text
FORMULA [C-01]:  T_sort = k * N * log2(N)          T_hash = k * N
WHERE:  N     number of event rows applied in one run
        k     constant per comparison or per hash probe (implementation-specific, not measured here)
CONSTRAINT:  the hash structure must fit in memory; the sort does not. Break-even is therefore
             set by memory, not by the asymptote.
NUMERICAL EXAMPLE:  N = 3,840,000 rows/day (a 2,400-order day at 1,600x is not the case; this is
             a 1,600-order-per-hour estate over 24 hours). With 16 bytes per key and load factor
             0.5, the key set costs 32 B/row * 3.84e6 = 122.9 MB, plus value slots ~48 B/row
             = 184 MB, total about 307 MB. That fits one process; a month of rows (115e6) would
             need about 9.2 GB and no longer fits a single worker.
INTERPRETATION:  for a daily batch the hash path is fine and simpler; for a month-scale replay the
             sort path or a partitioned hash is required, at a cost of log2(3.84e6) = 21.9x
             comparisons per row relative to the linear path in the comparison model.
```

Validity condition: the comparison-model factor `log2(N)` describes sorting, not a specific engine. A columnar engine with dictionary-encoded keys can group far faster than the asymptote suggests, and a streaming engine with pre-sorted input is linear. Quote the factor, never a wall-clock time, unless you have measured on your own estate.

**The cheaper regime.** Below roughly one million transitions per run, keep the state in a single process and skip partitioning entirely. The engineering cost of distributed dedupe exceeds the compute saving until the key set stops fitting, and a distributed dedupe introduces its own failure mode: partial application on a worker crash, which requires the run to be idempotent anyway — at which point per-row idempotence has already been solved and the distribution is optional.

### 9.2 Out-of-order state buffer

To apply transitions in `t_e` order with a bounded horizon `H`, the processor must hold every unit whose state is awaiting a possibly-later event.

```text
FORMULA [C-02]:  B = r_late * H * u
WHERE:  r_late  late-arrival rate, events per second arriving with t_e older than the current
                watermark (dimensionless share times arrival rate)
        H       horizon in seconds (declared policy, e.g. 21,600 s for a six-hour store batch)
        u       bytes per retained unit state (id, current state, pending transitions, timestamps)
NUMERICAL EXAMPLE:  arrival 44.4 events/s (3.84e6/day), late share 0.06, H = 21,600 s,
                u = 480 B.
                B = 44.4 * 0.06 * 21,600 * 480 = 27,617,280 B ~ 26.3 MiB.
                With H doubled to 43,200 s: 52.7 MiB. Linear in H, as expected.
INTERPRETATION:  a six-hour horizon costs single-digit megabytes for this estate, which is why the
                horizon can be chosen by policy (long enough to cover the worst batch) rather than
                by memory pressure. The constraint that bites is not memory but the latency of the
                real-time view, which is H behind by construction.
```

Validity condition: `r_late * H` must be below the number of distinct units in the horizon, or the model double-counts. In a small estate with 740 collections and a six-hour horizon, the late arrivals are dominated by a handful of nodes, so the aggregate rate hides concentration; compute `B` per node, not estate-wide, and size for the worst node. That node's buffer is the one that blocks the watermark, and one blocked node stalls every consumer downstream of the merge.

**The cheaper regime.** For nodes that batch daily and never stream, route them to a separate daily path and exclude them from the streaming watermark. The streaming view then covers only real-time nodes and stays minutes behind; the batch nodes appear in the reconciled daily view. Mixing the two in one pipeline forces the horizon to the slowest node, which makes the real-time view useless for all of them.

### 9.3 Heartbeat volume and retained rollups

```text
FORMULA [C-03]:  V_hb = n_nodes * (86,400 / p)          (events per day)
WHERE:  n_nodes   emitting nodes
        p         heartbeat period in seconds
NUMERICAL EXAMPLE:  61 nodes, p = 900 s -> 61 * 96 = 5,856 events/day.
                At 120 B/row raw, that is 702,720 B/day ~ 0.67 MiB/day, 245 MiB/year.
                Hourly rollups at 24 rows/node/day: 61 * 24 = 1,464 rows/day, 176 KB/day at
                120 B, 64 MiB/year for ten years.
INTERPRETATION:  raw heartbeats at 15-minute cadence are cheap in bytes and expensive in row
                count: they exceed the daily collection-event count (651) by about nine times,
                which matters for any per-row cost, quota, or scan-priced query engine.
```

Validity condition: the rollup must retain the fields the alerting rule needs — last-seen timestamp, count of events in the interval, and client version. A rollup that keeps only a count cannot distinguish "alive and idle" from "alive and broken", which is the whole point of the heartbeat. Keep three fields minimum; drop the rest.

### 9.4 Replay and restatement cost

```text
FORMULA [C-04]:  C_replay = N * (1 + m_join) * c_row + s * N
WHERE:  N       rows replayed
        m_join  mean joined rows per event (Section 4.3)
        c_row   per-row compute cost
        s       per-row storage write cost for the restated period
NUMERICAL EXAMPLE:  90 days at 3.84e6 events/day = 345.6e6 rows; m_join = 1.32;
                (1+m_join) = 2.32 -> 802e6 joined-row operations. If the engine sustains 50e6
                operations/minute, that is 16.0 minutes of pure join work, before shuffle and
                skew, so plan on a multiple of it rather than the figure itself.
INTERPRETATION:  a nine-month replay is a batch job measured in hours in practice, and the
                practical constraint is that it must be idempotent, because it will be re-run.
```

Validity condition: replay is only meaningful if the definition version is pinned per row. Replaying 90 days under today's definition produces a series that is internally consistent and historically false; the series is legitimate only if labelled as restated-on-definition-vN.

### 9.5 Human cost of the taxonomy itself

The scheme has a staffing cost that is routinely omitted from the business case: the producer contract must be reviewed, the alias table owned, the control totals reconciled, and the quarantine drained. A defensible estimate for a 60-node estate is a quarter of one analyst's time for the first two quarters, then roughly two days per month for maintenance, plus the store-side work to bring 14 legacy clients current. The store-side work is the item that actually delays the programme, because it competes with trading priorities; sequencing it before the metrics are promised is the difference between a two-quarter delivery and an open-ended one.

The cost that is not worth paying: a bespoke taxonomy service with its own API, its own UI, and its own permission model, built to serve one dashboard. That pattern appears because the taxonomy is politically contentious, and the service becomes the venue for the argument. The cheaper resolution is a written definition with a named owner and a diffable version, reviewed in the same place the schema changes are reviewed.

Takeaway: the arithmetic in this section is dominated by row count and horizon, not by cleverness; choose the simplest structure whose key set fits, and make the run idempotent so that every later decision is reversible.

## 10. Key Theorems / Results Reference

The results below are stated as invariants and dominance relations, because each can be checked by arithmetic on a single example rather than requiring an experiment. Each carries the condition that makes it true and the observation that would falsify it.

### R1 — Conservation of transitions

If every event instance is the unique witness of one legal transition, then for any unit and any window, `sum over states of (entries) = sum over states of (exits) + 1` for a unit that started in `committed` and is now terminal. Equivalently, the count of terminal states equals the count of units that entered the machine minus those still live.

Falsified by: any unit with two terminal states, or a unit whose state count is zero. Both indicate duplicate emission or a state that was overwritten. This is the cheapest integrity check in the sheet: one query, one number, and it catches the majority of taxonomy violations.

### R2 — Grain dominance for on-time rate

For any population where some orders contain a late unit and an on-time unit, order-grain on-time rate is greater than or equal to unit-grain on-time rate, with equality only when lateness is perfectly correlated within orders. The Section 5.3 trace shows the gap at 2.2 percentage points with 1.0723 parcels per order; the gap grows with both the split rate and the independence of lateness across parcels.

Falsified by: a case where the order-grain rate is below the unit-grain rate, which can only occur if orders are weighted differently between the two computations — usually because the order-grain computation excludes cancelled orders that the unit-grain one includes. That is not a falsification of the relation but of the claim that the two use the same population.

### R3 — Coverage correction bias

Applying `1/c` to a rate computed on covered rows is unbiased only if coverage is independent of the outcome. When coverage and outcome are correlated — as they are whenever the unreliable emitter is also the poor performer — the corrected rate is biased toward the well-covered nodes. In the Section 5.2 example, the correlated case gives 72.26% against a naive correction of 76.68%; the bias is 4.4 points and it points in the direction of flattering the estate.

Falsified by: a demonstration that covered and uncovered subsets have equal outcome rates. That is testable where a control total gives the outcome for both subsets — journal receipts give the outcome for both, so the test is available, and it usually fails.

### R4 — Fan-out inflation is convex in the parcel distribution

For a sum over a joined output where the left table is events and the right is units, the ratio of the joined sum to the true sum equals the mean parcel count per matched order, weighted by event volume. Because the multiplier is a weighted mean, it is dominated by the tail: a single 12-parcel order contributes 3x to its own event and raises the aggregate more than eleven 2-parcel orders of the same value.

Falsified by: a distribution where every order has exactly one parcel, in which case the multiplier is 1 and the join is safe. State this condition explicitly rather than relying on it.

### R5 — Promise-snapshot sufficiency

Promise accuracy computed against per-event snapshots equals the accuracy computed against a full revision history, provided that every revision is recorded as an event carrying both old and new values, and that the evaluation uses the snapshot valid at the terminal timestamp. Storing only the current value destroys sufficiency: any order revised after its terminal event is misclassified, and the Section 5.3 sensitivity shows a 1.7-point swing from that single defect.

Falsified by: a revision path that mutates the header without an event — for example a customer-service agent editing the promise in an admin screen with no audit row. That path is common, which is why the invariant is stated at write time rather than at read time.

### R6 — Heartbeat non-substitution

Heartbeat presence proves the emitter is alive and proves nothing about business events; event presence proves something happened and proves nothing about coverage of what did not. Neither can substitute for the other, and a monitoring design that alerts only on one will miss the failure the other detects. The Section 8.7 matrix has four cells — heartbeat present or absent crossed with events present or absent — and each cell maps to a different owner.

Falsified by: a node whose emitter is alive and reports heartbeats while its business-event validation rejects every row. That is cell (heartbeat yes, events no, quarantine non-empty), and it is the cell most often misdiagnosed as "quiet day".

### R7 — Replay determinism requires total order per key

A replay reconstructs the true state only if transitions for each unit are applied in a deterministic order. Ordering by `t_e` is deterministic; ordering by `t_i` is not, because ingest order depends on network timing. Ties in `t_e` must be broken by a stable secondary key, typically the transition identity, or two replays of the same data produce different states — which is observed as a metric that changes when the job is re-run unchanged.

Falsified by: two replays of identical input producing different counts. That outcome is not a hypothesis; it is a defect with a specific repair, and it should be a scheduled test rather than an incident finding.

Takeaway for Section 10: these seven statements are what make the taxonomy auditable. Each has a one-query falsification test, and the tests are cheap enough to run on every release.
