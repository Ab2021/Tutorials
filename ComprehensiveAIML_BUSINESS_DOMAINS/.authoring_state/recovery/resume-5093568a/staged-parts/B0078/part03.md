## 5. Worked Numerical Example

The scenario is one entity, 13 events, and two cutoffs. Every number below is ILLUSTRATIVE and was
recomputed from the ledger; no operational system produced it.

### 5.1 The setup

Feature definition, version `card_spend_7d_v2`:

```text
feature card_spend_7d  (entity: card_id, unit: USD, window W = 168 h)
  filter  t_e >= t_c - 168h  AND  t_e < t_c  AND  t_i <= t_c
  value   SUM(amount_usd)
  null    NULL when zero rows are visible (never 0.00)
```

Entity `C-4471`. The source table is append-only with columns `card_id`, `ts_event`, `amount_usd`,
and `lag_h`, where `lag_h = t_i - t_e` is measured at the ingest boundary — not copied from a
source-supplied field. Cutoff `t_c = 2026-03-07T00:00Z`.

| # | ts_event (UTC) | amount USD | lag_h | `t_i` (event + lag) | in window `[02-28T00, 03-07T00)` | visible at `t_c` |
|---|---|---|---|---|---|---|
| 1 | 2026-02-27 21:10 | 12.40 | 0.30 | 02-27 21:28 | no (before window) | yes |
| 2 | 2026-02-28 08:05 | 88.10 | 0.15 | 02-28 08:14 | yes | yes |
| 3 | 2026-03-01 09:15 | 40.00 | 0.10 | 03-01 09:21 | yes | yes |
| 4 | 2026-03-02 18:40 | 125.50 | 0.20 | 03-02 18:52 | yes | yes |
| 5 | 2026-03-03 22:55 | 7.60 | 1.05 | 03-03 23:58 | yes | yes |
| 6 | 2026-03-04 11:05 | 62.25 | 0.30 | 03-04 11:23 | yes | yes |
| 7 | 2026-03-04 19:20 | 33.80 | 31.5 | 03-06 02:50 | yes | yes (02:50 < cutoff) |
| 8 | 2026-03-05 22:30 | 18.75 | 26.0 | 03-07 00:30 | yes | **no** (after cutoff) |
| 9 | 2026-03-06 08:00 | 210.00 | 0.07 | 03-06 08:04 | yes | yes |
| 10 | 2026-03-06 12:45 | 9.90 | 2.40 | 03-06 15:09 | yes | yes |
| 11 | 2026-03-06 23:50 | 54.30 | 0.05 | 03-06 23:53 | yes | yes |
| 12 | 2026-03-07 06:12 | 19.99 | 0.05 | 03-07 06:15 | no (after cutoff) | no |
| 13 | 2026-03-08 14:10 | 75.00 | 0.10 | 03-08 14:16 | no (after cutoff) | no |

Row 7 is the instructive one: a 31.5-hour lag, but it landed at 02:50 on 03-06, which is before
the cutoff, so it *is* legal. Long lag does not imply invisible; only `t_i > t_c` does. A designer
who filters on "lag <= some maximum" instead of on `t_i <= t_c` would wrongly drop row 7 and keep
row 8 — inverting the correct decision on both.

### 5.2 The four computed values

```text
A  naive sum, all 13 rows, no filters
   = 12.40 + 88.10 + 40.00 + 125.50 + 7.60 + 62.25 + 33.80 + 18.75
     + 210.00 + 9.90 + 54.30 + 19.99 + 75.00
   = 757.59 USD          (13 rows, mean 58.28 USD)

B  event-window sum, filter on t_e only  (rows 2..11)
   = 757.59 - 12.40 (row 1) - 19.99 (row 12) - 75.00 (row 13)
   = 650.20 USD          (10 rows, mean 65.02 USD)

C  point-in-time sum, filter on t_e AND t_i  (rows 2..7, 9..11)
   = 650.20 - 18.75 (row 8)
   = 631.45 USD          (9 rows, mean 70.16 USD)      <- correct value

D  contamination of each wrong answer, relative to C
   (B - C)/C = 18.75/631.45 = 2.97%      late-row leakage left by a t_e-only join
   (A - C)/C = 126.14/631.45 = 19.98%    naive join, future rows included
   (A - B)/B = 107.39/650.20 = 16.52%    future rows plus the aged-out row 1
```

The three gaps are different defects. `A - B` mixes two causes: rows 12 and 13 had not happened
(a genuine future leak of `19.99 + 75.00 = 94.99`) and row 1 had aged out (an overcount of
`12.40`). `B - C` is the pure arrival leak. A team that "adds a window filter" fixes 16.52% of the
contamination and, if it stops there, ships a model trained on the remaining 2.97% plus whatever
other entities contribute on top.

Row-count and amount-count views disagree, which is a useful sanity constraint:
`1/10 = 10.0%` of window rows were late, but those rows carry only `8.08%` of window dollars
(`18.75/650.20`). Late rows here are smaller than average, so a count-based leak estimate
overstates the dollar impact by `10.0/8.08 = 1.24x`. Neither statistic is "the" leak; state which
one you measured.

### 5.3 Independent re-check by a different route

Enumerate and sum the visible rows directly, then confirm by subtraction:

```text
visible rows = 2, 3, 4, 5, 6, 7, 9, 10, 11        (nine rows)

 88.10
+40.00  = 128.10
+125.50 = 253.60
+  7.60 = 261.20
+ 62.25 = 323.45
+ 33.80 = 357.25
+210.00 = 567.25
+  9.90 = 577.15
+ 54.30 = 631.45        matches C by direct sum ✓

cross-check via subtraction:
  window sum 650.20 - excluded row 8 (18.75) = 631.45 ✓
  all rows   757.59 - (12.40 + 19.99 + 75.00 + 18.75)
            = 757.59 - 126.14 = 631.45 ✓
```

Three routes agree, and the partial sum after row 7 (`357.25`) is the number the grace-period
variant in 5.4 will reuse. The re-check is not decoration: the first draft of this example
subtracted the late row twice and produced 597.65, which no single route reproduces. Enumerate,
then subtract; when the two disagree, the visible set is miscounted, and the error is almost never
in the arithmetic.

### 5.4 Sensitivity: what changes the answer

| Change | Mechanism | Recomputed value | Rows visible |
|---|---|---|---|
| baseline `W = 168`, `G = 0` | reference | 631.45 | 9 |
| `W = 72 h` | window becomes `[03-04T00, 03-07T00)`; rows 2–5 leave | 370.25 | 5 |
| grace `G = 26 h` | also requires `t_e <= 2026-03-05T22:00`, dropping rows 9–11 | 357.25 | 6 |
| `W = 336 h` | window widens to 14 days; row 1 enters | 643.85 | 10 |
| `log1p` then sum | monotone transform, same rows | 34.6454 | 9 |
| count instead of sum | row count, not dollars | 9 | 9 |
| cutoff `2026-03-06T00:00Z` | window `[02-27T00, 03-06T00)`; rows 6,7 late | 335.85 | 6 |

Two of these deserve arithmetic. **Grace 26 h** applies `t_e <= t_c - 26h = 2026-03-05T22:00`, which
removes rows 9 (`08:00`, 210.00), 10 (`12:45`, 9.90) and 11 (`23:50`, 54.30) — `274.20` USD in
total — from the nine visible rows. `631.45 - 274.20 = 357.25`, so this grace discards 43.4% of the
feature to protect against a lag tail it already handled correctly. Row 7 at `19:20` survives
because it is before the band, which is the point of using `t_e` and not `t_i` in the grace rule:
the grace is a policy about *when things happened*, while the availability filter is a fact about
*when they arrived*.

**Count versus sum:** nine rows are visible, so the count feature is `9`, and the two aggregates
differ only in unit — dollars versus rows — not in row set. Any divergence between a count and a
sum feature on the same window is a null-handling difference, not a filtering difference.

**Cutoff moved back one day:** window `[02-27T00, 03-06T00)` contains rows 1–8 for
`12.40 + 88.10 + 40.00 + 125.50 + 7.60 + 62.25 + 33.80 + 18.75 = 388.40`. At that cutoff both late
rows are invisible (row 7 becomes visible at 03-06 02:50, row 8 at 03-07 00:30), so the visible sum
is `388.40 - 52.55 = 335.85` and the leak rate is `52.55/335.85 = 15.65%` — 5.3 times the 2.97% at
the later cutoff, with the same feature definition and the same ledger.

That last figure is the design fact worth memorizing: **the leak rate is a property of the cutoff
relative to the lag mass, not of the feature.** A backtest whose cutoffs sit close behind the last
source update shows heavy contamination; one whose cutoffs sit a month back shows almost none. Both
can be run on the same week of data, and the difference in measured leakage between them explains
most "the model worked in backtest" regressions.

### 5.5 Recomputing the aggregate two ways in SQL

```sql
-- Route 1: point-in-time, explicit and auditable
SELECT l.card_id, l.t_c, SUM(e.amount_usd) AS card_spend_7d
FROM   labels l
LEFT   JOIN events e
  ON   e.card_id = l.card_id
 AND   e.ts_event >= l.t_c - INTERVAL '168 hours'
 AND   e.ts_event <  l.t_c
 AND   e.ts_ingest <= l.t_c
GROUP  BY l.card_id, l.t_c;

-- Route 2: row-count check on the same join; must agree with route 1's non-null rows
SELECT l.card_id, l.t_c, COUNT(e.amount_usd) AS n_rows
FROM   labels l
LEFT   JOIN events e
  ON   e.card_id = l.card_id
 AND   e.ts_event >= l.t_c - INTERVAL '168 hours'
 AND   e.ts_event <  l.t_c
 AND   e.ts_ingest <= l.t_c
GROUP  BY l.card_id, l.t_c;
```

Three details carry the correctness. The `LEFT JOIN` keeps entities with no visible history so
that the null policy is exercised rather than silently dropping them — an `INNER JOIN` here is the
single most common cause of "the training set has 3% fewer rows than the label table and we do not
know which 3%". `COUNT(e.amount_usd)` rather than `COUNT(*)` counts non-null amounts, so a
nullable amount column does not inflate the row count. And both routes must use the *same* filter
clause text; duplicating it in two queries invites the drift that Section 8 catalogs as the
"definition forked during review" failure.

## 6. Visual / Geometric Intuition

### 6.1 The window as a shifted strip

Draw time as an axis with the cutoff as a vertical line. A naive join uses the strip
`[t_c - W, t_c)`. The point-in-time join uses the same strip, minus a wedge at the right whose
width varies per row — the set of rows whose lag pushes their visibility past `t_c`.

```text
  t_e axis  ----------------------------------------------> t_c
  |----------------- W = 168 h -----------------|----------|
  [            usable span 142 h for L = 26 h  ][ lag 26 h ]
  |<----- rows a PIT join can use ------------>|<-- excluded -->
                                               \_ not yet visible at t_c _/

  a row at event time u = t_c - s, where s = age in hours:
     legal iff s >= L,  contributing iff  L <= s < W
     so it feeds exactly the cutoffs on the age band  [L, W)
```

The last line is the geometric content: a row of fixed lag `L` feeds `W - L` hours' worth of
cutoffs and is dead weight for the remaining `L` hours of its nominal window. For a degenerate lag
distribution the lost fraction is exactly `L/W`.

### 6.2 The leakage triangle

For a fixed window, mass splits into three regions relative to the cutoff and the lag boundary:

```text
                          t_c
                           |
    true feature     |  A  |  B  |     A: t_e < t_c - G, t_i <= t_c    (kept)
    (legal rows)     |     |     |     B: t_e in [t_c-G, t_c)          (discarded)
                     |-----+-----|     C: t_e < t_c, t_i > t_c         (leak if used)
                     |  C  |     |     D: t_e >= t_c                   (future leak)
                     +-----+-----+
                        D (future)
```

`B` shrinks as `G` shrinks and `C` grows as `G` shrinks; the two are locked in trade. `D` is
removable outright by any correct `t_e < t_c` filter, which is why future leakage is a beginner
error and arrival leakage is not: `C` survives a filter that looks right.

### 6.3 Latency budget as a bar

```text
|<-- 30 ms network+gateway -->|<- 10 auth ->|<- 15 model ->|<-- 65 ms features -->|
0                            30             40             55                   120
```

The 65 ms segment covers the slowest view, and sequential fetches add. Six views at 5 ms each are
30 ms serial but roughly 6 ms parallel (slowest reply plus fan-out coordination), which decides
whether a 120 ms budget holds. A percentile over 90 days is never 5 ms; it is a cache hit or a miss,
and the miss path needs its own line in the budget or a precomputed fallback value.

### 6.4 Staleness as a survival curve

```text
  share of
  reads
  1.00 |**************************************
  0.97 |                                   *    TTL 7 d, refresh 0.5/day -> 3.0% null
  0.83 |                             *          TTL 7 d, refresh 0.25/day -> 17.4% null
  0.50 |                    *
  0.00 +---------+---------+---------+-------->  age (days)
       0         2         4         6       8
```

Same TTL, different refresh rate, a 5.8x difference in null share. The curve makes visible that
"TTL = 7 days" is not a freshness policy without the arrival rate `r` and the action taken at the
boundary.

### 6.5 Cardinality versus QPS

```text
  QPS
  1e5 |                     *   hot-key / cache problem
      |            *
  1e4 |     *                   <- 25M entities x 6 views, 18k lookups/s
      |  *
  1e2 +--*------*------*------*----> entities
        1e4    1e6    1e8    1e10
```

`1e4` entities at `1e5` QPS is a caching problem; `1e9` entities at `1e2` QPS is a storage and
rebuild problem. Identical API, different engineering, different interview answer.

### 6.6 Where the four failure layers sit

```text
 events --> [L1 capture] --> [L2 window/agg] --> [L3 materialize] --> [L4 consume] --> decision
               |                    |                    |                   |
         lag histogram        row-set diff          parity residual      unit/default diff
         duplicate rate       count agreement       staleness hist.      feature-order diff
```

Each layer has one cheap observable that localizes a mismatch, ordered by collection cost: L4 is
free (read the serving log), L3 is one join, L1 and L2 need retained history. Diagnose in that
order and stop at the first layer that fails.

