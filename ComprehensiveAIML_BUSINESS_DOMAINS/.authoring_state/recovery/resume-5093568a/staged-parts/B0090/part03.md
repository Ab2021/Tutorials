### 6.5 Selection under capacity: why the ranking must be applied to admission, not to the promise

The capacity gate converts the problem from "promise to everyone the model likes" into "admit the
best `K_slot` candidates into the tier and route the rest". The quantity that ranking should order is
the expected value of admitting a line, not the probability that the line is fast.

```text
FORMULA [D-06]: EV_admit(i) = p_i * v_2h + (1 - p_i) * (v_late - c_comp)  -  mu_i * c_pick
WHERE:
  EV_admit(i) [USD]   expected net value of promising line i at P2
  p_i         [fraction] predicted probability the line is notified within 120 min if admitted
  v_2h        [USD]   margin attributable to a kept two-hour promise
  v_late      [USD]   margin retained on a late pick (customer usually still collects)
  c_comp      [USD]   expected service-recovery cost per broken promise
  c_pick      [USD]   labour cost per picker-minute
  mu_i        [min]   expected service time for line i
CONSTRAINT:
  p_i must be estimated on the deployment regime, not on the priority-treated training regime;
  the gate is applied per slot before ranking, and only lines passing the gate may be ranked.
NUMERICAL EXAMPLE (constructed, one line in slot A):
  p = 0.86, v_2h = 9.00, v_late = 7.20, c_comp = 4.50, mu = 6.6 min, c_pick = 0.22
  EV = 0.86*9.00 + 0.14*(7.20 - 4.50) - 6.6*0.22
     = 7.74 + 0.378 - 1.452 = 6.666 USD
  compare a low-p line: p = 0.55 -> 4.95 + 0.45*2.70 - 1.452 = 4.95 + 1.215 - 1.452 = 4.713 USD
INTERPRETATION:
  The spread between a strong and a weak candidate is 1.95 USD per line. The marginal pick costs
  1.45 USD. A line with p = 0.30 has EV = 2.70 + 1.89 - 1.45 = 3.14 USD and is still worth admitting
  when capacity is free, which is why the gate — not the ranking — decides who is excluded.
```

Note what this arithmetic does and does not say. It says that in the off-peak slots, where capacity
is not binding, a threshold on `p` throws away value: every candidate has positive expected value and
the correct policy is to promise P2 broadly. It says that in the peak slots, where the 105th
candidate cannot be picked, the marginal decision is a comparison among candidates rather than
against zero.

That distinction explains a pathology the case's authors observed in the constructed shadow log: a
global 0.90 threshold suppressed 52 off-peak promises whose expected value was strongly positive,
because the model's `p` in off-peak slots clustered at 0.83 due to a base-rate effect in training.
The threshold was not too high in an absolute sense; it was applied to a quantity whose scale shifts
by slot, and a single cut across heterogeneous slots is a specification error regardless of its value.

Boundary where D-06 breaks: the formula treats `p_i` as independent across lines in a slot, which is
false when the lines compete for the same picker. Admitting 104 lines with `p = 0.86` is not the
same as admitting 104 lines whose events are correlated through a shared queue. If the slot's
realised load exceeds `K`, every admitted line's `p` falls together, and the per-line products
overstate value by roughly the correlation term. The discriminating check is whether the slot's
realised pick-task start times cluster; if the standard deviation of `W_q` within a slot is below
8 minutes, the lines are effectively batched and the independence assumption is wrong.

### 6.6 Why the model's own score moves when the policy changes

This is the mechanism that produced the inversion in section 2, and it can be shown arithmetically
rather than argued.

```text
FORMULA [D-07]: p_hat(i) = P(ready <= 120 | x_i, quoted = P2)  !=  P(ready <= 120 | x_i, do(P2))
WHERE:
  p_hat(i)   [fraction]  model output, conditioning on the historical policy having quoted P2
  x_i        [vector]    permitted features at the snapshot
  do(P2)     [operator]  the intervention of promising P2 under the new decision rule
  quoted=P2  [event]     the selection event; the training cohort is conditioned on it
CONSTRAINT:
  equality holds only if quoting P2 does not change the pick treatment or the queue position.
NUMERICAL EXAMPLE (constructed):
  training cohort: only lines the ledger cleared were quoted P2. Under the ledger, a P2 line is
  flagged priority at task creation and is picked before PD and PN lines in the same slot.
  Suppose 62% of the cohort received priority treatment and their kept rate is 0.897, while the
  38% that were quoted P2 without priority have a kept rate of 0.801.
  blend check: 0.62*0.897 + 0.38*0.801 = 0.5561 + 0.3044 = 0.8605 ~ 0.860 (baseline rate)
  now apply the model to lines the ledger would NOT have admitted: those lines, under the new
  rule, enter the same slot without displacing priority lines, so they receive priority 0% of the
  time and their expected kept rate is closer to 0.801 than to 0.860.
INTERPRETATION:
  The model's calibration is a weighted average of two regimes. Applying it to a population whose
  regime weights are different biases every prediction by up to 6 points, in the optimistic
  direction, exactly where the decision is hardest.
```

The blend check is the important part: it reproduces the baseline's 0.860 from two sub-populations,
which demonstrates that the cohort is a mixture rather than a homogeneous set. Once you see the
mixture, the counterfactual bias is not mysterious. The model learned `0.62 * 0.897 + 0.38 * 0.801`
because that is what the data contained, and a new policy that changes the mixture weights inherits
an estimate for a mixture it no longer has.

The discriminating check for this failure is a stratification of the training cohort by whether the
line received priority treatment, and a comparison of the model's calibration curve in each stratum.
If the calibration curves differ — in the constructed build they differ by 4 to 7 points in the
0.75–0.90 band — then the score is regime-dependent and a threshold on it is a threshold on the
historical policy. If they coincide, the bias is absent and the model is safer to deploy.

Boundary where D-07's warning does not apply: a fully randomised promise experiment, in which tier is
assigned independently of the model, breaks the conditioning because `quoted` no longer predicts
treatment. That is the clean fix and it costs conversion on the randomised-off arm. The case does not
have such an experiment, which is why the adopted design uses a fixed yardstick rather than a
corrected probability estimate.

### 6.7 The residual after the gate: what remains for a model to earn

With mechanical lateness removed by the gate, the 46 residual late lines are the addressable set. The
constructed attribution splits them by first failing event, assigned by reading the event sequence of
each line rather than by a model.

| First failing event | Lines | Share of residual | Addressable by ranking? | Addressable by data accuracy? |
|---|---|---|---|---|
| Stockout at pick (on-hand wrong) | 22 | 0.478 | yes, by avoiding the promise | yes, by cycle counts |
| Picker absence mid-slot | 9 | 0.196 | no | no, roster control |
| Notification integration delay > 12 min | 8 | 0.174 | no | partially, by expediting |
| Pick error requiring a re-pick | 5 | 0.109 | no | partially, by validation |
| Customer arrived before notification retry | 2 | 0.043 | no | no |

Check the shares sum to 1.000: `0.478 + 0.196 + 0.174 + 0.109 + 0.043 = 1.000`. And check the counts:
`22 + 9 + 8 + 5 + 2 = 46`, consistent with the residual from D-03 after the 6-line spill correction
(73 observed − 27 mechanical = 46).

Only the 22 stockout lines are both large and addressable. Two interventions reach them. The first is
accuracy: a cycle count on the 40 highest-velocity SKUs per store-week moves the on-hand belief closer
to truth, and if it recovers half of the 22, the fixed yardstick rises to `(447 + 11)/520 = 458/520 =
0.8808`. The second is avoidance: a model that ranks lines by stockout risk lets the gate defer the
risky ones to P4 rather than promise them at P2, which does not move the fixed yardstick at all but
does move the per-tier rate — because the deferred lines no longer count as broken two-hour promises.

That second effect is worth isolating because it is the honest version of what the original model was
trying to do. Under the gate plus risk ranking that defers those 22 lines, the constructed result is
465 P2 promises and 447 kept, or `447/465 = 0.9613`, against the baseline's `447/520 = 0.8596`. The
kept *count* does not rise at all: nothing became faster, 22 risky promises were simply not made. The
first lever that raises the count is the cycle-count programme, and if it recovers all 22 lines the
result is 487 P2 promises with 469 kept, `469/487 = 0.9630`, and a fixed yardstick of
`469/520 = 0.9019`. So the count moves by 22 only when the shelf is fixed, and the rate moves by 10
points when the promise is honest, and confusing those two contributions is how a fulfilment
programme reports a victory it did not have. Every other late line was late for a reason a tier
decision cannot reach: 9 picker absences, 8 notification delays, 5 re-picks, 2 early arrivals.

This decomposition is the finding, and it is stated next.

## 7. The Finding

The supported conclusion is a three-part, conditional statement, and each part was earned by a
separate piece of arithmetic.

**First: the machine-learning model cannot be deployed as a per-line threshold, and the reason is
structural, not a matter of tuning.** `W_q` is 34% of the promise in the peak slot and is not a line
attribute (D-01). The 0.90 threshold therefore admits lines in proportion to a quantity the store
cannot control, and the constructed result is 468 promises with 389 kept, `0.831`, against a baseline
of 520 promises with 447 kept, `0.860`. The model's discrimination is real — AUC 0.81, Brier 0.147 —
and it is applied to the wrong object.

**Second: the capacity gate is what makes the promise honest, and it buys no speed whatsoever.**
Applying `K_slot` from D-02 removes 33 promises that arithmetic had already doomed: 104 admitted of
128 in the 10:00–12:00 slot, 77 of 86 in the 16:00–18:00 slot, all 306 off-peak lines unchanged.
The P2 rate rises from `447/520 = 0.8596` to `447/487 = 0.9179`, a gain of 5.83 points, achieved by
promising fewer people rather than by picking faster. The fixed yardstick is unchanged at 0.8596,
which is the proof that the gain is definitional rather than operational. That is still worth having:
a promise kept 92% of the time is a better basis for a customer relationship than one kept 86% of the
time, and the 33 deferred customers are told the truth at quote time rather than disappointed later.

**Third: the model earns its place only when it is used to choose which lines not to promise, and the
size of that prize is 22 lines, not 73.** Of the 46 residual late lines, 22 fail on a stockout. Used
this way the ranker withholds those 22 promises, and the constructed result is 465 P2 promises with
447 kept, `447/465 = 0.9613` — a rate gain with no count gain, because withholding a bad promise does
not create a good one. The count moves only when the shelf is fixed: a cycle-count programme that
recovers all 22 gives 487 promises with 469 kept, `469/487 = 0.9630`, and lifts the fixed yardstick to
`469/520 = 0.9019`. So the honest attribution is 6.11 points of promise integrity from the gate and
the ranker together, and 4.23 points of speed from a person with a scanner.

The conclusion is conditional on four things, and it is withdrawn if any of them fails. Capacity must
be computable from the roster with the `L_eff` correction — if the store cannot state `L_eff`, the gate
is a guess and should not be deployed. The ready timestamp must be the notification time; if the store
can only observe staging scans, every rate in this case is overstated by the median staging-to-notify
gap. The tier mix must be reportable, because otherwise a policy that defers everything will look
excellent. And the abandonment cost of deferral must remain below the recovery cost of a broken
promise; the break-even is computed in the next paragraph.

Break-even on deferral, derived rather than asserted. Let `delta` be the abandonment probability for
a line moved from P2 to P4, `m` the margin on a collected order, and `c` the recovery cost avoided by
not breaking a promise. Deferring is worthwhile when `delta * m < c`, so the break-even abandonment is
`delta* = c/m`. With `c = 4.50 USD` and `m = 12.00 USD`, `delta* = 0.375`. The constructed funnel shows
a 3.8% abandonment rate on deferred lines, well under the 37.5% break-even, so the decision is robust
to a tenfold error in the abandonment estimate. If margins were thin — say `m = 5.00 USD` — the
break-even would be `0.90`, and any meaningful abandonment would reverse the recommendation. This is
the kind of sensitivity a manager can act on: the decision does not depend on precise funnel
measurement at current margins, and it does depend on them at low margins.

The claim that is *not* made: that this design was deployed, that it produced these numbers in
production, or that any retailer's promise accuracy is described here. The numbers are constructed,
the arithmetic is verified, and the structure of the argument is the transferable part.

Takeaway: separate the three questions — who should be promised, how many can be promised, and how
fast can the store actually pick — because collapsing them into one model score is what produced the
inversion.

## 8. System/Solution Design

The design has four components and one rule connecting them: the gate runs before the model, never
after it.

```text
                    +-------------------------------+
  [order line] ---> | 1. snapshot assembler         |  as-of join on t_quote
                    |    publish snap_id, refuse    |  lineage manifest check (C-01)
                    |    post-quote columns         |
                    +---------------+---------------+
                                    |
                                    v
                    +-------------------------------+     roster v41 ----+
                    | 2. slot capacity resolver     |                   |
                    |    K_slot = floor(P*L_eff/S)  |<------------------+
                    |    counts carried-in promises |     n_mean from quoted lines
                    +---------------+---------------+
                                    |
                        gate: slot_load < K_slot ?
                          |                        |
                          no                       yes
                          |                        |
                          v                        v
              +-------------------+   +-------------------------------+
              | route to next tier|   | 3. risk ranker (GBM)          |
              | re-run gate there |   |    orders candidates by       |
              +-------------------+   |    stockout risk, not by p    |
                          |           +---------------+---------------+ 
                          |                           |
                          v                           v
              +---------------------------------------------------+
              | 4. decision writer: tier, snap_id, gate inputs,   |
              |    rank, one-sentence rationale, revert handle    |
              +-------------------+-------------------------------+
                                  |
                                  v
                        [decision log, append-only]
                                  |
                                  +--> [nightly metric job: per-tier, pooled, fixed yardstick]
                                  +--> [review queue, capped at 40/store-day]
```

**Component 1 — snapshot assembler.** Its only job is to produce a frozen input set and refuse to
produce one that violates the lineage rule. The refusal is a build failure, not a warning, because a
warning is a thing people scroll past. The manifest is a list of `(column, source table, validity
start, permitted)` tuples, and the check is `max(validity_start) <= t_quote` evaluated per decision
batch, with an exception list that is empty in the shipped configuration.

**Component 2 — slot capacity resolver.** It owns three numbers per slot: `P` from the published
roster, `L_eff` from a configured availability factor with a per-store override, and `n_mean` from
the lines already quoted into that slot. It counts promises *landing* in the slot, including those
quoted in the previous ninety minutes whose deadlines fall inside it. It exposes `K_slot` and the
current `slot_load` and its verdict, and it is the only component allowed to answer "can this slot
take another P2 promise".

The resolver's failure behaviour is the design decision worth stating. If the roster is missing for a
slot — a late publication, a store-level override — the resolver does not default to the previous
day's roster, because yesterday's roster is a guess dressed as data. It returns `K_slot = 0` for that
slot, which routes every candidate to the next tier, which means the store promises P4 rather than
gambling on P2. That is a deliberate degradation: the customer gets a true four-hour promise instead
of a 74%-likely two-hour promise, and the store avoids the recovery cost. The resolver logs the
degradation reason so the next morning's review can see how many slots were degraded.

**Component 3 — risk ranker.** This is the model, used narrowly. Its output is an ordering of
candidates within a slot by the probability that the pick will fail or be slow for a reason the
promise could avoid — stockout risk, substituted-line risk, bulky-item risk. It does not set the tier;
it decides which candidates occupy the scarce admissions. Two properties follow: the model can be
replaced without touching the gate, and the model can be refit on a population that is not selected
by the promise, because its target is "pick failed or re-picked", which is observed for every line
regardless of what tier was promised.

That second property is the quiet fix for the D-07 bias. A target defined on the pick event rather
than on the promise event is not conditioned on the policy, so the counterfactual bias that broke the
first design cannot enter through the label. It does not eliminate all regime dependence — a
priority-treated line is still picked sooner — but it removes the specific conditioning that made
the score a restatement of the historical rule.

**Component 4 — decision writer.** It emits one row per line containing: `snap_id`, `t_quote`,
`gate_K`, `gate_load`, `gate_verdict`, `rank_score`, `tier_chosen`, `rationale` (a template filled
from the gate inputs and the top two risk drivers), and `revert_handle`. The rationale template is
deliberately mechanical — "P2 granted: slot 10:00 load 91 of 104, stockout risk 0.12, two lines" —
because a rationale a human cannot reproduce from the log is not an explanation.

Two invariants are enforced in the writer rather than in a downstream check, because a downstream
check reports a violation while a writer prevents one:

```text
INV-1  per effective window: count(effective actions per decision_id) <= 1
       enforced by a unique constraint on (decision_id, effective_window)
INV-2  per slot: slot_load_after <= gate_K, unless gate_verdict = 'degraded'
       enforced by a conditional write that fails if the increment would exceed K
INV-3  every action row references a snap_id present in the snapshot registry
       enforced by a foreign key, which also blocks replay against a rebuilt snapshot
```

INV-2 is the interesting one. A conditional write that fails when the load would exceed `K` turns a
race between two concurrent quotes into a retry rather than an oversubscription. Without it, a
resolver that reads `slot_load = 103, K = 104` and is called twice concurrently will admit both
candidates and exceed the gate by one. At 4 pickers this is a rounding error; at 40 concurrent quote
requests it is the difference between a gate and a suggestion. The constructed store sees about 9
concurrent quote requests per second at peak, which is enough to make the race routine.

Recovery and replay:

| Failure | Detection | Behaviour | Replay check |
|---|---|---|---|
| Snapshot build fails lineage | manifest assertion throws | batch blocked, alert to steward, quotes served by incumbent ledger | rebuild from raw and diff |
| Roster missing for a slot | resolver returns `K = 0` | all candidates degrade to next tier | count degraded slots per day |
| Ranker unavailable | health probe fails | candidates ordered by a static risk table (bulky, low-velocity, promo) | compare admission sets |
| Decision writer unavailable | write timeout | order held, customer shown "confirming" for up to 20 s, then P4 default | count holds and defaults |
| Metric job fails | no output by 05:00 | previous day's report re-published with a staleness banner | compare cohort sizes |
| Reversal requested | operator action | writer emits compensating row, never an update | confirm both rows survive |

The replay check for each row is the difference between an operable system and a system that seems
operable. A reversal that updates in place destroys the evidence of the original decision and makes
the override-rate metric uncomputable; a reversal that appends a compensating row preserves both.

Takeaway: put the capacity gate in front of the model and the invariants in the writer, because a
gate applied after ranking cannot undo a promise that has already been shown to a customer.
