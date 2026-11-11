## 5. Data, Tools & Method

### 5a Data Specification

**Decision grain.** One row is a *budget decision instance*: a channel-by-quarter allocation choice,
keyed `(quarter_id, channel_id)`, evaluated at a single decision instant `t_dec` = the last day of the
preceding quarter at 23:59:59 in the retailer's trading timezone. With seven channels and four
quarters of history there are 28 decision rows, but only one of them is *live* at a time; the other
27 are replay rows used to check that the procedure would have produced a defensible plan in each
past quarter. This distinction matters for leakage: a replay row may legitimately read outcomes that
the live row could not, but it must be scored with the rules of its own instant, not with today's
rules.

**Supporting grains and keys.** Five tables, each with one grain and one key discipline:

| Table | Grain | Key | Units | Earliest availability |
|---|---|---|---|---|
| `orders` | one settled web or store order | `order_id` | currency, gross then net of returns | settlement + 1 day |
| `touchpoints` | one impression, click, open, or visit | `touch_id` | count, cost | event + 0 to 48 hours |
| `paths` | one ordered touch sequence ending in a conversion or a null | `path_id` | attribution weight (pure number) | last touch + window |
| `spend` | channel by day | `(channel_id, date)` | currency, invoiced | invoice + 3 days |
| `channel_params` | channel by quarter | `(channel_id, quarter_id)` | iROAS multiple, elasticity, cap multiple | estimated after quarter close |

**Population.** Adults with at least one identifiable session in the 90 days before the decision
instant, resident in the trading country, excluding staff accounts, bots flagged by the traffic
filter, and orders cancelled before settlement. The population is deliberately stated as *identifiable
sessions*, not *customers*, because the identity layer cannot see the difference for a large share of
traffic (§2.2) and pretending otherwise inflates the denominator of every per-customer metric.

**Cut-off.** `t_dec` is the hard cut. Any record whose *processing* timestamp exceeds `t_dec` is
excluded even if its *event* timestamp precedes it, because the decision could not have seen it. This
is the single most consequential line in the data specification and the one most often relaxed "just
this once" during a live quarter.

**Units and currency.** All money in thousands of the trading currency (\$k), gross of tax, net of
known returns where the column says `net`. Revenue is recognised at order placement for the ledger and
at collection for the accrual; the two are never mixed inside one table. Elasticities are unitless
multipliers on a log-log spend curve. Caps are multiples of current spend and therefore unitless.

**What is not available at decision time — the exclusion list.** Seven items, each of which is
available to a retrospective analyst and unavailable to the decision rule:

1. Loyalty-scanned store collections that link a web browse to an in-store purchase (§2.2).
2. Return events, typically arriving 14 to 35 days after the order, which reduce recognised revenue
   after the attribution window closes.
3. The current quarter's actual spend, which is a decision output rather than an input.
4. Competitor promotional calendars, which are reconstructed after the fact.
5. Platform algorithm or auction changes, which are announced after they are deployed.
6. The margin rate for the *current* quarter if finance has not yet re-based it.
7. Any channel parameter estimated on data that includes the quarter being decided.

**Derived quantities and their provenance.** Three derived columns carry the analysis and each is
built from a distinct source with a distinct failure mode:

- `credited_revenue` per order sums the attribution weights of all touches that earned credit,
  multiplied by order value. It can exceed order value; that is the multiplicity defect.
- `incremental_revenue` per channel comes from the parameter table, which comes from prior
  experiments, not from the ledger.
- `settled_revenue` comes from the payment settlement file and is the only column that ties to the
  bank.

**Size and sparsity.** The live quarter has 42,600 settled orders, 48,300 credit events, 158,000
journeys in the path table of which 15,800 end in a conversion, and roughly 120 million display
impressions. The path table is sparse by construction: of the 15 possible ordered subsets of three
channels, six occur with meaningful volume, and the remaining nine sum to under 0.5% of journeys.
Sparsity is not a nuisance here; it is the reason Shapley and Markov agree so closely on this data
(§6.3) and the reason they diverge sharply once repeated touches are introduced.

**Privacy and identity constraints.** The identity graph is used to *count* journeys, never to score
individuals, and no output column is at customer grain. Where a link depends on a loyalty scan, the
journey is included only if the scan preceded the conversion — a rule that costs roughly 9% of
otherwise attributable conversions and buys temporal validity, which is the correct trade in a
measurement instrument that will be used to move money.

### 5b Tools & Stack

The stack is described by *function*, not by vendor, because the functions are what the case depends
on and the vendors are interchangeable. Seven components:

1. **Versioned extract.** A snapshot with an immutable identifier (`snap_id`) containing every table
   as of `t_dec`. Reproducibility is a property of the snapshot, not of the query text; a query rerun
   tomorrow against a live table is not a reproduction.
2. **Event store.** Append-only log of touchpoints with both event and processing timestamps. The
   append-only property is what makes the leakage audit in §5d mechanically checkable rather than a
   matter of trust.
3. **Path builder.** Deterministic, sessionised, order-anchored. Given the same events and the same
   window it must emit byte-identical paths; if it does not, no downstream comparison is meaningful.
4. **Parameter store.** Channel-level iROAS, elasticity, and cap values with interval bounds and a
   `source` column naming the experiment that produced each one. An empty `source` means the value is
   a prior, and priors are labelled as such wherever they are printed.
5. **Allocation solver.** A constrained optimiser over the seven spend multiples with the cap
   register as hard bounds; returns the corner solutions as well as the central one.
6. **Append-only decision log.** One row per committee decision: instant, plan, parameter version,
   dissent, and the checkpoint date. This is what makes §10 auditable and §12 diagnosable.
7. **Reporting layer.** The one-page recommendation (§3) plus the appendix tables. No interactive
   dashboard for the decision itself, because an interactive surface invites re-litigation of a
   locked plan.

Deliberately absent: a real-time bidding integration, a customer-level scoring service, and any
automated budget-shifting loop. Each would be appropriate for a different problem; for a quarterly
allocation decision they add failure surface without adding decision quality, and the last one in
particular would let the instrument move money faster than the committee can audit it.

### 5c Method Summary

**Baseline.** Last-click credited revenue, 28-day window, de-duplicated by order, refreshed nightly.
Retained not as a straw man but as a real alternative: it is the number the trading team reconciles
weekly and the number the plan must beat on the committee's own terms.

**Candidate methods, in increasing order of causal ambition.** Four were considered and one was
adopted in staged form:

- **Heuristic re-weighting** (position-based, time-decay). Cheap, decomposable, and unfalsifiable:
  any credit split can be justified by choosing the decay parameter, so it cannot be wrong and
  therefore cannot be right. Rejected as a decision basis; retained as a sensitivity check.
- **Cooperative-game allocation** (Shapley value over channel subsets). Principled, symmetric among
  channels, and decomposable, but it allocates *explained* conversions, not *caused* ones: it answers
  "how much of the observed total is attributable to this coalition" and not "what would happen if we
  turned it off".
- **Markov removal effect.** Closer to the causal question: remove a channel from the transition
  matrix and measure the drop in absorption probability. Still observational, still assumes the
  transition matrix is invariant to the removal, and it inherits whatever selection produced the
  observed paths.
- **Design-based measurement** (geo-matched holdouts, user-level holdouts, ghost bidding). The only
  family that identifies a counterfactual rather than assuming one. Expensive, slow, and often
  underpowered — §9 shows exactly how underpowered for this budget.

**Adopted procedure.** A three-layer decision: (i) a scale correction from the ledger, which is
arithmetic and needs no experiment; (ii) a sign test per channel from existing interval estimates,
which needs arithmetic and judgement; (iii) a magnitude test only where the sign is robust, using a
capped response curve and a design-based check where one is affordable. The layers are ordered by
cost, so the cheapest evidence is spent first and the most expensive evidence is reserved for the
moves that actually carry the decision.

**Why not simply run the experiment for everything.** Because the budget cannot buy the sample. §9
computes that detecting the plan's total revenue effect on total company revenue needs on the order of
1173× the geo pairs available at the achievable variance, and detecting the *email* channel's
contribution effect at customer level needs a holdout so large that the test costs more than the
channel. A measurement programme that demands experiments it cannot power produces either no decision
or a decision dressed as one; the layered procedure is the honest alternative.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance chain.** For every printed number the case can name: source table, snapshot identifier,
transformation, and the parameter version that supplied any coefficient. Four timestamps are retained
on every record — event, processing, effective (when the record became authoritative for reporting),
and correction (when a later record superseded it). A record with a correction timestamp before
`t_dec` must be replaced, not added to; double-counting corrections is the most common silent
inflation in a nightly-refresh ledger.

**Quality dimensions and the tests used.** Rather than a single "data quality score", five
independent checks, each with a pass threshold:

- *Completeness*: share of settled orders with at least one touch. Observed 61.4%; unattributed
  orders are the residual and are *not* distributed at random across channels.
- *Uniqueness*: duplicate `order_id` rate after settlement. Threshold 0.02%; duplicates above that
  indicate a re-sync, and a re-sync doubles credited revenue without doubling sales.
- *Timeliness*: median processing lag by channel. Display 4 hours, search 2 hours, affiliate 26
  hours, store collection 3.5 days. Any channel whose lag exceeds the window boundary is
  automatically truncated and must be flagged.
- *Consistency*: platform revenue versus ledger revenue, monthly. Tolerance 1.5%; a breach is a data
  incident, not a modelling tweak.
- *Validity*: impossible values — negative spend, conversion before first touch, order value of zero —
  counted and excluded with the rate reported.

**Exclusions and their rates.** Exclusions are never silent. Each is counted, and the count enters the
denominator of a stated coverage statistic: staff and bot traffic (3.1% of sessions), orders
cancelled pre-settlement (2.8%), journeys whose only touch is inside the loyalty-link gap (9.2% of
otherwise attributable conversions), channels with fewer than 500 conversions in the window (none in
this quarter), and spend rows without an invoice (0.4%). The exclusion set is stable across quarters,
which is what allows the same procedure to be replayed; a changing exclusion set quietly changes the
denominator and therefore the answer.

**Leakage, enumerated with a check per channel.** Leakage in this domain is not one bug; it is a
family of six, and each has a distinct discriminating test:

1. *Post-decision linkage.* Store collection data arrives after `t_dec` and links a browse to a
   purchase. Check: recompute the identity link using only records with processing time ≤ `t_dec`;
   the difference in conversions is the leak's size.
2. *Outcome-informed features.* A feature derived from the outcome — "did the customer return this
   item" — enters the model as if it were a predictor. Check: feature-availability audit against
   `t_dec`, one row per feature, signed off by the data owner.
3. *Window-boundary truncation.* Long-lag touches are cut at 28 days, so early-funnel channels lose
   credit systematically. Check: re-run at 7/28/90 days on a fixed cohort; the credit share that
   moves is the truncation sensitivity.
4. *Replay contamination.* A historical replay uses parameters estimated on data that includes the
   replayed quarter. Check: parameter timestamps must precede each replay instant.
5. *Correction double-count.* A superseded revenue record is added rather than replaced. Check: sum
   `correction` rows and compare to the reported correction rate.
6. *Selection on attribution.* Journeys that cannot be attributed are dropped, so the analysable
   population is the well-tracked one. Check: compare conversion rate of attributed versus
   unattributed orders; a gap means the dropped set is not exchangeable.

**Reproducibility artefact.** The deliverable includes the snapshot identifier, the parameter version,
the solver configuration, and the exact exclusion list with counts. A reader with the same snapshot
and the same parameters must obtain the same plan; a reader with a different snapshot must obtain a
plan that is comparable in structure even if the numbers move, which is the actual test of whether the
*procedure* is stable as opposed to whether one run is reproducible.

### 5e Reproducibility & Method-Selection Ledger

The ledger records what was chosen, what was rejected, and why — because the rejected options are
where the review will focus.

| Decision | Chosen | Rejected | Reason | Reversal condition |
|---|---|---|---|---|
| Credit allocation | Shapley over channel subsets | Last click; linear; time decay | Symmetric, decomposable, no free parameter to tune | Repeated-touch paths exceed 15% of volume (see §6.3) |
| Causal layer | Removal effect plus interval arithmetic | Pure Shapley | Shapley answers "explained", the decision needs "caused" | A design-based estimate exists for the channel |
| Scale correction | Ledger decomposition into multiplicity and unattributed | Single blanket inflation factor | A single factor hides opposite-signed errors | Settlement reconciliation shows a stable single factor |
| Curve form | Power curve with capped range | Linear response; polynomial fit | Monotone, one elasticity, capped at observed range | Spend history spans the proposed range |
| Uncertainty | Interval bounds on iROAS and elasticity, propagated by corners | Analytic variance propagation | Corners are auditable and conservative | Joint distribution of parameters is estimated |
| Plan selection | Maximin over corners with point-estimate reporting | Expected-value maximisation | Decision is quarterly and largely irreversible | Moves become cheaply reversible |
| Test design | Geo-matched holdout where powered; user holdout for brand | Channel-level A/B on platform | Platform lift studies cannot hold out a channel | Platform provides a true holdout |
| Reporting | One page plus appendix | Interactive dashboard | Locks the decision for the quarter | Cadence changes to monthly |

**Replay results as a stability check.** Applying the procedure to the four prior quarters produces
plans whose recommended spend multiples move by 8% to 14% between quarters, while the *direction* of
every move is stable in three of four quarters and the fourth flips only for social retargeting, whose
interval straddles the decision boundary in both directions. That flip is reported rather than
smoothed: a procedure that appears perfectly stable across replays is usually one that has been fitted
to the replays.

**Environment pinning.** Solver tolerances, rounding conventions (half-up at the currency unit, never
inside an intermediate), and the order of operations in the profit formula are fixed and documented,
because the profit delta in this case is small enough (\$352k on \$5,027k of revenue, 7.0%) that a
rounding change inside a loop can move the reported figure by a measurable amount. Every intermediate
in §6 and §11 is printed at full precision so the reader can find a disagreement rather than having to
trust one.
