---
case_id: "CASE-04-02-04"
case_title: "Cross Channel Attribution"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "commercial measurement lead"
source_index: "INDEX.md"
review_trigger: "Any change to the attribution window, the credit rule, the contribution-margin rate, the channel cap set, or the review capacity of the budget committee"
---

# Cross Channel Attribution

**Evidence status.** Every figure in this case is **ILLUSTRATIVE**: a constructed ledger, a
constructed path table, and constructed channel economics built to be internally consistent and
arithmetically re-derivable by a reader with a calculator. Nothing here is a measured result for any
retailer, agency, platform, or vendor, and no company architecture is described or implied. The
methods named — Shapley value allocation, Markov-chain removal effect, geo-matched holdouts, ghost
bidding, adstock transformation — are published techniques; the parameters attached to them in this
file are teaching inputs, not estimates from a live account. Where a number is the output of a
formula in this document, the arithmetic is shown so it can be checked; where a number is an input
chosen to make the example instructive, it is called an input. The case is `proposed`, not
`verified`, and carries no claim of deployed impact.

One structural warning before the reader begins. This case is about a measurement instrument that
was arithmetically correct and decisionally wrong. Every ratio in the first two sections computes
exactly, every table foots, and the plan the instrument recommends destroys value. The failure is
not a bug in a query. It is that the instrument answers "which touch gets credit" while the
committee is deciding "which dollar buys another dollar of contribution", and those two questions
have different denominators, different populations, and different notions of the word *effect*.

## 1. The One-Line Framing

A retail media budget of \$1,310k per quarter is allocated by last-click credited revenue, a metric
that is internally consistent, fully auditable, and inflated by a factor of 2.234 relative to
incremental contribution — so the honest decision is to cut spend by a third while raising
contribution profit by 8.9% of revenue, which requires the committee to accept a 5.6% decline in the
very number they are paid to grow.

Three things move together in that sentence and must be kept apart for the rest of the case. The
**ledger** is credited revenue, a bookkeeping object: it sums to a number larger than actual sales
because one order can be credited to several channels. The **estimate** is incremental revenue, a
causal object: it exists only relative to a counterfactual that was never observed. The **decision**
is a spend plan, an operational object: it must survive the worst plausible parameter corner, fit
inside channel caps, be reversible inside a quarter, and be explainable to a sceptical brand owner
in one page. Last-click serves the first object well and the third object badly, and the temptation —
which this case exists to make visible — is to treat the first object's precision as evidence about
the second.

The reusable pattern is **evaluation-mismatch**: a convenient score whose optimisation does not
improve the operational objective. The score here is credited revenue per dollar; the objective is
contribution after media cost. Optimising the score moves money toward channels that appear late in
purchase paths (brand search, retargeting, coupon affiliates), because late touches are
mechanically over-credited, and away from channels that appear early (prospecting, display), because
early touches are mechanically under-credited. The recommended plan therefore spends more on people
who were already going to buy and less on people who were not, and it reports a rising ROAS while
contribution falls. A ROAS that rises as the business shrinks is not a measurement success.

The output of the work is consequently bounded and modest: a staged reallocation with two channels
expanded under a cap, three cut, two held pending evidence that cannot be bought this quarter, and an
explicit statement of which decisions remain undecidable with the budget available. The last clause
is the part reviewers most often delete, and it is the part that makes the other clauses honest.

## 2. Context & Background

### 2.1 The commercial situation

The retailer sells through stores, a web storefront, and a marketplace listing, and it funds seven
media lines from one quarterly budget: non-brand paid search, brand paid search, social prospecting,
social retargeting, programmatic display, affiliate and coupon placements, and owned email. The
budget is reviewed quarterly by a committee of three: the commercial measurement lead (who owns this
case and the instrument), the head of digital trading (who owns the channel targets and is measured
on revenue), and the finance business partner (who owns the contribution-margin rate and signs the
accrual). Store operations is consulted but does not vote; its interest is that media-driven web
demand spills into store collection, and any plan that raises web orders raises store workload.

Two governance facts create the tension. First, the trading scorecard measures *credited* revenue by
channel, because that is what the platforms report and what the trading team can reconcile weekly.
Second, the finance accrual measures *total* revenue and *contribution* at the company level,
because that is what the statutory accounts need. Nobody in the room is measured on incremental
contribution by channel, and that is the vacuum into which the attribution model was installed. An
instrument becomes authoritative when it is the only number in the room, not when it is the best
number in the room.

### 2.2 Why omnichannel makes this harder than single-channel retail

In a pure e-commerce account, a purchase is a session and the path is a cookie. In an omnichannel
account, three complications sit on top of each other and each one bends the credit rule in a
different direction.

**Identity is partial and time-varying.** The same person may browse on a phone without signing in,
add to a basket on a laptop while signed in, and collect in a store where the only identifier is a
loyalty scan at the till. The identifier that links the first two touches to the third often becomes
available *after* the purchase — at the loyalty scan — which means the linkage is knowable to the
analyst and not knowable to the decision rule at the moment the decision is made. Any feature or
credit assignment that relies on it is temporally leaked.

**The conversion event has several legitimate definitions.** A web order is a conversion. A store
collection of a web order is the same conversion recorded twice if both systems count it. A store
walk-in sale influenced by a nearby-search impression is a conversion with no digital event at all.
The choice among these definitions changes the denominator of every rate in the model, and the
choices are made by different teams for different reasons, so the instrument inherits a mixture.

**Fulfilment latency separates event time from revenue time.** A basket reserved online and collected
in a store four days later produces a digital conversion at reservation and a revenue event at
collection; a return three weeks after that reverses part of the revenue after the attribution window
has closed. Credited revenue is therefore recognised earlier and more gross than settled revenue, and
the gap is a function of the channel mix rather than of media performance.

### 2.3 The instrument that was in place

The incumbent instrument is a last-click model with a 28-day lookback, de-duplicated by order rather
than by customer, refreshed nightly, and reconciled to platform-reported revenue within 1.5%. It is
a genuinely good bookkeeping system. Its properties are worth stating precisely because they are the
reason it survived: it is deterministic (the same order yields the same credit tomorrow), it is
decomposable (channel totals foot to the grand total by construction), it is fast, and it explains
itself to a brand owner in one sentence — "this channel was the last thing the customer touched".

Those four properties are exactly the properties of a *ledger*, and none of them is the property of
an *estimator*. Determinism is unrelated to unbiasedness. Decomposability means the parts sum to the
whole, which for a causal quantity is not a virtue but a coincidence: incremental revenue is not
conserved across channels the way credited revenue is, because removing two channels at once does not
remove the sum of their individual effects when they interact. Speed is a property of the computation,
not of the answer. Explainability is a property of the sentence, not of the estimate. The committee
had four good reasons to trust the number and no reason to ask what it measured, which is how a
ledger becomes a decision rule without anyone deciding that it should.

### 2.4 The two clocks

Every figure in this case has two timestamps and the analysis collapses if they are confused.

**Event time** is when the thing happened: the impression served, the click registered, the order
placed, the parcel collected, the return booked. **Processing time** is when the record became
available to the system: the nightly batch that loaded the platform report, the settlement file that
arrived from the payment processor, the store system that synced the collection. The attribution
window is defined on event time (28 days), the extract is defined on processing time (last complete
night), and the reconciliation is defined on neither (month-end close). A record with a late
processing time and an early event time is the archetypal leak in this domain: it belongs to a past
window, so it looks like history, but it was not visible when the decision was taken, so using it is
a look-ahead.

### 2.5 What the committee actually asked for

The request that opened the work was narrow and specific: "tell us which channel to move the
incremental \$200k to". That framing presupposes that the answer is a channel. The work found that
the answer is a *set of moves with different confidence levels plus two open questions*, and that the
\$200k was not incremental at all — it was already inside a budget whose marginal allocation was
negative in three lines. Restating the question consumed more calendar time than answering it, which
is normal in this domain and worth saying out loud, because a measurement lead who silently answers
the asked question produces a plan that gets rejected in the second meeting.

## 3. The Business Question

**Primary question.** Given the credited-revenue ledger, the causal estimates that can be bought
inside one quarter, and the committee's cap and reversibility constraints, which spend plan should be
adopted for the next quarter — and what is the plan's worst-case contribution, not only its central
estimate?

**Decomposition into five answerable sub-questions.** The primary question is not one decision. It
splits into five, and each has a different evidence requirement:

1. *Scale check.* Is the credited total an unbiased estimate of total revenue? Answerable to
   within a fraction of a percent from the ledger itself, with no experiment.
2. *Allocation check.* Which channels have a marginal contribution per dollar whose sign is stable
   across the plausible parameter range? Answerable by interval arithmetic on existing estimates.
3. *Magnitude check.* For the channels where the sign is stable, how far can spend move before the
   marginal turns negative? Answerable only where a response curve has been identified over the
   proposed range.
4. *Test check.* For the channels where the sign is not stable, what experiment would decide it, and
   how large must that experiment be? Answerable by a power calculation, and frequently the answer is
   "larger than the traffic available".
5. *Governance check.* Does the adopted plan violate a cap, a brand-safety constraint, a contractual
   commitment, or the review capacity of the committee? Answerable from the constraint register.

**Owner and cadence.** The commercial measurement lead owns the recommendation. The committee
decides quarterly; the plan is locked for the quarter and re-opened only on a documented trigger
(a cap breach, a platform policy change, a margin-rate change, or a variance larger than the
pre-registered tolerance). The locking matters: a plan that can be re-litigated weekly is a plan
nobody is accountable for.

**Available actions.** Four, and the fourth is the one that requires courage:

- *Adopt* a reallocation within existing caps, with the point estimate as the central case.
- *Stage* the reallocation, taking the moves whose sign is robust now and deferring the rest to a
  pre-registered checkpoint.
- *Hold* the current allocation and spend the quarter buying evidence on the undecidable channels.
- *Defer* — publish no change and state which evidence is missing, which is a legitimate answer and
  is not the same as doing nothing, provided the deferral is logged with an owner and a date.

**Objective function.** Contribution profit over the quarter,
`profit = m · R_incr − S`, where `m` is the contribution-margin rate (a company input, 0.55 in this
case), `R_incr` is incremental revenue attributable to the media plan, and `S` is media spend. This
is deliberately *not* revenue and *not* ROAS. Revenue enters only through the margin rate, which is
why a plan can raise revenue and lower profit or the reverse.

**Hard constraints.** Five, all binding, none negotiable inside the quarter:

1. No data timestamped after the decision instant may enter the estimate (§5d).
2. Channel spend stays inside the cap register: non-brand search ≤1.8× current, email ≤1.5×,
   social prospecting ≤0.2×, display ≥0 (can go to zero), brand search ≥0.5× (a contractual brand
   term), retargeting and affiliate uncapped in either direction.
3. Total spend may not exceed the approved envelope; unspent budget returns to finance and is *not*
   available for reallocation in the same quarter.
4. Any cut above 40% of a channel's current spend requires a named owner's written acceptance of the
   revenue consequence, because the trading scorecard will show it.
5. Review capacity is 18 candidate moves per quarter. Moves beyond capacity are deferred, not
   compressed.

**Out of scope.** Vendor selection, contract renegotiation, the correctness of the margin rate
itself, the design of the trading scorecard (recommended but not owned here), legal or privacy
determinations on identity resolution, and any claim about realised financial outcome.

**Answer format required by the committee.** One page: the adopted plan as a table of seven numbers,
the profit delta versus the status quo with its interval, the two or three moves that carry the
decision, the list of what was cut and why in one clause each, and the one paragraph naming what
remains unknown. A twenty-page deck is not an answer; the one page is the deliverable and the deck
is the appendix that defends it.

## 4. Assumptions & Constraints

Each assumption below carries an owner, a test that could falsify it, and the fallback that applies
if it fails. An assumption without a fallback is a wish, and the committee is entitled to treat it as
one.

| ID | Assumption | Why it is load-bearing | Test that could falsify it | Fallback if falsified |
|---|---|---|---|---|
| A-01 | Credited revenue exceeds actual revenue only through credit multiplicity and unattributed orders | The whole inflation factor rests on this decomposition | Reconcile attributed total against the settlement file for four consecutive weeks | Report attributed and settled side by side; never quote a ratio across the two |
| A-02 | Contribution-margin rate is constant at 0.55 across channels and weeks | Makes channel profit comparable on one scale | Recompute margin by category; compare weighted rate to the plan's mix | Use channel-specific margins and re-rank; expect email and affiliate to move most |
| A-03 | The 28-day window captures the decision-relevant lag for search and social | Sets which touches can earn credit | Compare credit shares at 7, 28 and 90 days on the same cohort | Extend the window for considered categories; accept the extra latency |
| A-04 | Channel-level lift transfers to the proposed spend range | Justifies moving spend 1.2× to 1.8× | Check the response-curve fit inside and outside the observed range | Cap at the top of the identified range; never extrapolate the curve |
| A-05 | Geo markets are exchangeable after matching on baseline sales and region | Makes the geo holdout a valid counterfactual | Placebo test: run the assignment on a pre-period with no campaign | Abandon the geo design for that channel and report it as unmeasured |
| A-06 | Loyalty linkage is available at decision time only for signed-in sessions | Prevents the largest leak in the identity layer | Audit what fraction of conversions gain their link post-conversion | Drop linked-only paths from features; accept a smaller analysable population |
| C-01 | Committee reviews at most 18 moves per quarter | Makes the plan executable, not merely optimal | Count moves in the last two adopted plans | Prioritise by robustness margin; defer the tail with a dated checkpoint |
| C-02 | Cuts are reversible inside one quarter without penalty | Makes staged expansion safe | Read the platform insertion orders for commitment terms | Convert staged expansion into a single-step move where reversibility fails |
| C-03 | Platform-reported revenue is reconciled to the ledger, not to the bank | Keeps the ledger auditable | Compare platform totals to settlement monthly | Treat platform revenue as an index, not a level |
| C-04 | The budget envelope is fixed for the quarter | Removes the option of scaling the whole plan | Confirm with finance before the plan is locked | Re-solve with the envelope as a decision variable, not a constraint |
| C-05 | Brand-search spend cannot fall below 0.5× current | Bounds the largest single cut in the plan | Read the brand term in the supplier contract | Hold brand at 0.5× and fund the gap elsewhere |

The most dangerous assumption is A-04, not because it is unusual but because it is invisible. The
response curve is fitted where spend has historically lived; the plan proposes to move outside that
range; and the curve will happily extrapolate. In this case the fitted optimum for non-brand search
is 8.06× current spend — a 3.4× increase from the proposed 1.8× cap — which is a clear signal that
the curve's functional form, not the market, is producing the number (§6.4). A cap is therefore not
a bureaucratic limitation on the optimiser; it is the statement that the model's validity ends where
the data ends, and the cap is doing the work the confidence interval should have done.

Two assumptions deserve separate comment because they interact. A-01 and A-03 are both about the
ledger, and they fail in opposite directions: multiplicity inflates credited revenue (one order,
several credits), while window truncation deflates long-lag credit (fewer touches earn credit than
caused the sale). In a channel mix that skews to late, low-lag touches — brand search, retargeting,
coupon affiliates — the first effect dominates and the ledger overstates. In a mix that skews to
early, high-lag touches — video, display, prospecting — the second dominates and the same ledger
understates. This is why a single inflation factor for the whole ledger is a simplification that must
be stated and then decomposed (§6.2); using it as a constant across channels is precisely the error
the incumbent instrument made, in the opposite direction.
