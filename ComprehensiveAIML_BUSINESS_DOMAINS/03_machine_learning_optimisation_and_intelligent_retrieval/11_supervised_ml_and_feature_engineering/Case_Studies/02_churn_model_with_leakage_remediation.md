---
case_id: "CASE-M11-02"
case_title: "Churn leakage remediation: withdrawing a retention model that could not be scored on the day it decided"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "draft"
reviewed: null
content_status: "authored draft; independent review open"
evidence_status: "illustrative teaching values; arithmetic verified locally; no measured company outcome"
implementation_stage: "proposed; no queue change executed"
source_status: "OPEN — original retention source bodies unavailable; method references named in Section 14a"
case_type: "predictive-ml"
domain: "B2B subscription retention and customer success"
primary_pattern: "temporal-validation"
secondary_patterns: ["leakage-control", "decision-threshold-under-capacity"]
evidence_level: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Customer-success operations lead (decision owner); retention analytics lead (recommender); finance business partner (budget approver)"
review_trigger: "billing snapshot semantics, CRM activity taxonomy, contact capacity, save-offer economics, contract-length mix, or churn definition change"
---

# Case study: Churn leakage remediation

> **Evidence statement, read first.** Every account count, rate, currency amount and hour figure below is a
> constructed teaching value. The arithmetic tying those values together has been recomputed by hand and by an
> independent expression evaluation, and the results appear in Section 6.8 and Section 11. No number is a
> measurement of a real company, no model was trained, no queue was run, and no save offer was made. Where the
> text states a mechanism rather than a quantity, the statement is a consequence of the definitions in Sections
> 3 and 5, not an empirical claim. Source bodies for the retention material referenced by this course were not
> available for reading, so no claim about their contents is made anywhere in this file.

## 1. The One-Line Framing

Given a monthly snapshot of 60,000 mid-market subscription accounts and only those signals that were actually
recorded before the planning cut, decide which **900 accounts** enter the human outreach queue for a nine-week
retention cycle, to protect contribution-margin revenue that would otherwise be lost to voluntary cancellation,
subject to a hard customer-success capacity of 900 contacts per cycle and a non-negotiable requirement that
every feature be reproducible from the warehouse as of the decision instant, compared with the deployed
**rules health score** that has ordered that queue for three years.

The unit of decision is the **account-cycle**: one account, one planning cut, one queue slot request. The unit
of measurement is the same account observed at its 90-day outcome boundary. Those two units are not the same
row and must not be joined as if they were, which is the first of several grain errors this case makes
deliberate.

The primary success condition in plain language: the accounts this queue selects must churn **less often** than
the accounts the health score would have selected under the same 900-slot capacity, and that claim must survive
recomputation on a forward holdout built only from decision-time-available inputs. The secondary condition, and
the one finance actually funds, is that the queue's expected retained contribution margin be positive after
contact cost.

**The supported surprise, stated before the evidence.** The model the analytics team built reported 52%
precision at the 900-slot capacity against a 6% base rate — a lift of 8.67× — and that number is the symptom,
not the achievement. At the moment the queue must be built, the field driving that performance does not exist
yet. The honest forward number is 18% precision, a lift of 3.0×, which is still commercially useful: it clears
the 7.75% break-even threshold and adds roughly 119,459 USD of expected margin per cycle over the health score.
So the case is not "the model was bad." It is "the model was evaluated on a protocol the deployment cannot
reproduce," and the remediated version is worth shipping while the celebrated version was not.

**Why the obvious simpler framing misses the point.** The obvious framing is "a better churn model means higher
precision, and precision at capacity is the metric, so maximise it and ship the winner." That framing is blind
to three things that decide this case. First, precision computed on rows whose features were assembled after the
outcome was already visible measures hindsight, and the protocol — not the model — produced the 52%. Second, a
queue is a *fixed-capacity ranking*, so the economic question is where the marginal account stops paying for its
contact, which is a threshold question the precision number cannot answer on its own. Third, the outcome that
matters is churn *among accounts that were not contacted*, so the evaluation population has to include a
comparison group the operations team is willing to leave alone; the moment a queue exists, the population it
did not contact becomes the only honest counterfactual available, and that group must be designed rather than
inherited.

Primary pattern: `temporal-validation`. Secondary: `leakage-control` and `decision-threshold-under-capacity`.
The decisive trade-off, stated before any evidence: the as-built model looks better than anything the function
has produced, and adopting it would consume 510 analyst hours per cycle and 108,900 USD of contact cost on a
ranking that cannot be produced on the morning it is needed.

**Takeaway:** a performance number is evidence about a protocol, not about a system; when the number is
implausible for the base rate, audit the protocol before celebrating the model.

## 2. Context & Background

### 2.1 The commercial shape

The teaching setting is a mid-market business-to-business software subscription. Accounts are companies; each
account holds one or more named seats; each account pays a monthly recurring charge under an annual or
multi-year contract. The commercial team reports on recurring revenue, not on user counts, so every metric here
is denominated in accounts or in monthly recurring revenue (MRR), never in logins or seats.

The behavioural unit is the **account-month**: an account observed at one monthly billing boundary. Accounts are
not independent across months — the same account appears up to twelve times in a year of history, with
correlated outcomes. That dependence is a first-class problem in Sections 6 and 9; it is the single most common
reason a retention model's uncertainty interval is overstated.

### 2.2 The workflow that the decision interrupts

Six weeks before a renewal date the renewal operations team produces a forecast. Four weeks before, customer
success plans outreach. Two weeks before, the account executive receives a defence brief. If the customer
intends to cancel, they submit notice through the account portal, and the **notice date** — not the contract end
date — is the event that matters for measurement.

The retention queue is built at the *planning* boundary, roughly ten weeks before the average renewal date,
because outreach has to be scheduled into human calendars and offers have to be approved by finance. That timing
is not a technical detail. It means the model must produce a score at a moment when the current month's usage is
only partially complete, when the renewal conversation has not yet happened, and when no cancellation signal
exists for most accounts.

Three clocks therefore coexist and are constantly confused:

| Clock | Definition | Typical value in this case | What it governs |
|---|---|---|---|
| Event time | When the customer behaviour occurred | within the trailing 30 days | feature semantic window |
| Ingestion time | When the warehouse recorded the event | 2–36 hours after event time | latency budget for the build |
| Availability time | When the value became queryable at the planning cut | event time + ingestion + transform | **the only legal cutoff** |
| Decision time `t0` | Monday 06:00 local, the snapshot instant | fixed per cycle | row set and population |
| Outcome time | `t0` + 90 days | per account | label maturity |
| Notice time | customer cancellation submission | 0–90 days after `t0` | the label itself |

The distinction between event time and availability time is where the case lives. A support ticket created at
23:40 on the Sunday before a 06:00 Monday cut has an event time inside the feature window and an availability
time after the cut for perhaps a third of accounts, depending on the CRM sync schedule. Using it is leakage.

### 2.3 Stakeholders and their conflicting objectives

| Stakeholder | Owns | Measured on | Optimises if unopposed |
|---|---|---|---|
| Customer-success operations lead | the outreach calendar and 900 slots | contact coverage and post-contact retention | accounts touched, the quantity they control |
| Retention analytics lead | queue ordering and its evidence | whether queued accounts churn less than comparable unqueued ones | offline discrimination, the metric that moves fastest |
| Finance business partner | discount and credit budget | cost per retained dollar of MRR | approving no offers until effect is proven |
| Renewal operations | renewal forecast | forecast accuracy against booked renewals | excluding uncertain accounts from the forecast |
| Data platform owner | snapshot semantics, feature availability | freshness and correctness | serving whatever is cheapest to materialise |
| Privacy and legal reviewer | permitted use of support, sentiment and usage data | documented purpose and retention limits | removing more fields than necessary |

The conflict is structural rather than personal. A queue that maximises discrimination at fixed capacity can be
a queue that spends the entire discount budget on accounts that would have stayed anyway, because the ranking
optimises *risk* while finance funds *incremental saves*.

### 2.4 The baseline and its documented design weaknesses

The deployed process is a **rules health score**: a weighted sum of five banded signals — logins in the trailing
30 days, open support-ticket count, days since the last executive-sponsor meeting, contract days remaining, and
whether an expansion opportunity is open. The top 900 accounts enter the queue.

Three weaknesses are properties of the design rather than of its parameters:

1. **It is not calibrated.** The sum has no probability interpretation, so it cannot be compared against the
   cost of a contact, and it cannot answer whether 900 contacts is the right number rather than the available
   number.
2. **Its weights were set in a workshop.** Bands and weights were agreed in a half-day session and not revisited
   in three years, during which the pricing model changed twice.
3. **It double-counts correlated signals.** Open tickets and days-since-sponsor are strongly associated, so the
   effective weight on "account in difficulty" exceeds any single band.

The health score's own precision is **not measured** in this file. That is deliberate: the argument is that an
unmeasured baseline plus an inflated model number produces a confident wrong decision, and filling that gap with
a fabricated baseline would repeat the error. Section 6 instead uses a *declared planning value* of 9.5%
precision at capacity, labelled illustrative, purely so the economic comparison has an arithmetic spine.

### 2.5 Scale, timing and boundaries

| Context field | Teaching value | Status |
|---|---|---|
| In-scope accounts | 60,000 mid-market accounts | illustrative |
| Median in-scope MRR | 1,400 USD per month | illustrative |
| Contribution margin on recurring revenue | 78% | illustrative |
| Expected remaining tenure at snapshot | 11 months | illustrative |
| Churn base rate over the 90-day label window | 6.00% | illustrative |
| Contact capacity per nine-week cycle | 900 accounts | illustrative |
| Analyst time per contacted account | 34 minutes | illustrative |
| Fully loaded contact cost | 121 USD per account | illustrative |
| Cycles per year | 5.78 (52/9) | derived |
| History available | 24 monthly snapshots | illustrative |
| Geography / currency / billing system | single region, single currency, single system | illustrative boundary |

Population exclusions are applied before modelling: accounts below a revenue floor (outreach not economically
justified), accounts in formal notice (outcome determined), accounts in their first 30 days (no usage baseline).
These are not leakage, but they change the denominator the precision number refers to, and a precision computed
on a different population is not comparable.

### 2.6 Prior attempts

Two prior attempts matter, both recorded as narrative because no artifact survives in the teaching setting.

The first added a gradient-boosted re-ranker on top of the health score. It was abandoned after two cycles
because operations saw "the same accounts in a slightly different order." That perception was probably correct:
a re-ranker trained on the same leaky features inherits the leak, and a leaky model's ordering is dominated by
whichever accounts sit closest to the extraction date rather than by which accounts are at risk.

The second built a churn model on twelve months of snapshots with a random row split. It reported AUC 0.94 and
a top-decile lift of eleven. Operations ran one cycle, observed that the top of the queue was full of accounts
that had already renewed, and stopped using it. The recorded reason was "model does not reflect reality." That
diagnosis is directionally right and mechanistically wrong: the model reflected the past very accurately, and
the queue needed the present.

**Takeaway:** the failure mode that repeatedly kills retention models is not weak signal; it is that the model
answers "which accounts were at risk in the past, as seen from today" instead of "which accounts are at risk
today, as seen from today."

## 3. The Business Question

### 3.1 The question as the decision owner put it

> "I have 900 slots and six cycles before the year closes. Your model says its top 900 are more than five times
> as likely to leave as the health score's top 900. Before I rebuild the calendar around it, tell me whether
> those 900 accounts are ones I can actually identify ten weeks before renewal — and if they are not, give me the
> honest number, because I would rather plan against a real 18% than a fictional 52%."

This is a constructed stakeholder question for teaching, not a quotation from a person.

### 3.2 The decision contract

| Field | Content |
|---|---|
| Question | Order the retention queue by the as-built model, the remediated model, or the incumbent health score — and at what capacity? |
| Decision owner | Customer-success operations lead; analytics lead recommends; finance approves offer budget |
| Trigger | Offline evaluation reported 52% precision at 900; operations was asked to commit four cycles |
| Deadline | Two weeks before the next planning cycle, when calendar blocks publish |
| Actions available | (a) keep health score; (b) adopt as-built; (c) adopt remediated with forward holdout; (d) adopt remediated plus randomized no-contact holdout; (e) defer one cycle to measure |
| Value at stake | 11–26 expected saves per cycle under the declared save rate, worth 0.13–0.31 M USD of contribution margin before contact cost |
| Downside of a wrong queue | 510 analyst hours and 108,900 USD of contact cost per cycle, displacing other customer-success work |
| Objective | Maximise incremental retained contribution margin per cycle at fixed capacity, subject to a reproducibility gate and a calibration gate |
| Hard constraints | Features reproducible as of `t0`; queue explainable to the account executive; capacity ≤900 per cycle; finance approval for any offer |
| Scope | 60,000 accounts, one region, one billing system, 5.78 cycles per year |
| Out of scope | pricing changes, product changes, involuntary churn from payment failure, expansion targeting, individual treatment assignment |

### 3.3 Why objective and proxy can disagree

The objective is retained contribution margin. The proxy the analytics team reported is precision at the contact
capacity. They agree only if three conditions hold: the probability of saving a contacted account is equal across
true churners; contact cost is equal across accounts; and accounts displaced from the queue have no value. None
holds exactly. High-value accounts receive better save offers, so save probability rises with revenue; contact
cost is roughly flat, so value per contact is not; and ranks 901+ have real but lower risk, so displacement is
not free.

The arithmetic that makes the disagreement concrete uses one value per *true churner contacted*:

```text
FORMULA [CH-01]: v = s * R * m * T
WHERE: v = expected retained contribution margin per true churner placed in the queue [USD/account];
       s = probability that a contacted true churner is saved [dimensionless, 0..1];
       R = monthly recurring revenue [USD/account-month];
       m = contribution margin fraction on recurring revenue [dimensionless];
       T = expected remaining tenure at snapshot [months].
CONSTRAINT: s must be estimated on contacted accounts of the same segment; v is an expectation,
            not a receipt, and assumes no displacement of other revenue.
NUMERICAL EXAMPLE: s=0.13, R=1400, m=0.78, T=11 -> v = 0.13*1400*0.78*11 = 1561.56 USD.
INTERPRETATION: each true churner occupying a slot is worth about 1.56k USD of expected margin,
               which is what a contact must be compared against.
```

```text
FORMULA [CH-02]: N(p) = C * (p * v - k)
WHERE: N = expected net margin of one cycle's queue [USD]; C = contact capacity [accounts];
       p = precision at capacity = share of queued accounts that would churn without contact;
       v as above; k = fully loaded cost per contact [USD/account].
CONSTRAINT: p measured on the same population and horizon as the queue; cost is linear in
            contacts; no offer spend beyond k is included.
NUMERICAL EXAMPLE: C=900, p=0.18, v=1561.56, k=121 ->
                   N = 900*(0.18*1561.56 - 121) = 900*(281.0808 - 121) = 144072.72 USD.
INTERPRETATION: the queue is worth a positive expected margin only while p*v > k; the
                break-even precision p* = k/v follows directly.
```

```text
FORMULA [CH-03]: p* = k / v
WHERE: p* = break-even precision at capacity [dimensionless]; k, v as above.
CONSTRAINT: same population and horizon; v estimated, not observed.
NUMERICAL EXAMPLE: p* = 121 / 1561.56 = 0.077487 -> 7.75%.
INTERPRETATION: below ~7.75% precision the queue destroys margin even though it "finds churners".
```

The guardrail that keeps the proxy honest is therefore a **value-weighted** comparison: the queue's expected
retained margin must improve, not merely its precision.

### 3.4 What an expensive false decision looks like

There are two distinct shapes, and they cost different things.

**Shape A — adopting the as-built model.** The 52% number is believed, four cycles are committed, and the queue
is rebuilt each cycle from features that either do not exist at `t0` (the build fails, and the team silently falls
back to last month's scores) or exist only because the pipeline reads the current warehouse state (the queue is
built from today's data and is therefore a ranking of accounts that already renewed). The cost is 4 × 108,900 =
435,600 USD of contact spend plus 4 × 510 = 2,040 analyst hours, against saves that may not materialise because
the ranking is dominated by accounts whose outcome is already decided.

**Shape B — rejecting everything.** Operations, having been burned twice, keeps the health score. The
opportunity cost is the incremental margin of the remediated queue, which Section 6 computes as 119,459 USD per
cycle. Because the health score is unmeasured, this cost is invisible: nobody is ever asked to justify the queue
they did not change. Shape B is the more likely failure in practice, because it requires no decision at all.

**Takeaway:** the expensive error is usually not a wrong model but a decision protocol that lets an unverifiable
number, or an unmeasured baseline, settle a recurring resource allocation.

## 4. Assumptions & Constraints

### 4.1 Assumption register

Each row states the assumption, the unit or clock it lives on, where it came from, and what happens if it fails.
Assumptions marked **reversal-critical** can flip the recommendation on their own; the rest change magnitude.

| ID | Assumption | Units / clock | Basis | If false |
|---|---|---|---|---|
| A1 | Warehouse state at `t0` is a faithful record of what was known at `t0` | timestamp | platform contract | features must be rebuilt from immutable event logs; A1 is reversal-critical |
| A2 | Notice date is the churn event | date | billing/portal definition | label changes and every number shifts; reversal-critical |
| A3 | 90 days is enough for the label to mature for the scored cohort | days | maturity curve in §5d | precision inflated by immature negatives; reversal-critical |
| A4 | Save probability is 13% for contacted true churners | dimensionless | declared planning value, not measured | break-even precision moves between 5.04% and 16.79% (§6.5) |
| A5 | Contact cost is 121 USD and roughly flat across accounts | USD/account | loaded analyst time plus tooling | value-weighted ranking must include account-level cost |
| A6 | Expected remaining tenure is 11 months | months | contract mix | scales all values proportionally; ranking unaffected |
| A7 | Contribution margin is 78% of MRR | dimensionless | finance declaration | scales all values linearly |
| A8 | Accounts are the decision unit; contacts are not shared across accounts | account | operations design | two accounts under one parent company break independence |
| A9 | The 900-slot capacity is hard, not elastic | accounts/cycle | staffing plan | capacity becomes a cost variable (§6.5) |
| A10 | No save offer exceeds the standard discount band | USD | finance policy | value per save rises; policy risk unquantified here |
| A11 | The health score's ordering is stable run to run | rank | its deterministic band sum | baseline comparison becomes noisy; reversal-critical for A/B claims |
| A12 | Excluded populations stay excluded across cycles | account set | policy | precision denominator changes; comparability breaks |
| A13 | A queued account receives exactly one outreach attempt per cycle | count | operations standard | repeated contacts multiply cost without modelled benefit |
| A14 | No cross-cycle interference: this cycle's contact does not change a later cycle's baseline risk | account | assumption, untested | effects attributed to later cycles are contaminated |

A1, A2, A3 and A11 are the four that can reverse the design; Sections 5d, 6.1 and 12 test each explicitly.

### 4.2 Constraints versus choices

A **constraint** is imposed by an authority outside the analytics team and cannot be optimised away: the 900-slot
capacity, the 34-minute handling time, the requirement that features be reproducible at `t0`, the requirement
that the queue be explainable, the prohibition on using free-text support sentiment without a documented lawful
basis, and the finance rule that any discount requires approval.

A **choice** is a design decision the team owns and must defend: the model family, the split design, the
calibration method, the rank-versus-threshold rule, the reason-code format, the number of features, and whether
to run a randomized no-contact holdout. Confusing the two produces the classic failure of "we cannot change the
capacity, so let us relax the reproducibility requirement" — one is fixed, the other is the whole point.

### 4.3 What the case deliberately does not assume

- It does not assume the 52% number is wrong because it is large. It assumes the number is *unexplained*, and
  the audit in §6.1 supplies the explanation.
- It does not assume the health score's 9.5% precision is real. It is a declared planning value used to size a
  comparison, and the file says so wherever the number appears.
- It does not assume that a contacted account would have churned without contact. Save probability is a separate
  quantity, estimated only by a randomized holdout, and no such holdout exists here.
- It does not assume the label is mature for the most recent vintages; §5d and §6.6 quantify the bias.
- It does not assume a model score authorises a contact. Operations and finance hold that authority.

**Takeaway:** the assumption register is the case's real content; a reader who can name which assumption is
load-bearing has understood more than a reader who can quote the precision.

## 5. Data, Tools & Method

### 5a. Data Specification

| Element | Specification |
|---|---|
| Grain | one row per (account_id, cycle_id) at the planning cut; one outcome row per (account_id, cycle_id) at `t0`+90d |
| Primary key | `account_id` (stable billing identifier), `cycle_id` (ISO week of the Monday cut) |
| Entity key risk | parent-company rollups and re-billed accounts can share an `account_id` history; §5d handles it |
| Time zone | all timestamps stored UTC; the planning cut is a local 06:00 converted to UTC per region |
| Snapshot semantics | `as_of` query restricted to availability_time <= `t0`; late-arriving corrections kept as versions |
| Feature windows | 30-day login window, 90-day ticket window, 180-day sponsor-meeting window, contract `days_remaining` at `t0` |
| Label | `notice_submitted` within `(t0, t0+90d]`, excluding involuntary payment failure |
| Label maturity | a `cycle_id` is mature only when `t0 + 90d + 14d` processing lag has elapsed |
| Population | active accounts, MRR above floor, not in notice at `t0`, age ≥30 days |
| Exclusions | fraud-suspended accounts, accounts in legal dispute, internal test tenants |
| Volume | 24 cycles × 60,000 accounts ≈ 1.44 M account-cycles before exclusions |
| Storage | features ≈ 120 columns × 8 bytes ≈ 0.96 KiB/row → ≈1.38 GiB uncompressed for 1.44 M rows |
| Refresh | full rebuild per cycle; incremental backfill for corrected vintages |
| Access control | row-level by region; support free-text excluded from the model artifact |

The storage figure is payload-only: 120 columns at 8 bytes is 960 bytes per row, and 1.44 M rows is
1,382,400,000 bytes ≈ 1.29 GiB (1 GiB = 1,073,741,824 bytes). Parquet compression and indexes are additional and
are not estimated here.

### 5b. Tools & Stack

| Layer | Tool class | Why | Failure if missing |
|---|---|---|---|
| Warehouse | columnar SQL engine with `AS OF` / snapshot support | point-in-time reconstruction is the case's core requirement | features silently read current state |
| Feature build | scheduled SQL plus versioned transformation code | reproducible per cycle | no audit trail for a queue |
| Modelling | gradient boosting and logistic regression with calibration | nonlinear fit plus an interpretable comparator | no credible baseline for explanation cost |
| Evaluation | grouped, time-ordered splits; bootstrap intervals | dependence across account-months | optimistic uncertainty |
| Serving | batch scoring job writing an immutable queue table | queue is consumed by humans, not by an API | no replayable artifact |
| Monitoring | calibration and rank-drift dashboards per cycle | detects population shift | drift discovered by complaint |
| Governance | feature availability ledger (name, source, availability rule, owner) | the artifact that makes leakage review possible | every audit restarts from zero |

### 5c. Method Summary

1. Freeze the decision contract: `t0`, population, capacity, action set, label, horizon, comparator (§3.2).
2. Build a feature availability ledger: for each candidate feature, record source table, maximum event time,
   ingestion lag distribution, transform, and the rule that makes it legal at `t0` (§5d).
3. Construct two evaluation worlds: **as-built** (whatever the pipeline produced) and **as-of** (rebuild with
   `availability_time <= t0`).
4. Split by time: train on cycles 1–18, validate on 19–21, hold out 22–24 untouched until one model is frozen.
   Group all rows of an account into one partition.
5. Fit three candidates on the as-of world: health-score re-ranker, logistic regression, gradient boosting.
6. Calibrate on the validation cycles using isotonic regression fitted per cycle, never on holdout.
7. Convert calibrated probability to a **rank-cut at 900**, because capacity binds; report the implied
   probability at the cut so finance can check it against `p* = k/v`.
8. Report precision at capacity, recall on the mature churner population, calibration by decile, and expected
   net margin with `CH-02`, each with an interval.
9. Compare on the frozen holdout; freeze the winner; specify the queue-build contract for serving (§8).
10. Pre-register the randomized no-contact holdout for the *next* cycles; this is the only design that can
    estimate save probability.

### 5d. Data provenance, quality, exclusions and leakage controls

**Leakage taxonomy used in the audit.** Four kinds appear in retention work, and each has a different check:

| Kind | Mechanism | Check | Result here |
|---|---|---|---|
| Target leakage | a feature records the outcome or its precursor | availability ledger + maximum feature event time inside the label window | found: `renewal_disposition` and `notice_flag` (§6.1) |
| Temporal leakage | feature value computed from data after `t0` | rebuild with `availability_time <= t0` and diff | found: current-state joins on usage and tickets |
| Group leakage | same account in train and holdout | partition by `account_id` | found in the earlier random split |
| Selection leakage | the evaluation population depends on the label or on treatment | compare population built before scoring with population after | found: accounts in notice included at scoring time |

**Availability ledger (excerpt).** Twelve of the 120 candidate features are shown; the ledger entry that matters
is the last column.

| Feature | Source | Event window | Ingestion lag | Legal at `t0`? | Note |
|---|---|---|---|---|---|
| `logins_30d` | product events | 30 d | ≤6 h p95 | yes | partial-day at cut; window is `[t0-30d, t0-1d]` |
| `tickets_open` | CRM | point-in-time | ≤2 h p95 | yes | counts only, no text |
| `days_since_sponsor` | CRM meetings | 180 d | ≤24 h | yes | missing for 11% of accounts → explicit null |
| `days_remaining` | billing | point-in-time | 0 | yes | deterministic from contract |
| `expansion_open` | CRM opportunity | 90 d | ≤24 h | yes | 4.2% of accounts change weekly |
| `mrr` | billing | point-in-time | 0 | yes | versioned on amendment |
| `support_sentiment` | free text | 90 d | ≤72 h | restricted | excluded from the model; lawful basis unresolved |
| `renewal_disposition` | CRM call notes | 90 d | ≤24 h | **no** | recorded during the renewal conversation, after `t0` |
| `notice_flag` | portal | point-in-time | 0 | **no** | records the outcome itself |
| `contract_signed_date` | billing | point-in-time | 0 | yes | used only to derive age |
| `usage_seats_active` | product | 14 d | ≤6 h p95 | yes | denominator drift when seats are added mid-window |
| `credit_memo_count` | finance | 180 d | ≤48 h | yes, if no offer is attached | exclude memos created by the retention team |

**Label maturity.** A cycle is scored mature when `t0 + 90d + 14d` has passed. In the 24-cycle history, 21 cycles
are mature at 6.0% churn and 3 are immature, showing 4.1% observed churn because notices are still arriving. If
those 3 cycles are pooled without a maturity filter, the pooled base rate is
(21×6.0 + 3×4.1)/24 = 5.7625%, which is 0.2375 percentage points low. That distortion propagates into every
precision denominator computed on the pool, and it is why §6.6 recomputes the headline comparison on mature
cycles only.

**Exclusions and their effect.** The revenue floor removes 8,400 accounts (14.0%), first-30-day accounts remove
2,100 (3.5%), and notice-at-`t0` removes 900 (1.5%). The 900 removed for notice are not random: they are the
highest-risk accounts in the file, so including them would inflate measured precision while contributing nothing,
because their outcome is already fixed.

### 5e. Reproducibility and method-selection ledger

Reproducibility means: given `(cycle_id, model_version, transform_version, population_version)`, an independent
engineer can regenerate the exact ordered queue of 900 account IDs and the reason codes behind each. The evidence
is a diff of ordered IDs, not an aggregate metric.

| Alternative | Considered because | Rejected because | Where revisited |
|---|---|---|---|
| Keep health score | zero build risk, explainable | unmeasured and uncalibrated; cannot answer the capacity question | §11 if remediation fails |
| Re-ranker on health score | cheapest change | inherits the leak and the uncalibrated scale | §12 failure 6 |
| Random-split GBM (prior attempt) | best offline AUC | group and temporal leakage; top of queue already renewed | §6.1 |
| Logistic regression on as-of features | interpretable, stable coefficients | forward precision 15.8% at capacity vs 18.0% for GBM — inside noise, so kept as the *explanation* fallback | §9.4 |
| GBM on as-of features | captures interactions | less stable coefficients; needs reason codes | chosen, §6.3 |
| Depth-limited GBM without calibration | simplest ranker | rank-cut needs an interpretable probability at the cut for finance | §5c step 6 |
| Threshold rule instead of rank-cut | stable per-account action | at 6% base rate a fixed threshold yields 300–3,000 tasks across cycles | §6.5 |
| Randomized no-contact holdout now | only way to estimate save probability | would cost a cycle of coverage before any queue change is justified | pre-registered for later cycles, §8.4 |

**Takeaway:** the method ledger exists so that a later reader can tell which choices were tested and which were
merely preferred; the rejected random-split model is the same family as the chosen one, differing only in
protocol.

<!--PART2-->
