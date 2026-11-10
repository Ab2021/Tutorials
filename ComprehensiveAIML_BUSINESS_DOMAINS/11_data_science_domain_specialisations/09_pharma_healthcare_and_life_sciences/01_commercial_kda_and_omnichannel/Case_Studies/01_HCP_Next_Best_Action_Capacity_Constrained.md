---
case_id: "CASE-09-01-001"
case_title: "Next-Best-Action for HCP Engagement Under Fixed Field Capacity"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Pharma commercial analytics - field force targeting and next-best-action"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["causal-confounding", "human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Commercial analytics lead with field operations co-owner"
source_index: "INDEX.md"
review_trigger: "Any change to call capacity, data-sharing consent rules, aggregate-spend policy, or brand strategy"
---

# Next-Best-Action for HCP Engagement Under Fixed Field Capacity

> **One-line framing:** Given one keyed HCP-week decision event and point-in-time
> prescription, engagement and access features, choose which 600 of 1,020 eligible
> prescribers receive the coming week's field calls so that incremental brand
> prescriptions per call rise, subject to a hard capacity ceiling, consent rules and a
> 30-second recommendation latency budget, compared with the current
> prior-quarter-volume decile ranking.
>
> **Evidence declaration:** every number in this file is a constructed teaching value.
> No company, brand, territory or prescriber population is described. Arithmetic is
> independently recomputed in Section 6 and re-executed as a standalone smoke at the
> end of authoring. Source bodies for the assigned commercial source brief were not
> available; the source limitation is recorded in Section 5e and Section 11.

## 1. The One-Line Framing

This case studies one recurring commercial decision: which prescribers a
fixed-size field force should contact next week. The unit of analysis is a single
HCP-week decision event, not a prescription row, not a call, and not a patient.
Every candidate action is one planned call to one eligible prescriber in one week.

The task category is allocation with a ranking component. It is allocation because
the binding resource is the number of calls the field force can physically deliver in
a week. It is ranking because the allocator must order more eligible prescribers than
it can serve. It is not pure forecasting: a forecast of next month's prescriptions
does not by itself answer who should be visited, because the decision needs the
difference between contacting and not contacting a specific prescriber.

The beneficiary is the brand team and the regional sales director, who jointly own
the call plan. The prescriber is not the beneficiary of the analytics; the prescriber
receives a visit whose content is governed by separate medical, legal and regulatory
review that this case does not touch.

The binding constraints are four:

1. Capacity: 40 representatives times 15 calls per week equals 600 deliverable calls
   per week, with no overtime permitted under the current operating policy.
2. Eligibility: only prescribers with consent and sufficient longitudinal data may be
   scored; 180 of 1,200 candidates are excluded before scoring.
3. Human authority: a regional manager must approve any plan change that moves more
   than 10 percent of a representative's call list, and every recommendation is
   advisory to the representative.
4. Latency: recommendations must be regenerated inside a 30-second weekly batch so
   that the plan is available before Monday routing.

The baseline is the current process: rank prescribers by prior-quarter total
prescriptions within territory, cut at capacity, and let representatives reorder the
list by judgement. That baseline is not a straw man. It is cheap, transparent,
stable week to week, and it embeds institutional knowledge about relationships that
no model in this case represents.

The primary success condition is a higher rate of incremental prescriptions per
delivered call, measured on prescribers who were actually contacted, against a
concurrent randomized comparison, in a pre-declared 12-week window. Section 9 defines
the full metric, including numerator, denominator, population and guardrails. This
file does not claim that condition was met; the pilot described here is illustrative
and its interval includes zero.

The reusable pattern is evaluation-mismatch, with causal-confounding and
human-in-the-loop as secondary patterns. The mismatch is concrete: the offline
metric that the team can compute cheaply is "did the prescriber respond", but the
decision requires "did the prescriber respond because of the call". Those two
quantities diverge exactly where the model is most confident, because the
highest-scoring prescribers under a response model are the ones who would have
prescribed anyway.

The simpler framing that misses the point is "predict next-quarter prescriptions and
visit the biggest predicted values". That framing answers a volume question, not an
incrementality question. It systematically allocates calls to prescribers whose
behaviour is already saturated, and it cannot distinguish a prescriber who responds
to any contact from one who responds only to a specific message.

A second naive framing is "predict who will respond to a call". That is closer, but
it still fails when the response model is trained on observational call data: calls
were not assigned at random, so the model learns where representatives already chose
to go. Section 4.4 shows the mechanism with an explicit numerical counterexample.

The interesting tension in this case is not model accuracy. It is that the
commercially correct answer may be to contact fewer prescribers, or different ones,
than the relationship-driven plan, and that this conflicts with representative
autonomy. Any design that ignores the human workflow will be overridden in the field
and will appear to fail even if its ranking is sound.

**Interview cue:** a reader should be able to state, in thirty seconds, the decision,
the population, the capacity number, the baseline and the evidence status without
naming a single algorithm.

**Acceptance:** input grain, output, beneficiary, baseline, constraints and evidence
basis are all identifiable from this section alone.

Takeaway: an allocation decision is defined by its capacity ceiling and its
counterfactual, not by the sophistication of its ranking model.

## 2. Context & Background

### 2.1 Domain and product

The setting is a mature branded medicine in a single therapeutic area, sold through
a field force in one country. Commercial operations for such a brand typically run a
weekly cycle: headquarters produces a target list, regional managers adjust it,
representatives execute visits, and prescription data arrives with a lag through
prescription audit panels or aggregated dispensing feeds.

Three data regimes coexist in this workflow, and conflating them is the single most
common analytical error in commercial pharma analytics:

- Prescription volume data, aggregated to the prescriber-week level, delayed by
  several weeks and revised after the fact.
- Engagement data, produced by the customer relationship system, near-real-time but
  self-reported by representatives and subject to late entry.
- Access and formulary data, updated episodically when payer decisions change, with
  effective dates that may precede the publication date.

A model that joins these three without respecting their clocks will produce a
recommendation that "knows" about a formulary change before the field did.

### 2.2 Workflow before and after the decision

Before the decision, the commercial data team refreshes the prescriber feature store
on Sunday night. Territory alignment, which maps prescribers to representatives, is
refreshed monthly and is authoritative for ownership but not for clinical
relationship. The analytics service scores eligible prescribers and emits a ranked
list per representative, truncated at that representative's capacity.

After the decision, each representative reviews the list in a mobile planning tool,
may swap up to a configured number of prescribers per week, records planned calls,
and executes them. Visit notes and call dispositions flow back into the customer
relationship system. Four weeks later, outcome data begins to mature.

The human step is not decoration. Representatives hold relationship knowledge that
the feature store does not encode, including recent conversations, local formulary
committees and personal preferences of prescribers. The design therefore treats the
model output as an ordering with reasons, not as an assignment.

### 2.3 Stakeholders

| Role | Interest | Authority | Failure they experience |
|---|---|---|---|
| Regional sales director | Call-plan quality and field morale | Approves plan changes above threshold | Cannot explain why a long-standing target was dropped |
| Representative | Efficient routing, credible recommendations | Executes or overrides individual calls | Receives a list that ignores local knowledge |
| Commercial analytics lead | Method validity and monitoring | Owns scoring service | Defends a lift claim that the design cannot support |
| Compliance partner | Promotional rules and data use | Can veto data sources or messages | Finds scoring on data outside the approved purpose |
| Data steward | Point-in-time correctness | Owns refresh schedule and lineage | Discovers a lagging feed after the plan shipped |
| Medical affairs | Scientific accuracy of content | Independently approves content | Sees a targeting list imply a clinical claim |

### 2.4 Baseline and its documented weaknesses

The baseline is a decile ranking on prior-quarter prescriptions, filtered by
territory, truncated at capacity, then manually reordered. Its weaknesses are known
and observable in the operating data:

- It is a volume rank, so it re-selects the same large prescribers every week
  regardless of whether that week's contact would change anything.
- It has no notion of eligibility changes, so a prescriber who lost formulary access
  can remain on the list until a representative notices.
- It cannot express the reason for a rank, which makes override analysis impossible.
- It is stable, which is a genuine strength: representatives can plan routes.

The proposed method must be compared against this baseline under the same capacity
and the same representatives. Comparing against "no targeting" would flatter any
method, because some targeting is obviously better than random calling.

### 2.5 Scale and timing

| Quantity | Illustrative value | Unit | Basis |
|---|---|---|---|
| Candidate prescribers in region | 1,200 | HCPs | Constructed |
| Eligible after consent and data-quality filters | 1,020 | HCPs | 85 percent of candidates |
| Excluded for insufficient longitudinal data | 120 | HCPs | Constructed |
| Excluded for consent or purpose limitation | 40 | HCPs | Constructed |
| Excluded for specialty mismatch | 20 | HCPs | Constructed |
| Representatives | 40 | people | Constructed |
| Calls per representative per week | 15 | calls/week | Operating policy |
| Weekly capacity | 600 | calls/week | 40 times 15 |
| Quarter length | 12 | weeks | Planning cycle |
| Quarterly call slots | 7,200 | calls/quarter | 600 times 12 |
| Prescription data lag | 3 to 5 | weeks | Constructed operating norm |
| Visit-note entry lag | 0 to 7 | days | Constructed operating norm |

The capacity figure is the anchor for every economic statement in this case. If
capacity were elastic, the decision would become "how many calls should we make",
which is a different question with a different owner and a different baseline.

### 2.6 Prior attempts

The team previously tried a propensity model that predicted the probability of any
prescription in the next quarter. It scored well offline and was rejected by the
field after six weeks because the top of the list was dominated by prescribers who
already wrote the brand heavily and who representatives described as "already
converted". That episode is the origin of this case's central methodological
constraint: the target must be incremental behaviour, and the evaluation must be
able to distinguish selection from effect.

The earlier attempt failed for a reason worth stating precisely. A propensity model
estimates P(response). Under a monotone relationship between baseline volume and
response probability, ranking by P(response) approximates ranking by baseline volume,
so the model reproduced the baseline while appearing novel. The correct target for
allocation is the conditional effect of contact, not the probability of the outcome.

### 2.7 Applicable requirements

Two requirement classes shape the design. First, data-use purpose limitation: the
approved purpose for prescriber-level data in this scenario covers commercial
engagement planning and excludes clinical decision support. Second, promotional
content approval: the recommendation may propose a contact and a topic category, but
the content itself must come from the approved material library. This case does not
determine legal applicability in any jurisdiction; it records the requirement as a
design constraint whose interpretation belongs to the compliance partner.

Takeaway: the baseline is a real competitor with real strengths, and the workflow
contains a human override that must be designed for rather than suppressed.

## 3. The Business Question

### 3.1 The question in the stakeholder's words

The following is an illustrative paraphrase, not a quotation from any person:

> "We have 600 calls a week and about a thousand prescribers we could visit. The
> current list is last quarter's biggest writers, and my representatives tell me half
> of those visits are social calls. Tell me which prescribers are actually worth a
> visit this week, and tell me why, so I can defend the list to the team."

### 3.2 Decision contract

| Field | Value |
|---|---|
| Question | Which eligible prescribers should receive this week's limited field calls, and in what order, to maximise incremental prescriptions per call? |
| Decision owner | Regional sales director, advised by the commercial analytics lead |
| Trigger and deadline | Weekly plan refresh, Sunday 22:00 local, available before Monday routing |
| Available actions | Adopt model order; adopt model order with manual swaps; keep baseline order; reduce calls to a subset; defer and collect more evidence |
| Value and downside | Incremental prescriptions and margin versus wasted calls, representative distrust, and compliance exposure |
| Objective | Incremental prescriptions per delivered call, aggregated over the quarter, with a guardrail on coverage equity across territories |
| Hard constraints | 600 calls per week; consent and purpose limitation; approved content only; recommendation latency under 30 seconds; human approval above the swap threshold |
| Scope | One country, one brand, 1,020 eligible prescribers, 40 representatives, 12-week quarter |
| Out of scope | Message content selection, pricing, patient support programmes, payer contracting, any clinical claim |

### 3.3 Objective versus proxy

The business objective is incremental prescriptions attributable to field contact.
The measurable proxy is the difference in prescription trajectory between contacted
and uncontacted comparable prescribers. That proxy is only valid when contact
assignment is either randomized or convincingly identified. In the observational
history available here it is not, because representatives chose whom to visit.

Three specific divergences between proxy and objective must be named:

1. Response versus incremental response. A prescriber who would have increased
   prescriptions without a visit counts as a response but contributes no increment.
2. Short-window versus sustained effect. A visit may accelerate a prescription that
   would have occurred later, producing a short-term difference with no cumulative
   gain.
3. Volume versus margin. Incremental prescriptions in a heavily discounted channel
   contribute less margin than the same count in another channel.

The evaluation in Section 9 therefore uses incremental prescriptions per call as the
primary metric, margin per incremental prescription as a secondary metric, and
coverage equity as a guardrail. A model that improves the primary metric while
collapsing one territory's coverage fails the guardrail regardless of its average.

### 3.4 What an expensive wrong decision looks like

The expensive error is not a bad single call. It is a systematic misallocation of a
scarce resource across a quarter. If 600 weekly calls are aimed at saturated
prescribers for 12 weeks, the organisation spends 7,200 call slots, roughly 2,700
field hours at 22.5 minutes per call including travel, and the counterfactual value
is invisible because no comparison group exists. The error is expensive precisely
because it is unfalsifiable under the current measurement design.

The second expensive error is organisational: a list that representatives do not
trust gets overridden silently, so the analytics investment produces no behavioural
change while the dashboards report adoption.

### 3.5 Decision frequency and reversibility

The decision recurs weekly and is cheaply reversible at the individual level, because
a representative can substitute a prescriber. It is expensive to reverse at the
programme level, because a quarter of field behaviour creates expectations among
prescribers and across the sales team. This asymmetry justifies a staged rollout:
pilot in two regions, measure, then expand, rather than switching all 40
representatives at once.

**Interview cue:** ask which objective is optimised, what the no-change counterfactual
is, and what an expensive false positive looks like in field hours.

Takeaway: the decision is a repeated constrained allocation whose true target is a
counterfactual, and the measurement design is part of the decision, not an afterthought.

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger |
|---|---|---|---|---|---|---|---|
| A-01 | Prescription volume for an eligible prescriber is observed with a lag of at most five weeks and is not revised by more than 10 percent after 8 weeks | Data | Compare two snapshots of the same historical weeks, measure revision distribution | Unverified | Features and labels are point-in-time wrong | Restrict to matured weeks; add maturity flag | Data vendor change |
| A-02 | Consent and purpose limitation cover engagement planning but not clinical content | Business | Compliance partner written confirmation | Unverified | Scoring itself may be impermissible | Exclude the data source; use territory aggregates | Policy change |
| A-03 | Representative capacity is exactly 600 calls per week with no overtime | Business | Regional operating policy | Confirmed for the scenario | Capacity overflow invalidates the allocation | Cap at lower number; queue | Policy change |
| A-04 | Contact effect is heterogeneous across prescribers, so targeting matters | Model | Pilot randomization | Unverified | Ranking by response equals ranking by volume | Fall back to baseline deciles | Pilot result |
| A-05 | Four-week response window is long enough for a visit to affect prescribing | Model | Pre-trend analysis, clinical input on detailing response time | Unverified | Effect measured too early reads as zero | Extend window; model decay | New evidence on detailing lag |
| A-06 | No interference between prescribers in the same practice | Causal | Check practice-level clustering; measure spillover | Unverified | Treatment effect contaminated across units | Randomize at practice level | Design review |
| A-07 | Representative overrides remain below 30 percent of list entries | Workflow | Override logging in planning tool | Unverified | Model effect diluted beyond recognition | Co-design with representatives; simplify list | Override rate alarm |
| A-08 | Recommendation latency budget of 30 seconds is achievable in the weekly batch | Infrastructure | Load test on production-sized feature matrix | Unverified | Plan ships late; representatives use stale list | Precompute features nightly; cache scores | Volume growth |
| A-09 | Territory alignment changes monthly and is stable within a week | Data | Alignment version log | Unverified | Prescriber assigned to two representatives | Enforce single-owner rule at scoring time | Reorganisation |
| A-10 | Margin per incremental prescription is approximately constant within a channel | Business | Finance-provided net revenue assumptions | Unverified | Economic ranking misallocates across channels | Channel-specific margin table | Contract change |

The three assumptions most likely to reverse the recommendation are A-01, A-04 and
A-07. A-01 breaks point-in-time correctness and can create a spurious offline lift.
A-04, if false, means the entire uplift framing collapses to volume ranking and the
programme should be abandoned rather than tuned. A-07 determines whether the
analytics reaches the prescriber at all.

### 4.2 Constraint register

| ID | Constraint | Source | Consequence | Escalation |
|---|---|---|---|---|
| C-01 | 600 calls per week | Regional operating policy | Allocation must be a strict truncation | Regional director |
| C-02 | Consent and purpose limitation | Compliance policy | Prescribers without consent are never scored | Compliance partner |
| C-03 | Approved promotional content only | Medical, legal and regulatory review process | Recommendation may name a topic category, not text | Medical affairs |
| C-04 | Human approval above 10 percent list change | Field governance | Automation cannot silently reshape a territory | Regional director |
| C-05 | Recommendation latency under 30 seconds | Planning tool contract | Scoring must be batched, not interactive per request | Platform owner |
| C-06 | No prescriber-level data leaves the approved environment | Data governance | No external model endpoints on raw features | Data steward |
| C-07 | Audit trail for every recommendation and override | Governance | Store inputs hash, model version, rank, reason code | Analytics lead |

### 4.3 What the constraints exclude

C-02 excludes 40 prescribers from scoring entirely. That exclusion is not a modelling
choice and must not be re-derived by the model. A common failure is to impute a score
for excluded prescribers "for completeness", which converts a governance boundary
into a technical default. The correct behaviour is an explicit
`not_eligible_reason` field carried through to the planning tool.

C-06 rules out hosted third-party inference on prescriber-level data in this
scenario. It does not rule out hosted inference on de-identified aggregates, but this
case does not need that path, so it is recorded as considered and rejected for
simplicity.

### 4.4 The assumption that decides the case

A-04 deserves separate treatment because it is the hinge of the entire design. If
contact effects were homogeneous, the optimal policy under a capacity ceiling would
be to call the prescribers with the highest expected outcome, because every call
would have the same marginal effect. Under homogeneity, uplift ranking and response
ranking coincide, and the simpler propensity model would be correct.

The commercial reality that motivates this case is heterogeneity: some prescribers
respond to visits, some are indifferent, and some respond negatively because an
unwelcome visit consumes practice time. The pilot described in Section 6 is designed
to estimate whether that heterogeneity exists and how large it is, not to prove that
the model works.

Takeaway: name the assumption whose failure would make the sophisticated method
strictly worse than the baseline, then test that assumption first.

## 5. Data, Tools & Method

### 5a. Data Specification

The decision grain is one row per eligible HCP-week decision event. Every other table
in the environment is either a feature source joined onto that grain or an outcome
attached later. Fixing the grain first prevents the most expensive class of error in
commercial pharma analytics, which is silently changing the denominator between the
training table, the scoring table and the evaluation table.

| Field | Specification |
|---|---|
| Unit of analysis | HCP-week decision event, key `hcp_id` + `iso_week`, one row per eligible prescriber per planning week |
| Volume and window | 1,020 eligible prescribers times 12 weeks equals 12,240 decision events per quarter in the pilot regions; 2 pilot regions of 6 representatives each |
| Time zone | All event timestamps in local territory time; the `iso_week` key is Monday 00:00 to Sunday 23:59 local |
| Inputs | Lagged prescription counts, call history, channel engagement, formulary status, specialty, practice size, territory assignment, prior campaign exposure |
| Outcome | Brand prescriptions dispensed in the four weeks after the decision week, attributed to the prescriber identifier |
| Population and exclusions | Exclude 40 without consent, 20 outside the target specialty, 120 with fewer than 26 observed weeks of history |
| Join semantics | `hcp_id` to territory via versioned alignment; prescription panel to `hcp_id` via a validated crosswalk with documented match rate |
| Splits | Time-based: weeks 1 to 8 for fitting, weeks 9 to 10 for selection, weeks 11 to 12 as a held-out scoring window; additionally two pilot regions are randomized for the causal comparison |

The feature table carries an explicit availability timestamp for every field. A
feature is admissible for the decision at week `w` only when
`available_at <= decision_at(w)`, where `decision_at(w)` is Sunday 22:00 at the end of
week `w - 1`. This single rule eliminates the majority of leakage defects observed in
commercial models, because prescription feeds, formulary feeds and engagement feeds
all arrive on different schedules.

| Feature group | Example fields | Native cadence | Availability rule used | Leakage risk |
|---|---|---|---|---|
| Prescription history | Weekly brand and class fills, 4/13/52-week rolling sums | Weekly with 3-5 week lag | Only weeks with `available_at <= decision_at` | High: lagged values can be joined on the wrong week index |
| Call history | Calls, planned calls, cancelled calls, call channel | Daily, entries late by up to 7 days | Freeze at decision time; late entries excluded from the current week's feature | High: late entry looks like activity before the decision |
| Engagement | Email opens, portal sessions, sample requests | Near real-time | Use value as of decision time, not end-of-week totals | Medium |
| Formulary | Coverage status, restriction flags, effective date | Episodic | Join on effective date, not publication date | High: publication precedes effective date |
| Structural | Specialty, practice size, territory, urban flag | Monthly or static | Versioned snapshot | Low |
| Campaign exposure | Prior campaign membership | Campaign close | Only completed campaigns before the decision | Medium |

Two specific traps deserve naming because they appear in nearly every audit of this
type of work.

The first is the "future fill" trap. A 4-week rolling prescription sum computed at
week `w` using prescription data through week `w + 3` is a legitimate descriptive
feature and an illegitimate predictive feature. It correlates strongly with the
outcome because it contains the outcome. The mitigation is structural: the feature
pipeline takes an explicit `as_of` parameter and refuses to emit a value whose source
week exceeds the allowed maturity, rather than relying on analyst discipline.

The second is the crosswalk trap. Prescription audit panels identify prescribers by a
proprietary identifier, while the customer relationship system uses an internal
identifier. The crosswalk between them has a documented match rate and a known
failure mode: two prescribers in the same practice can collapse onto one identifier.
That produces prescribers with implausibly high volume and representatives who
receive duplicate recommendations. The mitigation is a cardinality check per
practice per week, plus a review queue for crosswalk changes above a volume
threshold.

### 5b. Tools & Stack

The stack is deliberately boring. Boring infrastructure is a virtue in a regulated
commercial setting because every component must be explainable to a compliance
reviewer and reproducible by a successor analyst.

| Component | Role | Version discipline | Owner |
|---|---|---|---|
| Columnar warehouse | Stores feature snapshots, decision log, outcome tables | Schema versioned per release | Data platform |
| Transformation layer | Builds point-in-time features with `as_of` enforcement | Versioned SQL or dataframe jobs | Commercial analytics |
| Uplift model trainer | Fits response and effect estimates | Pinned library versions, fixed seeds | Commercial analytics |
| Scoring batch | Emits ranked lists per representative per week | Container image digest recorded | Platform |
| Planning tool | Presents list, captures overrides, records reasons | Weekly release cadence | Field operations |
| Model registry | Stores model artefact, metrics, approvals | Immutable versions | Platform |
| Decision log | Immutable record of rank, reason, override | Append-only | Data steward |
| Monitoring | Drift, override rate, coverage, latency | Dashboards with owners | Analytics lead |

No external inference endpoint touches prescriber-level features, per C-06. Any
language-model-generated explanation text in the planning tool is constrained to
template slots filled from approved reason codes, so the explanation cannot invent a
clinical claim.

### 5c. Method Summary

The chain is: point-in-time features, then two-model effect estimation, then
constrained allocation, then human review, then logging.

```text
input:  point-in-time feature row x_i for HCP i at week w
step 1: response model  m0(x) ~ P(prescribe | not contacted)
step 2: treated model  m1(x) ~ P(prescribe | contacted)
step 3: effect estimate tau(x) = m1(x) - m0(x)
step 4: rank by tau(x) * value, subject to eligibility and capacity
step 5: allocate greedily within representative capacity
step 6: emit top-k with reason codes and confidence band
output: ordered list of <= 15 HCPs per representative with reason codes
```

The simplest credible baseline is the prior-quarter volume decile rank. Two further
alternatives were considered.

| Candidate | Evidence considered | Strength | Limitation or avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Prior-quarter volume rank | Existing operating report | Transparent, stable, no model risk | Selects saturated prescribers; no effect notion | Chosen as baseline | Never; it is the comparator |
| Single propensity model P(prescribe) | Prior rejected attempt | Simple, one model, easy to monitor | Ranks by baseline volume under monotonicity; confounded by call selection | Rejected | If heterogeneity proves negligible |
| Two-model effect estimate | This design | Directly targets the allocation quantity | Requires overlap and a randomized or well-identified treated sample | Chosen | If overlap diagnostics fail |
| Contextual bandit | Considered | Learns online, allocates exploration | Needs high volume per arm; 600 calls/week is too sparse for stable per-prescriber learning | Not tested | If capacity rises by an order of magnitude |
| Rules on formulary change plus volume | Considered | No model risk, explainable | Cannot rank within the rule-triggered set | Not tested | If model lift proves negligible |

The two-model estimate is chosen because it maps directly onto the decision quantity.
The bandit is rejected on statistical grounds, not on sophistication grounds: with
600 calls per week across 1,020 prescribers, the average prescriber receives a call
roughly every two weeks at best, so per-prescriber exploration cannot converge within
a quarter. The rule-based alternative is retained as a monitoring fallback: if the
model's overlap diagnostics fail, the operationally safe response is to call
prescribers who just had a formulary change, which is defensible without a model.

Threshold selection is a separate decision from ranking. The allocation truncates at
capacity, so the effective threshold is the 600th ranked position each week. There is
no independent probability threshold, and inventing one would create a second,
inconsistent control on the same decision.

### 5d. Data provenance, quality, exclusions and leakage controls

| Source ID | Content | Owner | Access basis | Extraction | Known quality issue |
|---|---|---|---|---|---|
| SRC-01 | Prescription panel, weekly, prescriber level | Vendor under contract | Commercial licence | Weekly batch | 3-5 week lag, restatements |
| SRC-02 | Customer relationship system call log | Internal | Employment | Daily | Late entry up to 7 days |
| SRC-03 | Formulary and coverage feed | Market access team | Internal subscription | Episodic | Effective date precedes publication |
| SRC-04 | Territory alignment | Sales operations | Internal | Monthly | Mid-month reorganisations |
| SRC-05 | Consent and purpose register | Compliance | Internal | On change | Retroactive withdrawals |
| SRC-06 | Campaign membership history | Marketing operations | Internal | On campaign close | Overlapping campaign windows |

The quality controls that matter most are four reconciliation checks run before every
scoring batch.

1. Row-count reconciliation: eligible prescribers in the scoring table must equal
   candidates minus each exclusion reason, and the components must sum exactly. A
   mismatch blocks the batch rather than producing a short list.
2. Maturity reconciliation: no feature may reference a source week newer than the
   allowed maturity for that source. The check is a maximum over source weeks, so it
   fails loudly instead of silently averaging.
3. Ownership reconciliation: every scored prescriber must map to exactly one
   representative in the current alignment version. Zero matches and two matches are
   both errors with different fixes.
4. Consent reconciliation: every scored prescriber must appear in the consent
   register with a commercial engagement purpose. Withdrawn consent propagates within
   one batch, not within one quarter.

Missingness is handled explicitly rather than silently. Structural fields have
documented defaults; behavioural fields carry an explicit `no_history` indicator and
are never imputed with a population mean at the prescriber level, because mean
imputation erases exactly the heterogeneity the allocation depends on. For the 120
prescribers with short history, exclusion is preferred to imputation: a new prescriber
with 8 observed weeks cannot support an effect estimate, and pretending otherwise
would place them at an arbitrary position in the ranking.

The leakage control is enforced in the transformation layer, not in review. Each
feature definition declares `max_source_week_offset`, and the builder computes the
maximum referenced source week and refuses to emit the row when the offset is
violated. Two deliberate tests exercise the control: a feature that references the
current week should raise an error, and a feature that references week `w - 6` with a
5-week maturity should pass. In the authoring smoke, both behave as specified.

### 5e. Reproducibility and method-selection ledger

| Artefact | Location | Version or date | Purpose | Accessibility | Reproduction limit |
|---|---|---|---|---|---|
| Feature definitions | Transformation layer repository | Release tag per batch | Defines `as_of` semantics | Internal | Data not redistributable |
| Trainer configuration | Registry entry | Pinned version | Model hyperparameters, seed | Internal | Determinism across platforms not guaranteed |
| Scoring container | Registry | Image digest | Batch scoring | Internal | Depends on warehouse snapshot |
| Decision log | Append-only table | Per batch | Audit and override analysis | Restricted | Contains prescriber identifiers |
| Arithmetic smoke | This document, Section 6 | 2026-11-10 | Recompute worked numbers | Public within course | Teaching values only |
| Commercial source brief | `00_SOURCE_COVERAGE_EDGE_INDUSTRY.md`, `#pharma-commercial` | Visible scope only | Topic assignment | Public | Source bodies not available; see limitation below |

The assigned source entries for the commercial brief are named as planned topics in
the coverage file: field force, medical affairs, market access and patient services,
plus commercialisation agentic material. Those files were not available in this
environment. The consequence is precise and bounded: this case's *topic scope* was
informed by the visible topic names, while every mechanism, number and recommendation
is authored from first principles. No claim in this file should be read as a summary
of, or as agreement with, any unavailable source text. That gap is recorded as OPEN
and remains OPEN after authoring, because access was not obtained.

Takeaway: the method is only as trustworthy as its `as_of` enforcement and its
crosswalk cardinality, both of which are checked mechanically rather than by intent.

## 6. Analysis Walkthrough

### 6.1 Decisive result first

The decisive result is a coverage-overlap tradeoff, not an accuracy number. Under the
constructed pilot, the two-model estimator produces a mean estimated effect of 1.8
incremental prescriptions per contacted prescriber per four weeks in the top capacity
tier, against 0.4 in the baseline tier, while its estimated effect in the bottom tier
is negative. That pattern is what makes allocation valuable: effect varies by tier,
and the baseline selects a tier that mixes high- and low-effect prescribers.

The same result also contains its own warning. The estimator's standard error in the
top tier is 0.9, so the interval spans roughly 0.0 to 3.6. The pilot does not
establish a reliable per-prescriber effect; it establishes that the effect is not
obviously homogeneous. A decision to deploy on this evidence alone would be
unsupported.

### 6.2 Step ledger

| Step | Hypothesis or decision | Input snapshot | Method | Intermediate output | Interpretation | Status |
|---|---|---|---|---|---|---|
| 1 | Eligibility accounting is exact | Candidate list, consent register | Set arithmetic and reconciliation | 1,020 of 1,200 eligible | The denominator for everything downstream is fixed | Recomputed below |
| 2 | Baseline selection is concentrated | Prior-quarter volume, territory | Decile rank and truncation | Top decile covers 61 percent of call slots | Baseline is close to pure volume ranking | Derived |
| 3 | Response rate and effect diverge | Pilot randomization | Difference in means by tier | Response 34 percent treated versus 26 percent control in tier 1 | Effect is smaller than response | Derived |
| 4 | Effect heterogeneity exists | Same | Tier-wise differences with standard errors | Tier 1 effect 1.8, tier 3 effect 0.4 | Allocation by effect differs from allocation by response | Derived, wide intervals |
| 5 | Capacity truncation binds | Capacity 600, eligible 1,020 | Greedy allocation under capacity | 600 selected, 420 unselected | Selection is a policy, not a prediction | Recomputed |
| 6 | Overlap is adequate for ranking | Propensity of being contacted | Overlap check in the treated support | 78 percent of eligible prescribers within common support | Ranking is defensible inside support only | Derived |
| 7 | Economic value ordering matches effect ordering | Margin per prescription | Effect times margin | Ordering preserved under constant margin | Constant-margin assumption is load-bearing | Derived |
| 8 | Plan is explainable | Reason code generation | Rule-based attribution of top features | Two reason codes per recommendation | Explanation exists but is not causal | Derived |

### 6.3 Step 1 recomputation: eligibility accounting

The candidate population is 1,200 prescribers. Exclusions are 40 without consent or
outside the approved purpose, 20 outside the target specialty, and 120 with fewer than
26 observed weeks.

```text
excluded_total = 40 + 20 + 120 = 180 prescribers
eligible       = 1200 - 180 = 1020 prescribers
eligible_share = 1020 / 1200 = 0.85 = 85.0 percent
```

Independent check: 180 divided by 1,200 is 0.15, and 1 - 0.15 equals 0.85. The two
routes agree exactly. A third check uses the counts directly: 1,200 times 0.85 equals
1,020, which recovers the eligible count.

The important interpretation is not the percentage. It is that the three exclusion
reasons have different owners and different recheck triggers. Consent exclusions can
reverse within a day. Specialty exclusions reverse only if the brand's indication
changes. History exclusions reverse automatically as weeks accumulate. Treating them
as one bucket hides the fact that the eligible population changes for three unrelated
reasons, and a scoring batch that assumes a stable denominator will produce phantom
week-over-week variation.

### 6.4 Step 3 recomputation: response versus effect

The pilot randomizes contact at the prescriber level within two regions. In tier 1
(highest prior volume), 100 prescribers are randomized 50 to contact and 50 to no
contact. Over four weeks, the contacted group records 17 prescribers with at least one
brand prescription, and the control group records 13.

```text
treated_response = 17 / 50 = 0.34
control_response = 13 / 50 = 0.26
response_difference = 0.34 - 0.26 = 0.08 = 8 percentage points
relative_difference = 0.08 / 0.26 = 0.3077 = 30.77 percent
```

The 8-point difference is the estimated effect on the response indicator. Under
randomization, and assuming no interference, this is an unbiased estimate of the
average effect on response probability for tier 1 in this window. It is not a
prescription count, and it is not a claim about any real brand.

Now translate to prescriptions. Contacted prescribers average 6.4 prescriptions over
the four weeks; uncontacted average 5.4.

```text
treated_mean = 6.4 prescriptions
control_mean = 5.4 prescriptions
effect_on_count = 6.4 - 5.4 = 1.0 prescription per prescriber per four weeks
```

Note the internal tension a careful reader should notice: an 8-point response
difference and a 1.0 count difference are both plausible, but they are different
quantities with different variances. The count difference has a much larger standard
error relative to its mean because counts are heavy-tailed. That is why the case
reports both and refuses to convert one into the other by assuming a fixed count per
responder.

### 6.5 Step 4 recomputation: tier heterogeneity

| Tier | Treated mean (scripts / 4 weeks) | Control mean | Effect | Standard error of effect | Implied interval |
|---|---|---|---|---|---|
| 1 | 6.4 | 5.4 | 1.0 | 0.9 | -0.8 to 2.8 |
| 2 | 4.1 | 3.5 | 0.6 | 0.5 | -0.4 to 1.6 |
| 3 | 2.2 | 1.8 | 0.4 | 0.25 | -0.1 to 0.9 |
| 4 | 1.3 | 1.5 | -0.2 | 0.15 | -0.5 to 0.1 |

The tier 1 interval includes zero. That is the honest result of a 100-prescriber
pilot. The pattern is suggestive of heterogeneity, and the pattern is exactly what a
targeting programme would need, but the evidence does not reach a defensible
per-tier conclusion. Anyone who reports "tier 1 delivers 1.0 incremental
prescriptions" without the interval has overstated the pilot.

Combining tiers with weights proportional to eligible counts (tiers of 300, 300, 260
and 160 prescribers) gives a weighted mean effect.

```text
weights: 300, 300, 260, 160 ; total = 1020
numerator = 300*1.0 + 300*0.6 + 260*0.4 + 160*(-0.2)
          = 300 + 180 + 104 - 32 = 552
weighted_mean = 552 / 1020 = 0.5412 prescriptions per prescriber per four weeks
```

Independent check: the four effects average 0.45 under equal weights, and the
weighted value of 0.5412 is higher because the larger-effect tiers carry more weight.
Both are correct for different questions: the equal-weight figure answers "what
happens to a randomly chosen tier", the weighted figure answers "what happens to a
randomly chosen prescriber".

### 6.6 Step 5 recomputation: allocation arithmetic

Weekly capacity is 600 calls across 40 representatives at 15 each. Eligible
prescribers number 1,020, so the selection rate is:

```text
selection_rate = 600 / 1020 = 0.5882 = 58.82 percent
unselected      = 1020 - 600 = 420 prescribers per week
```

The selection rate above one half is a design fact with a consequence: the model is
not choosing a small elite, it is choosing slightly more than half the population.
Ranking precision differences in the middle of the distribution therefore matter
more than discrimination at the very top, which is the opposite of the intuition
carried over from fraud or anomaly detection work.

Across a 12-week quarter:

```text
quarterly_calls = 600 * 12 = 7200 calls
calls_per_eligible = 7200 / 1020 = 7.0588 calls per prescriber per quarter
```

A prescriber in the eligible population can expect about seven calls in a quarter if
all capacity is used at the selection rate above. That number disciplines the effect
story: an effect estimate per contact must be interpreted against roughly seven
contacts, not against one.

### 6.7 Step 6 recomputation: overlap

For each eligible prescriber, a propensity of being contacted is estimated from
history. Common support is defined as the overlap between the propensity distributions
of contacted and uncontacted prescribers, evaluated in deciles.

```text
propensity deciles with both arms present: 8 of 10
eligible prescribers inside support: 800 of 1020
support_coverage = 800 / 1020 = 0.7843 = 78.43 percent
outside_support = 220 prescribers
```

The 220 outside-support prescribers cannot be reliably ranked by an effect estimate
because there is no comparable untreated observation. Their handling is a policy
decision, not a technical one, and Section 8c records the choice: they are appended
after the supported ranking in volume order, with an explicit
`insufficient_support` reason code.

### 6.8 Step 7 recomputation: economic ordering

If margin per incremental prescription is constant at 42 currency units, effect
ordering and value ordering coincide because multiplication by a positive constant is
monotone.

```text
value_tier1 = 1.0 * 42 = 42.0 currency units per prescriber per four weeks
value_tier3 = 0.4 * 42 = 16.8 currency units per prescriber per four weeks
```

Now break the constant-margin assumption. Suppose tier 1 prescribers operate in a
channel with net margin 28 and tier 3 prescribers in a channel with net margin 55.

```text
value_tier1_revised = 1.0 * 28 = 28.0
value_tier3_revised = 0.4 * 55 = 22.0
```

The ordering is preserved here but the gap narrows from 25.2 to 6.0 currency units. A
slightly larger margin difference reverses the order. This is why A-10 is in the risk
register: a constant-margin assumption is convenient and fragile, and it is the kind
of assumption that is rarely revisited until someone notices the field force
systematically ignoring high-value channels.

### 6.9 Step 8: reason codes

Each recommendation carries up to two reason codes drawn from an approved taxonomy:
`high_effect_estimate`, `recent_access_change`, `low_recent_contact`,
`practice_growth_signal`, `insufficient_support`. Reason codes are descriptive
attributions of the model's inputs, not causal explanations. A recommendation
labelled `practice_growth_signal` does not mean practice growth causes prescribing;
it means that input contributed positively to the score.

The distinction matters for compliance and for trust. A reason code that reads like a
causal claim invites representatives to repeat it as a clinical statement. The
taxonomy is therefore phrased in terms of the analytics, not in terms of medicine.

### 6.10 What was tried first and why it did not work

The prior propensity model was actually tested in a limited rollout and rejected. Its
top-ranked prescribers overlapped the baseline's top decile by 84 percent, so it
reproduced the existing allocation while adding monitoring burden. The measured
overlap is the evidence, not an impression: when two rankings agree in the top decile
at 84 percent, the second one is not adding a decision.

Overlap was computed as the share of the model's top 600 that also appeared in the
baseline's top 600.

```text
overlap_count = 504
overlap_share = 504 / 600 = 0.84 = 84 percent
```

The stopping rule was agreed in advance: if top-capacity overlap exceeded 75 percent,
the model would not be deployed regardless of its AUC. This is a useful pattern to
copy, because it makes "the model is not adding a decision" an explicit, pre-declared
failure rather than a post-hoc judgement.

### 6.11 Sensitivity and robustness

| Stress test | Original condition | Alternative | Change in conclusion | Limitation | Decision consequence |
|---|---|---|---|---|---|
| Capacity | 600 calls/week | 480 calls/week | Selection rate falls from 58.8 to 47.1 percent; ranking precision matters more at the margin | Arithmetic only | Tighten eligibility before cutting capacity |
| Capacity | 600 calls/week | 900 calls/week | Selection rate rises to 88.2 percent; nearly everyone is called, so ranking value collapses | Arithmetic only | Reconsider whether targeting is worth the cost |
| Effect homogeneity | Tier effects 1.0, 0.6, 0.4, -0.2 | All tiers 0.5 | Allocation value over volume rank falls to zero | Constructed | Abandon targeting, keep volume rank |
| Margin | Constant 42 | 28 versus 55 by channel | Ordering preserved but gap narrows to 6 units | Constructed | Adopt channel-specific margin table |
| Support | 78.4 percent inside support | 55 percent inside support | Ranking covers barely half the eligible population | Constructed | Restrict programme to supported prescribers |
| Override rate | 20 percent | 45 percent | Realized allocation diverges from recommended allocation | Constructed | Treat as workflow failure, not model failure |
| Data lag | 3-5 weeks | 8 weeks | Point-in-time features become stale; effect estimate degrades | Constructed | Add staleness penalty and recheck A-01 |

The capacity sensitivity deserves emphasis because it is counterintuitive to
commercial audiences. Increasing capacity does not monotonically increase the value of
targeting; it decreases it. At 900 calls per week, 88 percent of the eligible
population is contacted, and the difference between a good ranking and a mediocre one
shrinks toward zero. Targeting earns its keep under scarcity.

The homogeneity sensitivity is the decisive one. If all tiers had the same effect,
the value of effect-based ranking over volume ranking would be zero, and the entire
programme would be an expensive way to reproduce a decile report. This is the
scenario to test first, not last.

### 6.12 Highest-leverage step and a branch that changes the answer

The highest-leverage step is Step 3, the randomized pilot. All downstream numbers
inherit its uncertainty, and no amount of model refinement compensates for a
contaminated pilot. If the pilot had been assigned at the territory level rather than
the prescriber level, interference within practices would have inflated the control
group's response and biased the effect toward zero; the design would then have
concluded "no effect" while an effect existed.

A branch where another reasonable choice changes the answer is the unit of
randomization itself. Prescriber-level randomization maximizes statistical efficiency
but violates the no-interference assumption when prescribers share a practice.
Practice-level randomization is more credible and roughly triples the required sample
for the same power. This case chooses prescriber-level randomization for the pilot and
adds a practice-clustered sensitivity analysis; a more conservative team would choose
practice-level and accept a longer pilot.

### 6.13 Observation, interpretation and identification

Observation: in the pilot, contacted prescribers recorded 6.4 versus 5.4 prescriptions
per four weeks in tier 1.

Interpretation: contact may raise prescribing, or the contacted group may differ in
unmeasured ways despite randomization failing, or representatives may have
supplemented contact with other actions.

Identification: the causal reading depends on successful randomization, no
interference, and no differential measurement of the outcome. The case asserts the
first, flags the second as a sensitivity analysis, and notes that prescription data is
recorded identically for both arms, which supports the third.

Takeaway: the pilot's job is to falsify homogeneity, and the most valuable output of
this walkthrough is the boundary at which targeting stops being worth doing.

## 7. The Finding

The recommended decision is to run the effect-based allocation as a two-region pilot
with representative override enabled, keep the volume-decile plan as the concurrent
comparator in the remaining regions, and treat the programme as unproven until the
pilot's incremental-prescriptions-per-call estimate separates from zero.

The supported magnitude is modest and imprecise. Within the constructed pilot, tier 1
contacted prescribers recorded 1.0 more prescriptions over four weeks than uncontacted
prescribers, with a standard error of 0.9, which is a difference that cannot be
distinguished from zero at conventional confidence. The tier ordering is consistent
with heterogeneity, and the weighted mean across tiers is 0.54 prescriptions per
prescriber per four weeks, but that figure inherits the same wide intervals and must
not be quoted as a realized gain.

The uncertainty has two parts that should not be merged. Statistical uncertainty from
a small pilot is the visible part; the identification uncertainty from prescriber-level
randomization inside shared practices is the part that would survive a larger sample.
Evidence scope is limited to two regions, one brand, one country and one quarter. The
pilot says nothing about other therapeutic areas, about long-run prescribing habits, or
about whether representatives would sustain a changed allocation beyond the novelty
period.

The main boundary is the capacity-scarcity condition. Targeting earns its value at 600
calls against 1,020 eligible prescribers, where selection rate is 58.8 percent.
Above roughly 85 percent selection rate the ranking contribution approaches zero and
the programme should be discontinued on economic grounds rather than refined.

The next justified action is not deployment. It is a practice-clustered re-analysis of
the pilot and a power calculation for a quarter-length expansion, both of which are
specified in Section 9. If the re-analysis preserves the tier ordering with narrower
intervals, expansion is justified; if it collapses the ordering, the case closes with a
negative result, which is a legitimate and useful outcome.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

```text
[Weekly warehouse snapshot, frozen 20:00 Sunday]
        |
        v
[Point-in-time feature build: as_of = decision_at]
        |
        +--> [maturity violation] --> [block batch + alert data steward]
        |
        v
[Eligibility filter: consent, specialty, history >= 26 weeks]
        |
        +--> [excluded HCP] --> [reason code written to log, not scored]
        |
        v
[Effect model: m0, m1 trained on matured labelled weeks]
        |
        v
[Effect estimate tau(x) + interval per HCP]
        |
        v
[Overlap classification: supported | insufficient_support]
        |
        v
[Constrained allocation: greedy by tau * margin, 15 per representative]
        |
        v
[Reason code generation from approved taxonomy]
        |
        v
[Model registry approval gate: metrics + diagnostics recorded]
        |
        v
[Emit ranked plan JSON + decision log append]
```

The offline path runs once per week and takes minutes, not hours, because scoring is
a batch over 1,020 rows. That is a deliberate design consequence of the capacity
constraint: the system never needs low-latency inference, only reliable batch
completion before the Sunday deadline.

Training cadence is monthly, not weekly. Retraining weekly on 12 new weeks of data
would make the model unstable for no statistical benefit, because the effect
estimates have standard errors of order 0.5 and would move randomly week to week.
Monthly retraining with a fixed feature set keeps the plan comparable across weeks,
which is what the field force needs in order to trust it.

The registry gate is not ceremonial. A model version is promoted only when four
artefacts are attached: the maturity check result, the eligibility reconciliation, the
overlap coverage figure, and the top-capacity overlap against the baseline. A model
that fails the overlap check is blocked regardless of its internal metrics.

### 8b. Online / Serving / Real-Time Path

There is no per-request real-time path for the recommendation itself. The online path
is the representative's planning session, which reads a precomputed plan.

```text
[Representative opens planning tool, Monday 07:00]
        |
        v
[Authenticate + load territory plan version for this week]
        |
        +--> [plan missing or stale] --> [show baseline decile list + banner]
        |
        v
[Render ordered list with reason codes and confidence bands]
        |
        v
[Representative action: accept | swap | skip]
        |
        +--> [swap within allowance] --> [log override + reason + timestamp]
        +--> [swap beyond allowance] --> [route to regional manager approval]
        |
        v
[Planned call records written to CRM]
        |
        v
[End-of-week reconciliation: planned vs delivered vs disposition]
```

Three behaviours at the boundary matter more than the happy path.

First, stale plan handling. If the plan version is older than the current week, the
tool must not silently present it. It shows the baseline list with a banner naming the
missing plan, because a stale model plan is worse than a fresh simple one: it carries
false authority.

Second, override allowance. The tool permits swaps up to a configured count per
representative per week. Beyond that, the swap requires manager approval per C-04.
The purpose is not to prevent judgement but to make large deviations visible, since
large deviations are the signal that either the model or the local knowledge is wrong.

Third, missing prescriber handling. When a representative searches for a prescriber who
is not on the list, the tool distinguishes "not eligible" from "not selected". Those
are different facts with different implications, and conflating them is how governance
exclusions leak back into the workflow.

Peak load is trivial in absolute terms: 40 representatives loading a plan within a
two-hour window is 40 requests, not 40 thousand. The interesting load is the weekly
batch, which must complete within a maintenance window and is dominated by the feature
build, not by scoring.

Cold start applies to new representatives and new territories, not to the service. A
representative with no history receives the baseline decile plan for their territory
with an explicit `cold_start` flag, and their overrides are logged separately so they
do not contaminate the effect estimates.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Effect estimate from two models | Propensity only; bandit | Pilot heterogeneity | Needs overlap; more complex | Overlap falls below 60 percent | Support diagnostics |
| DEC-02 | Prescriber-level randomization | Practice-level | Statistical efficiency | Interference risk | Practice clustering detected | Sensitivity result |
| DEC-03 | Greedy allocation | Optimal assignment with routing | Simplicity, explainability | Ignores travel time | Routes exceed feasible distance | Routing integration |
| DEC-04 | Advisory recommendation | Mandatory list | Field trust, autonomy | Dilution of effect | Override rate above 30 percent | Override monitoring |
| DEC-05 | Monthly retraining | Weekly; quarterly | Stability | Slower adaptation | Structural market change | Drift alarm |
| DEC-06 | Append unsupported prescribers in volume order | Exclude them; random order | Transparency | Inconsistent treatment | Unsupported share above 40 percent | Support monitoring |
| DEC-07 | Reason codes from fixed taxonomy | Free-text explanations | Compliance safety | Less expressive | Representatives ignore codes | Adoption study |

DEC-03 is worth defending. An optimal assignment that accounts for travel time would
require geospatial routing, representative home locations, and a solver. The gain
would be real but small relative to the effect-ordering gain, and the complexity would
obscure the recommendation's logic. The honest position is that routing optimization is
deferred, not dismissed, and the trigger for revisiting it is a measured routing
inefficiency rather than a preference for sophistication.

DEC-06 is a compromise that a stricter design would reject. Appending unsupported
prescribers in volume order means the bottom of the list is not effect-ranked. The
alternative, excluding them entirely, reduces capacity utilization below 100 percent,
which the operating policy does not permit. The chosen compromise is transparent in
the reason code, so the inconsistency is visible rather than hidden.

### 8d. Architecture artifacts and operational readiness

| Area | Answer for this design |
|---|---|
| Interfaces | Plan JSON with schema version, week key, territory, ranked entries, reason codes, confidence bands, model version, and a plan hash. Idempotency key is `week + territory + model_version`. |
| Data and state | Warehouse snapshots are immutable per week. The decision log is append-only. Territory alignment is versioned and joined by version, never by "latest". |
| Reliability | SLI is the share of territories with a published plan by Sunday 23:00. Target 99 percent monthly. Error budget consumption triggers a manual fallback to baseline. |
| Resilience | Feature build has a hard timeout; on timeout the batch publishes the baseline plan rather than a partial model plan. Retries are bounded to one, because a retried feature build on a stale snapshot is worse than a fallback. |
| Recovery | The unit of recovery is one weekly plan. Replay uses the recorded warehouse snapshot identifier, so a regenerated plan is byte-identical in ranking given the same model version. |
| Observability | Dashboards for maturity violations, eligibility reconciliation, overlap coverage, override rate by representative, plan publication time, and realized-versus-recommended divergence. |
| Deployment | Model version and plan schema version deploy independently. A schema change requires a planning-tool release first, then the batch, never the reverse. |
| Security | Prescriber identifiers are restricted to the commercial environment. Plan payloads are scoped to the representative's territory. Access is logged per session. |
| Economics | Cost per plan is dominated by feature build compute and analyst review time, not by inference. The material cost is field time, which is the resource being allocated. |
| Human workflow | Regional manager approves swaps beyond allowance. Compliance can veto a data source. The analytics lead owns the model but not the plan. |

Capacity and sizing arithmetic for the batch path:

```text
rows_scored          = 1020 HCPs per week
features_per_row     = 180
feature_cells        = 1020 * 180 = 183600 cells per week
bytes_per_cell       = 8 (float64)
feature_matrix_bytes = 183600 * 8 = 1468800 bytes = 1.4009 MiB
quarterly_snapshots  = 12
quarterly_bytes      = 1468800 * 12 = 17625600 bytes = 16.81 MiB
```

The feature matrix is small enough to fit in memory many times over, so the batch is
not compute-bound; it is bound by upstream data availability. That is the correct
diagnosis of why the plan sometimes publishes late, and it points at the data steward's
refresh schedule rather than at model optimization.

Cost per useful outcome, using illustrative values:

```text
analyst_hours_per_week   = 6
loaded_hourly_cost       = 55 currency units
weekly_analysis_cost     = 6 * 55 = 330
quarterly_analysis_cost  = 330 * 12 = 3960
quarterly_calls          = 7200
cost_per_call_of_analysis= 3960 / 7200 = 0.55 currency units per call
```

Against a margin of 42 currency units per incremental prescription and an effect of
0.03 prescriptions per call at the low end of plausible values, the analysis cost per
call is small relative to plausible value. That comparison is fragile in exactly one
place: if the effect is zero, the analysis cost is pure loss, which is why the pilot
gate exists before the quarterly spend.

Takeaway: the architecture is small, the hard parts are data clocks and human
authority, and the economics hinge on whether the effect is nonzero at all.

## 9. Evaluation & Validation

### 9.1 Primary metric

| Field | Definition |
|---|---|
| Metric name | Incremental prescriptions per delivered call |
| Numerator | Sum over contacted prescribers of (observed prescriptions minus counterfactual prescriptions), in prescriptions per four-week window |
| Denominator | Count of delivered calls with a recorded disposition, in the same window and population |
| Unit | Prescriptions per call |
| Population | Eligible prescribers in pilot regions, contacted arm only, excluding prescribers with fewer than 2 delivered calls |
| Aggregation | Sum of prescriber-level effects divided by total delivered calls; not an average of ratios |
| Window | Four weeks after each decision week, summed across the 12-week pilot |
| Comparator | Eligible prescribers in control regions using the volume-decile plan |

The choice of sum-of-effects over sum-of-calls, rather than a mean of per-prescriber
ratios, matters. A mean of ratios weights a prescriber with one call equally with a
prescriber with five, which is wrong for a capacity decision: the denominator is calls.
The sum-of-effects form correctly lets prescribers with more delivered calls contribute
more.

### 9.2 Pass and fail rule

The rule, declared before the pilot:

- Pass: the lower bound of the 95 percent interval on incremental prescriptions per
  call exceeds 0.02 prescriptions per call, and the coverage-equity guardrail holds.
- Inconclusive: the interval includes 0.02 but its upper bound exceeds 0.05.
- Fail: the upper bound is below 0.02, or the coverage guardrail is violated at any
  point.

The threshold 0.02 is derived, not arbitrary. At a margin of 42 currency units and an
analysis cost of 0.55 currency units per call, break-even requires roughly:

```text
break_even_effect = analysis_cost_per_call / margin_per_script
                  = 0.55 / 42 = 0.0131 prescriptions per call
```

Setting the pass threshold above break-even at 0.02 leaves headroom for implementation
overhead not captured in the analyst-cost estimate. The derivation is shown here so a
reviewer can disagree with the input values rather than with the target.

### 9.3 Guardrails

| Guardrail | Definition | Threshold | Owner |
|---|---|---|---|
| Coverage equity | Minimum territory-level share of eligible prescribers receiving at least one call per quarter | 60 percent | Regional director |
| Override rate | Swaps as a share of recommended entries | Below 30 percent | Field operations |
| Plan publication | Share of territories with a plan by Sunday 23:00 | 99 percent monthly | Platform |
| Eligibility integrity | Scored prescribers minus eligible prescribers | Exactly zero | Data steward |
| Consent propagation | Weeks between consent withdrawal and removal from scoring | At most 1 | Compliance |

Coverage equity is the guardrail that protects against the most likely unintended
consequence. An effect-based model will tend to concentrate calls on prescribers with
high estimated effect, which may correlate with practice size and urban location. A
territory whose eligible prescribers are mostly low-effect could be starved of
coverage, and the aggregate metric would not reveal it because it is dominated by
high-volume territories.

### 9.4 Test design

The design is a cluster-randomised comparison with prescriber-level assignment inside
pilot regions and a concurrent baseline in non-pilot regions.

| Element | Specification |
|---|---|
| Randomization unit | Prescriber within pilot region, stratified by prior-volume tier and practice |
| Exposure | Recommendation delivered in planning tool; delivered call recorded with disposition |
| Eligibility | 1,020 eligible prescribers, 600 weekly capacity in pilot regions proportionally |
| Interference | Within-practice spillover; measured in sensitivity analysis, not assumed absent |
| Sample-ratio check | Treated and control counts per tier reconciled weekly; alarm on deviation above 5 percent |
| Stopping rule | Fixed 12-week horizon; no interim peeking except for safety or guardrail breach |
| Blinding | Outcome data is vendor-provided and identical across arms; representatives are not blinded to arm |

The stopping rule deserves a note. Interim looks at an effect estimate with a standard
error of 0.9 will frequently cross zero in either direction, and stopping early on such
a signal is the most reliable way to produce a wrong answer. The fixed horizon is a
protection against the team's own impatience.

### 9.5 Sample, uncertainty and power

A power calculation for detecting an effect of 1.0 prescriptions per prescriber with a
standard deviation of 6 prescriptions per prescriber, at 80 percent power and two-sided
alpha 0.05:

```text
effect_size_d = 1.0 / 6 = 0.1667
required_n_per_arm ≈ 2 * (1.96 + 0.8416)^2 / d^2
                  = 2 * (2.8016)^2 / 0.02778
                  = 2 * 7.849 / 0.02778
                  = 15.698 / 0.02778
                  = 565.1 prescribers per arm
```

The pilot has 50 per arm in tier 1. That is roughly one eleventh of the required
sample, which is consistent with the observed standard error of 0.9 rather than the
assumed 6-unit standard deviation: the per-prescriber variance in the pilot is much
larger than 6, and the pilot's effective precision is far below the design target.

```text
implied_std_from_se = 0.9 * sqrt(50) = 0.9 * 7.0711 = 6.364 prescriptions
```

So the assumed 6 and the implied 6.36 are close, and the shortfall is driven by sample
size, not by an unusual variance. To detect 1.0 with the observed variance requires
about 574 per arm, which at 12 weeks and 600 weekly calls total is not achievable
inside one quarter. The honest conclusion is that the pilot can detect a large effect
and cannot detect a moderate one, and the programme design must account for that.

An alternative formulation uses the total number of calls rather than prescribers,
because the decision metric is per call. With 7,200 quarterly calls split across arms
and roughly 3.5 calls per prescriber, the effective sample is smaller than the call
count suggests; calls within a prescriber are correlated. This is why the case reports
prescriber-level sample sizes rather than call-level ones.

### 9.6 Validity threats

| Threat | Mechanism | Falsification check | Status |
|---|---|---|---|
| Leakage in features | Lagged prescription sums including future weeks | Maturity violation test; max source week audit | Controlled by design, tested |
| Selection into contact | Representatives call prescribers they already favour | Randomization in pilot; propensity overlap in observation | Controlled in pilot, uncontrolled historically |
| Interference | Shared practice members influence each other | Practice-clustered sensitivity analysis | Open |
| Outcome mismeasurement | Panel restatements after the window | Snapshot comparison at 8 weeks | Open |
| Novelty effect | Representatives perform better because the plan is new | Extend pilot; compare early versus late weeks | Open |
| Reporting bias in visits | Representatives log calls they did not make | Reconciliation against GPS or sample audit | Open |
| Denominator drift | Eligibility changes mid-quarter as history accumulates | Weekly reconciliation against fixed eligible list | Controlled |

The novelty effect is the threat most likely to produce a false positive in a real
programme, and it is invisible in a 12-week pilot. A team that expands after a
successful pilot may find the effect decaying in weeks 13 to 26. The only defence is to
plan a longer measurement window before making a permanent organizational commitment,
and to pre-declare what decay would mean.

### 9.7 Executed versus planned

| Evaluation element | Status |
|---|---|
| Arithmetic recomputation of eligibility, effects, allocation, power | Executed as authoring smoke, results in Section 6 |
| Maturity violation test | Executed as a logic test; behaviour described |
| Eligibility reconciliation | Executed as a set-arithmetic check |
| Randomized pilot | Not executed; illustrative values |
| Guardrail measurement | Not executed |
| Practice-clustered sensitivity | Planned |
| Long-run decay measurement | Planned |

Nothing in this file should be read as a measured result. The design is complete enough
to be executed, and the arithmetic is checked, but the pilot is hypothetical.

Takeaway: a pre-declared overlap screen and a break-even-derived pass threshold do more
for decision quality than a more accurate model would.

## 10. Action Taken

No action was taken as of 2026-11-10. This case is an illustrative design exercise.

| Action field | Record |
|---|---|
| Recommendation | Pilot effect-based allocation in two regions with override enabled, retain decile plan elsewhere |
| Approval status | Proposed; no approval recorded |
| Implementer and handoff | Commercial analytics builds scoring batch; field operations configures planning tool; regional director owns rollout |
| Effective scope | Would cover 2 regions, 6 representatives, 1,020 eligible prescribers, 12 weeks |
| Release strategy | Pilot with concurrent comparison, then decision gate |
| Stop and rollback | Stop if plan publication SLI falls below 95 percent for two consecutive weeks, or any guardrail breach; rollback is reverting the plan source to baseline deciles, which requires no schema change |
| Excluded actions | No routing optimization, no message content generation, no expansion beyond pilot regions, no use of prescriber data outside the approved purpose |

The rollback design is deliberately trivial. Because the planning tool already renders
a decile list when a model plan is absent, rollback is a configuration switch, not a
migration. Systems that require a data migration to revert a recommendation cannot be
safely piloted in a regulated commercial environment.

What would need to happen before any real action: compliance confirmation of purpose
limitation (A-02), a documented variance estimate from the pilot (A-05), and a signed
operating agreement with field operations covering override policy (C-04). None of
these is a technical step, and all three are prerequisites rather than follow-ups.

## 11. Outcome & Results

Outcome not yet measured. This section retains the measurement plan rather than
inventing results.

| Metric | Baseline value | Treatment value | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Incremental prescriptions per delivered call | Not established | Not measured | Not measured | Not computable | Not available | Illustrative design | Requires pilot execution |
| Response rate, tier 1 | 0.26 control | 0.34 treated | +0.08 | +30.77 percent | Wide, n=50 per arm | Illustrative arithmetic | Not a production measurement |
| Prescriptions per prescriber, tier 1 | 5.4 | 6.4 | +1.0 | +18.52 percent | SE 0.9 | Illustrative arithmetic | Interval includes zero |
| Coverage equity | Not measured | Not measured | Not measured | Not computable | Not available | Planned guardrail | Requires territory data |
| Override rate | Not measured | Not measured | Not measured | Not computable | Not available | Planned guardrail | Requires tooling |

Source declaration: all values above are constructed teaching values. They are not
measurements of any organization, brand or prescriber population, and they do not
establish realized impact.

The relative change of +18.52 percent is computed as 1.0 divided by 5.4. It is
reported for completeness only. A relative change on a quantity whose absolute
interval includes zero is close to meaningless, and quoting it in a business review
would misrepresent the evidence.

### 11.1 Measurement plan

| Element | Plan |
|---|---|
| Owner role | Commercial analytics lead, with regional director as decision owner |
| Proposed window | 12 weeks pilot, 12 weeks follow-on decay measurement |
| Data required | Delivered call dispositions, matured prescription data at 8 weeks post-window, territory coverage counts, override logs |
| Decision trigger | Pass rule from Section 9.2 evaluated once at week 12, then again at week 24 for decay |
| Escalation | Any guardrail breach escalates to the regional director within one week, independent of the primary metric |
| Reporting format | Interval-first reporting; point estimates never shown without their interval |

The interval-first reporting requirement is a deliberate organizational control. In
commercial reviews, a point estimate of "1.0 incremental prescriptions" travels further
than its interval, and by the time it reaches a brand plan it has become a committed
forecast. Requiring the interval in the same sentence is the cheapest available
protection against that drift.

### 11.2 What would constitute success

Success is not a positive point estimate. Success is a lower interval bound above the
break-even-derived threshold, with coverage equity intact, override rate below
threshold, and the effect persisting into the decay window. Anything less is
inconclusive, and inconclusive is a legitimate outcome that should be reported as
such rather than reframed as a win.

Takeaway: the absence of measured outcomes is stated plainly, and the measurement plan
is specific enough that a successor could execute it without re-deciding the design.

## 12. Failure Modes & What Went Wrong

| Failure ID | Trigger and symptom | Detection / discriminating test | Cause or hypothesis | Impact | Containment | Prevention / owner | Verification |
|---|---|---|---|---|---|---|---|
| F-01 | Offline AUC rises sharply after adding a feature set | Compare feature availability timestamps against decision time; recompute with the feature lagged one extra week | 4-week rolling prescription sum included weeks after the decision | Inflated offline evaluation; wrong prescribers ranked | Freeze model, revert feature set | Structural `as_of` enforcement in transformation layer; owner: analytics lead | Maturity violation test must fail on the offending feature; re-run shows degraded but honest AUC |
| F-02 | Two prescribers show identical, doubled prescription counts | Cardinality check per practice per week; inspect crosswalk change log | Panel-to-CRM crosswalk collapsed two prescribers | Duplicate recommendations; wrong territory assignment | Quarantine the crosswalk batch; re-run with previous crosswalk | Practice-level cardinality assertion; owner: data steward | Re-run produces distinct counts matching panel totals |
| F-03 | Representatives report that top-ranked prescribers "already write everything" | Top-capacity overlap against baseline decile; compare effect versus response ranking | Propensity model used instead of effect model, or effect model degenerate | Programme rejected by field; no behavioural change | Switch to baseline; investigate target definition | Pre-declared overlap screen at 75 percent; owner: analytics lead | Overlap drops below screen and reason codes reflect effect |
| F-04 | Plan publishes Monday 11:00 instead of Sunday 23:00 | Plan publication SLI by territory; trace batch stage timings | Upstream prescription feed arrived late; batch waited without timeout | Representatives plan without recommendations or with stale ones | Publish baseline fallback plan automatically | Batch timeout with explicit fallback; owner: platform | Inject a delayed feed; confirm fallback plan publishes before deadline |
| F-05 | Override rate climbs from 18 to 45 percent over three weeks | Override log by representative and reason | Model ranks prescribers representatives know to be unreachable, or list ignores routing | Realized allocation diverges from recommendation; effect estimate meaningless | Pause expansion; interview representatives; inspect reason distribution | Co-design sessions and routing-aware ordering; owner: field operations | Override rate returns below threshold and reasons shift to legitimate categories |
| F-06 | Coverage equity guardrail breaches in one territory | Territory-level coverage share versus 60 percent threshold | Effect estimates correlate with practice size; small-practice territory starved | Equity harm; relationship damage | Manual coverage floor for the territory | Coverage floor rule applied before truncation; owner: regional director | Territory coverage restored above 60 percent without breaching capacity |
| F-07 | Effect estimate flips sign between two monthly retrains | Compare model versions on identical held-out weeks; inspect training window composition | Small effective sample; high variance; weekly retraining | Unstable plans; loss of trust | Freeze to previous version; extend training window | Monthly retraining cadence; owner: analytics lead | Version-to-version rank correlation above a stability floor |
| F-08 | Reason code text appears in a representative's email as a clinical claim | Audit of exported reason text and call notes | Free-text explanation generation instead of fixed taxonomy | Compliance exposure | Disable free-text generation; retract template | Fixed taxonomy with approved wording; owner: compliance partner | Export audit finds only approved phrases |
| F-09 | Scored prescriber count exceeds eligible count by 12 | Eligibility reconciliation per batch | Consent withdrawal not propagated; history exclusion counted as eligible | Governance breach; scoring of non-consented prescribers | Block batch; regenerate from consent register | Consent reconciliation as a hard gate; owner: data steward | Scored minus eligible equals zero across five consecutive batches |
| F-10 | Prescriber contacted seven times in a quarter despite a three-call policy limit | Call count per prescriber per quarter from CRM | Model re-ranks the same prescriber weekly without a frequency cap | Prescriber friction; wasted capacity | Apply frequency cap and redistribute | Frequency constraint in allocation; owner: field operations | Cap violations drop to zero and capacity is reallocated |

F-01 is the failure most likely to be mistaken for success. A leakage-induced AUC
increase looks like a modelling improvement, passes every unit test, and is only
detectable by a clock audit. The discriminating test is not statistical; it is
temporal. The team must be able to answer, for every feature, "at what wall-clock time
did this value become knowable", and the answer must precede the decision time.

F-02 produces a signature that analysts often misread as a data-quality nuisance. Two
prescribers with doubled counts are not merely a scaling error: they break the
allocation unit, because capacity, coverage and effect are all defined per prescriber.
The correct repair is to quarantine the crosswalk batch and re-run, not to divide the
counts by two.

F-06 is the failure that the aggregate primary metric cannot see. A territory with
starved coverage contributes little to the numerator because it receives few calls, so
it cannot pull the average down much. Equity has to be measured at the territory
level, with its own threshold and its own escalation path.

F-10 is included because it is the most likely operational failure in a repeated
weekly allocation. Without a frequency constraint, a deterministic ranking converges
on the same top prescribers every week, and the quarterly call count per prescriber
grows linearly with the number of weeks. The model has no reason to rotate; rotation is
a policy requirement that must be imposed on the allocation.

### 12.1 Failed mitigations worth recording

A first mitigation attempt for F-05 was to add a free-text field where representatives
could explain overrides. It failed for two reasons: the field was rarely completed, and
when it was, the text could not be aggregated into a diagnostic. The replacement was a
fixed override-reason taxonomy with five options plus a free-text note, which produced
usable counts. This is a small example of a general principle: diagnostics that cannot
be aggregated cannot drive correction.

A second failed mitigation for F-07 was to average two consecutive model versions to
stabilize the plan. Averaging rankings is not order-preserving in general, and the
averaged plan had no interpretable reason codes because the two models disagreed on
which inputs mattered. The replacement was to freeze the version and extend the
training window, which addresses the variance rather than masking it.

Takeaway: the most dangerous failures here are temporal and organizational, and the
correct diagnostic for each is a clock audit or an override distribution, not a better
offline score.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | evaluation-mismatch, with causal-confounding and human-in-the-loop |
| General principle | When a scarce resource must be allocated across candidates, optimize the conditional effect of allocating to a candidate, not the probability of a favourable outcome; and screen the effect ranking against the incumbent ranking before deployment. |
| Preconditions | A capacity ceiling that binds; heterogeneous treatment effects; a way to obtain credible effect estimates (randomization or strong identification); an incumbent baseline worth beating; a human workflow that can override |
| Mechanism | Response probability conflates baseline propensity with treatment effect. Under monotonicity, ranking by response approximates ranking by baseline, reproducing the incumbent and adding cost without adding a decision. Effect-based ranking separates the two, but only where overlap exists. |
| Supporting evidence | The constructed pilot shows tier ordering consistent with heterogeneity and intervals that include zero; the prior propensity model's 84 percent top-capacity overlap with the baseline is the strongest single piece of evidence that response ranking is not a decision |
| Avoid when | Effects are homogeneous; capacity is not binding; overlap is poor; the intervention cannot be randomized or identified; the override rate is so high that realized allocation is unrelated to the recommendation |
| Portability test | Ask: if I ranked by predicted outcome instead of predicted effect, would the top of the list change? If the answer is "barely", the effect model is not needed and the simpler baseline is honest |
| Adaptation | In credit collections, the analogue is ranking by marginal recovery rather than by default probability; in retention, by incremental saves rather than churn risk; in clinical outreach, by preventable events rather than predicted events. Each requires its own identification strategy and capacity constraint |

The mechanism deserves a precise statement because it is frequently misapplied. A
response model estimates `E[Y | X]`. An effect model estimates `E[Y(1) - Y(0) | X]`.
These coincide only when the treatment has no effect on anyone, in which case both are
useless for allocation, or when the treatment effect is constant, in which case
ranking by either is equivalent. In the interesting case, where effects vary, the two
orderings differ, and the difference is exactly the value of the programme.

The portability test is deliberately cheap. It does not require building the effect
model; it requires asking whether reordering would happen at all. A team that cannot
articulate a plausible reordering should not invest in the machinery.

What is structural versus local: the capacity constraint and the need for a
counterfactual are structural and will appear in any allocation problem. The specific
600-call capacity, the two-model estimator, the fixed reason-code taxonomy and the
monthly retraining cadence are local implementation choices that another organization
would reasonably change.

A concrete counterexample where following this lesson would be harmful: a market where
the intervention is a mandated safety communication. There, the allocation is not
discretionary, effect estimation is irrelevant, and applying an uplift model would
introduce a selection mechanism into a process that must be universal. The lesson
applies to discretionary allocation under scarcity, not to obligations.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Actual target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| uses-method | [Cross-domain evidence-to-action sheet](../../Cheat_Sheets/01_Evidence_to_Action_Controls.md) | Point-in-time validity, queue capacity, authority boundary | The sheet generalizes the timing invariant; this case instantiates it for HCP-week grain |
| uses-method | [Course standards, Section 5](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) | Formula anatomy and evidence labelling | Numbers here are illustrative even where arithmetic is exact |
| contrasts-with | [HCP promotional response driver analysis](../../../../11_data_science_domain_specialisations/01_production_ml_case_library/03_kda_solutions_pharma/Case_Studies/01_HCP_Promotional_Response_Driver_Analysis_Launch_Brand.md) | That case explains response drivers; this case decides allocation | Driver importance is not effect; a strong driver can be non-actionable |
| related-planned | Market-access driver analysis (CASE-09-01-002, planned) | Access dynamics as an effect modifier | Not yet authored; plain-text reference only |

The contrast with the promotional-response case is the most instructive link. A driver
analysis answers "what correlates with prescribing", which is genuinely useful for
strategy. It does not answer "who should we visit", because correlation with prescribing
does not establish that a visit changes prescribing. Confusing the two is the failure
this case is built to prevent.

### 14b. What this case leaves open

| Open question | Why evidence is insufficient | Test that would resolve it | Priority | Owner role |
|---|---|---|---|---|
| Is the effect heterogeneous across prescribers, and by how much? | Pilot has 50 per arm and intervals spanning zero | Practice-clustered re-analysis plus a quarter-length expansion | High | Analytics lead |
| Does the effect decay after the novelty period? | No follow-on window measured | 12-week decay measurement | High | Analytics lead |
| How large is within-practice interference? | Prescriber-level randomization cannot measure it directly | Practice-level randomization in a second wave | Medium | Analytics lead |
| What is the realized override rate under real field conditions? | Tooling not deployed | Pilot telemetry | Medium | Field operations |
| Does effect ranking survive channel-specific margins? | Constant-margin assumption untested | Finance margin table by channel | Low | Finance partner |

### 14c. Intense interview pack

#### Q01 / Level 1 / Framing

**Candidate prompt and given facts:** A brand has 600 weekly field calls and 1,020
eligible prescribers. The current plan ranks by prior-quarter prescriptions. State the
decision this system supports and the target quantity it should rank.

**Clarifications to seek:** Whether capacity is hard or elastic; whether the outcome is
volume or margin; the decision frequency.

**Strong answer:** The decision is a weekly constrained allocation of 600 calls across
1,020 candidates. The target quantity is the conditional effect of a call on the
prescriber's prescribing over a defined window, not the probability of prescribing.
Ranking by the probability of prescribing approximates ranking by baseline volume under
monotonicity, which reproduces the incumbent plan. The correct ordering is by
`effect times margin`, truncated at capacity.

**Acceptable alternatives:** If effects are known homogeneous, ranking by expected
outcome is defensible because the orderings coincide. If margin is constant, effect
alone suffices.

**Follow-up:** What changes if capacity rises to 900 calls per week? Selection rate
becomes 88.2 percent and the value of refined ranking approaches zero; the programme
should be reconsidered on economic grounds.

**Red flags:** Answering "rank by predicted prescriptions" without addressing the
counterfactual; treating capacity as a soft preference; describing the model without
naming the decision owner.

**Scoring:** 0 for volume ranking without justification; 1 for naming allocation but
not effect; 2 for effect ranking with the monotonicity reasoning; 3 for additionally
naming the capacity boundary at which targeting stops paying.

**Answer support:** Sections 1, 3.3, 6.11.

#### Q02 / Level 1 / Data

**Candidate prompt and given facts:** Prescription data arrives with a 3-to-5 week lag.
A candidate feature is the 4-week rolling prescription sum. What must be true for it to
be admissible at week `w`?

**Clarifications to seek:** Definition of the decision timestamp; whether the sum is
computed on data through week `w` or later.

**Strong answer:** Every source week referenced by the feature must have
`available_at <= decision_at(w)`, where `decision_at(w)` is Sunday 22:00 at the end of
week `w - 1`. With a 5-week maximum maturity, a 4-week sum admissible at week `w` can
reference at most weeks ending `w - 5`. A sum computed through week `w + 3` contains
the outcome and is leakage.

**Acceptable alternatives:** Using a longer lag and accepting staleness, provided the
staleness is measured and its effect on ranking is reported.

**Follow-up:** How would you enforce this mechanically rather than by review? Declare
`max_source_week_offset` per feature and have the builder compute the maximum
referenced source week, failing the row when the offset is violated.

**Red flags:** "Just make sure the join is correct"; relying on analyst discipline;
treating a mature-looking number as evidence of maturity.

**Scoring:** 0 for accepting the sum; 1 for mentioning lag but not the decision clock;
2 for the timestamp inequality; 3 for the mechanical enforcement design.

**Answer support:** Sections 5a, 5d, F-01.

#### Q03 / Level 2 / Calculation

**Candidate prompt and given facts:** Region has 1,200 candidates. Exclusions: 40
consent or purpose, 20 specialty, 120 insufficient history. Capacity is 600 calls per
week. Compute eligible count, selection rate, and quarterly calls per eligible
prescriber over 12 weeks.

**Clarifications to seek:** Whether exclusions are disjoint; whether capacity is
weekly.

**Strong answer:** Excluded total is 40 + 20 + 120 = 180, so eligible is 1,020 and the
eligible share is 85.0 percent. Selection rate is 600 / 1,020 = 58.82 percent, leaving
420 unselected per week. Quarterly calls are 600 times 12 equals 7,200, which is 7.06
calls per eligible prescriber.

**Check:** 180 / 1,200 = 0.15, and 1 minus 0.15 equals 0.85, matching the direct route.

**Follow-up:** If the history exclusion reversed as new prescribers accumulated data,
what happens to the selection rate? Eligible count rises, selection rate falls, and the
denominator of every rate changes mid-quarter unless the eligible list is versioned.

**Red flags:** Adding overlapping exclusions twice; dividing calls by candidates rather
than eligible; reporting 58.82 percent as "about 59 percent of the market".

**Scoring:** 0 for arithmetic errors; 1 for correct eligible count only; 2 for all
three quantities; 3 for additionally noting denominator instability.

**Answer support:** Section 6.3, 6.6.

#### Q04 / Level 2 / Calculation

**Candidate prompt and given facts:** In tier 1, 50 treated prescribers record mean 6.4
prescriptions over four weeks (SE of the difference 0.9), and 50 controls record 5.4.
Response rates are 34 percent and 26 percent. Interpret the effect and say what the
pilot establishes.

**Clarifications to seek:** Whether assignment was randomized; whether the window is
pre-declared.

**Strong answer:** The count difference is 1.0 prescription per prescriber over four
weeks with an approximate 95 percent interval of 1.0 plus or minus 1.76, giving
approximately -0.76 to 2.76. The response difference is 8 percentage points, or 30.77
percent relative to the control rate of 26 percent. The pilot establishes that the
effect is not precisely estimated; it does not establish a positive effect. The two
quantities differ in variance and must not be converted into each other by assuming a
fixed count per responder.

**Acceptable alternatives:** Reporting a one-sided interval if the hypothesis was
directional and pre-declared, which still includes zero here.

**Follow-up:** How many prescribers per arm would detect 1.0 with a standard deviation
of 6? Approximately 565 per arm at 80 percent power and two-sided alpha 0.05.

**Red flags:** Quoting "1.0 incremental prescriptions" without the interval; treating
the 8-point response difference as a count; calling the result significant.

**Scoring:** 0 for a confident positive claim; 1 for computing 1.0 only; 2 for the
interval; 3 for the power implication and the response-versus-count distinction.

**Answer support:** Sections 6.4, 6.5, 9.5.

#### Q05 / Level 2 / Evaluation

**Candidate prompt and given facts:** The team proposes a pass threshold of 0.02
incremental prescriptions per call. Analysis costs 0.55 currency units per call and
margin is 42 per prescription. Is the threshold justified?

**Clarifications to seek:** Whether the cost figures are complete; whether margin is
constant across channels.

**Strong answer:** Break-even is 0.55 / 42 = 0.0131 prescriptions per call. A threshold
of 0.02 sits above break-even, leaving headroom of 0.0069 prescriptions per call, or
about 0.29 currency units of margin per call. The threshold is defensible because it
requires the estimate to clear break-even with margin for unmodelled overhead. It is
fragile if margin varies by channel, because break-even then varies by channel too.

**Acceptable alternatives:** A channel-specific threshold, computed the same way per
channel, is more precise but harder to communicate.

**Follow-up:** What if the interval's lower bound exceeds break-even but not the
threshold? That is the "inconclusive" band in Section 9.2 and should be reported as
inconclusive, not as a pass.

**Red flags:** Choosing 0.02 because "it looks small"; ignoring that analysis cost per
call depends on call volume; forgetting that the threshold applies to the interval
bound, not the point estimate.

**Scoring:** 0 for accepting or rejecting without arithmetic; 1 for computing
break-even; 2 for comparing thresholds correctly; 3 for the interval-bound rule and
channel caveat.

**Answer support:** Section 9.2.

#### Q06 / Level 1 / Definitions

**Candidate prompt and given facts:** Distinguish overlap, response rate and effect in
this design, and explain why overlap constrains the ranking.

**Clarifications to seek:** Whether overlap is measured on propensity or on feature
support.

**Strong answer:** Response rate is `E[Y | X]` in a single arm. Effect is
`E[Y(1) - Y(0) | X]`, requiring both potential outcomes. Overlap is the region where
both treated and untreated observations exist with comparable `X`. Outside overlap, the
counterfactual is unsupported, so an effect estimate is an extrapolation. In this case
78.43 percent of eligible prescribers lie inside support, and the remaining 220 are
appended in volume order with an explicit reason code.

**Acceptable alternatives:** Restricting the programme to supported prescribers, which
improves validity at the cost of capacity utilization.

**Follow-up:** What if support coverage fell to 55 percent? More than half the eligible
population would be ranked without evidence, and the programme should either narrow its
scope or abandon effect ranking for those prescribers.

**Red flags:** Treating overlap as a model diagnostic rather than an identification
condition; computing effect estimates outside support without disclosure.

**Scoring:** 0 for conflating the three; 1 for definitions only; 2 for the identification
argument; 3 for the handling of unsupported prescribers and its cost.

**Answer support:** Sections 4.4, 6.7, 8c.

#### Q07 / Level 3 / Debugging

**Candidate prompt and given facts:** Offline AUC improves from 0.71 to 0.83 after a
feature refresh, but the field team says the new top-600 list is "the same prescribers
as before". Diagnose.

**Clarifications to seek:** What changed in the refresh; whether the top-capacity
overlap was recomputed.

**Strong answer:** Two competing explanations must be separated. First, leakage: the
refresh may have introduced features containing future information, which raises AUC
and correlates with baseline volume. Second, task mismatch: even without leakage, a
response model's top set converges on high-volume prescribers, so the list legitimately
does not change. The discriminating test is to recompute top-capacity overlap against
the baseline and to run a maturity audit on the refresh. If overlap is high and
maturity is clean, the model is answering the wrong question. If maturity fails, the
AUC is invalid.

**Acceptable alternatives:** If overlap is high and maturity is clean, the correct fix
may be to change the target rather than the features.

**Follow-up:** The team proposes adding more features to differentiate the list. Why is
that unlikely to help? Because the ordering is driven by the target, not by feature
richness; a response target orders by response under almost any feature set.

**Red flags:** Concluding "the model is fine because AUC improved"; tuning
hyperparameters; blaming the field team.

**Scoring:** 0 for accepting the AUC; 1 for naming one hypothesis; 2 for both and a
discriminating test; 3 for the target-versus-features conclusion.

**Answer support:** Sections 6.10, F-01, F-03.

#### Q08 / Level 3 / Workflow

**Candidate prompt and given facts:** Override rate rises from 18 to 45 percent in
three weeks. The model's offline metrics are unchanged. What is happening and what do
you do?

**Clarifications to seek:** What the override reasons are; whether routing changed;
whether a new representative cohort started.

**Strong answer:** Unchanged offline metrics with rising overrides indicates a workflow
mismatch, not a model regression. The likely causes are that the list ignores routing
feasibility, or that it ranks prescribers representatives know to be unreachable, or
that a frequency cap is missing and the same prescribers recur. The discriminating
diagnostic is the distribution of override reasons plus a comparison of recommended
versus delivered calls. The immediate mitigation is to pause expansion, not to retrain.
The correction is to add routing-aware ordering or a frequency constraint, and to
co-design with the field.

**Acceptable alternatives:** If override reasons are legitimate local knowledge, the
correct response may be to treat the recommendation as one input among several and
measure the incremental value of the recommendation relative to representative
judgement.

**Follow-up:** If the override rate stays above 30 percent after correction, what does
that imply for the effect estimate? The realized allocation diverges from the
recommended allocation, so the estimated effect no longer describes what the system
would achieve if deployed.

**Red flags:** Retraining the model; blaming representatives; ignoring that the
delivered call, not the recommendation, is the treatment.

**Scoring:** 0 for retraining; 1 for naming workflow; 2 for the reason-distribution
diagnostic; 3 for the consequence for effect estimation.

**Answer support:** Sections 8b, F-05.

#### Q09 / Level 3 / Changed constraint

**Candidate prompt and given facts:** The region reorganizes mid-quarter, so 12 percent
of prescribers change representative. How does the pilot's validity change?

**Clarifications to seek:** Whether the reorganization respects randomization arms;
whether the comparison regions also reorganize.

**Strong answer:** Reorganization breaks the alignment between the recommendation and
the delivering representative, and it can move a prescriber between arms if the arms
were defined geographically. If arms are prescriber-level and reorganization is
administrative only, the effect estimate survives but the delivered-call denominator
changes. If reorganization changes which arm a prescriber is exposed to, the comparison
is contaminated and the affected prescribers must be excluded or analysed as a
separate stratum.

**Acceptable alternatives:** Freezing the pre-reorganization alignment for analysis
purposes while allowing the operational change, provided the delivered calls are
attributed to the pre-change representative.

**Follow-up:** What guardrail would catch this early? A weekly reconciliation of the
alignment version against the frozen analysis alignment, alarming when the changed
share exceeds 5 percent.

**Red flags:** Assuming administrative changes are analysis-neutral; dropping affected
prescribers without reporting the exclusion rate.

**Scoring:** 0 for ignoring it; 1 for naming contamination; 2 for arm-versus-denominator
distinction; 3 for the freeze-and-attribute design.

**Answer support:** Sections 5a, 9.4, 9.6.

#### Q10 / Level 2 / Design

**Candidate prompt and given facts:** Why not use a contextual bandit instead of a
two-model estimate?

**Clarifications to seek:** Volume per arm; horizon; whether per-prescriber learning is
needed.

**Strong answer:** A bandit needs repeated observations per decision context to
estimate per-context value. Here, 600 weekly calls across 1,020 prescribers means the
average prescriber is contacted roughly every two weeks at best, so effective sample
per prescriber is a handful per quarter. Exploration would consume scarce capacity that
the capacity constraint says is the binding resource. The bandit is not rejected for
being sophisticated; it is rejected because the data volume cannot support
per-prescriber learning within the planning horizon.

**Acceptable alternatives:** A bandit over coarse strata, such as tier times region, is
feasible and would be a reasonable extension after the pilot.

**Follow-up:** At what capacity would a bandit become viable? Roughly when calls per
week substantially exceed the eligible population, so that each prescriber is contacted
often enough to learn from.

**Red flags:** "Bandits are more modern"; ignoring that exploration cost is paid in the
same currency as exploitation; assuming one quarter is enough to converge.

**Scoring:** 0 for preference-based answers; 1 for naming exploration cost; 2 for the
sample-volume argument; 3 for the coarse-strata alternative.

**Answer support:** Section 5c.

#### Q11 / Level 3 / Verification

**Candidate prompt and given facts:** A vendor reports that the programme delivered
"12 percent prescription lift". What do you require before accepting it?

**Clarifications to seek:** Comparator; window; population; whether the comparison was
concurrent.

**Strong answer:** Require the comparator definition, the concurrent or historical
nature of the control, the population and denominator, the window, and whether the
control was subject to the same external shocks. A 12 percent lift against a
pre-period is not evidence if the market grew. Then require the interval and the
analysis population, and check whether the lift is in prescriptions or in
prescriber-reported intent. Finally, check whether coverage, override and denominator
guardrails held.

**Acceptable alternatives:** If only historical comparison exists, the claim should be
downgraded to "consistent with" rather than "delivered".

**Follow-up:** The vendor says the control was "matched prescribers". What additional
check? Whether matching was done on pre-treatment variables only, and whether the
matched control shows parallel pre-trends.

**Red flags:** Accepting a percentage without a denominator; accepting a pre-period
comparison as causal; not asking about the interval.

**Scoring:** 0 for accepting; 1 for asking for a comparator; 2 for the full list;
3 for the pre-trend and guardrail checks.

**Answer support:** Sections 9.1, 9.6, 11.

#### Q12 / Level 3 / Diagnosis

**Candidate prompt and given facts:** The plan is published on time but a territory's
scored count is 12 above its eligible count. What is your sequence?

**Clarifications to seek:** Whether the excess prescribers are consented; whether the
history filter changed.

**Strong answer:** Block the batch first, because non-consented scoring is a governance
breach that cannot be repaired retroactively. Then reconcile: compare the scored list
against the consent register and the specialty list, and check whether the history
exclusion was applied before or after the join. The most likely causes are consent
withdrawal not propagated and a filter applied at the wrong stage. The correction is to
make consent reconciliation a hard gate that fails the batch, and the verification is
five consecutive batches with zero difference.

**Acceptable alternatives:** If the excess is entirely from a legitimate re-inclusion
after accumulated history, the reconciliation definition is wrong rather than the
filter; the fix is to version the eligible list.

**Follow-up:** What does this failure teach about "defensive" scoring of ineligible
prescribers? Imputing scores for governance-excluded prescribers converts a boundary
into a default and is never acceptable.

**Red flags:** Dropping the excess rows to make the count match; treating it as a
cosmetic reconciliation issue.

**Scoring:** 0 for patching the count; 1 for blocking the batch; 2 for the reconciliation
sequence; 3 for the gate and versioning design.

**Answer support:** Section 5d, F-09.

#### Q13 / Level 1 / Authority

**Candidate prompt and given facts:** The model is confident that a prescriber should be
visited. Can the system schedule the visit?

**Clarifications to seek:** What authority the system holds; whether the representative
has approved.

**Strong answer:** No. The recommendation is advisory. The representative executes or
overrides, and a manager approves list changes above the configured threshold. Model
confidence is an estimate about an outcome, not a grant of authority. Scheduling
without human approval would also bypass the content-approval constraint, because a
scheduled visit implies a message.

**Acceptable alternatives:** Automated scheduling within a pre-approved plan, where a
human approved the plan structure in advance, is defensible if the individual content is
approved and the override path remains available.

**Follow-up:** What must be logged for the audit trail? Inputs hash, model version,
rank, reason codes, override decision, override reason and timestamps.

**Red flags:** Treating confidence as permission; conflating recommendation with
execution.

**Scoring:** 0 for yes; 1 for no without reason; 2 for the authority argument;
3 for the pre-approved-plan alternative.

**Answer support:** Sections 4.2, 8b, 10.

#### Q14 / Level 2 / Economics

**Candidate prompt and given facts:** Margin is 42 per prescription in one channel and
55 in another. Tier 1 effect is 1.0, tier 3 effect is 0.4. Does the ranking change?

**Clarifications to seek:** Which tier maps to which channel; whether margin is per
incremental prescription.

**Strong answer:** With constant margin 42, the values are 42.0 and 16.8, a gap of 25.2.
With 28 and 55 respectively, values are 28.0 and 22.0, a gap of 6.0, and the ordering
is preserved. If tier 1's channel margin were 20 instead of 28, values would be 20.0 and
22.0 and the ordering would reverse. So the constant-margin assumption is load-bearing
and must be tested, not assumed.

**Acceptable alternatives:** Using contribution margin after discounts and rebates,
which is the economically correct quantity but requires finance input.

**Follow-up:** How would you present this to a brand director? Show the reordering
threshold: the margin ratio at which the order flips, which here is 55 / 42 = 1.31.

**Red flags:** Assuming gross price equals margin; ignoring channel mix; computing
relative lift on values without stating the base.

**Scoring:** 0 for no computation; 1 for the constant-margin case; 2 for the revised
case; 3 for the reversal threshold.

**Answer support:** Section 6.8, A-10.

#### Q15 / Level 3 / Transfer

**Candidate prompt and given facts:** Apply this case's lesson to a patient-adherence
outreach programme with 2,000 patients and 400 outreach calls per month.

**Clarifications to seek:** Whether the intervention is discretionary; what the outcome
window is; whether consent covers outreach.

**Strong answer:** The structural analogy holds: 400 monthly calls across 2,000 patients
is a binding capacity constraint, so the target should be the incremental effect of
outreach on adherence, not the predicted risk of non-adherence. The differences matter:
adherence is measured through claims with a long lag, so the outcome window must be
declared before scoring; the population includes vulnerable patients, so the coverage
guardrail is a care-equity requirement rather than a commercial one; and consent rules
are stricter. A risk model would concentrate outreach on the highest-risk patients, who
may be the least responsive and the most costly to reach.

**Acceptable alternatives:** If outreach capacity exceeds the at-risk population, a
risk-based rule is sufficient and effect estimation is unnecessary.

**Follow-up:** Which assumption from this case is most likely to fail in the adherence
setting? The no-interference assumption, because patients share households and
clinicians.

**Red flags:** Copying the capacity number; ignoring the lag structure; treating
adherence risk as effect.

**Scoring:** 0 for a superficial analogy; 1 for the effect-target transfer; 2 for the
differences; 3 for the interference and consent analysis.

**Answer support:** Sections 3.3, 13, 4.1.

#### Q16 / Level 2 / Measurement

**Candidate prompt and given facts:** Why is the primary metric expressed as
sum-of-effects over sum-of-calls rather than a mean of per-prescriber ratios?

**Clarifications to seek:** Whether prescribers receive equal numbers of calls.

**Strong answer:** Calls are the scarce resource, so the metric must be per call. A mean
of per-prescriber ratios weights a prescriber with one call equally with one who
received five, which misstates the value of capacity. Sum-of-effects over sum-of-calls
weights each call correctly. The trade-off is that prescribers with many calls dominate
the estimate, which is appropriate for a capacity decision but means the metric is not
a per-prescriber average and should not be reported as one.

**Acceptable alternatives:** Reporting both, with the ratio-of-sums as primary and the
mean-of-ratios as a secondary descriptive statistic.

**Follow-up:** What happens to the metric if the allocation increases the variance in
calls per prescriber? The ratio-of-sums becomes more sensitive to high-call prescribers,
so the interval should account for clustering.

**Red flags:** Averaging ratios; reporting prescriptions per prescriber when the
decision is per call.

**Scoring:** 0 for the ratio mean; 1 for naming calls as the denominator; 2 for the
weighting argument; 3 for the clustering caveat.

**Answer support:** Section 9.1.

#### Q17 / Level 3 / Judgement

**Candidate prompt and given facts:** The pilot is inconclusive at week 12. The brand
director wants to expand anyway because "the trend is positive". What do you advise?

**Clarifications to seek:** What the interval bounds are; whether the guardrails held;
what the cost of a wrong expansion is.

**Strong answer:** Advise against expansion on trend language and propose the cheapest
decisive test instead. Expansion converts an inconclusive result into an unmeasured
programme, because once everyone receives the new plan there is no comparator. The
decisive option is to extend the pilot in the same two regions with practice-clustered
analysis, or to expand to one additional region while retaining a matched comparison.
If the effect is real, a quarter of delay costs little; if it is not, expansion locks in
field time and organizational commitment.

**Acceptable alternatives:** If the capacity is otherwise idle and the intervention is
low-risk, expanding with a holdback group is a reasonable compromise.

**Follow-up:** What would you need to see at week 24 to justify full expansion? A lower
interval bound above the break-even-derived 0.02, with guardrails intact and no decay
in the second window.

**Red flags:** Deferring to seniority; calling the trend evidence; expanding without a
holdback.

**Scoring:** 0 for agreeing; 1 for refusing without an alternative; 2 for the extended
pilot; 3 for the holdback design and the cost-of-delay reasoning.

**Answer support:** Sections 9.2, 11.1, 11.2.

#### Q18 / Level 2 / Failure

**Candidate prompt and given facts:** A prescriber was contacted seven times in a
quarter under a three-call policy limit. Which failure is this and how do you prevent
it?

**Clarifications to seek:** Whether the limit is a policy or a recommendation; whether
the plan regenerated the same prescriber weekly.

**Strong answer:** This is a policy-constraint failure in the allocation, not a data
error. A deterministic weekly ranking re-selects the same high-scoring prescribers, so
call counts grow with weeks. The fix is to add a frequency constraint to the
allocation, so that prescribers at the cap are excluded and capacity is redistributed to
the next-ranked candidates. Prevention is a weekly cap check before publication, and
verification is a quarter of zero cap violations with capacity utilization maintained.

**Acceptable alternatives:** A soft penalty that reduces the score of recently
contacted prescribers, which smooths rotation but does not guarantee the cap.

**Follow-up:** What does this constraint do to the effect estimate? It changes the
treatment each prescriber receives, so the estimate must be conditioned on the cap or
measured under the capped policy.

**Red flags:** Blaming the representative; treating it as a data-quality issue; adding
the cap after publication.

**Scoring:** 0 for no diagnosis; 1 for naming the cap; 2 for the allocation constraint;
3 for the effect-estimate consequence.

**Answer support:** Section 12, F-10.

Takeaway: the interview pack tests whether a candidate can defend a counterfactual
target, an interval-first evaluation, and a human authority boundary under pressure.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Pharma commercial analytics; field-force targeting and next-best-action for a branded medicine |
| Problem type | Constrained allocation and ranking under a hard weekly capacity ceiling |
| Method | Two-model conditional effect estimate with overlap gate, greedy capacity-constrained allocation, advisory recommendation with reason codes; baseline is prior-quarter volume decile rank |
| Primary result | Not measured. Illustrative pilot shows tier effect 1.0 (SE 0.9) versus 0.4 (SE 0.25), consistent with heterogeneity but not statistically separating from zero |
| Key lesson | Rank by the effect of allocating, not by the probability of the outcome; screen the new ranking's top-capacity overlap against the incumbent before deploying |
| Tags | evaluation-mismatch (primary); causal-confounding; human-in-the-loop |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-09-01-001 / 1.0 / 2026-11-10 |
| Evidence level | illustrative; all values constructed, arithmetic independently recomputed |
| Implementation stage | proposed; no pilot executed, no approval recorded |
| Baseline / primary metric | Prior-quarter volume decile rank; incremental prescriptions per delivered call, 4-week window, 12-week horizon |
| Key limitation | Pilot power is roughly one eleventh of the requirement; identification assumes no interference, untested |
| Review trigger / owner | Capacity policy change, consent or purpose change, or any guardrail breach; commercial analytics lead with regional director |
