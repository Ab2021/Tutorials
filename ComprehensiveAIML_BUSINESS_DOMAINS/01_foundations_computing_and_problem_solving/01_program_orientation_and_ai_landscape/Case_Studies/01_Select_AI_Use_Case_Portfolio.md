---
case_id: "CASE-01-001"
case_title: "Selecting an AI use-case portfolio under a reviewer-time cap and an authority gate"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "AI product portfolio planning"
primary_pattern: "evidence-to-operation decision"
secondary_patterns: ["human-in-the-loop", "capacity-gated sequencing"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "portfolio steering owner"
source_index: "INDEX.md"
review_trigger: "Change in eligible volume, routing fraction, review service time, reviewer headcount, harm cost, or any proposed data or action boundary"
---

# Selecting an AI Use-Case Portfolio Under a Reviewer-Time Cap and an Authority Gate

**Evidence declaration.** Every quantity in this file is either constructed for teaching or derived arithmetically from constructed inputs. Nothing here is a measurement of an organization, a vendor outcome, a deployment, or a procurement result. The setting, the three proposals, the reviewer team, the arrival profile, the error rates and the cost rates are all hypothetical. The arithmetic is checked independently and labelled `DERIVED`; the parameter values are labelled `ILLUSTRATIVE`. Where a question cannot be settled by arithmetic, the label is `OPEN`. Section 5e records how the arithmetic was verified and what it cannot establish.

## 1. The One-Line Framing

Given **three workflow interventions described at the level of one request**, decide **which ones enter a time-boxed pilot** for a portfolio steering owner, subject to **a cap on committed reviewer minutes per day, a tail-spillover gate, and a rule that no request may change an account state without recorded human approval**, compared with **the current fully manual workflow and a "fund by enthusiasm" ranking**.

The unit of analysis is one eligible request. The decision is made once, at the start of a 30-day pilot window, and is revisited only when an observed quantity crosses a stated threshold. The output is not one ranked list but three different verdicts: fund now, fund only after a redesign, and do not fund — plus one measurement track that carries no pilot authority.

The reusable pattern is **evidence-to-operation decision**: a proposal earns operating authority by clearing independent gates in a fixed order, and only the survivors are ranked by expected value. The secondary patterns are **human-in-the-loop** — the reviewer is a capacity-consuming component with a service-time distribution, not a rubber stamp — and **capacity-gated sequencing**, where a portfolio is assembled against a budget of minutes rather than a budget of money.

The surprising conclusion, and the reason this case exists, is that **the highest estimated net value among the three proposals is infeasible, and the second-highest gross benefit is the strongest fundable option only after its routing policy is redesigned**. A reviewer-time cap does not merely delay work; it changes which proposal is correct. A portfolio chosen by value alone picks an option that saturates the review team at its own mean load and overflows on any busy day.

The obvious simpler framing — "score each proposal, pilot the top one" — misses three things that decide the answer here. First, reviewer minutes are not interchangeable with money: they cannot be bought back at the margin within a 30-day window. Second, the mean is the wrong statistic when a stream can consume the entire team at its average, because a system running at 100% of capacity on average is over capacity on most days. Third, one of the proposals is disqualified by an authority boundary that no amount of benefit can offset, so the comparison set must be narrowed before any ranking is meaningful.

```text
inputs                                  decision              constraints            baseline
------                                  --------              -----------            --------
eligible volume per stream       --->   fund / redesign /     committed reviewer     current manual
routing fraction q                      defer / reject        minutes per day        workflow, no
service time t_rev per review     --->   + sequencing across   p95 spillover gate     measurement of
time saved per item                     a 30-day window       authority boundary     review load
error rate x loss per error                                                           at all
fixed run cost
```

A reader should be able to restate the decision in thirty seconds without naming a tool: *decide which three candidate workflows enter a 30-day pilot when the review team can absorb a bounded number of minutes per day and one candidate would act without approval.*

## 2. Context & Background

### 2.1 The operating environment

A fictional operations group handles **2,000 eligible requests per day** across three queues: written responses to customer questions, inbound document triage, and account maintenance. The group is staffed by operations specialists who perform each request end to end. A separate review function, six people on eight-hour shifts, exists to check work that leaves the group and to authorise anything that changes a customer record.

The six reviewers are not idle. They currently review escalations, handle exceptions and audit a sample of specialist work. The case models **75% availability after those scheduled non-pilot obligations**, yielding 6 × 480 × 0.75 = **2,160 reviewer-minutes per day** as the gross pilot window. A separate 65% commitment policy holds back part of that window for unplanned interruptions and recovery; it does not subtract the scheduled work a second time. This figure is a scenario assumption, not a time study; Section 4 records how it would be measured.

Three proposals arrive in the same budget cycle. Each is described by its sponsor in terms of a capability, and each must be converted into a decision contract before it can be compared.

| Proposal | Sponsor's description | What it actually does to a request | Requested authority |
|---|---|---|---|
| **A — drafting assist** | "an assistant that writes the first version of the reply" | proposes response text; a reviewer edits and sends | none beyond reading the request |
| **B — document triage** | "automatic routing of inbound documents to the right queue" | classifies each document and routes it, with review of the routed set | changes internal queue assignment |
| **C — account maintenance** | "automated handling of routine account changes" | executes the change directly, no review step | changes customer account state |

### 2.2 The workflow, before and after

Before intervention, a request is read by a specialist, processed, and either sent or escalated. Review happens on a sample and on every escalation. The elapsed time is long but the process has an authorisation trail: every outbound message and every account change has a named owner.

After intervention, the request path changes shape, and the changes differ by proposal. A inserts a generation step and moves the reviewer from author to editor. B inserts a classification step whose output is an internal routing decision, and moves the reviewer from processor to auditor of the routed set. C removes the human step entirely from the routine path and replaces it with an automated state transition.

That last difference is the whole case. A and B produce artefacts that a person can refuse. C produces an effect. The proof burden is not comparable, and no benefit figure converts an effect into an artefact.

### 2.3 Stakeholders and their interests

| Role | Interest in the outcome | What they can veto | What they cannot see |
|---|---|---|---|
| Portfolio steering owner | a defensible, reversible allocation of the pilot window | funding any proposal | daily review-minute consumption unless instrumented |
| Operations lead | throughput without new backlogs | routing changes inside the group | the review team's true availability |
| Review lead | predictable load and a defensible approval record | any change to what counts as reviewed | the sponsor's value model |
| Risk and compliance partner | no unapproved state change; authorised data only | any proposal crossing the authority boundary | engineering feasibility |
| Sponsors of A, B and C | their own proposal survives | nothing formally, but they shape the estimates | the capacity constraint's arithmetic |
| Customers | correct answers and correct account state | — (churn, complaints, redress) | that any of this is happening |

### 2.4 Why the current process persists

The manual workflow survives despite being slow because it supplies three things that no proposal in this case supplies on its own: an authorisation trail, a correction mechanism, and a person who can be asked why. Review capacity is the operational expression of those three properties. When a team adds automation without adding review capacity, it is not removing a bottleneck; it is relocating the bottleneck to a smaller, more expensive group and calling the result a throughput gain.

The group has also never instrumented review time. Nobody can say what a review actually costs in minutes, so every sponsor's benefit estimate is denominated in a unit that the organisation does not measure. That gap is the second thread of the case: a portfolio decision made on unmeasured reviewer minutes is a decision made on the sponsor's optimism.

### 2.5 Scale, timing and boundaries

The modelled window is 30 consecutive operating days. The eligible population is the 2,000 requests per day described above; there is no seasonality in the model, which is a simplification that Section 6.9 stress-tests. Geography, language and channel are held constant. The pilot is single-site. No vendor is selected by this decision, and no contract is signed.

The timing matters in one specific way: review capacity for a 30-day window is effectively fixed. Reviewers cannot be hired, trained and productive inside the window, and review work cannot be deferred indefinitely because a request has a customer waiting on the other end. Capacity is therefore a hard constraint on the portfolio, not a cost line that scales with demand.

## 3. The Business Question

### 3.1 The decision contract

| Contract field | This case |
|---|---|
| Question | Which of the three proposals may enter the 30-day pilot, in what sequence, without committing more reviewer minutes than the team can sustain and without permitting an unapproved state change? |
| Decision owner | portfolio steering owner, with the review lead holding a veto on load and the risk partner holding a veto on authority |
| Trigger and deadline | the annual budget cycle closes in three weeks; two sponsors have already begun building |
| Available actions | fund A only; fund B only; fund C only; fund any feasible combination; fund none; defer a proposal to a measurement track with no operating authority |
| Value and downside | freed specialist minutes and faster handling on one side; review backlog, correction work, customer harm and an unauthorised state change on the other |
| Objective | maximise expected net value per day across funded proposals, after hard gates eliminate infeasible ones |
| Hard constraints | committed review load ≤ 65% of available reviewer minutes; 95th-percentile load ≤ 90%; no state change without a recorded approval; only authorised data |
| Scope | 2,000 requests per day, one site, 30 operating days, six reviewers at 75% availability |
| Out of scope | vendor selection, legal determination, pricing, hiring decisions, multi-site rollout |

### 3.2 Why the objective is not the same as the proxy

The objective is net value, and net value is computed from three proxies: minutes of specialist time freed, a valuation of those minutes, and an expected harm term. Each proxy can move without the objective moving.

Freed minutes overstate value when the queue is not the binding constraint — if specialists are idle in the afternoon, an hour freed at 10 a.m. is worth less than an hour freed at 4 p.m. The dollar value per minute is a management accounting convention, not a market price. The harm term is an expected value over a distribution that has never been estimated at this site, so it carries more uncertainty than its point estimate suggests.

The guardrail is the review-load constraint itself. A proposal whose value depends on pushing review load toward the cap is a proposal whose value depends on a state the team cannot sustain, so the capacity gates are doing double duty: they filter infeasible options and they bound how much of the value estimate can be believed.

### 3.3 What an expensive wrong decision looks like

Four distinct failures are expensive here, and they are not the same failure.

Funding B as specified consumes the entire team at its own mean, so every busy day leaves a backlog that carries into the next day, and the operations lead discovers the problem only when escalations stop being reviewed.

Funding C produces a small number of unauthorised state changes that must be reversed and disclosed, and the cost of one such event exceeds the annual value of the proposal at any plausible error rate the team could measure in 30 days.

Funding nothing costs the organisation the learning it needs for the next budget cycle and leaves review time unmeasured for another year.

Funding A alone is safe and small; its risk is not harm but the opportunity cost of leaving the highest-density option (B after redesign) unfunded because the comparison set was never redesigned.

### 3.4 The question a sponsor would ask, and the answer shape

A sponsor asks: "my proposal has the largest benefit; why is it not funded?" The answer is not "because it is risky." The answer is a specific infeasibility with a number attached, followed by the smallest design change that makes the proposal fundable, followed by the arithmetic showing the redesigned version is better than the option that displaced it. Section 6 performs that sequence; Section 7 states the conclusion.

## 4. Assumptions & Constraints

### 4.1 The assumption register

| ID | Assumption or constraint | Type | How it would be verified | Status | If wrong | Fallback | Recheck trigger |
|---|---|---|---|---|---|---|---|
| `A-01` | 2,000 eligible requests per day | workload | arrival counts by queue over four weeks | unverified, illustrative | every capacity figure scales linearly | re-size from observed arrivals | volume moves ±10% |
| `A-02` | 30% of eligible requests fall in A's draftable set | scope | eligibility rule applied to a frozen sample | unverified, illustrative | A's pool and net value change proportionally | narrow A's scope | eligibility rule or mix changes |
| `A-03` | 40% of eligible requests fall in B's triage set | scope | same as `A-02` | unverified, illustrative | B's load and value scale | narrow B | mix changes |
| `A-04` | Reviewer availability is 75% of paid hours | capacity | time study over ten days | unverified, illustrative | the whole capacity budget moves | re-derive from the time study | meeting load or shift pattern changes |
| `A-05` | A review takes 4 minutes on average | service time | stopwatch sample of 35 reviews | unverified, illustrative | A's load and net value change | instrument before wave 1 | sample mean outside ±0.5 min |
| `A-06` | B' review takes 5 minutes on average | service time | sample of 200 routed items | unverified, illustrative | B' density falls | instrument | sample mean outside ±0.5 min |
| `A-07` | B's routed fraction is 45% as specified | policy | read the routing specification | stated by sponsor | B's load is set by it | renegotiate the threshold | threshold changes |
| `A-08` | A reviewer misses a defect in 0.2% of A reviews | quality | stratified audit of released outputs | unverified, illustrative | A's net value moves by the harm term | tighten rubric | audit rate outside the interval |
| `A-09` | A defective A output costs $60 to correct | harm | correction ledger | unverified, illustrative | A's net value changes | re-price | correction cost changes |
| `A-10` | A defective C' action costs $2,000 | harm | redress and reversal ledger | unverified, illustrative | C' net value changes sign | stop C' | cost estimate revised |
| `A-11` | A specialist minute is worth $0.50 | economics | management accounting rate | convention | all values scale | restate per unit | rate policy changes |
| `A-12` | A reviewer minute is worth $0.65 | economics | fully loaded rate ÷ productive minutes | convention | densities and shadow price scale | restate | pay or availability changes |
| `C-01` | Committed review load ≤ 65% of available minutes | capacity | policy decision, this case | by construction | portfolio shrinks or grows | adjust the policy, not the arithmetic | review lead changes the policy |
| `C-02` | 95th-percentile load ≤ 90% of available minutes | capacity | compound-Poisson calculation | by construction | tail-spillover risk returns | add pacing or capacity | arrival or service variability changes |
| `C-03` | No account state change without a recorded approval | authority | design review and audit log | required, non-negotiable | an unacceptable effect occurs | proposal-only mode | any change to authority |
| `C-04` | Only authorised data may be read | data boundary | data-flow review | required | the proposal is disqualified | narrow the data scope | any new source |
| `C-05` | Review capacity is fixed for the 30-day window | capacity | hiring lead time | consequence of the window | the portfolio grows | extend the window | window length changes |
| `C-06` | A daily audit reserve of 120 minutes | capacity | policy | by construction | effective capacity falls by 120 | fund the reserve explicitly | audit policy changes |

### 4.2 The three assumptions most likely to reverse the decision

The routing fraction in `A-07`, the reviewer availability in `A-04`, and the harm rate in `A-08` are the three that can flip the answer on their own, and Section 6.12 shows the arithmetic for each.

Routing fraction moves the load of the proposal that fails the tail gate, so it is the variable a redesign operates on. When the fraction falls from 0.45 to 0.25, the same proposal moves from 100% of capacity to 46%, while its net value rises from $1,496 to $1,882 per day because the $754 reduction in review cost exceeds the $288 increase in expected harm. That asymmetry — load falls faster than value — is the entire economic case for redesign.

Availability sets the denominator of every capacity statement. If the true figure is 60% rather than 75%, available minutes fall from 2,160 to 1,728, the 65% gate falls from 1,404 to 1,123, and the recommended portfolio no longer fits. The case would then have to choose between funding A and funding B', not both.

Harm rate is the term that most easily changes a sign. At A's full 600-item scope, net value reaches zero at a defect rate of 1.48%, 7.4 times the assumed 0.2%; at the recommended 266-item scope, its fixed cost makes the break-even 0.61%. C's proposal-only net value reaches zero at 0.13%, which is close enough to the assumed 0.1% that the decision cannot rest on the point estimate.

### 4.3 Constraints that are not negotiable, and why

The authority constraint `C-03` is not a parameter. It is the boundary between a proposal that a person can refuse and one that a person cannot. No benefit figure in this case is large enough to compensate for crossing it, because the harm term for an unauthorised state change is bounded by the number of affected customers multiplied by the cost of redress, and the case's own arithmetic shows that a 1% error rate on 40 daily changes already produces $800 of expected daily harm against $360 of gross benefit.

The data-boundary constraint `C-04` is similarly absolute, but it is easy to violate accidentally: a drafting assistant that has seen a document in a retrieval index has read data it may not be authorised to read. The gate is therefore about the proposal's data path, not its output.

The capacity constraints `C-01` and `C-02` are policy, not physics. They could be set at 80% and 100% respectively, and the portfolio would change. The case sets them where it does because a review team that is committed above roughly two thirds of its available time has no room for the escalations that the rest of its job requires, and because a stream whose 95th percentile exceeds the hard cap will spill over on one day in twenty.

### 4.4 Constraints the team controls

Review capacity looks external but is partly a choice: the team could reduce its audit sample, defer a lower-priority review category, or pool reviewers across skills. Pacing of arrivals is also a choice: a proposal can release work in a controlled hourly profile instead of at once. Section 6.8 shows that pacing alone converts a daily 720-minute backlog into a 61-minute transient, which is a design decision worth more than a headcount increase in this case.

The routing fraction is a choice too, and it is the one the case recommends changing. The distinction matters for the interview: a candidate who treats the 45% routing as physics will conclude the proposal is infeasible and stop; a candidate who treats it as policy will redesign it and find the best option in the portfolio.

## 5. Data, Tools & Method

### 5a. Data Specification

This is not a model-training case, so there is no training set, no label set and no split. The "dataset" is a decision ledger: one row per proposal, with columns that are either policy parameters, measured quantities, or illustrative estimates. The grain is **one proposal per decision**, expanded to **one request per day** when computing load.

| Field | Grain | Meaning | Units | Availability at decision time | Status |
|---|---|---|---|---|---|
| `stream_id` | proposal | stable identifier A, B, C | — | at intake | defined |
| `V` | proposal | eligible request volume for the stream | requests/day | estimated from queue counts | `ILLUSTRATIVE` |
| `q` | proposal | fraction of eligible requests that reach a reviewer | dimensionless [0,1] | set by routing policy | policy for B; structural for A and C |
| `t_rev` | proposal | reviewer service time per routed request | reviewer-minutes | must be sampled; unknown before instrumentation | `ILLUSTRATIVE` |
| `t_base` | proposal | specialist minutes per request under the current process | specialist-minutes | estimated from the manual workflow | `ILLUSTRATIVE` |
| `t_post` | proposal | specialist minutes per request after the intervention | specialist-minutes | estimated | `ILLUSTRATIVE` |
| `p_err` | proposal | probability a released output is defective | dimensionless | must be audited | `ILLUSTRATIVE` |
| `L` | proposal | cost of one defective release | dollars | from correction and redress ledgers | `ILLUSTRATIVE` |
| `F` | proposal | fixed daily run cost (engineering, monitoring, licences) | dollars/day | from a build estimate | `ILLUSTRATIVE` |
| `authority` | proposal | whether the proposal can change external state | categorical: artefact / internal / external | at design review | design fact |

**Volume and window.** 2,000 eligible requests per day for 30 operating days, giving 60,000 request-days in the window. Of these, 30% (600/day, 18,000 in the window) fall in A's draftable set and 40% (800/day, 24,000) fall in B's triage set. C's set is 40/day (1,200 in the window) because only routine changes are in scope.

**Inputs and their availability timestamps.** The three parameters that decide the answer — `q`, `t_rev` and `p_err` — are exactly the three that are hardest to know before the pilot. This is not an accident of the case; it is the structural problem of portfolio selection. The decision must therefore be made under uncertainty about its own most sensitive inputs, which is why Section 9 specifies instrumentation as part of the intervention rather than as an afterthought.

**Population and exclusions.** Ineligible requests are those outside the declared set: non-routine account changes, documents from a channel the classifier was not scoped for, and requests already escalated. Duplicate submissions are excluded by a stable request identifier. The case does not model the excluded population, and the exclusion rule must be frozen before outcomes are inspected, or the eligible set becomes a post-hoc choice that flatters whichever proposal is being defended.

**Join semantics.** A request appears once in the volume count and at most once in the review ledger, keyed by request identifier. Retries must reuse the identifier; if they do not, a retried review becomes a second review, load is overstated, and the capacity arithmetic is wrong in the conservative direction for load but the optimistic direction for value.

**Splits.** There is no train/test split. The analogous control is a **time split**: the 30-day window is analysed in three ten-day blocks so that a drifting routing fraction or service time is visible before the window ends. Section 9 makes the ten-day block the reporting unit for exactly this reason.

**Absent data, stated plainly.** The organisation has no measured review-time distribution, no defect-rate estimate, no correction ledger by proposal, and no time study of reviewer availability. Every one of those is `unverified` in Section 4.1. The case proceeds because a portfolio decision has to be made anyway, and because the gates can be designed to fail safe when the estimates are wrong.

### 5b. Tools & Stack

The analytical layer is arithmetic and a small queueing calculation, reproducible in a spreadsheet. Nothing here requires a modelling platform, and the case deliberately avoids one: a portfolio decision that depends on a bespoke optimiser is a decision that cannot be re-run by the person who has to defend it.

| Component | Role in this case | Status |
|---|---|---|
| Decision ledger (spreadsheet or versioned table) | holds `V`, `q`, `t_rev`, `p_err`, `L`, `F` per proposal with evidence labels | proposed; not built |
| Capacity arithmetic | converts routed volume and service time into committed reviewer minutes | performed in Section 6 |
| Gate test | applies authority, commitment and tail tests in fixed order | performed in Section 6 |
| Density ranking | orders feasible options by net value per committed reviewer minute | performed in Section 6 |
| Queue check (Erlang C) | estimates waiting time for a pooled or dedicated review team | performed in Section 6.7 |
| Pacing controller (proposed production) | releases routed work in an hourly profile rather than at once | proposed; not built |
| Review console with immutable ledger (proposed production) | records reviewer identity, decision, output version and approval identifier | proposed; not built |
| Audit sampler (proposed production) | draws a stratified sample of released outputs for defect adjudication | proposed; not built |
| Kill switch (proposed production) | reverts a stream to the manual path within one operating day | proposed; not built |

The three production components are described in Section 8. None exists. A reader should not infer from the presence of this table that a system was built, instrumented or tested.

### 5c. Method Summary

The method is a five-stage sequence, and the order is the method. Reordering it changes the answer.

```text
stage 1  authority and data gate      -> disqualifies, cannot be offset by value
stage 2  load arithmetic               -> committed reviewer minutes per stream per day
stage 3  commitment gate               -> mean load share against a 65% cap
stage 4  tail gate                     -> 95th-percentile load share against a 90% cap
stage 5  density ranking under the cap -> fill remaining capacity by value per minute
```

**Why the gates precede the ranking.** A ranking is a comparison, and a comparison requires a common measure. Value per day is a common measure across proposals, but it is the wrong measure for a proposal that crosses an authority boundary, because the harm from that crossing is not a term in the proposal's value — it is a term in the organisation's obligations. Putting the authority test inside a weighted score lets a large benefit buy its way past a boundary, which is exactly the failure the gate exists to prevent. Section 6.6 demonstrates this with numbers: a weighted score ranks the disqualifiable proposal first.

**Baseline.** The baseline is the current manual workflow with the review function doing what it does today. It is a real option, not a straw man: it costs nothing to continue, it carries no new harm, and it is the comparator against which freed minutes are counted. Any proposal that cannot beat the baseline on net value after its review cost should not be funded, and "fund nothing" is a legitimate output of the method.

**Alternatives considered and their disposition.**

| Candidate method | Evidence considered | Strength | Limitation | Disposition |
|---|---|---|---|---|
| Fund by gross estimated benefit | sponsor estimates | fast, familiar, no instrumentation | ignores capacity and authority; selects the disqualified proposal | **rejected** — demonstrated failure in 6.6 |
| Fund by benefit/cost ratio on money only | same estimates plus a build cost | normalises for size | treats reviewer minutes as free, so every proposal looks affordable | **rejected** — a ratio cannot express a hard cap |
| Weighted score across value, risk, feasibility | same plus a risk rating | balances competing concerns | compensates a non-compensable failure | **rejected** — see 6.6 |
| Sequential gates, then density ranking | the same estimates, restated per reviewer minute | preserves the boundary; makes the scarce resource explicit; auditable | requires a service-time estimate that the site does not have | **chosen** |
| Fund nothing and instrument for one quarter | none | no downside risk; produces the missing data | no learning on the intervention itself; review time still unmeasured | **considered, kept as the fallback** |
| Hire reviewers first, then fund everything | headcount lead time | removes the capacity constraint | adds $312/day per reviewer for value that does not cover it | **rejected** — 6.11 shows the arithmetic |

**The algorithmic core, stated without tool names.** Stage 2 is a product of a volume, a fraction and a service time. Stage 3 and 4 are comparisons of that product against two thresholds. Stage 5 is a two-constraint packing problem with one constraint relaxed, which in this instance has a knapsack structure: maximise the sum of per-stream net values subject to the sum of per-stream minutes being no greater than the remaining capacity, with each stream either funded at a chosen scope or not funded. Because each stream's scope is itself a choice variable (A can be funded at 600 items or at 266), the problem is a small integer program with a monotone objective in scope, and it is solved here by enumeration of the eight subsets with an inner search over A's scope.

**Termination and correctness of the enumeration.** With three proposals there are 2³ = 8 subsets. For each subset the inner scope choice is monotone — net value increases with scope until the capacity bound binds — so the optimum for a subset is found by setting each stream's scope to its maximum feasible value. The enumeration therefore terminates after eight capacity tests and one inner division, and it is exhaustive over the subset space. It is not exhaustive over fractional scopes, which is why Section 6.11 computes the linear-programming relaxation as a bound and reports the gap.

### 5d. Data provenance, quality, exclusions and leakage

**Provenance.** All parameter values originate in this document. They are constructed to be internally consistent and arithmetically interesting, not to represent any organisation. The queueing formulae are standard results, cited in Section 10 with their conditions. The statistical formulae for sample size and the rule of three are standard results, likewise cited.

| Source ID | Content | Type | Accessibility | What it can support | What it cannot |
|---|---|---|---|---|---|
| `SRC-01` | this file, Sections 4 and 5a | constructed parameter set | full | the arithmetic in Section 6 | any organisation's workload or cost |
| `SRC-02` | sibling sheet `01_AI_Use_Case_Charter.md` | method reference | local | charter fields, gate hierarchy | measured values |
| `SRC-03` | sibling sheet `02_AI_Market_Map.md` | method reference | local | review-feasibility and cost-per-output forms | measured values |
| `SRC-04` | sibling sheet `03_Technology_Due_Diligence.md` | method reference | local | rule of three, sample sizing, pass gates | measured values |
| `SRC-05` | Erlang C / M/M/c waiting-time result | standard result | public | the queue check in 6.7 under its assumptions | real service-time behaviour on this site |
| `RUN-01` | hand arithmetic and independent recomputation | derived | reproducible | every figure quoted | anything about an organisation |

**Quality and known defects in the inputs.** The service-time estimates `t_rev` are point estimates with no variance attached, so the tail gate in stage 4 has to assume a variance. The case uses a compound-Poisson variance in which each routed request contributes a review with variance `t_rev²`, which is the variance implied by a random number of reviews each of fixed duration. That understates real variance, because service time varies per review. Section 6.9 therefore reports the sensitivity to a doubled service-time standard deviation: the tail gate is not close to binding in the recommended portfolio, so the understatement does not reverse the decision. It would reverse it for the proposal that fails the commitment gate anyway.

**Exclusions.** Excluded from the analysis: specialist queueing effects (specialists are not modelled as a constrained resource), rework loops after a correction (a corrected request is counted once), reviewer learning during the window (service time is treated as constant), and the cost of building the three production components (only the fixed daily run cost `F` is counted). Each exclusion biases the analysis in a stated direction, and Section 12 records them.

**Leakage controls.** Three leakage paths matter here, and none is a machine-learning leakage path in the usual sense.

The first is **outcome leakage into eligibility**. If the eligible set is defined after the pilot by removing requests that the system handled badly, the measured defect rate is biased downward and the harm term is understated. The eligibility rule must be frozen and versioned before the window opens.

The second is **resolution leakage into routing**. If the routing policy can see a field that is populated only after a specialist resolves the request, then the routing decision is not available at the time it must be made in production, and the measured routed set is optimistic. The routing policy must be restricted to fields with an availability timestamp no later than the request's arrival.

The third is **selection leakage into the harm estimate**. Audit samples must be drawn from released output, not from reviewed output, or the audit measures the reviewer rather than the system. Section 9 specifies the sampling frame.

### 5e. Reproducibility and method-selection ledger

| Artifact | Location | Purpose | Accessibility | Reproduction limit |
|---|---|---|---|---|
| `RUN-01` | this file, Section 6 | full arithmetic trace of every figure | reproducible by hand | arithmetic only; no execution environment |
| `RUN-02` | this file, 6.7 | Erlang C evaluation for four configurations | reproducible with the formula in Section 10 | assumes Poisson arrivals, exponential service, no abandonment |
| `RUN-03` | this file, 6.9 | sensitivity grid over volume, routing fraction, service time, harm rate, availability | reproducible by hand | one-at-a-time variation, not a joint distribution |
| `RUN-04` | this file, 6.11 | enumeration over eight subsets plus LP relaxation | reproducible by hand | three proposals only; larger portfolios need an actual solver |

**Reproduction recipe.** Recompute the capacity budget as headcount × paid hours × 60 × availability. Multiply each stream's volume by its routed fraction and its service time to get committed minutes. Divide by the budget for the load share. Compute the compound-Poisson standard deviation from the routed count and the service time, add 1.645 standard deviations to the mean, and divide by the budget for the tail share. Compute each stream's net value as volume × (baseline minutes − post minutes) × value per specialist minute, minus review minutes × value per reviewer minute, minus volume × defect rate × loss per defect, minus fixed cost. Divide net value by committed minutes for the density. Every figure in Section 6 reproduces from that recipe, and the case states the arithmetic inline so that a reader can check each one.

**Method-selection rationale.** The gate-then-density method was chosen over a weighted score because the case contains a genuinely non-compensable constraint and a genuinely scarce resource. If both were compensable — say, three proposals with no authority differences competing for a budget in dollars — a weighted score or a Lagrangian relaxation would be defensible and simpler. The method is therefore conditional on the structure of the decision, not a universal recommendation. Section 13 states the condition explicitly.

## 6. Analysis Walkthrough

The walkthrough proceeds in the order the method prescribes, and it ends with the answer. The decisive result is stated first: **B as specified is infeasible because it consumes 100% of available reviewer minutes at its own mean, and it fails the tail gate at 108.7%; B after redesign is the highest-density option in the portfolio; A is fundable but only up to a scope that the redesigned B leaves available; C fails the authority gate as specified and is negligible as a proposal-only stream.**

### 6.1 Step 1 — Build the capacity budget

The budget is the denominator of every subsequent statement, so it is computed before any proposal is examined.

```text
FORMULA [CAP-01]: R = n_rev * H * 60 * alpha
WHERE: R = available reviewer minutes per day [reviewer-min/day]
       n_rev = reviewers assigned to pilot routing [people]
       H = paid hours per reviewer per day [h/day/person]
       60 = minutes per hour [min/h]
       alpha = fraction of paid hours available for pilot routing [-]
CONSTRAINT: alpha must be measured, not assumed; scheduled meetings, escalations,
       and existing non-pilot audit work are already inside the unavailable
       fraction. The later beta3 gate reserves additional pilot-window slack
       for unplanned interruptions and recovery, not those same obligations.
       R is a one-day budget; multi-day capacity is R times operating days
       only if no work carries over, which Section 6.8 tests.
NUMERICAL EXAMPLE: n_rev=6, H=8, alpha=0.75;
       R = 6 * 8 * 60 * 0.75 = 6 * 480 * 0.75 = 2880 * 0.75 = 2160 reviewer-min/day.
INTERPRETATION: 2,160 minutes is the entire pilot budget. Every proposal's
       review load is a share of this one number.
```

Two derived thresholds follow immediately. The **commitment gate** leaves 35% of the gross pilot window uncommitted for within-day variation, urgent interruptions, and recovery work, so a proposal may be committed only up to 0.65 × 2,160 = **1,404 minutes**. It does not reserve the scheduled non-pilot work already excluded by `alpha`. The **tail gate** allows the 95th percentile of a day's load to reach 0.90 × 2,160 = **1,944 minutes** before the proposal is treated as a spillover risk. The 120-minute pilot-audit reserve is then carved out of the committed portion, leaving 1,284 minutes for productive routing and 756 minutes uncommitted.

```text
FORMULA [CAP-02]: G3 = beta3 * R,   G4 = beta4 * R,   R_net = G3 - A_res
WHERE: G3 = commitment gate [reviewer-min/day]; beta3 = 0.65 [-]
       G4 = tail gate on a day's load [reviewer-min/day]; beta4 = 0.90 [-]
       A_res = audit reserve [reviewer-min/day]; R_net = routable committed minutes
CONSTRAINT: 0 < beta3 < beta4 < 1; A_res must be funded inside G3, otherwise
       the audit is the first thing dropped under load and the quality signal
       disappears exactly when it is needed.
NUMERICAL EXAMPLE: G3 = 0.65 * 2160 = 1404; G4 = 0.90 * 2160 = 1944;
       A_res = 120; R_net = 1404 - 120 = 1284 reviewer-min/day.
INTERPRETATION: the portfolio is packed against 1,284 productive minutes,
       not against 2,160, and the packing is reported against 1,404.
```

The difference between 1,284 and 2,160 is the single most common error in portfolio sizing: a team that packs against gross capacity believes it has 68% more room than it does. The case keeps both numbers visible in every table so the reader can see which one the constraint is applied to.

### 6.2 Step 2 — Convert each proposal into per-unit parameters

Each proposal is restated as six numbers. The restatement is where most proposals lose their ambiguity, because a sponsor's adjective has to become a volume, a fraction, a duration, a defect rate and a cost.

| Parameter | A drafting assist | B triage, as specified | B' triage, redesigned | C maintenance, as specified | C' maintenance, proposal-only |
|---|---:|---:|---:|---:|---:|
| `V` eligible requests/day | 600 | 800 | 800 | 40 | 40 |
| `q` routed to a reviewer | 0.2667 | 0.45 | 0.25 | 0 | 1.0 |
| reviews/day = `V·q` | 160 | 360 | 200 | 0 | 40 |
| `t_rev` reviewer minutes/review | 4 | 6 | 5 | 0 | 6 |
| committed minutes/day = `V·q·t_rev` | 640 | 2,160 | 1,000 | 0 | 240 |
| `t_base` specialist minutes now | 6 | 9 | 9 | 20 | 20 |
| `t_post` specialist minutes after | 2 | 1 | 1.2 | 2 | 4 |
| `p_err` defective release rate | 0.002 | 0 | 0.06 | 0.01 | 0.001 |
| `L` cost per defective release | $60 | $0 | $6 | $2,000 | $2,000 |
| `F` fixed cost/day | $250 | $300 | $300 | $60 | $60 |
| authority class | artefact | internal | internal | **external** | artefact |

Four of these rows deserve comment because they are where a proposal is usually mis-specified.

**B's routed fraction is a policy, not a measurement.** As specified, B routes 45% of 800 documents, which is 360 reviews. Its designer chose that threshold to protect quality. The threshold is the only parameter in the case that can be changed by a design meeting rather than by an experiment, and that is why the redesign is cheap.

**B's review time falls when the routed set shrinks.** As specified, a reviewer checks the full classification of a routed document, which takes 6 minutes. After redesign, the reviewer checks only the low-confidence cases, where the decision is genuinely open, and the check takes 5 minutes because the routing policy has already narrowed the question. This is a modelling choice and it is labelled as one: a redesigned policy that still required 6 minutes per routed item would produce a net value of $3120 − 200×6×0.65 − 288 − 300 = $1,732 rather than $1,882, which does not change the ranking.

**B's redesigned defect rate rises.** Routing 25% of documents instead of 45% means more documents take the auto-route path, so more routing errors can escape. In this case `p_err=0.06` is explicitly the defect rate across **all 800 released B' outputs**, matching the field definition in Section 5a and the `V·p_err·L` harm formula; it is not a conditional rate on the 600 auto-routed documents. With a $6 correction cost, expected harm is `800 × 0.06 × 6 = $288/day`. That harm is smaller than the $754/day review-cost reduction, but it is a real cost of the redesign. The redesign is correct because it converts an infeasible proposal into a feasible one while improving illustrative net value, not because its error cost is ignored.

**A's displayed routing fraction is rounded.** Its exact constructed fraction is `160/600=4/15`; `0.2667` is a display approximation, not a different policy. At a capped scope, round the required review count upward before checking capacity and cost. Use the exact fraction for continuous comparisons and state separately whether a count is a design commitment or a random arrival model.

**C's specified design has no review step at all.** That is what makes it attractive to its sponsor: 40 changes per day executed without a reviewer, gross value $360/day. It is also what disqualifies it, as 6.4 shows.

### 6.3 Step 3 — Compute net value per stream

Net value has four terms: value of specialist minutes freed, cost of reviewer minutes consumed, expected harm from defective releases, and fixed run cost. The derivation is a sum of four independently estimable quantities, and the case reports each separately so that a disagreement about the conclusion can be localised to one term.

```text
FORMULA [VAL-01]: NV = V*(t_base - t_post)*c_spec - V*q*t_rev*c_rev - V*p_err*L - F
WHERE: NV = net value [$/day]; V = eligible requests/day [requests/day]
       t_base, t_post = specialist minutes per request before/after [min/request]
       c_spec = value of a specialist minute [$/min]
       q = routed fraction [-]; t_rev = reviewer minutes per routed request [min]
       c_rev = value of a reviewer minute [$/min]
       p_err = defective release rate [-]; L = cost per defect [$]
       F = fixed run cost [$/day]
CONSTRAINT: minutes freed must be usable, i.e. the specialist queue must be
       binding; c_spec and c_rev are accounting conventions, not market prices;
       the harm term is an expected value, so NV is a mean and not a bound.
NUMERICAL EXAMPLE (A): V=600, t_base=6, t_post=2, c_spec=0.50,
       q=0.2667, t_rev=4, c_rev=0.65, p_err=0.002, L=60, F=250.
       gross = 600*(6-2)*0.50 = 600*4*0.50 = 1200
       review cost = 160*4*0.65 = 416
       expected harm = 600*0.002*60 = 72
       NV = 1200 - 416 - 72 - 250 = 462 $/day.
INTERPRETATION: A returns $462/day at its full scope, which is 38.5% of its
       gross figure. The review cost alone consumes 34.7% of gross.
```

The same formula applied to the six configurations produces the table that drives the rest of the case. Each row is computed from the parameters in 6.2 and each column can be checked by hand.

| Configuration | Gross $/day | Review cost $/day | Expected harm $/day | Fixed $/day | **Net $/day** | Net per item $ |
|---|---:|---:|---:|---:|---:|---:|
| A at 600 items | 1,200.00 | 416.00 | 72.00 | 250 | **462.00** | 0.770 |
| A capped at 266 items | 532.00 | 184.60 | 31.92 | 250 | **65.48** | 0.246 |
| B as specified | 3,200.00 | 1,404.00 | 0 | 300 | **1,496.00** | 1.870 |
| B' redesigned | 3,120.00 | 650.00 | 288.00 | 300 | **1,882.00** | 2.352 |
| C as specified | 360.00 | 0 | 800.00 | 60 | **−500.00** | −12.50 |
| C' proposal-only | 320.00 | 156.00 | 80.00 | 60 | **24.00** | 0.600 |

Three features of this table decide the case, and none of them is visible in a benefit ranking.

B as specified has the largest gross value of the feasible-authority options at $3,200, and the second-largest net value at $1,496. Its review cost is $1,404, which is 43.9% of gross. That review cost is not an inefficiency to be optimised away; it is the entire capacity budget, which 6.5 shows.

B' has a *smaller* gross value than B ($3,120 against $3,200) and a *larger* net value ($1,882 against $1,496), because it converts $754 of review cost into $288 of expected harm. Trading a reviewer minute for a corrected error is a good trade at these prices, and that trade is only visible when both terms are in the same equation.

C as specified has negative net value (−$500) even before the authority question is asked, because its 1% error rate on 40 daily changes produces $800 of expected daily harm against $360 of gross benefit. C' has positive but tiny net value ($24/day), about 1.3% of the recommended portfolio's net. Its narrow break-even margin motivates a separately governed measurement track rather than a funded pilot stream.

### 6.4 Step 4 — Apply the authority gate

The authority gate is applied before any capacity arithmetic is used to eliminate a proposal, because a proposal that fails it cannot be rescued by anything downstream.

| Proposal | What it produces | Can a person refuse it before it acts? | Gate result |
|---|---|---|---|
| A | draft text | yes — the reviewer edits and sends | pass |
| B, B' | internal queue assignment | yes — the reviewer sees and can re-route the routed set | pass (internal effect, reversible within the day) |
| C as specified | executed account change | **no** — the change is the output | **fail** |
| C' | proposed change for approval | yes — the reviewer approves or rejects | pass, with the approval identifier required in the audit record |

The gate's arithmetic is not a benefit calculation; it is a statement about which failure is recoverable. An A defect produces a wrong sentence that a reviewer can correct and a customer can be told about. A C defect produces a wrong account state that must be found, reversed, and disclosed, and the search cost — finding which of 40 daily changes was wrong — is itself larger than the proposal's gross value.

```text
FORMULA [GATE-01]: admissible(p) = authority_ok(p) AND data_ok(p) AND capacity_ok(p)
WHERE: authority_ok = the proposal's output is refusable before it takes
       external effect, or the effect is reversible inside one operating day
       with a recorded approval identifier
       data_ok = every input has an availability timestamp no later than
       the decision point and an authorisation covering this use
       capacity_ok = mean load <= beta3*R and p95 load <= beta4*R
CONSTRAINT: AND, not a weighted sum. Any single false makes the proposal
       inadmissible regardless of the other two and regardless of NV.
NUMERICAL EXAMPLE: C as specified has authority_ok = false because its output
       is an executed state change; NV = -500. C' has authority_ok = true and
       NV = +24. The gate, not the value, is what changed.
INTERPRETATION: the gate is a conjunction; the ranking is a comparison.
       Mixing them lets value buy an inadmissible proposal.
```

C's expected harm is worth stating as a break-even, because it shows how little the proposal needs to go wrong for the decision to be forced.

```text
FORMULA [HARM-01]: p_err* = (gross - review cost - F) / (V*L)
WHERE: p_err* = defect rate at which net value is zero [-]
CONSTRAINT: all other terms held at their stated values; the harm term is
       linear in p_err, which holds only while defects are independent and
       individually corrected.
NUMERICAL EXAMPLE (C as specified): gross = 360, review cost = 0, F = 60,
       V = 40, L = 2000; p_err* = (360-0-60)/(40*2000) = 300/80000 = 0.00375.
       At the assumed 0.01 the net value is 360 - 800 - 60 = -500.
       (C' proposal-only): (320-156-60)/(40*2000) = 104/80000 = 0.0013.
INTERPRETATION: C as specified destroys value if more than 3 changes in 1,000
       go wrong. C' destroys value if more than 1.3 in 1,000 go wrong — and the
       assumed rate is 1.0 in 1,000, so C' clears its break-even by 30%.
```

That last clause is why C' is a measurement track and not a funded stream. A stream whose net value is $24/day against a break-even defect rate of 0.13% — when the assumed rate is 0.10% — is a stream whose sign is decided by a parameter nobody has measured. Section 9 shows that measuring 0.13% to any precision requires about 2,300 observations, which is 58 operating days at 40 changes per day. The honest disposition is therefore: run it as an audit, not as a value stream.

### 6.5 Step 5 — Apply the commitment gate

The commitment gate compares each stream's mean load against 1,404 minutes. It is a simple division, and it eliminates one proposal outright.

| Configuration | Committed min/day | Share of R = 2,160 | Share of G3 = 1,404 | Gate result | Slack to G3 |
|---|---:|---:|---:|---|---:|
| A at 600 items | 640 | 29.63% | 45.6% | pass | 764 |
| B as specified | 2,160 | **100.00%** | **153.8%** | **fail** | −756 |
| B' redesigned | 1,000 | 46.30% | 71.2% | pass | 404 |
| C as specified | 0 | 0% | 0% | pass on capacity, fails authority | 1,404 |
| C' proposal-only | 240 | 11.11% | 17.1% | pass | 1,164 |
| A + B' | 1,640 | 75.93% | 116.8% | **fail** | −236 |
| A + B' + audit | 1,760 | 81.48% | 125.4% | **fail** | −356 |
| B' + C' + audit | 1,360 | 62.96% | 96.9% | pass | 44 |
| A capped at 266 + B' + audit | 1,404 | 65.00% | 100.0% | pass, exactly at the cap | 0 |
| A + B' + C' + audit | 2,000 | 92.59% | 142.5% | **fail** | −596 |

The B row is the pivot of the case. B's committed load is 2,160 minutes, which is the budget exactly, because 360 reviews × 6 minutes happens to equal 6 × 480 × 0.75. That exact equality is a coincidence of the constructed numbers, but the conclusion it illustrates is not: a proposal that consumes the whole budget at its mean has no room for a busy day.

The A + B' row shows the second-order effect. Two individually admissible proposals are jointly inadmissible, because 640 + 1,000 = 1,640 exceeds 1,404 by 236 minutes. Capacity is not a per-proposal property; it is a property of the portfolio, and it must be re-tested after each addition. This is why the method in 5c tests subsets rather than proposals.

The A + B' + audit row shows the third-order effect, and it is the one that decides the recommendation. Adding the 120-minute audit reserve pushes the combined load from 1,640 to 1,760, which is 356 minutes over the gate. Something has to give. The case's answer is to shrink A rather than drop it, because A's marginal density is high while its average density is low: A's fixed cost of $250/day is amortised over whatever scope survives, so A at 600 items earns $462 but A at 266 items earns only $65.48. Dropping A entirely loses $462; shrinking A to 266 items loses $396.50 and keeps $65.48. Neither is a rounding error, and Section 6.10 shows why shrinking A is nonetheless the right call.

### 6.6 Step 6 — Demonstrate why a weighted score fails here

The case rejects a weighted score, and it owes the reader a demonstration rather than an assertion. Three criteria are scored 0–1 and weighted: value, feasibility and safety, with weights 0.4, 0.3 and 0.3.

| Proposal | Value score | Feasibility score | Safety score | Weighted score | Rank |
|---|---:|---:|---:|---:|---:|
| A | 0.45 | 0.60 | 0.80 | 0.60 | 2 |
| B as specified | 0.75 | 0.40 | 0.50 | 0.57 | 3 |
| C as specified | 0.90 | 0.70 | 0.20 | 0.63 | **1** |

The weighted score ranks the disqualified proposal first, and it does so while giving safety a 30% weight. The arithmetic is not a trick: C scores 0.63 = 0.4(0.90) + 0.3(0.70) + 0.3(0.20) = 0.36 + 0.21 + 0.06, and its high value and feasibility scores overwhelm its low safety score. Any weighting that keeps safety below 50% will produce this outcome whenever a proposal is strong on two dimensions and weak on the third, which is precisely the profile of a proposal that crosses a boundary.

Raising the safety weight to 0.6 fixes this instance and breaks another. With weights 0.2/0.2/0.6, A scores 0.69, B scores 0.50 and C scores 0.44, so A outranks B' — but B' is the highest-density option in the portfolio at $1.88 per reviewer minute against A's $0.72. A single weighting cannot serve both purposes because the two purposes are different operations: the gate is a *conjunction* and the ranking is a *comparison*. Flattening a conjunction into a weighted sum is a category error, and no choice of weights repairs it.

```text
FORMULA [SCORE-01]: S(p) = sum_k w_k * s_k(p),  sum_k w_k = 1, s_k in [0,1]
WHERE: S = composite score [-]; w_k = weight on criterion k [-];
       s_k = normalised score of proposal p on criterion k [-]
CONSTRAINT: compensability — a high score on one criterion must be able to
       offset a low score on another for S to be meaningful. Where any
       criterion encodes a non-compensable constraint, S is invalid.
NUMERICAL EXAMPLE: w = (0.4, 0.3, 0.3); C = 0.4(0.90)+0.3(0.70)+0.3(0.20) = 0.63;
       A = 0.4(0.45)+0.3(0.60)+0.3(0.80) = 0.18+0.18+0.24 = 0.60.
INTERPRETATION: C wins by 0.03 while failing the authority constraint outright.
       The 0.03 is not noise; it is the mechanism by which the score conceals
       the failure.
```

A second, subtler defect of the score is that its inputs are unnormalised ratings. The value score for C is 0.90 because C has the largest gross figure per day ($360 from 40 items) — but gross per day is not the same as gross per reviewer minute, and C's actual density is −$1.233 per reviewer minute because its harm term exceeds its gross. A score built on ratings rather than on ratios will keep producing this error, and the repair is not a better weighting but a different object: rank by a quantity that already has the constraint's units in the denominator.

### 6.7 Step 7 — Check waiting time, not just load

The budget provides six productive hours per reviewer per operating day: `8×0.75=6`. For this stationary sensitivity calculation, assume those hours overlap as one 360-minute staffed window. The real calendar has not been measured; staggered shifts or interruptions require a time-varying model. Dividing daily arrivals by 1,440 calendar minutes would incorrectly spread work across an unstaffed night.

```text
FORMULA [QUE-01]: C(c,a) = [a^c/(c!(1-rho))] /
                    [sum_{k=0}^{c-1} a^k/k! + a^c/(c!(1-rho))]
                  W_q = C(c,a) / (c*mu - lambda)
WHERE: c = concurrent reviewers [people]
       lambda = reviews per operating day / 360 [reviews/staffed-minute]
       mu = 1/E[t_rev] [reviews/staffed-minute per reviewer]
       a = lambda/mu [erlangs]; rho = a/c [-]
       C = probability an arrival waits [-]
       W_q = mean waiting time before service [staffed minutes]
CONSTRAINT: M/M/c reference model: stationary Poisson arrivals, exponential
       independent service, one pooled FCFS queue, no abandonment or priorities.
       rho must be strictly below 1. At rho>=1 there is no finite stationary
       mean wait. Finite-day backlog still needs a separate trace calculation.
NUMERICAL EXAMPLE: 200 B' reviews at 5 minutes, 71 A reviews at 4 minutes,
       and 30 audit reviews at 4 minutes give 301 jobs and 1,404 service minutes.
       E[t_rev] = 1404/301 = 4.66445 minutes.
       lambda = 301/360 = 0.836111/min; mu = 301/1404 = 0.214387/min;
       a = 3.9 erlangs; c = 6; rho = 3.9/6 = 0.65;
       C(6,3.9) = 0.260933;
       W_q = 0.260933/(6*0.214387-0.836111) = 0.579577 minutes,
       or about 34.8 seconds. Conversion to seconds multiplies by 60;
       it does not change a reported minute value into a larger minute value.
INTERPRETATION: the recommended portfolio has finite reference-model waiting,
       not zero waiting. Its utilization agrees with 1404/2160=65%.
```

| Configuration | Reviews/day | Mean service min | Servers | Utilisation ρ | C(c,a) | Mean wait, staffed min |
|---|---:|---:|---:|---:|---:|---:|
| Recommended, pooled | 301 | 4.66445 | 6 | 0.6500 | 0.26093 | 0.57958 |
| A alone, dedicated | 160 | 4.000 | 2 | 0.8889 | 0.83660 | 15.05882 |
| A capped at 266, dedicated | 71 | 4.000 | 2 | 0.3944 | 0.22315 | 0.73702 |
| B' alone, dedicated | 200 | 5.000 | 2 | 1.3889 | not stationary | no finite stationary mean |
| B as specified, 4 servers | 360 | 6.000 | 4 | 1.5000 | not stationary | no finite stationary mean |
| B as specified, 5 servers | 360 | 6.000 | 5 | 1.2000 | not stationary | no finite stationary mean |
| C' alone, 1 server | 40 | 6.000 | 1 | 0.6667 | 0.66667 | 12.00000 |

The dedicated rows describe each named stream, without the shared audit workload. They diagnose staffing fragmentation; they are not complete funded portfolios. B' cannot be assigned only two reviewers in the same six-hour window: its 1,000 service minutes exceed their 720-minute capacity before audits. A full-scope two-reviewer queue is stable in this reference model but has little slack and a much longer mean wait.

Pooling can help when the same qualified people can serve the combined jobs, but the comparison also changes the available staffing allocation. It does not establish a measured improvement or justify moving restricted work to unauthorized reviewers. A single-server C' queue is mathematically stable here; 12 minutes may or may not satisfy its separately declared response requirement.

Exponential service is an assumption, not a demonstrated upper bound for real review work. A mixture of A, B' and audit service distributions is not generally exponential merely because its mean is known. Arrivals, service variation, qualifications and breaks must be measured before these reference values become planning evidence.

### 6.8 Step 8 — Test the intraday shape

Every figure so far is a daily total, and a daily total hides the shape of the day. Work arrives in a morning peak, and a team that is exactly staffed on the daily total is under-staffed for the morning. The case therefore runs a four-hour burst trace with a 30/30/20/20 arrival profile.

| Hour | B as specified: arrival minutes | Capacity (6 × 60) | Backlog after hour | Recommended, paced: arrival minutes | Backlog after hour |
|---|---:|---:|---:|---:|---:|
| 1 | 648 | 360 | 288 | 351 | 0 |
| 2 | 648 | 360 | 576 | 421 | 61 |
| 3 | 432 | 360 | 648 | 351 | 52 |
| 4 | 432 | 360 | **720** | 281 | **0** |

B as specified leaves 720 reviewer-minutes of backlog at the end of the four-hour arrival burst. The declared six-hour productive window still has two hours left, supplying exactly `2×360=720` minutes; with no further arrivals or interruptions the backlog clears at hour six, not the next day. The problem is zero daily headroom and a large transient queue. The recommended paced profile peaks at 61 backlog minutes and clears by hour four. Neither trace establishes a waiting-time percentile.

```text
FORMULA [BURST-01]: B_t = max(0, B_{t-1} + A_t - C)
WHERE: B_t = backlog after hour t [reviewer-min]
       A_t = arrival minutes in hour t [reviewer-min]
       C = hourly capacity [reviewer-min]; B_0 = 0
CONSTRAINT: work is served first-in-first-out within capacity; no overtime,
       no abandonment, no cross-day capacity borrowing. Backlog is in minutes
       of work, not requests, so a request's wait depends on its position.
NUMERICAL EXAMPLE: C = 6*60 = 360; A = (648, 648, 432, 432);
       B_1 = max(0, 0+648-360) = 288; B_2 = max(0, 288+648-360) = 576;
       B_3 = max(0, 576+432-360) = 648; B_4 = max(0, 648+432-360) = 720.
INTERPRETATION: B_4=720 is burst-end backlog, not day-end backlog.
       With A_5=A_6=0, B_5=360 and B_6=0. Any lost capacity or additional
       arrivals can then create carryover because the daily mean uses all capacity.
```

The drain arithmetic shows the second consequence. If the backlog is carried into the next day and the next day's arrivals are unchanged, the backlog does not clear; it accumulates at 720 minutes per day until something breaks. Clearing 720 minutes requires spare capacity, and at 100% load there is none.

```text
FORMULA [DRAIN-01]: T_drain = B / (C - A_bar)
WHERE: T_drain = time to clear backlog [h]; B = backlog [reviewer-min]
       C = hourly capacity [reviewer-min/h]; A_bar = mean hourly arrivals [reviewer-min/h]
CONSTRAINT: C > A_bar; constant rates; no new bursts during the drain.
NUMERICAL EXAMPLE (two cases): B as specified has C = 6*60 = 360 and
       A_bar = 2160/4 = 540, so C - A_bar = -180 and the backlog grows rather
       than drains. The recommended portfolio paced to peak at 61 minutes has
       A_bar = 1404/4 = 351; adding one temporary reviewer gives C = 420 and
       T_drain = 61/(420-351) = 61/69 = 0.88 h.
INTERPRETATION: the sign of (C - A_bar) is the whole question. Where it is
       negative, no pacing fixes the backlog and capacity must change; where it
       is positive, pacing plus a small buffer clears the transient.
```

The honest reading of the B row is that B's problem is not its daily total but its shape and its lack of slack. Pacing B's release would reduce the burst but not the total, and its total is already the entire budget. The case therefore does not recommend pacing B; it recommends redesigning B's routing so that the total falls, and then pacing the smaller stream.

### 6.9 Step 9 — Apply the tail gate

The commitment gate tests a mean. The tail check estimates a variable day's load. With independent Poisson review counts and fixed per-stream service durations, mean load is `sum(n_j*t_j)` and variance is `sum(n_j*t_j²)`. Adding 1.645 standard deviations is a normal approximation to the 95th percentile, not an exact compound-Poisson quantile or a guaranteed coverage bound.

```text
FORMULA [TAIL-01]: L_95 = m + z_0.95 * sqrt(sum_j n_j * t_rev,j^2)
WHERE: L_95 = 95th-percentile daily load [reviewer-min]
       m = sum_j n_j * t_rev,j = mean daily load [reviewer-min]
       n_j = routed reviews/day for stream j [reviews/day]
       t_rev,j = service time for stream j [min/review]
       z_0.95 = 1.645 [-]
CONSTRAINT: independent Poisson review counts; fixed per-stream service times;
       sufficient counts for a normal approximation. Variable durations add
       variance, but the approximation does not establish a one-sided bound.
NUMERICAL EXAMPLE (B as specified): n=360, t_rev=6;
       m = 2160; sd = sqrt(360*36) = sqrt(12960) = 113.84;
       L_95 = 2160 + 1.645*113.84 = 2160 + 187.3 = 2347.3.
       Share of R = 2347.3/2160 = 108.7%.
INTERPRETATION: the normal approximation places B's 95th-percentile load
       above the available budget. No measured exceedance frequency follows
       from this constructed calculation; the mean gate already rejects B.
```

The tail gate does real work on the *portfolio* rather than on the individual streams. Every stream passes on its own; the combination does not.

| Portfolio | Mean load | Mean share | SD | 95th percentile | p95 share | G4 = 1,944? |
|---|---:|---:|---:|---:|---:|---|
| A at 600 | 640 | 29.6% | 50.6 | 723.2 | 33.5% | pass |
| A + audit | 760 | 35.2% | 55.1 | 850.7 | 39.4% | pass |
| B as specified | 2,160 | 100.0% | 113.8 | 2,347.3 | 108.7% | **fail** |
| B' | 1,000 | 46.3% | 70.7 | 1,116.3 | 51.7% | pass |
| C' | 240 | 11.1% | 37.9 | 302.4 | 14.0% | pass |
| B' + C' | 1,240 | 57.4% | 80.2 | 1,372.0 | 63.5% | pass |
| B' + C' + audit | 1,360 | 63.0% | 83.2 | 1,496.8 | 69.3% | pass |
| A + B' | 1,640 | 75.9% | 86.9 | 1,783.0 | 82.5% | pass |
| A + B' + audit | 1,760 | 81.5% | 89.7 | 1,907.5 | 88.3% | pass |
| A + B' + C' | 1,880 | 87.0% | 94.9 | 2,036.1 | 94.3% | **fail** |
| A + B' + C' + audit | 2,000 | 92.6% | 97.4 | 2,160.2 | 100.0% | **fail** |
| **A at 266 + B' + audit** | **1,404** | **65.0%** | **81.3** | **1,537.8** | **71.2%** | **pass** |
| A at 266 + B' + C' + audit | 1,644 | 76.1% | 89.8 | 1,791.6 | 82.9% | pass |

The A + B' + audit row is the interesting near-miss: it passes the tail gate at 88.3% while failing the commitment gate at 125.4% of G3. A portfolio can be tail-safe and mean-infeasible, and the two gates are not redundant. Conversely, the A + B' + C' row fails the commitment gate at 133.9% of G3 and fails the tail gate at 94.3% of R. Neither gate subsumes the other, which is why both are in the design.

The recommended row sits exactly at the commitment gate — 1,404 of 1,404 minutes — with a 95th percentile of 1,538 minutes, which is 71.2% of capacity and 79.1% of the tail gate. That is a deliberate trade: the portfolio uses all of the commitment budget and leaves a 406-minute gap between its 95th percentile and the tail gate, because the tail gate is measured against the full budget rather than against the committed portion.

#### 6.9a What the tail gate would say if it were the only gate

If the commitment gate were removed and only the tail gate applied, B as specified would still fail (108.7%), and the A + B' + C' + audit portfolio would fail at exactly 100.0%. The commitment gate is therefore not doing all the work; it is catching the portfolios that are mean-infeasible but tail-tolerable, such as A + B' at 88.3% of the tail gate and 125.4% of the commitment gate. The two gates catch disjoint failure sets, which is the argument for keeping both rather than picking the stricter one.

#### 6.9b What the tail gate misses: service-time variance

The compound-Poisson variance assumes each review takes exactly `t_rev` minutes. If the true service time has a standard deviation `s_rev`, the daily load variance becomes `n·(t_rev² + s_rev²)`, and the 95th percentile rises. For the recommended portfolio with `s_rev` = 2 minutes, so that variance per review doubles:

```text
sd = sqrt(200*(25+4) + 71*(16+4) + 30*(16+4))
   = sqrt(200*29 + 71*20 + 30*20)
   = sqrt(5800 + 1420 + 600) = sqrt(7820) = 88.4 min
L_95 = 1404 + 1.645*88.4 = 1404 + 145.4 = 1549.4 min  -> 71.7% of R
```

The tail share moves from 71.2% to 71.7%, which changes nothing. The same calculation on B as specified moves its tail share from 108.7% to `(2160 + 1.645*sqrt(360*(36+4)))/2160 = (2160+312.2)/2160 = 114.5%`, which also changes nothing because the proposal had already failed. The variance assumption is therefore not decision-reversing at any point in this case, which is worth stating explicitly because it would be decision-reversing in a portfolio packed at 88% of the tail gate.

### 6.10 Step 10 — Rank the feasible options by density

Only options that pass all three gates are ranked, and the ranking statistic is net value per committed reviewer minute rather than net value. The choice of denominator is the substantive decision in this step.

| Option | Net $/day | Committed min/day | **Density $/reviewer-min** | Rank | Why it ranks here |
|---|---:|---:|---:|---:|---|
| B' redesigned | 1,882.00 | 1,000 | **1.882** | 1 | best value per scarce minute |
| A at 600 items | 462.00 | 640 | **0.722** | 2 | good density, large fixed-cost base |
| B as specified | 1,496.00 | 2,160 | **0.693** | — | fails both capacity gates |
| C' proposal-only | 24.00 | 240 | **0.100** | 3 | positive but immaterial |
| A capped at 266 items | 65.48 | 284 | **0.231** | — | capacity-limited variant of A |
| C as specified | −500.00 | 0 | undefined | — | fails authority and value |

A's density at full scope is 0.722, which is less than half of B's 1.882. A's marginal density, however, is much higher than its average density, because A's fixed cost is already paid. The marginal value of the last A item is `4 × 0.50 − 0.2667 × 4 × 0.65 − 0.002 × 60 = $1.1867` against `0.2667 × 4 = 1.0667` committed minutes, which is **1.112 $/min**. That marginal density is still below B's 2.182 $/min, so the correct sequencing is B' first, then A with whatever remains; A's scope is then set by the residual capacity rather than by its own eligibility.

```text
FORMULA [DENS-01]: d(p) = NV(p) / m(p)
                   d_marg(p) = (dNV/dV) / (dm/dV)
WHERE: d = average density [$ per reviewer-minute]
       d_marg = marginal density at the current scope
       dNV/dV = per-item net margin [$ per item]
       dm/dV = per-item committed minutes [reviewer-min per item]
CONSTRAINT: d_marg is the correct ordering statistic for a capacity-packed
       portfolio; d is correct only when scope is fixed. Both are reported
       because d is what a reader can check from the total row.
NUMERICAL EXAMPLE (A): dNV/dV = 4*0.50 - 0.2667*4*0.65 - 0.002*60
       = 2 - 0.6933 - 0.12 = 1.1867 $/item;
       dm/dV = 0.2667*4 = 1.0667 reviewer-min/item;
       d_marg = 1.1867/1.0667 = 1.112 $/reviewer-min.
       (B'): dNV/dV = 7.8*0.50 - 0.25*5*0.65 - 0.06*6 = 3.9-0.8125-0.36
       = 2.7275 $/item; dm/dV = 1.25 min/item; d_marg = 2.182 $/min.
INTERPRETATION: B' outranks A on both average and marginal density, so B' is
       funded at full scope and A is funded at the residual.
```

After reserving 120 audit minutes, B' consumes 1,000 of the 1,284 routable minutes. The residual is 284 minutes. A's continuous allowance is `284/((4/15)×4)=266.25` items; whole-item and whole-review constraints must then be applied.

At 266 items, continuous A reviews would be `266×4/15=70.9333`, costing $184.4267; the continuous stream net is $65.6533. Requiring 71 whole reviews raises that cost to $184.60 and reduces net to $65.48. The fully relaxed 266.25-item solution has A net $65.95, so the relaxed-to-integer portfolio gap is $0.47/day. Rounding the review count down to 70 would not cover the declared routing requirement.

### 6.11 Step 11 — Solve the packing problem and bound the gap

The tables first enumerate stream-level subsets before the shared audit cost. Their “admissible” flag means passing the stated stream gates, not release approval. Every nonempty funded portfolio must also add the 120-minute audit reserve and subtract its $78/day cost; row 16 shows that complete calculation. Empty baseline retains no new pilot cost. Fixed-cost and whole-review effects mean density ordering alone is not a general knapsack solver.

| # | Subset | Committed min | Gate G3 = 1,404 | p95 min | Gate G4 = 1,944 | Net $/day | Admissible |
|---:|---|---:|---|---:|---|---:|---|
| 1 | none | 0 | pass | 0 | pass | 0 | **yes, by default** |
| 2 | A at 600 | 640 | pass | 723.2 | pass | 462.00 | yes until jointly packed |
| 3 | B as specified | 2,160 | **fail** | 2,347.3 | **fail** | 1,496.00 | no |
| 4 | C as specified | 0 | pass | 0 | pass | −500.00 | no (authority, value) |
| 5 | A + B as specified | 2,800 | **fail** | 3,005.3 | **fail** | 1,958.00 | no |
| 6 | A + C as specified | 640 | pass | 723.2 | pass | −38.00 | no (authority, value) |
| 7 | B as specified + C | 2,160 | **fail** | 2,347.3 | **fail** | 996.00 | no |
| 8 | A + B + C | 2,800 | **fail** | 3,005.3 | **fail** | 1,458.00 | no |

Read as specified, the only stream-level choices are none and A. A contributes $462/day before the mandatory audit; its complete funded portfolio consumes 760 minutes and returns `$462−$78=$384/day`. This is the defensible literal-proposal comparison, not a claim that an unaudited A release is acceptable. The redesign changes the feasible comparison set.

| # | Subset, redesigned | Committed min | Gate G3 | p95 min | Gate G4 | Net $/day | Admissible |
|---:|---|---:|---|---:|---|---:|---|
| 9 | B' | 1,000 | pass | 1,116.3 | pass | 1,882.00 | yes |
| 10 | C' | 240 | pass | 302.4 | pass | 24.00 | yes |
| 11 | B' + C' | 1,240 | pass | 1,372.0 | pass | 1,906.00 | yes |
| 12 | A + B' | 1,640 | **fail** | 1,783.0 | pass | 2,344.00 | no (mean) |
| 13 | A + B' + C' | 1,880 | **fail** | 2,036.1 | **fail** | 2,368.00 | no |
| 14 | A capped 266 + B' | 1,284 | pass | 1,412.9 | pass | 1,947.48 | yes |
| 15 | A capped 266 + B' + C' | 1,524 | **fail** | 1,667.2 | pass | 1,971.48 | no (mean) |
| 16 | A capped 266 + B' + audit | 1,404 | pass (exactly) | 1,537.8 | pass | 1,869.48 | **yes, recommended** |
| 17 | A capped 266 + B' + C' + audit | 1,644 | **fail** | 1,791.6 | pass | 1,893.48 | no (mean) |

Note on row 14: without the audit reserve the mean is 1,284 and the standard deviation is `sqrt(200×25 + 71×16) = 78.3`, so the 95th percentile is `1,284 + 1.645×78.3 = 1,412.9`. Row 16 adds 120 mean minutes and the 30 audit reviews' `30×16` variance, yielding 1,537.8. The version with the reserve is recommended because the quality signal is not optional.

Note on row 17: adding C' raises the approximate tail load to 1,791.6 minutes, 92.2% of G4, and adds $24/day of stream value, but exceeds G3 by 240 minutes. C' therefore cannot be added to the recommended scope. A separate measurement track needs its own explicitly approved staffing and audit budget; calling it a trial does not create free capacity.

**Continuous-scope comparison with fixed costs retained.** The relaxed A scope is exactly `284/((4/15)×4)=266.25`. Its variable margin is `2−(4/15)×4×0.65−0.002×60=1.1866667` dollars/item. The relaxed portfolio net is `1,882+266.25×(89/75)−250−78=$1,869.95/day`, versus $1,869.48 after integer review constraints. The $0.47 gap bounds only this fixed-B' rounding comparison; it is not a proof that greedy density solves every fixed-charge portfolio.

```text
FORMULA [PACK-01]: maximise  sum_p NV_p(x_p)
                   subject to sum_p m_p(x_p) <= G3 - A_res
                              x_p in {0} U [x_p^min, x_p^max]
WHERE: x_p = scope of proposal p [items/day]; NV_p = its net value [$/day]
       m_p = its committed minutes [reviewer-min/day]
       G3 - A_res = 1,284 routable committed minutes
CONSTRAINT: each NV_p is monotone increasing and linear in x_p above a
       minimum viable scope; the fixed cost makes NV_p negative below
       x_p^min = F_p / (per-item margin).
NUMERICAL EXAMPLE: A's minimum viable scope is 250/1.1867 = 210.7 items;
       B' earns 2.7275*800 - 300 = 1882 at its full scope and needs
       300/2.7275 = 110.0 items to break even.
INTERPRETATION: A is worth funding only if at least 211 of its 600 items
       fit in the residual, and the residual is 266, so it clears the
       minimum with 55 items of margin.
```

The minimum viable scope is the quantity that decides whether a small residual is worth using at all, and it is the reason the recommendation shrinks A rather than deleting it. A residual of 150 minutes would fund only 140 A items, whose net value would be 140 × 1.1867 − 250 = −$83.9. At that point the correct answer is to drop A and leave the capacity uncommitted, which is the non-obvious conclusion that a fixed cost forces.

### 6.12 Step 12 — Test the decision against changed assumptions

A decision that survives only at its own parameter values is not a decision. The case varies the five parameters that Section 4.2 identified as decision-reversing, one at a time.

| Stress | Original | Alternative | Recommended portfolio | Change in conclusion |
|---|---|---|---|---|
| Eligible volume `N` | 2,000/day | 1,600/day | B' requires 800 review-minutes; A residual after audit is 484 minutes | At most 121 whole A reviews support 453 items; audited portfolio net=$1,654.64/day |
| Eligible volume `N` | 2,000/day | 2,400/day | B' requires `960×0.25×5 = 1,200` routing minutes, leaving 84 routable minutes for A after the audit reserve | A is about 78 items and net-negative; **fund B' + audit only, $2,240.40/day** |
| Routing fraction `q` (B') | 0.25 | 0.45 with the redesigned 5-minute check | B' load 1,800 min; mean share 83.3%, before audit | **B' becomes infeasible** at that scope; A + audit returns $384/day |
| Routing fraction `q` (A) | 0.2667 | 0.563 | A's net value hits zero at 0.563 | A's contribution vanishes |
| Reviewer availability `alpha` | 0.75 | 0.60 | R = 1,728; G3 = 1,123 | **A does not fit**; B' + audit only, $1,804 |
| Reviewer availability `alpha` | 0.75 | 0.90 | R=2,592; G3=1,684.8; A residual=564.8 min | At most 141 whole A reviews support 528 items; audited portfolio net=$2,180.04/day |
| A service time `t_rev` | 4 min | 8.44 min | A's net value hits zero | A drops; B' + audit only |
| A defect rate `p_err` | 0.002 | 0.0061 | A's capped-scope net value hits zero | A drops |
| B' defect rate `p_err` | 0.06 | 0.452 | B' stream net approaches zero at fixed scope and other inputs | **Repack**; A + audit returns $384/day if B' is dropped |
| B' saved minutes | 7.8 min | 3.10 min | B' net value hits zero | **B' drops** |
| C' defect rate | 0.001 | 0.0013 | C' net value hits zero | C' is already a trial, not a stream |
| Reviewers | 6 | 8 | G3=1,872; A at 600 fits with B' and audit | Portfolio gain=$396.52/day, below the illustrative $624/day added staffing cost |

Three stresses reverse the recommendation outright: volume at 2,400/day, availability at 0.60, and B' routing at 0.45. At 2,400 daily requests, B's 40% eligible set is 960 requests, so B' requires 1,200 routing minutes and leaves only 84 routable minutes for A after the audit reserve. That supports about 78 A items, below A's fixed-cost break-even; the correct portfolio is B' plus audit, not an A scope that destroys value. The response is not to find a more robust recommendation but to attach the three numbers to it as monitoring triggers. If observed volume exceeds 2,300/day, or availability falls below 0.68, or B's routed set exceeds 30%, the decision is reopened rather than adjusted quietly.

The reopening thresholds are management escalation triggers, not permission to exceed a hard gate. Every intake or scope revision must still satisfy the current authority, mean-capacity and tail checks. If those checks fail before a listed trigger is reached, stop additional admission and repack immediately; a previously approved daily scope is not a guarantee that changed routing or staffing still fits.

## 7. The Finding

The steering owner should fund **two of the three proposals, neither at its submitted specification, and sequence them in one wave with a scope cap on the first.**

Specifically: fund **B only after redesigning its routing policy** from a 45% routed fraction to 25% with a shorter per-item check, which moves it from 2,160 committed minutes per day (100% of available capacity, 108.7% at the 95th percentile) to 1,000 minutes (46.3%, 51.7%), at a cost of $754 less review spend and $288 more expected harm, for a net of **$1,882 per day** at $1.882 per reviewer-minute. Fund **A at a capped scope of 266 items per day**, not its 600-item eligibility, consuming the residual 284 minutes for **$65.48 per day**; at full scope A is jointly infeasible with B', and its value density of $0.722 makes it second in the queue. Do **not** fund **C as specified** — it fails the authority gate because its output is an executed account state change, and it fails the value test independently at −$500 per day on a 1% error rate against $2,000 redress cost. Convert C to a **proposal-only trial governed outside the pilot portfolio**, with 40 reviews per day at a value of $24 per day, because that value is too small and too fragile to justify capacity and its sign depends on a defect rate no one has measured.

The recommendation commits **1,404 minutes/day**, exactly G3. Its approximate p95 is **1,537.8 minutes**, or **71.2% of R and 79.1% of G4**. Net value is **$1,869.48/day**, or **$56,084.40 over 30 operating days**, including the shared audit cost. If B' remains at 45% routing, that scope is infeasible and A plus audit returns $384/day. At availability 0.60, B' plus audit still fits and returns $1,804/day, but A cannot be funded profitably in the remaining scope. These are illustrative conditional calculations, not measured savings or guaranteed percentiles.

## 8. System/Solution Design

### 8.1 The architecture that the recommendation implies

The portfolio decision only means something if the pilot can be operated and measured. The design below is the minimum operating structure that makes the three gates observable. None of it exists; it is a proposed design.

```text
                     +-------------------------------+
  requests --------->| intake and eligibility policy |--- excluded (logged)
                     +---------------+---------------+
                                     | eligible request + stable request_id
                                     v
                     +-------------------------------+
                     | stream policy (versioned)      |
                     |  A: draftable set, q_A         |
                     |  B': route if confidence < tau |
                     +---------------+---------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
   +-------------------+                          +----------------------+
   | review queue      |                          | auto-path (no review)|
   | priority by age   |                          | logged, audited      |
   +---------+---------+                          +----------+-----------+
             |                                               |
             v                                               |
   +-------------------+                                     |
   | reviewer console  |                                     |
   | approve/edit/     |                                     |
   | reject/escalate   |                                     |
   +---------+---------+                                     |
             |                                               |
             v                                               v
   +---------------------------------------------------------------+
   | immutable decision ledger                                     |
   | request_id, policy_version, output_version, reviewer_id,      |
   | decision, approval_id, timestamps, side_effect_id             |
   +------------------+----------------------------+---------------+
                      |                            |
                      v                            v
             +------------------+         +-------------------+
             | audit sampler    |         | load monitor      |
             | stratified       |         | mean + p95 share  |
             +--------+---------+         +---------+---------+
                      |                            |
                      v                            v
             +--------------------------------------------------+
             | gate controller: pacing, cap, kill switch        |
             +--------------------------------------------------+
```

### 8.2 The four state stores

| Store | Contents | Why it must exist | Failure if absent |
|---|---|---|---|
| Eligibility log | request identifier, eligibility decision, policy version, timestamp | freezes the eligible set before outcomes are seen | eligibility can be redefined after the fact, biasing the defect rate |
| Routing ledger | request identifier, routing decision, confidence, policy version | makes the routed fraction and its drift measurable | `q` is only known from the specification, never from behaviour |
| Decision ledger | request identifier, output version, reviewer identity, decision, approval identifier, timestamps | the only record that a review happened and who authorised what | no audit trail; C-type effects become unattributable |
| Side-effect ledger | side-effect identifier, target system, reversal status | bounds the damage from any unintended external effect | a state change cannot be found or reversed |

The four stores are separate because they have different retention, access and correction rules. Merging them into one "event log" is the common simplification that destroys the audit trail: a mutable log cannot prove an approval, and an approval record embedded in a mutable log is not a record.

### 8.3 Request lifecycle, including the failure path

```text
1. request arrives            -> stable request_id assigned; duplicate check
2. eligibility evaluated      -> eligible | excluded(reason)   [logged]
3. stream policy applied      -> draft | route+review | auto-route | propose-change
4. if review: enqueue         -> queue position, age visible
5. reviewer acts              -> approve | edit | reject | escalate
6. if approve: approval_id    -> bound to output_version and reviewer_id
7. release                    -> artefact sent | queue assignment | proposed change
8. audit sample drawn         -> independent adjudication, reason code
9. ledger reconciled          -> counts match intake minus exclusions
FAILURE PATHS
  F1 queue cap exceeded       -> stop new intake for the stream; hold baseline path
  F2 reviewer unavailable     -> degrade to baseline for that stream; alert owner
  F3 ledger write fails       -> block release (no release without a record)
  F4 audit finds defect class -> pause stream; revise rubric; re-audit before resume
  F5 unauthorised effect      -> revoke path; reverse via side-effect ledger; notify
  F6 policy version mismatch  -> reject the request to the baseline path
```

Two properties of this lifecycle are load-bearing and easy to get wrong.

**No release without a record.** If the ledger write can fail while the release succeeds, the audit trail has a hole exactly where the interesting cases are. The design makes release conditional on the ledger write, which means a ledger outage becomes a throughput outage. That is the correct trade: an unrecorded release is worse than a delayed one, because it cannot be reversed, disputed or counted.

**Idempotency by request identifier.** A retried review that creates a second queue entry inflates measured load, deflates measured service time per request, and makes the capacity arithmetic wrong in a direction nobody can see. The request identifier is the idempotency key, and the queue enforces uniqueness on it.

### 8.4 Offline and online paths

The offline path is the decision arithmetic and the evaluation plan. It runs monthly, on a frozen snapshot of the four ledgers, and produces the gate report in Section 9. The online path is the request lifecycle above. The two paths must not share a mutable code path, because an evaluation that imports the production policy object will silently adopt whatever policy is deployed at evaluation time, and the comparison then measures nothing.

| Path | Input | Output | Cadence | Owner |
|---|---|---|---|---|
| Offline: gate report | ledger snapshot, parameter estimates | load share, tail share, net value, gate verdicts | monthly, and at any trigger | portfolio analyst |
| Offline: audit adjudication | stratified sample of releases | defect rate with interval, reason codes | weekly | review lead |
| Offline: re-scoping | gate report + observed `q`, `t_rev` | revised scope for each stream | at trigger | steering owner |
| Online: routing and review | request + policy version | decision + ledger record | continuous | review lead |
| Online: pacing and cap | live arrival rate, queue depth | release rate adjustment | continuous | on-call operator |
| Online: kill switch | breach signal | stream reverted to baseline | on breach | on-call operator |

### 8.5 Key design choices and their rejected alternatives

| Choice | Alternative rejected | Why rejected |
|---|---|---|
| One pooled review queue with stream tags | fragmented dedicated queues | two dedicated reviewers cannot serve B' in the declared six-hour window; pooling still requires qualified reviewers (6.7) |
| Pacing at 25/30/25/20 | concentrate the day's work in four hours | the unpaced B trace has 720 burst-end backlog minutes and uses the remaining two staffed hours merely to clear (6.8) |
| Approval identifier on every release | reviewer identity only | identity says who acted; the approval identifier says which version they approved |
| Scope cap as a first-class field | scope implied by intake | an implied scope cannot be enforced or reported |
| Release blocked on ledger write | best-effort logging | an unrecorded release cannot be reversed or counted |
| Audit reserve funded inside the gate | audit funded from slack | slack is the first thing consumed under load, so the audit disappears when it matters |

### 8.6 Operational readiness

The design is not ready to operate, and the gaps are specific.

There is no measured `t_rev`. Until a sample exists, the load figures are the sponsor's estimates multiplied by an assumed service time, and the entire recommendation inherits that error. The first operational task is a stopwatch sample of 35 reviews for A and 200 for B', which Section 9 sizes.

There is no measured `q` for B's redesigned policy. The 25% figure is a design target; the routed fraction that a confidence threshold actually produces is an empirical question, and if the underlying classifier's confidence distribution is concentrated near the threshold, the routed fraction will be far from the target.

There is no harm ledger. Section 6.4 shows that C's net value is decided by a defect rate of 0.375% against an assumed 1.0%, and A's by 1.48% at full scope or 0.61% at the recommended capped scope, against an assumed 0.2%. Neither rate can be estimated from the current record because corrections are not coded by proposal.

The kill switch is untested. A switch that has never been thrown is a switch that may not work, and the failure mode is discovered at the moment of maximum stress.

## 9. Evaluation & Validation

### 9.1 What has to be measured, and why the list is short

The evaluation has one job: determine whether the three parameters that decide the recommendation are anywhere near their assumed values, before the portfolio's value is claimed. Those parameters are the routed fraction `q`, the review service time `t_rev`, and the defect rate `p_err`. Everything else — net value, density, capacity share — is arithmetic on top of them.

The evaluation therefore has a deliberately narrow primary question and a set of secondary questions that matter for operations but not for the decision.

| Metric | Numerator | Denominator | Units | Population | Window | Comparator | Status |
|---|---|---|---|---|---|---|---|
| Committed review load | sum of reviewer minutes on pilot reviews | operating days | reviewer-min/day | all routed reviews in the stream | rolling 1 day, reported as 10-day block mean | G3 = 1,404 | not measured |
| Tail load | same, 95th percentile of daily totals | — | reviewer-min/day | same | 30 days | G4 = 1,944 | not measured |
| Routed fraction `q` | routed requests | eligible requests in the stream | dimensionless | stream's eligible set | 10-day blocks | 0.2667 (A), 0.25 (B') | not measured |
| Review service time `t_rev` | sum of review durations | reviews completed | reviewer-min | completed reviews | 10-day blocks | 4 min (A), 5 min (B') | not measured |
| Defect rate `p_err` | defects adjudicated in audit sample | releases sampled | dimensionless | released outputs, stratified by stream and reviewer | weekly | 0.002 (A), 0.06 (B'), 0.001 (C') | not measured |
| Specialist minutes per request | sum of handling minutes | requests completed | specialist-min/request | stream's eligible set | 10-day blocks | 6→2 (A), 9→1.2 (B') | not measured |
| Correction cost | total correction and redress cost | defects found | $/defect | audit-detected and complaint-driven defects | 30 days | $60 (A), $6 (B'), $2,000 (C') | not measured |

Every row is `not measured`. The case has no execution. The table exists so that a reader can see what a real pilot would have to produce, and so that the arithmetic in Section 6 can be labelled honestly as conditional on values nobody has observed.

### 9.2 Guardrails and stop conditions

A guardrail is not a metric with a target; it is a condition that ends the pilot. The distinction matters because a guardrail that is reported alongside value metrics competes with them, and loses.

| Guardrail | Trigger | Response | Owner |
|---|---|---|---|
| Authority breach | any release with an external effect and no approval identifier | revoke the path; reverse via side-effect ledger; notify risk partner | risk partner |
| Tail-spillover | 95th-percentile daily load > 1,944 min for two consecutive 10-day blocks | cut intake for the largest stream by 25%; re-run the gates | review lead |
| Persistent backlog | end-of-day backlog > 200 min on 3 days in a block | reduce release rate; add a reviewer or narrow scope | on-call operator |
| Defect rate drift | observed rate above the break-even in `HARM-01` for the stream | pause the stream; revise the rubric; re-audit | review lead |
| Review-time drift | observed `t_rev` > 1.25 × assumed for two blocks | re-run the capacity arithmetic; re-scope | steering owner |
| Data-boundary breach | any input without an authorisation record | stop the stream immediately; escalate | risk partner |
| Ledger gap | any release without a ledger record | block release until reconciled | engineering owner |

Two of these are worth explaining because they are frequently mis-specified.

The **tail-spillover** trigger requires two consecutive blocks rather than one day, because a single busy day is exactly the event the tail gate is calibrated to tolerate. Triggering on one day produces an oscillation: the pilot is cut after every burst, never accumulates enough volume to measure anything, and the organisation concludes the intervention does not scale.

The **defect-rate drift** trigger is stated against the stream's own break-even rather than against a fixed percentage. A 2% defect rate is catastrophic for A, whose break-even is 1.48% at full scope and 0.61% at the recommended cap, and tolerable for B', whose break-even is 45.2%. A single global quality threshold would either block B' unnecessarily or fail to protect A.

### 9.3 How many observations are needed

These parameters have different evidence costs. Thirty operating days is the proposed management window, not a promise that the funded audit sample can estimate every rare-event rate precisely.

**Routed fraction.** A proportion with a 95% confidence half-width of `h` needs `n = 1.96² p(1−p)/h²`. For `p` = 0.2667 and `h` = 0.035:

```text
FORMULA [SAMP-01]: n = z² p(1-p) / h²
WHERE: n = observations [requests]; z = 1.96 for 95% [-]
       p = proportion being estimated [-]; h = desired half-width [-]
CONSTRAINT: simple random sample; p not near 0 or 1; observations independent.
NUMERICAL EXAMPLE: p=0.2667, h=0.035;
       n = 3.8416 * 0.2667 * 0.7333 / 0.001225 = 3.8416*0.19556/0.001225
         = 0.75125/0.001225 = 613.3 -> 614 requests.
       At the recommended A scope, 266 arrivals/day give
       h = 1.96*sqrt(0.2667*0.7333/266) = 0.05315.
       Over 30 such days, n=7,980 gives h=0.00970.
INTERPRETATION: about 614 independent scoped requests are needed for the
       stated half-width, spanning roughly 2.3 operating days at 266/day.
```

**Review service time.** A mean with a 95% half-width `h` needs `n = (1.96 s/h)²`, where `s` is the standard deviation. With `s` = 1.5 minutes and `h` = 0.5 minutes, `n = (1.96 × 1.5/0.5)² = 34.6`, so **35 reviews**. With `s` = 2.0 the requirement rises to 61; with `h` = 0.25 it rises to 138. The case's Section 4.1 verification plan for `A-05` therefore specifies a 35-review sample, and the plan acknowledges that a 35-review sample cannot detect a 0.25-minute bias.

**Defect rate.** This is the expensive one, and the arithmetic is the reason C' is a trial rather than a stream.

```text
FORMULA [SAMP-02]: n = z² p(1-p) / h²   (proportion, small p)
WHERE: as SAMP-01; for small p, p(1-p) ≈ p.
CONSTRAINT: normal approximation degrades when np < 5; use an exact method then.
NUMERICAL EXAMPLE (A): p=0.002, h=0.001 (i.e. ±50% relative);
       n = 3.8416*0.002*0.998/0.000001 = 7667.8336 -> 7,668 audited releases.
       A's p_err covers all released A outputs, reviewed and unreviewed.
       At 266 outputs/day, 7,668 releases exist after about 28.8 days,
       but existence is not an audit. At an illustrative 4 minutes per audit,
       120 audit minutes/day fund at most 30 audits/day across A and B',
       or 900 in 30 days even if all were allocated to A.
INTERPRETATION: the funded window cannot deliver the stated +/-0.001 precision.
       Predeclare allocation and use an exact/binomial interval for sparse
       defects; otherwise fund more evidence or retain the baseline.
```

```text
FORMULA [RULE3-01]: p_upper(95%) ≈ 3/n   when zero defects are observed
WHERE: n = observations with no defect [-]; 3 = -ln(0.05) rounded
CONSTRAINT: independence; every observation actually audited; no defects found.
NUMERICAL EXAMPLE (C'): 40 releases/day * 30 days = 1,200 audited releases,
       zero defects -> p_upper = 3/1200 = 0.0025.
       C's break-even is 0.0013, so zero defects in 1,200 observations does not
       establish that C' is safe: the upper bound is 1.9 times the break-even.
       To reach p_upper = 0.0013 requires n = 3/0.0013 = 2,308 releases,
       which is 2,308/40 = 57.7 operating days.
INTERPRETATION: C' cannot be cleared inside a 30-day window. Its disposition
       as a separately governed trial follows from the arithmetic, not from
       caution.
```

The rule of three approximates a one-sided 95% zero-event upper bound: the exact binomial upper limit is `1−0.05^(1/n)` under independent identical trials and complete ascertainment. Valid two-sided exact or Wilson intervals at zero events are not degenerate; the naive Wald interval is. Zero observed defects alone establishes neither zero risk nor clinical, legal or operational safety.

### 9.4 The evaluation protocol

The protocol fixes four things before the window opens, and each one closes a specific way of fooling oneself.

**Freeze the eligibility rule.** The rule that decides which requests are in each stream is versioned and hashed at the start. Requests are tagged with the rule version that admitted them. A request that later looks miscategorised is not removed from the denominator; it is reported as a miscategorisation. Removing it would bias the defect rate downward, because the requests most likely to look miscategorised after the fact are the ones that went wrong.

**Freeze the audit frame.** The audit samples released outputs, not reviewed outputs, and it samples within stream and within reviewer. Sampling reviewed outputs measures the reviewer; sampling only within a stream hides a reviewer-specific problem; sampling only recent outputs hides drift.

**Pre-register the analysis.** The three primary quantities, their windows, their comparators and the stop rules are written down before the data exists. The case's Section 6 arithmetic is the pre-registration: the decision rule is "compare observed load share to 65%, observed tail share to 90%, observed net value to zero", and it does not change after the window.

**Report by 10-day block.** Thirty days in one block hides a trend. Three blocks of ten days make a drifting routed fraction visible in time to act, which is the operational value of the block structure. The cost is wider intervals per block; the case reports the block means and the full-window interval rather than pretending the block intervals are independently decisive.

### 9.5 What would falsify the recommendation

The recommendation is falsifiable in four specific ways, and the case states them so that a later reader can check whether it was wrong rather than arguing about interpretation.

| Falsifier | Observation that would establish it | How it would be observed |
|---|---|---|
| B' fails its current capacity plan | observed q at least 0.45 or mean review time at least 8 minutes at 800/day | routing and duration ledger; recompute all funded load including audit |
| A cannot retain its recommended scope | observed availability no greater than 0.68, or A mean review time at least 5.3 minutes | staffed-time study and duration sample; repack rather than merely retaining 266 items |
| C's stated authority classification is wrong | reviewed design records show the action cannot change external state, or a valid recorded approval already binds that exact effect | inspect actual effect semantics, permission and approval records; evaluate the corrected variant |
| The portfolio destroys value | observed scoped handling savings, review/audit cost and harm produce nonpositive complete portfolio net | reconcile actual decision cohorts and costs against the frozen baseline, then compare fund-nothing |

Zero external effects in proposal-only C' cannot falsify the authority requirement for autonomous C. The two designs deliberately have different effect boundaries. Absence of recorded incidents also cannot justify removing an authorization gate; missing instrumentation is not evidence of safety.

### 9.6 Uncertainty and what the numbers do not mean

Three uncertainties are large enough to state beside the conclusion.

The point estimates of net value are means of a distribution whose variance is unknown. B' at $1,882/day has a harm term of $288/day built from an assumed 6% defect rate and an assumed $6 correction cost. If the correction cost is $12, the harm term doubles to $576 and B' net value falls to $1,594 — still the best option, but a 15% reduction for one parameter.

The daily tail approximation and the M/M/c waiting calculation use different conditional models. Neither establishes an upper bound for real review work. The service-time variance sensitivity in Section 6.9b explores one assumption; it does not validate arrival correlations, interruptions, qualification constraints or the shape of waiting-time tails. Measure these before making an operational latency claim.

The specialist-minute valuation of $0.50 is the parameter least connected to anything observable. It is a management accounting rate, and the case uses it because a portfolio comparison needs a common unit, not because it is right. B's ranking is insensitive to it: at $0.30 per specialist minute, B' still earns `800×(7.8×0.30 − 0.8125 − 0.36) − 300 = 800×(2.34−1.1725) − 300 = 800×1.1675 − 300 = $634`, which is still positive and still the largest of the feasible options. A is not insensitive: its per-item margin falls to `4×0.30 − 0.6933 − 0.12 = 0.3867`, so its break-even volume rises to `250/0.3867 = 646` items, which exceeds its 600-item eligibility. At a $0.30 specialist minute, A cannot break even at any scope it can reach, and the recommendation becomes B' alone.

## 10. Action Taken

### 10.1 What the steering owner does with the finding

The action is a decision record plus a set of pre-conditions, not a deployment. Four items are decided, and each has a named owner and a date.

| # | Action | Owner | Precondition | Consequence if not done |
|---:|---|---|---|---|
| 1 | Approve B for redesign, not for pilot, with target q=0.25 and a 5-minute check | steering owner with B's sponsor | none; design work can start immediately | if redesign fails, repack to A + audit at $384/day, not an unaudited $462 portfolio |
| 2 | Approve A for a capped pilot of 266 items/day with a hard intake cap | steering owner with operations lead | 35-review time sample for `t_rev` within 5 days | A's load is a guess; the cap may be wrong in either direction |
| 3 | Reject C as specified; authorise C' as a separately governed measurement track with no value claim | steering owner with risk partner | side-effect ledger and approval identifier requirement | C would either be funded on a false premise or dropped without learning |
| 4 | Fund the 120-minute audit reserve inside the capacity gate, not from slack | review lead | audit frame and sampling rule pre-registered | the quality signal disappears under load, exactly when needed |
| 5 | Instrument the four ledgers and the review-time sample before any stream goes live | engineering owner | ledger write must block release | the pilot produces no usable evidence |
| 6 | Set the reopening triggers: volume > 2,300/day, availability < 0.68, B' routed fraction > 0.30, A `t_rev` > 5.3 min | steering owner | none | the decision silently drifts out of validity |

The sequencing of these items is not arbitrary. Item 5 must precede items 2 and 3, because a stream that runs without a ledger cannot be audited, and a stream that cannot be audited cannot be evaluated or reversed. Item 1 does not depend on 5, because redesign is design work and does not consume review capacity.

### 10.2 The decision record, in the form an auditor would want

```text
DECISION  fund A (capped 266 items/day) + B' (redesigned, 800 items/day) one wave
          reject C as specified; authorise C' as measurement track
DATE      2026-11-09 (illustrative)
OWNER     portfolio steering owner
BASIS     Section 6 arithmetic on illustrative parameters; no execution
GATES     authority: A pass, B' pass, C fail, C' pass (with approval id)
          commitment: portfolio 1404/1404 min = 100.0% of G3
          tail: p95 1537.8/1944 = 79.1% of G4
VALUE     $1,869.48/day net of fixed costs; $56,084 over 30 operating days
EVIDENCE  ILLUSTRATIVE parameters, DERIVED arithmetic
NOT CLAIMED  any measured outcome, any vendor capability, any production result
REOPEN IF  volume > 2300/day; alpha < 0.68; B' q > 0.30; A t_rev > 5.3 min;
           any unauthorised external effect; any ledger gap
FALSIFIER  observed B' q >= 0.45 or t_rev >= 8.0 min (6.12, 9.5)
```

### 10.3 What was explicitly not done

No vendor was contacted. No contract was signed. No engineering work was commissioned beyond the design change in item 1. No reviewer was hired. No data was accessed. No production system was touched. The case is an analytical exercise, and its action is a decision record that a real organisation would then have to execute.

This matters for how the file should be read. A reader who takes the recommendation as evidence that B' works has misread it. The recommendation is that B' *would* be the best option *if* its redesigned parameters hold, and the pilot exists to find out whether they do.

### 10.4 The alternative action that was seriously considered

The most serious competitor to the recommendation is **fund nothing and instrument for a quarter**. It costs nothing, carries no new harm, and produces the three missing parameters — `q`, `t_rev`, `p_err` — from the manual workflow rather than from an intervention.

The case rejects it for one reason and accepts it as the fallback for another. It is rejected because the manual workflow does not produce `t_rev` in the relevant sense: there is no comparable review step for drafted text, so measuring the manual path does not tell you how long an edit of generated text takes. It is retained as the fallback because if the time sample for `t_rev` comes back above 5.3 minutes for A, or above 8.0 minutes for B', the correct response is not to re-scope but to stop, and the organisation then needs the measurement quarter it should have run earlier.

## 11. Outcome & Results

### 11.1 The only result that exists

**Nothing has been measured.** The only outputs of this case are arithmetic consequences of constructed inputs. Stated precisely:

| Result | Value | Evidence label | What it establishes | What it does not |
|---|---|---|---|---|
| Capacity budget | 2,160 reviewer-min/day | `DERIVED` from `ILLUSTRATIVE` inputs | the denominator of every gate | the team's real availability |
| Commitment gate | 1,404 min/day | `DERIVED` | the packing constraint | that 65% is the right policy |
| B as specified load share | 100.0% mean, 108.7% p95 | `DERIVED` | B is infeasible as specified | B's real routed fraction |
| B' load share | 46.3% mean, 51.7% p95 | `DERIVED` | B' is feasible if its parameters hold | that a redesign achieves 0.25 |
| Recommended net value | $1,869.48/day | `DERIVED` | the ranking under these parameters | any realised saving |
| 30-day portfolio value | $56,084 | `DERIVED` | arithmetic over the window | an earned amount |
| A-only funded portfolio | $384.00/day | `DERIVED` | $462 stream value less $78 mandatory audit | no measured saving |
| C as specified net | −$500.00/day | `DERIVED` | C destroys value before the gate is applied | C's real defect rate |
| C' clearance horizon | 57.7 days | `DERIVED` from `RULE3-01` | C' cannot be cleared in 30 days | anything about C' in production |

### 11.2 The 30-day arithmetic, shown once

```text
wave 1 (days 1-30)   A at 266 items + B' at 800 items + audit
  A:  266 * (4*0.50)          =  532.00 gross
      71  * (4*0.65)          =  184.60 review
      266 * 0.002 * 60        =   31.92 harm
      fixed                   =  250.00
      net                     =   65.48 /day        [A]
  B': 800 * (7.8*0.50)        = 3120.00 gross
      200 * (5*0.65)          =  650.00 review
      800 * 0.06 * 6          =  288.00 harm
      fixed                   =  300.00
      net                     = 1882.00 /day        [B']
  audit: 120 min * 0.65       =   78.00 /day
  TOTAL net                   = 65.48 + 1882.00 - 78.00 = 1869.48 /day
  30 days                     = 1869.48 * 30 = 56,084.40
  committed minutes           = 284 + 1000 + 120 = 1404 of 1404 (100.0% of G3)
                                and 1404 of 2160 = 65.0% of R
  p95 load                    = 1404 + 1.645*81.3 = 1537.8 = 71.2% of R
```

The audit cost is subtracted once at the portfolio level rather than allocated to each stream, because the audit sample is drawn across the portfolio. Allocating it proportionally would change each stream's density slightly and the ranking not at all: A's density would fall from 0.722 to 0.667, and B''s from 1.882 to 1.827.

### 11.3 What a real outcome record would have to contain

The case's own finding cannot be promoted to a result without five things, and stating them is part of the deliverable.

A **numerator and denominator** for every rate, with the denominator's population and window. "B' reduced handling time" is not a result; "the mean specialist handling time for the 24,000 eligible documents in the 30-day window fell from a sampled baseline of 9.0 to 1.2 minutes, n=800 sampled" is.

A **comparator** with the same measurement procedure. The baseline cannot be a different quarter's management report, because the reporting procedure differs.

An **interval**, not a point. Section 9.3 sizes the samples; a result reported without its interval cannot be compared to a break-even.

A **harm record** including defects that were found by complaint rather than by audit, because complaint-driven defects are a different and usually more expensive population.

A **rollback record** for anything that was reverted, because a reverted stream's partial results are not the stream's results.

### 11.4 The result that would change the recommendation

If the time sample returns `t_rev` = 6.0 minutes for A rather than 4.0, A's committed minutes at 266 items become `71 × 6 = 426`, which no longer fits the 284-minute residual. The portfolio would then be B' + audit at 1,120 minutes and $1,804/day, with 284 minutes uncommitted. That is a $65.48/day reduction and a materially different recommendation: A would be dropped rather than capped, and the residual capacity would be held back rather than spent.

If B's redesign achieves `q` = 0.35 rather than 0.25, B' load becomes `800 × 0.35 × 5 = 1,400` minutes, and B' plus the audit reserve becomes 1,520 minutes, which exceeds the 1,404 gate. B' would then need either a shorter check or a smaller scope, and the arithmetic in 6.12 is the template for the re-run.

Neither of these is a hypothetical failure mode; both are the expected outcome of a pilot whose job is to replace assumed parameters with measured ones. The case's recommendation is therefore best read as a **prior plus a measurement plan**, not as a conclusion that survives contact with data unchanged.

## 12. Failure Modes & What Went Wrong

This section distinguishes failures that were demonstrated by the constructed arithmetic from operational failures that remain untested. No entry is evidence that a real organization experienced the event.

### 12.1 Failure register and discriminating checks

| ID | Failure mechanism | Status | Misleading symptom | Discriminating check | Bounded response | Recovery evidence |
|---|---|---|---|---|---|---|
| `FM-01` | B is funded at its submitted 45% routing fraction | demonstrated analytically | daily total appears equal to capacity | recompute `800×0.45×6=2,160` against G3=1,404 and p95=2,347.3 | reject the submitted design; reduce routing and review duration together | B' at 1,000 mean minutes and 1,116.3 p95 only if its inputs are observed |
| `FM-02` | A and B' are approved separately but not repacked together | demonstrated analytically | each stream individually passes | sum all stream minutes and the 120-minute reserve before approval | cap A at 266 items or reject the combination | decision ledger records 1,404, not two independent pass labels |
| `FM-03` | An authority gate is scored rather than enforced | demonstrated analytically | C obtains the highest weighted score | inspect whether a false authority predicate can be offset by a score | make authority an AND gate; convert C to proposal-only C' | every release has an approval identifier before an external effect |
| `FM-04` | Routing fraction drifts above the design target | untested operational risk | benefit remains high while backlog begins late in the day | routing ledger numerator/denominator by 10-day block; reopen above 0.30 | lower intake, narrow eligibility, or stop B' | observed q, scope version, and revised gate calculation agree |
| `FM-05` | Review service time is longer or more variable than assumed | untested operational risk | mean looks acceptable but age and p95 rise | duration sample by stream; compare mean to 4/5 minutes and recompute variance | reduce scope; pace releases; pause if A exceeds 5.3 minutes | new service-time estimate, p95 calculation, and no carried backlog |
| `FM-06` | The audit reserve is treated as optional slack | demonstrated design error | value improves briefly because 120 minutes disappear from the ledger | compare committed load including and excluding audit; inspect audit completion | fund the reserve inside G3 and block expansion when it is consumed | audit frame completes and total remains no greater than 1,404 |
| `FM-07` | A retry creates a second review | untested operational risk | review minutes rise while eligible volume is stable | join intake and routing records on stable `request_id`; count duplicate keys | idempotent enqueue; route duplicates to existing record | one request identifier maps to at most one active review |
| `FM-08` | A ledger write fails after an output is released | untested operational risk | throughput looks normal but approval coverage falls | reconcile releases to immutable ledger rows and approval IDs | block release on failed ledger write; send request to baseline | zero unmatched releases at end-of-day reconciliation |
| `FM-09` | Eligibility or audit sampling is changed after defects are visible | untested evaluation risk | defect rate improves while exclusions rise | compare frozen rule and sampling-frame hashes with current versions | retain original denominator; report miscategorisation separately | versioned eligibility log and pre-registered audit frame |
| `FM-10` | C' is treated as cleared after zero defects in 30 days | demonstrated statistical error | point estimate is zero | apply `3/n`: `3/1,200=0.25%`, above C' break-even 0.13% | retain measurement-track status through at least 2,308 audited releases | documented count, zero-event bound, and approval-only operation |

### 12.2 Failed and successful recovery paths

The failed recovery is “add reviewers after the queue is full.” It does not repair the authority failure in C, does not restore omitted ledger records, and cannot be executed inside the stated 30-day staffing window. It also mistakes a policy gate for a staffing estimate: B as submitted consumes 2,160 minutes before audit, while the committed budget is 1,404. Adding capacity after backlog appears changes the experiment and invalidates the pre-registered comparison.

The successful recovery is narrower. When B' routing rises above target, stop accepting new B' work, preserve the request and policy version in the ledger, and return new requests to the manual baseline. Finish or explicitly cancel already-queued reviews; do not silently discard them. Then recompute load from observed `q` and `t_rev`, choose a new scope only if both gates pass, and record the change as a new policy version. This recovery preserves authority, denominators, and reversibility; it does not claim that the original design succeeded.

An authority breach has a stricter path: revoke the external-effect path, identify affected requests through the side-effect ledger, reverse each reversible state transition, notify the named risk owner, and preserve the records for adjudication. “Correct the model and continue” is not recovery because it leaves the unknown blast radius unresolved. A missing side-effect identifier makes that recovery incomplete, which is why C cannot graduate from proposal-only status without that ledger.

### 12.3 Failure boundaries

The arithmetic does not model correlated errors, adversarial traffic, reviewer fatigue, specialist queueing, or the build cost of the proposed console and ledgers. Each can make the illustrative value estimate too high. Conversely, it does not credit learning that might reduce review time. Neither omission authorizes an adjustment to the decision record without a measured replacement parameter.

The key boundary is semantic rather than statistical: a reversible internal routing error and an unapproved external account change are not two points on one risk scale. The former can be sampled, corrected, and costed; the latter must first satisfy the authority condition. Treating them as comparable expected-loss rows is the failure this case is designed to prevent.

## 13. Transferable Lesson

### 13.1 The reusable rule

For a portfolio with a non-compensable boundary and a scarce operating resource, use **gates before ranking**:

```text
1. Reject any option that lacks authority, authorised inputs, or a recoverable path.
2. Convert each surviving option to the scarce resource's units.
3. Reserve mandatory control work inside, not beside, the usable budget.
4. Test mean and tail load for the complete portfolio, not its members.
5. Rank only survivors by incremental value per scarce unit.
6. Re-run the decision when an observed parameter crosses its trigger.
```

This rule applies to review minutes, safety-engineer hours, on-call error budgets, warehouse dock slots, clinician sign-off capacity, or any resource that cannot be purchased or deferred within the decision window. It does **not** imply that every decision needs a queueing model. If capacity is elastic, effects are reversible, and every criterion is genuinely compensable, a simpler cost-benefit comparison may be sufficient.

### 13.2 What transfers and what does not

| Element | Transfers | Does not transfer automatically |
|---|---|---|
| Authority gate | distinguish a proposal from an effect; require a refusal/reversal path | the specific approval form or legal requirement |
| Capacity calculation | volume × routing fraction × service time, then portfolio sum | the 65% and 90% thresholds |
| Tail check | mean capacity is insufficient when daily variation matters | a normal/compound-Poisson approximation without checking arrivals and service times |
| Density ranking | compare incremental value in the scarce unit | ranking before disqualifying gates or treating fixed costs as marginal |
| Measurement track | learn under constrained authority when clearance needs more evidence | claiming an unmeasured trial delivers production value |

The counterexample matters. Suppose every proposal only drafts internal notes, all drafts are reviewable, reviewer capacity can be contracted weekly, and no hard authority boundary exists. A sequential gate may add ceremony without changing the answer; a budgeted value model with sensitivity analysis may be clearer. The method is justified here by fixed reviewer capacity and the account-state boundary, not by the presence of AI.

### 13.3 Compact derivation of the decision logic

The recommended portfolio is not “B' because it has the highest value.” It is:

```text
R       = 6 × 8 × 60 × 0.75 = 2,160 reviewer-min/day
G3      = 0.65R = 1,404 reviewer-min/day
route budget = G3 − 120 audit minutes = 1,284 reviewer-min/day
B' load = 800 × 0.25 × 5 = 1,000 reviewer-min/day
A residual = 1,284 − 1,000 = 284 reviewer-min/day
A scope = floor(284 / (0.2667 × 4)) = 266 items/day with 71 whole reviews
portfolio net = 65.48 + 1,882.00 − 78.00 = $1,869.48/day
30-day arithmetic = 30 × $1,869.48 = $56,084.40
```

Those results are `DERIVED` from `ILLUSTRATIVE` inputs. The transfer is the chain of units and gates, not the dollar value, thresholds, or an asserted deployment outcome.

## 14. Connections & Cross-References

### Related material and source boundaries

| Connection | Why it is relevant | Status and boundary |
|---|---|---|
| [AI Use-Case Charter](../Cheat_Sheets/01_AI_Use_Case_Charter.md) | frames owner, decision, baseline, and authority before solution selection | local method reference; availability/link target must be checked by the integration owner |
| [AI Market Map](../Cheat_Sheets/02_AI_Market_Map.md) | supports feasibility and cost-per-output comparisons | local method reference; does not supply this case's quantities |
| [Technology Due Diligence](../Cheat_Sheets/03_Technology_Due_Diligence.md) | motivates validation, rule-of-three reasoning, and gates | local method reference; does not validate C' |
| Section 6.7 | distinguishes daily capacity from queue waiting | illustrative Erlang C calculation only |
| Section 6.9 | distinguishes mean commitment from p95 spillover | constructed compound-Poisson approximation only |
| Section 9.3 | relates clearance evidence to zero-event sample size | `RULE3-01` is conditional on independent complete audits |

`SRC-05` names standard Erlang C and zero-event results, but the source body was not reviewed as part of this illustrative case. Its precise bibliographic version, publication date, and access date remain `OPEN`. The formulas are used here as explicitly conditional teaching mechanisms, not as evidence of operating performance.

### Interview pack

#### Q1. Why is B rejected before B' is ranked?

**Answer.** B as specified requires `800×0.45×6=2,160` review-minutes/day, exceeding the 1,404-minute commitment gate and reaching 108.7% of total capacity at p95. It is therefore not in the feasible comparison set.

**Reasoning.** Ranking an infeasible option invites a false trade-off between value and a hard operating constraint.

**Follow-up.** What if B has the largest gross benefit? Gross benefit changes neither the 2,160-minute load nor the gate result.

**Acceptable alternatives.** State that B may be redesigned or deferred for measurement, provided the new parameters are separately gated.

**Red flags.** “Fund it and hire later”; comparing B's $1,496 net directly to B' without capacity.

**Scoring.** 0 = ranks first; 1 = notes high load; 2 = computes a gate failure; 3 = explains why gate precedes ranking.

**Answer support.** Sections 5c, 6.2, 6.5, and 6.9.

#### Q2. Recompute the capacity budget and explain why 2,160 is not the packing limit.

**Answer.** `6×8×60×0.75=2,160` reviewer-minutes/day, but G3 is `0.65×2,160=1,404`; after the 120-minute audit reserve, only 1,284 minutes are routable.

**Reasoning.** Existing obligations and control work must be funded before discretionary routing.

**Follow-up.** What fails if audit is outside G3? The quality signal is dropped first under load.

**Acceptable alternatives.** A different documented reserve or gate is acceptable; packing against gross paid hours is not.

**Red flags.** Calling 2,160 the usable pilot budget or subtracting audit twice.

**Scoring.** 0 = no units; 1 = 2,160 only; 2 = gives 1,404; 3 = also derives 1,284 and explains reserve placement.

**Answer support.** Sections 2.1 and 6.1.

#### Q3. Why is C disqualified even if a revised cost estimate makes its expected value positive?

**Answer.** C executes an external account-state change without a recorded approval. It fails the authority predicate before value is considered.

**Reasoning.** An executed effect is not a draft a person can refuse; its reversibility and search cost are separate obligations.

**Follow-up.** What redesign makes C discussable? C' proposes the change, requires approval and a side-effect ledger, then remains a measurement track.

**Acceptable alternatives.** A documented pre-authorised, reversible state transition could be evaluated only if it meets the stated authority policy.

**Red flags.** Raising C's safety-score weight; treating $2,000 loss as a complete authority control.

**Scoring.** 0 = uses value only; 1 = says “risky”; 2 = identifies missing approval; 3 = distinguishes proposal from effect and names recovery.

**Answer support.** Sections 6.4, 8.3, and 12.2.

#### Q4. Derive the recommended 266-item cap for A.

**Answer.** B' uses 1,000 routable minutes and audit uses 120, leaving `1,404−1,000−120=284` minutes. A consumes about `0.2667×4=1.0667` minutes/item, so `284/1.0667=266.2`; round down to 266 items and 71 whole reviews.

**Reasoning.** Rounding review count down would under-cover routed items; the cap is a portfolio residual, not A's standalone optimum.

**Follow-up.** Why not use all 600 A items? A+B'+audit would consume 1,760 minutes and fail G3.

**Acceptable alternatives.** A conservative cap below 266 that preserves whole-review coverage.

**Red flags.** Dividing 284 by four; ignoring q; using 2,160 as residual.

**Scoring.** 0 = unsupported cap; 1 = identifies residual; 2 = includes q and service time; 3 = addresses rounding and joint gate.

**Answer support.** Sections 6.5, 6.10, and 6.11.

#### Q5. Identify the numerical error in a claim that the 2,400/day stress merely shrinks A.

**Answer.** At 2,400 eligible requests/day, B's 40% scope is 960 requests/day and B' requires `960×0.25×5=1,200` routing minutes, not 1,500. This is still below the 1,284 routable-minute budget but leaves only 84 minutes for A; with audit the commitment is exactly 1,404. The earlier 1,500 figure incorrectly treated all 2,400 requests as B-eligible.

**Reasoning.** Inputs use B volume `V=0.40N`, not total N. The residual supports about 78 A items and A's fixed cost makes that scope net-negative.

**Follow-up.** What is the action? Fund B'+audit only at its scaled scope; do not fund A at negative incremental portfolio value.

**Acceptable alternatives.** Recompute with an explicitly changed B eligibility share, but state that changed assumption.

**Red flags.** Scaling B from 800 to 1,200 without preserving its 40% eligibility share; claiming no re-run is needed.

**Scoring.** 0 = accepts the error; 1 = spots volume scaling; 2 = computes 1,200; 3 = also applies A's fixed-cost boundary.

**Answer support.** Sections 5a, 6.12, and 13.3.

#### Q6. What does the tail gate add after the commitment gate?

**Answer.** The commitment gate limits mean load; the tail gate asks whether a variable day spills toward total capacity. A+B'+audit fails mean at 1,760 minutes but passes p95 at 1,907.5, while A+B'+C' fails p95 at 2,036.1.

**Reasoning.** A mean-safe portfolio can have unsafe variation, and a tail-safe portfolio can still reserve too little time for existing obligations.

**Follow-up.** What assumption is weak? Fixed service time and independent Poisson review counts; measure service-time variance.

**Acceptable alternatives.** A simulation using observed arrivals and durations once data exists.

**Red flags.** Saying one gate makes the other redundant.

**Scoring.** 0 = repeats threshold; 1 = distinguishes mean/p95; 2 = gives one counterexample; 3 = names model limitation and measurement.

**Answer support.** Sections 6.9 and 9.1.

#### Q7. A reviewer says “add two people and fund A at full scope.” Is that automatically correct?

**Answer.** No. Eight reviewers yield G3=`0.65×8×480×0.75=1,872` minutes, so A+B'+audit at 1,760 fits. Its net is `$462+$1,882−$78=$2,266`, only `$2,266−$1,869.48=$396.52/day` more than the capped portfolio, below the illustrative $624/day cost of two added reviewers.

**Reasoning.** Feasibility and economic desirability are different tests.

**Follow-up.** What evidence could change the answer? A different staffing cost, a longer window, or A's measured per-item margin.

**Acceptable alternatives.** Hire if the staffing decision has value outside this portfolio and that value is explicitly modelled.

**Red flags.** Treating capacity addition as free or using gross A benefit.

**Scoring.** 0 = accepts; 1 = recomputes capacity; 2 = compares costs; 3 = separates portfolio and enterprise value.

**Answer support.** Section 6.12.

#### Q8. Why is zero C' defects in 30 days insufficient for clearance?

**Answer.** At 40 releases/day, 30 days gives 1,200 audits. With zero defects, the one-sided 95% upper bound is about `3/1,200=0.25%`, above C' break-even 0.13%.

**Reasoning.** Zero observed events does not mean zero true rate; clearance asks whether the rate could still destroy value.

**Follow-up.** How long for the stated bound? `3/0.0013=2,308` releases, or about 57.7 operating days.

**Acceptable alternatives.** An exact binomial bound that is no less conservative under its stated assumptions.

**Red flags.** Reporting the point estimate zero as proof of safety.

**Scoring.** 0 = clears C'; 1 = mentions sample size; 2 = computes bound; 3 = relates it to break-even and authority.

**Answer support.** Sections 6.4 and 9.3.

#### Q9. Diagnose rising backlog when mean daily review minutes remain below 1,404.

**Answer.** Inspect intraday arrivals, pacing, dedicated-versus-pooled assignment, and duplicate queue entries before raising the gate. A morning burst can create a transient backlog despite a safe daily mean.

**Reasoning.** Daily capacity is an aggregate; backlog evolves hourly as prior backlog plus arrivals minus service.

**Follow-up.** What evidence distinguishes a burst from slow review? Hourly arrival-minute trace versus duration samples and reviewer availability.

**Acceptable alternatives.** Use observed queue-age percentiles alongside the trace.

**Red flags.** Declaring the portfolio infeasible from one backlog observation or treating the tail model as an hourly queue model.

**Scoring.** 0 = adds headcount; 1 = names pacing; 2 = identifies competing causes; 3 = specifies the discriminating data and bounded action.

**Answer support.** Sections 6.7, 6.8, and 12.1.

#### Q10. Why must the eligibility rule be frozen before outcomes are reviewed?

**Answer.** Removing requests that later look hard or misrouted changes the defect-rate denominator after seeing the outcome, biasing the measured rate downward.

**Reasoning.** The decision needs a population defined at request time, not a successful subset defined retrospectively.

**Follow-up.** What should happen to a miscategorised request? Retain it in the original denominator and report a reason-coded miscategorisation.

**Acceptable alternatives.** A prospective rule version for later requests, with results reported by version.

**Red flags.** “Clean the data” by removing bad cases; sampling only reviewed outputs.

**Scoring.** 0 = accepts removal; 1 = names bias; 2 = states denominator; 3 = supplies versioned remedy.

**Answer support.** Sections 5d and 9.4.

#### Q11. When is a weighted score acceptable?

**Answer.** When all criteria are compensable and no criterion encodes a hard authority, safety, or data boundary. It is not acceptable when a high value score can purchase an unapproved external effect.

**Reasoning.** A weighted sum assumes substitution between criteria; a conjunction does not.

**Follow-up.** How would you use a score here? Only to rank proposals after authority, data, mean, and tail gates pass.

**Acceptable alternatives.** A constrained optimization with gates encoded as feasibility constraints.

**Red flags.** Setting the safety weight above 50% and calling the category error solved.

**Scoring.** 0 = uses score globally; 1 = says weights matter; 2 = explains compensability; 3 = provides gate-then-rank design.

**Answer support.** Sections 5c and 6.6.

#### Q12. What would make “fund nothing and instrument” the correct decision?

**Answer.** It becomes correct if observed A review time is above 5.3 minutes, B' review time is at least 8.0 minutes, routing remains near the submitted level, or the ledger/authority prerequisites cannot be built.

**Reasoning.** In those conditions the calculation cannot support a safe feasible pilot, and manual measurement becomes the least-irreversible way to replace the decisive assumptions.

**Follow-up.** Why is it not the initial recommendation? Manual work does not measure generated-draft editing time, so it cannot resolve every relevant parameter.

**Acceptable alternatives.** A non-operating shadow mode that records routing and review estimates without release, if it respects the data boundary.

**Red flags.** Treating no action as proof of no value; launching a pilot without instrumentation.

**Scoring.** 0 = “always pilot”; 1 = names uncertainty; 2 = names a stop trigger; 3 = connects triggers, measurement limits, and reversibility.

**Answer support.** Sections 10.4 and 11.4.

## 15. Quick Reference Card

| Field | Case reference |
|---|---|
| **Decision** | Fund B' only after redesign at 800 B-eligible items/day; fund A at 266 items/day; reject C as specified; keep C' as approval-only measurement. |
| **Evidence and stage** | `ILLUSTRATIVE` inputs; `DERIVED` arithmetic; proposed analytical case; no deployment or measured outcome. |
| **Binding gates** | Authority and data path first; G3=`1,404` reviewer-min/day including 120 audit minutes; G4=`1,944` p95 reviewer-min/day. |
| **Core capacity check** | `R=6×8×60×0.75=2,160`; recommended commitment=`284+1,000+120=1,404`; p95=`1,537.8`, or 71.2% of R. |
| **Core value check** | `$65.48 + $1,882.00 − $78.00 = $1,869.48/day`; `×30=$56,084.40`. These are not realised savings. |
| **Disqualifying fact** | C executes external account changes without a recorded approval; value cannot offset the authority failure. |
| **Reopen / stop** | Reopen above 2,300 daily requests, below 0.68 availability, B' q above 0.30, or A review time above 5.3 min; stop for an authority, data, or ledger breach. |
| **Measurement minimum** | Capture eligibility, routing, decision, and side-effect ledgers; audit released outputs; retain C' as a trial until its evidence boundary is met. |
| **Failure check** | Never rank before gates; never pack against 2,160 gross minutes; never omit the audit reserve or change denominators after outcomes. |
| **Review record** | Content completed 2026-11-10; external source bodies and operational measurements remain `OPEN` / not-measured. |
