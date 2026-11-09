---
collection: "program-orientation-ai-landscape"
file_id: "01"
topic: "AI use-case charter"
level: 1
status: "draft; unreviewed"
version: "2.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Decision owner, eligible population, data route, baseline, consequence, or workload assumption changes"
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["problem-framing", "baseline", "human-oversight", "economics"]
source_status: "OPEN — directory inventory consulted; GLCC, learn, raw, and speechtotxt bodies unavailable"
owner_role: "business decision owner with technical and risk counterparts"
source_index: "INDEX.md"
---

# AI Use-Case Charter

## 1. File Header

> **Purpose:** Convert a proposed AI idea into a falsifiable decision contract before selecting a model, vendor, or automation pattern.
> **Prerequisites:** Basic percentage arithmetic; ability to identify a workflow owner and distinguish an illustrative estimate from a measured result.
> **Key outcome:** Given a workflow proposal, write the eligible unit, decision, baseline, intervention boundary, economic hypothesis, risk gates, and promotion or rollback rule.
> **This file does NOT cover:** model-training recipes, legal certification, vendor selection, or a claim that a source container establishes a deployment outcome.
> **Level:** 1 Foundations.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED decision logic; ILLUSTRATIVE examples and arithmetic; OPEN source-body boundary.
> **Retrieval cue:** If a proposal begins with a product name or says “we need AI,” start at Section 2.

**Source boundary.** [directory.txt](../../../../Templates/directory.txt) inventories visible source-tree names.
The entries `GLCC`, `learn`, `raw`, and `speechtotxt` remain unavailable bodies.
Their names do not establish their contents, methods, results, or authorship.
This sheet constructs a teaching method rather than reconstructing those materials.
No example below is a measured deployment, procurement result, or claim about an organization.

## 2. Concept Map / Mental Model

```text
[Workflow trigger]
        |
        v
[Eligible unit and decision owner]
        |
        v
[Current baseline and consequence]
        |
        +-- unknown --> [instrument baseline before choosing technology]
        |
        v
[Candidate intervention and human boundary]
        |
        v
[Evidence, economics, risks, and capacity]
        |
        +-- hard gate fails --> [narrow scope / retain baseline / collect evidence]
        |
        v
[Versioned evaluation and operating contract]
        |
        +-- pass --> [limited rollout]
        +-- breach --> [rollback or safe fallback]
```

A charter is a compact agreement about a decision, not a description of a technology.
It starts with an observed workflow trigger rather than a capability label.
It names the smallest unit on which a decision may be made.
It gives one accountable owner authority to accept, reject, or pause the intervention.
It records the current path so a later comparison has a credible counterfactual.
It separates assistance from automation because side effects change the proof burden.
It makes uncertainty visible instead of hiding it inside a single ROI number.
It turns risks into observable conditions, owners, and responses.
It ends with a reversible gate rather than a promise to “pilot and learn.”

### 2.1 The charter as a chain of claims

A workflow claim states what happens now and to whom.
A decision claim states what could change after a system output.
A baseline claim states how the current path performs on the same eligible units.
An intervention claim states precisely what the proposed system may and may not do.
An evidence claim states what data, labels, tests, and uncertainty support the proposal.
An economics claim states how value, cost, harm, and capacity are counted.
A governance claim states who owns a gate, an exception, and an incident.
An operating claim states how the service degrades, stops, and recovers.
A charter fails if any claim is silently replaced by an adjective such as “intelligent,” “secure,” or “transformative.”

### 2.2 Charter lifecycle

1. Observe a repeated workflow decision.
2. Freeze a first eligibility rule.
3. Name the current decision-maker and fallback.
4. Measure the baseline on eligible units.
5. Specify the intervention’s input, output, and authority.
6. State the smallest decision that could create useful evidence.
7. Identify non-negotiable harm, privacy, safety, and authority constraints.
8. Estimate economics with uncertainty ranges rather than a single promise.
9. Define an evaluation population unavailable to tuning.
10. Define promotion, pause, rollback, and re-review conditions.

### 2.3 Questions that expose solution-first framing

What action changes if the output is correct?
What action changes if the output is absent?
Who bears the consequence of a wrong output?
Which requests are eligible, and which are deliberately excluded?
What current process is the intervention replacing or assisting?
What evidence would show that the current process is already sufficient?
Which constraint cannot be traded for faster handling or lower cost?
What event makes the system stop rather than continue automatically?
Who has authority to accept residual risk?
When will the charter be reconsidered after inputs, policy, or volume change?

**Takeaway:** A viable AI proposal is a chain from an observed decision to a controlled consequence, not a request for a fashionable capability.

## 3. Foundational Definitions

| Term | Meaning | Units/type | Valid use | Nonexample |
|---|---|---|---|---|
| Decision | A choice that changes a workflow, priority, communication, or state | named action | define value and accountability | “improve productivity” |
| Decision owner | Person or role authorized to accept a recommendation or retain baseline | role | owns gates and exceptions | a distribution list |
| Eligible unit | One item meeting a frozen inclusion rule | request, document, claim, order | denominator for metrics and costs | all activity vaguely described |
| Trigger | Observable event that starts the workflow | event | define timing and inputs | a quarterly ambition |
| Baseline | Current credible process on the same eligible units | process plus measured outputs | comparator for improvement | weakest alternative imagined later |
| Intervention | Bounded change to the workflow | input-output-action contract | state what technology does | “deploy AI” |
| Assistance | Output informs a human who owns the final action | advisory contract | draft, classify, retrieve, summarize | automatic record change |
| Automation | System executes a state-changing step | side effect | only with authority and recovery | a suggestion awaiting approval |
| Abstention | Deliberate refusal to provide an actionable output | policy outcome | route uncertainty safely | silent failure |
| Fallback | Safe behavior when the intervention cannot proceed | process | retain service and authority | “try again later” without owner |
| Eligible population | All units satisfying inclusion rules in a stated window | set | define metric denominator | convenience sample |
| Material slice | Subgroup with different risk, workflow, or likely performance | partition | inspect uneven effects | arbitrary demographic grouping |
| Ground truth | Defined reference used for evaluation | label, record, adjudication | score a declared claim | model output grading itself |
| Leakage | Use of information unavailable at the decision time | validity failure | invalidate evaluation | a valid post-action audit field |
| Coverage | Fraction of eligible units receiving a usable intervention response | fraction | measure reach | quality on answered units only |
| Quality | Agreement with a declared rubric or target on a declared population | metric | compare to a baseline | an unscoped “accuracy” number |
| Harm | Adverse consequence caused or amplified by the intervention | currency, count, severity | set gates and escalation | any disliked output |
| Guardrail | Measurable limit that constrains deployment | threshold or invariant | stop or narrow operation | aspirational principle |
| Hard gate | Condition that cannot be offset by a favorable score elsewhere | pass/fail | prohibit unacceptable behavior | a lightly weighted preference |
| Review capacity | Sustainable human handling capability in a window | units/window or hours/window | size routing policy | nominal headcount only |
| Fixed cost | Cost not varying with eligible-unit count over the declared window | currency/window | economics estimate | per-request API charge |
| Variable cost | Cost that changes with processed unit count | currency/unit | volume sensitivity | annual integration work |
| Expected value | Probability-weighted consequence under stated premises | currency/window | prioritize uncertain options | observed profit without comparator |
| Counterfactual | Credible outcome had the intervention not been used | reference outcome | assess incremental value | another team’s unrelated result |
| Promotion | Decision to expand authorized use after declared evidence passes | governance event | move from trial to broader use | continuing because a demo impressed viewers |
| Rollback | Controlled return to a known-safe path | recovery action | contain harm or invalid evidence | deleting logs after failure |
| Evidence ledger | Record linking claim, artifact, version, limitation, owner, and expiry | auditable record | prevent evidence drift | a slide with citations |
| OPEN evidence | Missing or unavailable support with a stated resolution need | status | preserve uncertainty | assumed positive evidence |

### 3.1 Charter fields and their minimum content

| Field | Minimum statement | Question it answers | Common omission |
|---|---|---|---|
| Workflow | trigger, user, current sequence, fallback | where does work begin and end? | only naming a department |
| Decision | action, timing, owner, consequence | what changes? | describing an output without an action |
| Eligible unit | inclusion, exclusion, time window | what is counted? | denominator drift |
| Baseline | current method, metrics, cost, quality limits | compared with what? | comparing to no process |
| Intervention | inputs, outputs, authority, abstention | what may it do? | hidden automation |
| Data | provenance, availability time, rights, retention | what supports it? | assuming access equals permission |
| Evaluation | split, rubric, comparator, slices | how will it be judged? | tuning on final test cases |
| Economics | benefit, cost, harm, capacity, uncertainty | why prioritize it? | gross benefit only |
| Risk | failure mode, severity, owner, control | what must not happen? | generic risk list |
| Operations | version, monitor, alert, fallback, rollback | who runs it? | treating launch as completion |
| Gate | pass, fail, pause, review date | what follows evidence? | “iterate” without authority |

### 3.2 Required distinctions

A unit count is not a rate.
A rate is not a probability estimate unless its population and window are stated.
A probability estimate is not calibration without comparison to observed frequencies.
A model score is not a business outcome.
A business outcome is not incremental value without a counterfactual.
A capability is not a permission.
A permission is not an implemented control.
A control is not proof that it operated under load.
A pilot is not a production deployment.
A deployment is not evidence of benefit.
A measured output is not automatically a causal effect.
An unavailable source is not weak evidence; it is OPEN evidence.

### 3.3 Evidence labels used in this sheet

**DERIVED** marks a logical consequence of stated definitions and assumptions.
**ILLUSTRATIVE** marks constructed teaching inputs and outcomes.
**VERIFIED** may mark arithmetic checked from stated illustrative inputs.
**SOURCED** would require a nearby primary reference and scoped claim.
**OPEN** marks a missing artifact, unavailable source body, or unresolved assumption.
A label on an example never transfers to a real organization or vendor.

**Takeaway:** Precise terms prevent a charter from substituting a technology label, a score, or an aspiration for an accountable decision.

## 4. Core Formulas with Full Derivation

### 4.1 Base: eligible coverage

Validity before use: count each eligible unit once in a frozen population and window.
Use this formula to measure reach, not correctness or realized value.
Do not use it when eligibility changes after reviewing intervention outcomes.

Derivation: partition the eligible set into units that receive a usable response and units that do not.
The fraction served is the served count divided by the eligible count.

```text
FORMULA [CHART-01]: c = n_served / n_eligible
WHERE: c is coverage [fraction]; n_served [units/window] is the number of
       eligible units receiving a usable response; n_eligible [units/window]
       is the number meeting the frozen eligibility rule.
CONSTRAINT: 0 <= n_served <= n_eligible; count retries once; use one window.
NUMERICAL EXAMPLE: ILLUSTRATIVE n_served=680 requests/week and
                   n_eligible=800 requests/week; c=680/800=0.85=85%.
INTERPRETATION: 120 eligible requests were not served; quality among 680
                responses cannot describe their experience.
```

Dimension check: units/week divided by units/week is dimensionless.
Arithmetic check: `0.85 * 800 = 680` served units.
Boundary: a stricter filter can raise quality on responses while lowering coverage.
Decision consequence: report coverage beside every quality claim.

### 4.2 General: expected incremental net value

Validity before use: baseline and intervention refer to comparable eligible units.
Benefits, harms, and costs must use the same currency and time window.
Probabilities are assumptions or estimates, not causal proof by themselves.
Do not double-count a savings already achieved by the baseline.

Derivation begins with expected net contribution for one eligible unit.
A beneficial outcome occurs with probability `p_g` and consequence `b`.
A harmful outcome occurs with probability `p_h` and consequence `h`.
Variable processing cost is `k` per eligible unit.
Expected contribution per unit is `p_g*b - p_h*h - k`.
Multiplying by `N` eligible units and subtracting fixed cost `F` gives the result.

```text
FORMULA [CHART-02]: V = N*(p_g*b - p_h*h - k) - F
WHERE: V [currency/window] is estimated incremental net value;
       N [eligible units/window]; p_g and p_h [fractions] are probabilities
       of incremental beneficial and harmful outcomes; b and h
       [currency/outcome] are their consequences; k [currency/unit] is
       variable intervention cost; F [currency/window] is fixed cost.
CONSTRAINT: 0 <= p_g,p_h <= 1; define outcome attribution and the same
            decision window; benefit and harm can co-occur only if modeled
            explicitly rather than treated as mutually exclusive by accident.
NUMERICAL EXAMPLE: ILLUSTRATIVE N=4,000 cases/month; p_g=0.18;
                   b=$24/case; p_h=0.012; h=$180/case; k=$0.65/case;
                   F=$4,000/month.
                   V=4,000*(0.18*24 - 0.012*180 - 0.65)-4,000
                    =4,000*(4.32-2.16-0.65)-4,000
                    =4,000*1.51-4,000=$2,040/month.
INTERPRETATION: the proposal has a positive illustrative estimate only if
                the stated incremental rates and consequences are credible.
```

Independent arithmetic: benefit is `$17,280/month`.
Expected harm is `$8,640/month`.
Variable cost is `$2,600/month`.
Subtracting those and fixed cost gives `$2,040/month`.
Units reduce as `(cases/month)*(dollars/case)=dollars/month`.

Sensitivity one: if `p_h` rises from `0.012` to `0.020`, expected harm becomes `$14,400/month`.
The resulting value is `17,280 - 14,400 - 2,600 - 4,000 = -$3,720/month`.
Sensitivity two: if `b` falls from `$24` to `$18`, benefit is `$12,960/month`.
The resulting value is `12,960 - 8,640 - 2,600 - 4,000 = -$2,280/month`.
Sensitivity three: if volume doubles while fixed cost stays fixed, the model gives `$8,080/month`.
That scale result is invalid if review capacity, quality, or harm probabilities change with volume.

Counterexample: a proposal may have positive expected value but be prohibited because a single harm crosses a non-negotiable safety, rights, or privacy boundary.
Decision consequence: use `V` for prioritization after hard gates, never to purchase permission to violate one.

### 4.3 Variant: review-capacity feasibility

Validity before use: each routed unit consumes one known review slot in the stated window.
The basic inequality assumes stable arrival and completion rates.
It excludes service-time variation, rework, vacations, escalation, and queueing delay.

Derivation: routed workload equals eligible volume times review fraction.
A workflow remains within simple capacity when routed workload does not exceed sustainable review capacity.

```text
FORMULA [CHART-03]: q*N <= r
WHERE: q [fraction] is the fraction routed for human review;
       N [eligible units/window]; r [review units/window] is sustainable
       review capacity under the declared staffing and service standard.
CONSTRAINT: 0 <= q <= 1; use the same window for N and r; do not use
            scheduled capacity if interruptions and rework are material.
NUMERICAL EXAMPLE: ILLUSTRATIVE N=1,200 requests/day, q=0.16,
                   and r=180 requests/day. Routed load=0.16*1,200=192,
                   which exceeds capacity by 12 requests/day.
INTERPRETATION: the policy creates a growing queue in this simplified model;
                it is not feasible at the declared throughput.
```

Independent arithmetic: `192 - 180 = 12` uncompleted arrivals per day.
Five similar workdays add at least `60` requests before considering variance.
The maximum feasible routed fraction is `180/1,200=0.15=15%`.
A threshold adjustment that lowers `q` must be re-evaluated for missed harmful cases and coverage.

Worked service-time conversion: if a reviewer completes 15 requests/hour for six productive hours/day, one reviewer supplies 90 requests/day.
Two reviewers supply 180 requests/day under that idealized assumption.
If average handling rises to 7.5 minutes, throughput is 8 requests/hour, or 96 requests/reviewer/day at twelve productive hours across two people.
The capacity estimate must use measured or conservatively planned productive time, not calendar hours.

### 4.4 Variant: break-even beneficial-outcome rate

Validity before use: use only when `b` is positive and estimated harm, costs, and volume are stated.
This rearrangement exposes the benefit rate required for nonnegative estimated value.
It does not establish that the rate is attainable.

Start with `V >= 0` in CHART-02.
Move fixed, harm, and variable terms to the other side.
Divide by positive `N*b`.

```text
FORMULA [CHART-04]: p_g >= p_h*(h/b) + k/b + F/(N*b)
WHERE: all symbols retain CHART-02 definitions; the right side is the
       break-even incremental beneficial-outcome fraction.
CONSTRAINT: N>0 and b>0; do not treat correlated benefit and harm as
            independent merely because this accounting expression is simple.
NUMERICAL EXAMPLE: ILLUSTRATIVE p_h=0.012, h=$180, b=$24, k=$0.65,
                   F=$4,000/month, N=4,000/month.
                   p_g >= 0.012*(180/24)+0.65/24+4,000/(4,000*24)
                        =0.090+0.027083+0.041667
                        =0.15875, or 15.875%.
INTERPRETATION: the assumed 18% benefit rate clears break-even by only
                2.125 percentage points; evidence uncertainty is material.
```

Arithmetic check: `0.012*7.5=0.09`.
Arithmetic check: `$0.65/$24=0.027083`.
Arithmetic check: `$4,000/$96,000=0.041667`.
Boundary: when `N` is small, fixed cost dominates and the required benefit rate rises sharply.
Decision consequence: a low-volume workflow may justify a manual or rule-based improvement instead.

### 4.5 Limits and counterexamples

A positive estimate is not a deployment authorization.
A high average score does not waive a zero-tolerance disclosure or action failure.
A frozen evaluation does not prove future performance after policy, population, or version changes.
A baseline with missing time or correction work understates its true cost.
A benefit estimate based on a correlated proxy can overstate realized value.
A queue inequality does not predict tail delay under bursty arrivals.
A human reviewer is not a control if review occurs after an irreversible action.
An abstention is not safe if it has no routed owner or time limit.

**Takeaway:** Derivations expose assumptions that a single ROI or accuracy number would otherwise hide.

## 5. Worked Numerical Example

### 5.1 Scenario and decision

**ILLUSTRATIVE teaching scenario; not measured.**
A regional equipment distributor receives warranty-claim packets from repair partners.
Each packet may require a request for missing documentation before a warranty analyst decides whether to continue review.
The current workflow uses a checklist and analyst-written email.
A proposed assistant extracts packet facts and drafts a missing-document request.
It may not approve, deny, alter a warranty record, or communicate externally without analyst review.
The decision owner is the warranty operations lead.
The service owner is the operations systems manager.
The risk counterpart is the information-security and records-management representative.

### 5.2 Charter excerpt

| Charter field | Illustrative statement | Why it matters |
|---|---|---|
| Trigger | packet enters “documentation incomplete” queue | avoids applying tool to complete packets |
| Eligible unit | one English-language packet with a permitted attached document and no active dispute flag | fixes denominator and exclusions |
| Decision | analyst sends, edits, or rejects one request for missing documentation | names human action |
| Baseline | analyst reads packet, consults checklist, drafts request | credible comparator |
| Intervention | assistant proposes missing items and draft wording with source spans | bounded assistance |
| Authority | analyst approves every outbound message | prevents hidden automation |
| Fallback | checklist and manual drafting remain available | preserves service on outage or abstention |
| Quality rubric | correct missing-item identification, no invented requirement, correct routing | makes “good draft” scoreable |
| Harm guardrail | zero unapproved external send; zero prohibited attachment disclosure | hard gates |
| Review capacity | four analysts can review 320 eligible packets/day at planned service time | operational feasibility |
| Promotion gate | all hard gates pass, quality improves or matches baseline, queue stays bounded, cost model remains within range | makes decision predeclared |

### 5.3 Baseline measurement plan

The baseline window is four ordinary operating weeks.
The baseline eligibility rule is applied before any analyst outcome is inspected.
Each eligible packet receives a unique packet identifier.
Each packet records intake timestamp, analyst assignment, completion timestamp, and status.
The baseline records draft time separately from total queue delay.
The baseline records correction and escalation events separately from first-pass completion.
The baseline records whether an attachment type was unavailable to the proposed intervention.
The baseline does not infer quality from speed alone.
The baseline sample includes ordinary and difficult packets in their observed mix.
The baseline report presents results by repair-partner channel only when sample size and policy permit meaningful interpretation.

### 5.4 Illustrative economics

Suppose 3,000 eligible packets arrive each month.
Suppose the assistant can produce a usable proposal for 78% of eligible packets.
Suppose analysts accept or lightly edit 65% of proposed drafts.
Suppose each accepted or lightly edited proposal saves 4 analyst minutes relative to baseline.
Suppose analyst loaded cost is $54/hour.
Suppose assistant usage cost is $0.22 per served packet.
Suppose review and exception handling add 0.5 minutes on every served packet beyond baseline handling.
Suppose integration and operating fixed cost is $3,600/month.
Suppose a materially wrong proposed requirement has an expected correction or customer-friction cost of $45.
Suppose its probability among served packets is estimated at 0.8%, pending validation.

The number served is `3,000 * 0.78 = 2,340 packets/month`.
The number benefiting from a usable draft is `2,340 * 0.65 = 1,521 packets/month`.
Gross saved minutes are `1,521 * 4 = 6,084 minutes/month`.
Gross saved hours are `6,084 / 60 = 101.4 hours/month`.
Gross labor value is `101.4 * $54 = $5,475.60/month`.
Usage cost is `2,340 * $0.22 = $514.80/month`.
Added review cost is `2,340 * 0.5/60 * $54 = $1,053/month`.
Expected wrong-requirement cost is `2,340 * 0.008 * $45 = $842.40/month`.
Illustrative net is `$5,475.60 - $514.80 - $1,053 - $842.40 - $3,600 = -$534.60/month`.

This proposal is not economically justified by the first point estimate.
It may still justify a narrow evidence-gathering evaluation if the evaluation cost is bounded and learning has value.
It must not be presented as a labor-saving deployment based on gross minutes alone.

### 5.5 Sensitivity analysis

If usable-draft acceptance rises from 65% to 75%, beneficial packets become `2,340*0.75=1,755`.
Saved labor value then becomes `1,755*4/60*$54=$6,318/month`.
With other assumptions unchanged, illustrative net becomes `$307.80/month`.
That small positive estimate remains fragile.

If fixed cost falls from $3,600 to $2,500, the original scenario becomes `$565.40/month` positive.
If wrong-requirement probability rises from 0.8% to 1.5%, expected harm becomes `$1,579.50/month`.
The original scenario then becomes `-$1,271.70/month`.
If eligible volume doubles but analyst capacity does not, queue and review feasibility must be re-estimated before using the volume result.
If intervention coverage falls because non-permitted attachments increase, the benefit estimate falls while fixed cost remains.

### 5.6 Review-capacity check

Assume the workflow routes every proposed draft to analysts.
That is 2,340 reviews/month.
At 20 working days, the average routed load is 117 reviews/day.
The declared review capacity is 320 eligible packets/day across four analysts.
The assistant review load is feasible only if it replaces enough baseline drafting rather than adding a second full review pass.
If it adds 0.5 minutes but requires separate verification of every extracted fact, the service-time assumption is invalid.
A time-and-motion observation during evaluation must test that assumption.

### 5.7 Evaluation protocol

Hold out a packet set from prompt, rule, and retrieval changes.
Score each packet against a rubric created before candidate outputs are reviewed.
Require a reviewer to mark every cited source span as present, absent, or misleading.
Score missing-item recommendations against the current authorized checklist.
Score routing decisions against an adjudicated reference process.
Record abstentions as outcomes rather than discarding them.
Compare candidate-assisted and baseline handling on the same eligibility definition.
Record reviewer correction minutes, not merely model latency.
Record prohibited disclosure attempts separately from ordinary quality errors.
Stop the evaluation immediately on a hard-gate violation according to the test protocol.

### 5.8 Decision statement

The illustrative charter recommends a proposal-only evaluation, not production rollout.
The assistant must have no authority to send messages or modify warranty state.
The evaluation must test whether actual correction time offsets the claimed drafting savings.
The owner must retain the manual checklist path throughout the evaluation.
Promotion requires revised economics with observed review time and a passed hard-gate record.
A negative or uncertain outcome is a valid decision result.

**Takeaway:** A worked charter can show that an attractive demo has negative or fragile economics once review, error, and fixed costs are counted.

## 6. Visual / Geometric Intuition

Think of every proposed intervention as a point in a four-axis decision space.
The horizontal axis is consequence of a wrong output.
The vertical axis is uncertainty about output quality or causal benefit.
The depth axis is breadth of data and authority boundary.
The fourth axis is reversibility of the resulting action.

```text
higher consequence
       ^
       |                    [state-changing automation]
       |                         / high proof burden
       |              [human-reviewed assistance]
       |                    /
       |      [search/checklist]
       +---------------------------------> broader data and authority boundary
       low uncertainty                 high uncertainty
```

Moving right can mean more data leaves a controlled environment.
Moving right can also mean more systems and identities participate in the workflow.
Moving upward can mean a wrong result affects money, rights, safety, or customer trust.
Moving into depth can mean the action becomes harder to reverse.
A proposal near the origin can often be tested with a simple baseline and bounded trial.
A proposal far from the origin needs stronger evidence, narrower permissions, explicit recovery, and more independent review.
A higher-capability model does not move a proposal closer to the origin.
A human review step lowers risk only when it occurs before the consequential action and has time and authority to act.

### 6.1 Evidence cone

```text
                 deployment claim
                       /\
                      /  \
             operating evidence
                    /      \
           representative evaluation
                  /          \
          task definition and baseline
                /              \
          observed workflow and owner
```

The lower layers support the upper layers.
A deployment claim without a baseline rests on no credible comparison.
A representative evaluation without an eligibility rule cannot establish its population.
Operating evidence without version information cannot establish what actually ran.
A charter records missing lower layers as OPEN instead of filling them with optimistic inference.

### 6.2 Reversibility ladder

Level 0 is information display with no workflow action.
Level 1 is a draft that a person can edit before use.
Level 2 is a recommendation that changes queue priority after approval.
Level 3 is a reversible system update with audit and tested undo.
Level 4 is a nonreversible or externally consequential action.
Higher levels require more stringent authorization, validation, monitoring, and recovery proof.
The ladder is about consequences, not about how sophisticated the model appears.

**Takeaway:** Proof burden grows with consequence, uncertainty, control breadth, and irreversibility—not with product branding.

## 7. Comparison / Selection Table

| Condition | Smallest credible intervention | Why | Do not choose when | Fallback |
|---|---|---|---|---|
| Current answer exists in authorized records | search or retrieval with visible source | user can inspect evidence | corpus is stale or access rule cannot be applied | manual search |
| Stable deterministic policy exists | rule, validation, or checklist | transparent and cheap | exceptions dominate and are measured | human triage |
| Numeric or categorical future outcome is defined | statistical or ML prediction | target and threshold can be evaluated | labels are delayed, invalid, or leak future information | rules plus data collection |
| User needs a draft but retains authority | human-reviewed generation | assists language or structure | review cannot occur at required granularity | templates and retrieval |
| Several options have explicit constraints | optimization or decision support | tradeoffs can be stated | objective/constraints are unknown | manual planning with instrumented data |
| Action changes a record | bounded automation | can reduce handoff after controls pass | identity, idempotency, approval, and rollback are absent | propose-only workflow |
| Data cannot leave a boundary | controlled local deployment or no intervention | data route is a hard constraint | permitted operating model is unavailable | retain baseline |
| Work arrives in bursts but can wait | queued or batched assistance | bounds cost and concurrency | service deadline is immediate | smaller synchronous baseline |
| Outcome has irreversible consequence | decision support with authorized approval | preserves human accountability | approver is nominal or unavailable | retain manual control |

### 7.1 Intervention selection questions

Can a rule solve the decision with lower error and lower governance burden?
Does the proposed system need facts, prediction, generation, optimization, or execution?
Are its inputs available at the moment of decision?
Can the output be inspected before a consequence occurs?
Does the workflow have a valid baseline to beat?
Can the team measure the relevant harm and correction work?
Is the proposed data route permitted in the actual configuration?
Can the system abstain safely when confidence or evidence is insufficient?
Can the owner explain why a simpler intervention is inadequate?
Can the team return to the baseline without losing state or auditability?

### 7.2 Gate hierarchy

A legality, safety, rights, or data-route prohibition is a hard gate.
An unavailable accountable owner is a hard gate for consequential use.
An untested nonreversible action is a hard gate.
An economics estimate below threshold is normally a prioritization gate, not proof of harm.
A quality result with wide uncertainty is an evidence gate requiring more evaluation.
A capacity shortfall is an operating gate requiring scope, staffing, or routing change.
A weak baseline is an evaluation gate requiring improved measurement.
An unavailable source body is an OPEN evidence status, not a reason to fabricate a result.

### 7.3 Charter review roles

The business owner defines the decision and acceptable business consequence.
The workflow owner validates the current baseline and feasible fallback.
The technical owner defines system boundaries, versioning, and observability.
The data owner validates provenance, availability time, and permitted use.
The security or privacy counterpart validates relevant control and data routes.
The risk or compliance counterpart identifies non-negotiable boundaries where applicable.
The evaluator maintains frozen tests and records scoring limitations.
The operator owns alerts, incident response, and rollback execution.
No role may be listed merely as a consultation ornament.
Each role needs an explicit approval, review, or operational responsibility.

**Takeaway:** Select the least powerful intervention that can meet a measurable need without violating a declared gate.

## 8. Failure Modes & Edge Cases

### CHART-F01 — Solution-first proposal

**Category:** CONCEPT_CONFUSION.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** The proposal begins with “use an agent” but cannot name an action owner or baseline.
**Competing causes:** enthusiasm for a technology; a hidden workflow pain; pressure to copy a competitor; lack of discovery work.
**Discriminating diagnostic:** Ask for one eligible unit, one changed action, one current comparator, and one consequence of being wrong.
**Expected signal:** Answers remain at slogan level or change between stakeholders.
**Immediate mitigation:** Pause product or vendor selection.
**Correction:** Write the decision and baseline fields with accountable owners.
**Prevention:** Require a signed problem statement before technical evaluation funding.
**Verification:** Independent reader can predict the action for a sample eligible unit.
**Mitigation risk:** Excessive discovery can delay low-risk experimentation.
**Bounded response:** Time-box discovery and retain only a narrow, observable workflow.

### CHART-F02 — Baseline laundering

**Category:** VIOLATED_ASSUMPTION.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** The intervention is compared with an outdated, unusually slow, or uninstrumented process.
**Competing causes:** genuine historical baseline; selective measurement; recent process improvement; different eligibility rules.
**Discriminating diagnostic:** Recompute both paths on the same frozen units and record exclusions.
**Expected signal:** Claimed gain shrinks or reverses when correction, escalation, and wait time are included.
**Immediate mitigation:** Remove unsupported ROI claim from the decision record.
**Correction:** Measure current workflow end to end using a shared eligibility definition.
**Prevention:** Version baseline protocol and forbid post-hoc denominator changes.
**Verification:** A reviewer can trace every reported metric to same-window units in both paths.
**Mitigation risk:** Instrumentation may add temporary workflow burden.
**Bounded response:** Sample a representative window rather than logging indefinitely.

### CHART-F03 — Future-information leakage

**Category:** VIOLATED_ASSUMPTION.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** Offline quality is excellent but live usefulness falls sharply.
**Competing causes:** future status field used as input; post-decision note included in retrieval; duplicate entities across split; temporal policy change.
**Discriminating diagnostic:** For every feature and document, prove its availability timestamp precedes the decision timestamp.
**Expected signal:** One or more high-value inputs disappear under point-in-time reconstruction.
**Immediate mitigation:** Invalidate the affected result rather than adjusting the score cosmetically.
**Correction:** Rebuild data extraction and split using decision-time availability.
**Prevention:** Maintain a feature and evidence availability ledger.
**Verification:** Re-run frozen evaluation without any post-decision artifact.
**Mitigation risk:** Lower valid score may discourage useful narrower use.
**Bounded response:** Narrow to inputs demonstrably available at decision time.

### CHART-F04 — Review-capacity illusion

**Category:** SCALE_FAILURE.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** The system routes uncertain cases to humans, but the queue grows.
**Competing causes:** volume growth; threshold too conservative; service-time tail; rework; staffing absence; duplicate arrivals.
**Discriminating diagnostic:** Compare arrival timestamps, route decisions, completion timestamps, and correction reasons.
**Expected signal:** Routed arrivals exceed completed reviews in the relevant peak window.
**Immediate mitigation:** Cap intake or disable low-value routing; do not silently abandon queued cases.
**Correction:** Change staffing, scope, threshold, or workflow with joint quality-and-harm evaluation.
**Prevention:** Specify capacity with productive hours, peak demand, and escalation allowance.
**Verification:** Backlog is non-increasing through declared peak periods.
**Mitigation risk:** A higher threshold can hide uncertainty and increase unreviewed error.
**Bounded response:** Monitor abstentions, coverage, and harmful errors together.

### CHART-F05 — Average-score masking a hard failure

**Category:** NUMERICAL_EDGE.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** A team argues that 98% quality offsets a prohibited disclosure or unauthorized action.
**Competing causes:** hard gate was not defined; severity was misclassified; evidence was averaged incorrectly; escalation authority unclear.
**Discriminating diagnostic:** Inspect whether the failure violates a stated zero-tolerance or non-compensable constraint.
**Expected signal:** The metric combines events that should have separate decision consequences.
**Immediate mitigation:** Stop the affected action path.
**Correction:** Add an independent pass/fail guardrail and test it before aggregate scoring.
**Prevention:** Define hard gates during charter review, not after an incident.
**Verification:** Evaluation report shows gate results separately from quality averages.
**Mitigation risk:** Too many hard gates can make any learning impossible.
**Bounded response:** Restrict hard gates to genuinely non-negotiable harms and document rationale.

### CHART-F06 — Hidden automation

**Category:** CONCEPT_CONFUSION.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** A “draft” is automatically sent, filed, or used to prioritize without an authorized human choice.
**Competing causes:** integration default; ambiguous UI; timeout behavior; workflow shortcut; misunderstood approval semantics.
**Discriminating diagnostic:** Trace an output from creation through every downstream state change and identify the actor of record.
**Expected signal:** A state change occurs before meaningful review or under a fallback identity.
**Immediate mitigation:** Disable write permissions and return to propose-only mode.
**Correction:** Separate draft creation, review, approval, and execution states with durable audit.
**Prevention:** Test negative paths, timeouts, and absent-approver behavior before rollout.
**Verification:** Replay shows no state change without the required approval event.
**Mitigation risk:** Added approval can create delay and reviewer fatigue.
**Bounded response:** Automate only reversible low-consequence steps after separate validation.

### CHART-F07 — Proxy-value trap

**Category:** CONCEPT_CONFUSION.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** Faster handling or more generated text is reported as business value without evidence of decision improvement.
**Competing causes:** measurement convenience; incentive misalignment; missing outcome data; short evaluation window.
**Discriminating diagnostic:** Ask how the proxy connects to the owner’s actual decision and what would falsify that connection.
**Expected signal:** No counterfactual or mechanism links proxy movement to desired consequence.
**Immediate mitigation:** Re-label proxy as process signal, not realized value.
**Correction:** Add decision-quality, correction, downstream outcome, or controlled comparison measures.
**Prevention:** Review metrics with the business owner before implementation.
**Verification:** Report distinguishes operational proxy from outcome and incremental estimate.
**Mitigation risk:** Outcome measurement can take longer than a pilot.
**Bounded response:** Use a preregistered proxy with explicit uncertainty and recheck date.

### CHART-F08 — Stale authorization or source material

**Category:** VIOLATED_ASSUMPTION.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** The assistant gives a plausible answer based on withdrawn policy or revoked access.
**Competing causes:** stale index; missing source version; permission cache; no document owner; retrieval ranking bias.
**Discriminating diagnostic:** Query a known superseded item and compare retrieved version, access decision, and source timestamp.
**Expected signal:** Obsolete or unauthorized content appears in the candidate context.
**Immediate mitigation:** Disable affected corpus or return no answer for the topic.
**Correction:** Version sources, enforce authorization before display, and set refresh ownership.
**Prevention:** Include stale-source regression cases and expiry monitoring.
**Verification:** Superseded material fails retrieval or is visibly marked unavailable under test.
**Mitigation risk:** Aggressive freshness filters lower coverage.
**Bounded response:** Route missing-evidence requests to manual search with explicit status.

### CHART-F09 — Economics excludes correction work

**Category:** NUMERICAL_EDGE.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** Unit cost looks low while analysts report more edits, escalations, or customer follow-up.
**Competing causes:** omitted review time; variable output length; retries; duplicated work; change-management training; inaccurate labor rate.
**Discriminating diagnostic:** Reconcile time logs, usage records, and corrected outputs to the cost model.
**Expected signal:** Actual minutes or calls exceed a model term that was assumed negligible.
**Immediate mitigation:** Freeze financial claims and update the estimate.
**Correction:** Add correction, escalation, retry, and support terms with ranges.
**Prevention:** Make each estimate traceable to an input owner and refresh trigger.
**Verification:** Recomputed model matches observed pilot inputs within declared tolerance.
**Mitigation risk:** Excess detail can obscure material drivers.
**Bounded response:** Model the largest uncertain drivers first and disclose exclusions.

### CHART-F10 — Fallback exists only on paper

**Category:** SCALE_FAILURE.
**Provenance:** HYPOTHETICAL teaching scenario.
**Symptom:** An outage or safety stop leaves staff unable to continue the workflow.
**Competing causes:** manual path retired; access missing; staff untrained; backlog procedure absent; state cannot be recovered.
**Discriminating diagnostic:** Execute a tabletop or controlled failover using ordinary staff and current permissions.
**Expected signal:** Users cannot find inputs, interpret status, or resume safely within declared time.
**Immediate mitigation:** Stop expansion and restore the known-safe process.
**Correction:** Document, train, and periodically test manual operation and state reconciliation.
**Prevention:** Treat fallback as a release requirement, not an appendix.
**Verification:** Drill meets service and reconciliation criteria without privileged improvisation.
**Mitigation risk:** Maintaining two paths costs time.
**Bounded response:** Limit automation scope until fallback maintenance is affordable.

### 8.1 Failure diagnosis sequence

Start with the observable symptom rather than a preferred cause.
List at least two mechanisms that could create that symptom.
Choose the smallest diagnostic that separates those mechanisms.
Record expected signals before looking at results.
Apply an immediate mitigation that preserves evidence and avoids larger harm.
Correct the causal mechanism rather than tuning a score blindly.
Verify with the previously failing path and a representative ordinary path.
Record mitigation side effects, remaining uncertainty, and re-review trigger.

**Takeaway:** A charter is credible when it anticipates competing explanations and makes each safeguard testable.

## 9. Complexity & Cost Analysis

### 9.1 Charter production cost

A charter is inexpensive compared with an unconstrained build, but it is not free.
The primary cost is structured stakeholder attention.
The minimum work is proportional to the number of distinct workflow decisions and material failure boundaries.
A single scope with `d` independent decisions needs at least O(`d`) decision contracts.
A risk register with `f` material failure mechanisms needs at least O(`f`) diagnostic and owner entries.
A comparison of `a` alternatives across `g` hard gates requires O(`a*g`) explicit checks.
These are planning counts, not runtime complexity claims.

### 9.2 Evaluation workload sizing

Let `n` be test cases.
Let `m` be reviewer minutes per case.
Let `j` be independent reviewers per case.
Let `a` be adjudication hours.
Evaluation labor hours are approximately `n*m*j/60 + a`.

```text
FORMULA [CHART-05]: H_eval = n*m*j/60 + a
WHERE: H_eval [hours]; n [cases]; m [minutes/case]; j [reviewers/case];
       a [hours] is separately estimated adjudication and protocol work.
CONSTRAINT: reviewer minutes must include source inspection and correction;
            independent reviewers require a resolution process for disagreement.
NUMERICAL EXAMPLE: ILLUSTRATIVE n=120, m=7 minutes, j=2, a=9 hours.
                   H_eval=120*7*2/60+9=28+9=37 hours.
INTERPRETATION: a “small” evaluation consumes nearly one full workweek before
                sampling, instrumentation, or engineering time.
```

Arithmetic check: `120*7*2=1,680 minutes`.
`1,680/60=28 hours`.
`28+9=37 hours`.
Boundary: halving cases reduces labor but may remove rare, high-consequence scenarios from coverage.
Decision consequence: budget evaluation review explicitly rather than treating it as volunteer overhead.

### 9.3 Cost categories to declare

| Cost category | Examples | Common exclusion | Recheck trigger |
|---|---|---|---|
| Discovery | workflow mapping, baseline instrumentation | stakeholder time | scope changes |
| Build | integration, data preparation, configuration | testing rework | interface changes |
| Usage | inference, retrieval, storage, egress | retries and output growth | workload or price changes |
| Human review | approval, correction, escalation | context switching | routing policy changes |
| Operations | monitoring, incident handling, release work | on-call burden | service level changes |
| Risk reserve | expected remediation or compensating control | correlated failure | harm model changes |
| Exit | export, replacement, retraining, retention work | contract delay | vendor or architecture changes |

### 9.4 Capacity versus latency

Capacity is sustainable completed units per window.
Latency is elapsed time for an individual unit.
A workflow can have adequate average capacity and unacceptable tail latency.
A workflow can have low average latency and insufficient capacity during bursts.
A charter must state which service property matters to the decision owner.
A priority queue can protect urgent work while increasing delay for other work.
That choice is a business and risk decision, not a mere system optimization.

### 9.5 Evidence-maintenance cost

Every versioned candidate adds evidence refresh work.
Every data-route change can invalidate a privacy or security conclusion.
Every workflow-policy change can invalidate a baseline or rubric.
Every material volume change can invalidate capacity and cost estimates.
A charter names these triggers so maintenance is deliberate rather than forgotten.

**Takeaway:** The affordable option is the one whose evaluation, review, operation, and exit costs remain acceptable—not the one with the lowest quoted model price.

## 10. Key Theorems / Results Reference

| Result or principle | Statement used here | Validity / limit | Evidence status |
|---|---|---|---|
| Linearity of expectation | expected aggregate contribution is the sum of stated expected components | does not justify probability or causality estimates | DERIVED |
| Conservation of queue work | backlog changes by arrivals minus completions | variable arrivals and service require queue analysis beyond averages | DERIVED |
| Point-in-time validity | evaluation inputs must be available before the decision they inform | requires accurate availability timestamps | DERIVED |
| Counterfactual comparison | incremental value needs a credible reference outcome | observational comparison can be confounded | DERIVED |
| Least privilege | permission should be limited to required authority | requires system-specific identity and control implementation | SOURCED concept; candidate evidence OPEN |
| Defense in depth | multiple controls may reduce single-point failure | controls can share common failure modes | HEURISTIC |
| Separation of duties | consequential approval and execution may require different authorities | applicability depends on workflow and policy | HEURISTIC |
| Goodhart-style warning | optimizing a proxy can degrade the underlying objective | not a numerical theorem or universal prediction | HEURISTIC |

### 10.1 Invariants for a chartered intervention

Every counted metric has a declared numerator, denominator, population, and window.
Every claimed improvement has a named baseline or is marked OPEN.
Every side effect has a named authority boundary.
Every hard gate has a pass/fail consequence and owner.
Every evaluation input is available at the decision time or excluded.
Every automated write has identity, idempotency or duplicate handling, and recovery behavior.
Every fallback is executable by the people expected to use it.
Every illustrative number is labeled as illustrative and not retold as observed performance.
Every source-body gap stays OPEN until the body is available and examined.

### 10.2 Evidence record template

| Field | Record |
|---|---|
| Claim ID | stable short identifier |
| Claim | exact bounded statement |
| Scope | population, workflow, configuration, and time |
| Artifact | report, log, test result, contract, or source |
| Version/date | candidate and artifact version or date |
| Support | what the artifact directly supports |
| Limitation | what it cannot establish |
| Owner | person or role responsible for recheck |
| Expiry trigger | change that invalidates reliance |
| Status | DERIVED, ILLUSTRATIVE, VERIFIED, SOURCED, HEURISTIC, or OPEN |

### 10.3 Result misuse warnings

Expected value cannot turn an unacceptable action into an acceptable one.
A capacity average cannot promise prompt service under bursts.
A quality metric cannot reveal a missing eligibility population.
A valid baseline cannot justify a different workflow without transfer evidence.
A test can establish only the configuration and conditions actually tested.
An audit log proves neither correctness nor user benefit by itself.
A human-in-the-loop label proves nothing until timing, authority, and workload are observed.

**Takeaway:** Results and invariants are useful because they state both what follows and where the argument stops.

## 11. Interview / Exam Questions

### Q1. What must be defined before choosing a model?

**Answer:** Define the workflow trigger, eligible unit, decision, owner, baseline, consequence, and fallback first.
**Reasoning:** These fields determine whether a model is needed and how it could be evaluated.
**Follow-up:** Ask which field would change if the output were unavailable.
**Acceptable alternative:** A rule, search path, or manual process may be selected after the definition.
**Red flag:** Starting with a model family or vendor name.
**Scoring:** Full credit names decision and baseline; partial credit names only a user need.

### Q2. Compute coverage when 680 of 800 eligible requests receive a usable response.

**Answer:** `680/800=0.85`, or 85% coverage.
**Reasoning:** Served and eligible counts use the same weekly population.
**Follow-up:** What happened to the other 120 requests?
**Acceptable alternative:** Report 15% unserved alongside coverage.
**Red flag:** Calling coverage an accuracy score.
**Scoring:** Full credit supplies arithmetic and denominator boundary.

### Q3. Why is gross labor time saved not enough to approve a charter?

**Answer:** Review, correction, fixed, usage, harm, and capacity costs may exceed gross saved time.
**Reasoning:** The worked example’s gross savings were positive while net estimate was negative.
**Follow-up:** Which cost would you measure first if correction time is unknown?
**Acceptable alternative:** Require a bounded evaluation before a full estimate.
**Red flag:** Treating generated output count as realized value.
**Scoring:** Full credit names comparator and at least three omitted terms.

### Q4. Derive the break-even benefit rate from CHART-02.

**Answer:** Rearranging `V>=0` yields `p_g >= p_h*(h/b)+k/b+F/(N*b)` when `N>0` and `b>0`.
**Reasoning:** Move harm, variable, and fixed terms to the benefit side, then divide by `N*b`.
**Follow-up:** What happens to the fixed-cost term when volume rises?
**Acceptable alternative:** Explain algebra verbally with correct conditions.
**Red flag:** Dividing by zero or omitting expected harm.
**Scoring:** Full credit gives formula, conditions, and interpretation.

### Q5. A proposal has 95% offline quality but its review queue grows. Diagnose it.

**Answer:** Compare routed arrivals with completions, routing fraction, service time, rework, and eligibility changes.
**Reasoning:** Quality does not determine the number of reviewed units or their handling time.
**Follow-up:** What immediate mitigation preserves safety?
**Acceptable alternative:** Cap intake while preserving manual fallback.
**Red flag:** Raising confidence threshold without checking missed harmful cases.
**Scoring:** Full credit gives competing causes and a discriminating check.

### Q6. When is a human-in-the-loop claim inadequate?

**Answer:** When the person lacks authority, receives output after the side effect, lacks time/capacity, or is unable to inspect relevant evidence.
**Reasoning:** Review must causally control the action.
**Follow-up:** How would you test that in a timeout path?
**Acceptable alternative:** Trace actor, timestamp, and state transition.
**Red flag:** Accepting a UI label as proof.
**Scoring:** Full credit identifies timing, authority, and workload.

### Q7. What is wrong with evaluating a claim using documents created after the decision?

**Answer:** It is leakage because the system had information unavailable at decision time.
**Reasoning:** The result overstates deployable performance.
**Follow-up:** What artifact detects it?
**Acceptable alternative:** A point-in-time availability ledger.
**Red flag:** Calling it acceptable because the document is now accessible.
**Scoring:** Full credit explains invalidation and repair.

### Q8. An intervention has positive expected value but could send an unauthorized external message. What decision follows?

**Answer:** Treat unauthorized sending as a hard gate; retain propose-only mode or remediate and retest.
**Reasoning:** Expected value does not compensate a non-negotiable authority violation.
**Follow-up:** What evidence proves remediation?
**Acceptable alternative:** Replay an approval and timeout test with audit trace.
**Red flag:** Averaging the event into quality.
**Scoring:** Full credit distinguishes hard gate from weighted metric.

### Q9. What is a plausible wrong method for baseline selection?

**Answer:** Compare the candidate to a deliberately weak historical process instead of the current workflow on the same eligible units.
**Reasoning:** It launders the baseline and inflates improvement.
**Follow-up:** What must be frozen before comparison?
**Acceptable alternative:** Eligibility rule, window, rubric, and exclusions.
**Red flag:** “Any manual process is an acceptable baseline.”
**Scoring:** Full credit explains denominator and counterfactual risk.

### Q10. How would a doubled volume change the warranty-draft example?

**Answer:** Recompute served count, review load, usage cost, fixed-cost allocation, service capacity, quality slices, and harm exposure.
**Reasoning:** Linear arithmetic alone is insufficient when staffing, queues, or input mix change.
**Follow-up:** Which result may improve mechanically?
**Acceptable alternative:** Fixed cost per unit may fall if capacity remains valid.
**Red flag:** Doubling net value without checking review capacity.
**Scoring:** Full credit names operational and economic rechecks.

### Q11. What evidence is required to call an illustrative number measured?

**Answer:** A defined population, time window, method, source artifact, version or environment, result, and limitations.
**Reasoning:** Arithmetic verification does not turn constructed inputs into observation.
**Follow-up:** How should unavailable source bodies be recorded?
**Acceptable alternative:** Mark them OPEN with needed evidence.
**Red flag:** Treating a filename as empirical support.
**Scoring:** Full credit distinguishes illustrative, verified arithmetic, and sourced measurement.

### Q12. A team wants to automate a reversible record update. What charter additions are needed?

**Answer:** Add authority, identity, idempotency, validation, audit, rollback, failure recovery, and a bounded rollout gate.
**Reasoning:** Reversibility lowers but does not eliminate control requirements.
**Follow-up:** What test catches duplicate execution?
**Acceptable alternative:** Replay the same idempotency key and inspect durable state.
**Red flag:** Assuming reversibility makes review unnecessary.
**Scoring:** Full credit includes recovery and verification.

### Q13. How should a charter handle uncertainty in benefit and harm estimates?

**Answer:** State assumptions, ranges or sensitivities, evidence maturity, and the decision consequence when a key assumption changes.
**Reasoning:** A point estimate hides brittle prioritization.
**Follow-up:** Which input in CHART-02 changed the illustrative example from positive to negative?
**Acceptable alternative:** Harm probability or benefit consequence, with arithmetic.
**Red flag:** Reporting only the favorable scenario.
**Scoring:** Full credit includes uncertainty and re-review trigger.

### Q14. What makes a fallback real rather than aspirational?

**Answer:** Current access, trained users, preserved input/state, named ownership, and a tested recovery or manual procedure.
**Reasoning:** A document alone does not restore service under stress.
**Follow-up:** How often should it be tested?
**Acceptable alternative:** At a risk-appropriate cadence and after material change.
**Red flag:** “The old system still exists somewhere.”
**Scoring:** Full credit includes execution evidence and reconciliation.

### Q15. Defend retaining the baseline after an evaluation.

**Answer:** Retain it when hard gates fail, evidence is insufficient, economics are fragile, capacity is infeasible, or improvement is not credible on comparable units.
**Reasoning:** A charter makes non-adoption a valid controlled outcome.
**Follow-up:** What should the record preserve after rejection?
**Acceptable alternative:** Evidence, limitations, decision owner, and recheck trigger.
**Red flag:** Treating retention as project failure.
**Scoring:** Full credit connects the outcome to predeclared gates.

**Takeaway:** Strong answers identify the decision contract, test competing explanations, and name the consequence of changing a constraint.

## 12. Quick Reference Card

| Cue | Rule | Use when | Critical assumption | Avoid when |
|---|---|---|---|---|
| Start | Name trigger, eligible unit, decision, owner, and consequence | any AI proposal | workflow can be observed | outcome is undefined |
| Compare | Measure current path on the same eligible units | claiming improvement | baseline is current and comparable | denominator or window changes |
| Bound | State inputs, outputs, authority, abstention, and fallback | designing intervention | control boundary is known | “assist” hides automation |
| Value | `V=N(p_g*b-p_h*h-k)-F` | preliminary prioritization | assumptions share a window | called observed ROI |
| Break even | `p_g >= p_h*(h/b)+k/b+F/(N*b)` | testing benefit plausibility | `N>0`, `b>0` | harm or fixed cost omitted |
| Size review | `q*N <= r` | human review is required | stable capacity approximation | tail delay and rework ignored |
| Gate | Separate hard constraints from scored preferences | rights, safety, data, authority matter | gate owner can act | average score waives violation |
| Evaluate | Freeze population, rubric, version, and comparator | running a pilot | decision-time inputs are available | tuning contaminates final test |
| Operate | Name monitor, alert, fallback, rollback, and re-review trigger | moving beyond a demo | owners and access exist | fallback is untested |
| Stop | Retain baseline or narrow scope when a gate fails | evidence is weak or harm is unacceptable | non-adoption is allowed | uncertainty is relabeled success |
