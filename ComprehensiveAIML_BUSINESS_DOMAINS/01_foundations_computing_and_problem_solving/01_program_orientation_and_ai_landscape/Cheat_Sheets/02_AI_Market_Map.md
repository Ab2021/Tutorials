---
collection: "program-orientation-ai-landscape"
file_id: "02"
topic: "AI market map: capability layers, deployment modes, and control boundaries"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Any change to a candidate system's data route, authority to act, review capacity, or the commercial terms of a selected vendor"
notation_style: "plain-text"
prerequisites: ["01_AI_Use_Case_Charter.md"]
pattern_tags: ["problem-framing", "build-buy", "deployment-choice", "human-oversight"]
source_status: "OPEN — inventory of source containers visible; their bodies unavailable"
owner_role: "product and technical decision owner"
source_index: "INDEX.md"
---

# AI Market Map: Capability, Control, and Deployment Choice

## 1. File Header

> **Purpose:** Place an AI proposal on a decision map before any vendor, model, or
> product comparison begins, so that the *kind* of decision, the *asset* it consumes,
> and the *control boundary* it changes are explicit and testable.
> **Prerequisites:** [AI Use-Case Charter](01_AI_Use_Case_Charter.md); comfort with
> distinguishing a count from a rate, a point estimate from an interval, and an
> estimate from a measurement.
> **Key outcome:** Given a workflow description and a set of operating constraints,
> you can name the smallest credible capability class and deployment mode, compute
> its coverage, review load, unit cost and break-even volume, and state the single
> observation that would overturn the choice.
> **This file does NOT cover:** model training recipes and hyperparameter practice;
> procurement law, contract clauses or jurisdiction-specific regulation; any claim
> that a named commercial product possesses a particular capability, price, or
> certification.
> **Level:** 1 — Foundations
> **Version / reviewed:** 2.0 / not reviewed
> **Evidence:** DERIVED classification rules and arithmetic; ILLUSTRATIVE workloads
> and money values; SOURCED only where an external method reference is named inline;
> OPEN where a source body is unavailable.
> **Retrieval cue:** If a proposal says "we need AI" and cannot name the decision, the
> action, and the boundary it touches, start at Section 2 and do not leave it until
> those three words are on the page.

### 1.1 What a "market map" is for

A market map answers one question: *which class of thing is being purchased or built,
and what evidence does that class owe before it is allowed to act?* It is not a
landscape diagram of logos, a funding summary, or a maturity curve with a company
placed on it. Those artifacts are marketing surfaces. They help nobody decide whether
a given workflow should use retrieval instead of generation, or a managed API instead
of a controlled local runtime, or no automation at all.

The map earns its name because it has coordinates. Every proposal lands somewhere,
and the coordinates determine what must be proved. A proposal that cannot be
*placed* — because nobody can say what decision it changes — is not early in its
lifecycle, it is undefined. Undefined proposals absorb budget indefinitely because
there is no condition under which they can succeed or fail.

Three coordinate axes are enough for practical work. **Capability** is the
transformation: retrieve evidence, predict a label or value, generate an artifact,
allocate under constraints, or execute a sequence of actions. **Control** is where
inference, data, and state live, and who may change them: local runtime, managed API,
managed workflow platform, or an autonomous actor with credentials. **Evidence** is
the proof burden implied by consequence: a reversible suggestion inside a document
needs a weaker protocol than an irreversible change to a payment or a patient record.

Keep the axes separate. A common error is to collapse them and speak of "an AI
product", which mixes a model family with a hosting arrangement and an authority
level. Two systems can use the same model weights and land in completely different
places on the map, because one only proposes text while the other can move money.

### 1.2 The five layers any offering decomposes into

Comparisons fail when they compare across layers. Split every offering into five
layers, and evaluate each on its own terms.

**Assets.** Data with a provenance and a permitted use, labels with a validation
history, compute, identity, and the network paths between them. An asset question
sounds like "may this field leave this environment, for this purpose, for this long?"
It is answered by a data owner and a privacy or records function, not by a benchmark.

**Capability.** A declared input-to-output contract: given this record, return a
ranked list, a probability, a draft, a feasible allocation, or a tool call sequence.
Capability is comparable only within a task. Two vendors' "extraction" offerings may
extract different fields under different tolerances, which makes their headline
accuracy numbers unrelated quantities.

**Delivery.** How the capability reaches the workflow: a library inside an existing
service, a hosted API, a packaged application, a managed workflow product, an
on-device component, or a batch job. Delivery determines latency shape, failure
semantics, upgrade control, and where logs accumulate.

**Operation.** Who watches it at 03:00, who owns an incident, who approves a version
change, how access is granted and revoked, and how the system is rolled back. Every
deployment has an operator. If no operator is named, the deployment is incomplete
regardless of how good the model is.

**Accountability.** Who is answerable for the consequence of a wrong output or a
wrong action, and through what mechanism they can be made whole: an approval step, an
appeal path, an SLA credit, an insurance policy, or a contractual indemnity. This
layer is frequently absent from technical comparisons and is usually the one that
decides the purchase.

A vendor comparison that evaluates only the capability layer answers the least
consequential question. The expensive surprises live in delivery, operation, and
accountability.

### 1.3 Scope in and out

In scope: classifying a proposal; selecting the smallest sufficient capability
class; choosing a deployment mode under constraints; sizing review load, unit cost,
and break-even volume; specifying an acceptance protocol that makes vendors
comparable; naming promotion gates and disqualifiers; diagnosing the recurring ways
these choices go wrong.

Out of scope and explicitly handed off: the detailed charter method lives in
[AI Use-Case Charter](01_AI_Use_Case_Charter.md); claim-level vendor interrogation
lives in [Technology Due Diligence](03_Technology_Due_Diligence.md); statistical
estimation and experiment design belong to the data-analytics part; distributed
systems reliability practice belongs to the platform part. This sheet uses those
results, it does not re-derive them.

### 1.4 How to use the sheet in sequence

A workable order that fits inside a single working session:

1. Write the eligible unit and the decision in one sentence each (Section 3).
2. Classify the decision type and read off candidate capability classes (Section 7).
3. Apply the control gate: list every field that crosses a boundary and every
   authority the system would hold (Sections 2 and 7).
4. Compute coverage, review feasibility, unit cost, and break-even volume
   (Section 4 then Section 5 as a template).
5. Name the acceptance protocol and the evidence gate (Sections 7 and 9).
6. Record the disqualifier and the fallback (Section 12).

If step 3 eliminates every option, the correct output is "retain the baseline",
written down and signed by the decision owner. That is a finished analysis, not a
failed one.

### 1.5 Evidence posture and source boundary

Nothing in this sheet asserts that any real product has a given capability, price,
latency, or certification. Every workload, rate, and currency figure is constructed
for teaching and labelled ILLUSTRATIVE even where the arithmetic is checked twice.
Method references are cited where a named result is used; those references support
the method, not any illustrative number attached to it.

The source inventory for this module lists several containers whose bodies were not
available when this sheet was written. A container name is a provenance identifier;
it is not evidence of content, and it cannot be used to infer what a lesson "should"
say. Where a topic depends on unavailable material, the limitation is recorded as
OPEN beside the affected claim rather than filled with plausible-sounding invention.

### 1.6 Symbols introduced in this sheet

| Symbol | Meaning | Units | Kind | Domain |
|---|---|---|---|---|
| `N` | eligible units in a window | units/window | count | positive integer |
| `n_served` | units receiving a usable response | units/window | count | `0..N` |
| `c` | coverage | fraction | rate | `0..1` |
| `p_g` | probability of a beneficial outcome per served unit | fraction | probability | `0..1` |
| `p_h` | probability of a harmful outcome per served unit | fraction | probability | `0..1` |
| `b` | money value of a beneficial outcome | currency/unit | scalar | `>=0` |
| `h` | money cost of a harmful outcome | currency/unit | scalar | `>=0` |
| `k` | variable operating cost | currency/unit | scalar | `>=0` |
| `F` | fixed cost per window | currency/window | scalar | `>=0` |
| `V` | expected incremental value | currency/window | scalar | any sign |
| `q` | fraction of served units routed to human review | fraction | rate | `0..1` |
| `r` | review capacity | units/window | rate | `>=0` |
| `a` | adoption | fraction | rate | `0..1` |
| `y` | acceptance fraction of outputs | fraction | rate | `0..1` |
| `rho` | utilization of a service or reviewer pool | fraction | rate | `0..1` stable |

**Takeaway:** A proposal is placeable only when its decision, authority, and data
route are all nameable; everything else in this sheet is arithmetic performed on
those three declarations.

## 2. Concept Map / Mental Model

### 2.1 The placement sequence

```text
[Workflow trigger]
        |
        v
[Eligible unit defined?] --no--> [Stop: write the charter first]
        |yes
        v
[Decision type: retrieve / predict / generate / allocate / act]
        |
        +--> deterministic and stable? -------> [rule / validator baseline]
        |
        +--> answer exists in controlled text? -> [retrieval with citations]
        |
        +--> target numeric with labels? ------> [statistical or ML prediction]
        |
        +--> drafting / transformation? -------> [generative assistance]
        |
        +--> feasible allocation under limits? -> [optimisation]
        |
        +--> state change across steps? -------> [bounded automation]
        |
        v
[CONTROL GATE: which fields leave the boundary?
               which credentials does it hold?
               what can be undone?]
        |fail --> [narrow authority | change boundary | retain baseline]
        |pass
        v
[CAPACITY & ECONOMICS: coverage, review load, unit cost, break-even]
        |fail --> [reduce scope | change mode | retain baseline]
        |pass
        v
[EVIDENCE GATE: acceptance protocol, slice results, rollback drill]
        |
        v
[NAMED OPERATOR + MONITORING + REVIEW DATE]
```

The order matters. Control before economics, because a cheaper option inside a
prohibited boundary has no value. Economics before evidence, because a perfectly
validated system whose review load exceeds staffing cannot be operated, and there is
no point buying an expensive evaluation for it.

### 2.2 Why the gate order is not negotiable

Teams frequently run the sequence backwards. They start with a vendor demo, then
measure whether it is accurate, then discover it cannot receive the data, then
return to the market. The time lost is not the main cost; the main cost is that the
decision has already been socially made by the time the constraint appears, so the
constraint gets negotiated away instead of the proposal being narrowed.

Running control first has a second benefit. It shrinks the candidate set before any
evaluation spend, which is where most of the money goes in a comparison programme.
A gate that removes three of five candidates in an afternoon is worth more than a
leaderboard that ranks all five on a task nobody validated.

### 2.3 The map as a stack, not a list

```text
+---------------------------------------------------------------+
| ACCOUNTABILITY  approval | appeal | indemnity | SLA | audit    |
+---------------------------------------------------------------+
| OPERATION       monitoring | incident | release | access | cost |
+---------------------------------------------------------------+
| DELIVERY        library | API | app | workflow | device | batch |
+---------------------------------------------------------------+
| CAPABILITY      retrieve | predict | generate | allocate | act  |
+---------------------------------------------------------------+
| ASSETS          data rights | labels | compute | identity | net |
+---------------------------------------------------------------+
```

Read the stack bottom-up when assessing feasibility, and top-down when assessing
risk. Feasibility questions ("can this be built and fed?") are answered at the
bottom. Risk questions ("what happens when it is wrong, and who answers?") are
answered at the top. A proposal that is strong at the bottom and silent at the top
is a prototype, not a deployment.

### 2.4 Two loops that decide most outcomes

**The narrowing loop.** Start wide (all capability classes that could serve the
decision), apply the control gate, apply capacity and economics, apply the evidence
gate. Each application either narrows the set or produces a written exception with
an owner. The loop terminates when one option remains, or when none does and the
baseline is retained.

**The re-entry loop.** After deployment, three signals push a proposal back into the
map: a data-route change (new field, new subprocessor, new region), an authority
change (the system may now act rather than propose), and a quality or cost drift that
invalidates the sizing. Each signal reopens a specific gate rather than the whole
analysis, which is what makes maintenance tractable.

### 2.5 Reading the map with a concrete example

Consider a proposal to "use AI to handle supplier emails". Placed on the map, it is
not one proposal but four, each with a different coordinate:

- Classify each message into a request type: a prediction with a small label set, low
  consequence, cheap to review.
- Extract structured fields into an ERP staging record: an extraction task where
  errors are correctable before commit, so review can be sampled rather than total.
- Draft a reply for a human to send: generation, reversible, moderate review cost.
- Answer the supplier directly with commitments: generation plus action, high
  consequence, requiring authority limits, template constraints, and audit.

The fourth variant is roughly the same model behind the first, and yet it needs an
order of magnitude more evidence, because the consequence changed. That gap is the
entire point of the map: the technology did not change, the coordinates did.

**Takeaway:** Place the proposal on capability, control, and evidence axes in that
gate order, and let the coordinates rather than the demo determine what must be
proved.

## 3. Foundational Definitions

### 3.1 Core vocabulary

| Term | Formal meaning | Unit / type | Valid use | Nonexample |
|---|---|---|---|---|
| Eligible unit | One item about which a decision is made, counted once in a fixed window | count (ticket, request, record, shipment) | denominator for coverage, cost, and load | "our users" |
| Decision | The choice the workflow makes after seeing an output | action in a policy | sizing and gating | "leverage AI" |
| Capability | Declared input-to-output contract | transformation | comparing like with like | vendor brand |
| Deployment mode | Where inference and state run, and who operates them | control arrangement | boundary and latency analysis | model quality |
| Control boundary | The set of environments, identities, and jurisdictions data and authority may cross | policy set | hard gates | a security badge |
| Autonomy | The set of state changes the system may perform without prior human approval | authority level | risk classification | "agentic" as a label |
| Proof burden | Evidence required before a given authority level is granted | protocol + gate | promotion decisions | test suite size |
| Retrieval | Selecting and presenting existing evidence | ranked records | grounded lookup | free-form answer |
| Prediction | Estimating a defined target before the outcome exists | probability or value | thresholded action | post-hoc scoring |
| Generation | Producing a candidate artifact from context | candidate text, code, media | assisted work under review | factual certification |
| Allocation | Choosing a feasible assignment under constraints | assignment vector | explicit trade-offs | unconstrained ranking |
| Automation | Executing a state-changing step | side effect | gated by identity and reversibility | a recommendation |
| Abstention | Deliberate routing away from automated handling | fraction of units | safety and quality control | a system crash |
| Human in the loop | A named person can inspect, change, reject, or escalate | ownership contract | consequence above validated autonomy | a checkbox nobody reads |
| Disqualifier | A condition under which an option is removed regardless of score | predicate | selection | a weighted penalty |
| Gate | A pass/fail condition checked before the next stage | predicate + owner | promotion | a dashboard chart |
| Acceptance protocol | The frozen procedure by which outputs are judged | document | vendor comparison | a demo script |
| Benchmark | A published measurement on a stated task and dataset | reported number | rough orientation | proof of fitness |
| Unit economics | Cost and value per eligible unit | currency/unit | pricing and break-even | monthly invoice total |
| Break-even volume | Volume at which two options cost the same | units/window | build-versus-buy | payback period alone |
| Review capacity | Reviewer throughput available in a window | units/window | feasibility | headcount |
| Drift | Change in inputs, labels, or environment that alters performance | distributional change | monitoring | a single bad output |
| Version pinning | Fixing the exact artifact a deployment uses | identifier | reproducibility | "latest" |
| Shadow mode | Running a candidate without changing the workflow action | deployment stage | pre-pilot evidence | a pilot with silent use |

### 3.2 Capability is a contract, not a category

The word "capability" is doing technical work here. It means that for a declared
input and declared output, there is a testable statement of what the system will
produce and what it will refuse. "Summarisation" is not a capability in this sense,
because it does not say what a summary must contain, how long it may be, whether it
may introduce claims absent from the source, or what it does with a document that
exceeds the context window.

A usable capability statement reads more like: given one supplier message of up to
4,000 characters, return a label from a fixed set of nine, with an explicit
`unknown` option, and never return a free-text justification. Now the statement can
be tested, priced, and compared. Two vendors can be held to it. Disagreement about
quality becomes disagreement about a measurable quantity rather than about taste.

### 3.3 Deployment mode is about control, not location alone

It is tempting to define deployment mode geographically — "cloud" versus "on-premise".
That framing is obsolete and misleading, because a managed service can run inside a
customer's cloud tenancy, and a self-hosted model can call an external index. Define
the mode by three properties instead: who operates the runtime, where the data and
the model weights reside, and who holds the authority to change the version or the
policy.

Under that definition, a "self-hosted" deployment whose weights are pulled from an
unpinned remote source on every start is operationally managed by someone else and
is not reproducible. Conversely, a vendor-hosted API with a pinned model version, a
data-processing agreement, and no training on customer content can be a defensible
control boundary for many workloads. Location is one input to the decision, not the
decision.

### 3.4 Autonomy is a graded authority, not a binary

Autonomy is best described as the set of effects the system may cause without prior
approval, plus the conditions under which those effects are permitted. Useful grades:

- **Propose only.** The system emits suggestions; a human performs every action.
- **Propose with pre-fill.** Suggestions populate a form; a human commits it.
- **Act with sampling.** The system acts on cases above a confidence threshold and
  routes the rest; a sample of actions is audited afterwards.
- **Act within limits.** The system acts freely under amount or scope caps, and
  escalates above them.
- **Act with rollback.** The system acts broadly but every action is reversible
  within a stated window.
- **Irreversible autonomy.** Effects cannot be undone; only pre-action validation
  constrains harm.

Each grade requires strictly more evidence than the one above it. A team that
reports "we deployed an agent" without naming the grade has said nothing about risk,
because the same technology stack spans all six grades.

### 3.5 Proof burden, stated plainly

Proof burden is the amount and kind of evidence required before an authority level is
granted. It has four components: a *protocol* (how cases are selected and judged), a
*population* (which slices must be represented), a *threshold* (what performance is
required, with uncertainty), and a *gate owner* (who may waive it, and on what
record). Without all four, "we validated it" is an opinion.

The burden rises with irreversibility, with the size of the affected population, with
the asymmetry between benefit and harm, and with the difficulty of detecting a bad
outcome after the fact. It falls when outputs are cheap to check, when errors are
self-announcing, and when the human reviewer is genuinely authoritative rather than
decorative.

### 3.6 Market vocabulary that misleads

**"Foundation model"** names a training regime, not a fitness claim. A foundation
model can be excellent at open-ended drafting and unusable for a nine-way
classification with a hard refusal requirement, because the failure is in the
contract, not the weights.

**"Agentic"** names an architecture that lets a model choose and sequence tool calls.
It says nothing about permissions, stopping conditions, idempotency, or audit. Those
must be designed and are the substance of the risk.

**"Copilot"** names an interaction pattern with a human in the loop. It implies
review, but a review that consists of pressing Tab is not a control. Verify that the
reviewer can actually detect and reject a bad output at the required rate.

**"Fine-tuned on your data"** names an adaptation method. It does not imply that
your data is excluded from other uses, that the adaptation generalises beyond the
tuning distribution, or that the vendor's evaluation is comparable to yours.

**"Enterprise-grade"** names nothing measurable. Ask for the specific control, the
specific test, or the specific clause.

### 3.7 Symbol roles, restated for arithmetic safety

Every formula in Section 4 combines a count, a rate, or a currency amount with at
least two other quantities. Three arithmetic hazards recur and are worth naming once:

*Mixing denominators.* A rate computed on served units (say 93% correct) cannot be
multiplied by total eligible units to obtain correct outcomes, because the unserved
units are not represented. Multiply a rate only by the population it was measured on.

*Mixing windows.* A monthly fixed cost divided by a daily volume produces a
meaningless per-unit figure unless the window is converted first. Carry the window as
a unit and cancel it explicitly.

*Treating an estimate as an observation.* In this sheet, `p_g` and `p_h` are almost
always assumed values. An expected-value result inherits every error in them, and the
sensitivity analysis is what makes the result honest.

**Takeaway:** Define the eligible unit, the capability contract, and the autonomy
grade before comparing anything; unstated authority is the most expensive omission in
an AI proposal.

## 4. Core Formulas with Full Derivation

### 4.0 Shared setup for every formula in this section

All formulas below describe one window of operation, taken as one calendar month
unless stated otherwise, over a population of eligible units that has been frozen
before measurement. Freezing matters: if the eligibility rule is edited between the
baseline measurement and the candidate measurement, every ratio computed afterwards
compares two different populations and inherits an unknown bias.

The quantities fall into three dimensional families that must never be added
directly: counts (`N`, `n_served`, `n_use`), rates (`c`, `q`, `y`, `a`, `rho`), and
money (`b`, `h`, `k`, `F`, `V`). Rate times count yields a count; count times money
per unit yields money; money per window divided by count per window yields money per
unit. Every numerical example below performs that cancellation explicitly so the
habit transfers.

### 4.1 Base: eligible coverage

**Validity before use.** The inclusion rule is frozen and written down. Each eligible
unit is counted exactly once, retries and duplicates are collapsed to one unit, and
the window is identical for numerator and denominator. Coverage measures reach; it
says nothing about whether the reached units received a correct answer.

**Derivation.** Coverage is a part-over-whole ratio over a finite set. Let the set of
eligible units in the window be `E` with `|E| = N`, and let `S` be the subset that
received a usable response, with `|S| = n_served`. Because `S` is a subset of `E`,
the count of served units cannot exceed the count of eligible units, and the ratio
`|S| / |E|` lies in the closed unit interval. Usable is defined by a rule fixed in
advance — the response arrived, parsed, and passed the minimum format contract — not
by whether the reader liked it.

```text
FORMULA [MAP-01]: c = n_served / N
WHERE: c coverage [fraction, 0..1];
       n_served usable responses [units/window], counted once per unit;
       N eligible units [units/window], same population, same window.
CONSTRAINT: same eligibility rule and window for both quantities; retries collapse
            to one unit; "usable" defined before measurement, not after.
NUMERICAL EXAMPLE: N = 12,000 eligible supplier messages/month;
                   n_served = 9,300 messages received a parsed, contract-valid
                   response; c = 9,300 / 12,000 = 0.775 = 77.5%.
INTERPRETATION: any quality statistic measured on the 9,300 served units describes
                at most 77.5% of the population; the remaining 2,700 units are
                outside the claim and must be reported as unserved, not as failures
                or as successes.
```

**Boundary and counterexample.** Coverage can rise while quality falls, and it can
fall while measured quality rises. If a strict safety filter rejects the hardest
30% of units, coverage drops to 70% and accuracy on the remainder may climb to 96%;
the *number* of correct outcomes can be lower than before and the harm concentrated
in the rejected units, which are now handled by whatever the fallback does. Reporting
a single accuracy figure without coverage allows exactly this substitution. Report
the pair, and report the fallback's performance on the excluded units.

**Dimension check.** units/window divided by units/window is dimensionless. A common
error is to compute coverage with a daily numerator over a monthly denominator,
which inflates or deflates by the ratio of working days; the result is a number with
no meaning that nonetheless looks plausible.

### 4.2 General: expected incremental value

**Validity before use.** Baseline and intervention are evaluated on comparable
eligible units in the same window. `p_g` and `p_h` refer to the same action and the
same outcome definition. `b` counts only value that the baseline does not already
deliver; attributing baseline value to the intervention is the single most common
way these estimates are inflated. Costs `k` and `F` are fully loaded: if reviewers,
integration engineers, or incident responders are needed, their cost belongs in the
model.

**Derivation.** Take one eligible unit. Under the intervention, one of three
mutually exclusive things happens with respect to value: a beneficial outcome occurs
with probability `p_g`, contributing `+b`; a harmful outcome occurs with probability
`p_h`, contributing `-h`; or neither occurs, contributing zero. The expected
contribution of one unit is therefore the probability-weighted sum
`p_g*b - p_h*h`, which is the definition of expectation over a finite outcome set,
applied per unit. Subtract the variable cost `k` incurred per unit regardless of
outcome, giving expected net contribution `p_g*b - p_h*h - k`. Summing over `N`
independent units multiplies by `N` by linearity of expectation — linearity requires
no independence assumption, only that the same distribution applies to each unit, so
heterogeneity across slices must be handled by evaluating slices separately. Finally
subtract fixed cost `F`, which does not scale with volume.

```text
FORMULA [MAP-02]: V = N * (p_g*b - p_h*h - k) - F
WHERE: V expected incremental value [currency/window];
       N eligible units [units/window];
       p_g, p_h outcome probabilities [fraction, 0..1], p_g + p_h <= 1;
       b benefit per beneficial outcome [currency/unit];
       h cost per harmful outcome [currency/unit];
       k variable cost per processed unit [currency/unit];
       F fixed cost [currency/window].
CONSTRAINT: increments relative to a measured baseline; probabilities describe the
            same action and window; b excludes value already delivered by the
            baseline; k, F fully loaded with labour and incident cost.
NUMERICAL EXAMPLE: ILLUSTRATIVE N = 12,000/month, p_g = 0.18, b = $9.00,
                   p_h = 0.012, h = $70.00, k = $0.25, F = $4,200/month.
                   per unit: 0.18*9.00 = 1.62 ; 0.012*70.00 = 0.84 ;
                   1.62 - 0.84 - 0.25 = 0.53 currency/unit.
                   V = 12,000*0.53 - 4,200 = 6,360 - 4,200 = $2,160/month.
INTERPRETATION: the proposal is marginally positive under these assumptions. The
                margin is thin enough that a 20% error in p_h alone turns it
                negative, so the decision hinges on harm measurement quality, not
                on the apparent positive sign.
```

**Independent arithmetic check.** Beneficial value `12,000*0.18*$9.00 = $19,440`.
Expected harm `12,000*0.012*$70.00 = $10,080`. Variable cost `12,000*$0.25 = $3,000`.
Fixed cost `$4,200`. Net `19,440 - 10,080 - 3,000 - 4,200 = $2,160`. The two routes
agree, and the units resolve to currency per month because units/month times
currency/unit cancels to currency/month.

**Sensitivity that changes the decision.** Raise `p_h` from 0.012 to 0.015: expected
harm becomes `12,000*0.015*$70 = $12,600`, and `V = 19,440 - 12,600 - 3,000 - 4,200
= -$360/month`. A change of three-tenths of one percentage point flips the sign.
Any decision resting on this estimate must therefore fund a measurement of `p_h`
before scaling, and must state the harm definition that produced `h`.

**Boundary and counterexample.** Linearity hides correlation. If harmful outcomes
cluster — one bad model version produces a thousand wrong commitments in an hour
rather than a steady trickle — the mean over a month can look identical while the
tail destroys the business case. Expected value is a statement about the mean; a
decision that cannot survive a bad week needs a different analysis, addressed in
Section 4.9.

### 4.3 Variant A: review feasibility under a routing policy

**Validity before use.** Every routed unit consumes one review slot; review is not
parallelisable within a unit; the routing fraction `q` is stable over the window.
Queueing behaviour, absence, rework, and escalation each add load beyond this
skeleton, so the inequality is a necessary condition, not a sufficient one.

**Derivation.** Let `q` be the fraction of eligible units routed to human review. The
arrival of review work is `q*N` units per window. If the pool can complete `r` units
per window at a sustainable pace, feasibility requires arrivals not to exceed
capacity: `q*N <= r`. Rearranged, the maximum supportable routing fraction at a given
volume is `q_max = r/N`.

```text
FORMULA [MAP-03]: q*N <= r   equivalently   q_max = r/N
WHERE: q routing fraction [fraction, 0..1];
       N eligible units [units/window];
       r sustainable review capacity [units/window];
       q_max maximum supportable routing fraction [fraction].
CONSTRAINT: one slot per routed unit; steady-state rates; excludes queueing delay,
            rework, absence, context switching, and escalation effort.
NUMERICAL EXAMPLE: ILLUSTRATIVE N = 12,000/month, r = 1,760 reviews/month
                   (11 reviewers * 160 reviews each); q_max = 1,760/12,000 = 0.1467.
                   A policy routing 18% implies 2,160 arrivals/month, which exceeds
                   capacity by 400/month.
INTERPRETATION: the policy is infeasible as staffed, independent of model quality.
                Either the routing threshold rises (fewer routed, more unchecked),
                or capacity rises by 400/month, or volume falls, or the review
                granularity changes from per-unit to per-batch.
```

**Converting capacity into labour.** Capacity is often known as headcount rather than
units. If a review takes `s` minutes, then

```text
FORMULA [MAP-03b]: H = s * q * N / 60
WHERE: H reviewer-hours [h/window]; s service time per review [min/unit];
       q, N as above; 60 converts minutes to hours.
CONSTRAINT: excludes rework and escalation; s must be a measured mean of the
            *routed* population, which is the harder subset by construction.
NUMERICAL EXAMPLE: s = 4.5 min, q = 0.18, N = 12,000/month ->
                   H = 4.5*0.18*12,000/60 = 162 reviewer-hours/month.
INTERPRETATION: 162 hours/month is about one full-time reviewer at 160 productive
                hours, before rework; a second reviewer is needed for absence cover
                and escalation, which makes the true cost roughly two heads, not one.
```

**Boundary and counterexample.** A routing policy that sends only low-confidence
units to review selects the hardest cases, so `s` measured on the full population is
an underestimate. Measure `s` on routed units only. A policy that routes *high*
confidence units instead selects easy cases and gives a flattering `s` while leaving
the dangerous ones unchecked.

### 4.4 Variant B: utilisation and the queue that never drains

**Validity before use.** Arrivals form a stationary process over the window, and the
reviewer pool is pooled — any reviewer can take any unit. Utilisation is the ratio of
offered load to capacity.

**Derivation.** Offered load is `lambda = q*N` units per window; capacity is
`r` units per window; utilisation `rho = lambda / r`. As `rho` approaches 1, waiting
time grows without bound in any queueing model with variability, because the
probability that arrivals momentarily exceed capacity rises toward certainty. The
practical consequence: a plan at `rho = 0.98` is not "98% efficient", it is a plan
with an unbounded backlog under ordinary variability.

```text
FORMULA [MAP-04]: rho = (q*N) / r
WHERE: rho utilisation [fraction, 0..1 for stability]; q, N, r as above.
CONSTRAINT: stationary arrivals; pooled reviewers; rho < 1 for a stable queue,
            and in practice rho <= 0.85 for a service with meaningful variability.
NUMERICAL EXAMPLE: ILLUSTRATIVE q*N = 1,500/month, r = 1,760/month ->
                   rho = 0.852; backlog is stable but waits are material.
                   Same arrivals with r = 1,540 (one reviewer absent) -> rho = 0.974;
                   the queue grows through the window.
INTERPRETATION: the margin exists but is thin. Staffing that exactly matches mean
                load converts ordinary variance into a backlog that must be paid
                down with overtime, which then increases absence and reduces
                capacity further.
```

**Independent check.** `0.852 * 1,760 = 1,499.5`, recovering the arrival rate within
rounding. Units cancel: (units/window)/(units/window) is dimensionless.

**Boundary and counterexample.** Pooling helps only if the work is genuinely
interchangeable. If review requires domain authority that only two of eleven
reviewers hold, the effective capacity is theirs, and utilisation against total
headcount is a fiction that will be discovered during the first escalation.

### 4.5 Variant C: cost per accepted output

**Validity before use.** Acceptance is judged by the same rubric on the same
population, and rework cost per rejected output is either included or explicitly
excluded with a stated reason.

**Derivation.** Raw cost per attempted output is total cost divided by attempted
units. The useful quantity is cost per *accepted* output, because rejected outputs
deliver nothing and consume capacity. If accepted outputs number `y*N` where `y` is
the acceptance fraction, then cost per accepted output is total cost divided by
`y*N`, which exceeds raw cost per attempt by the factor `1/y`.

```text
FORMULA [MAP-05]: C_acc = (k*N + F + w*(1-y)*N) / (y*N)
WHERE: C_acc cost per accepted output [currency/unit];
       k variable cost per attempt [currency/unit];
       F fixed cost [currency/window];
       w rework cost per rejected output [currency/unit];
       y acceptance fraction [fraction, 0<y<=1];
       N attempted units [units/window].
CONSTRAINT: acceptance rubric frozen before measurement; y measured on the target
            population and slice mix; w includes reviewer time, not only compute.
NUMERICAL EXAMPLE: ILLUSTRATIVE k = $0.25, F = $4,200, N = 12,000,
                   y = 0.62, w = $0.90.
                   numerator: 3,000 + 4,200 + 0.90*0.38*12,000
                            = 3,000 + 4,200 + 4,104 = 11,304.
                   denominator: 0.62*12,000 = 7,440 accepted units.
                   C_acc = 11,304 / 7,440 = $1.5194 per accepted output (1.52).
INTERPRETATION: the advertised $0.25 per call is 16.5% of the real cost per usable
                output. Pricing and business cases built on the per-call figure
                understate cost by a factor of about six, and the gap grows as y
                falls.
```

**Independent check.** Divide both sides by `1/y`: `1.5194 * 0.62 = 0.942` per
attempt, which equals `11,304/12,000`. The two computations agree.

**Boundary and counterexample.** If acceptance is measured by the same reviewers who
are measured on throughput, `y` will drift upward as they learn what is easy to
accept. Freeze the rubric and rotate reviewers, or sample-audit accepted outputs.

### 4.6 Variant D: break-even volume between two modes

**Validity before use.** Two options serve the same eligible unit with the same
quality contract, so the comparison isolates cost structure. If quality differs, the
comparison must first convert quality difference into cost, for example through
`C_acc`.

**Derivation.** Write total cost as a function of volume for each option: `T1(N) =
F1 + k1*N` and `T2(N) = F2 + k2*N`. Setting them equal gives `F1 + k1*N = F2 +
k2*N`, so `N*(k1 - k2) = F2 - F1`, and `N* = (F2 - F1)/(k1 - k2)`. The expression is
meaningful only when `k1 != k2`; if the variable costs are equal the option with the
lower fixed cost dominates at every volume.

```text
FORMULA [MAP-06]: N* = (F2 - F1) / (k1 - k2)
WHERE: N* break-even volume [units/window]; F1, F2 fixed cost [currency/window];
       k1, k2 variable cost [currency/unit]; k1 != k2.
CONSTRAINT: linear cost structures over the relevant range; the option with higher
            fixed cost must also have lower variable cost, otherwise no crossing.
NUMERICAL EXAMPLE: ILLUSTRATIVE option 1 managed API: F1 = $0, k1 = $0.42/unit.
                   option 2 self-hosted: F2 = $3,600/month (platform + operator
                   share), k2 = $0.11/unit (compute + maintenance).
                   N* = (3,600 - 0) / (0.42 - 0.11) = 3,600 / 0.31 = 11,612.9 units.
INTERPRETATION: below roughly 11,613 units/month the managed API is cheaper; above
                it the self-hosted option wins, *given* that the operating labour
                estimate of $3,600/month is real and not wishful. At the 12,000-unit
                volume of earlier examples the difference is only about $120/month,
                which is inside the estimation error, so the tie is decided by
                control and staffing arguments rather than by cost.
```

**Independent check at N = 11,613.** Option 1: `0.42*11,613 = $4,877.46`. Option 2:
`3,600 + 0.11*11,613 = 3,600 + 1,277.43 = $4,877.43`. Difference three cents, from
rounding `N*` to an integer.

**Boundary and counterexample.** Fixed cost is rarely as fixed as the model assumes;
platform cost rises in steps when a new availability zone or licence tier is
required, and an operator share is a headcount that arrives in whole people. Plot the
step function, not the smooth line, and re-run at each step boundary.

### 4.7 Variant E: a latency budget that must close

**Validity before use.** The interaction is synchronous and a user waits for the
result. Component latencies are measured at the percentile that matters, not as
means, and retries are budgeted as expected extra calls rather than ignored.

**Derivation.** A request that performs `m` sequential remote calls, each with
p95 latency `t_c`, plus retrieval `t_r`, plus local overhead `t_o`, has p95 total
latency approximately `m*t_c + t_r + t_o`, provided the components are not strongly
tail-correlated. Adding expected retry traffic with retry probability `pi` multiplies
the call term by `1/(1-pi)` in expectation. The budget constraint requires the total
to stay within the interaction deadline `D`.

```text
FORMULA [MAP-07]: t_total = (m * t_c / (1 - pi)) + t_r + t_o <= D
WHERE: t_total p95 response latency [ms]; m sequential model calls [count];
       t_c p95 latency per call [ms]; pi retry probability [fraction, 0<pi<1];
       t_r retrieval latency [ms]; t_o local overhead [ms];
       D interaction deadline [ms].
CONSTRAINT: components not strongly tail-correlated; sequential (not parallel)
            structure; p95 arithmetic is an approximation, not a convolution.
NUMERICAL EXAMPLE: ILLUSTRATIVE m = 2, t_c = 1,400 ms, pi = 0.04, t_r = 180 ms,
                   t_o = 60 ms, D = 3,500 ms.
                   call term = 2*1,400/0.96 = 2,916.7 ms;
                   t_total = 2,916.7 + 180 + 60 = 3,156.7 ms, inside budget by
                   343.3 ms.
INTERPRETATION: the headroom is 9.8% of the deadline. If a third call is added,
                the term becomes 3*1,400/0.96 = 4,375 ms and the design fails the
                budget outright; the architecture, not the model, must change.
```

**Boundary and counterexample.** Summing p95s overstates the total when components
are independent (the chance that all three are simultaneously at p95 is small), and
understates it when they share a cause — for example a single network path whose
congestion slows the model call, retrieval, and the logging write together. Use
measured end-to-end p95 wherever possible and treat the sum as an upper bound for
planning.

### 4.8 Variant F: adoption and the offer denominator

**Validity before use.** Meaningful use is defined before observation; the offered
population is the set that had a usable opportunity to use the intervention; outages
are recorded rather than silently excluded.

**Derivation.** Adoption is a part-over-whole ratio over the offered population, the
same structure as coverage but with a different numerator definition. Meaningful use
is a count of units or users meeting a declared rule (for example, a cited retrieval
result incorporated into a weekly deliverable), not a count of logins.

```text
FORMULA [MAP-08]: a = n_meaningful_use / n_offered
WHERE: a adoption [fraction, 0..1]; n_meaningful_use [users or units/window]
       meeting the declared meaningful-use rule; n_offered [same unit/window]
       with a usable opportunity.
CONSTRAINT: opportunity defined before uptake is observed; single-sign-in events
            and forced launches do not count as meaningful use.
NUMERICAL EXAMPLE: ILLUSTRATIVE n_meaningful_use = 264 analysts, n_offered = 480
                   analysts with access and relevant cases; a = 264/480 = 0.55.
INTERPRETATION: 55% adoption says nothing about benefit; the 216 non-users may
                have had no relevant cases at all, in which case the denominator
                is wrong and the true adoption among the eligible subset is higher.
```

### 4.9 Limits and counterexamples for the whole family

**The coverage paradox.** Maximising `c` and maximising correct outcomes are
different objectives whenever unserved units are handled differently from served
ones. Formally, correct outcomes `= c*N*y + (1-c)*N*y_fb`, where `y` is the
acceptance fraction on served units and `y_fb` is the fallback's success rate. If the
fallback is good (`y_fb` high relative to `y`), raising coverage can *reduce* total
correct outcomes by displacing better fallback handling. Always carry `y_fb`.

**The irreversibility asymmetry.** Expected value treats a harmful outcome as the
same kind of quantity as a beneficial one, scaled by `h`. For irreversible harms this
is wrong: a firm can absorb many small losses and only one existential one, so the
decision rule must bound the *worst case* separately, not merely price it. A
practical form is a separate gate: the maximum single-event loss must be below a
stated ceiling and the probability of that event must be estimated independently,
usually by a different person from the one who built the estimate of `p_h`.

**The small-denominator hazard.** A slice with 40 units and 3 errors yields an error
rate of 7.5% with a confidence interval so wide that it is compatible with both 1%
and 20%. Decisions to expand or kill a deployment on such a slice are decisions made
from noise. Either aggregate slices to a minimum count or report intervals and let
the decision wait.

**The frozen-baseline fallacy.** A baseline measured once, before the intervention,
continues to be used as the comparator long after the surrounding workflow changed.
If the fallback process improved for unrelated reasons, the intervention's measured
advantage shrinks without any change to the intervention. Re-measure the baseline on
the same schedule as the candidate, or state that the comparison is historical.

**What these formulas cannot do.** They cannot establish causality, cannot validate
`p_g` or `p_h`, cannot substitute for a controlled comparison when the question is
"did the intervention cause the improvement", and cannot certify a vendor's claim.
They convert declared assumptions into consequences quickly, which is their entire
job: they tell you which assumption the decision is sensitive to, so that measurement
effort goes where it changes an answer.

**Takeaway:** Coverage, expected value, review feasibility, cost per accepted output,
break-even volume, and latency budget are six cheap calculations that eliminate most
bad proposals before any evaluation spend; each is only as honest as its frozen
population and its fully loaded costs.

## 5. Worked Numerical Example

### 5.1 The setting

**Evidence:** ILLUSTRATIVE throughout. The arithmetic is checked twice by independent
routes; the workload is constructed for teaching and describes no real organisation.

A shared-services group receives **24,000 supplier messages per month**. Historically
a team of six agents reads each message, classifies it into one of nine request types,
extracts four fields into an enterprise system, and either answers directly or
escalates to a buyer. Average handling time is 7.2 minutes per message; that gives
`24,000 * 7.2 / 60 = 2,880` agent-hours per month, or eighteen full-time equivalents
at 160 productive hours each. The group is asked to evaluate three AI options against
the current baseline and to recommend one, or to recommend retaining the baseline.

The eligible unit is one inbound supplier message, deduplicated on message ID. The
window is one calendar month. The decision is the classification label plus the four
extracted fields, which determine whether a reply is drafted automatically or the
message is escalated. Two of the four fields feed a payment instruction downstream,
which makes extraction errors consequential even though the AI itself never pays.

### 5.2 The four options

**Option A — retrieval-assisted search.** Agents query a cited index of past messages,
contracts, and supplier records. The system returns ranked evidence with links; the
agent reads and acts. No generated text, no extraction, no state change.

**Option B — assisted generation with mandatory citation.** The system proposes a
classification, the four extracted fields, and a draft reply. An agent must accept or
edit every field before it flows downstream; the reply is never sent without an
explicit send action by the agent.

**Option C — bounded automation with sampling.** The system classifies, extracts, and
drafts, and commits the result automatically when its confidence exceeds a threshold
and the extracted amount is under $500; a 10% sample of committed cases is audited
after the fact. Amounts above $500 escalate to an agent.

**Option D — retain the baseline.** The six-agent team continues as-is, with modest
tooling improvements to search only.

### 5.3 Declared parameters

| Quantity | A: retrieval | B: assisted | C: bounded auto | D: baseline |
|---|---:|---:|---:|---:|
| Eligible units/month `N` | 24,000 | 24,000 | 24,000 | 24,000 |
| Served units/month `n_served` | 21,600 | 19,200 | 16,800 | 24,000 |
| Coverage `c` | 0.900 | 0.800 | 0.700 | 1.000 |
| Acceptance fraction on served `y` | — (evidence view) | 0.66 | 0.71 | — |
| Variable cost/attempt `k` | $0.02 | $0.31 | $0.44 | $0.00 |
| Fixed cost/month `F` | $1,100 | $4,900 | $9,600 | $0 |
| Routed to human `q` | 1.00 | 1.00 | 0.30 | 1.00 |
| Mean review minutes `s` | 5.1 | 6.0 | 7.5 | 7.2 |
| Reversible before commit? | yes | yes | partially | yes |

The numbers deserve a note, because they are where most proposals go wrong. Coverage
for C is the lowest of the three because a confidence threshold removes hard cases;
those cases are not failures of the model, they are routed to humans, and the human
cost is already counted in `q = 0.30` for the escalated share plus the audited share.
Review time `s` differs across options because the routed population differs: B
routes everything, so its mean is close to the population mean; C routes only the
hard 30%, so its per-case time is longer, which is exactly the underestimate teams
make when they multiply the population mean by the routed volume.

### 5.4 Step 1 — review labour for each option

Using [MAP-03b] with `H = s*q*N/60`:

```text
Option A: H = 5.1 * 1.00 * 21,600 / 60 = 1,836.0 reviewer-hours/month.
Option B: H = 6.0 * 1.00 * 19,200 / 60 = 1,920.0 reviewer-hours/month.
Option C: routed = 0.30*24,000 = 7,200 units; but the audited 10% of the
          16,800 automated units adds 1,680 units of audit work at a lighter
          2.0 min each. So two loads:
            escalated: 7.5 * 7,200 / 60 = 900.0 h
            audit:     2.0 * 1,680 / 60 =  56.0 h
            total H = 956.0 reviewer-hours/month.
Option D: H = 7.2 * 24,000 / 60 = 2,880.0 reviewer-hours/month.
```

Converted to full-time equivalents at 160 productive hours per month:

```text
A: 1,836.0/160 = 11.475 -> 12 FTE (round up; staffing is integral)
B: 1,920.0/160 = 12.000 -> 12 FTE, with zero slack
C:   956.0/160 =  5.975 ->  6 FTE
D: 2,880.0/160 = 18.000 -> 18 FTE
```

Option B at exactly 12.000 FTE has utilisation `rho = 1.000` against its staffing,
which Section 4.4 says is infeasible under variability: the first absence creates a
backlog. Practical staffing for B is 14 FTE (`rho = 0.857`). Option C at 5.975 has the
same problem in miniature — 6 FTE gives `rho = 0.996` — so 7 FTE (`rho = 0.854`) is
the honest number. This single correction, applied to both options, changes the cost
comparison materially and is the kind of thing a proposal built on "we reviewed the
math" usually omits.

### 5.5 Step 2 — fully loaded monthly cost

Assume a loaded reviewer cost of **$6,400/month** including benefits, workspace, and
supervision, applied identically to every option so the comparison is fair.

```text
A: model/platform 1,100 + variable 0.02*21,600 = 432 -> 1,532
   labour 12 FTE * 6,400 = 76,800
   total = 78,332

B: fixed 4,900 + variable 0.31*19,200 = 5,952 -> 10,852
   labour 14 FTE * 6,400 = 89,600
   total = 100,452

C: fixed 9,600 + variable 0.44*16,800 = 7,392 -> 16,992
   labour 7 FTE * 6,400 = 44,800
   total = 61,792

D: variable 0 -> labour 18 FTE * 6,400 = 115,200
   total = 115,200
```

Check one line by an independent route: B variable cost `19,200 * 0.31`. Compute
`19,200*0.3 = 5,760` and `19,200*0.01 = 192`; sum `5,952`. Correct. Check C:
`16,800*0.44 = 16,800*(0.4 + 0.04) = 6,720 + 672 = 7,392`. Correct.

The ranking is C < A < B < D on total monthly cost. That ranking is not the decision,
and the next two steps show why.

### 5.6 Step 3 — cost per accepted output

Use [MAP-05] with rework cost `w = $1.40` per rejected output for A and B (agent
re-does the work) and `w = $2.60` for C, because a rejected automated case has
already been committed and must be reversed, which involves the payables team.

```text
A: accepted units = 21,600 (evidence view; every served unit is acted on by an
   agent, so acceptance is not meaningful). Cost per served message =
   78,332 / 21,600 = $3.627.
B: numerator = 5,952 + 4,900 + 1.40*(1-0.66)*19,200 = 5,952 + 4,900 + 9,139.20
             = 19,991.20 ; accepted = 0.66*19,200 = 12,672 ;
   C_acc = 19,991.20/12,672 = $1.5776.
C: numerator = 16,992 + 2.60*(1-0.71)*16,800 = 16,992 + 2.60*4,872 = 16,992 + 12,667.20
             = 29,659.20 ; accepted = 0.71*16,800 = 11,928 ;
   C_acc = 29,659.20/11,928 = $2.4865.
D: 115,200 / 24,000 = $4.800 per message.
```

Check B: `9,139.20 / 6,528 = 1.40`; `1.40*6,528 = 9,139.20`. The rejected count is
`0.34*19,200 = 6,528`. Correct. Check C: `0.29*16,800 = 4,872`; `4,872*2.60`:
`4,872*2 = 9,744`, `4,872*0.6 = 2,923.20`, total `12,667.20`. Correct.

Now the ranking changes: **B ($1.578) < C ($2.487) < A ($3.627) < D ($4.800)** per
accepted output. Option C's low total cost is partly an artefact of low coverage: it
produces 11,928 accepted outputs against B's 12,672, so its per-output cost is higher
even though its monthly spend is lower. Whether that matters depends on whether the
group needs the volume — and whether the 4,872 rejected automated cases caused
downstream harm beyond the $2.60 rework.

### 5.7 Step 4 — the harm the cost table omits

Option C commits extraction before review. Two of the four fields feed payment. A
wrong amount under $500 is reversed at $2.60 of labour if caught in the audit sample,
but the audit is a 10% sample, so a wrong amount is caught with probability roughly
0.10 within the audit window, and otherwise persists until the supplier disputes it.
Assume, ILLUSTRATIVE, that an undisputed wrong payment costs `h = $180` to unwind and
reconcile, and that among the 4,872 automated-and-rejected-in-rubric cases, roughly
`p_h = 0.06` become an actual wrong payment rather than a cosmetic extraction slip.

```text
Expected wrong-payment harm for C = 4,872 * 0.06 * 180 = $52,617.60/month.
```

Check: `4,872*0.06 = 292.32` events; `292.32*180 = 52,617.6`. Correct.

Add this to C's total: `61,792 + 52,617.60 = $114,409.60/month`, which is **worse
than the baseline** ($115,200) only by a hair, and worse than A and B by a wide
margin. The apparent cost leader is destroyed by one harm term that the cost table
never contained. Note also that `p_h = 0.06` is an assumed parameter, and the
decision's sensitivity to it is enormous: at `p_h = 0.02` the harm is `$17,539.20`
and C totals `$79,331.20`, still worse than A's `$78,332` but close; at `p_h = 0.01`
C totals `$70,220.80` and wins on cost.

**This is the crux.** The decision between A and C is decided entirely by a parameter
nobody has measured. The correct next action is not to pick the option whose assumed
parameter flatters it, but to measure `p_h` on a shadow run before committing.

### 5.8 Step 5 — the sensitivity that actually settles it

Solve for the harm probability at which C equals A:

```text
61,792 + 4,872 * p_h * 180 = 78,332
4,872*180 = 876,960 ; 78,332 - 61,792 = 16,540
p_h* = 16,540 / 876,960 = 0.01886
```

So C is cheaper than A only if fewer than about **1.89%** of the automated-rubric
failures become wrong payments. Against A, at `p_h = 0.06`, C is `$36,077.60` worse
per month. That is the one number to carry into the decision meeting.

Compare C against B: `61,792 + 876,960*p_h = 100,452` gives `p_h* = 38,660/876,960 =
0.04408`, so C beats B only below **4.41%**. Since C also delivers less coverage, and
B's failures are caught before commit, the burden of proof is on C.

### 5.9 Step 6 — latency and the interaction that must close

For the agent-facing options, the interaction deadline is 4,000 ms. Using [MAP-07]
with `m = 1` model call at p95 `t_c = 1,650` ms, retry probability `pi = 0.05`,
retrieval `t_r = 210` ms, overhead `t_o = 75` ms:

```text
A: no model call; t_total = t_r + t_o = 210 + 75 = 285 ms.
B: t_total = 1*1,650/0.95 + 210 + 75 = 1,736.8 + 285 = 2,021.8 ms.
C: same 2,021.8 ms on the synchronous path; the automated path completes
   asynchronously within a 30 s service-level objective, so the 4,000 ms
   deadline does not bind for it, but the 30 s does.
```

Check B: `1,650/0.95 = 1,736.84`. All options close the 4,000 ms budget; the latency
criterion therefore does not discriminate, and saying so explicitly prevents a team
from using latency as a tie-breaker it has not earned.

### 5.10 Step 7 — the recommendation and its falsifier

The arithmetic supports **Option A as the deployment-ready choice and Option B as
the candidate under evaluation**, with C deferred pending a shadow measurement of
`p_h` and a reversal drill. A is cheapest per accepted output among the safe options,
its control boundary is narrow (ranked evidence, no generated text, no state change),
and its review burden is the lowest of the full-coverage options at 12 FTE.

The falsifier, stated so it can be checked: *if a shadow run shows that B's
acceptance fraction on the production slice mix is at or above 0.74, and its review
time on routed cases stays at or below 6.0 minutes, then B's total monthly cost falls
below A's and B becomes the recommendation.* Verify with two consecutive months of
shadow data, the second month being the decision month.

Compute that boundary to make it concrete. B's cost with `y = 0.74` and `s = 6.0`:
labour is unchanged at 14 FTE because `s` and `q` did not change, so `$89,600`;
variable `$5,952`; fixed `$4,900`; rework `1.40*0.26*19,200 = $6,988.80`; total
`$107,440.80` — which is *not* below A's `$78,332`. The stated falsifier is therefore
wrong as written, and the arithmetic exposes it. The correct statement of the
condition under which B beats A on total cost requires reducing labour, which means
raising coverage or acceptance enough that fewer units need human handling — a
different mechanism entirely.

This is worth pausing on, because it is the most useful thing in the worked example:
**the recommendation was right but the accompanying falsifier was incoherent, and
only the arithmetic revealed it.** A proposal whose falsification condition cannot be
satisfied is not falsifiable; it is a slogan with numbers attached.

### 5.11 Re-derived falsifier

B beats A on total cost only if B's labour falls. Let B's review load be `q*N` units
at `s` minutes; the FTE count is `s*q*N/(60*160)`. For B to use fewer than 12 FTE at
`N = 19,200` and `s = 6.0`, we need `q < 12*60*160/(6.0*19,200) = 115,200/115,200 =
1.0` — which is exactly 1.0, meaning no reduction is possible while every served unit
is reviewed. B cannot beat A on labour under the assisted model as defined. It could
only do so if a fraction of units were auto-accepted, which is Option C's design.

So the honest recommendation is: **A is the deployment choice; B is not justified by
cost under its declared design and is justified only if its quality improves
downstream outcomes in ways A cannot, which is an unmeasured claim.** C remains a
candidate whose cost advantage depends on an unmeasured harm parameter, with a
break-even at `p_h* = 1.89%` against A.

```text
RECOMMENDATION SUMMARY (ILLUSTRATIVE)
  Deploy:      A, retrieval with citations; 12 FTE; $78,332/month.
  Reject now:  C, bounded automation; breaks even only if p_h < 1.89%
               against A; requires shadow measurement and reversal drill.
  Not justified: B on cost grounds; 14 FTE and $100,452/month for lower
               coverage than A. Reconsider only with measured downstream
               benefit that A cannot supply.
  Baseline:    D at $115,200/month is the comparator; the proposal must beat
               it, and A beats it by $36,868/month (32.0%).
```

Check the last figure: `115,200 - 78,332 = 36,868`; `36,868/115,200 = 0.32003`, so
32.0%. Correct.

### 5.12 What the example teaches beyond its numbers

Three transferable habits emerge. First, **coverage and acceptance must be multiplied,
never assumed**; a system that serves 70% at 71% acceptance delivers about half the
units its demo suggested. Second, **review time must be measured on the routed
population**, because routing selects hard cases and the population mean is an
optimistic substitute. Third, **the harm term is the term nobody measures and the term
that decides the outcome**; in this example it moved option C from first to last.

A fourth, subtler habit: state the falsifier and then *compute whether it is
satisfiable*. An unsatisfiable falsifier reads like rigour and functions as its
opposite.

**Takeaway:** Cost tables rank options by the quantities that are easy to count;
decisions are made by the quantities that are not, so the worked example must always
carry coverage, routed-case service time, and expected harm before it ranks anything.

## 6. Visual / Geometric Intuition

### 6.1 The proposal plane

Draw two axes. The horizontal axis is **control span**: how much data crosses a
boundary and how much authority the system holds, from narrow (reads a document you
already may read, emits text) to broad (reads across systems, holds credentials,
changes records). The vertical axis is **consequence per error**: how much a single
wrong output costs, from a mildly annoying draft to an irreversible financial or
safety event.

```text
 consequence per error
        ^
  high  |  . retrieval on         . autonomous payment
        |    regulated records      authorization
        |                              x  (needs caps + pre-validation)
        |  . classification of     .
        |    safety-relevant         . agent with write access
        |    documents                 x
        |
 medium |  . search over           . assisted drafting with
        |    internal policy         . send-on-behalf
        |      x                   x
        |
  low   |  . search over        . autocomplete
        |    public docs   x       x
        +--------------------------------------------------> control span
          narrow                                  broad
```

Every point's **proof burden** is roughly the product of its coordinates: the further
up and to the right, the more evidence before deployment. The diagonal band from
bottom-left to top-right is where most real systems live, because teams naturally
pair narrow authority with low consequence. The dangerous quadrant is top-right, and
it is populated almost entirely by systems that started in the middle and were
granted more authority without the coordinates being redrawn.

### 6.2 The narrowing funnel

```text
 all capability classes that could touch the decision     (wide)
        |
        | control gate: data route, credentials, reversibility
        v
 permitted classes                                        (~half)
        |
        | capacity gate: review load <= capacity, rho <= 0.85
        v
 operable classes                                          (~two thirds)
        |
        | economics gate: V > 0 with fully loaded costs
        v
 affordable classes                                       (~half)
        |
        | evidence gate: acceptance protocol satisfiable
        v
 deployable classes                                        (one, or none)
```

The funnel is a diagnostic, not a forecast: if it produces zero at the bottom, the
answer is the baseline, and the useful output is the record of which gate emptied it.
If it produces five, a gate was applied too loosely — usually the evidence gate,
which teams skip because it requires writing a protocol before seeing results.

### 6.3 The autonomy ladder as a staircase with landing zones

```text
   irreversible autonomy      |  requires: pre-action validation,
                              |  caps, independent harm estimate,
                              |  incident drill, named accountable owner
   ---------------------------+
   act with rollback          |  requires: reversal within stated window,
                              |  reversal tested, not merely designed
   ---------------------------+
   act within limits          |  requires: amount/scope caps enforced in code,
                              |  escalation path exercised
   ---------------------------+
   act with sampling          |  requires: audit protocol, sample size,
                              |  authority to reverse found errors
   ---------------------------+
   propose with pre-fill      |  requires: reviewer can reject at the required
                              |  rate; UI does not coerce acceptance
   ---------------------------+
   propose only               |  requires: output clearly marked as candidate
   ===========================+
   baseline                   |  reference point for all comparisons
```

Each landing zone is a set of obligations. Climbing a step without meeting them is
the single most common way a well-validated prototype becomes an incident. Note that
the ladder is about *authority*, not about model capability: a system can be
technically capable of the fourth step and still be correctly deployed at the first.

### 6.4 The cost crossover picture

```text
 total cost
   |                                     /
   |                            option 2 (high fixed, low variable)
   |                        . /
   |                    .    /
   |               .        /
   |          .            /          option 1 (low fixed, high variable)
   |     .                /      .
   |  .                  /   .
   |.___________________/_._____________________  volume
                 N*     ^
                        break-even; below it, option 1 wins
```

The picture has a wrong version that appears in most decks: it draws both lines from
the origin with a single slope. The lines must start at their fixed costs, which means
option 2 starts above zero. When drawn correctly, the region where option 2 is cheaper
is a triangle whose left vertex is `N*`; a proposal that claims savings from
self-hosting at volumes below `N*` is reading the wrong side of its own picture.

The stepped reality is worth drawing too: fixed cost is a staircase, not a line, so
the crossover moves each time a licence tier or an availability zone is added.

### 6.5 The two-population picture for coverage

```text
 all eligible units  N
 +-----------------------------------------------------------+
 |                                                           |
 |   served  n_served            |   unserved                |
 |   quality y measured here     |   quality not measured    |
 |                               |   handled by fallback     |
 |                               |   (cost and error often    |
 |                               |    unknown)                |
 +-----------------------------------------------------------+
                               ^
                     coverage boundary c = n_served/N
```

Most evaluation reports draw a box around the left region and call it the system. The
right region is the system too, in the sense that it determines total outcomes. Any
comparison that scores only the left box will prefer systems that serve less, because
they can concentrate on easy cases.

### 6.6 The review-load picture with the safety margin

```text
 capacity r  |====================================================|
             |  usable band: rho <= 0.85                          |
 arrivals    |==========================================|          |
 (q*N)       |                                          | dead zone |
             |                                          | rho -> 1  |
             +---------------------------------------------------> 
                0.70      0.80   0.85        0.95   1.00
```

The band from `rho = 0.85` to `rho = 1.00` is not a margin; it is a zone where the
queue is stable on average but grows during any period of above-average arrivals or
absence. Teams that staff to `rho = 0.98` and then measure their backlog over a full
quarter see a number that looks like a staffing failure. It is not — it is the
predicted behaviour of the design.

### 6.7 Geometry of the sensitivity slope

In Section 5.8 the break-even harm probability was `p_h* = 16,540 / 876,960`. Geometrically,
`16,540` is the vertical gap in dollars between the two cost lines at the current
volume, and `876,960` is the slope of the harm term in dollars per unit of
probability. A break-even probability is therefore a *slope ratio*: vertical gap
divided by exposure. This is why the number is so small when exposure is large. A
system that processes 4,872 units at `$180` of harm each has `$876,960` of exposure
per unit of probability, so a gap of only `$16,540` buys a probability window of under
two percent. Enlarging exposure shrinks the safe region, which is the geometric
statement of the intuition that scale makes harm parameters more rather than less
important.

**Takeaway:** Draw the proposal's coordinates, the gate funnel, and the two-population
box before computing anything; the pictures catch the errors — wrong cost lines,
hidden fallback populations, staffing to `rho = 1` — that arithmetic performed on the
wrong structure will faithfully preserve.

## 7. Comparison / Selection Table

### 7.1 Capability-class selection

| Decision the workflow makes | First choice | Why it is first | Disqualify when | Fallback |
|---|---|---|---|---|
| Look up a fact that exists in an authorised corpus | retrieval with citations and an abstain option | preserves source inspection; failures are visible as bad ranking rather than invented text | corpus is stale, unauthorised, or the needed fact is absent | structured search with field filters |
| Assign a fixed label set to an item | supervised classifier or constrained extraction | testable contract, calibratable, cheap at volume | labels unavailable, unstable, or label delay exceeds decision cadence | rules with an exception queue |
| Estimate a number before the outcome exists | statistical or ML regression on point-in-time features | explicit target, residual diagnostics, intervals | target undefined or features only available after the outcome | human estimate with recorded reasoning |
| Produce a draft a person will finish | generative assistance with review | compresses first-draft effort; errors are caught at review | reviewer cannot detect bad output at the required rate | template plus retrieval |
| Choose among feasible assignments under limits | optimisation with explicit objective and constraints | makes trade-offs inspectable and infeasible cases detectable | constraints are unmodelled or the objective owner is unassigned | greedy heuristic with manual override |
| Change state in a downstream system | bounded automation | removes handoffs if reversible and capped | identity, idempotency, caps, or rollback missing | propose-only integration |
| Execute a multi-step plan across tools | none of the above by default | each tool needs its own contract and authority | authority cannot be bounded or steps cannot be logged | single-step automation per tool |
| Policy is stable and exceptions are enumerable | rules and validators | deterministic, auditable, no training set | exceptions dominate the volume | classifier for the exception residue |

Two rules govern the table. First, **the first choice is the smallest thing that
answers the decision**, not the most capable thing the team can build. Second, **a
disqualifier removes an option outright**; it is not a penalty score. Selection
tables that allow a strong score to overcome a hard constraint produce the failures
catalogued in Section 8.

### 7.2 Deployment-mode selection

| Condition | Mode | What it buys | Disqualify when | What must be measured before commit |
|---|---|---|---|---|
| Data may not leave a controlled environment | local or in-tenancy runtime, pinned versions | data route stays inside the boundary; versions fixed | the operating team cannot patch, monitor, or recover it | patch cadence, incident ownership, capacity headroom |
| Boundary is acceptable and speed to integration dominates | managed inference API with contractual data terms | no runtime operations; elastic capacity | subprocessor or retention terms conflict with the data agreement | actual retention configuration, rate limits, version policy |
| The value is coordinating state and approvals across systems | workflow platform with explicit approval states | state ownership and audit come with the product | authority propagates further than intended | authority propagation, duplicate handling, tool failure behaviour |
| Connectivity, latency, or privacy forces execution at the interaction point | embedded or on-device model | local responsiveness; data stays on device | device fleet heterogeneity cannot be controlled | device distribution, thermal and power limits, update path |
| Decisions tolerate hours of delay | batch with rerun semantics | throughput and cost efficiency | freshness requirement is under an hour | idempotent reruns, partial failure recovery, staleness signalling |
| A team wants control over runtime without a platform function | self-hosted service | version and integration control | no operator is funded for 24/7 response | deployment reproducibility, rollback of artefacts, on-call rota |

The middle column is deliberately phrased as "what it buys" rather than "advantages",
because each mode buys something specific and costs something specific. A comparison
organised as a pros-and-cons list invites the reader to trade a real constraint
against a vague preference. A comparison organised by condition and disqualifier
forces the constraint to be named.

### 7.3 Autonomy-grade selection

| Evidence held | Highest defensible grade | Not yet defensible | Next evidence required |
|---|---|---|---|
| Offline evaluation on a frozen protocol, no field run | propose only | propose with pre-fill | shadow run on production-like inputs with a discrepancy log |
| Shadow run with discrepancy log, reviewers agree with outputs above the target rate | propose with pre-fill | act with sampling | measured reviewer rejection rate under real time pressure |
| Reviewer rejection rate measured, audit protocol written, reversal authority assigned | act with sampling | act within limits | caps enforced in code and an escalation path exercised end-to-end |
| Caps enforced, escalation exercised, single-event loss bounded below the ceiling | act within limits | act with rollback | reversal actually performed on real cases, timed and recorded |
| Reversal performed and timed within the stated window under load | act with rollback | irreversible autonomy | independent estimate of the worst-case single event, accepted by a second owner |
| Independent worst-case estimate accepted, drills passed, harm ceiling signed | irreversible autonomy | — | continuing monitoring with a defined re-review trigger |

Read this table downward as an evidence ladder and upward as a checklist. The critical
discipline is that the *grade claimed must not exceed the evidence row*, and the
evidence rows are ordered so that each is genuinely harder to obtain than the last.
Teams routinely claim the third row's grade on the first row's evidence and describe
the gap as "we will monitor it in production", which inverts the ladder.

### 7.4 Build, buy, and the third option

| Situation | Prefer | Reason | Revisit trigger |
|---|---|---|---|
| Capability is generic, low differentiation, fast-moving | buy or managed service | internal effort would chase a moving target for no strategic gain | price change above the internal-run cost, or a control requirement appears |
| Capability is generic but the data route forbids external processing | buy the weights or licence the model, run in boundary | keeps control without inventing the capability | boundary rule changes, or a compliant managed offer appears |
| Capability is the product differentiator | build, with a purchased base model | the adaptation and evaluation are the asset | base model cost or licence terms change materially |
| Volume is below the break-even `N*` of Section 4.6 | buy | fixed cost of self-operation dominates | volume grows past `N*` for two consecutive quarters |
| The workflow, not the model, is the hard part | buy the workflow platform, not the model | integration and state handling are where the effort is | platform authority model proves insufficient |
| The requirement is a hard constraint nobody can meet | do not buy; retain baseline | a constraint failure is not a procurement problem | the constraint is revised with a recorded decision |

The third option that these tables omit is **retain the baseline**. It is a positive
choice with its own cost profile, own improvement path, and own risk — and it is
frequently the highest-value option because the proposal's benefits are estimated and
the baseline's costs are observed.

### 7.5 Choosing an acceptance protocol

| Protocol | Use when | Cost and effort | What it cannot tell you |
|---|---|---|---|
| Frozen labelled test set, single rater per case | labels are cheap and the task has objective ground truth | low; risk of rater idiosyncrasy | how users behave under time pressure |
| Frozen test set with two raters and adjudication | judgement is involved and disagreement is plausible | medium; measures inter-rater agreement | field outcome, only output quality |
| Shadow comparison on live traffic | the distribution differs from any test set | medium; requires logging and a discrepancy process | benefit until the action changes |
| A/B test on the workflow action | causality of the workflow change is the question | high; needs power, ethics, and stable assignment | which component caused the effect |
| Reversal drill | automation is proposed at any grade above sampling | low cost, high information | accuracy, only recoverability |
| Adversarial probe set | the threat is deliberate manipulation | medium; needs a threat model | ordinary-case performance |

Pick the protocol that answers the question actually being asked. "Is the output
good?" needs a labelled set. "Does the workflow improve?" needs an A/B test.
"Can we recover?" needs a drill and nothing else will substitute.

### 7.6 A decision worksheet

Fill this before comparing any product. Each row must carry a value or the word
`unknown`.

| Field | Value | Owner | If unknown |
|---|---|---|---|
| Eligible unit and window | | product | stop; define it |
| Decision and action taken from output | | product | stop; define it |
| Baseline performance on the same units | | analytics | measure before proceeding |
| Data fields consumed, and their permitted uses | | data owner | gate closed |
| Environments the data may traverse | | security/privacy | gate closed |
| Authority held by the system, in operations terms | | product/ops | assume propose-only |
| Reversibility and reversal time | | ops/engineering | assume irreversible |
| Review fraction and routed-case service time | | ops | measure on routed cases |
| Fixed and variable cost, fully loaded | | finance/platform | assume worst case |
| Harm definition, `h`, and the estimate source for `p_h` | | domain owner | gate closed for automation |
| Acceptance protocol and threshold | | analytics/domain | no comparison permitted |
| Named operator and escalation path | | ops | gate closed |

"Gate closed" means the proposal cannot advance to comparison, not that it is
rejected. The distinction matters administratively: a closed gate has an owner and a
question, so it can be reopened by producing the missing value.

### 7.7 Comparing vendors without being misled

| Reported claim | What it usually measures | What to ask for instead |
|---|---|---|
| "94% accuracy" | a single aggregate on an unknown class mix | per-class results, the confusion matrix, and the abstention policy |
| "Beats human experts on benchmark X" | performance on a public test set | results on your frozen protocol with your population |
| "Handles 40 languages" | some capability in each language | per-language results, especially on the tail languages you need |
| "SOC 2 certified" | controls over a defined system boundary | the boundary description and whether your use is inside it |
| "Processes 1M tokens" | a context window limit | cost and quality at your real document length, including the tail |
| "No training on your data" | a contractual commitment | the retention configuration that implements it, and the logs that show it |
| "99.9% uptime" | availability of an endpoint | latency percentiles and behaviour under rate limiting |
| "Enterprise-grade security" | nothing specific | named controls, test reports, and the incident notification clause |

The pattern is consistent: for every claim, ask for the population, the protocol, the
denominator, and the configuration. A vendor that cannot supply any of the four is
answering a different question than the one you need.

### 7.8 Selection under conflicting constraints

When two constraints point in opposite directions — a data boundary that forbids
external processing and a volume that makes internal operation uneconomic — there are
three legitimate resolutions and one illegitimate one.

Legitimate: **narrow the scope** so that a compliant option becomes affordable (fewer
fields, fewer languages, a smaller population); **change the boundary** through a
recorded decision by the owner who holds it, with the residual risk accepted in
writing; or **retain the baseline** and invest in the manual process.

Illegitimate: treat the constraint as a weighted penalty and let a strong capability
score outweigh it. Weighted scoring across a hard constraint is how organisations end
up with deployments that were "approved" and are nevertheless non-compliant on the
day of audit.

**Takeaway:** A selection table earns its place only if every row carries a
disqualifier, and if the baseline appears as a real option rather than a rhetorical
foil.

## 8. Failure Modes & Edge Cases

### 8.1 How to read these entries

Each entry has a stable ID, the observable symptom, at least two competing causes
that produce that symptom, a check that discriminates between them, a bounded fix,
and a verification that the fix worked. The provenance of the pattern is stated; most
are DERIVED from the arithmetic in Section 4 and describe how that arithmetic is
bypassed in practice, not observed incidents at any named organisation.

### 8.2 MAP-F01 — The proposal has no falsifiable claim

**Symptom.** The requirement reads "we need AI for the supplier workflow". Success is
described with verbs like modernise, empower, or accelerate. No baseline number
exists, and the acceptance discussion is deferred to after a pilot.

**Competing causes.** (a) Solution-first framing: a capability was chosen before the
decision was described, often because a demo was compelling. (b) A political purpose:
the project exists to signal investment, and a falsifiable claim would risk the
signal. (c) Genuine early exploration, where the decision really is unclear and the
correct next step is discovery, not procurement.

**Discriminating check.** Ask three questions and record the answers verbatim: which
decision changes, which action follows the output, and what number must move. If the
first two can be answered but not the third, it is (a). If none can be answered and
the sponsor resists writing them, it is (b). If the sponsor agrees to a two-week
discovery with named interviews, it is (c).

**Bounded fix.** For (a) and (c): pause the comparison and write the charter fields
from Section 7.6. For (b): do not attempt to fix it inside the project; escalate the
missing decision to the accountable owner and record that the project cannot proceed
without it.

**Verification.** A signed one-page charter exists containing the eligible unit, the
decision, the action, the baseline number, the harm definition, and the owner's name.
Re-check that the charter is referenced in the next review, not merely filed.

**Residual risk.** A charter can be written to satisfy the process while the real
decision stays elsewhere. The tell is that the charter's metric never appears in any
subsequent status report.

### 8.3 MAP-F02 — Coverage collapse disguised as quality improvement

**Symptom.** Reported accuracy rises from 81% to 94% after a change, while the
workflow's total correct outcomes fall and the complaint volume rises.

**Competing causes.** (a) A stricter abstention or safety filter removed hard cases
from the served population. (b) The eval set was regenerated with an easier sampling
rule. (c) A genuine model improvement that also happens to shift the served
population, so both effects are real but only one is reported.

**Discriminating check.** Recompute coverage `c` on both sides of the change using the
same frozen eligibility rule. If `c` fell, compute `c*N*y + (1-c)*N*y_fb` for both
periods using the fallback's measured success rate. If the total is lower, the
improvement is an artefact of the population.

**Bounded fix.** Report coverage and quality as an inseparable pair in every
evaluation artefact. Where a filter causes the shift, evaluate the filter itself as a
separate decision with its own cost: it trades served volume for served accuracy, and
that trade needs an owner.

**Verification.** Two consecutive reports show the served and unserved populations
with quality measured on the served units and the fallback outcome on the rest, and
the totals reconcile to `N`.

### 8.4 MAP-F03 — Review capacity was computed from the wrong population

**Symptom.** The review queue grows steadily; reviewers report working through
backlog at the end of each week; escalations are delayed.

**Competing causes.** (a) The routing fraction rose because the confidence threshold
was tightened, so arrivals grew faster than planned. (b) Service time per routed case
is longer than the population mean used in the sizing, because routing selects the
hard cases. (c) Absence, rework, and escalation were excluded from `r`, so true
capacity is lower than assumed.

**Discriminating check.** Instrument three quantities separately for two weeks:
arrivals by day, completion timestamps by reviewer, and service time measured only on
routed cases. Compare `rho = q*N/r` computed from measurements against the plan. If
arrivals match plan but service time is higher, the cause is (b). If service time
matches but arrivals are higher, it is (a). If neither matches but completion
timestamps show large gaps, it is (c).

**Bounded fix.** Re-size with measured routed-case service time and staff to
`rho <= 0.85`. Where the threshold caused the rise, re-evaluate the threshold jointly
with the harm estimate rather than unilaterally loosening it to reduce load.

**Verification.** Backlog trend is non-increasing over a four-week window that
includes at least one absence, and the utilisation measure stays below 0.85
throughout.

**Residual risk.** Loosening the threshold reduces load and increases harm. Any fix
that trades load for unchecked volume must record the harm change, not just the queue
improvement.

### 8.5 MAP-F04 — Vendor comparison across incomparable protocols

**Symptom.** Three vendors report 91%, 88%, and 86%; the team selects the first.
Six weeks later, on the organisation's own cases, the ranking inverts.

**Competing causes.** (a) Different populations: each vendor evaluated on its own
curated sample, with different difficulty and class mix. (b) Different scoring rules,
for example one counting an abstention as a failure and another excluding it. (c)
Different denominators, where one vendor reports accuracy per document and another per
field, making the second look worse on multi-field documents.

**Discriminating check.** Obtain, for each vendor, the class mix, the abstention
treatment, and the unit of the denominator. Then run all three on the same 300-case
frozen set with the same rubric. Any vendor that cannot run on the frozen set is
disqualified from the numeric comparison and may only be compared qualitatively.

**Bounded fix.** Publish a single acceptance protocol before the comparison, including
the case-selection rule, the rubric, the abstention policy, and the denominator
definition. Report raw outputs alongside scores so that disagreements can be
inspected.

**Verification.** The protocol document is versioned and dated before evaluation;
each vendor's result sheet cites the same protocol version; a sample of 20 outputs per
vendor is independently re-scored and the agreement rate is reported.

### 8.6 MAP-F05 — The control boundary passes on paper and fails in configuration

**Symptom.** The contract states that no customer content is retained for training.
The security review of the actual configuration shows prompt and response logging
enabled, retained for thirty days, in a region outside the agreed boundary.

**Competing causes.** (a) Default logging settings differ from the contractual
commitment and nobody changed them. (b) An integration component — an observability
tool, a debugging proxy, or an evaluation harness — copies payloads to a third
location that is not in the data-flow diagram. (c) The commitment applies to a
different product tier than the one deployed.

**Discriminating check.** Produce a field-level data-flow diagram from the deployed
configuration, not from the contract. For each field, record every location it is
written to, the retention period, and the identity that can read it. Compare against
the approved boundary. The check must be performed by someone who can read the
configuration, and must include the observability layer.

**Bounded fix.** Disable the offending sink or route it inside the boundary. Where a
field is unnecessary, remove it from the request — the cheapest control is not sending
the data. Where the field is necessary, the decision belongs to the boundary owner,
with the residual risk recorded.

**Verification.** A fresh deployment from the repository, instrumented with the
logging configuration in place, shows no field crossing the boundary outside the
approved list. Repeat the check on any integration change, because new components
reintroduce the problem.

### 8.7 MAP-F06 — Automation repeats a harmful action

**Symptom.** A supplier receives three identical payment adjustments for one dispute;
support spends two days unwinding them.

**Competing causes.** (a) A timeout triggered a retry against a non-idempotent
endpoint, so each attempt created a new effect. (b) A duplicate inbound event — a
redelivery from the message broker or a double submission from the upstream form —
produced two independent workflow runs. (c) The workflow legitimately decided twice,
for example after a state refresh changed the extracted amount, and nothing in the
design rejected the second decision.

**Discriminating check.** Look for an idempotency key on the outbound call and a
deduplication key on the inbound event. Replay one event with the same keys and
inspect the downstream state history. If the replay creates a second effect, (a) or
(b). If the state history shows two distinct decisions with different inputs, (c).

**Bounded fix.** For (a): attach an idempotency key derived from the business event,
not from the attempt. For (b): deduplicate on the upstream event identity with a
window longer than the redelivery horizon. For (c): require the workflow to read the
current state and refuse to act on a stale decision, with an explicit conflict
resolution path.

**Verification.** A replay test sends the same event three times with the same key and
produces exactly one downstream effect, confirmed by reading the state history. The
test runs in the deployment pipeline, not once by hand.

### 8.8 MAP-F07 — The rules baseline was never considered

**Symptom.** A six-month modelling effort is proposed for a classification whose
policy is written, stable, and has fewer than two hundred enumerated exceptions.

**Competing causes.** (a) Prestige or funding dynamics favour a learned method. (b)
Nobody measured the exception volume, so the problem looked harder than it is. (c) The
policy genuinely needs ranking or scoring within a category that rules cannot express.

**Discriminating check.** Measure exception volume over a quarter and the rate at
which the policy changes. Fewer than a few percent of cases falling outside the rules,
with infrequent policy change, indicates a rules solution. Frequent policy change with
stable exceptions indicates a maintenance problem that a learned model will also have,
in a harder-to-audit form.

**Bounded fix.** Implement and measure the rule baseline first; its residual error
becomes the specification for anything learned. This is not a detour — it is the
comparator every later claim needs.

**Verification.** The rule baseline's accuracy and its maintenance cost are recorded
as the reference, and any learned alternative is evaluated against it on the same
population.

### 8.9 MAP-F08 — Retrieval returns an unauthorised document

**Symptom.** A pilot user is shown, and cites, a document from a business unit they
have no right to read.

**Competing causes.** (a) Access filtering happens after ranking and is applied to the
displayed page rather than the retrieved set, so the citation survives in a summary.
(b) The access-control index is stale, and a revoked user or reassigned document is
still represented. (c) Identity mapping failed at the connector, so the retrieval
service queried with a service identity holding broader rights than the user.

**Discriminating check.** Run a probe under a test identity with a known-narrow
entitlement set: request the same query as a broad-entitlement user and as the narrow
one, and diff the retrieved sets. If the sets are identical, the filter is absent or
running with a service identity (c). If the narrow set still contains revoked
material, the index is stale (b). If the citation appears in a summary but not in the
retrieved set, filtering is post-hoc (a).

**Bounded fix.** Enforce authorisation at query time using the requesting identity,
before ranking. Rebuild the access index on a schedule tied to entitlement change
events, and carry the user identity through summarisation so that any generated text
inherits the caller's entitlements.

**Verification.** The probe runs in CI as a regression test: the narrow identity must
retrieve a strict subset, and a revoked document must be absent within one
entitlement-propagation window.

### 8.10 MAP-F09 — Fluent output read as grounded output

**Symptom.** A generated answer contains a confident procedural instruction that no
source supports; the instruction is acted on.

**Competing causes.** (a) Retrieval returned nothing relevant and the model answered
from parametric memory. (b) Retrieval returned an outdated version of a policy that
has since changed. (c) Retrieval returned the right document but the model paraphrased
past the qualifier, turning "may, if approved" into "should".

**Discriminating check.** For a sample of outputs, record whether each claim is
entailed by a cited passage. Separately record the document version and effective
date. If claims fail entailment with no relevant citation, (a). If entailment holds
against a superseded version, (b). If the passage contains the qualifier and the
output drops it, (c).

**Bounded fix.** Require claim-level citation with an abstain-when-absent rule;
filter the corpus by effective date at query time; add a rubric item that scores
qualifier preservation and rejects outputs that harden a conditional into an
instruction.

**Verification.** An adversarial set of queries with no supporting source produces
abstentions rather than answers, and a set of superseded-document queries returns the
current version. Rubric scores for qualifier preservation are reported alongside
overall acceptance.

### 8.11 MAP-F10 — A score shipped as though it were a policy

**Symptom.** A well-calibrated risk score is in production; nobody can say which
threshold is in force, or why the routed volume is 40% when the capacity plan assumed
12%.

**Competing causes.** (a) The threshold was chosen on a test set whose class mix
differs from production. (b) The threshold was chosen for a target precision without
checking the implied volume against capacity. (c) Different services apply different
thresholds because the value was duplicated in configuration rather than centralised.

**Discriminating check.** Plot, on a held-out set drawn from production, the routed
volume and the harm-weighted error at each threshold. Compare the production routed
volume against the plan. If production volume differs while the threshold is
identical, the class mix differs (a). If volume matches the plan's threshold but
exceeds capacity, the threshold was never capacity-checked (b). If services disagree
on the routed set for the same input, (c).

**Bounded fix.** Treat the threshold as a joint decision of the domain owner (harm) and
the operations owner (capacity), record it with the evidence behind it, and serve it
from one configuration source.

**Verification.** The routed volume observed in production is within a stated band of
the planned volume for two consecutive weeks, and a configuration audit shows one
threshold value in force.

### 8.12 MAP-F11 — Multi-step tool use granted without an authority model

**Symptom.** A workflow that was demonstrated as a text assistant has, after several
increments, the ability to issue refunds and update supplier records, and nobody can
enumerate what it can do.

**Competing causes.** (a) Incremental credential grants, each individually reasonable,
accumulated into broad authority. (b) Tool descriptions are interpreted by the model,
so untrusted content in a retrieved document can suggest a tool call the designer
never intended. (c) A shared service identity means every step holds the union of all
tools' permissions.

**Discriminating check.** Produce an authority inventory: for each tool, the
operations it can perform, the identity it uses, the input validation applied, and the
worst-case effect of one call. Then attempt an injection probe: place an instruction
inside a retrieved document and observe whether a tool call follows. And read the
identity configuration: if all tools share one identity, (c) is present regardless of
the other causes.

**Bounded fix.** Give each tool a distinct identity with least privilege; validate
tool inputs against a schema that does not accept free-form targets; require explicit
approval for any state-changing call above a stated effect size; and treat all
retrieved content as untrusted input that cannot alter the permitted tool set.

**Verification.** The authority inventory is reproducible from configuration by
someone outside the team; the injection probe produces no state change; and a
worst-case single-call drill shows the effect ceiling is enforced.

### 8.13 MAP-F12 — Adoption measured as logins

**Symptom.** A dashboard reports 92% adoption; interviews find that most users open
the tool once, see an unhelpful result, and return to their previous method.

**Competing causes.** (a) The numerator counts sessions rather than meaningful use.
(b) The denominator includes people with no relevant cases, deflating or inflating the
rate depending on direction. (c) Use is mandated, so opening the tool is compliance
rather than adoption.

**Discriminating check.** Recompute with a meaningful-use rule declared in advance
(for example, a cited result incorporated into a deliverable within seven days) and a
denominator restricted to people with at least one relevant case in the window. If the
rate collapses, (a). If the rate rises sharply, the old denominator was wrong (b). If
usage clusters immediately after reminders, (c).

**Bounded fix.** Publish both the opportunity-based denominator and the meaningful-use
numerator, and stop reporting login counts as adoption. Pair the metric with a
reason-coded sample of non-use, because the fix for poor output quality is different
from the fix for missing workflow integration.

**Verification.** Two consecutive reports show numerator, denominator, and the rule
definitions; a reason-coded sample of at least thirty non-users is attached, and at
least one actionable cause has an owner.

### 8.14 MAP-F13 — The evaluation set was contaminated

**Symptom.** Offline results are excellent; field results are far worse, and the gap
does not close with more data.

**Competing causes.** (a) Near-duplicate documents appear in both training and test
splits. (b) The test set was assembled from cases the team had already discussed, so
the rubric was effectively tuned to it. (c) The test set is genuinely clean but drawn
from a period whose distribution differs from current traffic.

**Discriminating check.** Run a near-duplicate search between splits on content
hashes and embeddings; a non-trivial overlap confirms (a). Compare rubric revision
dates against test-set assembly dates; rubric items added after assembly suggest (b).
Compare feature distributions and outcome base rates between the test period and the
last month of traffic; a shift confirms (c).

**Bounded fix.** Deduplicate by content before splitting, ideally split by time rather
than at random; freeze the rubric before assembling the test set; keep a permanently
held-out recent window that is never inspected during development.

**Verification.** A duplicate scan reports zero overlap at the chosen similarity
threshold; the held-out recent window produces results within a stated band of the
field measurement.

### 8.15 MAP-F14 — Cost per call mistaken for cost per outcome

**Symptom.** A business case built on $0.30 per call delivers a cost line six times
higher in the first month.

**Competing causes.** (a) Acceptance is low, so cost per accepted output is
`1/y` times higher. (b) Retries and multi-call chains were unbudgeted. (c) Review and
rework labour, which dominate, were omitted because they sit in a different budget.

**Discriminating check.** Reconstruct the first month's spend by line: inference,
retries, review labour, rework, incident time. Compare against the model terms `k`,
`F`, `w`. Any line absent from the model is the cause. If the inference line alone
exceeds the model, count calls per unit to detect chains (b).

**Bounded fix.** Rebuild the model with [MAP-05], including `w` and chains, and report
`C_acc` rather than per-call price. Where chains are unavoidable, measure the actual
call multiplier rather than assuming one.

**Verification.** The next month's reconstructed spend is within a stated percentage
of the model, per line, with the largest line item justified by a measurement.

### 8.16 MAP-F15 — Version drift changes behaviour silently

**Symptom.** Outputs change character over a weekend with no deployment in the
organisation's own change log.

**Competing causes.** (a) A managed endpoint was updated upstream under a floating
version alias. (b) A retrieval index was rebuilt and ranking shifted. (c) An upstream
data source changed format or freshness.

**Discriminating check.** Record the resolved model version in every response log; if
it changed, (a). Compare index version identifiers and a fixed probe query's ranked
output across the boundary; if rankings moved, (b). Check source freshness timestamps
and schema versions; a changed schema or a stalled feed confirms (c).

**Bounded fix.** Pin versions explicitly and require a change record for any pin
update; snapshot the index version with each evaluation; monitor source freshness as a
first-class signal with an alert on staleness.

**Verification.** A fixed probe query, run daily, produces a recorded ranking whose
changes are explained by a change record; response logs show a resolved version that
matches the pin.

### 8.17 Edge cases worth naming separately

**Zero-denominator slices.** A language, region, or document type with no cases in the
window produces a rate of undefined value, not zero. Reporting it as zero invites a
decision to drop a slice that simply was not present this month.

**Seasonal eligibility.** If eligibility depends on a calendar (fiscal close, harvest,
enrolment), the monthly window compares different populations across months. Compare
like months, or normalise by an eligibility count rather than a fixed denominator.

**Bimodal consequence.** When errors are mostly trivial and occasionally severe, the
mean harm hides the mode that matters. Report the distribution, and gate the severe
mode separately as Section 4.9 requires.

**Overlapping entitlements.** Two identity systems with different group definitions
produce a retrieved set that satisfies neither policy. Test with an identity whose
entitlements are defined in only one system, and observe what happens.

**The reviewer who cannot say no.** If the interface makes acceptance the default and
rejection requires typing a reason, measured acceptance is a UI artefact. Measure
rejection in an interface where both actions cost the same effort.

**Time-zone windows.** A "month" computed in local time and a "month" computed in UTC
differ by up to a day of volume, which matters when a threshold is close. Fix the
window definition in the protocol, not in each report.

**Takeaway:** Every failure above is a bypass of one of the six calculations in
Section 4; the discriminating check is always the same move — recompute the quantity
from measurements rather than from the plan, and compare the two.

## 9. Complexity & Cost Analysis

### 9.1 What "complexity" means for a market map

There is no algorithm here whose asymptotic class the reader must derive. The
analogue is the **scaling behaviour of each cost and effort term as volume, variety,
and authority grow**, plus the marginal cost of the evidence needed to justify each
step up the authority ladder. Two proposals can be identical in unit price and differ
by an order of magnitude in operating cost, because their dominant terms grow at
different rates.

The four scaling families that matter are: *linear in volume* (inference, review
labour proportional to routed units, storage of request logs); *linear in variety*
(per-slice evaluation, per-language quality work, per-source connector maintenance);
*stepwise in authority* (each additional permission adds review, testing, and incident
surface in discrete jumps); and *constant per deployment* (fixed platform cost, base
evaluation effort). Getting the family right is most of the estimate.

### 9.2 Cost and effort table

| Operation | Input size | Time / cost model | Space / state | Dominant assumption |
|---|---|---|---|---|
| Retrieval query | `D` indexed documents, `k` returned | index-lookup dominated; measure p95, do not extrapolate from a mean | index storage plus per-query working state | authorisation filtering happens before display |
| Constrained classification | `N` units, `L` labels | `O(N)` calls or `O(N)` rows through one batch scoring pass | feature buffer or request payload | label set fixed; drift monitored |
| Generative assistance | `N` requests, `t_in + t_out` tokens each | tokens × price × `N`, plus retry multiplier `1/(1-pi)` | context plus log retention | output length and retry rate are measured, not assumed |
| Human review | `q*N` routed units | `s` minutes per routed unit × `q*N` / 60 hours | queue state and audit trail | `s` measured on routed cases |
| Acceptance audit | sample of `m` accepted outputs | `s_audit` minutes each, constant in `N` if `m` fixed | audit record | sampling is random within stratum |
| Reversal of a bad action | `(1-y)*N` rejected units | `w` per reversal, includes downstream reconciliation | reversal log and state history | reversal is possible at all |
| Per-slice evaluation | `S` slices, `C` cases each | `O(S*C)` judging effort | case bank and rubric versions | cases per slice above the noise floor |
| Connector maintenance | `M` sources | `O(M)` ongoing effort, roughly constant per source per quarter | schema and credential state | sources change at a bounded rate |
| Version pin and re-validate | per pin change | one full acceptance cycle per change | evaluation artefacts | acceptance protocol stays stable |

### 9.3 Worked sizing: the twelve-month cost of the Section 5 recommendation

Take Option A from Section 5 as the deployed choice: 12 FTE at `$6,400` loaded,
`$1,100` fixed platform cost, `$0.02` variable per served unit at 21,600 served units,
and an acceptance protocol that is re-run quarterly. Add the recurring evaluation and
version-management effort, which proposals usually omit entirely.

```text
Monthly run cost (from Section 5.5)          $78,332
Annual run cost    = 78,332 * 12             $939,984
Quarterly acceptance cycle: 4 per year, each requiring
  case-bank refresh         24 h
  scoring and adjudication  36 h
  slice analysis            12 h
  write-up and gate review   8 h
  total 80 h per cycle; 4 cycles = 320 h
  at $85/h loaded analyst time = $27,200/year
Annual total                                 $967,184
```

Check the multiplication: `78,332*12 = 78,332*10 + 78,332*2 = 783,320 + 156,664 =
939,984`. Correct. `320*85 = 27,200`. Correct. The evaluation overhead is **2.8%** of
run cost — `27,200/967,184 = 0.02812` — which is small in this case only because the
workload is large. At a tenth of the volume, the same fixed evaluation effort would be
28% of run cost, and the small deployment would be disproportionately expensive to
keep honest.

### 9.4 Marginal cost of climbing one authority grade

Authority is the term that scales in steps. For the same workload, moving from propose
to act-with-sampling adds, in the ILLUSTRATIVE case of Section 5:

```text
  cap enforcement in code and configuration audit      40 h  ->  $3,400
  reversal drill design and first execution            32 h  ->  $2,720
  audit sampling protocol and tooling                  48 h  ->  $4,080
  harm estimate rework with an independent reviewer    24 h  ->  $2,040
  incident runbook and on-call integration             36 h  ->  $3,060
  acceptance protocol extension for the acted subset   40 h  ->  $3,400
  total one-off                                          220 h -> $18,700
```

Check: `3,400+2,720+4,080+2,040+3,060+3,400 = 18,700`. And `220*85 = 18,700`, so the
hourly rate is applied consistently. Against an annual run cost near `$967,000` this
looks trivial, which is precisely the trap: the one-off engineering is not the cost.
The recurring cost is the audit sample and the incident readiness, which appear as
additional review load month after month.

```text
Recurring additions at act-with-sampling, per month:
  audit sample: 2.0 min * (sample of 1,680) / 60 = 56.0 h  -> $4,760
  on-call share: 0.25 FTE-equivalent                        -> $1,600
  monthly total                                              $6,360
Annualised recurring                                          $76,320
```

Check: `56.0*85 = 4,760`; `0.25*6,400 = 1,600`. Correct. So the true annual price of
the authority step is `$18,700 + $76,320 = $95,020` — about 10% of run cost, and more
than three times the one-off engineering that occupies all the planning attention.

### 9.5 Cost of a wrong deployment mode

The expensive mistake is not choosing the second-best mode; it is choosing a mode that
must be abandoned after integration. Model the abandonment cost explicitly, because it
is the quantity that justifies spending on the control gate up front.

```text
ILLUSTRATIVE abandonment of a managed-API integration after 7 months:
  integration engineering already spent        1,100 h  -> $93,500
  evaluation effort spent                        320 h  -> $27,200
  workflow change management and training        180 h  -> $15,300
  re-integration on the compliant mode         1,300 h  -> $110,500
  duplicated run cost during overlap (2 months)          -> $12,400
  total                                                  $258,900
```

Check: `93,500+27,200+15,300+110,500+12,400 = 258,900`. The control gate that would
have prevented this cost roughly `$18,700`-scale effort (the section 9.4 shape, applied
to boundary analysis rather than authority). The ratio is about 14:1 in favour of
spending on the gate. That ratio, not a compliance slogan, is the argument for doing
boundary work first.

### 9.6 Latency as a cost, with units

Latency is a cost when it consumes human attention or breaches a deadline. Convert it
to money so it can be compared with other terms.

```text
FORMULA [MAP-09]: L = u * (t_int - t_base) / 3600 * v
WHERE: L attention cost of added latency [currency/window];
       u interactions [count/window]; t_int, t_base mean interaction latency
       [s]; 3600 converts seconds to hours; v loaded attention value
       [currency/h].
CONSTRAINT: latency within the range where users wait rather than abandon;
            above the abandonment threshold the cost is lost work, not
            waiting time, and a different model applies.
NUMERICAL EXAMPLE: ILLUSTRATIVE u = 21,600/month, t_int = 2.0 s,
                   t_base = 0.3 s, v = $62/h ->
                   L = 21,600 * 1.7 / 3,600 * 62 = 21,600*0.000472*62 = $632.4/month.
INTERPRETATION: at this volume the attention cost of latency is a rounding error
                against a $78,332 monthly run cost; do not trade quality for
                latency here. At 1,000 interactions/month with a 9 s increase,
                L = 1,000*9/3,600*62 = $155/month, still small, but the
                abandonment threshold matters more than the arithmetic.
```

**Boundary.** The formula assumes users wait. Above the abandonment threshold, latency
destroys the interaction rather than slowing it, and no linear attention model
applies; measure abandonment directly rather than extrapolating.

### 9.7 Storage and retention as a growing term

Logs that support audit and evaluation are the only cost that grows without bound if
retention is unmanaged, because they accumulate per request forever.

```text
FORMULA [MAP-10]: S_ret = N * b * R
WHERE: S_ret retained storage [bytes]; N requests/month [count/month];
       b bytes per request record including prompt, response, metadata [bytes];
       R retention period [months].
CONSTRAINT: excludes indexes, replicas, backups, and compression; payload size
            varies with document length, so use the measured mean and state
            the tail.
NUMERICAL EXAMPLE: ILLUSTRATIVE N = 21,600/month, b = 6,400 bytes (about 1,600
                   tokens of text plus metadata), R = 18 months ->
                   S_ret = 21,600*6,400*18 = 2,488,320,000 bytes
                        = 2.48832e9 / 1.073741824e9 = 2.3174 GiB.
INTERPRETATION: small in absolute terms for this workload, so retention is a
                policy question rather than a cost question here. The same
                calculation at 12 million requests/month with 40 KB records
                and 36-month retention gives 17.28 TB, where retention becomes
                a budget line and a discovery liability.
```

Check the second case: `12e6 * 40,000 * 36 = 1.728e13` bytes; divide by `1e12` for
decimal terabytes gives `17.28 TB`. Correct.

**Boundary.** Compression and deduplication can change this by a large factor, and
so can storing hashes instead of payloads. The decision that matters is whether the
payload must be retained for audit at all; if a hash plus a rubric score suffices,
the storage term collapses.

### 9.8 Complexity of the evaluation programme

Evaluation cost scales with the number of slices that must be reported, not with
volume, and slices multiply: language, document type, region, channel, supplier
segment. If `S` slices each need `C` cases to keep the noise acceptable, the judging
effort is `S*C` case-judgements per cycle regardless of how many requests the system
serves.

| Slices `S` | Cases per slice `C` | Judgements per cycle | Minutes each | Hours | Cycles/year | Hours/year |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 75 | 300 | 4.0 | 20.0 | 4 | 80 |
| 8 | 75 | 600 | 4.0 | 40.0 | 4 | 160 |
| 8 | 150 | 1,200 | 4.0 | 80.0 | 4 | 320 |
| 16 | 150 | 2,400 | 4.0 | 160.0 | 4 | 640 |
| 16 | 150 | 2,400 | 4.0 | 160.0 | 12 | 1,920 |

Read the last two rows together: the same programme costs three times more annually if
run monthly rather than quarterly. The choice of cadence is therefore a cost decision
with a quality consequence, and it should be made against a drift risk estimate rather
than by default.

At `$85/h`, the third row costs `80*4 = 320` hours per year, `$27,200` — which is the
figure used in Section 9.3. At the fourth row it is `$54,400`, and at the fifth
`$163,200`, at which point evaluation is a material fraction of a large deployment's
cost and the case for automated pre-scoring with human adjudication of a sample
becomes strong.

### 9.9 Total cost of ownership, assembled

A complete monthly view for a deployed system, in the order the terms appear in a
real budget:

| Term | Formula | Section 5 Option A | Grows with |
|---|---|---:|---|
| Inference / platform | `F + k*n_served` | $1,532 | volume, linearly |
| Review labour | `v * s*q*N/60` | $76,800 | routed volume, linearly |
| Audit labour | `v * s_a*m` | $0 (propose only) | sample size |
| Rework / reversal | `w*(1-y)*N` | not material for evidence-only | rejection rate |
| Evaluation programme | `v * S*C*t_j/60 * cycles` | $27,200/year | slices × cadence |
| Version and pin management | engineering hours | ~$8,000/year | number of pins |
| Incident readiness | share of on-call | ~$19,200/year | authority grade |
| Retention storage | `N*b*R*price` | small | volume × retention |
| Exit / portability reserve | estimate | not quantified here | integration depth |

Summed for Option A: `1,532 + 76,800 = 78,332/month`, `939,984/year`, plus
`27,200 + 8,000 + 19,200 = 54,400` of recurring programme cost, for `$994,384/year`.
The point of the table is the ordering: labour and evaluation dominate, and inference
price — the number every vendor pitch leads with — is under 2% of the total. Any
business case that turns on a per-call price difference between two vendors is
optimising the smallest term.

**Takeaway:** Size the labour, the evaluation cadence, and the authority step, not the
per-call price; in every worked case here the model cost is a rounding error against
the cost of the people who must check, audit, and recover.

## 10. Key Theorems / Results Reference

### 10.1 How to use this section

Each entry names a result, states the conditions under which it may be invoked,
records what it does *not* establish, and labels the evidence. The results are used as
reasoning instruments, not as authority: invoking one incorrectly is worse than
reasoning from first principles, because it carries the appearance of rigour.

| Result | Statement as used here | Conditions / limits | Evidence |
|---|---|---|---|
| Linearity of expectation | the expected total of per-unit contributions equals the sum of per-unit expectations, so total expected value scales as `N` times the per-unit expectation | holds without independence; requires that the same outcome distribution applies to each unit, so heterogeneous slices must be summed separately | DERIVED (elementary probability) |
| Subset monotonicity of a part-over-whole ratio | if `S` is a subset of `E` then `0 <= |S|/|E| <= 1`, and coverage is non-increasing when units are removed from `S` | requires `S` measured on the same `E`; a changed eligibility rule makes the comparison invalid | DERIVED |
| Law of total expectation across a served/unserved partition | total expected correct outcomes equal `c*N*y + (1-c)*N*y_fb` | requires `y_fb` to be measured, not assumed; the partition must be exhaustive and disjoint | DERIVED |
| Queue stability condition | a work-conserving system is stable in the long run only if the long-run average arrival rate does not exceed the long-run average service rate | averages are insufficient at `rho` near 1; variability determines waits, so stability is necessary but not sufficient for an acceptable service level | DERIVED (workload conservation); practice guidance is HEURISTIC |
| Break-even of two linear cost functions | total costs cross at `N* = (F2-F1)/(k1-k2)` when `k1 != k2` | linearity over the relevant range; step changes in fixed cost create multiple crossings | DERIVED |
| Least privilege | a process should hold only the permissions required for its declared function | requires a threat model and an enforceable mechanism; a document's permission list is not enforcement | SOURCED — see [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final), 2020, accessed 2026-11-09 |
| Idempotency as a safety condition for retries | an operation safe to retry must produce the same durable effect for repeated identical requests | requires a stable key derived from the business event, not from the attempt, and deduplication storage at least as long as the retry horizon | DERIVED (distributed-systems practice) |
| Calibration | predicted probabilities should match observed frequencies on the declared population | requires a declared population, sufficient sample per bin, and re-estimation under drift | SOURCED method reference — [NIST/SEMATECH e-Handbook](https://www.itl.nist.gov/div898/handbook/), accessed 2026-11-09 |
| Tail risk is not captured by a mean | two distributions with identical means can differ arbitrarily in worst-case loss | any decision that cannot absorb the worst case must gate on a tail quantile separately | DERIVED |
| Cost per accepted unit exceeds cost per attempt by `1/y` | from `C_acc = (k*N + F + w*(1-y)*N)/(y*N)` | requires acceptance measured on the target population with a frozen rubric | DERIVED (Section 4.5) |
| Selection under a hard constraint is lexicographic | when one criterion is a constraint rather than a preference, options violating it are removed before any trade-off is computed | requires the constraint to be declared by its owner; converting a constraint into a penalty changes the decision rule silently | DERIVED (decision theory) |
| Sampling error grows as the slice shrinks | the standard error of a proportion scales as `sqrt(p(1-p)/n)`, so small slices cannot support fine distinctions | independence within the slice; clustered cases inflate the error beyond this expression | SOURCED method (standard estimation theory) |
| Context-window sufficiency | a single-pass prompt can only condition on content that fits the window, so long documents require chunking, retrieval, or summarisation with corresponding information loss | each transformation loses or distorts information; the loss must be measured, not assumed away | DERIVED |
| Reversibility reduces required proof burden | where an action can be undone at bounded cost, the expected cost of an error is bounded by the reversal cost rather than the full consequence | requires the reversal to be tested under load, not merely designed, and the reversal window to exceed the detection delay | HEURISTIC, with the arithmetic consequence shown in Section 4.2 |

### 10.2 Three results that are commonly misapplied

**Linearity of expectation, misapplied to causal claims.** The result says nothing
about whether the intervention caused an outcome. It aggregates whatever per-unit
probabilities you supply. Teams sometimes present an expected-value calculation as
though it demonstrated benefit; it demonstrates only the arithmetic consequence of
assumed probabilities.

**Queue stability, misapplied as a service-level guarantee.** Stability says the
backlog will not grow without bound in the long run. It does not say the 95th
percentile wait is acceptable, and a stable queue at `rho = 0.97` can have waits
measured in days. The design target of `rho <= 0.85` is practice guidance, not a
theorem, and should be labelled as such.

**Least privilege, misapplied as a procurement question.** Least privilege is a
property of the deployed configuration and the identities in use. A vendor's
capability to enforce fine-grained permissions is not the same as the customer's
deployment actually doing so. The check is always the configuration, never the
datasheet.

### 10.3 Results the map deliberately does not use

**Benchmark leaderboards.** A published ranking measures a task on a dataset with a
specific protocol. It cannot rank options for a different task, and its uncertainty is
usually not reported in a form that supports a decision. Used as orientation, fine;
used as selection, it is the error described in MAP-F04.

**Maturity models.** A level assignment compresses five independent layers into one
ordinal, which destroys exactly the information the map needs. A vendor at "level 4"
in a maturity framework may hold no credentials and store no data, or the reverse.

**Composite scores.** Any weighted sum of capability, cost, control, and risk requires
exchange rates between incommensurable quantities, and those rates encode a policy
decision nobody has made explicitly. The map keeps the quantities separate and decides
lexicographically, which is slower to explain and far harder to game.

### 10.4 An invariant to carry through every comparison

For any proposed deployment:

```text
INVARIANT [MAP-I1]:
  correct_outcomes + incorrect_outcomes + abstentions = N
  and every reported rate names the population it was measured on.
```

This is trivially true and surprisingly often violated. When a report's numbers do not
reconcile to `N`, the missing units are the interesting part: they are the ones that
timed out, were filtered, were escalated without a recorded outcome, or fell outside
the window. Reconciling to `N` is the cheapest audit available and it catches a large
share of the failures in Section 8 before they reach a decision meeting.

**Takeaway:** Cite a result only with its conditions; when a report cannot reconcile
its outcomes to the eligible population, treat the gap as the finding.

## 11. Interview / Exam Questions

### 11.1 How answers are scored

Each question below carries a checked answer, the reasoning that earns credit, a
follow-up a strong candidate anticipates, and a **red flag** — the specific phrase or
omission that indicates the candidate is reciting rather than reasoning. Answers are
graded on whether the arithmetic is performed, the population is named, and the
decision consequence is stated. A candidate who names the right framework but cannot
compute coverage for a two-line scenario has not learned the material.

### 11.2 Recall and definition

**Q1. What are the three axes of the market map, and why is the order of application
not arbitrary?**

*Answer.* The axes are capability (what transformation, on what input, producing what
output contract), control (which data crosses which boundary, and what authority the
system holds), and evidence or proof burden (what must be demonstrated before that
authority is granted). They are applied in that order, with the control gate before
the economics gate and the evidence gate last, because a cheaper option inside a
prohibited boundary has no value and a perfectly validated system whose review load
exceeds capacity cannot be operated. Applying economics first produces a preferred
option that may later be removed, which distorts the decision socially even after the
constraint is found.

*Reasoning credit.* Naming the gate order and stating the failure it prevents.

*Follow-up.* "What if the economics are so favourable that the constraint 'must' be
revisited?" Answer: that is a legitimate resolution, but it is a decision by the owner
of the constraint, recorded with residual risk — not a recalculation performed by the
proposal team.

*Red flag.* Listing axes as dimensions of a diagram without stating any gate ordering
or consequence.

**Q2. Distinguish a capability from a deployment mode, using one concrete system.**

*Answer.* A capability is a declared input-to-output contract: for example, given one
supplier message up to 4,000 characters, return one label from a fixed set of nine,
with an explicit `unknown`, and no free-text justification. A deployment mode is where
that contract is executed and who controls it: a managed API under a data-processing
agreement with a pinned version, a self-hosted service inside a tenancy with a named
operator, or an embedded model at the interaction point. The same capability can be
delivered in any mode; the mode determines data route, latency shape, failure
semantics, upgrade control, and where logs accumulate, but not the accuracy claim.

*Reasoning credit.* Producing a testable contract rather than a category word.

*Follow-up.* "Which mode for a workload where a prohibited field is present?" Answer:
the mode is chosen by the boundary, not by price; if no mode satisfies it, narrow the
scope or retain the baseline.

*Red flag.* Describing deployment mode as "cloud versus on-premise" without naming
operator, residency, or version control.

**Q3. What are the six autonomy grades, and what changes between them?**

*Answer.* Propose only; propose with pre-fill; act with sampling; act within limits;
act with rollback; irreversible autonomy. What changes is the set of effects the system
may cause without prior approval, and therefore the evidence required: measured
reviewer rejection rates, enforced caps, a tested reversal within a stated window, and
an independently estimated worst-case single event accepted by a second owner. The
technology stack can be identical across all six, which is why "we deployed an agent"
communicates nothing about risk.

*Reasoning credit.* Naming that authority, not capability, is graded.

*Follow-up.* "Which grade can you defend with offline evaluation only?" Answer: none
above propose-only; offline evidence cannot establish field behaviour under time
pressure.

*Red flag.* Treating autonomy as binary, or claiming a grade on the evidence of a demo.

### 11.3 Calculation

**Q4. A workflow has 18,000 eligible units per month. The system serves 13,500 of them
with 84% acceptance; the fallback succeeds on 91% of the units it handles. Compute
coverage and the total expected correct outcomes, and state what the acceptance figure
alone would have claimed.**

*Answer.* Coverage `c = 13,500/18,000 = 0.75`. Served correct: `0.75*18,000*0.84 =
11,340`. Unserved correct: `0.25*18,000*0.91 = 0.25*16,380 = 4,095`. Total
`11,340 + 4,095 = 15,435` correct outcomes, which is `15,435/18,000 = 85.75%` of
eligible units. The acceptance figure alone would have been reported as 84%, and a
reader could reasonably — and wrongly — conclude that the system is worse than the
fallback at 91%. The system is better overall only because it handles units the
fallback also handles well; the honest comparison is against the fallback's
performance *on the served subset*, which may be lower than 91%.

*Reasoning credit.* Performing the law-of-total-expectation split and noticing that
`y_fb` must be measured on the same served subset to make a valid comparison.

*Follow-up.* "What if the fallback's accuracy on the served subset is 96%?" Then the
system's 84% is a degradation on those units, and the correct conclusion is to route
them to the fallback.

*Red flag.* Multiplying 84% by 18,000 and presenting that as correct outcomes.

**Q5. With `N = 20,000`, `q = 0.12`, and reviewer capacity of 1,900 reviews per month,
is the policy feasible? How many reviewers at 155 reviews each?**

*Answer.* Arrivals `0.12*20,000 = 2,400/month` against capacity 1,900: infeasible by
500/month, `rho = 2,400/1,900 = 1.263`, an unstable queue. Required capacity at
`rho <= 0.85` is `2,400/0.85 = 2,823.5`, so 2,824 reviews/month. At 155 reviews per
reviewer that is `2,823.5/155 = 18.2`, so 19 reviewers. Note the difference between
"enough to match mean load" (16 reviewers at `2,400/155 = 15.5`) and "enough to hold
`rho <= 0.85`" (19): staffing to the mean is the error the question is designed to
expose.

*Reasoning credit.* Computing both the naive headcount and the stability-adjusted one.

*Follow-up.* "What if review is only possible by two specialists?" Then effective
capacity is theirs, and utilisation against the full pool is a fiction discovered at
the first escalation.

*Red flag.* Dividing by 155 and stopping at 16.

**Q6. Unit price is $0.28 per call, acceptance is 55%, rework costs $1.10 per rejected
output, fixed cost is $2,400 per month, and volume is 15,000 attempts. Compute cost
per accepted output.**

*Answer.* Variable `15,000*0.28 = $4,200`. Rework `1.10*0.45*15,000 = 1.10*6,750 =
$7,425`. Total `4,200 + 2,400 + 7,425 = $14,025`. Accepted `0.55*15,000 = 8,250`.
`C_acc = 14,025/8,250 = $1.70` exactly. The per-call price is 16.5% of the real cost.

*Reasoning credit.* Including `w` and dividing by accepted rather than attempted.

*Follow-up.* "At what acceptance would `C_acc` fall to $1.20?" Solve
`(4,200 + 2,400 + 1.10*(1-y)*15,000)/(y*15,000) = 1.20`. Expanding:
`6,600 + 16,500 - 16,500y = 18,000y`, so `23,100 = 34,500y`, `y = 0.6696`. Note that
no acceptance above 1.0 ever reaches $1.00, because fixed and variable alone are
`6,600/15,000 = $0.44` per attempt — a useful check that the target is achievable at
all.

*Red flag.* Reporting $0.28 as the cost of an output.

**Q7. An expected-value model gives `V = +$2,160` for a proposal. Name three
assumptions that could each flip the sign, and state what you would measure first.**

*Answer.* Three: `p_h` (the harm probability) — in the Section 5 example a rise from
0.012 to 0.015 flipped the sign; `b` (the incremental benefit), which is inflated if
baseline value is counted as intervention value; and `F`, if reviewer staffing was
excluded or if staffing is at `rho = 1.0` and must be raised for variability. The
first measurement should be `p_h`, because it is both the least observed term and the
one with the largest leverage: the harm term is `p_h*h*N`, and `h` is usually large
enough that a tenth of a percentage point matters.

*Reasoning credit.* Identifying leverage rather than listing parameters generically.

*Follow-up.* "How would you estimate `p_h` before deployment?" A shadow run with
outcomes observed downstream, or an audit of a sample of actions if the system already
acts; never a survey of stakeholder concern.

*Red flag.* Proposing to "run a sensitivity analysis" without naming which parameter
the decision is most sensitive to.

### 11.4 Judgment and diagnosis

**Q8. Accuracy improved from 79% to 92% after a change, but total complaint volume
rose. Give the three most likely mechanisms and one check that separates them.**

*Answer.* (a) A stricter abstention or safety filter removed hard cases, so accuracy
rose on a smaller, easier served population while unserved cases went to a weaker
fallback. (b) The evaluation set was regenerated with an easier sampling rule, so the
number changed while the system did not. (c) A genuine improvement that also shifted
the population, so both effects are real. The separating check: recompute coverage
`c` on both sides with the frozen eligibility rule and compute
`c*N*y + (1-c)*N*y_fb` for both periods using a measured `y_fb`. If `c` fell and the
total fell, (a). If `c` and `y_fb` are unchanged and the field complaint data moved
independently of the eval set, (b) — confirm by re-scoring a stored sample from the
old eval set with the new system.

*Reasoning credit.* Naming the total-outcome decomposition and insisting on `y_fb`.

*Follow-up.* "What would convince you the improvement is real?" Consistent
improvement on a frozen, recent, deduplicated set plus a field metric that is not the
eval set.

*Red flag.* Accepting the accuracy figure and searching for a communications
explanation.

**Q9. A data-processing agreement says no customer content is used for training, but
the security review finds thirty-day request logging in another region. What happened,
and how do you decide whether it is a blocker?**

*Answer.* The most likely causes are default logging left enabled, an observability or
evaluation component copying payloads outside the boundary, or a product tier
mismatch. Decide by producing a field-level data-flow diagram from the *deployed
configuration* rather than the contract: for each field, every sink, the retention
period, and the identities that can read it. It is a blocker if a field whose
permitted use excludes that region or purpose crosses the boundary; it is a fix if the
sink is unnecessary and can be disabled. Removing fields from the request is the
cheapest control, because unsent data cannot leak.

*Reasoning credit.* Distinguishing contractual commitment from configuration fact.

*Follow-up.* "Who decides?" The boundary owner (data or privacy function), with the
residual risk recorded if they accept it; not the proposal team.

*Red flag.* Treating the contract clause as evidence of the deployment's behaviour.

**Q10. A pilot reports 91% adoption. You are asked to verify. What does 91% most
likely measure, and what would you report instead?**

*Answer.* Most likely logins or sessions against a headcount denominator that includes
people with no relevant cases. Report instead `a = n_meaningful_use/n_offered` with a
meaningful-use rule declared in advance (for example, a cited result incorporated into
a deliverable within seven days) and a denominator of people with at least one
relevant case in the window. Attach a reason-coded sample of non-users, because the
fix for poor output quality differs from the fix for missing workflow integration. Also
note whether use is mandated: forced use is compliance, not adoption.

*Reasoning credit.* Replacing the numerator definition rather than disputing the
number.

*Follow-up.* "What if the recalculated rate is 34% and the sponsor is unhappy?"
Adoption is not a target; benefit is. A tool used by 34% with high override-free
acceptance may deliver more value than one opened by 91%.

*Red flag.* Reporting a second percentage from the same logs.

**Q11. You inherit a deployed system whose outputs changed over a weekend with no
internal change. List three causes and the check for each.**

*Answer.* (a) A managed endpoint updated under a floating alias: check the resolved
model version recorded in response logs; a change identifies the cause. (b) A
retrieval index was rebuilt and ranking shifted: compare a fixed probe query's ranked
output and the index version identifier across the weekend. (c) An upstream source
changed schema or stalled: check source freshness timestamps and schema versions. The
durable fix is version pinning with a change record, index-version snapshots with each
evaluation, and freshness monitoring with staleness alerts.

*Reasoning credit.* Naming the instrumentation that would have caught it, not only the
cause.

*Follow-up.* "Which of the three is most dangerous?" (b), because a ranking shift
degrades groundedness silently and is invisible in latency and error metrics.

*Red flag.* Proposing to "roll back" a system that was never pinned.

**Q12. A colleague proposes a weighted score to select a vendor: 40% capability, 30%
cost, 20% control, 10% risk. What is wrong, and what would you do instead?**

*Answer.* The weights are exchange rates between quantities that are not
commensurable, and they silently convert a constraint (control) into a preference. A
vendor scoring 8/10 on control but failing a hard boundary can still win on the
weighted total. Instead, apply the constraints lexicographically: remove any option
that violates a hard boundary, then compare the remainder on cost per accepted output
and on capability under one frozen protocol. If a residual trade-off remains, state it
explicitly with the two owners who own the two sides.

*Reasoning credit.* Identifying the constraint-as-penalty error.

*Follow-up.* "When is weighting legitimate?" When all criteria are genuine preferences
with an owner who can articulate the exchange rate, and when no criterion is a
compliance or safety constraint.

*Red flag.* Accepting the weighting and proposing better weights.

**Q13. The proposal must handle a workflow where the source data cannot leave a
controlled environment, but internal operating capability is thin. What are the three
legitimate resolutions?**

*Answer.* Narrow the scope so a compliant option becomes affordable (fewer fields,
fewer languages, a smaller population, a lower frequency); change the boundary through
a recorded decision by the owner who holds it, with residual risk accepted in writing;
or retain the baseline and invest in the manual process. The illegitimate fourth is to
treat the boundary as a weighted penalty. If internal capability is thin, the
self-hosted option must include a funded operator, because an unoperated deployment
inside the boundary is a compliance success and an availability failure.

*Reasoning credit.* Including the baseline as a positive option and naming the operator
requirement.

*Follow-up.* "What if the boundary owner agrees to relax it?" Then the relaxation is
the decision, recorded with its residual risk and a revisit trigger; the technical
team documents rather than decides it.

*Red flag.* Choosing the cheapest compliant-looking option without an operator.

### 11.5 Two questions that separate senior candidates

**Q14. Give a case where a higher offline score is a reason *not* to deploy.**

*Answer.* When the score is achieved by abstaining on the hard cases and the fallback
on those cases is better than the system would have been. Formally, if `y_fb` on the
abstained subset exceeds `y`, coverage restriction lowers total correct outcomes.
Another case: when the score was achieved on a set that rewards style over entailment,
so fluency rises while groundedness falls; the score is measuring the wrong construct.
And a third: when the higher score comes with longer outputs that exhaust reviewer
attention, so measured acceptance in an evaluation setting does not predict field
acceptance.

*Reasoning credit.* Recognising that a metric can be improved by degrading the thing
it is supposed to measure.

*Follow-up.* "How would you detect the third case?" Measure review time and rejection
rate under field conditions, not in a lab where reviewers are fresh.

*Red flag.* Asserting scores are always informative if the test set is clean.

**Q15. You have 300 cases, a rubric, and three vendors. Design the comparison in five
minutes and state its principal weakness.**

*Answer.* Freeze the protocol before seeing results: case-selection rule (stratified
across the slices that matter, with the tail slices included), rubric with explicit
abstention handling, denominator defined (per document or per field, and the same for
all), two raters with adjudication on disagreements. Each vendor runs on the same
cases; raw outputs are retained so that disagreements can be inspected; 20 outputs per
vendor are re-scored blind to check the rubric's reliability. Principal weakness: 300
cases across, say, six slices gives 50 per slice, which supports a coarse ranking but
not fine distinctions, and the whole exercise measures output quality under
non-adversarial conditions rather than field behaviour.

*Reasoning credit.* Naming a specific weakness of the design rather than a generic
caveat.

*Follow-up.* "What would you add with more budget?" A shadow run on live traffic,
which addresses the population and the time-pressure gaps that no static set can.

*Red flag.* Proposing to "use the vendors' own benchmarks and normalise".

**Takeaway:** The questions that discriminate are the ones requiring an arithmetic
step and a population definition; answers that name frameworks and stop there are
indistinguishable from memorised vocabulary.

## 12. Quick Reference Card

| Cue | Rule | Use when | Critical assumption | Avoid when |
|---|---|---|---|---|
| Place first | Name the eligible unit, the decision, the action, and the owner before comparing anything | any "we need AI" proposal | the workflow is observable and someone owns it | the outcome cannot be described as a number that moves |
| Separate layers | Capability, delivery, operation, and accountability are independent axes | comparing vendors or modes | the task is stable enough to write a contract for | a brand name is standing in for a capability statement |
| Gate on control | Data route, credentials, and reversibility are pass/fail, never weighted | sensitive data, records, or payments | the deployed configuration is inspectable | the only evidence is a contract clause |
| Coverage pair | Report `c = n_served/N` beside every quality figure | any evaluation report | the eligibility rule is frozen and retries are collapsed | a single accuracy number is offered as the whole story |
| Total outcomes | `c*N*y + (1-c)*N*y_fb` before and after any change | a filter, threshold, or abstention rule changed | `y_fb` is measured, not assumed | the fallback's performance is unknown and the change is large |
| Value | `V = N(p_g*b - p_h*h - k) - F` relative to a measured baseline | early prioritisation | `b` excludes baseline value; costs are fully loaded | the result is presented as measured ROI |
| Harm leverage | Find `p_h*` at which the decision flips; fund measurement of `p_h` | harm is estimated rather than observed | exposure `h*N` is the slope that sets the window | the proposal quotes `V` without its break-even |
| Review load | `H = s*q*N/60`, with `s` measured on routed cases | any human review in the loop | routed cases are the hard ones | `s` came from the population mean |
| Stability | Staff to `rho <= 0.85`, not to the mean | designing a reviewer pool or service | arrivals vary and reviewers take leave | work is genuinely pooled and variability is negligible |
| Accepted cost | `C_acc = (kN + F + w(1-y)N)/(yN)` | pricing or business case | acceptance rubric frozen before measurement | the per-call price is being used as cost |
| Break-even | `N* = (F2-F1)/(k1-k2)`, then check for fixed-cost steps | build versus buy | both options serve one quality contract | the comparison is really about control, not cost |
| Authority grade | Claim no grade above the evidence row achieved | promotion decisions | evidence rows are ordered by difficulty | "we will monitor in production" substitutes for a gate |
| Baseline | Retain the baseline is a full option with observed costs | constraints conflict or evidence is missing | the baseline's cost is measured | the baseline appears only as a strawman |
| Reconcile | Outcomes plus abstentions must equal `N` | auditing any report | the window is defined once | the report's numbers do not add up and nobody asks |
| Pin | Record the resolved version and index identity in every response | any managed endpoint or rebuilt index | logs capture it | "latest" appears in a deployment manifest |

The card is read top to bottom as a sequence. Placement precedes separation, separation
precedes gating, gating precedes arithmetic, and arithmetic precedes any claim of
benefit. Skipping a row is the mechanism behind every failure in Section 8.
