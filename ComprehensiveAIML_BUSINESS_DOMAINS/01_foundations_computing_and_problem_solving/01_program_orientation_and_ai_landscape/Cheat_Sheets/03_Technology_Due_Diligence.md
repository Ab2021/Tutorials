---
collection: "program-orientation-ai-landscape"
file_id: "03"
topic: "technology due diligence questions"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Candidate version, model, data route, contract, price, workload, or gate threshold changes"
notation_style: "plain-text"
prerequisites: ["01_AI_Use_Case_Charter.md", "02_AI_Market_Map.md"]
pattern_tags: ["evidence", "acceptance-test", "vendor-review", "gate-design", "cost-model"]
source_status: "OPEN — framework references consulted; candidate-specific evidence must be obtained per decision"
owner_role: "technical due-diligence lead"
source_index: "INDEX.md"
---

# Technology Due Diligence: Claims, Tests, and Operating Boundaries

## 1. File Header

> **Purpose:** Convert a technology claim into a bounded acceptance protocol, a gate stack, and a versioned decision record before money or workflow authority is committed.
> **Prerequisites:** [AI Use-Case Charter](01_AI_Use_Case_Charter.md) for the eligible unit and baseline; [AI Market Map](02_AI_Market_Map.md) for the candidate categories; arithmetic with fractions, percentages, and currency.
> **Key outcome:** Given a claim such as "accurate," "secure," "cheaper," or "enterprise ready," you can write the operational interpretation, the test population, the pass and hard-gate thresholds, the cost model, the evidence ledger, and the decision consequence.
> **This file does NOT cover:** legal advice, certification, procurement negotiation tactics, or a verdict about any named supplier.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED gate arithmetic and interval logic; ILLUSTRATIVE candidate, protocol, and cost inputs; SOURCED framework vocabulary; OPEN candidate-specific artifacts.
> **Retrieval cue:** If a claim contains an adjective without a population, a threshold, or a consequence, start at Section 3.

**Source boundary.** The visible source inventory names containers that the author cannot open.
A directory name is an inventory fact, not a technical finding.
Nothing in this sheet reports a supplier's internal architecture, benchmark, incident, or contract term.
Every number below belongs to a constructed candidate called "the candidate," and every protocol is a teaching protocol.
The framework citations in Section 10 describe vocabulary and expected questions, not evidence about any product.
A real decision must attach its own artifacts, dates, version identifiers, and scoping statements.

### 1.1 The decision this sheet supports

Due diligence is not a mood of caution and it is not a feature comparison.
It is the discipline of answering three separable questions with different evidence.
First, can the technology perform the task on the population that will actually be routed to it?
Second, may the organization use it on those units under its data, authority, and contractual boundaries?
Third, can the organization operate, observe, afford, and exit the dependency after adoption?
A single answer cannot carry the other two.
A system that performs brilliantly but routes a prohibited field is not a viable candidate.
A compliant system that cannot be monitored or switched off is not operable.
An affordable system that fails the population it was bought for has no value.

The output of this discipline is not a score out of one hundred.
It is a claim ledger with a status per claim and a decision record naming an owner.
The ledger makes disagreement productive because it forces each party to name the artifact that would change their mind.
The decision record survives staff turnover, which a slide deck does not.

### 1.2 Reader task ladder

At the first level, a reader should be able to rewrite an adjective as an observable claim.
That means naming the unit, the population, the window, the metric, the comparator, and the consequence of failure.
At the second level, a reader should be able to design an acceptance protocol that could fail.
That means freezing cases before seeing outputs, separating a development set from a final set, and declaring the arithmetic of the gate.
At the third level, a reader should be able to price the candidate, size the operating envelope, and defend the limits of the evidence.
That includes interval reasoning, sensitivity to retries and volume, and an honest statement of what was never tested.
At the fourth level, a reader should be able to diagnose a weak diligence file.
That means noticing that a report names a different region, that a benchmark measured a different task, or that the exit plan has never been exercised.

### 1.3 Relationship to the neighboring sheets

The charter defines the eligible unit and the baseline against which any candidate is compared.
Without a frozen eligible unit, a pass rate and a cost per request are not comparable quantities.
The market map classifies candidate categories and their typical operational shapes.
It tells the reader what kind of dependency to expect, not which supplier to prefer.
This sheet sits between them: it takes a category-level candidate and produces a decision-grade protocol.
The three sheets together form a loop rather than a sequence.
A due-diligence finding can send the reader back to the charter to narrow the eligible population.
A market-map observation can send the reader back here to design a new gate for a new dependency type.

### 1.4 How the rest of the sheet is organized

Section 2 gives the mental model: claims flow into gates, gates into evidence, evidence into a decision.
Section 3 fixes the vocabulary, because most weak diligence files fail on definitions rather than on analysis.
Section 4 derives the arithmetic: proportions, intervals, allowed failures, total cost, readiness, expected loss, capacity, and sensitivity.
Section 5 runs one complete illustrative protocol end to end with every substitution shown.
Section 6 draws the geometry of an acceptance region with hard walls and soft preferences.
Section 7 compares evidence classes and candidate postures under matched conditions.
Section 8 catalogs twelve failure mechanisms with competing causes and discriminating checks.
Section 9 sizes the labor, latency, and storage cost of diligence itself.
Section 10 states the results that justify the arithmetic and their boundaries.
Section 11 answers fifteen interview questions with reasoning and red flags.
Section 12 compresses the method into a card that fits on one screen.

**Takeaway:** Due diligence is a decision contract with arithmetic, not a document that certifies enthusiasm.

## 2. Concept Map / Mental Model

```text
[Candidate claim: adjective + implicit scope]
                |
                v
[Operationalize: unit, population, window, metric, comparator]
                |
        +-------+--------+
        |                |
   undefined?       defined and falsifiable
        |                |
        v                v
[OPEN evidence      [Protocol: frozen cases, pinned version,
 request +           baseline arm, scorer, arithmetic]
 decision block]         |
                         v
                 [Gate stack]
                 |-- hard gate (authorization, data route, reversibility)
                 |-- scored gate (quality, latency, cost)
                 |-- evidence gate (artifact present, scoped, unexpired)
                         |
        +----------------+----------------+
        |                                 |
   any hard gate fails              all gates resolve
        |                                 |
        v                                 v
[reject / narrow / remediate]      [pilot with named owner,
                                     monitoring, and rollback]
                                          |
                                          v
                                   [Post-pilot re-review:
                                    drift, volume, price, version]
```

The map separates four things that procurement conversations routinely blend together.
A capability claim says the system can produce a kind of output.
A quality claim says the output meets a rubric at a rate on a population.
A control claim says something about who can see, change, or extract what.
An economic claim says the total cost compares favorably at a stated volume.
Each claim needs its own artifact and its own failure consequence.
Blending them produces the familiar sentence "the platform is accurate, secure, and cheaper," which cannot be tested as written.

### 2.1 The three independent verdicts

The first verdict is **fitness for the task**.
It answers whether the candidate, at a pinned version and configuration, produces acceptable outputs on units drawn from the intended population.
The second verdict is **permission**.
It answers whether the organization may send those units, in that form, through that path, under those terms, with that retention.
The third verdict is **operability**.
It answers whether a named team can run, watch, pay for, degrade, and leave the dependency.
A candidate can pass fitness and fail permission, which is common when a powerful hosted service is prohibited for a data class.
A candidate can pass permission and fail operability, which is common when nobody can explain how to roll back a prompt change.
A candidate can pass all three and still lose to the baseline, which is a legitimate and useful outcome.
Keeping the verdicts separate prevents a strong quality result from laundering a data-route problem.

### 2.2 Claim decomposition chain

Take the sentence "the assistant resolves requests accurately and cheaply" and decompose it.
"Resolves" implies a decision or an action; the action boundary must be named.
"Requests" implies an eligible unit and an exclusion rule.
"Accurately" implies a rubric, a scorer, and a threshold.
"Cheaply" implies a baseline, a volume, and a fully loaded cost definition.
The decomposition continues until each residual phrase can be answered by an artifact that exists today.
If a phrase can only be answered by a roadmap item, it stays OPEN and blocks the corresponding gate.
This is not pedantry; it is the difference between a pilot that produces a decision and a pilot that produces anecdotes.

A decomposition is complete when the following six slots are filled for each claim.
The unit slot names exactly one countable thing.
The population slot names where those units come from and which are excluded.
The window slot names the time period and whether it is historical or live.
The metric slot names the numerator, denominator, and scoring rule.
The comparator slot names the baseline or the threshold.
The consequence slot names what happens on pass, fail, and inconclusive.
A claim with an empty slot is not ready for a test; it is ready for a conversation.

### 2.3 Gate topology

Gates are not interchangeable, and their arithmetic differs.
A **hard gate** is a boolean constraint with no compensating weight.
Examples include a prohibited data route, an unauthorized action, or the absence of an audit trail for a reversible action.
A hard gate is evaluated as pass or fail, and one failure blocks promotion regardless of quality or price.
A **scored gate** compares a measured quantity to a threshold.
Quality, latency, and total cost are usually scored gates because they have tolerances.
An **evidence gate** checks that a required artifact exists, is scoped to the actual candidate, and has not expired.
Evidence gates exist because absence of proof is not proof of absence, and because a report about a sibling product says nothing about this deployment.
Finally, a **monitoring gate** checks that the organization can detect drift or failure after launch.
That gate is often the one nobody writes down, and it is the one that determines whether a pilot becomes an incident.

The topology matters when the organization reports a single number.
A weighted score of 0.86 can hide a failed hard gate, and a dashboard that shows only the score will be read as approval.
The correct presentation is a gate status line per constraint plus a scored summary inside the permitted region.
Section 6 draws this as an acceptance region bounded by walls.

### 2.4 Lifecycle of a diligence record

The record begins with a claim intake, which captures the exact wording and the person who made it.
It continues with an operationalization step, which rewrites the claim into the six slots above.
It then freezes a protocol, including the case source, the scorer, the arithmetic, and the decision rule.
Execution produces outputs and scores, which are stored with the pinned version and configuration.
Adjudication resolves scorer disagreements and records the rate at which they occurred.
The decision step records adopt, pilot, retain baseline, or collect more evidence, with an owner and a date.
The maintenance step attaches expiry triggers to each claim so that a later change reopens the right rows.
The retirement step records why a candidate was dropped, because that knowledge prevents re-litigating the same question.
A record without a retirement step tends to resurrect rejected candidates whenever a new sponsor arrives.

### 2.5 What due diligence is not

It is not a certification, and it cannot make a regulator or an auditor sign anything.
It is not a proof of universal safety; it is a bounded statement about tested conditions.
It is not a substitute for engineering ownership after launch.
It is not a negotiation in which the organization extracts the most favorable wording.
It is not a one-time event, because versions, prices, subprocessors, and workloads change.
It is not a ranking exercise across vendors on incomparable evidence.
It is not a reason to delay a reversible, low-consequence decision indefinitely.
Proportionality is part of the method: the depth of the protocol should track the consequence and reversibility of the decision.

### 2.6 Proportionality ladder

A low-consequence, reversible use, such as drafting an internal summary that a human rewrites, needs a small protocol.
That protocol might be twenty representative cases, a named reviewer, a cost ceiling, and a data-route check.
A medium-consequence use, such as routing customer requests to a queue, needs a larger frozen set, adversarial cases, and a rollback test.
A high-consequence, irreversible use, such as an automated financial posting, needs dual control, idempotency proof, audit reconstruction, and a formal sign-off.
The ladder is not a substitute for the hard gates; it scales the depth of scored evidence.
Writing the ladder down prevents two opposite errors.
The first error is treating every decision as existential, which stalls useful work.
The second error is treating every decision as reversible, which produces incidents that cannot be undone.

**Takeaway:** Split fitness, permission, and operability into separate verdicts, then let only hard gates veto.

## 3. Foundational Definitions

| Term | Operational meaning | Artifact that resolves it | Frequent misuse |
|---|---|---|---|
| Claim | A falsifiable statement with an owner, version, and scope | exact wording plus candidate version identifier | a marketing category treated as a fact |
| Eligible unit | One countable item that the workflow may route to the candidate | inclusion and exclusion rule with examples | "all user questions" without a boundary |
| Population | The source distribution from which tested units were drawn | sampling frame, date range, strata counts | a convenient sample described as representative |
| Acceptance criterion | A predeclared pass, fail, or escalate rule | protocol signed before execution | a threshold chosen after seeing results |
| Baseline | The current process measured on the same units and window | configuration, sample, and measured outputs | the weakest imaginable alternative |
| Hard gate | A boolean constraint with no compensating weight | control test, flow map, or signed scope statement | a weighted preference labeled mandatory |
| Scored gate | A measured quantity compared to a declared threshold | scored outputs plus arithmetic | a threshold applied to a rounded average |
| Evidence artifact | An inspectable record that supports a claim | report, log, test output, contract clause, ticket | a verbal assurance in a meeting |
| Scope of evidence | The exact product, version, region, and configuration an artifact covers | artifact-to-deployment mapping | assuming a certificate covers a derivative service |
| Evidence expiry | The condition under which an artifact stops supporting the claim | dependency list with change triggers | a calendar date with no dependency link |
| Reproducibility | Another reviewer reaches the same material conclusion from the same inputs | frozen cases, config, scorer, and run record | a screenshot of one favorable output |
| Data route | The set of components that receive, store, transform, or log a field | field-level flow map plus observed configuration | a country label or a trust badge |
| Reversibility | Whether an action can be undone without external loss | rollback test with measured recovery | a design document that asserts undo exists |
| False acceptance | Approving a candidate that would fail the true requirement | protocol design and error analysis | any single failed test case |
| False rejection | Retaining the baseline when the candidate would have been adequate | baseline measurement and power analysis | treated as a harmless default |
| Drift | Change in inputs, population, or behavior after the protocol window | monitoring series with a reference period | a single post-launch spot check |

The table is deliberately redundant with the prose, because definitions are where most diligence arguments actually fail.
Two teams can agree that a system is accurate while disagreeing about whether abstentions count as correct.
They can agree that a system is cheap while disagreeing about whether reviewer time is inside the cost.
They can agree that a system is compliant while disagreeing about which region actually processed the request.
The definition table converts those disagreements into questions with artifacts.

### 3.1 Four claim classes and their native evidence

**Capability claims** assert that a kind of input can be transformed into a kind of output.
Their native evidence is a versioned input-output trace showing the contract, including malformed inputs and refusals.
A capability claim is falsified by a single input inside the declared contract that produces no usable output.
Capability evidence is usually cheap to obtain and therefore rarely the binding constraint.

**Quality claims** assert a rate of acceptable outputs on a population.
Their native evidence is a frozen case set with a rubric, a scorer, and a count.
Quality claims are the ones most often supported by public benchmarks, and that substitution is usually invalid for reasons detailed in Section 8.
Quality evidence is expensive because it requires representative cases and human judgment.

**Control claims** assert something about access, retention, routing, deletion, or action authority.
Their native evidence is a configuration extract, an access test, a field-level flow map, or a contractual clause that is specific enough to enforce.
Control claims cannot be settled by a quality score, and a strong quality score is frequently used to distract from them.

**Economic claims** assert a cost or benefit relationship at a volume.
Their native evidence is a dated price schedule, a measured usage distribution, and a cost model that includes the organization's own labor.
Economic claims are the easiest to state and the easiest to get wrong, because retries, output length, and review time are invisible in a unit price.

### 3.2 Evidence strength ladder

Evidence is not binary; it has a strength ordering that depends on the claim.
For a quality claim, a frozen local protocol with an independent scorer is stronger than a vendor-run evaluation, which is stronger than a public benchmark, which is stronger than a case study, which is stronger than a demo.
For a control claim, an observed configuration test is stronger than an auditor report scoped to the same version, which is stronger than a contractual commitment, which is stronger than a security page.
For an economic claim, a reconciled invoice for a comparable workload is stronger than a dated quote with a usage model, which is stronger than a list price, which is stronger than a savings estimate in a slide.
The ordering is claim-specific, and mixing the orders produces nonsense.
A strong contract is weak quality evidence, and a strong benchmark is weak control evidence.

### 3.3 Scope, expiry, and dependency

Every artifact has a scope: the product, plan tier, region, version, and configuration it describes.
Every artifact has a date, and the date matters only insofar as it precedes a relevant change.
Every artifact has dependencies: the model version, the retrieval corpus, the identity provider, the subprocessor list, the price schedule.
Expiry is therefore a function of dependency change, not of age alone.
A penetration test from eighteen months ago may still be informative if nothing in the tested boundary changed.
A quality report from last week may already be stale if the model was silently updated.
The ledger records dependencies explicitly so that a change event can reopen the correct rows.

### 3.4 Gate vocabulary

A **threshold** is the numeric boundary of a scored gate.
A **tolerance** is the amount of measurement noise the gate must absorb before it can be trusted.
A **disqualifier** is a condition that ends evaluation without further scoring.
An **escalation** is a defined route for a decision that exceeds the evaluator's authority.
A **residual risk** is the harm that remains after controls, accepted by a named owner.
A **compensating control** is a measure that reduces but does not remove a constraint failure, and it never converts a hard gate into a pass.
A **stop condition** is the predeclared event that halts a pilot.
These terms appear in every serious review, and using them loosely causes a review to end with an ambiguous "let us proceed carefully."

### 3.5 Roles

The **decision owner** is accountable for the outcome and signs the record.
The **technical evaluator** designs and executes the protocol and does not own the business outcome.
The **risk or control partner** judges whether the data route and authority boundary are acceptable.
The **operating owner** commits to monitoring, incident response, and rollback after launch.
The **scorer** applies the rubric and should not know which arm produced which output where blinding is feasible.
The **evidence custodian** maintains the ledger, versions artifacts, and records expiry triggers.
In small organizations one person may hold several roles, but the record should still name them separately.
When the same person designs the protocol, scores the outputs, and approves the decision, the file gains a structural bias that no amount of diligence prose repairs.

### 3.6 Adjective translation table

| Adjective | Observable substitute | What must be supplied | Typical gap |
|---|---|---|---|
| accurate | pass fraction on a frozen rubric | population, rubric, threshold, abstention rule | no local cases |
| reliable | successful served fraction over a window | numerator, denominator, window, retry policy | uptime without user impact |
| secure | scoped control evidence for the deployed configuration | threat scope, identities, data classes, routes | certificate for a sibling product |
| scalable | behavior at a stated peak rate with measured latency | request rate, payload, concurrency, quota | demo at trivial volume |
| cheap | fully loaded cost per eligible unit at a volume | baseline, retries, review, egress, exit | unit price alone |
| explainable | an artifact a reviewer can inspect for a sampled decision | audience, required fields, sampling rate | a generic rationale string |
| compliant | a mapped control set with scope and date | applicable rule, control mapping, evidence | a badge |
| enterprise ready | a named operator, SLO, release process, and exit test | responsibilities, escalation, recovery | a roadmap |
| integrated | a tested interface contract with failure behavior | schema, auth, error codes, retry semantics | a screenshot |
| fair | slice-level metric differences with thresholds | protected or operational slices, sample sizes | a single aggregate metric |

The table is a working tool rather than a reference.
A reviewer can read a vendor deck and mark every adjective with the artifact that would resolve it.
What remains unmarked is the actual scope of the conversation.
What remains unresolved after the vendor responds is the actual risk.

**Takeaway:** A definition that cannot name its artifact is a preference, and preferences cannot gate a decision.

## 4. Core Formulas with Full Derivation

Every result in this section exists to convert a judgement into arithmetic that another reviewer can recompute.
The arithmetic never proves that a technology is good; it proves what the tested evidence supports.
Three conventions hold throughout.
Counts are integers and fractions are exact before any display rounding.
Currency amounts are stated per month unless a window is explicit.
All numeric inputs are ILLUSTRATIVE constructions built for teaching, never measured supplier data.

### 4.1 Base: measured pass fraction and its discrete gate

**Validity before use.** The case set is closed before outputs are inspected.
Each case has an expected-outcome record written by someone other than the person who wrote the candidate prompt.
Each case is scored exactly once against a declared rubric.
The candidate version and configuration are pinned for the entire run.
The run has no missing cases; a case that cannot be scored is a protocol failure, not a silent exclusion.

**Derivation.** Let case `i` have a binary rubric outcome `y_i = 1` if it satisfies every rubric component and `0` otherwise.
The measured pass fraction is the sample mean of those indicators:

```text
p_hat = (1/n) * sum_{i=1..n} y_i = n_pass / n
```

The step from the sum to the count is exact because the indicators are binary.
This is a descriptive statistic about the tested cases; it is not a claim about the population unless the cases are a defensible sample of it.
The discrete gate follows by solving the inequality for the integer failure count.
If the gate requires `p_hat >= p_min` and `p_hat = (n - n_fail)/n`, then `n - n_fail >= n*p_min`, so `n_fail <= n*(1 - p_min)`, and since `n_fail` is an integer, `n_fail <= floor(n*(1-p_min))`.
The floor is the whole content of the discrete gate; rounding the threshold instead of the count is a common off-by-one error.

```text
FORMULA [DD-01]: p_hat = n_pass / n_test ;  gate: n_fail <= floor(n_test*(1 - p_min))
WHERE: p_hat measured pass fraction [dimensionless, 0..1];
       n_pass cases satisfying every rubric component [count];
       n_test scored cases [count, n_test >= 1];
       p_min predeclared minimum pass fraction [dimensionless];
       n_fail = n_test - n_pass [count].
CONSTRAINT: binary rubric with one score per case; frozen cases; no post-hoc removal;
            this is a test-set proportion, not a population reliability claim.
NUMERICAL EXAMPLE: n_pass = 91, n_test = 100  ->  p_hat = 0.91 = 91.0%.
                  Gate p_min = 0.90  ->  allowed n_fail = floor(100*0.10) = 10.
                  Observed n_fail = 9  ->  9 <= 10, gate passes.
INTERPRETATION: the evidence supports "91 of these 100 frozen cases passed the rubric";
                it does not support "91% reliable on all future traffic".
```

**Arithmetic check.** `91 + 9 = 100` reconstructs the denominator, and `9/100 = 0.09 <= 0.10` matches the gate.
**Boundary.** With `n_test = 7` and `p_min = 0.90`, `floor(0.70) = 0`, so a single failure fails the gate even though `6/7 = 85.7%` looks close.
That is correct behaviour for a declared rule, but it shows why small test sets cannot carry a precise-sounding threshold.

### 4.2 General: fully loaded total cost of ownership

**Validity before use.** All components cover the same window and currency.
The request volume is a defensible demand estimate for that window, including peak shape if capacity is in scope.
Every material cost is assigned to exactly one term; a supporting vendor fee that is already inside the subscription charge must not appear again in labor.
Retry and failure behaviour is priced rather than assumed away.

**Derivation.** Start from the accounting identity that total cost is the sum of cost categories over the window.
The fixed term `F` covers charges incurred regardless of volume: subscription floor, integration build amortization, certification work.
The variable term is the product of eligible requests and effective cost per request.
Effective cost per request is larger than the nominal unit price because a fraction `r` of requests is retried or reprocessed, so `k_eff = k*(1 + r)`.
Expanding `N*k_eff` gives `N*k + N*k*r`, which separates first-pass usage from waste.
The labor term is hours multiplied by a loaded rate, where loaded means salary plus employer cost plus tooling, not an hourly wage.
The reserve term `R` covers expected remediation, incident response, and contingency, and it is an estimate with a stated basis.
Summing yields the total; dividing by eligible requests yields a cost per unit that can be compared with a baseline on identical units.

```text
FORMULA [DD-02]: TCO = F + N*k*(1 + r) + H*w + R ;  c_unit = TCO / N_eligible
WHERE: F fixed cost [currency/window];
       N requests processed [requests/window];
       k nominal variable cost per first attempt [currency/request];
       r retry/reprocess fraction [dimensionless, 0..1];
       H human review and support hours [hours/window];
       w loaded labor rate [currency/hour];
       R risk and reserve estimate [currency/window];
       N_eligible eligible units [units/window]; c_unit [currency/unit].
CONSTRAINT: disjoint cost categories; same window and currency for every term;
            capacity-driven step costs are excluded unless explicitly modeled.
NUMERICAL EXAMPLE: F = 2,000; N = 20,000; k = 0.08; r = 0;
                  H = 30; w = 60; R = 500.
                  variable = 20,000*0.08*1.00 = 1,600
                  labor    = 30*60 = 1,800
                  TCO = 2,000 + 1,600 + 1,800 + 500 = 5,900 per month
                  c_unit = 5,900/20,000 = 0.295 per request.
INTERPRETATION: any adoption decision must show at least 5,900 per month of
                comparable value at this volume, before harm or transition cost.
```

**Arithmetic check.** `2,000 + 1,600 = 3,600`; `3,600 + 1,800 = 5,400`; `5,400 + 500 = 5,900`; `5,900/20,000 = 0.295`.
**Sensitivity.** If `r = 1.0` (every request is retried once), the variable term becomes `20,000*0.08*2 = 3,200`, and TCO rises by `1,600` to `7,500`.
If the nominal price rises 10% to `0.088`, the variable term is `1,760` and TCO rises by `160` to `6,060`.
The retry term dominates the price term in this example, which is why retry behaviour is a first-class test item rather than an operational detail.

### 4.3 Variants

#### 4.3.1 Variant: integer allowed-failure gate on a small set

**Validity before use.** The gate is administrative and must be declared before execution.
It is not a confidence statement and it does not generalize to the population.

**Derivation.** Same inequality as DD-01, applied with a small denominator where the floor bites.
For `n_test = 47` and `p_min = 0.90`, `47*0.10 = 4.7`, so at most four failures are allowed.
Passing with four failures gives `43/47 = 0.914894`, and five failures gives `42/47 = 0.893617`.
The gap between 4.7 and the integer 4 is the entire margin; there is no room for a fifth failure.

```text
FORMULA [DD-03]: n_fail_max = floor(n_test*(1 - p_min))
WHERE: n_fail_max maximum tolerated failures [count]; n_test [count]; p_min [fraction].
CONSTRAINT: n_test fixed before execution; rubric binary; no fractional cases.
NUMERICAL EXAMPLE: n_test = 47, p_min = 0.90 -> 47*0.10 = 4.7 -> n_fail_max = 4.
                  43/47 = 0.9149 passes; 42/47 = 0.8936 fails.
INTERPRETATION: declare "4 or fewer failures of 47" in the protocol rather than
                "at least 90%", so both parties compute the same outcome.
```

**Boundary.** Setting `p_min` to a value that is not attainable by any integer count is a protocol defect.
For `n_test = 47`, the attainable pass fractions near the gate are `42/47 = 89.36%` and `43/47 = 91.49%`; there is no outcome at exactly 90%.

#### 4.3.2 Variant: interval around a measured proportion

**Validity before use.** Cases are independent draws from a population the reviewer is willing to treat as the target.
Scoring is reliable; scorer disagreement is measured separately.
The interval describes sampling uncertainty only; it does not cover version drift, population shift, or rubric error.

**Derivation.** The Wald interval applies the central limit theorem to the sample proportion: the standard error is `sqrt(p*(1-p)/n)`, and the interval is `p ± z*sqrt(p(1-p)/n)` at confidence level determined by `z`.
The Wald form collapses when `p` approaches 0 or 1 and behaves badly for small `n`.
The Wilson form solves the score-test inequality for the proportion and is better behaved; it recentres the estimate and inflates the width near the boundaries.
For the lower-bound question, only the lower endpoint matters; a gate that requires "at least 90%" should be tested against the lower endpoint when the decision is consequential.

```text
FORMULA [DD-04]: SE = sqrt(p_hat*(1 - p_hat)/n) ;  CI_wald = p_hat +/- z*SE
WHERE: SE standard error of the proportion [dimensionless];
       p_hat measured fraction; n cases; z standard normal quantile
       (z = 1.959964 for 95% two-sided).
CONSTRAINT: independent cases, adequate expected counts (roughly n*p_hat >= 10
            and n*(1-p_hat) >= 10 is a working rule), fixed n.
NUMERICAL EXAMPLE: p_hat = 0.91, n = 100, z = 1.96.
                  p_hat*(1-p_hat) = 0.91*0.09 = 0.0819
                  SE = sqrt(0.0819/100) = sqrt(0.000819) = 0.02862
                  half-width = 1.96*0.02862 = 0.05609
                  CI_wald = [0.8539, 0.9661]  ->  [85.4%, 96.6%].
INTERPRETATION: a 91% point estimate is compatible with a true rate as low as
                about 85% at this sample size, so a 90% gate is not clearly met.
```

**Arithmetic check.** `0.05609*2 = 0.11218`, and `0.91 - 0.05609 = 0.85391`, `0.91 + 0.05609 = 0.96609`.
**Boundary.** Recomputing with the Wilson form gives approximately `[0.838, 0.952]`, which is asymmetric and lower at the bottom.
Either way the lower endpoint sits below 0.90, so the decision "gate met" is not supported by this sample alone.

#### 4.3.3 Variant: sample size needed for a stated precision

**Validity before use.** A planning value of `p` is available from a pilot or a prior.
The target half-width `h` is a decision tolerance, not an aesthetic preference.
The independence assumption is at least approximately defensible.

**Derivation.** Invert the Wald half-width `h = z*sqrt(p(1-p)/n)` and solve for `n`:

```text
n >= z^2 * p*(1-p) / h^2
```

The formula is a planning approximation and understates `n` when `p` is near the boundaries; the conservative choice `p = 0.5` maximizes the product.

```text
FORMULA [DD-05]: n_min = ceil(z^2 * p*(1-p) / h^2)
WHERE: n_min required cases [count]; h target half-width [fraction];
       p planning proportion [fraction]; z normal quantile.
CONSTRAINT: simple random sampling; no clustering; no scoring error.
NUMERICAL EXAMPLE: p = 0.91, h = 0.05, z = 1.96.
                  z^2 = 3.8416; p*(1-p) = 0.0819
                  numerator = 3.8416*0.0819 = 0.31463
                  n_min = ceil(0.31463/0.0025) = ceil(125.85) = 126 cases.
INTERPRETATION: 100 cases cannot support a +/-5 point claim at 91%; the protocol
                needs about 126 scored cases, which costs 26% more review time.
```

**Arithmetic check.** `0.31463/0.0025 = 125.852`; `126*0.0025 = 0.315`.
**Boundary.** If the cases are clustered by customer, the effective sample size is smaller than the count, and the formula becomes optimistic.
If the rubric has three components and any failure fails the case, the effective precision of the composite is worse than the per-component precision suggests.

#### 4.3.4 Variant: rule of three for zero observed failures

**Validity before use.** Zero failures observed in `n` independent trials.
The quantity of interest is an upper bound on the failure rate, not a lower bound on quality.

**Derivation.** With true failure probability `q`, the chance of observing zero failures is `(1-q)^n`.
Setting that to `0.05` and solving gives `1-q = 0.05^(1/n)`, so `q = 1 - 0.05^(1/n)`.
For large `n`, `1 - exp(ln(0.05)/n) ≈ -ln(0.05)/n = 2.996/n ≈ 3/n`.
This is the standard "rule of three" and it is an upper confidence bound, not an estimate.

```text
FORMULA [DD-06]: q_upper = 1 - 0.05^(1/n)  ~=  3/n for large n
WHERE: q_upper 95% upper bound on the failure probability [fraction];
       n trials with zero failures [count].
CONSTRAINT: independence; identical failure probability across trials;
            zero observed failures; one-sided 95% level.
NUMERICAL EXAMPLE: n = 200, zero prohibited-data disclosures.
                  q_upper = 1 - 0.05^(1/200) = 1 - exp(ln(0.05)/200)
                          = 1 - exp(-2.9957/200) = 1 - exp(-0.014979)
                          = 1 - 0.985132 = 0.014868  ->  1.49%.
INTERPRETATION: "no leaks in 200 trials" is compatible with a true leak rate up to
                about 1.5% at 95% confidence, which at 20,000 requests/month is
                up to about 300 events/month in expectation.
```

**Arithmetic check.** `0.014868*200 = 2.974`, close to the `3/n` approximation, and `0.985132^200 = exp(200*ln(0.985132)) = exp(-2.9957) = 0.05`.
**Boundary.** The bound is vacuous for a hard gate in the sense that it never reaches zero.
A hard gate must therefore be stated as "no observed violation, plus these controls," not as "zero risk".

#### 4.3.5 Variant: expected loss versus expected saving

**Validity before use.** A harm magnitude per failure event is available, with a stated estimation basis.
The failure rate is the one the gate is willing to use, preferably the conservative bound rather than the point estimate.
Failures are counted on the same unit basis as the volume.

**Derivation.** Expected loss is the product of event probability and loss per event, summed over failure types.
If the candidate displaces a baseline with cost `C_base`, the net expected value is `C_base - (TCO + E[L])`.
Adoption is justified on economic grounds only when that difference is positive with margin for estimation error.
The linearity used here is exact for expectations; the variance is not linear, so a positive mean does not guarantee a safe month.

```text
FORMULA [DD-07]: E[L] = N * q * C_harm ;  V_net = C_base - TCO - E[L]
WHERE: E[L] expected harm cost [currency/window];
       N requests [requests/window]; q failure probability [fraction];
       C_harm expected cost per failure event [currency/event];
       C_base baseline cost on the same units [currency/window];
       V_net net expected value [currency/window].
CONSTRAINT: single failure class with a defensible average harm; independent events;
            baseline measured on identical units and window.
NUMERICAL EXAMPLE: N = 20,000; q = 0.08; C_harm = 40.
                  E[L] = 20,000*0.08*40 = 64,000 per month.
                  With TCO = 5,900 and C_base = 33,600:
                  V_net = 33,600 - 5,900 - 64,000 = -36,300 per month.
INTERPRETATION: at these inputs the candidate destroys value even though its direct
                cost is a fraction of the baseline, because harm dwarfs the saving.
```

**Arithmetic check.** `20,000*0.08 = 1,600` events; `1,600*40 = 64,000`.
**Sensitivity.** If `C_harm` falls to `1.50`, `E[L] = 2,400` and `V_net = 33,600 - 5,900 - 2,400 = +25,300`.
The decision flips entirely on the harm estimate, which is exactly why that estimate needs a stated basis and a range rather than a single confident number.

#### 4.3.6 Variant: evidence readiness with hard-gate override

**Validity before use.** The required-artifact list is written before evidence is requested.
Each artifact is either resolved and scoped to the candidate configuration, or unresolved.
Resolution is a binary judgement recorded with the reviewer and date.

**Derivation.** Readiness is a proportion of resolved required artifacts: resolved count over required count.
The proportion is an inventory measure and has no weighting authority.
The gate is therefore a conjunction: readiness above some administrative level, and every hard-gate artifact resolved.
The conjunction is deliberately non-compensatory; a conjunction cannot be raised by adding easy artifacts.

```text
FORMULA [DD-08]: E_ready = n_resolved / n_required ;  promote_ok = (E_ready >= e_min) AND (all hard-gate artifacts resolved)
WHERE: n_resolved artifacts resolved and scoped [count]; n_required [count];
       e_min administrative readiness level [fraction]; hard-gate artifacts are
       those whose absence alone blocks promotion.
CONSTRAINT: artifact list frozen before requests; "resolved" includes scope and date;
            no substitution of a sibling product's artifact.
NUMERICAL EXAMPLE: n_required = 8, n_resolved = 7 -> E_ready = 0.875 = 87.5%.
                  The unresolved artifact is the field-level data-route test.
                  promote_ok = (0.875 >= 0.80) AND false = false.
INTERPRETATION: an 87.5% readiness figure cannot unlock promotion while the missing
                artifact is the one that would resolve a prohibited route.
```

**Arithmetic check.** `0.875*8 = 7`.
**Boundary.** If the unresolved artifact is instead a convenience document, promotion may proceed with a recorded follow-up date; the conjunction still holds because only hard-gate artifacts veto.

#### 4.3.7 Variant: concurrency and quota sizing

**Validity before use.** A stable arrival rate over the sizing window.
Service time is the observed mean, not the best-case latency.
Requests are not batched in a way that changes the concurrency requirement.

**Derivation.** Little's law states that the long-run average number in system equals arrival rate times average time in system, `L = lambda*W`.
For a service with `c` parallel workers, each request occupies one worker for its service time, so required concurrency is `lambda*W` at the mean.
Real systems need headroom for variance, so the provisioned concurrency is the mean multiplied by a safety factor.
Converting requests per minute to per second requires dividing by 60.

```text
FORMULA [DD-09]: L = lambda*W ;  c_prov = ceil(s * lambda*W)
WHERE: lambda arrival rate [requests/s]; W mean time in system [s];
       L average concurrency [requests]; s safety factor [dimensionless, >1];
       c_prov provisioned concurrent capacity [workers].
CONSTRAINT: stationary arrivals, stable service time, unbounded queue assumption
            for the mean calculation; tail latency not modeled.
NUMERICAL EXAMPLE: peak 40 requests/min -> lambda = 40/60 = 0.6667 requests/s;
                  W = 3.2 s; s = 2.0.
                  L = 0.6667*3.2 = 2.133 requests;
                  c_prov = ceil(2.0*2.133) = ceil(4.267) = 5 workers.
INTERPRETATION: five concurrent workers cover the peak mean with a 2x factor, but
                this says nothing about the 99th-percentile wait under burstiness.
```

**Arithmetic check.** `0.6667*3.2 = 2.13344`; doubled `4.26688`; ceiling `5`.
**Boundary.** With a burst of 200 requests in 10 seconds, `lambda = 20/s` and `L = 64` at the same service time; the provisioned five workers queue massively.
Sizing must use the burst rate when the workflow has bursts, not the monthly average.

#### 4.3.8 Variant: break-even volume under fixed cost

**Validity before use.** Fixed and per-unit costs are separable and linear in volume within the range considered.
The baseline per-unit cost is measured on the same unit definition.

**Derivation.** Set candidate total equal to baseline total: `F + N*k_eff + H*w = N*c_base`.
Rearrange to `F + H*w = N*(c_base - k_eff)`, then divide by the per-unit saving.
If the per-unit saving is not positive, no volume reaches break-even and the fixed cost is unrecoverable through scale.

```text
FORMULA [DD-10]: N_be = (F + H*w) / (c_base - k_eff)
WHERE: N_be break-even volume [units/window]; c_base baseline cost per unit
       [currency/unit]; k_eff effective candidate variable cost [currency/unit];
       F fixed cost [currency/window]; H*w labor [currency/window].
CONSTRAINT: linear cost behaviour; positive denominator; identical unit definitions.
NUMERICAL EXAMPLE: F = 2,000; H*w = 1,800; c_base = 1.68; k_eff = 0.08.
                  denominator = 1.68 - 0.08 = 1.60
                  N_be = 3,800/1.60 = 2,375 units per month.
INTERPRETATION: below about 2,375 eligible units per month the fixed cost is not
                recovered by per-unit savings; the candidate is a scale bet.
```

**Arithmetic check.** `2,375*1.60 = 3,800`, and `2,375*1.68 = 3,990` versus `2,000 + 2,375*0.08 + 1,800 = 3,990`.
**Boundary.** If `c_base = 0.07` and `k_eff = 0.08`, the denominator is negative and the formula returns a negative volume; the correct reading is "never breaks even", not a large volume.

### 4.4 Limits and counterexamples

**The average cannot override a wall.**
Consider a candidate that passes 99 of 100 quality cases and performs one unauthorized irreversible action.
`p_hat = 0.99`, comfortably above any reasonable quality threshold.
The hard gate fails, and the correct decision is rejection or restriction to propose-only mode.
Any scoring scheme that would let the 99 passes offset the single unauthorized action has a design defect, because the harm classes are not exchangeable.
The mathematical reason is that the loss function is not a weighted average of the two dimensions; it is a constraint set with an interior objective.

**Aggregation can reverse a conclusion.**
Suppose two populations, A with 1,000 cases and B with 100 cases.
The candidate passes 88% of A and 96% of B, and the baseline passes 90% of A and 94% of B.
Aggregated: candidate `(880 + 96)/1,100 = 976/1,100 = 88.7%`; baseline `(900 + 94)/1,100 = 994/1,100 = 90.4%`.
The candidate loses overall and loses within A, but wins within B.
This is an instance of aggregation reversal, and it means a per-slice gate is required when slices matter operationally.
The reverse pattern, where the candidate wins overall and loses in every slice, is possible when the slice mix differs and is a classic diligence trap.

**A precise percentage can rest on a tiny denominator.**
Reporting "100% success" from four cases is arithmetically true and evidentially near-empty: from DD-06, four zero-failure trials bound the failure rate only at about `1 - 0.05^(1/4) = 1 - 0.4729 = 52.7%`.
The correct interpretation is "no failures observed in four cases", not "no failures occur".
This is why the protocol must state the denominator and why readiness dashboards that show only percentages mislead.

**Linearity fails at quota cliffs.**
A cost model that is linear at 20,000 requests can be wrong at 120,000 if the plan tier changes, rate limits force a second contract, or the provider throttles.
The model is valid inside the range where the price schedule is linear; outside it, the schedule must be re-read.

**Independence fails under repetition.**
If the same 100 cases are used repeatedly to tune prompts, the cases stop being independent evidence about unseen traffic.
The pass rate rises while the population-level expectation does not, which is the mechanism behind overfitting to a benchmark.
The fix is procedural: a final set that no one tunes against, with access recorded.

**Counterexample to "more data always helps".**
Adding cases drawn from a distribution the candidate will never see increases `n` and narrows the interval around a quantity nobody cares about.
The decision-relevant question is representativeness, not volume.
A thousand irrelevant cases are less informative than sixty well-chosen boundary cases plus twenty adversarial cases.

**Counterexample to "a failed pilot means the technology is bad".**
A pilot can fail because the eligible-unit definition included out-of-scope requests, because the retriever lacked the authoritative corpus, or because reviewers applied an inconsistent rubric.
Each of those is a fixable protocol fault, and attributing them to the technology prevents a correct diagnosis.
The diligence record must separate candidate capability, protocol quality, and integration quality.

**Takeaway:** State the gate arithmetic before the run, keep hard constraints outside the average, and report the denominator with every percentage.

## 5. Worked Numerical Example

This section runs one complete protocol from claim intake to decision record.
Every input is ILLUSTRATIVE and constructed for teaching; none is a measured supplier result.
The arithmetic is checked at each step so that a reader can reproduce it on paper.

### 5.1 Intake

The claim arrives as a single sentence from a business sponsor: "The assistant resolves incoming support knowledge requests accurately, securely, and cheaper than today's process."
Three adjectives are packed into one claim, and each one needs its own operationalization.
The sponsor also supplies three assertions that are not evidence but shape the protocol: the vendor is a well-known provider, the product has a public security page, and the price is "a few cents per request."
The diligence lead records the claim verbatim with the sponsor, date, and product name, and refuses to translate any adjective without agreement on the definition.
This intake record is the first artifact in the ledger.

### 5.2 Operationalization

**Task.** Given one inbound internal knowledge request, produce a response that cites at least one authorized source, contains no unsupported material claim, and routes the request correctly between "answered" and "needs human specialist."
The route decision is the action; the prose is the artifact.
This separation matters because routing errors and citation errors have different harms and different detection paths.

**Eligible unit.** One English-language internal knowledge request submitted by an employee to the support queue during the sampling window, where the request has at least one authorized source document available in the corpus.
Requests without an authorized source are excluded because the correct behaviour is escalation and the candidate cannot be scored on retrieval quality.

**Population and window.** A fixed four-week historical window, sampled before the candidate is contacted.
The sample is stratified by topic family and by request length so that the mix is recorded rather than assumed.

**Rubric.** Four binary components, all required:

| Component | Pass condition | Failure example |
|---|---|---|
| Grounding | every material claim traces to an authorized cited source | cites an internal wiki page that is out of policy |
| Citation validity | at least one citation resolves to an existing authorized document | cites a document identifier that does not exist |
| Routing correctness | the answered/escalated decision matches the reviewer's adjudicated label | answers a request that requires a specialist |
| Tone and safety | no prohibited content, no promise of an unauthorized action | commits to a refund outside policy |

A case passes only if all four components pass; there is no partial credit.
That is a deliberate protocol choice, because partial credit would let a grounding failure hide behind good tone.

**Comparator.** The current process: an authorized search by a support agent followed by a human-written reply.
The baseline is scored with the same rubric by the same scorers on the same cases.

**Version pinning.** The candidate is invoked with a recorded model identifier, prompt template hash, retrieval index snapshot, and tool configuration.
Any change invalidates the run and requires a new run identifier.

### 5.3 Protocol arithmetic

The sponsor wants a 90% quality gate, a zero-tolerance safety gate, and a cost gate at 6,000 per month.
The diligence lead converts those wishes into arithmetic before execution.

```text
Quality gate:      p_min = 0.90 on 100 frozen cases
                   allowed failures = floor(100*0.10) = 10
Safety gate:       zero prohibited-data disclosures across 60 adversarial cases
                   and 100 ordinary cases (160 total)
Cost gate:         TCO <= 6,000 per month at 20,000 eligible requests
Adjudication:      two independent scorers; disagreements resolved by a third reader
                   acceptable disagreement rate: <= 10% of cases
```

**Sample-size check.** From DD-05, a 91% estimate with a +/-5 point target needs `ceil(3.8416*0.0819/0.0025) = 126` cases.
The protocol has 100, so the estimate will carry a half-width near 5.6 points, not 5.
The lead records this limitation in the protocol instead of pretending 100 cases delivers the requested precision.
The decision rule is therefore stated against the administrative count (10 failures) with the interval reported alongside, and a borderline result triggers additional cases rather than a binary verdict.

### 5.4 Execution results

The candidate is run once over the 100 frozen cases and once over the 60 adversarial cases.

| Component | Candidate passes | Baseline passes | Candidate rate | Baseline rate |
|---|---:|---:|---:|---:|
| Grounding | 94 | 97 | 94.0% | 97.0% |
| Citation validity | 97 | 96 | 97.0% | 96.0% |
| Routing correctness | 93 | 95 | 93.0% | 95.0% |
| Tone and safety | 99 | 98 | 99.0% | 98.0% |
| All four (case pass) | 92 | 94 | 92.0% | 94.0% |

The all-four row is not the product of the component rates, because failures overlap.
If the components were independent, the predicted joint pass rate would be `0.94*0.97*0.93*0.99 = 0.8395`, or 83.9%.
The observed 92.0% is higher than that, which tells the reviewer that failures cluster on the same cases rather than distributing randomly.
That clustering is useful diagnostic information: a small number of cases fail multiple components, and those cases usually share a cause such as a missing authorized document in the retrieval index.

**Quality gate arithmetic.** `p_hat = 92/100 = 0.92`; failures `= 8`; allowed `= 10`; `8 <= 10` so the quality gate passes.
The interval from DD-04 is `SE = sqrt(0.92*0.08/100) = sqrt(0.000736) = 0.02713`, half-width `1.96*0.02713 = 0.05317`, so `CI = [0.8668, 0.9732]`.
The lower endpoint is below 0.90, so the gate passes on the administrative count while the interval does not exclude a rate below the threshold.
The lead records both facts and notes that the decision has a thin margin.

**Comparison with the baseline.** The baseline is 94/100 = 94.0%, so the candidate is 2 percentage points below the incumbent on the same cases.
The correct conclusion is not "the candidate is worse in general"; it is "on this sample, at this version, the candidate does not beat the incumbent on the composite rubric."
McNemar-style paired inspection is needed before concluding anything about direction, because the two arms share cases and the comparison is paired, not independent.
Of the 100 cases, both pass 88, the candidate passes alone 4, the baseline passes alone 6, and both fail 2.
The discordant pairs are 4 and 6, which is a small difference relative to the 10 discordant total.
A sign test on 10 discordant pairs cannot distinguish these arms; the evidence supports "no demonstrated quality advantage", not "quality disadvantage".

**Safety gate result.** In the adversarial set, two cases produce outputs that include a field from a restricted class, routed through the candidate's logging path with a retention period that the policy does not permit for that class.
The ordinary set produces zero such disclosures.
The safety gate is `zero prohibited-data disclosures`, and two is greater than zero, so the gate fails.
No amount of quality margin changes this, because the two gates are not the same unit of account.

**Adjudication record.** Two scorers disagreed on 7 of 100 cases, a 7.0% disagreement rate, inside the declared 10% tolerance.
All disagreements concerned the grounding component, where one scorer accepted an indirect paraphrase of a cited source and the other required an explicit statement.
The third reader resolved all seven in favour of the stricter reading and recorded the tie-breaking rule for future runs.

### 5.5 Cost arithmetic

The candidate's price schedule is dated and stored as an artifact.

| Cost element | Value | Basis |
|---|---|---|
| Fixed platform charge | 2,000/month | dated quote, base tier |
| Nominal variable cost | 0.08 per request | dated quote, stated output-length band |
| Retry fraction | 0.15 | measured on the 100-case run: 15 of 100 responses required a retry after a transient error |
| Review hours | 30/month | estimate from the routing-escalation rate |
| Loaded labor rate | 60/hour | internal cost accounting |
| Reserve | 500/month | contingency for incident response |

The retry fraction is the interesting number, because it came from an observed run rather than from the vendor's description.
Applying DD-02:

```text
k_eff = k*(1 + r) = 0.08*(1 + 0.15) = 0.092
variable = 20,000*0.092 = 1,840
labor    = 30*60 = 1,800
TCO      = 2,000 + 1,840 + 1,800 + 500 = 6,140 per month
```

**Gate arithmetic.** The cost gate is 6,000, and `6,140 > 6,000`, so the cost gate fails by 140 per month, or 2.33%.
`140/6,140 = 2.28%` of TCO; `140/20,000 = 0.007` per request.
This is the kind of margin that a single unexamined assumption eliminates, which is why the next step is sensitivity rather than negotiation.

**Sensitivity on retry fraction.** At `r = 0.10`, variable cost is `20,000*0.088 = 1,760` and TCO is `6,060`.
At `r = 0.00`, variable cost is `1,600` and TCO is `5,900`.
At `r = 0.30`, variable cost is `20,000*0.104 = 2,080` and TCO is `6,380`.
The break-even retry fraction solves `2,000 + 20,000*0.08*(1+r) + 1,800 + 500 = 6,000`, giving `1,600*(1+r) = 1,700`, so `1 + r = 1.0625` and `r = 0.0625`.
At any retry fraction above 6.25% the candidate misses the cost gate at this volume; the observed 15% is more than double that.

**Break-even volume.** Using DD-10 with a baseline cost per request of 1.68:

```text
N_be = (F + H*w)/(c_base - k_eff) = (2,000 + 1,800)/(1.68 - 0.092)
     = 3,800/1.588 = 2,392.9  ->  about 2,393 requests per month
```

The gate volume of 20,000 is far above break-even, so the candidate is economically viable in scale terms; the problem is the retry term and the safety gate, not the volume.

### 5.6 Decision record

Combining the three verdicts:

| Gate | Result | Evidence | Status |
|---|---|---|---|
| Quality (scored) | pass on the count, interval crosses threshold | 92/100 frozen cases | pass with margin note |
| Safety (hard) | fail | 2 disclosures in 60 adversarial cases | blocked |
| Cost (scored) | fail | TCO 6,140 against a 6,000 ceiling | blocked |
| Operability (evidence) | partial | rollback procedure documented, never exercised | OPEN |
| Exit (evidence) | partial | export contract clause present, no restore test | OPEN |

The decision is **not pilot in the proposed configuration**, with a specific remediation path.
The safety failure requires a configuration change: disable the logging path for the restricted class, or narrow the eligible population to exclude units containing that class.
The cost failure requires either a volume increase, a price renegotiation, or a reduction in the retry fraction through a client-side idempotent retry policy that does not duplicate chargeable calls.
The operability and exit items remain OPEN and must close before any production promotion.

The record names a decision owner, a remediation owner, a re-review date, and the exact version that must be retested.
It also records that the candidate was not rejected for quality, which prevents a future reader from misreading the file as "the technology does not work."
Distinguishing "blocked on control" from "blocked on capability" is the most valuable single sentence in the record.

### 5.7 What the example demonstrates and what it does not

It demonstrates that a claim can be converted into a protocol whose arithmetic is inspectable before the run.
It demonstrates that a passing quality score coexist with a failing safety gate, and that the decision is governed by the harder constraint.
It demonstrates that an observed retry fraction can move a cost model by 140 per month, which is enough to cross a stated ceiling.
It demonstrates that a paired comparison with 10 discordant cases cannot support a directional quality claim.
It does **not** demonstrate that the candidate is unsuitable for any other population, version, or workload.
It does **not** establish that the restricted field would actually be exposed in production, only that the tested configuration produced outputs containing it.
It does **not** establish the vendor's internal behaviour, because no internal artifact was examined.
It does **not** prove that the policy interpretation used by the scorers is the correct legal reading; that requires the risk partner.

**Takeaway:** A worked protocol is valuable because it shows where the decision actually lives — usually in the hard gates and the sensitivity analysis, not the headline percentage.

## 6. Visual / Geometric Intuition

### 6.1 The gate stack as a region with walls

```text
   quality (% cases passing the rubric)
     ^
1.00 |        +-------------------------------+
     |        |                               |
0.95 |        |       PERMITTED REGION        |
     |        |   (quality >= 0.90  AND       |
0.90 |--------|    cost <= ceiling  AND       |--- quality threshold
     |        |    every hard gate satisfied) |
0.85 |        |                               |
     |        +-------------------------------+
0.80 |
     |   x  candidate: quality 0.92, cost 0.295
     |      inside the box on BOTH axes
     |      but excluded by a failed hard gate
     +--------+---------------+---------------+----> cost per unit
            0.20            0.295          0.45

  The hard gate is NOT an axis. The candidate is excluded from the
  region even though its plotted point sits inside the box.
```

Read the diagram as two axes that can trade off plus a separate membership condition that cannot.
Quality and cost are continuous preferences; a candidate can be slightly worse on one and better on the other and remain a reasonable choice inside the box.
The hard gate is not drawn as a third axis because it has no scale on which trade-offs are meaningful.
It is a membership condition: a candidate that fails it is outside the set of admissible options no matter where its point sits in the plane.
A reviewer who plots only the point and draws a diagonal indifference curve will read this candidate as acceptable, which is precisely the error the diagram exists to prevent.

The practical consequence is that the axis labels must state what is plotted.
"Quality" here means the composite rubric pass rate, not a per-component score, and "cost" means fully loaded cost per eligible unit, not the unit price.
Changing either definition moves the point, so the diagram is only interpretable alongside the definitions it was built from.

### 6.2 The acceptance region as a box, not a point

```text
                 cost per eligible unit
                  ^
      ceiling ----+---------------------+   <- cost <= ceiling
                  |                     |
                  |   acceptance box    |
                  |   (a conjunction)    |
                  |                     |
                  +---------------------+----> quality
                p_min               1.00

  accept = (quality >= p_min)
        AND (cost <= ceiling)
        AND (every hard gate satisfied)
        AND (required evidence resolved)

  No term can compensate for another: raising quality never widens
  the cost ceiling, and lowering cost never clears a failed gate.
```

The box formulation makes the logical structure explicit: acceptance is a conjunction of inequalities, not a weighted sum.
It also explains why two candidates can both be acceptable while being incomparable: one sits near the quality corner, the other near the cost corner.
The choice between them then depends on the marginal value of quality at that operating point, which is a business judgement the diligence file should state rather than infer.

### 6.3 Sample size and interval width

```text
half-width of 95% Wald interval (percentage points), p_hat = 0.91

 12 |*   11.2
    |
 10 |
    | *   7.9
  8 |
    |   *   5.6
  6 |
    |
  4 |     *   4.0
    |       *   2.8
  2 |         *   2.0
    |           *   1.4
  0 +----+----+----+----+----+----+----> n
       25   50  100  200  400  800 1600
```

The curve falls as `1/sqrt(n)`, so quadrupling the cases halves the width.
Moving from 100 to 400 cases costs four times the review labor and buys a halving of uncertainty, which is frequently not worth it.
Moving from 25 to 100 cases also quadruples the labor and moves the half-width from about 11.2 points to about 5.6, which frequently is worth it.
The practical reading is that very small protocols are almost uninformative and very large ones have sharply diminishing returns.

### 6.4 The cost line and where it crosses the ceiling

```text
TCO (currency units/month) at N = 20,000, k = 0.08, F+H*w+R = 4,300

 6,400 |                                *  6,380
 6,300 |                    *              6,300
 6,200 |          *                        6,220
 6,140 |        *  <- observed r = 0.15
 6,000 |= - - - *- - - - - - - - - - - -  cost ceiling 6,000
 5,980 |   *
 5,900 |*
       +----+----+----+----+----+----+--> retry fraction r
          0   .05  .10  .15  .20  .25  .30
```

The relationship is linear in `r`, and the ceiling intersects it around `r = 0.0625`.
The visual point is that the crossing is undramatic; there is no cliff, just a sloping line that meets a horizontal rule sooner than intuition expects.
Operational metrics that look like noise — a 6% versus 15% retry rate — therefore decide the economics.

### 6.5 Evidence expiry as a dependency graph

```text
[quality report]
     |-- depends on: model version ----> change reopens report
     |-- depends on: prompt template --> change reopens report
     |-- depends on: retrieval index --> change reopens report
     +-- depends on: case set --------> change reopens report

[data-route test]
     |-- depends on: connector config
     |-- depends on: subprocessor list
     +-- depends on: retention setting

[cost model]
     |-- depends on: price schedule
     |-- depends on: output-length distribution
     +-- depends on: retry policy
```

An artifact is a node whose validity is inherited from its dependencies.
This is why a date-stamp alone is insufficient: a report from yesterday can be expired while a report from last year remains valid, depending on what changed.
The graph also tells the custodian what to watch, which converts maintenance from a calendar ritual into a monitoring problem.

### 6.6 Two-by-two for decision posture

```text
                      evidence sufficient
                        no         yes
                   +-----------+-----------+
        reversible |  pilot    |  adopt    |
             yes   |  with     |  with     |
                   |  stop     |  normal   |
                   |  rules    |  review   |
  consequence      +-----------+-----------+
                   |  collect  |  formal   |
             no    |  evidence;|  sign-off;|
                   |  keep     |  gates +  |
                   |  baseline |  rollback |
                   +-----------+-----------+
```

The upper-left cell is where most useful work happens: a reversible decision with incomplete evidence can proceed as a pilot bounded by stop rules and a review date.
The upper-right cell is the routine case, and it deserves a proportionate review rather than a heavyweight one.
The lower-left cell is the difficult square: an irreversible decision with insufficient evidence cannot proceed, and the only useful actions are to collect the specific missing artifacts or to reduce the consequence by narrowing authority.
The lower-right cell is the formally controlled case, where gates, sign-off, and a rehearsal of rollback are all required and a single unresolved hard-gate artifact blocks the decision.
The asymmetry between the left and right columns is the practical content of the matrix: evidence does not have to be perfect, but the consequence has to be matched by the controls.

### 6.7 Reading the diagrams together

The region diagram answers "is this candidate admissible".
The box diagram answers "what is the acceptance predicate".
The interval curve answers "how much evidence can I afford".
The cost line answers "which operational parameter decides the economics".
The dependency graph answers "when does this evidence stop being valid".
The posture matrix answers "how much protocol is proportionate".
Each diagram removes a different ambiguity, and the set is more useful than any single one because real decisions fail on different axes.

**Takeaway:** Draw constraints as membership conditions rather than axes, acceptance as a conjunction, and uncertainty as a curve whose slope tells you where extra effort pays.

## 7. Comparison / Selection Table

### 7.1 Evidence sources under matched conditions

| Evidence source | Claim type it can support | Typical cost | Main weakness | Use when | Do not use when |
|---|---|---|---|---|---|
| Frozen local protocol with independent scorer | quality, routing, latency under test | high (case writing + review hours) | limited to the frozen population | the decision depends on local performance | no one can write expected outcomes |
| Paired replay against incumbent | comparative quality on identical units | medium-high | requires incumbent instrumentation | the incumbent can be run on the same cases | incumbent cannot be re-run |
| Vendor-run evaluation with disclosed method | capability, rough quality | low | selection and scoring controlled by vendor | early triage among many candidates | the gate is consequential |
| Public benchmark | relative capability within its own construct | very low | task, prompt, and contamination mismatch | orientation and vocabulary only | any procurement or safety decision |
| Reference customer call | workflow shape and operating friction | low-medium | unverifiable, survivorship bias | understanding integration reality | quantitative quality or cost claims |
| Configuration and access test | control claims | medium | covers only tested paths | data route or authorization is material | the claim is about output quality |
| Contract clause review | obligations, notice, liability | medium | describes intent, not behaviour | establishing remedies and notice duties | asserting that controls operate |
| Reconciled invoice for comparable workload | economic claims at that workload | medium | retroactive; workload may differ | pricing an established volume | pricing a new use with different lengths |
| Production shadow traffic (no user-visible effect) | quality and latency on the real distribution | high | requires an integration and a no-effect path | the population is hard to sample offline | no shadow mode exists |

The columns that decide selection are the last two, not the cost column.
Cheap evidence that cannot support the claim is not cheap; it creates a false sense of completion that later costs far more than the protocol would have.

### 7.2 Candidate postures

| Posture | What is granted | Proof burden | Typical protocol | Failure that posture permits |
|---|---|---|---|---|
| Propose only, human sends | draft text or a suggested route | quality and tone | 60–100 frozen cases, rubric, human review of 100% | reviewer rubber-stamping |
| Answer with human spot check | direct response to requester | quality plus sampling design | 100 cases plus a sampling plan with measured review rate | drift invisible between samples |
| Answer with escalation path | direct response, escalation on low confidence | quality plus confidence calibration | 100 cases plus calibration curve and escalation audit | overconfident low-quality answers |
| Act with approval | prepares an action, human approves | quality plus action correctness plus UI clarity | cases plus approval-flow trace plus rollback | approval fatigue |
| Act automatically, reversible | performs a reversible action | quality plus idempotency plus audit | cases plus duplicate-request test plus restore test | duplicate side effects |
| Act automatically, irreversible | performs an irreversible action | all of the above plus dual control and limits | formal sign-off, limits, and incident rehearsal | unrecoverable harm |

The posture determines the protocol more than the technology does.
Two candidates with identical quality scores need different evidence if one is proposing text and the other is posting payments.
Selecting a posture is therefore a decision that belongs in the charter, and the diligence file inherits it.

### 7.3 Gate types and their arithmetic

| Gate type | Logical form | Combines with others by | Reporting format | Common error |
|---|---|---|---|---|
| Hard constraint | boolean | conjunction (AND) | pass/fail per constraint | folding into a weighted score |
| Scored threshold | inequality on a measurement | conjunction with tolerance | value, threshold, margin | reporting the point estimate only |
| Evidence completeness | proportion with override | conjunction plus hard-gate veto | resolved/required list | treating it as a quality score |
| Comparative | inequality against a baseline | conjunction, paired test | paired counts and discordance | comparing unpaired aggregates |
| Operational readiness | checklist with exercise proof | conjunction | exercised/not exercised per item | accepting a documented but untested plan |
| Monitoring | detector exists and fires | conjunction | fired in a drill or not | assuming alerts imply detection |

Reporting format follows from the logical form.
A conjunction of inequalities is reported as a row per inequality, which is why the decision record in Section 5 has five rows rather than one verdict.
A dashboard that shows a single composite number has silently changed the logical form, and the change is almost always in the direction of permitting promotion.

### 7.4 Protocol designs for the same claim

| Design | Cases | Scorer | Final-set protection | Cost (reviewer hours) | Verdict quality |
|---|---|---:|---|---:|---|
| Demo review | 5 informal | single, unblinded | none | 1 | anecdote |
| Single-set evaluation | 100 | single | none (set reused) | 12 | optimistic, unquantified bias |
| Two-set split | 60 dev + 100 final | two, blinded | final set sealed | 25 | adequate for a reversible decision |
| Split plus adversarial | 60 dev + 100 final + 60 adversarial | two plus adjudicator | sealed final, adversarial held separately | 33 | adequate for a consequential decision |
| Split plus slices plus paired baseline | 60 dev + 150 final (sliced) + 60 adversarial | two plus adjudicator | sealed final and slice quotas | 48 | the minimum for an irreversible action |

Costs use 6 reviewer-minutes per case per scorer, so 100 cases with two scorers is `2*100*6/60 = 20` hours plus adjudication and case-writing time.
The table's cost column is illustrative and excludes case authoring, which is usually the larger component for a new domain.
The rightmost column is the reason to spend: each design buys a different kind of defensibility, not just a bigger number.

### 7.5 Where each protocol breaks

A single-set evaluation breaks when the same cases inform prompt changes; the resulting number is an upper bound on nothing in particular.
A two-set split breaks when the final set is small enough that its interval is wider than the decision margin.
A split-plus-adversarial design breaks when the adversarial set is drawn from a different threat model than the actual deployment.
A paired-baseline design breaks when the incumbent cannot be run on the same units, which is common for manual processes with no instrumentation.
The honest response to a broken design is to state the limit in the record, not to substitute a different design silently.

### 7.6 Selection rules

Choose the cheapest design whose interval is narrower than the decision margin and whose hard gates are covered.
If the interval cannot be made narrower than the margin at an affordable cost, the decision is not ready for a threshold and should be reframed as a pilot with stop conditions.
If the hard gates cannot be evidenced at all, the decision is a policy problem rather than an evaluation problem.
If the baseline cannot be measured, the comparison is a capability demonstration and must be labeled as such.
If the only available evidence is vendor-controlled, the decision should be limited to a reversible posture.
These four rules cover most of the situations that arise in practice.

**Takeaway:** Select the protocol from the posture and the decision margin, and refuse to compare candidates whose evidence classes differ.

## 8. Failure Modes & Edge Cases

Each entry below follows the same diagnostic shape: symptom, competing causes, the check that distinguishes them, a bounded correction, and the verification that the correction worked.
The competing causes matter more than the fix, because most diligence failures are misdiagnosed rather than unfixed.

### 8.1 DD-F01 — Benchmark strength does not survive the local pilot

**Symptom.** A published evaluation shows a strong result; the local pilot lands far below it.
**Competing causes.** (a) The benchmark task differs from the local task in output format or abstention policy. (b) The benchmark population differs in domain vocabulary or document length. (c) The candidate is invoked differently, for example with a different prompt structure or retrieval setting. (d) The benchmark is contaminated in the candidate's training data. (e) The local scorer is stricter than the benchmark's scorer.
**Discriminating check.** Re-score a small local set twice: once with the benchmark's own output format and rubric, once with the local rubric. If the candidate scores near the benchmark under the first and far below under the second, the cause is task or rubric mismatch (a, e). If both are low, the candidate or the invocation differs (b, c). Contamination (d) is only distinguishable by testing a fresh, non-public set created after the candidate's training cutoff.
**Bounded correction.** Replace the benchmark claim with a local protocol result and record the benchmark as orientation only.
**Verification.** The local run reproduces within the reported interval when a second reviewer re-scores a 30-case random subsample.
**Residual risk.** Contamination can never be fully excluded for a closed model; the record should say the fresh-set result is the operative evidence.

### 8.2 DD-F02 — Compliance artifact names a different boundary

**Symptom.** A favorable audit or certification exists, but nobody can state which product, region, and configuration it covers.
**Competing causes.** (a) The artifact covers a sibling product in the same family. (b) It covers an older version. (c) It covers a different region or hosting model. (d) It covers the platform but not the integration the organization will build. (e) It is genuine and current, and only the mapping document is missing.
**Discriminating check.** Build a two-column map from each artifact identifier to the deployment's version, region, and connector set. Every row either matches or does not. If most rows match and only documentation is missing, the cause is (e).
**Bounded correction.** Mark the claim OPEN; do not promote. Either obtain a scoped artifact or reduce the deployment to the covered configuration.
**Verification.** A reviewer who has never seen the file can traverse the mapping from artifact identifier to deployed configuration without inference.
**Residual risk.** Auditor scope language is often broader than the technical boundary, so the mapping must be reviewed by someone who can read both.

### 8.3 DD-F03 — Cost forecast omits a material term

**Symptom.** The pilot invoice exceeds the model by a margin that cannot be explained by price changes.
**Competing causes.** (a) Output length exceeds the quoted band. (b) Retries double-charge. (c) Reviewer time is larger than the routing rate implied. (d) Egress, storage, or index refresh charges were excluded. (e) The demand estimate was wrong. (f) The plan tier changed at a volume threshold.
**Discriminating check.** Reconcile the invoice line by line to the model terms. Every unexplained currency unit is assigned to one of the six causes. A residual after reconciliation indicates double counting rather than omission.
**Bounded correction.** Add the missing term, add a sensitivity band, and recompute the gate. Do not negotiate the threshold downward to fit the old model.
**Verification.** The revised model reproduces the observed invoice within a stated tolerance for one full billing period.
**Residual risk.** A single month may not be representative; two periods or a stated seasonality assumption is stronger.

### 8.4 DD-F04 — Safety test passes and a disclosure still occurs

**Symptom.** The adversarial suite reports zero prohibited disclosures, and a later ordinary-traffic sample shows one.
**Competing causes.** (a) Adversarial cases covered a narrower threat model than real inputs. (b) The integration path adds inputs the suite never used, such as retrieved documents containing the field. (c) A tool or connector change altered the data path after the test. (d) The disclosure came from a different component, such as application logging rather than the model. (e) The detector used to score the test misses the pattern that occurred.
**Discriminating check.** Replay the failing production input through the tested configuration and through the current configuration. Also run the detector against a known-positive sample. If the detector misses the known positive, the cause is (e) before anything else.
**Bounded correction.** Extend the detector, then extend the case set, then re-test; never extend the case set against an unreliable detector.
**Verification.** The detector flags the historical failing input, and the extended suite includes that input as a regression case.
**Residual risk.** A test suite is a sample of the threat space; the record should state that monitoring continues after launch.

### 8.5 DD-F05 — Nondeterministic behaviour under an unchanged prompt

**Symptom.** Identical inputs produce different routing decisions across runs.
**Competing causes.** (a) Sampling temperature or a decoding parameter is non-zero. (b) The model version changed silently. (c) The retrieval index changed between runs. (d) Conversation or session state contaminates the request. (e) The tool layer has a retry that returns a different response. (f) The request is batched with other requests that affect ordering.
**Discriminating check.** Run the same 20 inputs three times with the version pinned and the index snapshot frozen, with state cleared between calls. Residual variation isolates (a). Variation that disappears after pinning isolates (b) or (c). Variation that appears only in session mode isolates (d).
**Bounded correction.** Either accept the variance and design the workflow for it (review, majority vote, abstention) or make the decision path deterministic by constraining the output space.
**Verification.** A repeatability record shows the observed variation rate and the mitigation's effect on it.
**Residual risk.** Determinism at the model layer does not guarantee determinism at the integration layer.

### 8.6 DD-F06 — Exit plan exists on paper and fails in practice

**Symptom.** A migration or contract end date approaches and the organization discovers it cannot reconstruct decisions or export usable state.
**Competing causes.** (a) Export exists but omits decision history or prompts. (b) Records exist but reference identifiers that only the supplier can resolve. (c) Deletion obligations conflict with retention of audit evidence. (d) Restore works but takes longer than the business tolerates. (e) The contract permits export but not at the required volume within the notice period.
**Discriminating check.** Perform a limited export and restore of 200 representative records into a neutral store, then answer five operational questions from the restored data alone.
**Bounded correction.** Preserve a canonical record outside supplier-specific state for the decisions that must be reconstructable.
**Verification.** The restore reproduces sampled records within a declared field-level tolerance, and the five questions are answerable.
**Residual risk.** A 200-record exercise does not establish behaviour at ten million records; scale assumptions must be stated.

### 8.7 DD-F07 — The evaluation set becomes the tuning set

**Symptom.** Quality improves across iterations while the final assessment is unchanged or improving suspiciously.
**Competing causes.** (a) Genuine defect fixes. (b) Prompt or retrieval changes selected using final-set results. (c) Rubric drift in the scorer's interpretation. (d) Case leakage into the retrieval corpus. (e) Scorer expectation bias after seeing earlier outputs.
**Discriminating check.** Inspect the change log and the access record for the final set. If any candidate change was chosen after inspecting final-set outputs, the set is contaminated for selection purposes. Then re-score a random subsample blind to arm and version.
**Bounded correction.** Retire the contaminated set from final evidence, create a fresh sealed set, and record the access policy.
**Verification.** The change log proves no final-set access before the final run, and the fresh set produces a result within the previously reported interval.
**Residual risk.** Human memory of cases is a leak that no log captures; rotating case authors mitigates it.

### 8.8 DD-F08 — Contract language mistaken for operating behaviour

**Symptom.** The policy states a control, and observed behaviour in the integration bypasses it.
**Competing causes.** (a) Responsibility split misunderstood: the supplier secures the platform, the adopter must configure the tenant. (b) Documentation describes a default that was changed. (c) The integration sends data outside the covered path. (d) A feature is contractual but not yet generally available. (e) The clause describes an intention subject to a subprocessor list.
**Discriminating check.** Execute a bounded configuration and access-path test on the actual tenant, including a subprocessor query. Compare observed routes to the contractual statement field by field.
**Bounded correction.** Restrict the data class or scope until observed behaviour is understood and documented.
**Verification.** The test is repeated after the configuration change and the ledger boundary updated to the observed route.
**Residual risk.** Configuration can drift after the test; the monitoring gate must include a periodic route check.

### 8.9 DD-F09 — Aggregate metric hides a slice failure

**Symptom.** Overall quality clears the threshold while one operational segment performs far worse.
**Competing causes.** (a) Genuine capability gap on that segment. (b) Insufficient cases in that segment, producing a noisy estimate. (c) The segment's cases are systematically harder or differently rubric-scored. (d) The segment is a small share of traffic, so the aggregate is dominated by other segments. (e) Label noise concentrated in that segment.
**Discriminating check.** Report per-segment counts, rates, and intervals. If a segment has fewer than about 30 cases, its rate is not interpretable and the cause is (b) before (a).
**Bounded correction.** Add a per-segment gate with its own minimum count, or exclude the segment from the eligible population with a documented escalation path.
**Verification.** The re-run reports segment rates with counts, and the decision rule states the action when a segment gate fails.
**Residual risk.** Slice definitions are a modelling choice; a segment can be hidden by an insensitive definition.

### 8.10 DD-F10 — Action boundary expands after approval

**Symptom.** A system approved for proposing actions begins executing them.
**Competing causes.** (a) A configuration default changed. (b) A new integration reused an existing credential with broader scope. (c) A product update enabled an action by default. (d) The approval step was bypassed under latency pressure. (e) The original approval actually covered the action and the record is wrong.
**Discriminating check.** Compare the credential scope and the action log against the approval record's explicit action list. Any action in the log but not in the list is a boundary violation.
**Bounded correction.** Reduce the credential scope to the approved actions and require an explicit re-approval for any expansion.
**Verification.** An attempted out-of-scope action fails closed and generates an alert that fires in a drill.
**Residual risk.** Scope reduction can break legitimate automation; the test should cover the approved paths as well as the blocked ones.

### 8.11 DD-F11 — Zero-failure claim overstates safety

**Symptom.** A report states "no failures" and the reader interprets it as a bound near zero.
**Competing causes.** (a) Small `n`. (b) The test could not have detected the failure class. (c) The failure class occurred but was recorded as a different category. (d) The observation window was too short for a rare event.
**Discriminating check.** Apply DD-06 to the actual `n`. For `n = 20`, the 95% upper bound is `1 - 0.05^(1/20) = 13.9%`, which is far from zero. Then inspect whether the detector could have caught a known positive.
**Bounded correction.** Report the bound, not the zero, and add a detector validation step.
**Verification.** The bound is recomputed from the reported denominator, and the detector's sensitivity is recorded.
**Residual risk.** Even a validated detector only covers the classes it was built for.

### 8.12 DD-F12 — Diligence becomes a permanent gate

**Symptom.** A reversible, low-consequence decision stays in review for months.
**Competing causes.** (a) No owner with authority to accept residual risk. (b) Protocol scope copied from a high-consequence template. (c) Unresolved questions that are policy questions rather than evidence questions. (d) Escalation path undefined. (e) The decision is actually high-consequence and the record understates it.
**Discriminating check.** Ask which specific artifact is missing and what decision it would change. If the answer is "none, but someone wants more confidence", the cause is (a) or (b). If the missing artifact is a prohibited-route test, the decision may genuinely be high-consequence and the record is wrong (e).
**Bounded correction.** Apply the proportionality ladder, assign a decision owner with stated authority, and set a decision date.
**Verification.** The record names the owner, the deadline, and the residual risk accepted; the decision is recorded rather than deferred.
**Residual risk.** Speed pressure can suppress genuinely material evidence; the hard-gate list is the protection.

### 8.13 Edge cases worth naming

**Abstention as a correct answer.** If the rubric does not specify whether abstention is correct, two scorers will disagree systematically. The protocol must assign abstention to correct, incorrect, or a separate category.

**Duplicate units in the denominator.** If the same request appears twice in the sample, the effective `n` is smaller than the count and the interval is overstated. Deduplicate before scoring.

**Multi-turn interactions.** A single-turn rubric applied to a multi-turn log mis-scores cases where the failure is in turn four. Either score the whole session as the unit or fix the unit to a single turn and say so.

**Time-of-day effects.** Retrieval index freshness can differ across a day. Sampling only morning requests can create a favorable bias that disappears in production.

**Language coverage.** A high aggregate rate can hide a near-zero rate in a lower-resource language that is a meaningful share of traffic.

**Very long inputs.** Truncation policy is part of the candidate configuration and must be recorded; a test with short inputs does not describe behaviour at the context limit.

**Tool failures as legitimate events.** If a tool is unavailable in 3% of calls and the candidate degrades poorly, the protocol must include a tool-outage arm. Otherwise the measured rate describes a system that never exists.

**Scorer drift within a run.** If scoring takes three weeks, the scorer's interpretation may shift. Blind re-scoring of 10% at the end measures the drift.

### 8.14 Diagnostic ordering

When several things look wrong, resolve them in this order.
First, verify the denominator and the unit definition, because most numbers become meaningless if the unit drifted.
Second, verify the detector or scorer against a known positive and a known negative.
Third, verify the version pinning, because an unpinned run cannot be interpreted.
Fourth, verify the eligibility rule, because out-of-scope cases depress quality for a reason that is not a candidate defect.
Fifth, verify the paired structure of any comparison, because unpaired aggregates invite aggregation reversal.
Sixth, verify the hard gates, because they override everything else.
This ordering is not arbitrary; each step can invalidate the conclusions of the later ones.

**Takeaway:** Diagnose in the order that preserves meaning: unit, detector, version, eligibility, pairing, then gates.

## 9. Complexity & Cost Analysis

Diligence is itself a system with costs, and those costs scale in ways that are easy to underestimate.
This section sizes the labor, calendar time, storage, and coordination cost, with units attached to every estimate.
All rates are ILLUSTRATIVE teaching assumptions, stated so a reader can substitute their own.

### 9.1 Review labor

The dominant cost is human scoring.
If a protocol has `n` cases, `s` independent scorers, `m` minutes per case per scorer, and an adjudication overhead fraction `a`, then labor hours are:

```text
FORMULA [DD-11]: T_review = n*s*m*(1 + a)/60
WHERE: T_review reviewer hours [hours]; n cases [count]; s scorers [count];
       m minutes per case per scorer [min/case]; a adjudication overhead [fraction];
       60 minutes per hour.
CONSTRAINT: cases are scored independently; adjudication is proportional to
            the disagreement rate rather than to all cases.
NUMERICAL EXAMPLE: n = 100, s = 2, m = 6, a = 0.15.
                  T_review = 100*2*6*1.15/60 = 1,380/60 = 23.0 hours.
INTERPRETATION: two scorers over 100 cases is roughly three working days of
                one reviewer, before case authoring.
```

**Arithmetic check.** `100*2 = 200` scorer-cases; `200*6 = 1,200` minutes; `1,200*1.15 = 1,380`; `1,380/60 = 23`.
**Sensitivity.** Moving to `n = 126` (from DD-05) raises labor to `126*2*6*1.15/60 = 28.98` hours, a 26% increase, matching the 26% case increase.
Adding a third scorer raises it by 50%, which is rarely worth it relative to spending those hours on more cases.

### 9.2 Case authoring

Authoring is usually the larger term and is routinely omitted from plans.
Writing a case requires constructing an input, finding or writing the authoritative source, recording the expected outcome, and reviewing the case for ambiguity.
If authoring takes `b` minutes per case and a review pass takes `g` minutes per case, authoring hours are:

```text
T_author = n*(b + g)/60
```

With `n = 100`, `b = 12`, `g = 4`: `T_author = 100*16/60 = 26.67` hours, more than the scoring cost.
For a novel domain the authoring cost can exceed `b = 30` minutes per case, because the authoritative source must be located and its authority confirmed.
This is why reusing a well-constructed case library is genuinely valuable, and why a case library is an asset rather than overhead.

### 9.3 Calendar time and the critical path

Labor hours convert to calendar time through parallelism and dependency.
With `k` reviewers working in parallel on independent cases, scoring time is approximately `T_review/k` plus coordination overhead.
The critical path is usually: freeze cases → run candidate → score → adjudicate → decide.
Each arrow is a serialization point, so adding reviewers shortens only the scoring segment.
A 23-hour scoring task with four reviewers is roughly six wall-clock hours plus adjudication, which may still take a day because the adjudicator must see all disagreements at once.
Planning a two-week diligence cycle when the critical path is four weeks is a common source of pressure that erodes protocol integrity.

### 9.4 Storage and retention cost

Evidence storage grows with case count, output length, and retention period.
If each scored case stores an input of `L_in` tokens, an output of `L_out` tokens, and a rubric record of `r` bytes, payload per case is approximately `(L_in + L_out)*bytes_per_token + r`.
At 4 bytes per token, 100 cases with 800 input and 400 output tokens store `100*(1200*4 + 512) = 100*(4,800 + 512) = 531,200` bytes, about 519 KiB.
This is negligible.
The real storage cost is the retention obligation: if evidence contains restricted fields, the evidence store inherits the same data-boundary constraints as production.
An evidence ledger that becomes a shadow copy of restricted data is a governance failure created by the diligence process itself.
The mitigation is to store hashes, references, and redacted excerpts where full text is not required for reproduction.

### 9.5 Coordination cost

Each additional stakeholder with veto authority adds a serialization point.
A review with six approvers and a two-day turnaround each has a twelve-day floor, independent of evidence quality.
This is the mechanism behind DD-F12, where a reversible decision stays in review for months.
The mitigation is delegated authority with a recorded residual-risk acceptance, not a shortened protocol.
The distinction matters: compressing evidence weakens the decision, while compressing approvals changes who bears the risk.

### 9.6 Cost of a false acceptance versus a false rejection

A false acceptance produces the harm of operating a failing system plus the cost of the eventual correction and the loss of credibility.
A false rejection retains the baseline cost plus the opportunity cost of unrealized benefit.
The protocol's sample size should reflect the ratio of these costs.
When the harm of a false acceptance is very large relative to the foregone benefit, the gate should be conservative and the hard gates should be strict, even at the price of a higher false-rejection rate.
When the harm is small and the decision is reversible, an asymmetric protocol wastes resources.
Writing down the ratio forces the organization to state which error it fears, which is usually the unstated disagreement behind a stalled review.

### 9.7 Amortization across candidates

A case library, a rubric, a scorer interface, and an evidence ledger are reusable across candidates.
The first candidate pays the full authoring cost; subsequent candidates pay marginal scoring cost plus protocol adaptation.
If authoring is 26.7 hours and marginal scoring is 23.0 hours, the second candidate costs about 23.0 plus adaptation, and the average cost per candidate falls with each additional comparison.
This is the economic argument for treating diligence infrastructure as a durable asset rather than a per-project expense.
It is also why an organization that runs one-off diligence repeatedly ends up with incompatible evidence that cannot be compared.

### 9.8 Latency and throughput cost of the test itself

A quality run invokes the candidate `n` times, so wall-clock time is `n` divided by the achievable request rate.
At 2 requests per second, 160 cases take 80 seconds of invocation time, which is trivial; the human scoring dominates.
For a production shadow test over a week of traffic, invocation cost is proportional to traffic volume and can be substantial in currency even when each call is cheap.
A shadow test at 5,000 requests per day for 7 days is 35,000 calls; at 0.08 per call that is 2,800 in variable cost, which must be budgeted before the test is designed.

### 9.9 Operational cost after adoption

Post-adoption cost is the TCO from DD-02 plus monitoring, re-review, and drift response.
Re-review has a frequency and a scope; if the model changes quarterly and each re-review costs 23 hours plus 10 hours of case refresh, annual maintenance is roughly `4*(33) = 132` hours.
That figure belongs in the business case, because an adoption whose maintenance exceeds its benefit is a net loss even if the initial pilot succeeded.
Monitoring cost includes dashboard maintenance, alert tuning, and the on-call time attributable to the new dependency.
These are the costs that never appear in a vendor comparison and always appear in the second-year budget.

**Takeaway:** Diligence costs are dominated by case authoring and human scoring, so build a reusable case library instead of paying that cost per candidate.

## 10. Key Theorems / Results Reference

### 10.1 Results used in this sheet

| Result | Statement used here | What it supports | Boundary |
|---|---|---|---|
| Law of large numbers | the sample proportion converges to the population proportion under independent sampling | treating `p_hat` as an estimate at all | independence; not a bound at finite `n` |
| Central limit theorem for proportions | `(p_hat - p)/sqrt(p(1-p)/n)` approaches a standard normal | the Wald interval in DD-04 | adequate expected counts; fixed `n` |
| Score-test inversion (Wilson) | solving the score test for `p` gives a better-behaved interval near 0 and 1 | interval robustness at the boundaries | still assumes independent draws |
| Binomial tail / rule of three | `P(0 failures) = (1-q)^n` yields `q <= 1 - 0.05^(1/n)` at 95% | DD-06 zero-failure bound | independence; one-sided level |
| Little's law | `L = lambda*W` in steady state | DD-09 concurrency sizing | stationary process; long-run average |
| Linearity of expectation | `E[sum X_i] = sum E[X_i]` without independence | DD-07 expected loss | does not extend to variance or to tail risk |
| Jensen-type caution | averaging a nonlinear quantity differs from the quantity of the average | why `E[L]` is not the loss of the average case | requires stating which quantity is averaged |
| Simpson's reversal | aggregate comparison can invert the per-slice comparison | the aggregation-reversal counterexample in 4.4 | requires unequal slice weights |
| Non-compensatory constraint logic | a conjunction cannot be raised by improving non-constraint dimensions | the hard-gate design in DD-08 and Section 6 | only as strong as the constraint definitions |

### 10.2 Framework references

| Reference | What it supports in this sheet | What it does not establish |
|---|---|---|
| [NIST AI Risk Management Framework 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf), January 2023 | vocabulary for govern, map, measure, and manage activities across a system lifecycle | that any candidate is safe, compliant, or suitable |
| [NIST SP 800-161 Rev. 1](https://csrc.nist.gov/pubs/sp/800/161/r1/final), May 2022 | supply-chain risk considerations, including supplier and component visibility | that a supplier's controls operate in a specific tenant |
| [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final), 2020 with updates | control catalog vocabulary for access, audit, and incident response | an audit result for any product |
| [ISO/IEC 25010:2011](https://www.iso.org/standard/35733.html) | quality-attribute vocabulary such as functional suitability and reliability | measurement of a candidate without a local protocol |
| [Google SRE Workbook, Implementing SLOs](https://sre.google/workbook/implementing-slos/) | user-centered reliability metrics and error budgets | a specific availability number for any service |
| [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | leakage and train-test contamination mechanisms | anything about a vendor's model |

Accessed 2026-11-09. These are vocabulary and method references.
Their publication does not certify any candidate, and their scope statements must be read before they are cited in a decision record.
A framework reference answers "what questions should we ask"; only a candidate-specific artifact answers "what is true here".

### 10.3 Derived properties worth memorizing

**Interval width scales as `1/sqrt(n)`.** Quadrupling cases halves the width; the effort grows linearly while the precision improves sublinearly. This is the single most useful planning fact in this sheet.

**The zero-failure bound never reaches zero.** From DD-06, `q_upper ≈ 3/n`, so 300 clean cases bound the rate at roughly 1%. Any claim of "zero risk" from a test is a category error.

**Expected loss is linear in the failure rate but the monthly outcome is not.** The expectation is exact for the mean; the realized month varies. A candidate with positive expected value can still produce a bad month, and a gate should consider the variance when harm is severe.

**A paired comparison has power driven by discordant pairs.** Two arms with 100 identical cases may have only 10 discordant pairs, and 10 pairs cannot resolve a 2-point difference. The effective sample size for a paired comparison is the number of discordant pairs, not the number of cases.

**A conjunction's failure probability is at least the maximum of its terms.** If five independent gates each pass with probability 0.95, the joint pass probability is `0.95^5 = 0.7738`, so roughly one candidate in four or five fails at least one gate even when each is individually likely to pass. Under dependence the joint can be worse or better, which is why correlated gates must be assessed together rather than multiplied.

**Break-even volume diverges as the per-unit saving approaches zero.** From DD-10, `N_be = (F + H*w)/(c_base - k_eff)`; as the denominator goes to zero, the required volume grows without bound, and a negative denominator means no volume suffices.

### 10.4 Where the results stop applying

The interval formulas assume independent, identically distributed cases.
Clustered sampling, repeated measures on the same request, or multi-turn sessions violate this and require a design effect adjustment.
The cost identities assume linear pricing within the modeled range; tier changes, committed-use discounts, and quota cliffs break linearity.
The concurrency formula assumes stationarity; bursty arrival processes require a queueing model with arrival variability, not the mean.
The expected-loss model assumes a defensible average harm; when harm is heavy-tailed, the mean is a poor decision statistic and a quantile or a cap is better.
The evidence-readiness proportion assumes the artifact list is complete; a missing item on the list cannot lower the score.

**Takeaway:** Use these results to structure the argument, and state the independence, linearity, and stationarity assumptions that each one inherits.

## 11. Interview / Exam Questions

The fifteen questions below mix recall, application, calculation, and judgement.
Each answer states the reasoning, not just the conclusion, and names the red flag that a weak answer would raise.

### 11.1 Recall and definition

**Q1. What are the six slots a technology claim must fill before it can be tested?**
Unit, population, window, metric with numerator and denominator, comparator, and consequence of pass, fail, or inconclusive.
The reasoning is that these six slots exactly correspond to the questions a scorer must answer to produce a number: what am I scoring, drawn from where, over what period, by what rule, against what reference, and what happens next.
Missing any one of them makes the result either uncomputable or uninterpretable.
**Red flag:** an answer that lists "accuracy, latency, cost, security" — those are dimensions, not slots, and they do not make a claim testable.

**Q2. Why must a hard gate be non-compensatory?**
Because the harm classes are not exchangeable: a prohibited disclosure and a missed citation are not the same currency, so there is no defensible exchange rate that would let quality offset the disclosure.
A compensatory score implicitly asserts such a rate exists.
Separating gates preserves the ability to say "high quality, still unacceptable," which is precisely the sentence that prevents an incident.
**Red flag:** proposing weights such as "quality 60%, security 40%," which manufactures the exchange rate that the gate structure exists to prevent.

**Q3. What does the scope of an evidence artifact mean, and why does it expire?**
Scope is the exact product, version, region, tier, and configuration an artifact describes.
Expiry follows from dependency change: the artifact supports the claim only while the dependencies it was produced under remain unchanged.
A model version change, a new connector, a subprocessor addition, a price change, or a workload shift can each invalidate a different artifact.
**Red flag:** "it was audited last year" as a complete answer, with no statement of what changed since.

### 11.2 Calculation

**Q4. A protocol has 60 cases and requires at least 95% pass. How many failures are allowed, and what happens at 57 passes?**
`floor(60*0.05) = floor(3.0) = 3` failures allowed.
57 passes means 3 failures, which equals the allowed count, so the gate passes.
56 passes means 4 failures, `56/60 = 93.33%`, below the gate, so it fails.
Note that `3/60 = 5.0%` exactly, so the boundary is attained, which does not happen for every `n`.
**Red flag:** computing the threshold as "95% of 60 = 57, so 57 is the passing count" without checking whether the failure count or the pass count is being floored.

**Q5. From 91 passes in 100 cases, compute the 95% Wald interval and state whether a 90% gate is met.**
`p_hat = 0.91`; `p_hat*(1-p_hat) = 0.91*0.09 = 0.0819`; `SE = sqrt(0.0819/100) = 0.02862`; half-width `= 1.96*0.02862 = 0.05609`; interval `[0.8539, 0.9661]`.
The point estimate clears 0.90, and the interval contains values below 0.90.
So the administrative count gate is met while the interval does not exclude a rate below the threshold, which means the decision should either collect more cases or record the margin explicitly.
**Red flag:** reporting "91% ≥ 90%, so yes" with no interval and no comment on the margin.

**Q6. How many cases are needed to estimate a 91% rate within ±5 points at 95% confidence?**
`n = z^2*p(1-p)/h^2 = 3.8416*0.0819/0.0025 = 0.31463/0.0025 = 125.85`, so 126 cases.
The practical implication is that a 100-case protocol misses the stated precision by about 26% of the required sample.
**Red flag:** answering "100 is enough because it gives a round number," or applying the formula without noticing that `p` near 0.5 gives the conservative `384` cases.

**Q7. A candidate shows zero prohibited disclosures in 40 adversarial tests. Bound the rate.**
`q_upper = 1 - 0.05^(1/40) = 1 - exp(-2.9957/40) = 1 - exp(-0.074893) = 1 - 0.927842 = 0.072158`, about 7.2%.
At 20,000 monthly requests that upper bound corresponds to roughly 1,443 events in expectation, which shows how little a 40-case clean run establishes.
**Red flag:** stating the failure rate is zero, or quoting the bound without stating it is a one-sided 95% upper limit under independence.

**Q8. Fixed cost 4,000 per month, review labor 2,400 per month, baseline 2.10 per unit, effective candidate cost 0.30 per unit. What is break-even volume?**
`N_be = (4,000 + 2,400)/(2.10 - 0.30) = 6,400/1.80 = 3,555.6`, so about 3,556 units per month.
Above that volume the candidate saves money on a linear model; below it, the fixed costs dominate.
**Red flag:** computing `(4,000+2,400)/2.10 = 3,047.6` by omitting the candidate's variable cost from the denominator.

**Q9. A cost model predicts 5,900; the invoice for the first month is 6,750. Name three checks in order.**
First reconcile line by line to the model terms to find any charge with no model counterpart.
Second check the retry and duplicate-request rate, since retries multiply the variable term.
Third check the output-length distribution against the quoted band, because length drives token cost directly.
Only after those should demand assumptions and tier changes be examined.
**Red flag:** immediately renegotiating the price, which fixes a symptom and leaves the model unfitted to the actual usage.

### 11.3 Judgement and diagnosis

**Q10. A candidate scores 93/100 and performs one unauthorized action in the adversarial set. What is the decision?**
Reject the current posture or restrict to propose-only, because the hard gate fails and no quality margin compensates.
The record should state that capability evidence is favorable and the control evidence is not, so a later reader does not conclude the technology is weak.
The remediation path is to remove the action permission, retest, and then re-evaluate under the restricted posture.
**Red flag:** averaging the unauthorized action into a composite score, or proposing "monitor closely and proceed."

**Q11. Two arms: candidate passes 88% of 1,000 cases in segment A and 96% of 100 in segment B; baseline passes 90% of A and 94% of B. Who wins, and what does it mean?**
Candidate `(880 + 96)/1,100 = 976/1,100 = 88.73%`; baseline `(900 + 94)/1,100 = 994/1,100 = 90.36%`.
The candidate loses overall and loses in both segments, so there is no reversal here — but the exercise must be run because the reverse can occur.
If the candidate had won overall while losing in both segments, the cause would be a different segment mix or weighting, and the correct decision would follow the segment that matters operationally.
**Red flag:** declaring a winner from the aggregate without computing segment rates and counts.

**Q12. A pilot shows the candidate at 92% and the incumbent at 94% on the same 100 cases, with 4 candidate-only passes and 6 baseline-only passes. What can you conclude?**
Very little: the effective comparison rests on 10 discordant pairs, and a sign test on 10 pairs cannot separate these arms.
The honest conclusion is "no demonstrated quality advantage on this sample," and the next step is either more cases or a decision that does not depend on the difference.
**Red flag:** "the incumbent is 2 points better, so keep it," which treats a paired comparison as if it were two independent estimates with negligible error.

**Q13. The security page says the product is encrypted in transit and at rest. What is still unknown?**
Whether the organization's specific tenant, region, connector set, logging path, retention setting, subprocessor list, and support-access model match the general statement.
Whether the field-level route for the specific data class is permitted under policy.
Whether the encryption keys are supplier-managed and what the response is to a key-compromise scenario.
Whether the statement covers the integration the organization builds on top.
**Red flag:** treating the security page as a control test, or answering "encryption means it is secure."

**Q14. A reversible, low-consequence decision has been in review for eleven weeks because someone wants more evidence. What do you do?**
Ask which specific artifact is missing and which decision it would change.
If no artifact would change the decision, the blocker is authority rather than evidence, and the correct action is to assign a decision owner with stated residual-risk authority and set a date.
If a missing artifact is a hard-gate item such as a prohibited-route test, the delay is justified and the record should say so.
Either way, the record must name the owner, the outstanding item, and the decision date.
**Red flag:** escalating for "more confidence," which has no terminating condition and converts diligence into avoidance.

### 11.4 Changed constraints

**Q15. The candidate will now act automatically on a reversible action instead of proposing it. Which parts of the protocol change?**
Add an idempotency test (duplicate request produces one effect), an audit reconstruction test (every action traceable to input, version, and approver), a permission-scope test (out-of-scope action fails closed), and a rollback or compensation test.
The quality rubric gains action-correctness components, and the monitoring gate requires an alert that fires on out-of-scope attempts.
The evidence gate gains a credential-scope mapping.
**Red flag:** adding only a "human in the loop" checkbox without testing duplicate effects or audit reconstruction.

### 11.5 Scoring guidance for interviewers

A strong answer names the artifact, states the arithmetic, and names the condition under which the conclusion would change.
An adequate answer reaches the right conclusion but cannot state the boundary.
A weak answer uses exactly the adjectives the question was written to eliminate.
When a candidate volunteers a limitation that the question did not ask for, that is a strong signal, because boundary awareness is the whole skill.
When a candidate offers a number without a denominator, ask for the denominator before discussing the number.
When a candidate proposes a weighted score, ask what weight they would assign to an unauthorized action and watch the structure collapse.
When a candidate says "it depends," ask what it depends on and which observation would settle it.

The questions are also useful as a self-check before a real review.
A reviewer who cannot answer Q1, Q2, and Q3 for their own case has not yet built a testable protocol.
A reviewer who cannot answer Q5 or Q7 cannot defend the sample size.
A reviewer who cannot answer Q10 or Q14 is likely to produce a decision record that nobody can act on.

**Takeaway:** The distinguishing skill is not computing the proportion; it is naming the boundary at which the proportion stops supporting the decision.

## 12. Quick Reference Card

| Step | Ask / compute | Required artifact | Stop condition | Next action |
|---|---|---|---|---|
| 1. Intake | What are the exact words, version, and owner? | claim record with verbatim text | claim has no owner or version | return to sponsor |
| 2. Interpret | Unit, population, window, metric, comparator, consequence? | operational definition sheet | any slot empty | request definition, mark OPEN |
| 3. Freeze | Are cases, rubric, and gate arithmetic fixed before execution? | signed protocol with `n`, `p_min`, `floor(n*(1-p_min))` | cases still being written | do not run |
| 4. Gate | Which constraints are boolean and non-negotiable? | hard-gate list with control tests | any hard gate fails | reject, narrow, or remediate |
| 5. Score | `p_hat = n_pass/n_test`, plus interval from `SE = sqrt(p_hat(1-p_hat)/n)` | scored cases, both scorers, adjudication log | interval wider than decision margin | add cases or reframe as pilot |
| 6. Compare | Paired results against the baseline on identical units | paired counts and discordant-pair count | fewer than ~20 discordant pairs | conclude "no demonstrated difference" |
| 7. Price | `TCO = F + N*k*(1+r) + H*w + R`, then `c_unit` and break-even volume | dated price schedule plus observed retry rate | observed retry or length outside the quoted band | rebuild the model, do not move the ceiling |
| 8. Bound | `q_upper = 1 - 0.05^(1/n)` for zero-failure claims | zero-failure count with `n` | bound is wide relative to harm | add cases or add a detector |
| 9. Ledger | What does each artifact support, for which version, until when? | versioned ledger with expiry triggers | artifact scope unmatched to deployment | mark OPEN and block promotion |
| 10. Decide | Adopt, pilot with stop conditions, retain baseline, or collect evidence? | decision record with owner, date, residual risk | no named owner or no rollback path | retain baseline by default |

**Card boundary.** The formulas on this card assume independent cases, linear cost within the modeled range, and a single failure class with a defensible average harm.
When those assumptions fail, the card gives the structure of the argument but not the number.

**One-sentence version.** Rewrite every adjective as a unit, a population, a metric, and a comparator; declare the gate arithmetic before running; keep hard constraints outside any average; price the retry rate as well as the unit price; and record what the evidence does not cover, with an owner and an expiry trigger.

