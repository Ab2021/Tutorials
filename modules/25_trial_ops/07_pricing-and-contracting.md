# Pricing and Contracting: Value-Based Agreements and Risk-Sharing Arrangements

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 07 of 07
> Estimated study time: 4 hours
> Prerequisites: topics 04 and 05
> Version 2026.3

## 1. Why this topic exists

A price is not a number. It is a set of contingent payments: a base price, a rebate, a cap, a
performance condition, and a reconciliation process. The contract determines what the manufacturer
actually receives and what the payer actually pays, and it allocates the risk that the product
performs differently in practice than it did in the trial.

Risk-sharing arrangements are attractive in theory. They let a payer pay more when a product works
and less when it does not, and they let a manufacturer with uncertain real-world evidence obtain
access it might otherwise be denied. In practice, they are difficult to measure, expensive to
administer, and vulnerable to disputes. The published evidence on their financial impact is modest,
and the administrative burden is real.

The purpose of this topic is to make the contract quantitative. A value-based agreement is a bet on
a measured outcome. The bet has a probability, an expected payment, a variance, and a cost of
administration. A sponsor that cannot compute those four numbers is negotiating with a story.

### 1.1 The pattern that repeats

1. A payer is uncertain whether the trial effect will replicate in its population.
2. The manufacturer offers an outcome-based rebate to secure access.
3. The outcome is measured in a registry that is incomplete, or with a surrogate that is contested.
4. The threshold is set where the expected rate makes the rebate unlikely to trigger.
5. The contract is announced as innovative, and the effective price barely changes.
6. Two years later, neither side can demonstrate that the contract improved access or reduced
   uncertainty.

The failure is not the idea of risk-sharing. It is the absence of a measurement design that could
answer the question, and a threshold that was chosen for negotiation rather than for inference.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Manufacturer | Predictable revenue | Depends on payer data and behaviour |
| Payer | Budget certainty | Depends on manufacturer reporting |
| Patient | Access | Depends on a contract they never see |
| Clinician | Prescribing clarity | Criteria may be opaque |
| Regulator | Nothing directly | Contract is outside the label |
| Evaluator | Ability to assess the contract | Data are confidential |

### 1.3 The industry question

> What outcome, measured in which population, over what period, at what threshold, with what rebate,
> and with what evidence that the measurement is credible?

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** pricing arrangements into financial, performance-based, and service-based types, and
   explain what risk each transfers.
2. **Design** an outcome metric with a defined population, threshold, measurement window, and
   attribution rule.
3. **Compute** the probability that a performance condition triggers, the expected rebate, and the
   total cost including administration.
4. **Evaluate** whether a contract is adequately powered to distinguish performance from noise.
5. **Compare** an outcome-based contract with a simple rebate on expected cost and variance.
6. **Identify** the legal constraints on value-based arrangements in the United States and the
   European Union.
7. **Assess** the published evidence on managed entry agreements and explain why financial impact has
   been modest.
8. **Negotiate** a contract with a walk-away position and a dispute-resolution clause.

## 3. The core mental model

### 3.1 Risk allocation

```text
                     Risk transferred to manufacturer
                                   ^
                                   |
   Simple discount  ---------------+---------------  Full outcomes guarantee
   (price risk)                    |                   (performance risk)
                                   |
   Volume rebate  ------------------+-----------------  Payment by result
   (volume risk)                   |                   (patient-level outcome)
                                   |
                                   +------------------> Risk transferred to payer
```

Every contract sits somewhere on this line. A simple discount transfers price risk to the
manufacturer and leaves volume and performance risk with the payer. A payment-by-result contract
transfers performance risk to the manufacturer and leaves the payer with the risk that the
measurement is wrong.

### 3.2 The contract as a measurement system

```text
 Population -> Treatment -> Outcome -> Measurement -> Payment
      |            |           |            |             |
   eligible     adherence   endpoint     data source   reconciliation
   criteria     duration    definition   completeness   formula
```

A contract fails if any link is weak. The most common weakness is the measurement link: the outcome
is defined in a way that cannot be observed reliably in routine data, or the population observed is
not the population treated.

### 3.3 The value of risk-sharing

```text
Value of risk-sharing = value of information + risk premium
                        - administration cost - measurement error cost
```

The value of information is what the payer or manufacturer would pay to learn the true effect. The
risk premium is what each side pays to avoid the downside. The administration and measurement costs
are certain, and they are often larger than the value of the information for a single product.

A contract is worth doing when the uncertainty is material, the outcome is measurable, and the
administrative cost is small relative to the value at stake. Those conditions are less common than
the popularity of the arrangement suggests.

### 3.4 The design triangle

| Requirement | Question | Failure mode |
|---|---|---|
| Measurability | Can the outcome be observed in routine data? | Registry incomplete; surrogate contested |
| Attribution | Can the effect be attributed to the product? | Population differences; co-interventions |
| Enforceability | Can the payment be reconciled and disputed? | Ambiguous formula; no audit right |

All three must hold. A contract with a measurable outcome that cannot be attributed, or an
attributable outcome that cannot be measured, is not enforceable in practice even if it is
enforceable in law.

### 3.5 The expected payment

For a binary trigger, the expected rebate is:

```text
Expected rebate = P(trigger) x rebate rate x eligible revenue
```

The probability depends on the true outcome rate, the threshold, and the sample size. A threshold
far above the expected rate produces a low probability and a small expected rebate, which may make
the contract look cheap to the manufacturer and worthless to the payer.

### 3.6 Variance matters as much as the mean

Two contracts can have the same expected cost and very different variance. A manufacturer with a
thin balance sheet may prefer a certain rebate to a variable one, even at a higher expected cost.
A payer may prefer a variable payment because it protects the budget in a bad year.

The contract should be evaluated on both mean and variance, and the party bearing the variance
should be the one best able to absorb it.

### 3.7 Evidence on managed entry agreements

Managed entry agreements are used in at least 28 of 41 OECD and EU countries, but financial
agreements dominate. Performance-based agreements are concentrated in a few countries, notably
Italy, where they are administered through AIFA's monitoring registries. Published evaluations have
found that the financial recovery is small relative to total expenditure and that the data are
often not used to reduce uncertainty.

The lesson is not that risk-sharing never works. It is that the measurement and administration
costs are easy to underestimate, and that a contract without a plan to use the data is a price
discount with extra steps.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Value-based agreement (VBA) | Contract linking payment to value or outcome | Used for any rebate | Agreement |
| Risk-sharing agreement | Contract that allocates performance risk | Confused with a discount | Agreement |
| Managed entry agreement (MEA) | Formal arrangement for entry with conditions | Used as a synonym for VBA | Agreement |
| Performance-based agreement | Payment tied to a measured outcome | Confused with coverage with evidence | Agreement |
| Payment by result (PbR) | Rebate for patients who do not achieve the outcome | Applied to population averages | Agreement |
| Cost-sharing agreement | Manufacturer pays a share of treatment cost | Confused with a patient copay | Agreement |
| Capping | Maximum manufacturer liability or payer spend | Confused with a rebate | Agreement |
| Price-volume agreement | Price falls as volume rises | Confused with a simple rebate | Agreement |
| Coverage with evidence development (CED) | Coverage conditional on data collection | Confused with a registry | Policy |
| Outcomes guarantee | Manufacturer refunds if outcome not achieved | Confused with a warranty | Agreement |
| Warranty | Payment or replacement if the product fails | Confused with a guarantee | Agreement |
| Annuity payment | Payment spread over time | Confused with instalments | Payment |
| Rebate | Post-sale payment back to the payer | Confused with a discount | Currency |
| Clawback | Recovery of part of the payment | Confused with a rebate | Currency |
| Reconciliation | Calculation of the amount due under the contract | Treated as an accounting task | Process |
| Attribution | Judgement that the outcome is due to the product | Assumed | Method |
| Counterfactual | What would have happened without the product | Not observed | Concept |
| Surrogate endpoint | Intermediate marker used instead of a final outcome | Assumed validated | Endpoint |
| Registry | Systematic collection of patient data | Assumed complete | Data source |
| Patient-level data | Data at the individual patient level | Assumed available | Data type |
| Gaming | Manipulation of measurement to trigger payment | Confused with error | Behaviour |
| Best price | Lowest price to a payer under Medicaid rules | Confused with net price | Currency |
| Medicaid VBP | Value-based purchasing arrangements under Medicaid | Confused with commercial VBA | Policy |
| Anti-kickback | US prohibition on remuneration for referrals | Confused with a rebate | Law |
| GDPR | EU data protection regulation | Treated as an IT issue | Law |
| Effective price | Price after all rebates and adjustments | Confused with list price | Currency |
| Contract year | Period over which the contract is reconciled | Confused with calendar year | Period |
| Exit clause | Condition under which the contract ends | Absent | Clause |

### 4.1 Financial and performance-based agreements

A financial agreement changes the price or the budget exposure without tying payment to a clinical
outcome. Examples are discounts, rebates, price-volume agreements, and caps. A performance-based
agreement ties at least part of the payment to a measured clinical outcome, either at the patient
level (payment by result) or at the population level (coverage with evidence development).

The distinction matters because the data requirements are different. A financial agreement needs
claims or sales data; a performance agreement needs clinical outcome data, which are harder to
obtain and validate.

### 4.2 Payment by result and cost-sharing

Payment by result refunds the cost for patients who do not achieve the outcome. Cost-sharing has the
manufacturer pay a share of the cost for all treated patients, often for a defined period. PbR
requires patient-level outcome data; cost-sharing requires only treatment data. PbR transfers more
risk but is harder to administer.

### 4.3 Rebate, discount, and clawback

A discount reduces the invoice price. A rebate is paid after the sale. A clawback recovers part of a
payment already made. The accounting treatment and the cash timing differ, and so does the
incentive: a rebate can be designed to reward volume, while a clawback can be designed to recover
money when an outcome is not met.

## 5. The mechanics, step by step

This procedure designs and implements a value-based or risk-sharing contract. It assumes a product
with payer uncertainty about real-world performance and a payer willing to consider a conditional
arrangement.

### Step 1. Identify the uncertainty worth sharing

**Input:** the trial evidence, the payer's concerns, and the expected real-world performance.

**Output:** a statement of the specific uncertainty, why it matters to the payer, and what evidence
would resolve it.

**Rationale:** A contract should target a real uncertainty. If the payer's concern is budget impact
rather than effectiveness, an outcome-based contract addresses the wrong problem; a cap or a
volume-based rebate is more appropriate.

### Step 2. Choose the contract type

**Input:** the uncertainty, the data available, and the legal constraints.

**Output:** a contract type with a rationale and a rejected alternative.

**Rationale:** The type determines the data and the administration. A financial agreement is simple
and certain; a performance agreement is complex and uncertain. The choice should be justified by
the value of the information, not by the novelty of the arrangement.

### Step 3. Define the outcome and the threshold

**Input:** the clinical evidence and the payer's decision rule.

**Output:** a metric with a definition, a measurement window, a threshold, and a rebate formula.

**Rationale:** The threshold determines the probability of payment. A threshold set at the trial's
point estimate will trigger about half the time; a threshold set far above it will almost never
trigger. The threshold should be chosen so that the contract is meaningful to both parties, and it
should be justified by clinical relevance, not by the negotiation.

### Step 4. Define the population and attribution

**Input:** the label, the payer's covered population, and the expected off-label use.

**Output:** inclusion and exclusion criteria, a run-in period, and a rule for attributing the outcome
to the product.

**Rationale:** The population observed in a registry may differ from the trial population. If the
contract includes patients who would not have been treated in the trial, the outcome rate may fall
for reasons unrelated to the product, and the manufacturer will pay for a population difference.

### Step 5. Design the data collection

**Input:** the available registries, claims data, and electronic health records.

**Output:** a data specification with sources, completeness checks, and a governance agreement.

**Rationale:** The contract is only as good as the data. A registry that captures 60% of treated
patients will produce an estimate with unknown bias. A claims-based outcome may be coded
inconsistently. The data specification should state the expected completeness and the handling of
missing data before the contract is signed.

### Step 6. Compute the expected payment

**Input:** the expected outcome rate, the threshold, the sample size, and the rebate formula.

**Output:** the probability of triggering, the expected rebate, the variance, and the total cost
including administration.

**Rationale:** The expected payment is the number the contract is worth. A contract with a 4% chance
of triggering a 30% rebate costs about 1.2% of revenue in expectation, which may be less than a
simple discount and may buy the same access.

### Step 7. Price the risk premium

**Input:** the variance of the payment and the parties' risk tolerance.

**Output:** the certainty equivalent of the contract for each party.

**Rationale:** A risk-averse manufacturer values a variable payment below its expected value. The
difference is the risk premium. A contract that is fair in expectation may be unacceptable to a
company that cannot absorb the downside.

### Step 8. Check the legal constraints

**Input:** the jurisdiction and the contract terms.

**Output:** a legal assessment covering best price, anti-kickback, data protection, and
competition law.

**Rationale:** In the United States, Medicaid's best-price rule historically discouraged
outcome-based contracts because a single low price could reset the best price. The 2020 value-based
purchasing rule allows multiple best prices for arrangements offered to all states, subject to
reporting and record-keeping. In the EU, patient-level data collection must satisfy the GDPR. These
constraints shape what is possible, not only what is legal.

### Step 9. Negotiate the terms

**Input:** the contract design and the walk-away positions.

**Output:** an agreed contract with a term, a reconciliation date, a dispute process, and an exit
clause.

**Rationale:** The negotiation is about the threshold, the rebate, the population, and the data. A
manufacturer that concedes a low threshold and a broad population has accepted a large expected
payment; one that concedes a high threshold has accepted a contract that may never trigger.

### Step 10. Build the operations

**Input:** the contract and the data specification.

**Output:** a process for enrolment, data capture, reconciliation, invoicing, and audit.

**Rationale:** The operational build is where most contracts fail. The registry must be accessible
to the sites, the data must be linked to the rebate calculation, and the finance team must be able
to invoice or credit the payer. A contract without an operational owner is a press release.

### Step 11. Monitor and reconcile

**Input:** the data and the contract terms.

**Output:** a reconciliation statement with the outcome rate, the payment due, and the supporting
data.

**Rationale:** Reconciliation should be scheduled and auditable. A reconciliation that depends on a
manual spreadsheet and a goodwill conversation will be disputed.

### Step 12. Handle disputes and outliers

**Input:** the reconciliation statement and any disagreement.

**Output:** a resolution under the contract's dispute clause.

**Rationale:** Disputes cluster around attribution, missing data, and outliers. The contract should
specify how each is handled before the first reconciliation, not after.

### Step 13. Evaluate the contract

**Input:** the outcomes, the payments, the administration cost, and the access achieved.

**Output:** an evaluation of whether the contract delivered its objectives.

**Rationale:** The evaluation is the only way to learn whether the arrangement was worth doing. It
should compare the outcome with the counterfactual and report the administrative cost, which is
usually omitted.

### Step 14. Decide on renewal or exit

**Input:** the evaluation and the changed evidence.

**Output:** a decision to renew, renegotiate, or exit.

**Rationale:** A contract designed to resolve uncertainty should end when the uncertainty is
resolved. An evergreen contract becomes a price discount with additional reporting.

### Step 15. Govern and disclose

**Input:** the contract terms and the legal requirements.

**Output:** a governance record and, where required, a public disclosure.

**Rationale:** Confidentiality is common but not universal. Some jurisdictions require disclosure of
the existence and terms of managed entry agreements. A governance record that cannot be disclosed
without revealing patient data is a design flaw.

### 5.15 Decision table: contract type

| Type | Uncertainty addressed | Data needed | Administration |
|---|---|---|---|
| Simple discount | Price | None | Low |
| Volume rebate | Budget | Sales | Low |
| Price-volume agreement | Budget | Sales | Low to moderate |
| Cap | Budget | Sales | Moderate |
| Payment by result | Effectiveness in practice | Patient-level outcomes | High |
| Coverage with evidence development | Effectiveness and budget | Registry | High |
| Service-based agreement | Delivery and support | Service data | Moderate |

### 5.16 Decision table: outcome metric

| Metric | Measurability | Attribution | Best fit |
|---|---|---|---|
| Overall survival | High in a registry | Moderate | Serious diseases with clear endpoints |
| Progression-free survival | Moderate | Moderate | Oncology, with imaging data |
| Hospitalisation | High in claims | Low | Chronic diseases with clear events |
| Patient-reported outcome | Moderate | Low | Symptom-driven conditions |
| Adherence | High in pharmacy data | High | Products with adherence-dependent effect |
| Cost offset | High in claims | Low | Budget-driven contracts |

### 5.17 Decision table: dispute resolution

| Dispute | Likely cause | Resolution |
|---|---|---|
| Outcome rate differs from expectation | Population or data differences | Independent re-analysis |
| Missing data | Registry incompleteness | Pre-specified imputation; sensitivity analysis |
| Attribution | Co-interventions or off-label use | Exclude the affected patients; document |
| Rebate calculation | Formula ambiguity | Reconcile against the contract definition |
| Data access | Confidentiality or governance | Independent custodian |

### 5.18 Edge cases

**Payer rejection.** If the payer rejects the contract, the product may still be approved but not
covered. The contract is a coverage tool, not an approval tool.

**Price negotiation failure.** If the parties cannot agree on the threshold or the rebate, a simple
discount may be the fallback. The fallback should be priced before the negotiation starts.

**Small population.** A rare disease may have too few patients for a statistically meaningful
outcome. A registry across countries or a longer window may help, or the contract may need to rely
on a surrogate.

**Cross-border contracts.** A contract in one country may be referenced by another, or may conflict
with the reference-pricing rules. The terms should be reviewed for spillover.

**Generic or biosimilar entry.** A contract that runs beyond loss of exclusivity may be unenforceable
or irrelevant. The term should be aligned with the product lifecycle.

### 5.19 Specify the reconciliation

**Input:** the contract terms, the data specification, and the finance process.

**Output:** a reconciliation procedure that states the period, the data cut, the calculation, the
evidence package, and the payment timeline.

**Rationale:** Most disputes are caused by ambiguity in the reconciliation, not by disagreement
about the outcome. The procedure should state which patients are included, which data version is
used, how missing values are handled, and who signs off.

A reconciliation that depends on a manual spreadsheet is a reconciliation that will be questioned.
The calculation should be reproducible from the data package, and the data package should be
retained for audit.

### 5.20 Evaluate whether the contract worked

**Input:** the outcome data, the payments, the administrative cost, and the access metrics.

**Output:** an evaluation against the contract's objectives, with a counterfactual where possible.

**Rationale:** A contract can meet its terms and fail its purpose. The evaluation should ask whether
the arrangement increased access, reduced uncertainty, or reduced budget risk, and at what cost. A
contract that transferred a small amount of money at a large administrative cost has not worked,
even if the reconciliation was clean.

The counterfactual is difficult: the alternative was a different contract or no coverage. An
explicit comparison with the fallback simple discount is the minimum.

### 5.21 Design the data governance

**Input:** the data specification, the legal regime, and the parties' access needs.

**Output:** a data governance agreement naming the controller, the custodian, the permitted uses,
the retention period, and the audit rights.

**Rationale:** A performance contract requires patient-level data to flow between organisations that
may not trust each other. The governance agreement determines whether the measurement is credible
and whether it is lawful.

In the EU, the GDPR requires a lawful basis, minimisation, and a defined retention period. In the
United States, the contract must respect anti-kickback rules and, for Medicaid, the value-based
purchasing reporting requirements. A governance design that ignores these constraints produces a
contract that cannot be operated.

**Branch point:** an independent custodian can hold the data and calculate the rebate without
disclosing patient-level records to either party. It adds cost and removes a source of dispute.

### 5.22 Model the contract in the payer's budget

**Input:** the expected payment, the variance, and the payer's budget.

**Output:** a distribution of the payer's net spend under the contract, compared with the simple
rebate and with no contract.

**Rationale:** A payer cares about the budget in a bad year, not only the expected cost. A contract
with the same expected cost and a wider variance may be less attractive, because the budget must
absorb the upper tail. The model should report the 90th or 95th percentile of spend, not only the
mean.

The comparison should include the administrative cost on both sides. A contract that shifts cost
between organisations but adds $1.2 million of administration has made the system poorer unless it
also improved access or evidence.

### 5.23 Document the contract

**Input:** the agreed terms, the data governance, and the operational process.

**Output:** a contract file containing the agreement, the measurement plan, the reconciliation
procedure, the legal review, and the evaluation plan.

**Rationale:** A contract that exists only as a signed agreement and a spreadsheet is not operable.
The file is what allows a new team to run the reconciliation, an auditor to check it, and an
evaluator to judge whether it worked.

The file should also record the rejected alternatives and the reason. When the contract is reviewed
in two years, the question will be whether the assumptions that justified it still hold.

### 5.24 Metrics for a value-based contract

| Metric | Definition | Use |
|---|---|---|
| Trigger probability | Modelled chance the condition is met | Expected cost |
| Expected rebate | Probability x rebate x revenue | Budget |
| Administrative cost | Cost to operate the contract | Net value |
| Data completeness | Share of treated patients captured | Measurement validity |
| Reconciliation time | Days from period end to payment | Operational performance |
| Dispute rate | Share of reconciliations disputed | Contract clarity |
| Access effect | Change in treated patients | Contract objective |

## 6. Worked example

### 6.1 The product and the proposed contract

A therapy for a chronic condition costs $30,000 per patient per year and is expected to treat 1,000
patients in the contract population. The payer is concerned that the hospitalisation rate observed
in the trial, 15%, will be higher in routine practice.

The proposed agreement is a population-level payment by result:

- if the observed hospitalisation rate at 12 months exceeds 17%, the manufacturer rebates 30% of
  revenue for the year;
- the outcome is measured in a registry covering the treated population;
- the administrative cost of the registry and reconciliation is $1,200,000 per year.

The fallback offered by the payer is a simple 5% rebate.

### 6.2 Probability that the condition triggers

With 1,000 patients and a true rate of 15%, the standard error of the observed rate is:

```text
SE = sqrt(0.15 x 0.85 / 1,000) = 0.0113
```

The probability that the observed rate exceeds 17% is the upper tail of a normal distribution with
mean 0.15 and standard deviation 0.0113:

```text
z = (0.17 - 0.15) / 0.0113 = 1.77
P(trigger) = 1 - Phi(1.77) = 0.038
```

There is a 3.8% chance that the rebate is triggered.

### 6.3 Expected cost against the fallback

```text
Revenue = 1,000 x $30,000 = $30,000,000
Expected rebate = 0.038 x 0.30 x $30,000,000 = $344,351
Administrative cost = $1,200,000
Total expected cost = $1,544,351
Simple 5% rebate = 0.05 x $30,000,000 = $1,500,000
```

The outcome-based contract costs $1.54 million in expectation, almost exactly the same as the simple
5% rebate. It is not cheaper. Its value, if any, is in the access it buys or the uncertainty it
reduces, not in the expected payment.

### 6.4 Diagnostic 1: the threshold is too far from the expected rate

A threshold of 17% against an expected rate of 15% is 1.77 standard errors away. The contract
triggers in less than 4% of scenarios, so the payer receives almost no protection against the
scenario it was worried about, and the manufacturer pays almost nothing.

The contract is cosmetic. It looks like risk-sharing because it has an outcome and a rebate, but the
expected transfer is small and the probability of a meaningful transfer is smaller.

A useful diagnostic is the ratio of the threshold gap to the standard error. A gap below one
standard error makes the contract a coin flip; a gap above three makes it a near-certainty that it
will not trigger.

### 6.5 Diagnostic 2: the contract is underpowered

The contract uses 1,000 patients, but detecting a difference between a 15% and a 17% hospitalisation
rate at 80% power and a two-sided 5% significance level requires about 5,269 patients.

```text
n = 2 x (1.96 + 0.84)^2 x 0.16 x 0.84 / (0.02)^2 = 5,269
```

With 1,000 patients, the observed rate is dominated by noise. A contract that triggers on noise
pays for a random fluctuation; a contract that does not trigger may miss a real deterioration. The
measurement cannot support the decision it is meant to inform.

The remedy is not necessarily a larger trial. It can be a longer measurement window, a registry
covering multiple years, a surrogate with less noise, or a contract that shares risk at the
population level rather than the patient level.

### 6.6 Vary one assumption: threshold at 16%

If the parties agree a threshold of 16% instead of 17%, with the same true rate and sample size:

```text
z = (0.16 - 0.15) / 0.0113 = 0.89
P(trigger) = 1 - Phi(0.89) = 0.188
Expected rebate = 0.188 x 0.30 x $30,000,000 = $1,691,213
Total expected cost = $1,691,213 + $1,200,000 = $2,891,213
```

The contract now costs $2.89 million, nearly double the simple rebate. The threshold change of one
percentage point moves the expected cost by $1.35 million, because it moves the threshold from 1.77
to 0.89 standard errors from the mean.

The conclusion changes: the contract is no longer cosmetic, and the manufacturer should prefer the
simple rebate unless the access benefit is worth more than $1.39 million.

### 6.7 The manufacturer's bet

The calculation assumes the true rate is 15%. If the manufacturer believes the true rate is 18%,
because its own trial population was healthier than the contract population, the probability of
triggering at a 17% threshold is:

```text
z = (0.17 - 0.18) / 0.0118 = -0.85
P(trigger) = 0.80
Expected rebate = 0.80 x 0.30 x $30,000,000 = $7,152,990
Total expected cost = $8,352,990
```

The same contract costs $1.54 million if the rate is 15% and $8.35 million if it is 18%. The
contract is a bet on the true rate, and the manufacturer's private information about that rate is
the reason it may be willing or unwilling to sign.

### 6.8 The payer's variance

The expected cost is not the whole story. The rebate is a random variable: zero most of the time, and
30% of revenue when it triggers.

For the base case, the variance of the rebate is approximately:

```text
Var = p(1 - p) x (rebate x revenue)^2
    = 0.038 x 0.962 x ($9,000,000)^2
    = 2.96 x 10^12
Standard deviation = $1,720,000
```

The payer's budget under the contract therefore has a standard deviation of about $1.72 million,
against a certain $1.5 million under the simple rebate. The contract has a similar mean and a
larger variance. A risk-averse payer should prefer the simple rebate unless the contract buys
something else.

### 6.9 When the contract is worth doing

The contract is worth doing if the value of the information or the access it buys exceeds the
difference in expected cost plus the administrative burden.

```text
Value of contract = value of access + value of information
                  - expected rebate - administration
                  - risk premium
```

In the base case, the expected rebate plus administration is $1.54 million, and the simple rebate is
$1.5 million. The contract must therefore create at least $40,000 of additional value before it
breaks even, and more if the parties are risk-averse. That is a low bar if the contract is the only
route to coverage, and a high bar if the payer would have covered the product anyway.

### 6.10 What the example teaches

1. An outcome-based contract has an expected cost that can be computed before signature.
2. The threshold relative to the expected rate determines whether the contract is cosmetic or
   material.
3. The sample size determines whether the measurement can support the payment.
4. A one-point change in the threshold can double the expected cost.
5. The contract is a bet on the true rate, and the parties' beliefs about that rate drive the
   negotiation.
6. A contract that is fair in expectation can still be unattractive because of its variance, and a
   contract that is cheap in expectation can be worthless because it never triggers.

## 7. Code, query, or calculation

The model computes the trigger probability, the expected rebate, the total cost including
administration, and the sample size required for a well-powered contract. It uses only the standard
library.

**Assumptions:**

- 1,000 patients, annual price $30,000, revenue $30,000,000;
- true hospitalisation rate 15% in the base case;
- threshold 17%, rebate 30% of revenue;
- administrative cost $1,200,000 per year;
- fallback simple rebate 5%;
- normal approximation to the binomial;
- power calculation for a two-sided test at 5% significance and 80% power.

```python
import math

def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def contract(n=1_000, p=0.15, threshold=0.17, rebate=0.30,
             price=30_000, admin=1_200_000):
    se = math.sqrt(p * (1 - p) / n)
    prob = 1 - phi((threshold - p) / se)
    revenue = n * price
    expected = prob * rebate * revenue
    return {
        "se": se,
        "prob": prob,
        "revenue": revenue,
        "expected_rebate": expected,
        "total": expected + admin,
    }

base = contract()
print(f"standard error {base['se']:.4f}")
print(f"probability of trigger {base['prob']:.3f}")
print(f"annual revenue USD 2026 {base['revenue']:,.0f}")
print(f"expected rebate {base['expected_rebate']:,.0f}")
print(f"expected rebate plus admin {base['total']:,.0f}")
print(f"simple 5% rebate {0.05 * base['revenue']:,.0f}")

low = contract(threshold=0.16)
print(f"threshold 16%: probability {low['prob']:.3f}  expected {low['expected_rebate']:,.0f}  "
      f"total {low['total']:,.0f}")

high = contract(p=0.18)
print(f"true rate 18%: probability {high['prob']:.3f}  expected {high['expected_rebate']:,.0f}  "
      f"total {high['total']:,.0f}")

z_a, z_b = 1.96, 0.84
p1, p2 = 0.15, 0.17
pbar = (p1 + p2) / 2
n_req = 2 * (z_a + z_b) ** 2 * pbar * (1 - pbar) / (p2 - p1) ** 2
print(f"sample size for 80% power {math.ceil(n_req):,}")
```

**Expected output:**

```text
standard error 0.0113
probability of trigger 0.038
annual revenue USD 2026 30,000,000
expected rebate 344,351
expected rebate plus admin 1,544,351
simple 5% rebate 1,500,000
threshold 16%: probability 0.188  expected 1,691,213  total 2,891,213
true rate 18%: probability 0.795  expected 7,152,990  total 8,352,990
sample size for 80% power 5,269
```

**Interpretation:**

1. The base contract costs almost the same as a 5% simple rebate. It is not a cheaper way to buy
   access; it is a different way to describe the same price.
2. Lowering the threshold by one point raises the expected cost by $1.35 million and makes the
   contract material.
3. If the true rate is 18%, the contract costs $8.35 million, more than five times the base case.
   The manufacturer's belief about the rate is the key negotiation variable.
4. The contract needs about 5,269 patients to distinguish 15% from 17% with conventional power. At
   1,000 patients, it is measuring noise.

**What to vary:** the sample size. Doubling it to 2,000 reduces the standard error to 0.008 and
raises the trigger probability at a 17% threshold to 0.6%, which makes the contract even more
cosmetic. Power and trigger probability move in opposite directions when the threshold is fixed
above the expected rate, which is why the threshold and the sample size must be designed together.

## 8. Data, standards, and regulatory context

Value-based contracts sit at the intersection of pricing law, data protection, and public
procurement. The legal constraints differ by jurisdiction, and they determine which contract designs
are feasible, not only which are permitted.

### 8.1 United States

The Medicaid value-based purchasing final rule (85 FR 87000, 31 December 2020) established a
framework for value-based arrangements in Medicaid. Its key features are:

- a broad definition of a value-based purchasing arrangement, linking payment to observed or
  expected therapeutic or clinical value in a selected population;
- permission for manufacturers to report multiple best prices for a single dosage form and strength,
  but only if the arrangement is offered to all states;
- treatment of a value-based arrangement as a bundled sale, so that a single patient's poor outcome
  does not reset the best price to near zero;
- an exception to the three-year restatement rule where outcomes must be evaluated beyond that
  window;
- reporting and record-keeping requirements.

The rule became effective on 1 March 2021, with the multiple-best-price and related provisions
applying from 1 January 2022. As of 2020, only nine states had authorised value-based arrangements
through a state plan amendment or a similar mechanism, which indicates how slowly the framework was
adopted.

The anti-kickback statute remains a constraint. A discount or rebate that is not structured within a
safe harbour can create liability. The discount safe harbour and the warranty provisions are the
usual routes for price reductions and guarantees; an outcome-based arrangement requires careful
structuring and legal review.

### 8.2 European Union

The GDPR governs the processing of patient-level data used to measure outcomes. A contract that
requires a registry must have a lawful basis, a defined purpose, data minimisation, a retention
period, and a transfer mechanism if data cross borders.

Pricing and reimbursement law remains national. Italy uses AIFA's monitoring registries to administer
risk-sharing and payment-by-result agreements for selected medicines. Spain uses the Valtermed
registry for certain high-cost medicines. Belgium and the Netherlands have used managed entry
agreements with outcome conditions. The United Kingdom uses patient access schemes, including
outcome-based arrangements, within the NICE process.

The European rules do not create a single contract template. A contract that is lawful in one member
state may not be lawful in another, and the data governance must be designed per jurisdiction.

### 8.3 Transparency and confidentiality

Contract terms are often confidential, but the existence of a managed entry agreement may need to be
disclosed. The European Federation of Pharmaceutical Industries and Associations disclosure code
requires manufacturers to publish transfers of value to health care professionals and
organisations, but not necessarily the terms of a pricing agreement with a payer.

The tension between confidentiality and evaluation is real. If the terms and the outcomes are never
disclosed, independent evaluation is impossible, and the evidence base for risk-sharing remains
weak.

### 8.4 What the law does not settle

The law does not determine the outcome metric, the threshold, the measurement window, the attribution
rule, or the administrative budget. Those are design choices, and they determine whether the
contract is useful. A legally compliant contract can still be a poor contract.

> **Note:** This section is educational and reflects published rules at the access date. It is not
> legal advice. A value-based arrangement requires jurisdiction-specific legal review.

## 9. Industry practice and case evidence

### 9.1 How common are managed entry agreements

The OECD reported in 2019 that managed entry agreements were used in at least 28 of 41 OECD and EU
countries. Financial agreements were the majority. Italy was the clearest exception, with 71 active
agreements for 48 active ingredients as of October 2019, of which 46 were performance-based and 25
financial, administered through AIFA's web-based monitoring registries.

The same report found that coverage with evidence development had a poor record of reducing
uncertainty, and that payment-by-result data were often not aggregated or analysed for evidence.

### 9.2 What the financial evidence shows

A 2020 policy analysis concluded that performance-based agreements were failing in practice: Italy's
individual agreements were costly, recovered only about 5% of expenditure, and around 85% were based
on surrogate endpoints.

A 2024 analysis of Italian managed entry agreements from 2019 to 2021 found total payback of
€327.5 million, or 0.9% of €41.1 billion in expenditure, with a median payback share of 3.8% for all
agreements, 3.7% for financial agreements, and 3.3% for outcome-based agreements.

A 2023 review of Italian practice found that managed entry agreements applied to only 7% of
reimbursed medicines between late 2019 and late 2023, and that sales caps accounted for 88% of
agreements. Outcome-based agreements had largely disappeared.

These figures are from one country and one period. They do not prove that risk-sharing cannot work.
They do show that the financial effect is usually small relative to total spend, and that the
administrative cost can exceed the transfer.

### 9.3 The Medicaid experience

The Medicaid value-based purchasing rule was intended to make outcome-based contracts easier by
allowing multiple best prices. Adoption was slow: nine states had authorised arrangements as of
2020, and CMS projected up to $228 million in combined federal and state savings by 2025, a small
fraction of Medicaid drug spending.

> **Inference:** the slow adoption reflects the complexity of the reporting, the difficulty of
> measuring outcomes in a mobile population, and the fact that a simple rebate is easier to
> administer and often achieves a similar price.

### 9.4 What a well-designed contract looks like

The contracts that appear to work share four features:

- a small number of well-defined outcomes, measured in routine data;
- a threshold that is clinically meaningful and close enough to the expected rate to be informative;
- an independent data custodian or a registry with high completeness;
- a scheduled evaluation and an exit clause.

The contracts that fail often have a novel outcome that cannot be measured, a threshold chosen for
negotiation, and no plan to use the data.

### 9.5 The evidence gap

The published evaluations share a limitation: they can measure what was paid, but not what would
have happened without the agreement. The counterfactual is a different price, a different access
decision, or a different product, and none of those is observed.

This is why the evidence on managed entry agreements is dominated by descriptive studies and why
the financial conclusions are modest. A contract can be justified by access that would otherwise
have been denied, and that benefit does not appear in a payback calculation.

A sponsor or payer evaluating a proposal should therefore ask two questions separately: did the
contract transfer money, and did it change access or evidence? The first is measurable from the
reconciliation; the second requires a comparison that most contracts are not designed to support.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Threshold far from expected rate | Trigger probability below 5% | Negotiation convenience | Contract is cosmetic | Compute probability before signing |
| Underpowered measurement | Sample below power requirement | Noise dominates | Payment for random variation | Power the contract |
| Unmeasurable outcome | No routine data source | Registry incomplete | Dispute or invalid estimate | Choose a measurable metric |
| Population mismatch | Registry broader than label | Off-label use included | Manufacturer pays for population difference | Define eligibility and audit |
| Attribution ignored | Co-interventions not recorded | Effect not isolated | Wrong payment | Pre-specify attribution rule |
| Administration underestimated | No operations owner | Manual reconciliation | Cost exceeds transfer | Budget the operations |
| Legal structure wrong | Rebate outside safe harbour | Compliance risk | Liability | Legal review before signature |
| Data governance absent | No lawful basis for patient data | GDPR breach | Contract cannot operate | Data governance agreement |
| No evaluation | Contract renewed without review | Uncertainty persists | Evergreen discount | Scheduled evaluation and exit |
| Dispute clause missing | Ambiguous formula | Disagreement | Payment delayed | Pre-specified resolution |
| Confidentiality blocks learning | Terms not disclosed | No independent evaluation | Evidence gap | Publish non-commercial summary |
| No baseline for evaluation | Access data not collected before launch | Evaluation added later | Cannot show effect | Collect baseline at contract start |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| True outcome rate is known | Computes trigger probability | Real-world population differs | Registry pilot |
| Normal approximation is adequate | Computes probability | Small n or extreme rate | Exact binomial check |
| Outcome is measured completely | Validates payment | Registry incomplete | Completeness audit |
| Population is stable | Enables attribution | Churn and off-label use | Eligibility audit |
| Administration cost is fixed | Compares options | Scale and complexity | Vendor quotes |
| Rebate is the only transfer | Simplifies the model | Other discounts and services | Full contract review |
| Parties are risk-neutral | Simplifies comparison | Risk aversion | Certainty equivalent |
| Contract term matches evidence | Enables evaluation | Evidence arrives later | Align term to data |
| Legal framework is stable | Enables design | Rule changes | Legal monitoring |

### 11.1 Limits of the model

The model uses a normal approximation to a binomial outcome. For small populations or rates near
zero or one, the approximation is poor, and an exact calculation is needed.

The model treats the outcome rate as fixed. In reality it evolves with clinical experience, patient
selection, and co-interventions, so the probability of triggering changes over the contract term.

The model also treats the administrative cost as certain. In practice it varies with the number of
reconciliations, disputes, and data requests, and it is often higher in the first year.

## 12. Exercises

### Exercise 1. Classify the arrangements (recall and application, 30 minutes)

For each arrangement, name the type and the risk transferred: (a) 20% discount off invoice; (b) rebate
of 10% if annual volume exceeds 5,000 patients; (c) full refund for patients who discontinue before
three months; (d) manufacturer pays 50% of the cost for the first six months; (e) coverage conditional
on registry enrolment.

**Deliverable:** a table with arrangement, type, risk transferred, and data required.

**Acceptance criteria:** each type is correctly named; the risk transfer is specific; the data
requirement follows from the type.

**Time:** 30 minutes.

### Exercise 2. Compute the trigger probability (calculation, 45 minutes)

A contract covers 2,000 patients. The expected event rate is 12%, the threshold is 15%, the rebate is
25%, and annual revenue is $40,000,000. Administration costs $800,000.

**Deliverable:** standard error, trigger probability, expected rebate, total expected cost, and a
comparison with a 3% simple rebate.

**Acceptance criteria:** normal approximation stated; arithmetic shown; the comparison includes
administration; the conclusion states whether the contract is material or cosmetic.

**Time:** 45 minutes.

### Exercise 3. Power the contract (calculation, 30 minutes)

Using the same rates, compute the sample size needed to detect a difference between 12% and 15% with
80% power and a two-sided 5% significance level.

**Deliverable:** the sample size and a statement of whether the contract is adequately powered.

**Acceptance criteria:** formula shown; assumptions stated; the conclusion links power to the
contract's ability to support payment.

**Time:** 30 minutes.

### Exercise 4. Compare the options (calculation, 45 minutes)

Using Exercises 2 and 3, compare three options: the outcome-based contract, a 3% simple rebate, and a
cap at $2,000,000 of manufacturer liability. Compute expected cost and variance for each.

**Deliverable:** a comparison table and a recommendation for a risk-averse manufacturer and for a
risk-averse payer.

**Acceptance criteria:** expected cost and variance both reported; the recommendations differ if risk
aversion differs; the administrative cost is included.

**Time:** 45 minutes.

### Exercise 5. Design the metric and threshold (design, 60 minutes)

A product for a chronic skin condition has a trial effect on flare rate at 12 months. The payer is
concerned about real-world adherence.

**Deliverable:** a contract design with a metric, threshold, population, measurement window, data
source, and rebate formula.

**Acceptance criteria:** the metric is measurable in routine data; the threshold is justified against
the expected rate; the population and attribution rule are explicit; the rebate formula is
unambiguous.

**Time:** 60 minutes.

### Exercise 6. Review the legal and data constraints (critique, 45 minutes)

A contract proposes patient-level rebates based on a registry that includes patients treated
off-label, with data transferred to the manufacturer's global warehouse.

**Deliverable:** a review identifying the legal and data issues and the changes required.

**Acceptance criteria:** at least three issues identified; each linked to a specific rule or
principle; the changes preserve the contract's objective where possible.

**Time:** 45 minutes.

### Exercise 7. Evaluate a failed contract (critique, 45 minutes)

A contract ran for two years, triggered once, and cost $2.4 million to administer. The payer says it
improved access; the manufacturer says it did not.

**Deliverable:** an evaluation plan and a judgement on what the contract achieved, with the evidence
needed to decide.

**Acceptance criteria:** the counterfactual is addressed; the administrative cost is compared with
the transfer; the access claim is tested with data.

**Time:** 45 minutes.

### Exercise 8. Negotiate the terms (design, 60 minutes)

The payer proposes a 14% threshold and a 40% rebate. The manufacturer's expected rate is 12%. The
simple rebate fallback is 6%.

**Deliverable:** a negotiation brief with the expected cost of the payer's proposal, a counter-offer,
and a walk-away position.

**Acceptance criteria:** expected cost computed; the counter-offer is justified by the value of
access; the walk-away is explicit; the brief states what evidence would change the position.

**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

| Arrangement | Type | Risk transferred | Data |
|---|---|---|---|
| 20% discount | Financial | Price | None |
| Volume rebate | Financial | Volume | Sales |
| Refund on discontinuation | Performance-based (PbR) | Effectiveness and persistence | Patient-level |
| 50% cost share for six months | Financial / cost-sharing | Early budget | Treatment |
| Coverage with registry | Performance-based (CED) | Effectiveness and budget | Registry |

A strong answer distinguishes cost-sharing, which applies to all treated patients, from PbR, which
applies only to those who do not achieve the outcome.

### 13.2 Exercise 2

```text
SE = sqrt(0.12 x 0.88 / 2,000) = 0.00727
z = (0.15 - 0.12) / 0.00727 = 4.13
P(trigger) = 1 - Phi(4.13) = 0.00002
Expected rebate = 0.00002 x 0.25 x $40,000,000 = $200
Total expected cost = $200 + $800,000 = $800,200
Simple 3% rebate = 0.03 x $40,000,000 = $1,200,000
```

The outcome contract is cheaper in expectation by about $400,000, but it triggers almost never. The
payer receives no protection against the 15% scenario it was concerned about. The contract is
cosmetic.

### 13.3 Exercise 3

```text
n = 2 x (1.96 + 0.84)^2 x 0.135 x 0.865 / (0.03)^2
  = 2 x 7.84 x 0.1168 / 0.0009
  = 2,035
```

The contract has 2,000 patients, which is just below the required sample. It is marginally
underpowered, and the trigger probability is so low that power is not the binding problem; the
threshold is.

### 13.4 Exercise 4

For the outcome contract, the standard deviation of the rebate is approximately:

```text
SD = sqrt(p(1-p)) x rebate x revenue
   = sqrt(0.00002 x 0.99998) x 0.25 x $40,000,000
   = 0.00447 x $10,000,000 = $44,700
```

The simple rebate has zero variance and a cost of $1,200,000. The cap has a cost that depends on the
volume distribution, but it is bounded at $2,000,000.

A risk-averse manufacturer prefers the outcome contract because its expected cost and variance are
low. A risk-averse payer prefers the simple rebate because it is certain and larger, or the cap
because it bounds the downside. The recommendations differ because the parties value the same
distribution differently.

### 13.5 Exercise 5

A strong design uses flare rate at 12 months, measured in a registry with a completeness target of at
least 90%, a threshold at the upper end of the trial confidence interval, and a rebate of 20% of
revenue for the affected year. The population is the label population, with off-label patients
excluded. The attribution rule excludes patients who discontinue before the measurement window.

A weak design uses a surrogate that is not recorded in routine data, a threshold equal to the trial
point estimate, and a population defined by the prescriber rather than the label.

### 13.6 Exercise 6

The review should identify:

- **Off-label inclusion.** The population is broader than the label; outcomes may differ for reasons
  unrelated to the product. Restrict to the label or pre-specify a subgroup analysis.
- **GDPR.** Transfer to a global warehouse requires a lawful basis, minimisation, and a transfer
  mechanism. Use an independent custodian or aggregate the data.
- **Anti-kickback or equivalent.** Patient-level rebates may create inducement risk. Legal review is
  required.
- **Data quality.** The registry may not capture all treated patients. Set a completeness threshold
  and a missing-data rule.

### 13.7 Exercise 7

The evaluation should compare access before and after the contract, the outcome rate with the
counterfactual, and the administrative cost with the transfer. If the contract triggered once and
cost $2.4 million to administer, the transfer was small relative to the cost, and the access claim
must be tested against prescription and fill data.

A likely conclusion is that the contract functioned as a small discount with a large administrative
overhead. That is a finding worth reporting, not a failure to hide.

### 13.8 Exercise 8

```text
SE at p = 0.12, n = 1,000 (assumed) = 0.0103
z = (0.14 - 0.12) / 0.0103 = 1.94
P(trigger) = 0.026
Expected rebate = 0.026 x 0.40 x revenue
```

If revenue is $30,000,000, the expected rebate is $312,000. Add administration of $1,200,000 and the
total is $1.51 million, close to a 5% simple rebate. The counter-offer should be a threshold at 15%
or a lower rebate, and the walk-away should be a net price below the simple-rebate equivalent unless
the access benefit is documented.

### 13.9 Rubric for the design exercises

| Criterion | Weight | Weak | Strong |
|---|---:|---|---|
| Metric measurability | 25% | No routine data source | Named source with completeness target |
| Threshold justification | 20% | Round number | Linked to expected rate and clinical relevance |
| Attribution | 20% | Ignored | Population and co-intervention rules stated |
| Legal and data governance | 20% | Not addressed | Jurisdiction-specific review and custodian |
| Evaluation and exit | 15% | None | Scheduled review and exit clause |

A design that scores well on the metric but ignores attribution will produce a contract that cannot
be enforced.

## 14. Interview questions

**1. What is a value-based agreement?**
Strong: a contract linking payment to a measured outcome or value, with a defined population,
threshold, and data source. Weak: calls any rebate value-based.

**2. What risk does a simple discount transfer?**
Strong: price risk to the manufacturer; volume and performance risk stay with the payer. Weak: says
it transfers all risk.

**3. How do you set the threshold in an outcome-based contract?**
Strong: from the expected rate, the clinical relevance, and the power to detect a difference, with
the trigger probability computed. Weak: picks the trial point estimate.

**4. Why can an outcome-based contract be cosmetic?**
Strong: when the threshold is so far above the expected rate that it almost never triggers. Weak:
assumes any outcome contract changes behaviour.

**5. What sample size does the contract need?**
Strong: enough to distinguish the expected rate from the threshold with conventional power, which is
often thousands of patients. Weak: uses the contract population without a power calculation.

**6. How do you compare an outcome contract with a simple rebate?**
Strong: expected cost, variance, administration, and the value of access or information. Weak:
compares only the headline rebate percentage.

**7. What are the main legal constraints in the US?**
Strong: Medicaid best price and the 2020 value-based purchasing rule, anti-kickback, and reporting.
Weak: says outcomes contracts are illegal.

**8. Why have managed entry agreements had modest financial impact?**
Strong: small payback relative to spend, administrative burden, surrogate endpoints, and data that
are not used to reduce uncertainty. Weak: says they are ineffective without evidence.

**9. How do you evaluate whether a contract worked?**
Strong: access, outcome, administrative cost, and a counterfactual, with a scheduled review. Weak:
counts the number of patients enrolled.

**10. When would you walk away from a contract?**
Strong: when the expected net price falls below the value of the alternative, or when the data cannot
support the payment. Weak: never walks away.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| Anti-kickback | US prohibition on remuneration for referrals | 8.1 |
| Attribution | Judgement that an outcome is due to the product | 3.4 |
| Best price | Medicaid benchmark | 8.1 |
| Cap | Maximum liability or spend | 4 |
| Clawback | Recovery of part of a payment | 4.3 |
| Contract year | Reconciliation period | 4 |
| Cost-sharing | Manufacturer pays part of the cost | 4.2 |
| Counterfactual | What would have happened without the product | 4 |
| Coverage with evidence development | Coverage conditional on data | 4.1 |
| Effective price | Price after all adjustments | 4 |
| Exit clause | Condition for ending the contract | 4 |
| Gaming | Manipulation of measurement | 4 |
| GDPR | EU data protection regulation | 8.2 |
| Managed entry agreement | Formal arrangement for entry with conditions | 4 |
| Medicaid VBP | US Medicaid value-based purchasing framework | 8.1 |
| Outcome metric | Measured variable that triggers payment | 5.3 |
| Outcomes guarantee | Refund if outcome not achieved | 4 |
| Payment by result | Rebate for patients who do not achieve the outcome | 4.2 |
| Performance-based agreement | Payment tied to a measured outcome | 4.1 |
| Price-volume agreement | Price falls as volume rises | 4 |
| Rebate | Post-sale payment to the payer | 4.3 |
| Reconciliation | Calculation of the amount due | 5.19 |
| Registry | Systematic patient data collection | 5.5 |
| Risk premium | Value a party places on avoiding variance | 5.7 |
| Risk-sharing | Allocation of performance risk | 3.1 |
| Surrogate endpoint | Intermediate marker | 4 |
| Threshold | Outcome level that triggers payment | 3.5 |
| Trigger probability | Chance the condition is met | 3.5 |
| Value-based agreement | Contract linking payment to value | 4 |
| Warranty | Payment or replacement if the product fails | 4 |

## 16. Sources

All URLs accessed 26 October 2026.

1. Centers for Medicare and Medicaid Services. *Medicaid Program; Establishing Minimum Standards in
   Medicaid State Drug Utilization Review and Supporting Value-Based Purchasing for Drugs Covered in
   Medicaid*, final rule, 85 FR 87000, 31 December 2020.
   https://www.federalregister.gov/documents/2020/12/31/2020-28567/medicaid-program-establishing-minimum-standards-in-medicaid-state-drug-utilization-review-dur-and
2. Centers for Medicare and Medicaid Services. *CMS Issues Final Rule to Empower States,
   Manufacturers, and Private Payers to Create New Payment Methods for Innovative New Therapies
   Based on Patient Outcomes*.
   https://www.cms.gov/newsroom/press-releases/cms-issues-final-rule-empower-states-manufacturers-private-payers-create-new-payment-methods
3. Wenzl M, Chapman S. *Performance-based managed entry agreements for new medicines in OECD
   countries and EU member states*. OECD Health Working Paper No. 115, 2019.
   https://health.ec.europa.eu/system/files/2020-01/2019_entryagreements_newmedicines_oecdeu_en_0.pdf
4. Dabbous M, et al. Managed Entry Agreements: Policy Analysis From the European Perspective.
   *Value in Health*. 2020. https://www.valueinhealthjournal.com/article/S1098-3015(20)30041-3/fulltext
5. Trotta F, et al. Financial Outcomes of Managed Entry Agreements for Pharmaceuticals in Italy.
   *JAMA Health Forum*. 2024.
   https://jamanetwork.com/journals/jama-health-forum/fullarticle/2813672
6. AIFA. *Managed entry agreements and monitoring registries*.
   https://www.aifa.gov.it/en/web/guest/-/registri-farmaci-sottoposti-a-monitoraggio
7. Italian Ministry of Health. *Valtermed*.
   https://www.salute.gov.it/portale/valtermed/homeValtermed.jsp
8. NICE. *Patient access schemes*.
   https://www.nice.org.uk/about/what-we-do/patient-access-schemes-liaison-unit
9. European Medicines Agency. *Health technology assessment*.
   https://www.ema.europa.eu/en/human-regulatory-overview/research-development/health-technology-assessment
10. ISPOR. *Performance-based risk-sharing arrangements: good practices*.
    https://www.ispor.org/
11. European Federation of Pharmaceutical Industries and Associations. *Disclosure code*.
    https://www.efpia.eu/ethics-compliance/transparency/
12. US Government. *42 CFR 447.502, Definitions*.
    https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-C/part-447/subpart-I/section-447.502
13. US Government. *42 CFR 1001.952, Exceptions to the anti-kickback statute*.
    https://www.ecfr.gov/current/title-42/chapter-V/subchapter-B/part-1001/section-1001.952

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_global-regulation.md`](06_global-regulation.md)
- Market access: [`05_market-access.md`](05_market-access.md)
- HTA: [`04_health-technology-assessment.md`](04_health-technology-assessment.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
