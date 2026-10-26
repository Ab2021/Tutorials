# Privacy-Enhancing Technologies

> **Module 16 - Privacy, Security, and Ethics**
> Topic 03 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

Privacy-enhancing technologies (PETs) are sold as answers: differential privacy makes data
anonymous, federated learning keeps data on the device, homomorphic encryption lets you compute
without seeing, synthetic data has the statistics without the people. Each claim contains a
truth and a missing clause. The truth is that the mechanism provides a defined protection under
stated assumptions. The missing clause is the assumption: a privacy budget that must be accounted
for, a threat model that must exclude the server, a key that must be protected, a generative model
that may have memorised its training records.

The commercial pressure to deploy a PET is now strong. Procurement questionnaires ask for
differential privacy, regulators point to PETs as safeguards, and the European Health Data Space
requires secondary use through secure processing environments. A team that cannot evaluate a PET
claim will either overpay for a mechanism it does not need or deploy one that does not provide the
protection it advertises. Both errors are expensive: the first wastes budget and delays a
programme, the second creates a false sense of safety that survives until an audit or an attack.

Three failure patterns recur.

**The epsilon that is never accounted for.** A team deploys a differentially private dashboard with
a per-query epsilon and no ledger. Each query looks safe at epsilon 0.05. After sixty queries the
cumulative loss is three, the annual budget is one, and nobody has noticed because composition was
never computed. Differential privacy is a budget, not a per-release checkbox; a deployment without
accounting is not differentially private in any meaningful operational sense.

**Federated learning treated as a privacy guarantee.** A consortium trains a model across hospitals
without centralising data, and describes the result as privacy-preserving. The gradients are still
shared. Gradient inversion and membership inference can recover training information from them,
especially with small batches, and without secure aggregation the server sees every update. The
architecture changed where the data sits; it did not by itself bound what the model reveals.

**Synthetic data assumed to be safe because it is fake.** A generative model is trained on patient
records and the samples are released as "synthetic, therefore anonymous". The model can memorise
rare records, reproduce identifiers from free text, and leak membership through overfitting. A
synthetic dataset needs the same disclosure assessment as a real one, plus an assessment of the
generator.

This topic teaches PET selection as an engineering decision with a threat model, a guarantee, a
cost, and a verification step. The organising question is not "which PET is best" but "which
guarantee does this decision need, under which trust assumption, and what does it cost in utility,
latency, and operational complexity". The four families covered are differential privacy,
federated learning, secure computation (secure multi-party computation and homomorphic
encryption), and synthetic data. Trusted execution environments and tokenization appear where they
change the trust boundary.

The distinction that runs through the topic is between a mathematical guarantee, an architectural
control, and a contractual control. Differential privacy is a mathematical guarantee with a
parameter. Federated learning and secure computation are architectural controls that remove a
specific party's access. Synthetic data is a statistical approximation whose privacy properties
must be measured, not assumed. A programme needs all three kinds, and no single mechanism provides
all three.

A fourth pattern is the **PET that is never operated**. A pilot demonstrates differential privacy
on a static extract, the vendor contract ends, and the production pipeline continues without the
ledger, the gateway, or the tests. The mechanism was real during the pilot and is decorative
afterwards. The control that prevents this is operational ownership: a named team responsible for
the budget, the library version, the tests, and the quarterly reconciliation. A PET without an
owner is a procurement artefact.

> **Note:** A PET is not a compliance certificate. It changes the risk profile of a processing
> activity and must be described in the assessment alongside governance, access control, and
> contract terms.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a mathematical guarantee, an architectural control, and a statistical
   approximation, and classify each PET into the correct category.
2. **Compute** the Laplace and Gaussian noise scale for a query given sensitivity and epsilon, and
   **explain** what the resulting interval means for a decision.
3. **Account** for a privacy budget across queries using basic and advanced composition, and
   **detect** an over-spend before it happens.
4. **Choose** between central differential privacy, local differential privacy, federated
   learning, secure computation, and synthetic data for a stated trust model.
5. **Evaluate** a federated learning deployment for gradient leakage, secure aggregation, and
   heterogeneity.
6. **Assess** a synthetic dataset for memorisation, membership leakage, and utility loss.
7. **Write** a PET selection memo that states the guarantee, the assumptions, the cost, and the
   verification method.
8. **Critique** a vendor claim that a product is "differentially private" or "privacy-preserving"
   without a parameter, a unit of privacy, or an accounting method.

## 3. The core mental model

Every PET changes one of three things: what a party can see, what a computation reveals, or what a
statistic depends on. The mental model is a trust-boundary diagram with the mechanism placed on the
boundary it changes.

```text
   DATA SUBJECTS            DATA HOLDER                 ANALYST / BUYER
   +-----------+           +-------------+             +-------------+
   | patient   |  raw data | hospital    |  output     | researcher  |
   | device    |---------->| server      |------------>| vendor      |
   +-----------+           +-------------+             +-------------+
        |                        |                           |
        | local DP               | central DP                |
        | (noise before send)    | (noise on output)         |
        |                        |                           |
        +---- federated -------- + ---- secure agg ----------+
              (model updates)          (encrypted shares)

   TRUST BOUNDARY A: does the server see raw records?
   TRUST BOUNDARY B: does the analyst see record-level output?
   TRUST BOUNDARY C: does the model reveal its training data?
```

The diagram makes the trade-offs visible. Local differential privacy protects against the data
holder, because noise is added before the data leave the device; the cost is much lower accuracy
for the same epsilon, because noise is added once per record rather than once per query. Central
differential privacy gives better accuracy because the curator sees the raw data and adds noise to
the answer; the cost is that the curator must be trusted. Federated learning removes the transfer
of raw records but not the transfer of information about them; secure aggregation removes the
server's view of individual updates but not the final model's leakage. Synthetic data removes the
records from the release but not the influence of the records on the generator.

A second model is the guarantee ladder:

```text
strongest, most assumptions
   formal DP with a stated epsilon, delta, unit of privacy, and composition
   formal DP without accounting (a number on a slide)
   empirical privacy testing (membership inference attacks, red team)
   architectural separation (federated, MPC, TEE)
   contractual prohibition (DUA says do not re-identify)
   de-identification under a standard (Topic 02)
weakest, fewest assumptions
```

The ladder is not a ranking of usefulness; it is a ranking of how much the claim depends on
assumptions being true. A formal DP guarantee is strong because it holds regardless of the
adversary's auxiliary data, provided the implementation is correct and the budget is respected. An
architectural control is weaker because it holds only against the party excluded by the
architecture. A contract is weakest because it depends on enforcement.

Three properties follow.

**Every PET has a unit of privacy.** Event-level DP protects one record; user-level DP protects all
records from one person. A model trained with event-level DP across years of records from the same
patient does not give that patient user-level protection. The unit must be stated, and it must
match the harm.

**Every PET has a trust assumption.** Central DP trusts the curator. Local DP trusts the device
randomisation and the channel. Secure computation trusts the implementation and the key management.
Federated learning trusts the aggregator unless secure aggregation is added. Synthetic data trusts
the generator and the release process. A PET deployed outside its trust assumption provides less
protection than advertised.

**Every PET has a cost.** The cost is measured in utility (noise, suppression, fidelity), latency
(secure computation is slower), operational complexity (key management, orchestration), and
governance (budget ownership, audit). A PET that cannot be operated will be bypassed, and a bypassed
PET provides no protection.

### What the ladder does not say

The ladder ranks claims by how much they depend on assumptions, not by how much protection they
provide in a given deployment. A contract can be stronger than a weak DP implementation if the
implementation is wrong, and a formal guarantee with a large epsilon can be weaker than a
well-operated secure environment. The ladder is a prompt to ask what the claim rests on, not a
substitute for the assessment.

It also does not rank mechanisms by cost. Secure computation is high on the ladder and expensive
to operate; local DP is formal and inexpensive to deploy but costly in accuracy. The selection
decision therefore uses three axes at once: the guarantee, the trust model, and the cost. A team
that optimises only the guarantee will choose a mechanism it cannot run.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| PET | Privacy-enhancing technology: a mechanism that reduces privacy risk while preserving some utility. | Treated as a synonym for compliance. | Category. |
| Differential privacy (DP) | A property of a randomised mechanism such that neighbouring datasets produce similar output distributions, parameterised by epsilon and delta. | Used as a marketing adjective. | Formal definition. |
| Epsilon | The privacy loss parameter; smaller means stronger protection and more noise. | Compared across mechanisms without checking the definition. | Real number; per query or total. |
| Delta | The probability of a privacy failure beyond epsilon in approximate DP. | Set arbitrarily large. | Probability; NIST suggests <= 1/n^2. |
| Sensitivity | The maximum change in a query's output from adding or removing one record. | Ignored when choosing noise. | Query-specific; same unit as the output. |
| Laplace mechanism | Adds Laplace noise with scale sensitivity/epsilon; satisfies pure epsilon-DP. | Used with the wrong sensitivity. | Mechanism. |
| Gaussian mechanism | Adds Gaussian noise calibrated to sensitivity, epsilon, and delta; satisfies approximate DP. | Calibrated with the wrong delta. | Mechanism. |
| Privacy budget | The total epsilon (and delta) allocated to a set of analyses over a period. | Treated as per-query only. | Real number; per period. |
| Basic composition | The total loss of sequential mechanisms is the sum of their epsilons. | Ignored. | Accounting rule. |
| Advanced composition | A tighter bound for many mechanisms, using epsilon, delta, and the number of queries. | Assumed always better. | Accounting rule. |
| Unit of privacy | The entity protected by DP: a record (event-level) or a person (user-level). | Left unstated. | Definition choice. |
| Local DP | Noise added at the data source before transmission. | Confused with central DP. | Trust model. |
| Central DP | A trusted curator adds noise to query answers or releases. | Used when the curator is not trusted. | Trust model. |
| Shuffle model | A trusted shuffler between local randomisation and the aggregator, improving accuracy. | Assumed to be local DP. | Trust model. |
| DP-SGD | Differentially private stochastic gradient descent, clipping per-example gradients and adding noise. | Implemented without clipping. | Training algorithm. |
| Federated learning | Training a model across clients without centralising raw data. | Called privacy-preserving without qualification. | Architecture. |
| Secure aggregation | A protocol that lets a server compute the sum of client updates without seeing individual updates. | Assumed to protect the final model. | Protocol. |
| Gradient leakage | Reconstruction of training data from shared gradients. | Assumed impossible. | Attack. |
| Membership inference | Determining whether a record was in the training set. | Confused with DP. | Attack. |
| MPC | Secure multi-party computation: computing a function over private inputs without revealing them. | Assumed to be cheap. | Protocol. |
| Homomorphic encryption | Computing on ciphertext so that decryption yields the result of the computation on plaintext. | Assumed to support arbitrary workloads efficiently. | Cryptographic scheme. |
| TEE | Trusted execution environment: hardware-isolated computation with remote attestation. | Trusted without supply-chain review. | Architecture. |
| Synthetic data | Records generated by a model fitted to real data. | Assumed anonymous. | Data type. |
| Memorisation | A generative model reproducing training records or rare details. | Ignored in release review. | Attack or measurement. |
| Fidelity | How closely synthetic data reproduce the real joint distribution. | Confused with privacy. | Metric. |
| Utility | Usefulness of a PET output for the stated decision. | Reported without a task metric. | Task-specific. |
| Privacy-utility trade-off | The frontier along which stronger privacy reduces utility. | Treated as a single optimum. | Curve. |
| Side channel | A leak through timing, memory, logs, or error messages rather than the intended output. | Ignored in DP deployments. | Attack surface. |
| Trust boundary | The line across which a party's access changes. | Drawn implicitly. | Architecture. |
| Attestation | Evidence that a TEE is running expected code. | Assumed without verification. | Protocol. |
| DPIA | Data protection impact assessment; PETs are safeguards within it. | Replaced by a PET. | Process. |
| Privacy hazard | A practical failure mode in DP deployment catalogued by NIST SP 800-226. | Unknown to implementers. | Risk category. |

## 5. The mechanics, step by step

The procedure below selects and deploys a PET for a stated decision. It is written for a data or
platform team that has to choose among mechanisms and then defend the choice. The order matters:
the trust model is fixed before the mechanism, and the budget is fixed before the first query.

### Step 1. State the decision and the utility metric

**Input:** A decision that the PET output will support, and the analysis that consumes it.
**Output:** One sentence naming the decision and one metric that defines acceptable utility.
**Rationale:** PETs trade privacy for utility, and the trade can only be evaluated against a task.
"Our dashboard needs tract-level diabetes prevalence within plus or minus 10 percent" is a
criterion. "We need accurate data" is not. The metric should be the one the decision uses: relative
error for a rate, absolute error for a count, calibration for a model, or recall at a fixed
threshold for a classifier.

Branch point: if the decision needs record-level data rather than a statistic, differential
privacy on aggregates is the wrong tool. The choice is between secure computation, a controlled
environment, and a contract.

### Step 2. Draw the trust boundaries

**Input:** System architecture, contracts, deployment model.
**Output:** A diagram naming every party and what each can see.
**Rationale:** The trust model determines which mechanism is even applicable. A hospital that
operates its own analytics platform can use central DP. A consortium of hospitals that does not
trust a central server needs federated learning with secure aggregation, or MPC. A consumer app
that does not want to hold raw data needs local DP. A cloud deployment where the cloud provider is
not trusted needs confidential computing or client-side encryption.

The diagram should mark three boundaries: whether the server sees raw records, whether the analyst
sees record-level output, and whether the model or release can reveal training data. A mechanism
that protects one boundary while leaving another open is not a solution to the whole problem.

Branch point: if the party that operates the mechanism is also the adversary in the threat model,
the mechanism must not depend on that party's honesty. Central DP fails this test; local DP, MPC,
and confidential computing can pass it.

### Step 3. Name the adversary and the harm

**Input:** Threat model from Topic 04, data sensitivity, population.
**Output:** A harm statement per adversary: reconstruction, membership, attribute inference,
linkage, or singling out.
**Rationale:** Different PETs address different harms. Differential privacy bounds what can be
learned about any individual from the output, including membership and attribute inference, under
its definition. Secure computation hides inputs from the parties computing on them but does not
bound what the output reveals. Federated learning hides raw records but not gradient information.
Synthetic data aims to preserve distributional properties while not reproducing records, and its
privacy must be measured against memorisation and membership attacks.

A practical harm statement names the person or group, the inference, and the consequence: "a
patient with a rare diagnosis in a small county could be identified from a published count, leading
to disclosure to an employer". The statement drives both the mechanism and the epsilon.

### Step 4. Choose the guarantee category

**Input:** Trust boundaries, harm statement, utility metric.
**Output:** A guarantee category: formal DP, architectural separation, synthetic approximation, or
a combination.
**Rationale:** The category determines what can be promised. Formal DP can promise a bound on
influence for any adversary, given the stated unit and budget. Architectural separation can promise
that a named party does not see the data, but not that the output is safe. Synthetic data can
promise that no real record is released, but not that no real record is recoverable, unless the
generator is trained with DP or the output is tested.

| Mechanism | Protects against | Guarantee type | Typical cost | Fails when | Health example |
|---|---|---|---|---|---|
| Central DP | Analysts, recipients, and any adversary with auxiliary data | Formal, given a trusted curator | Noise; budget management | Curator is untrusted or budget unmanaged | Public health statistics |
| Local DP | The data holder and the server | Formal, weaker per epsilon | Much more noise; limited queries | Device compromise; small populations | Wearable telemetry |
| Federated learning + secure aggregation | The server seeing individual updates | Architectural, plus DP if added | Communication; orchestration; heterogeneity | Gradients leak; no DP; small cohorts | Multi-site model training |
| Secure multi-party computation | Each party seeing the others' inputs | Architectural, cryptographic | Latency; limited operations | Implementation bugs; collusion beyond threshold | Cross-insurer linkage |
| Homomorphic encryption | The computing party seeing plaintext | Architectural, cryptographic | Very high compute; restricted operations | Key compromise; side channels | Cloud inference on PHI |
| Trusted execution environment | The host operator seeing memory | Architectural, hardware-based | Attestation complexity; supply chain | Side-channel attacks; misconfiguration | Cloud analytics |
| Synthetic data | Recipients seeing real records | Statistical approximation | Fidelity loss; generator risk | Memorisation; membership leakage | Data sharing for methods development |

The table is a starting point, not a decision rule. A programme often combines mechanisms: federated
learning with secure aggregation and DP-SGD, or synthetic data generated inside a TEE. The
combination must be described as a whole, because the weakest component bounds the guarantee.

### Step 5. Select the mechanism and the unit of privacy

**Input:** Guarantee category, query workload, population.
**Output:** A named mechanism, a unit of privacy, and a parameter choice.
**Rationale:** For DP, the mechanism follows from the query type and the sensitivity. Counting and
summing queries use the Laplace or Gaussian mechanism. Selection among a small set of options uses
the exponential mechanism. Training uses DP-SGD, which requires per-example gradient clipping and
noise. The unit of privacy is event-level if one record is the protected entity, and user-level if
all records from one person are protected. User-level DP is harder and more appropriate for
longitudinal data.

| Query type | Mechanism | Sensitivity | Note |
|---|---|---|---|
| Count of records | Laplace | 1 | Add or remove one record changes the count by at most 1. |
| Sum of a bounded value | Laplace | Bound | Clamp each value to the bound before summing. |
| Mean of a bounded value | Laplace on sum and count | Bound and 1 | Compute the ratio of two noisy quantities; account for both. |
| Histogram | Laplace per bin | 1 per bin | Composition across bins consumes budget; use parallel composition where disjoint. |
| Selection among options | Exponential | Utility range | Used for choosing a category, not for counts. |
| Model training | DP-SGD | Clipping norm | Noise multiplier, sampling rate, and steps determine epsilon. |
| Synthetic data | DP generative model | Varies | Post-hoc testing is not a substitute for DP training. |

The unit choice changes the arithmetic. Event-level DP with epsilon 1 on a dataset of 50,000
records protects each record's contribution. User-level DP on the same dataset, where each user
contributes 20 records, must account for the group of 20, and the effective noise is larger for the
same epsilon. A deployment that claims user-level protection while accounting per record is
overstating the guarantee.

### Step 6. Set the budget and the accounting method

**Input:** Analysis plan, number of queries, release cadence, risk appetite.
**Output:** A budget per period, a per-query allocation, and an accounting rule.
**Rationale:** The budget is the operational core of DP. Basic composition sums the epsilons of
sequential mechanisms. Advanced composition gives a tighter bound when many mechanisms run, at the
cost of a delta term. Parallel composition applies when mechanisms operate on disjoint subsets,
because the worst case is the maximum rather than the sum.

A budget should be set per period (annual, per release cycle) and owned by a named role. The
allocation should be recorded before the first query, with a reserve for ad hoc requests. The
accounting must include every release, including reruns, because a rerun with fresh noise is a new
mechanism and consumes budget.

Branch point: if the analysis plan is open-ended, the budget will be exhausted by exploration. The
fix is to separate exploratory analysis (run inside the trusted environment on raw data, with
access controls) from published statistics (DP, with a fixed budget).

### Step 7. Compute parameters and test utility

**Input:** Sensitivity, epsilon, delta, query workload, utility metric.
**Output:** Noise scale, expected interval width, and a utility test result.
**Rationale:** The arithmetic is simple and frequently skipped. For the Laplace mechanism, the
scale is sensitivity divided by epsilon, and the two-sided 95 percent interval half-width is
approximately 2.996 times the scale. For the Gaussian mechanism, the scale depends on delta and is
larger for small delta. The utility test asks whether the interval is narrow enough for the
decision, at the planned number of queries, for the smallest population in scope.

A useful rule is to compute the signal-to-noise ratio for the smallest cell. If the expected count
is 40 and the interval half-width is 60, the release cannot support the decision, regardless of how
defensible the epsilon looks. The correct response is to change the geography, the query, or the
release mode, not to shave the epsilon.

### Step 8. Implement with a vetted library and verify

**Input:** Mechanism specification, library, test data.
**Output:** An implementation with tests for sensitivity, clipping, noise calibration, and budget.
**Rationale:** Differential privacy is easy to state and easy to implement incorrectly. NIST
SP 800-226 catalogues privacy hazards that include floating-point leaks, timing channels,
hyperparameter tuning on the private data, and inconsistent sensitivity. The implementation should
use a well-tested library, document the version, and include tests that check the noise scale
against the specification and the budget against the ledger.

Verification should include an empirical check where feasible: a membership inference attack on a
model, a reconstruction attempt on a synthetic dataset, or a comparison of released statistics
against the known raw values in a test environment. Empirical checks do not prove the guarantee,
but they catch implementation errors.

### Step 9. Add residual controls

**Input:** Deployment, threat model, contract.
**Output:** Access control, output review, contract terms, and monitoring.
**Rationale:** A PET reduces one class of risk and leaves others. A DP release still needs access
control for the raw data, output review for small cells that the mechanism does not cover, a
contract for the recipient, and monitoring for repeated queries that exhaust the budget. A
federated deployment still needs authentication of clients, secure aggregation, and a defence
against poisoning. A synthetic release still needs a disclosure assessment.

Residual controls should be described in the same document as the guarantee, because the reader
will otherwise assume the PET covers everything.

### Step 10. Monitor, audit, and re-budget

**Input:** Budget ledger, query logs, incident reports.
**Output:** A monthly reconciliation and an annual re-assessment.
**Rationale:** Budgets drift. A new analyst runs a query without the ledger; a dashboard reruns
after a data refresh; a model is retrained with a larger epsilon. Monitoring compares the ledger to
the query log and flags discrepancies. The audit also checks that the mechanism parameters in
production match the approved specification, because configuration drift is a common failure.

### Step 11. Document the guarantee honestly

**Input:** All outputs above.
**Output:** A PET statement with the mechanism, parameters, unit, budget, assumptions, utility, and
residual controls.
**Rationale:** The statement is what procurement, the DPIA, and the regulator will read. It must
avoid two errors: overclaiming ("anonymous") and underclaiming ("we added some noise"). A good
statement is specific: "central differential privacy, event-level, Laplace mechanism, epsilon 0.05
per query, annual budget 1.0, 95 percent interval half-width 60 at tract level, with a DUA and
output review". A reader can challenge each element.

### Composition in practice

Three composition rules cover most deployments, and the mistakes are usually about which rule
applies rather than about the arithmetic.

| Pattern | Rule | Worked example | Trap |
|---|---|---|---|
| Sequential queries over the same records | Basic composition sums epsilon; advanced composition gives a tighter bound for many queries | 60 queries at 0.05 cost 3.0 basic, 2.702 advanced | Treating a rerun as free |
| Queries over disjoint subsets | Parallel composition: total cost is the maximum epsilon, not the sum | 50 tracts, one query each at 0.05, cost 0.05 | A person appearing in two subsets breaks disjointness |
| Independent datasets | Budgets are separate | Two registries with no overlap | Assuming overlap when there is none, or vice versa |
| A query rejected by the ledger | No cost, but the rejection itself may leak | Small-cell query refused | Error messages reveal cell size |

Parallel composition is the most valuable and the most often missed. A dashboard that queries each
tract once can spend 0.05 in total, not 50 x 0.05, provided no person contributes to more than one
tract. That condition is false for anyone who moved during the year, for shared households, and for
records that span facilities. The composition claim is only as good as the disjointness argument,
and the argument must be written down.

Budget hygiene has five practical rules:

1. One ledger per dataset, not per team or per dashboard.
2. A reserve of 10 to 20 percent for ad hoc requests and corrections.
3. A gateway that rejects a query before execution when the budget is insufficient, with a
   response that does not reveal why.
4. Idempotency keys so a retry does not consume budget twice.
5. A monthly reconciliation between the ledger and the query log.

The gateway is the control that turns a policy into an operating property. Without it, the ledger
is a record of what happened rather than a limit on what can happen.

### Choosing epsilon

NIST SP 800-226 describes epsilon below 1 as generally reasonable and 0.1 as strong for many
purposes, while noting that the choice depends on the unit, the population, and the sensitivity of
the data. That is guidance, not a threshold. A practical selection process has four inputs:

- The sensitivity of the statistic: a count of diagnoses is more sensitive than a count of
  appointments.
- The population per cell: the same epsilon is weaker for a cell of 20 than for a cell of 20,000,
  because the noise is the same while the group is smaller.
- The number of queries: a per-query epsilon that is safe once is not safe sixty times.
- The adversary: a public release faces more adversaries than an internal dashboard.

The output of the process is a range, not a single number, and the range should be documented with
its rationale. A team that cannot explain why it chose 0.05 rather than 0.1 or 0.02 has not chosen
a parameter; it has copied one.

### Verification tests

A PET deployment needs tests that fail when the mechanism is wrong. Six are worth building into
the pipeline:

| Test | What it checks | How it fails |
|---|---|---|
| Noise-scale test | The implemented scale equals sensitivity / epsilon | A configuration change silently reduces noise |
| Sensitivity test | The query's sensitivity matches the derivation | A join adds records and changes sensitivity |
| Ledger reconciliation | The budget matches the query log | A query bypasses the gateway |
| Utility test | The interval meets the decision threshold at the smallest cell | A geography change breaks the threshold |
| Membership test | An attack does not beat chance by more than expected | The mechanism is not applied to the model |
| Side-channel review | Errors, timings, and logs do not reveal the answer or the cell size | A rejected query reveals a small cell |

The first four are unit tests and belong in continuous integration. The last two are assessment
activities and belong in the release review. A deployment that has only the first has tested the
arithmetic, not the system.

### Edge cases

**Small cells.** DP noise can produce negative counts and implausible values. Clamping negatives to
zero is permitted post-processing and does not break the guarantee, but it introduces bias,
especially for small counts, where the expected value after clamping can be much higher than the
true value. The release should state the clamping rule and, where possible, publish at a geography
where counts are large enough that clamping is rare.

**Quasi-identifiers.** DP protects against inference about individuals from the released statistics,
but the choice of cells is itself a disclosure channel. A histogram with a bin for a rare diagnosis
in a small area reveals the presence of at least one case, even with noise, if the noise is small
relative to one. Cell boundaries and bin definitions must be chosen with the same care as the
epsilon.

**Re-identification through linkage.** DP bounds what the output reveals about a record, but it does
not prevent linking a released statistic to an individual through auxiliary data when the statistic
is about a small group. The protection is strongest when the unit of privacy matches the group and
the group is large.

**Free-text PHI.** DP applies to queries and models, not to raw text. A pipeline that computes a DP
count over records but releases a sample of notes has not protected the notes. Text releases need
de-identification (Topic 02) or a text-specific mechanism.

**Metadata leakage.** Epsilon accounting covers the intended output, not logs, timings, error
messages, or the fact that a query was rejected. A rejected query can itself reveal that a cell is
small. Side channels are a named hazard in NIST SP 800-226 and must be part of the deployment
review.

**Consent withdrawal.** A DP model trained on a record cannot easily unlearn it. Withdrawal stops
future training, but the model retains the influence. The consent instrument should state this, and
the organisation should have a retraining policy with a defined cadence.

**Minors and deceased patients.** The unit of privacy and the budget do not change for minors, but
the harm is weighted more heavily, and some jurisdictions require stronger safeguards. For deceased
patients, the legal protection may lapse while the family's interest in genetic or inherited
conditions remains; a DP release that includes family-revealing statistics should be assessed for
that harm.

**Cross-border transfer.** DP can support a transfer by reducing the identifiability of the output,
but it does not automatically make a transfer lawful. The recipient's ability to combine the output
with other data, and the legal regime in the destination, are part of the assessment. A DP release
to a country without adequate protection is still a transfer of personal data if the input was
personal data and the output is not anonymous.

**Federated heterogeneity.** Clients differ in population, label prevalence, and data quality. A
federated model can underperform on the smallest or most different client, and the aggregation can
amplify a biased client. The evaluation must be per client, not only on the pooled metric.

**Synthetic data leakage.** A generator trained on a small dataset can memorise records. Testing for
memorisation requires a nearest-neighbour analysis, a membership inference attack, and a review of
rare categories. A synthetic dataset that passes a fidelity check can fail a privacy check.

**Key management.** Secure computation and encryption move the trust from the data holder to the
key holder. A key stored next to the ciphertext provides no protection. Key rotation, separation of
duties, and recovery procedures are part of the mechanism, not an operational detail.

## 6. Worked example

### Setting

A county health department publishes a monthly diabetes dashboard for 50 census tracts, covering
50,000 adults. The dashboard has five indicators per month: diagnosed diabetes count, controlled
HbA1c count, new diagnoses, emergency department visits, and a screening rate. Twelve months times
five indicators is 60 queries per year. The department adopts central differential privacy with an
annual budget of epsilon 1.0 and delta set by the NIST rule delta <= 1/n^2.

### Step 1: the plan as proposed

The analytics team proposes epsilon 0.05 per query, reasoning that 0.05 "sounds small". The
composition arithmetic tells a different story.

For 60 queries at epsilon 0.05:

- Basic composition: 60 x 0.05 = 3.000.
- Advanced composition with delta = 1 / 50,000^2 = 4.0 x 10^-10:
  epsilon x sqrt(2k ln(1/delta)) + k x epsilon x (e^epsilon - 1)
  = 0.05 x sqrt(2 x 60 x 21.6396) + 60 x 0.05 x (e^0.05 - 1)
  = 0.05 x 50.96 + 3 x 0.05127
  = 2.548 + 0.154
  = 2.702.

### Diagnostic 1: the plan exceeds the budget

Both bounds exceed the annual budget of 1.0. The plan is three times over on the simple bound and
2.7 times over on the tighter one. The first diagnostic is visible before any data are released:
the query plan cannot fit the budget it was given.

The ledger makes the timing concrete. At 0.05 per query, the budget is exhausted after 20 queries,
which is the fifth month of a five-indicator dashboard. Months 5 through 12 would be released
without budget, which means the annual claim of epsilon 1.0 is false for two thirds of the year.

### Step 2: the utility arithmetic

For a count query, sensitivity is 1. With epsilon 0.05, the Laplace scale is 1 / 0.05 = 20. The
two-sided 95 percent interval half-width is scale x ln(20) = 20 x 2.9957 = 59.9.

The median tract has 38 adults with diagnosed diabetes. A release with a half-width of 59.9 gives
an interval from roughly -22 to 98 around the true value, before any clamping. Relative to the
estimate, the half-width is 158 percent. The dashboard cannot distinguish a tract with 38 cases
from a tract with 5 or with 90.

For the smallest tracts, with a true count of 10, the interval extends from -50 to 70. The
department clamps negative values to zero, which is permitted post-processing, but the clamp is not
neutral. A simulation of 20,000 draws from the exact Laplace distribution gives:

| True count | Mean after clamping | Clamped share |
|---:|---:|---:|
| 10 | 15.97 | 30.7% |
| 38 | 39.39 | 7.4% |
| 12,000 | 11,999.86 | 0.0% |

### Diagnostic 2: the interval and the clamp bias

For a true count of 10, the released value averages 15.97, a 59.7 percent upward bias, and 30.7
percent of draws are clamped. The bias is systematic and largest where the count is smallest, which
is exactly where the equity analysis matters. The second diagnostic is therefore not only that the
interval is wide but that the correction for negative values distorts the small-area estimates in a
predictable direction.

If the department instead used epsilon 0.01 per query to fit the budget, the scale would be 100 and
the half-width 299.6, which is eight times the median tract count. The plan would fit the budget
and be useless. This is the trade-off in one line: budget compliance and utility are separate tests,
and passing one says nothing about the other.

### Vary one assumption: raise the annual budget

Suppose the department raises the annual budget from 1.0 to 4.0, on the argument that a count of
diagnosed diabetes is a low-sensitivity statistic and that Apple's local DP deployments use
per-feature epsilons of 2 to 8. The 60-query plan at epsilon 0.05 now costs 3.0 on the basic bound
and 2.702 on the advanced bound, both within 4.0. The budget diagnostic disappears.

The utility diagnostic does not. The Laplace scale is still 20, the tract half-width is still 59.9,
and the clamping bias for a true count of 10 is still 15.97. Raising the budget changed the
compliance story and left the decision problem unchanged. The conclusion changes from "the plan is
non-compliant and unusable" to "the plan is compliant and unusable", which is not an improvement
for the dashboard's users.

This is the most useful result in the example. Teams often treat epsilon as the dial that fixes a
DP deployment. It fixes the privacy loss; it does not fix the signal-to-noise ratio at the
geography where the decision is made. If the smallest unit has 10 events, no epsilon that fits a
sensible budget will make that unit's estimate reliable.

### The corrected design

The department changes two things, and only one of them is about epsilon.

1. **Geography.** The dashboard moves from census tracts to the 12 counties in the service area.
   The smallest county has 1,200 adults with diabetes; the mean is about 4,167.
2. **Query count.** The dashboard publishes one query per indicator per quarter rather than per
   month, which is 12 queries per year rather than 60. At 0.05 per query, the basic bound is
   12 x 0.05 = 0.600 and the advanced bound is 0.516, both within the annual budget of 1.0.

The utility arithmetic at county level:

- Smallest county, 1,200 cases: half-width 59.9, or 5.0 percent of the count.
- Mean county, 4,167 cases: half-width 59.9, or 1.4 percent.
- Clamping: for a true count of 1,200, the share of negative draws is effectively zero, so the
  bias disappears.

The plan now fits the budget and meets the decision criterion of plus or minus 10 percent at the
smallest geography. The cost is temporal resolution: quarterly rather than monthly, and county
rather than tract. That cost is a decision for the programme owner, stated in the PET memo, not a
technical default.

### Summary table

| Plan | Epsilon per query | Queries | Basic | Advanced | Half-width | Fits 1.0 | Meets utility |
|---|---:|---:|---:|---:|---:|---|---|
| A: tract, monthly | 0.05 | 60 | 3.000 | 2.702 | 59.9 | No | No |
| B: tract, monthly | 0.01 | 60 | 0.600 | 0.516 | 299.6 | Yes | No |
| C: tract, monthly, budget 4.0 | 0.05 | 60 | 3.000 | 2.702 | 59.9 | Yes (4.0) | No |
| D: county, quarterly | 0.05 | 12 | 0.600 | 0.516 | 59.9 | Yes | Yes |

The table is the deliverable. It shows that plan B and plan D have the same budget cost and
radically different utility, and that plan C fixes compliance without fixing the decision. A
reviewer can challenge any cell: the sensitivity assumption, the delta, the query count, or the
utility threshold. That is what a defensible PET statement looks like.

### A second variation: user-level privacy

The department's original claim was event-level: each record is protected. A patient in the
diabetes registry contributes on average three records per year (two visits and one lab panel).
If the harm is about the patient rather than about a single visit, the unit of privacy should be
the patient, and the sensitivity of a count query becomes 3 rather than 1, because adding or
removing one patient changes the count by up to 3.

At epsilon 0.05, the Laplace scale becomes 3 / 0.05 = 60, and the 95 percent half-width becomes
60 x 2.9957 = 179.7. That is more than four times the median tract count of 38. The event-level
release that looked marginal at tract level is unusable at patient level with the same epsilon, and
the guarantee it provided was about visits, not people. This is the second assumption that a PET
statement must state explicitly, and it is the one most often omitted in vendor material.

If the department wants patient-level protection at the same utility, it must either raise epsilon
by roughly a factor of three (which may be defensible for a low-sensitivity count), reduce the
query count, or move to a larger geography. There is no fourth option that preserves both the
claim and the estimate.

### The PET statement

The deliverable for the programme is a statement that a reviewer can challenge. For the corrected
design it reads:

```text
Mechanism: central differential privacy, Laplace, event-level
Parameter: epsilon 0.05 per query; delta 4.0e-10 (1/n^2 at n = 50,000)
Budget: 1.0 per calendar year; 0.6 allocated to the quarterly dashboard, 0.4 reserved
Workload: 12 count queries per year at county level
Utility: 95% interval half-width 59.9; 5.0% of the smallest county count (1,200)
Residual controls: DUA, access control to raw data, output review, ledger gateway
Known limits: county-level only; no tract estimates; no patient-level guarantee
Review: 2026-12-31 or on any change to geography, query count, or epsilon
```

Six questions a reviewer should ask, and the statement should answer:

1. What is the unit of privacy, and does it match the harm?
2. Is the budget annual and cumulative, and who owns it?
3. Is the sensitivity derived, or assumed?
4. What is the utility at the smallest cell, not the average?
5. What does the mechanism not protect against?
6. What would cause the statement to be re-issued?

A statement that answers all six is unusual, which is the point. Most PET documentation describes
the mechanism and omits the assumptions, and the assumptions are what determine whether the
mechanism protects anyone.

## 7. Code, query, or calculation

The script below implements the budget ledger, both composition bounds, the interval calculation,
and a simulation of the clamping bias. It uses only the standard library. The population and
counts are synthetic.

```python
"""Epsilon budget ledger and utility check for a DP dashboard."""
import random
from math import exp, log


def laplace_scale(sensitivity, epsilon):
    return sensitivity / epsilon


def ci95_halfwidth(scale):
    """Two-sided 95% interval half-width for Laplace(0, scale)."""
    return scale * log(20)


def basic_composition(epsilons):
    return sum(epsilons)


def advanced_composition(epsilons, delta):
    k = len(epsilons)
    e_max = max(epsilons)
    return e_max * (2 * k * log(1 / delta)) ** 0.5 + k * e_max * (exp(e_max) - 1)


class Budget:
    def __init__(self, total):
        self.total = total
        self.spent = 0.0
        self.log = []

    def spend(self, epsilon):
        if self.spent + epsilon > self.total + 1e-12:
            return False
        self.spent += epsilon
        self.log.append(epsilon)
        return True

    def remaining(self):
        return self.total - self.spent


def laplace_draw(mu, scale, rng):
    u = rng.random() - 0.5
    return mu - scale * (1 if u > 0 else -1) * log(1 - 2 * abs(u))


def clamp_bias(true_count, scale, draws=20_000, seed=7):
    rng = random.Random(seed)
    values = [max(0.0, laplace_draw(true_count, scale, rng)) for _ in range(draws)]
    neg = sum(1 for v in values if v == 0.0) / draws
    return sum(values) / draws, neg


if __name__ == "__main__":
    n_population = 50_000
    delta = 1 / n_population ** 2
    print(f"delta (NIST 1/n^2 rule): {delta:.2e}")

    for name, plan in (("plan A 60 x 0.05", [0.05] * 60),
                       ("plan B 60 x 0.01", [0.01] * 60)):
        print(f"{name}: basic={basic_composition(plan):.3f}, "
              f"advanced={advanced_composition(plan, delta):.3f}")

    for eps in (0.05, 0.01):
        scale = laplace_scale(1, eps)
        print(f"epsilon={eps}: scale={scale:.1f}, "
              f"95% half-width={ci95_halfwidth(scale):.1f}")

    b = Budget(1.0)
    for i in range(60):
        if not b.spend(0.05):
            print(f"budget exhausted after {i} queries; "
                  f"remaining={abs(b.remaining()):.3f}")
            break

    for true_count in (10, 38, 12_000):
        scale = laplace_scale(1, 0.05)
        mean_clamped, neg_rate = clamp_bias(true_count, scale)
        print(f"true={true_count}: mean after clamp={mean_clamped:.2f}, "
              f"clamped share={neg_rate:.1%}")
```

Expected output, verified on Python 3.11:

```text
delta (NIST 1/n^2 rule): 4.00e-10
plan A 60 x 0.05: basic=3.000, advanced=2.702
plan B 60 x 0.01: basic=0.600, advanced=0.516
epsilon=0.05: scale=20.0, 95% half-width=59.9
epsilon=0.01: scale=100.0, 95% half-width=299.6
budget exhausted after 20 queries; remaining=0.000
true=10: mean after clamp=15.97, clamped share=30.7%
true=38: mean after clamp=39.39, clamped share=7.4%
true=12000: mean after clamp=11999.86, clamped share=0.0%
```

Four readings of the output. Plan A exceeds the budget on both bounds. Plan B fits the budget and
produces a half-width of 299.6, which is useless at tract level. The ledger exhausts after 20
queries, which is the timing diagnostic. The clamp simulation shows the upward bias at small counts
and its disappearance at county scale.

The simulation uses the exact Laplace distribution by inverse transform, so the numbers are
reproducible. The seed is fixed. A real deployment would also test the tail behaviour, the
floating-point handling of very small counts, and the effect of filtering records before the
query, because each is a named hazard in NIST SP 800-226.

To adapt the script, replace the query plan with the real workload, set delta from the population
size, and add a check that the smallest cell in the publication geography meets the utility
threshold. The ledger should be the system of record for every query, not a spreadsheet maintained
by hand.


## 8. Data, standards, and regulatory context

PETs are technical mechanisms, but their use is shaped by law and by standards that define how a
mechanism may be described.

### NIST SP 800-226 and the evaluation of DP claims

NIST Special Publication 800-226, *Guidelines for Evaluating Differential Privacy Guarantees*, was
published in final form on 6 March 2025. It is the most useful public checklist for a DP
deployment. Three contributions matter here.

First, the "differential privacy pyramid": a claim must be evaluated from the top down, starting
with epsilon and the unit of privacy, then the algorithm and its correctness, then utility and
bias, then the query and threat model, then side channels and security, and finally the data
collection itself. A vendor that quotes epsilon but cannot describe the unit of privacy or the
composition method has not supplied the top of the pyramid.

Second, a catalogue of privacy hazards: floating-point leaks, timing channels, hyperparameter
tuning on private data, inconsistent sensitivity, and backend behaviour that reveals more than the
intended output. These are the failure modes that a mathematically correct mechanism can still
have.

Third, parameter guidance: delta should be no larger than 1/n^2 for a dataset of n records;
epsilon below 1 is generally reasonable, and 0.1 is strong for many purposes; custom
implementations should be avoided in favour of tested libraries. The guidance is not a regulation,
but it is the reference an assessor is likely to use.

### HIPAA

HIPAA does not recognise differential privacy as a de-identification method. A DP release can
support an expert determination under 45 CFR 164.514(b)(1), because the expert can use the
guarantee as part of the argument that re-identification risk is very small. It cannot be used as
Safe Harbor, because Safe Harbor is defined by the removal of 18 identifier categories, not by a
noise mechanism. A covered entity that releases DP statistics about small cells should also check
the actual-knowledge clause and the small-cell risk that DP does not eliminate.

Federated learning and secure computation are security controls, not de-identification. A covered
entity that uses them remains responsible for the access, minimum necessary, and business associate
requirements around the data and the computation.

### GDPR

The GDPR does not mention differential privacy by name. It provides two hooks. Article 25 requires
data protection by design and by default, and Article 32 requires security appropriate to the risk;
both can be satisfied in part by PETs. Recital 26 defines anonymous information by reference to
whether a person is identifiable, considering all means reasonably likely to be used. A DP output
is not automatically anonymous: if the output still relates to an identifiable person, or if the
guarantee is weak relative to the population, the data remain personal.

The EDPB has been cautious about treating PETs as anonymisation. Its pseudonymisation guidelines
(01/2025) treat pseudonymisation as a risk-reduction measure, not as removal from scope, and the
same logic applies to DP: the mechanism reduces risk, and the legal conclusion requires an
assessment of the output in context. For a DP release with a small epsilon and a large population,
that assessment may conclude the data are anonymous; for a release with a large epsilon or a small
population, it may not.

### European Health Data Space

Regulation (EU) 2025/327 creates health data access bodies that issue data permits for secondary
use and requires access through secure processing environments. The regulation applies from
26 March 2027 for governance provisions and from 26 March 2029 for most of the secondary-use
chapter. For a team building a PET-enabled pipeline, the practical effect is that the environment,
not only the mechanism, is regulated: the permit defines the purpose, the data, and the permitted
outputs, and the secure processing environment is where the analysis runs. Differential privacy,
synthetic data, and secure computation can be used inside that environment, but they do not replace
the permit.

### EU AI Act

Regulation (EU) 2024/1689, the Artificial Intelligence Act, entered into force on 1 August 2024 and
applies in stages. For high-risk systems, Article 10 requires data governance covering the
relevant design choices, data collection, preparation, and examination for biases. PETs are not
named as a requirement, but they are one way to satisfy the data-governance and privacy
expectations for training data. A DP-trained model does not by itself satisfy the Act's
transparency, human-oversight, or post-market monitoring obligations.

### Standards and conventions

- ISO/IEC 20889:2018 provides de-identification terminology that includes perturbation and
  synthetic data; it is a vocabulary, not a guarantee.
- ISO/IEC 27559:2022, *Privacy enhancing data de-identification framework*, provides a framework
  for assessing de-identification, including the role of the adversary and the release context.
- The NIST Privacy Framework (version 1.0) places PETs under the Protect-P and Control-P functions
  and requires them to be linked to identified privacy risks, not deployed as a catalogue.

> **Note:** Regulatory content in this file is educational and is not legal advice. The legal
> characterisation of a PET output, especially whether it is anonymous, requires a jurisdiction-
> specific assessment.

The classification to keep straight: a DP epsilon is a mathematical parameter, not a legal
standard; a secure processing environment is an architectural and regulatory control; a data use
agreement is a contractual control; and NIST SP 800-226 is guidance, not law. A PET statement that
mixes these categories in one sentence ("our solution is GDPR-compliant differential privacy") is
not reviewable.

## 9. Industry practice and case evidence

### Deployed differential privacy

**Apple, local DP, 2016-2017.** Apple deployed local differential privacy in iOS 10 and macOS Sierra
and described the system in *Learning with Privacy at Scale* (December 2017). Data are randomised
on the device before transmission, with per-use-case epsilon values and daily donation limits. The
published overview lists epsilon 2 for health data types with one donation per day, epsilon 4 for
emoji and Safari domains, and epsilon 8 for QuickType with two donations per day. Two lessons
follow. First, local DP is feasible at consumer scale, but the epsilon values are much larger than
the "below 1" that NIST describes as generally reasonable for central DP, because local
randomisation pays a accuracy penalty per record. Second, per-feature budgets are an operational
necessity: a single budget for all features would either exhaust immediately or be meaningless.

**US Census Bureau, 2020.** The 2020 Census redistricting data were produced with the TopDown
Algorithm under differential privacy, using a total privacy-loss budget of epsilon 19.61, split as
17.14 for the persons file and 2.47 for housing units. The published parameter is technically rho
under zero-concentrated differential privacy. The number is far above the range NIST describes as
strong, and it is defensible only in context: the data are a national census, the utility
requirements are legal and constitutional, and the alternative was a much less protected release.
The case is the best public example of the point that epsilon is meaningless without the
population, the unit, and the utility requirement.

**Google RAPPOR, 2014.** RAPPOR was an early local-DP system for collecting Chrome client
statistics. It demonstrated that randomised response with hashing could produce useful frequency
estimates without collecting raw values. It also illustrated the accuracy cost of local DP and the
need for large populations, which is why later systems moved toward shuffle models and secure
aggregation.

### Federated learning and its limits

**Google keyboard prediction, 2017.** McMahan and colleagues described federated averaging for
training a next-word prediction model across mobile devices without centralising user data. The
work established the architecture that most federated health projects now follow: local training,
update transmission, server aggregation.

**Gradient leakage, 2019.** Zhu, Liu, and Han showed that shared gradients can be inverted to
reconstruct training inputs and labels, including images and text, with high fidelity. The attack
does not require the server to be malicious in the traditional sense; it requires the server to see
individual gradients. The defences they evaluate include gradient pruning, larger batches, and
cryptographic protection such as secure aggregation.

The practical lesson for health consortia is that federated learning without secure aggregation and
without DP is an architectural control against raw-data transfer, not a privacy guarantee. A
hospital that participates in such a federation has reduced one risk (the centralisation of
records) and retained another (the leakage of information through updates).

### Secure computation and synthetic data in practice

**Secure multi-party computation** has been deployed in production for sensitive benchmarking, for
example the Danish sugar beet auction in 2008, where competing growers computed a market-clearing
price without revealing individual bids. The scale was modest by modern standards, but the case
established that MPC can be operated, not only demonstrated.

**Synthetic data** is widely used for methods development, software testing, and data sharing when
record-level access is difficult. Its privacy properties are inconsistent across vendors. The
assessment questions are: was the generator trained with DP; what is the nearest-neighbour distance
between synthetic and real records; what is the membership inference advantage; and does the
synthetic data reproduce rare categories or free-text identifiers. A synthetic dataset that
preserves the joint distribution well is, other things equal, more likely to be close to real
records, which is the tension the assessment must resolve.

**Synthetic data in health systems.** Several vendors now offer synthetic clinical datasets for
analytics development, and some health systems publish synthetic extracts alongside real ones. The
evaluation is uneven because there is no accepted minimum test. A defensible evaluation reports
three things: a nearest-neighbour distance distribution between synthetic and real records, a
membership inference advantage on the generator, and a subgroup fidelity check for rare
categories. A synthetic dataset that reports only "statistical similarity" has reported utility
without privacy.

**Trusted execution environments.** Cloud providers offer confidential computing with remote
attestation, and health systems use it to run analytics on data that the cloud operator should not
see. The control shifts trust to the hardware vendor and to the attestation service, which is why
supply-chain review and attestation verification are part of the assessment rather than an
implementation detail.

The Census case also shows the cost. The 2020 disclosure avoidance system was contested by users
who argued that noise degraded small-area and tribal estimates, and the Bureau published detailed
accuracy metrics in response. The dispute is not a reason to avoid DP; it is evidence that the
utility analysis must be public and specific, because the populations most affected by noise are
often the ones with the least statistical power.

> **Note:** The deployments above are documented in the cited publications. Statements about what a
> programme should have done are inference, not findings.

### Where practice is weaker than the claim

Three patterns recur in PET assessments, and they are inference from published evaluations rather
than from a single study. First, epsilon is quoted without the unit of privacy, so a reader cannot
tell whether the guarantee is event-level or user-level. Second, composition is not accounted, so
the quoted epsilon is a per-query number presented as an annual one. Third, the evaluation
measures utility on the full population and not on the smallest subgroup, so the mechanism looks
adequate while the subgroup estimates are unusable. Each is checkable: ask for the unit, the
ledger, and the subgroup error.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Epsilon without a budget | Per-query epsilon quoted; no ledger | Composition ignored | Actual privacy loss far exceeds the claim | Budget ledger; composition at plan time |
| Unit of privacy unstated | "Epsilon 1" with no event or user qualifier | Record-level protection claimed for people | Longitudinal records under-protected | State unit; use user-level for longitudinal data |
| Sensitivity wrong | Noise scale computed from intuition | Query sensitivity underestimated | Guarantee invalid | Derive sensitivity formally; test with edge cases |
| Delta too large | Delta 1e-3 on a small dataset | Approximate DP weakened | Privacy failure probability too high | Delta <= 1/n^2 per NIST |
| Local DP for small populations | Per-record noise on a 2,000-person cohort | Local randomisation accuracy cost | Useless estimates | Central DP or larger population |
| Federated learning without secure aggregation | Server logs individual updates | Gradient leakage | Reconstruction of training records | Secure aggregation; DP-SGD; clipping |
| Federated learning without DP | Membership inference succeeds | Model memorises clients | Membership and attribute leakage | DP-SGD; limit access to model |
| Synthetic data assumed safe | No nearest-neighbour or membership test | Generator memorises rare records | Re-identification from synthetic samples | DP training; leakage testing; review |
| Clamping bias ignored | Many negative counts clamped to zero | Post-processing shifts small-cell estimates | Systematic overstatement in small areas | Publish at larger geography; report bias |
| Side channels | Rejected queries reveal cell size | Timing, errors, logs | Disclosure outside the mechanism | Constant-time responses; log review |
| Custom DP implementation | No library version; bespoke noise code | Floating-point and calibration errors | Invalid guarantee | Use vetted library; test calibration |
| Key co-located with ciphertext | Encryption key in the same cloud account | Key compromise defeats encryption | Full disclosure | Key management; separation of duties |
| TEE without attestation | "Confidential computing" claimed, no verification | Misconfiguration or wrong code | Host sees data | Remote attestation; supply-chain review |
| Transfer assumed lawful | "It is DP, so we can send it anywhere" | Legal analysis skipped | Unlawful transfer | Transfer assessment; DPA terms |

### Reading the failure table

The rows are ordered from mechanism-level failures to deployment-level failures. The first group
(epsilon, unit, sensitivity, delta) invalidates the guarantee itself. The second group (local DP
for small populations, federated without secure aggregation, synthetic without testing) invalidates
the fit between mechanism and context. The third group (clamping, side channels, keys, attestation,
transfer) invalidates the operation. In an assessment, the first group should be resolved before
the second, because a context error on top of an invalid guarantee is not worth analysing.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The trust model is correct | Mechanism choice depends on who is untrusted | A party assumed honest is compromised | Threat model review; attestation |
| Sensitivity is correctly bounded | Noise calibration depends on it | Unbounded values or correlated records | Formal derivation; clipping tests |
| The budget is fully accounted | DP guarantee is cumulative | Reruns and ad hoc queries not logged | Reconcile ledger to query log |
| The unit of privacy matches the harm | Event-level may not protect a person | Longitudinal or household records | Recompute with user-level unit |
| The implementation matches the specification | Libraries can be misconfigured | Version drift, parameter changes | Configuration tests; code review |
| The population is large enough for the noise | Utility depends on counts | Small cells and rare conditions | Subgroup error analysis |
| Clamping is documented and bounded | Post-processing affects estimates | Large bias at small counts | Simulate; publish bias note |
| Synthetic data do not memorise | Release safety depends on it | Overfit generator; rare records | Nearest-neighbour; membership attack |
| Secure computation implementation is correct | Protocol security depends on it | Bugs, side channels, collusion | Audit; formal verification where possible |
| Keys are protected | Encryption and MPC depend on keys | Co-location, weak rotation | Key management audit |
| The PET output is not automatically anonymous | Legal conclusion requires assessment | Weak epsilon, small population | Anonymisation assessment; counsel |

Two limits are structural. First, a PET protects a defined asset against a defined adversary; it
does not protect the organisation from poor access control, phishing, or an insider with legitimate
credentials. The Anthem case in Topic 01 is a reminder that the largest breaches often use valid
credentials. Second, the utility cost falls unevenly. A DP release that is adequate in aggregate
can be unusable for the smallest subgroup, which is often the group the equity analysis needs. The
assessment must report subgroup utility, not only the pooled metric.

## 12. Exercises

**Exercise 1 - Classify the mechanism (recall, 20 minutes).**
Classify differential privacy, federated learning, secure aggregation, homomorphic encryption, a
TEE, and synthetic data as a mathematical guarantee, an architectural control, or a statistical
approximation. Deliverable: table with one sentence of justification per row. Criteria: DP is
mathematical; FL and MPC are architectural; synthetic data is statistical; at least one row notes a
combination.

**Exercise 2 - Compute the noise (calculation, 30 minutes).**
For a count query with sensitivity 1, compute the Laplace scale and the 95 percent half-width at
epsilon 0.1, 0.05, and 0.01. Deliverable: calculations with units. Criteria: scale = 1/epsilon;
half-width = scale x ln(20); correct values to one decimal.

**Exercise 3 - Account for the budget (calculation, 40 minutes).**
A dashboard runs 100 queries at epsilon 0.02 and 20 queries at epsilon 0.1. Compute basic and
advanced composition at delta 1e-9. State whether an annual budget of 2.0 is respected. Deliverable:
calculation. Criteria: correct sums; advanced bound applied with the maximum epsilon; conclusion
stated.

**Exercise 4 - Choose a trust model (design, 45 minutes).**
Three hospitals want to train a shared model without a trusted central server. Recommend an
architecture and justify it against central DP and a data clean room. Deliverable: one-page memo.
Criteria: names the trust boundary; addresses secure aggregation and DP; states the residual risk.

**Exercise 5 - Evaluate a vendor claim (critique, 45 minutes).**
A vendor says its product is "differentially private with epsilon 0.5". List five questions you
would ask before accepting the claim. Deliverable: questions with the reason each matters.
Criteria: covers unit, composition, sensitivity, implementation, and side channels.

**Exercise 6 - Test synthetic data (application, 60 minutes).**
Using a synthetic dataset you construct, run a nearest-neighbour check between synthetic and real
records and a simple membership score. Deliverable: script and results. Criteria: distance metric
stated; leakage interpreted, not just reported; limitations named.

**Exercise 7 - Design a PET statement (design, 60 minutes).**
Write the PET statement for the dashboard in Section 6 after the corrected design. Deliverable:
half a page. Criteria: mechanism, epsilon, delta, unit, budget, utility, residual controls, and
review date all present; no claim of anonymity.

**Exercise 8 - Budget failure analysis (critique, 45 minutes).**
The ledger shows 1.4 spent against a budget of 1.0. List the plausible causes in order of
likelihood and the control that would have prevented each. Deliverable: table. Criteria: at least
five causes; each linked to a control; distinguishes technical from governance causes.

## 13. Self-check answers

**Answer 1.** Differential privacy is a mathematical guarantee: it bounds the influence of one
record on the output distribution. Federated learning is an architectural control: it changes where
training happens but does not bound what the updates reveal. Secure aggregation is an architectural
and cryptographic control: it hides individual updates from the server but not the aggregate.
Homomorphic encryption is an architectural control: it hides plaintext from the computing party.
A trusted execution environment is an architectural control based on hardware isolation, and it
depends on attestation and supply-chain trust. Synthetic data is a statistical approximation: it
releases generated records, and its privacy properties must be measured rather than assumed. A
deployment can combine categories, for example federated learning with secure aggregation and
DP-SGD, and the combination must be described as a whole because the weakest component bounds the
claim.

**Answer 2.** At epsilon 0.1: scale = 1 / 0.1 = 10; half-width = 10 x 2.9957 = 29.96, so about
30.0. At epsilon 0.05: scale = 20; half-width = 59.9. At epsilon 0.01: scale = 100; half-width =
299.6. The half-width grows linearly as epsilon falls, which is why small-area estimates become
unusable quickly.

**Answer 3.** Basic composition: 100 x 0.02 = 2.0, plus 20 x 0.1 = 2.0, total 4.0. Advanced
composition uses k = 120 and the maximum epsilon 0.1:
0.1 x sqrt(2 x 120 x ln(1e9)) + 120 x 0.1 x (e^0.1 - 1)
= 0.1 x sqrt(240 x 20.7233) + 12 x 0.10517
= 0.1 x sqrt(4973.6) + 1.262
= 0.1 x 70.52 + 1.262
= 7.052 + 1.262 = 8.314.
The advanced bound is worse than the basic bound here because the maximum epsilon is large relative
to the number of queries; advanced composition helps when many small-epsilon mechanisms are
composed. Either way, the annual budget of 2.0 is exceeded.

**Answer 6.** Compare, for each synthetic record, the distance to its nearest real record and the
distance from a real record to its nearest synthetic record. If the distributions overlap heavily,
the generator is close to copying. A membership score trains a classifier to distinguish synthetic
from real records; an AUC near 1.0 means the synthetic data are distinguishable, which is a warning
about fidelity, not proof of leakage. Leakage is indicated by synthetic records that are very close
to real records in a rare category, or by a membership advantage against the generator itself. The
limitation is that these tests are empirical and depend on the distance metric; they do not
establish a formal bound.

**Answer 5.** Ask: what is the unit of privacy (event or user); what is the composition method and
the total budget across queries; what sensitivity was assumed for each query and how was it
derived; which library and version implements the mechanism, and how is it tested; and what side
channels or auxiliary releases could leak beyond the mechanism. A vendor that answers only the
first question has supplied a number, not a guarantee.

**Answer 8.** Plausible causes, in order: ad hoc queries not routed through the ledger; reruns after
a data refresh counted as new mechanisms; a second team using the same dataset with its own budget;
a parameter change in production that increased epsilon; and a composition calculation that omitted
a query family. The controls are, respectively: a single query gateway; idempotency keys; a
dataset-level budget owner; configuration management with approval; and an automated composition
check at plan time.

**Rubric for Exercise 4.** Full credit requires a named trust boundary, secure aggregation, a DP
mechanism with a unit and budget, a statement of residual risk (model leakage, poisoning,
heterogeneity), and an evaluation plan per site. A recommendation of "federated learning" without
secure aggregation or DP does not pass.

**Rubric for Exercise 6.** Full credit requires a defined distance metric, a comparison
distribution rather than a single nearest neighbour, a membership score with its assumptions, and
an interpretation of whether the result indicates memorisation. Reporting a number without an
interpretation does not pass.

**Rubric for Exercise 7.** Full credit requires all seven elements, a utility statement at the
smallest geography, and no claim that the output is anonymous. A statement that says "epsilon 1"
without the unit or the budget loses half the marks.

**Rubric for Exercise 8.** Full credit requires at least five causes, each with a control, and a
distinction between technical failures (ledger, configuration) and governance failures (ownership,
approval). A list of causes without controls does not pass.

## 14. Interview questions

**1. What does epsilon actually mean?**
Strong answer: a bound on how much the output distribution can change when one record is added or
removed, with smaller epsilon meaning stronger protection and more noise; it is meaningless without
delta, the unit of privacy, and the composition method. Weak answer: "the privacy level".

**2. Why is differential privacy not a de-identification method under HIPAA?**
Strong answer: HIPAA defines Safe Harbor by identifier removal and expert determination by a
risk assessment; DP can support an expert determination but is not a listed method, and small-cell
risk remains. Weak answer: "it makes data anonymous".

**3. What is the difference between local and central DP?**
Strong answer: local adds noise at the source and protects against the data holder at a large
accuracy cost; central adds noise to answers and requires trusting the curator. Weak answer: "local
is more private".

**4. Why can federated learning leak data?**
Strong answer: gradients carry information; inversion and membership attacks can reconstruct
training data; secure aggregation and DP are needed. Weak answer: "the data never leave the
device".

**5. How do you account for a privacy budget?**
Strong answer: basic composition sums epsilon; advanced composition gives a tighter bound with
delta; parallel composition applies to disjoint data; every query and rerun must be logged. Weak
answer: "we set epsilon per query".

**6. What is the unit of privacy and why does it matter?**
Strong answer: event-level protects one record, user-level protects all records from a person;
longitudinal data require user-level or the guarantee is misleading. Weak answer: "a record".

**7. When would you choose synthetic data over DP?**
Strong answer: when the recipient needs record-level data for methods development or software
testing, and the generator can be trained with DP or tested for leakage; not when the recipient
needs guaranteed inference bounds. Weak answer: "synthetic data is always safer".

**8. What does a secure processing environment add beyond a PET?**
Strong answer: it controls where the analysis runs, who can access it, and what outputs leave,
which is the model the EHDS requires; a PET inside it still needs a purpose and a permit. Weak
answer: "it is a secure cloud".

**9. How do you test a DP implementation?**
Strong answer: check the noise scale against the specification, test sensitivity and clipping,
reconcile the ledger to the query log, run an empirical membership attack, and review side channels
per NIST SP 800-226. Weak answer: "we trust the library".

**10. A vendor claims "military-grade privacy". What do you ask?**
Strong answer: asks for the mechanism, the parameter, the unit, the budget, the library, the
threat model, and the evidence; rejects the phrase as unverifiable. Weak answer: asks for a
certification.

## 15. Key terms

- **Advanced composition** - tighter budget bound for many mechanisms, with a delta term.
- **Basic composition** - total privacy loss is the sum of epsilons.
- **Central DP** - a trusted curator adds noise to answers.
- **Clamping** - bounding values before a query; also post-processing negative counts to zero.
- **Delta** - probability of privacy failure beyond epsilon in approximate DP.
- **Differential privacy** - a formal bound on the influence of one record on an output.
- **DP-SGD** - differentially private training with per-example clipping and noise.
- **Epsilon** - privacy loss parameter; smaller is stronger.
- **Federated learning** - training across clients without centralising raw data.
- **Fidelity** - how closely synthetic data match the real distribution.
- **Gaussian mechanism** - approximate-DP mechanism using calibrated Gaussian noise.
- **Gradient leakage** - reconstruction of training data from shared gradients.
- **Homomorphic encryption** - computation on ciphertext.
- **Laplace mechanism** - pure-DP mechanism using calibrated Laplace noise.
- **Local DP** - noise added at the source before transmission.
- **Membership inference** - determining whether a record was used in training.
- **Memorisation** - a model reproducing training records.
- **MPC** - secure multi-party computation over private inputs.
- **PET** - privacy-enhancing technology.
- **Privacy budget** - total epsilon and delta allocated over a period.
- **Privacy hazard** - a practical DP failure mode catalogued by NIST.
- **Secure aggregation** - summing updates without revealing individual ones.
- **Sensitivity** - maximum output change from one record.
- **Shuffle model** - local randomisation plus a trusted shuffler.
- **Side channel** - leakage through timing, errors, or logs.
- **Synthetic data** - generated records fitted to real data.
- **TEE** - hardware-isolated execution environment.
- **Trust boundary** - the line where a party's access changes.
- **t-closeness** - a distributional privacy criterion.
- **Unit of privacy** - event-level or user-level entity protected by DP.
- **Utility** - usefulness of the output for the decision.
- **Zero-concentrated DP** - a DP variant used by the 2020 Census.

## 16. Sources

All URLs accessed 26 October 2026.

**Standards and guidance**
- NIST, *SP 800-226, Guidelines for Evaluating Differential Privacy Guarantees*, March 2025.
  https://csrc.nist.gov/pubs/sp/800/226/final
- NIST, *SP 800-226* (PDF). https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-226.pdf
- NIST, *Privacy Framework, Version 1.0*. https://www.nist.gov/privacy-framework
- ISO/IEC 20889:2018, *Privacy enhancing data de-identification terminology and classification of
  techniques*.
- ISO/IEC 27559:2022, *Privacy enhancing data de-identification framework*.

**Regulation**
- Regulation (EU) 2016/679 (GDPR), Articles 25 and 32, Recital 26.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Regulation (EU) 2025/327 (European Health Data Space).
  https://eur-lex.europa.eu/eli/reg/2025/327/oj
- Regulation (EU) 2024/1689 (Artificial Intelligence Act).
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Data Protection Board, *Guidelines 01/2025 on Pseudonymisation*.
  https://www.edpb.europa.eu/public-consultations/guidelines-012025-on-pseudonymisation_en
- 45 CFR 164.514, HIPAA de-identification standard.
  https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514

**Deployments and case evidence**
- Apple Machine Learning Research, *Learning with Privacy at Scale*, December 2017.
  https://machinelearning.apple.com/research/learning-with-privacy-at-scale
- Apple, *Differential Privacy Overview*.
  https://www.apple.com/privacy/docs/Differential_Privacy_Overview.pdf
- US Census Bureau, *Census Bureau Sets Key Parameters to Protect Privacy in 2020 Census Results*,
  9 June 2021.
  https://www.census.gov/newsroom/press-releases/2021/2020-census-key-parameters.html
- Abowd, J. et al. *The 2020 Census Disclosure Avoidance System TopDown Algorithm*, 2022.
  https://www.census.gov/library/working-papers/2022/adrm/CED-WP-2022-002.html

**Methods and attacks**
- Dwork, C., Roth, A. *The Algorithmic Foundations of Differential Privacy*. Foundations and
  Trends in Theoretical Computer Science, 2014. https://doi.org/10.1561/0400000042
- McMahan, H. B. et al. *Communication-Efficient Learning of Deep Networks from Decentralized
  Data*. AISTATS, 2017. https://arxiv.org/abs/1602.05629
- Zhu, L., Liu, Z., Han, S. *Deep Leakage from Gradients*. NeurIPS, 2019.
  https://arxiv.org/abs/1906.08935
- Shokri, R., Stronati, M., Song, C., Shmatikov, V. *Membership Inference Attacks Against Machine
  Learning Models*. IEEE Symposium on Security and Privacy, 2017.
  https://doi.org/10.1109/SP.2017.41

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_deidentification-and-anonymization.md`](02_deidentification-and-anonymization.md)
- Next topic: [`04_threat-modeling.md`](04_threat-modeling.md)
- Related topic: [`05_access-control-and-cryptography.md`](05_access-control-and-cryptography.md)
- Survey file: [`../../course/16_privacy_security_ethics.md`](../../course/16_privacy_security_ethics.md)
- Labs: [`../../labs/`](../../labs/)
