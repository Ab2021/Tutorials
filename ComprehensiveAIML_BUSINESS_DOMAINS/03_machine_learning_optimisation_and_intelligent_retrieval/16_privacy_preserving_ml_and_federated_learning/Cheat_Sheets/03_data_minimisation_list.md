---
title: "Data Minimisation and Synthetic-Data Limits"
topic: "Data minimisation, purpose limitation, de-identification strength, and the limits of synthetic data"
collection: "Part 03 - Machine Learning, Optimisation and Intelligent Retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Data minimisation and synthetic-data limits

> **Evidence boundary.** Every numeric example in this sheet is CONSTRUCTED and labelled ILLUSTRATIVE.
> Derivations are arithmetic that you can re-run by hand; no benchmark, vendor claim, or organisation
> result is asserted. Where a bound comes from a published theorem, the theorem is named in section 10
> and the bound is derived here rather than quoted.

## 1. File Header

**Purpose of this sheet.** Give a reviewer a repeatable way to decide three questions about a dataset:
which fields must be kept for a stated purpose, which fields must be dropped or coarsened, and whether a
synthetic release is an acceptable substitute for the real data under that purpose.

**Decision this sheet supports.** "Release level" - the choice between row-level real data, de-identified
real data, differentially private aggregates, differentially private synthetic data, and non-private
synthetic data. The sheet also supports the smaller everyday decision of field-by-field retention.

**Who uses it.** A data owner who must answer a request, an ML engineer who wants training data, a
privacy engineer who must sign off, and an incident responder who must judge whether a release already
made was adequate. Each of the four reads different sections and should not skip section 4.4.

**Prerequisites.** You must be able to state a population and a denominator, read a histogram, and
distinguish a probability per record from an expected count over a table. Those two are confused more
often than any other pair in this topic, and section 5 makes the confusion explicit.

**Units used throughout.** Records are counted as rows in the release unit (patient-visit, customer-month,
order). Field cardinality is a count of distinct values. Epsilon is dimensionless and additive across
sequential releases. Bytes are decimal (10^3), and storage figures state the retention window in days.

**Outcome if the sheet is applied correctly.** You can write a retention schedule that names fields,
granularity, retention window, and deletion mechanism; you can state the re-identification probability
bound that your chosen release level buys; and you can say which claims about a synthetic release are
unsupported by that release.

**Exclusions.** This sheet does not give legal advice, does not decide whether a purpose is lawful, and
does not cover federated training topology (sheet 02) or the choice between privacy mechanisms as a
family (sheet 01). It also does not cover access control, key management, or network security except where
deletion depends on them.

**Why a list, not a narrative.** Requests arrive as field lists. The unit of negotiation in practice is
the field, then the granularity of that field, then the retention window on that field. A sheet organised
by those three axes is usable in a review meeting; a sheet organised by legal doctrine is not.

**The three-axis mental model.** Every retained field consumes three budgets: a purpose budget (does a
registered purpose require it), a granularity budget (how much does coarsening cost in utility), and a
time budget (how long does it need to exist at that granularity for that purpose). Minimisation is the
act of removing fields that fail at least one of the three.

**Non-goal that trips reviewers.** Minimisation is not the same as de-identification. Dropping fields
reduces the attack surface; it does not create a guarantee. A release can be minimal and still
re-identifiable if the retained quasi-identifier set is unique in the population.

**Reading order for a live review.** Section 2 for the mental model, section 3 for definitions you must
use precisely, section 4.1 and 4.2 for the two numbers you will be asked to defend (k and epsilon),
section 5 for a worked end-to-end pass, section 8 for the checks that catch mistakes.

**Version note.** This is version 1.0 of the sheet. The review trigger is a change in the underlying
data model, a change in the purpose register, or a change in the release level. Each of those three
invalidates a different part of the sheet.

**Evidence labels used inside.** DERIVED means the arithmetic follows from a stated formula. ILLUSTRATIVE
means the input numbers were chosen to make the arithmetic checkable, not observed. OPEN means the
question has no settled answer and the sheet says so rather than picking one.

**What this sheet refuses to claim.** That k-anonymity provides a privacy guarantee; that synthetic data
is automatically safe; that a small epsilon is always preferable regardless of utility; that minimisation
saves meaningful storage cost. All four claims are common and all four are wrong for stated reasons in
sections 4.4 and 9.

**Signature of a good retention schedule.** One row per field, each row naming: purpose identifier,
retention window in days, granularity at retention, deletion mechanism, owner, and the verification test
that proves deletion happened. If any of the seven is missing, the schedule is a wish list.

## 2. Concept Map / Mental Model

The first mental model is a funnel. A request arrives wide, and each stage removes something. The order
matters: you cannot coarsen a field you have already deleted, and you cannot delete a field after
aggregating it into a cell that cannot be un-aggregated.

```text
requested fields (42)
   |
   +-- purpose test ...... no registered purpose ---------> drop (and log refusal)
   |
   +-- necessity test .... purpose achievable without it --> drop or coarsen
   |
   +-- granularity test .. coarser level still adequate ---> coarsen (age -> 5-yr band)
   |
   +-- uniqueness test ... QI set unique in population ----> suppress / generalise
   |
   +-- time test ......... window shorter than purpose ----> shorten + schedule deletion
   |
   +-- release level ..... rows / aggregates / DP / synthetic
```

The second model is the release ladder. Each rung buys a different kind of statement, and the statements
are not interchangeable. Reading the ladder upward, the guarantee gets stronger and the utility claim
gets weaker.

```text
rung 5  DP synthetic microdata      guarantee: DP on the release      utility: task-specific, measured
rung 4  DP aggregates               guarantee: DP per query family    utility: counts only
rung 3  non-DP synthetic microdata  guarantee: none                   utility: high but unproven
rung 2  k-anonymised rows           guarantee: 1/k per QI match       utility: near-real
rung 1  raw rows with access control guarantee: none (policy only)    utility: full
```

The third model is the synthetic-data pipeline, and it exists to show where the guarantee can be lost.
Three places lose it: the training data selection, the generator's privacy mechanism, and the release
post-processing.

```text
real rows --> [selection] --> training set --> [generator] --> synthetic rows --> [filtering] --> release
                  |                                |                                  |
            selection bias                   no DP noise =>                  filtering on real rows
            changes coverage                 no guarantee                    re-introduces leakage
```

The fourth model is the utility-privacy plane. Every release level is a point on it, and the useful
question is not "which point is best" but "which points are on the frontier for this task". Two releases
can be incomparable when one has better marginal fidelity and the other better rare-stratum coverage.

```text
utility
  ^
  |            raw (not releasable)
  |     k-anon rows *
  |   non-DP synth *        * DP synth (eps=8)
  |                          * DP synth (eps=1)
  |        DP aggregates *
  +----------------------------------> privacy (weaker -> stronger)
```

The fifth model is the field lifecycle. A field is not "in" or "out"; it moves between states, and each
transition has an owner and a test. Auditors ask for the transitions, not the end state.

```text
collected -> validated -> minimised -> used -> archived(coarse) -> deleted -> key destroyed
     |            |            |          |            |              |            |
   consent     schema      purpose     model or     retention      tombstone    unrecoverable
   record      check       register    report       window         row          if crypto-shredded
```

**How to use the models together.** Start at the funnel to produce the field list. Place the resulting
release on the ladder to name the guarantee. If the rung is 3, 4, or 5, use the pipeline model to find
the leakage point and the plane model to argue about the operating point. Use the lifecycle model to
write the schedule. Any review that skips one of the five models produces a specific, predictable defect,
and section 8 names each one.

## 3. Foundational Definitions

Each definition below states the meaning, the unit, and the validity condition that makes it usable.
Terms are ordered from data items to release-level guarantees.

- **Data item** - the smallest unit of collection that a purpose can reference; unit: one field on one
  record. A data item is not a column in a table when the table denormalises, because denormalised
  copies are separate items for retention purposes.
- **Attribute** - a named property of the release unit, with a declared type and cardinality. The
  cardinality matters because it decides how much a generalisation step can coarsen.
- **Direct identifier** - an attribute whose value names or locates a single person without inference;
  unit: attribute. Email, national identifier, full name, device advertising identifier. Removal is
  necessary and cheap.
- **Quasi-identifier (QI)** - an attribute that, combined with others, can single out a record; unit:
  attribute set. Age, sex, region, and date of service are the classic set. The QI set is the object you
  must defend, never an individual attribute.
- **Sensitive attribute** - the attribute whose disclosure is the harm being prevented; unit: attribute.
  In health data this is a diagnosis, an outcome, or a medication. The QI set and the sensitive attribute
  are chosen together.
- **Purpose** - a registered, bounded activity that justifies processing; unit: registry entry with an
  identifier. A purpose without an identifier cannot be tested against a query, so purpose limitation
  fails silently.
- **Purpose limitation** - the invariant that every access maps to a registered purpose. Measured as the
  fraction of queries whose declared purpose is in the register; a value below 1.0 is a finding, not a
  rounding error.
- **Necessity test** - the question "can the purpose be achieved without this field at this
  granularity". It has three outcomes: keep, coarsen, drop. There is no fourth outcome called "keep for
  later".
- **Minimisation ratio** - retained distinct fields divided by requested distinct fields; unit:
  dimensionless, range 0 to 1. It is a completeness-of-review indicator, not a quality score: a ratio of
  1.0 is legitimate only when the necessity test was performed and passed for every field.
- **Retention window** - the number of days a field exists at a stated granularity; unit: days. Windows
  attach to (field, granularity) pairs, because a coarse archive may outlive the row-level copy.
- **Deletion SLA** - the promised maximum days between a deletion trigger and the field becoming
  unreadable; unit: days. Distinct from the retention window, which is about time since collection.
- **Tombstone** - a marker that a record was deleted, retained to keep referential counts consistent.
  A tombstone must not contain the deleted values, or the deletion did not happen.
- **Crypto-shredding** - deletion by destroying the per-subject key rather than the ciphertext rows;
  unit: key. It converts an O(fields x backups) delete into an O(keys) delete, and it is only as strong
  as the key store's isolation.
- **Pseudonymisation** - replacing an identifier with a token that a re-keying table can reverse; unit:
  token. Reversible by design, therefore not anonymisation, and the re-keying table is a direct
  identifier for retention purposes.
- **Tokenisation determinism** - whether the same input maps to the same token across datasets. A
  deterministic token enables joins across datasets, which restores linkability; a randomised token
  prevents joins and destroys utility that depended on the join.
- **Aggregation floor** - the minimum cell count below which a cell is suppressed or merged; unit:
  records. It is a policy number, and section 4.4 shows why it is weaker than a differential privacy
  claim of the same nominal strength.
- **Suppression** - removing a record or cell entirely; unit: records or cells. It preserves the values
  of surviving cells exactly, which is why it is preferred when exact counts matter.
- **Generalisation** - replacing a value with a broader category; unit: levels of a hierarchy. It keeps
  every record, changes every value, and therefore has a computable utility cost.
- **Generalisation hierarchy** - the ordered set of levels for one attribute; unit: tree depth. Age as
  exact, 5-year, 10-year, 20-year is a depth-4 hierarchy with cardinalities 90, 14, 7, 4 over a working
  adult range.
- **Microaggregation** - grouping records by proximity and replacing values with group averages; unit:
  group size. It preserves means better than generalisation but creates values that never existed, which
  breaks downstream range checks.
- **k-anonymity** - the property that every combination of QI values in the release appears at least k
  times. Unit: k, an integer >= 1. It bounds the per-record re-identification probability by 1/k under
  the assumption that the adversary knows only the QI values.
- **l-diversity** - the property that each QI group contains at least l distinct values of a sensitive
  attribute; unit: count of distinct values. It addresses the homogeneity attack but not skew.
- **Entropy l-diversity** - the stronger variant requiring the sensitive distribution in every group to
  have entropy at least log(l). Unit: bits when log base 2. Section 4.4 gives the cardinality ceiling
  that makes it unusable for binary attributes.
- **t-closeness** - the property that the sensitive-attribute distribution inside every group is within
  earth-mover distance t of the global distribution; unit: t in [0,1] for normalised categorical
  distributions. It is the only one of the three that accounts for global skew.
- **Discernibility metric (CDM)** - the sum over groups of the squared group size; unit: records squared.
  It measures how much the release blurs records and is the standard cost for a generalisation choice.
- **Differential privacy (DP)** - a property of a randomised mechanism: changing one record's input
  changes the output distribution by at most a factor e^epsilon, with failure probability delta. Unit:
  epsilon dimensionless, delta a probability.
- **Sensitivity** - the maximum change in a query's output when one record changes; unit: same as the
  query. It is the number that must be bounded before any noise can be calibrated.
- **Composition** - the accounting rule for multiple releases over the same data; unit: epsilon summed
  or composed. Without composition accounting, a stream of individually safe releases becomes unsafe.
- **Post-processing immunity** - any function of a DP output is DP at the same epsilon. This is why a DP
  aggregate can be safely transformed but a non-DP synthetic release cannot be repaired.
- **Amplification by subsampling** - the reduction in epsilon obtained by running a mechanism on a random
  subsample; unit: dimensionless factor. It is the main reason DP-SGD is usable at all.
- **Synthetic record** - a row produced by a generative model rather than copied from a person. It has no
  subject, so a "the subject consented" argument does not apply to it.
- **Generator** - the model that produces synthetic records; unit: parameters. Its privacy properties, not
  the release's format, determine whether the release is private.
- **Marginal fidelity** - agreement between real and synthetic distributions for one column or a small
  column set; unit: total variation distance in [0,1]. Cheap to measure, easy to over-report.
- **Joint fidelity** - agreement for the full column set; unit: distance on the joint distribution. It is
  the quantity that decides whether the synthetic data is usable for a multivariate task.
- **Coverage** - the fraction of real strata that appear in the synthetic release with adequate mass;
  unit: fraction of strata. The usual silent failure of synthetic data.
- **Mode dropping** - the generator's tendency to omit rare modes entirely while fitting common ones
  well. It is a coverage failure that marginal fidelity cannot see.
- **Memorisation** - the generator reproducing a real record closely enough to be a disclosure; unit:
  distance between synthetic and nearest real record. Detected by a distance-based test, not by looking
  at the data.
- **Membership inference** - an attack that decides whether a specific record was in the training set.
  Measured by attack AUC or by advantage; unit: advantage in [0,1].
- **Attribute inference** - an attack that infers a sensitive attribute from released attributes. It
  survives even when membership inference fails, which is why both must be tested.
- **Singling out** - the ability to isolate one individual in the release. Related to but weaker than
  re-identification; an aggregate release can permit singling out without naming anyone.
- **Linkability** - the ability to join two releases on shared attributes. Deterministic
  pseudonymisation is the usual cause.
- **Disclosure control** - the set of release-time edits (suppression, generalisation, noise) applied to
  a table. The term describes the edit, not the guarantee.
- **Data provenance** - the record of which source records and transformations produced a release; unit:
  lineage edges. Required to answer "was this synthetic row derived from a person who withdrew".
- **Canary record** - a deliberately inserted, unique record used to test whether a release leaks. If
  the canary is recoverable from the release, the release leaks by construction.

## 4. Core Formulas with Full Derivation

Each block uses the same five-part anatomy: FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE,
INTERPRETATION. Every block ends with an explicit validity condition, because a formula used outside its
validity condition is the most common defect in a privacy review.

### 4.1 Base case

The base case is a single release of a single table, with one generalisation choice and one k value. Two
numbers decide whether it may be released: the per-record re-identification bound and the utility cost of
the coarsening that produced the bound.

**Block 4.1.1 - k-anonymity bound.**

```text
FORMULA
  P(re-identify record i | adversary knows QI of i) <= 1 / k_i
  k_i = size of the QI group containing record i

WHERE
  k_i   : integer >= 1, records in the equivalence class of record i
  i     : index over released records
  bound : probability per record, dimensionless

CONSTRAINT
  Valid only if the adversary knows the QI values of the target and nothing else about the release.
  The bound is a per-record probability, not a guarantee about the table.

NUMERICAL EXAMPLE
  N = 50,000 records, QI set = {age band (14 levels), sex (2), region (12)}
  cells = 14 x 2 x 12 = 336
  mean cell size = 50,000 / 336 = 148.81 records
  policy k = 5  =>  bound per record = 1/5 = 0.20

INTERPRETATION
  A 20% per-record chance is a weak bound for a table this size. If the adversary is right 1 time in 5
  and can test their guess against an external source, one attempt per target is enough to be dangerous.

VALIDITY CONDITION
  Holds for the QI set as released. If two QI sets are released for the same records at different times,
  the effective k is the minimum over the intersection, not the maximum.
```

**Block 4.1.2 - discernibility cost of a generalisation choice.**

```text
FORMULA
  CDM = SUM over groups g of |E_g|^2
  CDM_norm = CDM / N^2

WHERE
  |E_g|     : number of records in equivalence class g
  N         : total records, here 50,000
  CDM       : records squared
  CDM_norm  : dimensionless, in (1/N, 1]

CONSTRAINT
  Requires that the groups partition the released records. Suppressed records must be excluded from
  both numerator and denominator or the metric becomes incomparable across releases.

NUMERICAL EXAMPLE (uniform approximation, ILLUSTRATIVE)
  level A: 336 cells, mean 148.81 -> CDM = 336 x 148.81^2 = 336 x 22,144.4 = 7.44e6
  level B: 168 cells, mean 297.62 -> CDM = 168 x 88,577.7 = 1.49e7
  level C:  96 cells, mean 520.83 -> CDM =  96 x 271,264 = 2.60e7
  normalised: 7.44e6/2.5e9 = 0.00298 ; 1.49e7/2.5e9 = 0.00595 ; 2.60e7/2.5e9 = 0.01040

INTERPRETATION
  Each halving of cell count roughly doubles CDM. Doubling the cell count costs 2x the blur, so the
  cheapest generalisation that reaches the policy k is the correct one; over-generalising is a real cost.

VALIDITY CONDITION
  The uniform approximation is wrong whenever cells differ by more than about 3x in size. Use the exact
  sum when a real histogram is available; the ratio between levels survives, the absolute value does not.
```

**Block 4.1.3 - utility retention of a synthetic release.**

```text
FORMULA
  retention = AUC_TSTR / AUC_TRTR
  gap       = AUC_TRTR - AUC_TSTR

WHERE
  AUC_TRTR : model trained and tested on real data
  AUC_TSTR : same model family trained on synthetic, tested on real held-out data
  both     : dimensionless in [0.5, 1]

CONSTRAINT
  Requires an identical model family, identical hyperparameters, and a real test set that the generator
  never saw. Reporting TSTR alone hides the baseline and is not evidence.

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  AUC_TRTR = 0.806 ; AUC_TSTR = 0.761
  retention = 0.761 / 0.806 = 0.9442
  gap       = 0.045

INTERPRETATION
  A 4.5-point AUC gap is acceptable for triage and unacceptable for a payment decision. The number is
  task-specific; the same generator can be adequate for one purpose and inadequate for the next.

VALIDITY CONDITION
  Holds only for the task and prevalence tested. A synthetic set that retains AUC at 30% prevalence can
  lose far more at 2% prevalence because the generator never learned the rare class.
```

### 4.2 General case

The general case adds the two things that make real reviews hard: repeated releases over the same data
(composition) and an adversary who is not restricted to the QI set (leakage measurement).

**Block 4.2.1 - epsilon composition across a release family.**

```text
FORMULA
  basic    : eps_total = SUM_{j=1..k} eps_j
  advanced : eps_total = eps * ( sqrt(2k ln(1/delta)) + k(e^eps - 1) )
             (k mechanisms each (eps, 0)-DP, composed at (eps_total, delta))

WHERE
  k     : number of sequential releases on the same data
  eps   : per-release epsilon, dimensionless
  delta : total failure probability, here 1e-6
  ln    : natural logarithm

CONSTRAINT
  Advanced composition is a bound, not an equality, and it only improves on basic composition when
  e^eps - 1 < 1, i.e. eps < ln 2 = 0.6931. Above that, the k(e^eps - 1) term alone exceeds k*eps.

NUMERICAL EXAMPLE (k = 12, eps = 0.1, delta = 1e-6)
  basic    = 12 x 0.1 = 1.20
  sqrt(2k ln(1/delta)) = sqrt(24 x 13.8155) = sqrt(331.572) = 18.209
  eps * 18.209 = 1.821 ; k*eps*(e^0.1 - 1) = 1.2 x 0.10517 = 0.126
  advanced = 1.821 + 0.126 = 1.947   (worse than basic)

NUMERICAL EXAMPLE (k = 100, eps = 0.05, delta = 1e-6)
  basic    = 5.00
  sqrt(2k ln(1/delta)) = sqrt(200 x 13.8155) = sqrt(2763.1) = 52.57
  advanced = 0.05 x (52.57 + 100 x 0.05127) = 0.05 x 57.70 = 2.885   (better than basic)

INTERPRETATION
  At eps = 0.1 you need roughly 35 queries before advanced composition pays off; below that, report the
  basic sum and stop. Never report the smaller of two bounds without stating which bound was used.

VALIDITY CONDITION
  Requires that every release touches the same underlying records. Independent datasets compose to the
  max, not the sum; mixing the two rules is the standard accounting error.
```

**Block 4.2.2 - membership advantage bound and its inversion.**

```text
FORMULA
  forward  : advantage <= (e^eps - 1) / (e^eps + 1) = tanh(eps / 2)
  inverse  : eps >= ln( (1 + advantage) / (1 - advantage) )

WHERE
  advantage : TPR - FPR of the best membership test, in [0,1]
  eps       : the mechanism's epsilon
  tanh      : hyperbolic tangent

CONSTRAINT
  The forward bound applies to (eps, 0)-DP. The inverse is a lower bound on eps implied by an observed
  advantage; it is not an upper bound and must never be reported as "the model's epsilon".

NUMERICAL EXAMPLE
  eps = 0.1 -> tanh(0.05) = 0.04996  (about 5 points of advantage)
  eps = 1.0 -> tanh(0.50) = 0.46212
  eps = 2.0 -> tanh(1.00) = 0.76159
  observed advantage 0.36 -> eps >= ln(1.36/0.64) = ln(2.125) = 0.754

INTERPRETATION
  An attack that reaches 0.36 advantage proves the release is at least eps = 0.75 equivalent, even if
  the accountant said 0.1. The attack result is evidence; the accountant's number is a claim.

VALIDITY CONDITION
  The bound is tight only for the worst-case neighbouring datasets and only for pure DP. Under
  (eps, delta)-DP add delta to the advantage and re-state the bound as approximate.
```

**Block 4.2.3 - calibration of Gaussian noise from a stated epsilon.**

```text
FORMULA
  sigma = sensitivity * sqrt(2 ln(1.25 / delta)) / eps

WHERE
  sigma       : noise standard deviation, in query units
  sensitivity : maximum L2 change from one record, in query units
  delta       : failure probability, here 1e-5 or 1e-6
  eps         : target epsilon

CONSTRAINT
  Assumes the query has bounded L2 sensitivity and that records are independent. Unbounded-sensitivity
  queries (means over an unknown denominator) need a bounded denominator first.

NUMERICAL EXAMPLE (sensitivity = 1)
  delta = 1e-5: 2 ln(1.25/1e-5) = 2 x 11.736 = 23.472 ; sqrt = 4.845
    eps = 1.0 -> sigma = 4.845 ; eps = 0.1 -> sigma = 48.45
  delta = 1e-6: 2 ln(1.25e6) = 2 x 14.039 = 28.078 ; sqrt = 5.000
    eps = 1.0 -> sigma = 5.000

INTERPRETATION
  Tightening delta from 1e-5 to 1e-6 costs 3% more noise. Tightening eps from 1.0 to 0.1 costs 10x more
  noise and 100x more variance. Budget decisions are dominated by epsilon, so negotiate epsilon first.

VALIDITY CONDITION
  Only for a single query. For a family, calibrate per query then compose; calibrating once for the family
  and reusing sigma understates the total epsilon.
```

**Block 4.2.4 - relative error of a noisy count.**

```text
FORMULA
  sd(noise) = sigma * sqrt(2)            (Gaussian, per coordinate)
  rel_err   = sd(noise) / true_count

WHERE
  sigma     : noise scale from Block 4.2.3
  true_count: the cell's real count, known to the analyst only for evaluation
  rel_err   : dimensionless

CONSTRAINT
  Requires true_count > 0. The formula says nothing about bias, only about spread; DP counts are
  unbiased, but a post-hoc rounding or flooring step introduces bias that this formula misses.

NUMERICAL EXAMPLE (eps = 0.1, sensitivity 1, delta = 1e-5, sigma = 48.45)
  sd = 48.45 x 1.4142 = 68.52
  cell of 154 -> 68.52/154 = 0.445   (44.5% relative error)
  cell of 12  -> 68.52/12  = 5.71    (571% relative error, sign meaningless)

INTERPRETATION
  At eps = 0.1 a per-cell count is unusable at small cells. Either raise epsilon, merge cells until counts
  are large, or publish a different statistic. Publishing anyway produces a number that looks precise.

VALIDITY CONDITION
  Applies to independent noise per cell. Correlated noise from a top-down histogram method has different
  per-cell variance and must be evaluated from that method's own variance formula.
```

### 4.3 Variants

**Block 4.3.1 - entropy l-diversity ceiling.**

```text
FORMULA
  H(group) = - SUM_v p_v log2 p_v  >=  log2(l)

WHERE
  p_v : fraction of the group with sensitive value v
  H   : entropy in bits
  l   : required number of well-represented values

CONSTRAINT
  H is maximised at log2(cardinality). If log2(cardinality) < log2(l) the requirement is infeasible for
  every group, so the release cannot satisfy it at all.

NUMERICAL EXAMPLE
  binary sensitive attribute (complication yes/no): max H = log2(2) = 1.000 bit
  l = 2 requires H >= 1.000 -> feasible only at exactly 50/50
  l = 3 requires H >= 1.585 -> infeasible, since 1.585 > 1.000
  a group with 6 of 8 positive: p = 0.75 -> H = -(0.75 log2 0.75 + 0.25 log2 0.25)
                                     = -(0.75 x -0.415 + 0.25 x -2.000) = 0.811 bits

INTERPRETATION
  For a binary outcome, entropy l-diversity with l >= 3 is not a hard requirement, it is an impossible
  one. A reviewer who writes it into policy has produced an unsatisfiable spec.

VALIDITY CONDITION
  Requires a categorical sensitive attribute with a known value set. For continuous attributes, discretise
  first; the choice of bins changes H and therefore changes the verdict.
```

**Block 4.3.2 - t-closeness distance for a binary attribute.**

```text
FORMULA
  t_group = | p_group - p_global |        (binary attributes, EMD reduces to absolute difference)

WHERE
  p_group  : fraction positive inside the equivalence class
  p_global : fraction positive in the whole table
  t_group  : earth-mover distance in [0,1]

CONSTRAINT
  Only for binary attributes. For k categories, EMD is half the L1 distance between the two distributions
  and can exceed the largest single-category difference.

NUMERICAL EXAMPLE (p_global = 0.30)
  group with 6 of 8 positive: p_group = 0.75 -> t = |0.75 - 0.30| = 0.45
  group with 3 of 8 positive: p_group = 0.375 -> t = 0.075
  group with 2 of 8 positive: p_group = 0.25 -> t = 0.05
  policy t <= 0.20 -> the first group fails, the other two pass

INTERPRETATION
  t-closeness catches the group that k-anonymity and l-diversity both accept: eight records, k = 8, two
  distinct values so l-diversity passes, but the group tells the adversary the outcome with 75% confidence.

VALIDITY CONDITION
  Global distribution must be computed on the same population as the release. Using a national baseline
  for a regional table produces t values that do not describe the table.
```

**Block 4.3.3 - secure-aggregation round survival.**

```text
FORMULA
  S ~ Binomial(n, 1 - p_drop)
  round succeeds iff S >= t
  E[S] = n(1 - p_drop) ; sd(S) = sqrt(n p_drop (1 - p_drop))

WHERE
  n      : clients invited per round
  p_drop : independent per-client dropout probability
  t      : reconstruction threshold (shares needed)
  S      : survivors, integer

CONSTRAINT
  Assumes independent dropout. Correlated dropout (one network partition, one OS version) breaks the
  independence and makes the tail far heavier than the binomial says.

NUMERICAL EXAMPLE (n = 100, t = 70)
  p_drop = 0.15: E[S] = 85, sd = sqrt(100 x 0.15 x 0.85) = 3.571
    z = (69.5 - 85)/3.571 = -4.34 -> P(fail) about 7e-6  (negligible)
  p_drop = 0.30: E[S] = 70, sd = sqrt(100 x 0.30 x 0.70) = 4.583
    z = (69.5 - 70)/4.583 = -0.109 -> P(fail) about 0.457  (a coin flip)

INTERPRETATION
  A 15-point change in dropout moves the failure probability by five orders of magnitude. Threshold
  choice, not cryptography, is what makes a secure-aggregation deployment survive a bad week.

VALIDITY CONDITION
  Holds when dropout is independent and when t is fixed. If t is chosen adaptively after seeing S, the
  probability statement no longer applies to the deployed rule.
```

**Block 4.3.4 - coverage of a rare stratum.**

```text
FORMULA
  coverage_s = ( synthetic records in stratum s ) / ( real records in stratum s )

WHERE
  s        : a cell of the cross-classification that matters for the task
  coverage : dimensionless; 1.0 means the release preserves the stratum's mass per real record

CONSTRAINT
  Coverage must be measured on strata defined before looking at the synthetic data, otherwise the choice
  of strata is fitted to the result. Report both the stratum definition and its real count.

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  real: age 80+ and complication yes -> 240 of 50,000 = 0.48%
  synthetic: 71 such rows
  coverage = 71 / 240 = 0.296
  weighted by prevalence: synthetic share = 71/50,000 = 0.14% vs real 0.48%

INTERPRETATION
  30% coverage on the smallest clinically important stratum is the typical synthetic-data failure. It is
  invisible in any per-column marginal and it is exactly the stratum where the model matters.

VALIDITY CONDITION
  Requires the real stratum count to be known and the synthetic release to be unweighted. A weighted
  synthetic release must be reweighted before coverage is computed, or the ratio is meaningless.
```

### 4.4 Limits and counterexamples

**Limit 1 - k-anonymity has no adversarial guarantee.** The bound 1/k assumes the adversary knows only
the QI values. Give the adversary one extra fact - a prescription date, a clinic visit - and the group
collapses. The counterexample is a group of 8 records that are 6/2 on the outcome: k = 8 looks strong,
and Block 4.3.2 shows the group leaks the outcome at 75% confidence. k bounds a quantity the adversary
rarely has.

**Limit 2 - the expected-count fallacy.** From 1/5 per record, an analyst computes 49,988/5 = 9,998 and
claims "we expect ten thousand re-identifications". That mixes a per-record probability with an expected
count over a table and, worse, implies the adversary attacks every record. The correct statement is: for
any single target the adversary has a 20% chance given the QI; the number of *attempts* the adversary can
make is bounded by their external data, not by the table.

**Limit 3 - minimisation does not reduce record counts.** Dropping fields 42 -> 9 leaves 50,000 records.
If exposure is priced per record, minimisation of fields changes nothing: 50,000 x 180 = 9,000,000 either
way. What changes exposure is reducing the release *unit* to 336 aggregates: 336 x 180 = 60,480, a factor
of 149 smaller. Field minimisation reduces severity per record; unit minimisation reduces the count.

**Limit 4 - synthetic data without DP has no guarantee.** A generative model trained on real data and
sampled without a privacy mechanism inherits no bound. It may be empirically hard to attack, which is not
the same as safe. Post-processing cannot fix it, because post-processing immunity only protects DP inputs.
The counterexample that matters: a generator that overfits reproduces rare rows closely, and the release
team never measures nearest-neighbour distance because the deliverable "looks synthetic".

**Limit 5 - utility measured on the wrong task.** TSTR retention of 0.944 measured on the full population
says nothing about the rare stratum where coverage is 0.296. If the purpose is rare-event detection, the
correct evaluation is TSTR restricted to the rare stratum, and that number will be far lower.

**Limit 6 - epsilon without a release count.** "eps = 1" is not a privacy statement until it names how
many releases compose to it. One release at 1.0 and twelve releases at 0.1 are 1.0 and 1.2 respectively;
the second is worse while each individual number looks better.

**Limit 7 - aggregation floors are not differential privacy.** A floor of 5 cells means a cell of 6 is
published exactly. An adversary with two neighbouring queries (add one record, or remove one) learns the
change. DP with eps = 1 and sensitivity 1 would report a count with sd 4.845, which is a different and
defensible claim. The floor answers "is this cell small", not "what can be inferred".

**Limit 8 - the impossible l-diversity spec.** As computed in Block 4.3.1, l = 3 on a binary sensitive
attribute is unsatisfiable. Policies get written with l = 3 because it sounds stronger than l = 2, and
the implementation then quietly reports l = 2. The failure is invisible unless the cardinality is checked
when the policy is written.

**Limit 9 - composition rules cannot be mixed.** Summing epsilons across datasets that are disjoint
overstates; taking the max across releases on the same records understates. A release family that spans
both cases needs the sum over each dataset and the max across datasets.

**Limit 10 - deletion that does not delete.** Crypto-shredding deletes a key, not a value. If the key is
also escrowed for backup restore, the value is recoverable and the deletion claim is false. The test is
not "did the delete job exit 0" but "can the value be recovered from every replica including the offline
one", and that test is the only one that counts.

## 5. Worked Numerical Example

**Scenario (ILLUSTRATIVE).** A regional diabetes service receives a request for data to support an annual
complications audit. The request lists 42 fields covering 50,000 patient-visits over seven years. The
stated purpose is: "estimate the proportion of patients with a recorded complication in each region, by
age band and diabetes type, for the most recent complete year". Nothing in that sentence requires
row-level identity, exact dates, or free-text notes.

**Step 1 - purpose test.** Each requested field is checked against the purpose sentence. Fields whose
values the sentence cannot consume are refused with a logged reason: free-text notes (12 fields), exact
service date (1), national identifier (1), postcode (1), clinician identifier (1), four billing fields.
That is 20 refusals, leaving 22 candidates.

**Step 2 - necessity test.** Of the 22 candidates, 8 duplicate the same information at finer granularity
(four date-derived fields, two address-derived fields, two diagnosis-code variants). Keeping the coarser
version of each pair removes 8 fields, leaving 14. A further 5 fields carry no information the audit
consumes (two administrative flags, three optional laboratory panels). The kept set is 9 fields.

```text
minimisation ratio = 9 / 42 = 0.2143  (21.4% of the requested fields retained)
refusal log entries = 20 purpose + 8 necessity(duplicate) + 5 necessity(unused) = 33
fields coarsened     = 4 of the 9 (age, date, region, HbA1c)
```

The refusal log is the deliverable here, not the ratio. A ratio of 0.21 with 33 logged reasons is
defensible; a ratio of 0.21 with no log is indistinguishable from an unexamined request.

**Step 3 - granularity choice.** The retained QI set is age band, sex, and region. Three candidate
hierarchies are compared on the two numbers from section 4.1: the smallest cell (which sets k) and CDM
(which prices the blur).

```text
level A: age 5-yr (14), sex (2), region (12) -> 336 cells
  mean cell 148.81 ; smallest cell 3 ; CDM 7.44e6 ; CDM_norm 0.00298
level B: age 10-yr (7), sex (2), region (12) -> 168 cells
  mean cell 297.62 ; smallest cell 6 ; CDM 1.49e7 ; CDM_norm 0.00595
level C: age 20-yr (4), sex (2), region (12) ->  96 cells
  mean cell 520.83 ; smallest cell about 12 ; CDM 2.60e7 ; CDM_norm 0.01040
```

Policy requires k >= 5. Level A fails on four cells holding 3 records each; level B passes with margin;
level C passes with more blur than needed. The correct choice is level B: the cheapest level that
satisfies policy. Choosing C "to be safe" doubles CDM again for no policy benefit.

```text
records affected by the four failing cells = 4 x 3 = 12
suppression alternative: keep level A, suppress 12 records
  released = 50,000 - 12 = 49,988 ; loss = 12/50,000 = 0.024%
  CDM at level A with suppression = 7.44e6 (unchanged for surviving cells)
  utility: age precision retained, 12 records lost
```

Two correct answers exist and they trade different things. Generalising to level B costs every record
some precision and loses no records; suppressing at level A keeps precision and loses 12 records. For an
audit whose output is a proportion, losing 12 records from cells that are already tiny biases the small
cells; coarsening does not. Prefer level B here, and record the reasoning.

**Step 4 - the k-number versus the expected-count number.** The reviewer is asked "how many patients can
be re-identified". Two answers follow from the same k = 6.

```text
per-record bound: 1/6 = 0.167  (16.7% chance for one target, given QI knowledge)
naive expected count: 49,988 / 6 = 8,331  (WRONG framing: assumes every record attacked)
```

The first number is a bound and is correct as stated. The second is the fallacy from Limit 2 and must be
corrected in the review: the adversary's attempt budget is set by their external data, not by our table.

**Step 5 - sensitive-attribute check.** The outcome is binary (complication recorded yes/no). Entropy
l-diversity is checked before it is promised.

```text
max entropy for binary = log2(2) = 1.000 bit
policy asks l = 3 -> needs H >= 1.585 bits -> infeasible
measured minimum group entropy at level B: H_min = 0.811 bits -> l_eff = 2^0.811 = 1.75
t-closeness check, p_global = 0.30:
  group 6/8 positive -> t = 0.45  (fails t <= 0.20)
  group 3/8 positive -> t = 0.075 (passes)
```

The finding is that the release satisfies k but not the policy's l, and fails t on one group. The honest
resolution is to merge the failing group into a neighbour and re-measure, not to lower t silently.

**Step 6 - cost of the retention schedule.** Storage is computed to show that storage is not the argument.

```text
42 fields: 18 numeric x 8 B = 144 B ; 24 text x 40 B = 960 B -> 1,104 B/record
  50,000 x 1,104 B = 55.2 MB
 9 fields: 6 numeric x 8 B = 48 B ; 3 text x 40 B = 120 B -> 168 B/record
  50,000 x 168 B = 8.4 MB
  reduction = 1 - 168/1,104 = 0.8478  (84.8%)
 retention 7 years = 365 x 7 = 2,555 days
 hot storage at 0.023 USD/GB-month: 0.0552 GB x 0.023 = 0.00127 USD/month
```

At one hundredth of a dollar per month, storage cost does not justify the review. The justified benefits
are: fewer fields to leak, fewer fields to answer for in a subject access request, and a shorter deletion
surface (9 fields instead of 42 across three replicas).

**Step 7 - synthetic alternative.** The requester asks for synthetic rows instead, to avoid the review.

```text
generator: DP-SGD, clip C = 1.0, sigma = 1.1, batch 256, 30 epochs, N = 50,000
  steps    = 30 x ceil(50,000/256) = 30 x 196 = 5,880
  q        = 256/50,000 = 0.00512
  eps approx = (q/sigma) x sqrt(2 x steps x ln(1/delta))
             = (0.00512/1.1) x sqrt(2 x 5,880 x 11.5129)
             = 0.004655 x sqrt(135,392) = 0.004655 x 367.96 = 1.713
fidelity: mean marginal TV = 0.031 ; mean pairwise correlation gap = 0.08
utility:  AUC_TRTR 0.806 ; AUC_TSTR 0.761 ; retention 0.944 ; gap 0.045
coverage: age 80+ with complication -> 240 real, 71 synthetic -> 0.296
attack:   membership AUC 0.53 -> advantage 0.06  (2 x 0.53 - 1)
```

The synthetic release at eps = 1.7 has a defensible privacy story and a task-specific utility story that
holds for the main population and fails for the smallest stratum. Because the audit's purpose is a
proportion by region and age band, and the smallest band's coverage is 0.296, the synthetic release is
not adequate for that specific purpose. It would be adequate for a purpose restricted to bands with
coverage above 0.8.

**Independent arithmetic re-check.** Every figure above is recomputed from its stated inputs.

```text
9/42        = 0.214286 -> 21.4%           (matches)
336 x 148.81^2 : 148.81^2 = 22,144.4 ; x336 = 7,440,518 -> 7.44e6   (matches)
168 x 297.62^2 : 297.62^2 = 88,577.7 ; x168 = 14,881,053 -> 1.49e7  (matches)
7.44e6 / 2.5e9 = 0.002976                (matches 0.00298)
49,988 / 6     = 8,331.3                 (matches 8,331)
18 x 8 + 24 x 40 = 144 + 960 = 1,104     (matches)
50,000 x 1,104 = 55,200,000 B = 55.2 MB  (matches)
30 x 196 = 5,880 ; 2 x 5,880 x 11.5129 = 135,392 ; sqrt = 367.96
0.00512 / 1.1 = 0.0046545 ; x 367.96 = 1.7126  (matches 1.713)
71/240 = 0.29583                         (matches 0.296)
```

**Sensitivity note.** The conclusion that level B is correct is robust to a +/-20% error in cell sizes:
level A's smallest cell would have to rise from 3 to 5, a 67% change, to pass policy. The conclusion that
the synthetic release is inadequate for the small stratum is robust to a doubling of coverage (0.296 to
0.59 still fails an 0.8 threshold). The conclusion that storage is not the justification is robust to a
10x error in unit price: 0.0127 USD/month remains negligible. The least robust number is the epsilon
approximation in step 7, which depends on the accountant and the sampling assumption; a factor-of-two
error there changes the release decision, so it must be recomputed with the deployed accountant rather
than reused from this sheet.

## 6. Visual / Geometric Intuition

**The generalisation lattice.** Coarsening options for three attributes form a lattice, and the lattice
nodes are ordered by CDM. Reviewers should walk the lattice upward until the policy k is met and stop.

```text
(age14, sex2, region12) 336 cells  CDM 7.44e6   k_min 3   FAIL
        |                    |
(age7,  sex2, region12) 168 cells  CDM 1.49e7   k_min 6   PASS  <-- cheapest pass
        |                    |
(age7,  sex2, region6 )  84 cells  CDM 2.98e7   k_min 12  PASS  (over-generalised)
        |
(age4,  sex2, region12)  96 cells  CDM 2.60e7   k_min 12  PASS  (over-generalised)
```

Two nodes can pass at similar cost from different directions. The choice between them is a utility
question - which attribute does the model actually use - not a privacy question.

**The suppression-versus-generalisation frontier.** For a fixed k, the two tools differ in what they
destroy. The frontier below is drawn for the 50,000-record example.

```text
records lost (%)
  2.0 |  *
  1.5 |     *
  1.0 |        *
  0.5 |           *
  0.0 |              *---------------- suppression at level A (12 records, 0.024%)
      +----------------------------------> precision lost (CDM_norm)
        0.00  0.003  0.006  0.010  0.015
```

Each point is a distinct (suppression set, generalisation level) pair. Moving right costs precision for
all records; moving up costs records for a few cells. A release that sits at the top-right is strictly
worse than one of its neighbours and should never be chosen.

**The epsilon-to-advantage curve.** Block 4.2.2 is a curve, and reading it as a table of three numbers
hides the shape: advantage rises steeply between eps = 0.5 and eps = 2, then saturates.

```text
eps    0.1    0.5    1.0    2.0    4.0    8.0
tanh   0.050  0.245  0.462  0.762  0.964  0.999
```

The practical reading: below eps = 0.5 an attack has little power; above eps = 2 an attack is nearly
decisive; between them, the deployment's other controls decide the outcome. This is why "eps = 1" is a
defensible default and "eps = 8 plus no monitoring" is not.

**The synthetic-data trilemma as a triangle.** Fidelity, coverage, and privacy cannot all be maximised by
one generator at fixed capacity. Each vertex is a failure mode you can name.

```text
            fidelity
               /\
              /  \
   memorisation  /    \  mode dropping
   (privacy lost)/      \ (coverage lost)
                /________\
          privacy        coverage
```

A generator that memorises sits at the fidelity-privacy edge. A generator that only fits marginals sits at
the fidelity-coverage edge with poor joint structure. The DP generator sits near the centre and pays in
fidelity, which is the price of a guarantee rather than a hope.

**The coverage heat map.** Coverage is a per-stratum quantity, so it belongs in a grid, not a scalar.

```text
                region 1  region 2 ... region 12
age 40-49  yes    0.94      0.91         0.88
age 40-49  no     0.97      0.96         0.95
age 80+    yes    0.31      0.28         0.35
age 80+    no     0.62      0.58         0.66
```

Reading the grid: the common strata are fine, the rare positive stratum in the oldest band is the failure.
A single mean coverage of 0.88 across this grid would be arithmetically true and operationally useless.

**The purpose-creep funnel.** Access queries enter wide and most should be refused; the metric that
matters is the refusal rate, and a refusal rate of zero is a red flag rather than a success.

```text
queries per quarter: 1,000
  declared purpose in register     940
  purpose in register but field out of scope  610  -> refused on field scope
  field in scope but granularity too fine      420  -> rerun at coarse granularity
  approved as requested                        210  -> 21% of requests, 79% adjusted or refused
```

**The retention timeline.** Windows attach to granularity, and the coarse archive outliving the row-level
copy is normal rather than suspicious.

```text
t=0      collect 42 fields
t=0      minimise to 9 fields (same day, before first write to the analytical store)
t=+1y    freeze the audited year, coarsen age to 10-yr in the archive copy
t=+2y    drop the row-level copy; keep aggregates of 168 cells
t=+7y    destroy aggregate table and per-subject keys
```

The 5-year gap between dropping rows and destroying aggregates is legitimate only if the aggregates are
at a granularity where the QI set no longer singles out records - which is a k claim that must be
re-measured after coarsening, not inherited.

## 7. Comparison / Selection Table

Release levels, compared on what a reviewer actually needs to know.

| Level | Guarantee statement | Utility ceiling | Effort | Typical failure |
|---|---|---|---|---|
| Raw rows + access control | policy only, no bound | full | low | insider read; no bound to cite |
| Pseudonymised rows | none; reversible by design | near-full | low | join restores identity |
| k-anonymised rows | 1/k per QI match | high on marginals | medium | homogeneous sensitive groups |
| l-diverse rows | limits group homogeneity | medium | medium | infeasible for binary outcomes |
| t-close rows | bounded distribution skew | medium | high | needs reliable global baseline |
| DP aggregates | eps per query family | counts and sums only | medium | small cells unusable at low eps |
| DP synthetic | eps on the release | task-measured, often lower | high | coverage loss in rare strata |
| Non-DP synthetic | none | high but unproven | high | memorisation, undetected |

Selection rules that follow from the table, in priority order:

1. If the purpose consumes only counts and sums, choose DP aggregates. It is the cheapest defensible
   release and it composes with a stated epsilon.
2. If the purpose needs row-level structure for model training and the strata are dense, choose DP
   synthetic and evaluate it on the rare strata, not on the whole population.
3. If the purpose needs exact small counts and the population is small, do not release; renegotiate the
   purpose. A suppression floor will not produce a bound you can defend.
4. Never choose non-DP synthetic as a way to avoid a privacy review. It converts a review into an
   unmeasured risk, which is worse for everyone including the requester.
5. Prefer k-anonymity only when the QI set is genuinely coarse and the adversary's external data is
   known to be limited. Document that assumption in the release note, because it is the load-bearing one.

Field-level techniques compared on what they change.

| Technique | Changes | Keeps | Destroys | Best used when |
|---|---|---|---|---|
| Drop field | removes the column | other columns | all utility of that column | no registered purpose uses it |
| Generalise | values to wider bins | every record | within-bin resolution | bins still support the task |
| Suppress row | removes records | exact values | counts in small cells | exact values matter more than counts |
| Microaggregate | values to group means | record count | plausible single values | means matter, ranges do not |
| Pseudonymise | identifier to token | join within dataset | cross-dataset join | linkage is the risk being controlled |
| Add noise | values perturbed | distribution shape | exact values | aggregates are the deliverable |
| Aggregate to cells | rows to counts | exact counts per cell | row-level structure | purpose is a proportion or total |

Two selection traps worth naming. First, microaggregation produces values that pass no range check - a
mean of HbA1c bands can land on a value no patient had - so a downstream schema validator will reject the
release. Second, pseudonymisation does not reduce the QI set at all; a tokenised record with age, sex, and
region is as linkable as an untokenised one to anyone holding the same three fields.

## 8. Failure Modes & Edge Cases

Each entry states the mechanism, the competing causes you must separate, and the check that discriminates
between them. The checks are chosen so that they can be run from the release artefacts, not from memory.

**F1 - k reported on the pre-generalisation data.** The audit computes k on the raw table, then the
release applies generalisation, and the reported k is the smaller of the two. Competing causes: (a) the
metric was computed before the edit; (b) the release was regenerated after a schema change; (c) the
population changed because late-arriving records were appended. Discriminating check: recompute k from
the released file itself, reading only its columns, and compare to the reported number. If they differ,
the report describes a different dataset than the release.

**F2 - QI set defined by the requester, not the adversary.** The QI set omits a field the requester will
also receive (for example year of birth in a linked table). Competing causes: (a) incomplete QI
enumeration across the release family; (b) a field added after the QI review; (c) a joinable public
dataset that supplies the missing field. Discriminating check: list every field in every release in the
family and run a closure test - take the union of fields that any two releases share and re-measure k on
the union. A drop in k identifies the missing attribute.

**F3 - homogeneity attack on a passing k.** Every record in a group shares the outcome, so k = 8 leaks
the outcome with certainty. Competing causes: (a) a genuinely correlated sensitive attribute; (b) a
generalisation that grouped by the outcome by accident; (c) a small population where the outcome is rare.
Discriminating check: compute the maximum single-value share inside each group. Any group above about
0.9 needs t-closeness or suppression; the k metric will not flag it.

**F4 - synthetic generator memorised the tail.** Nearest-neighbour distance from synthetic to real rows
is small for the rare stratum. Competing causes: (a) overfitting from too many epochs; (b) a stratum so
small that any faithful generator reproduces it; (c) a test set that overlaps the training set, which
inflates the apparent fidelity and hides memorisation. Discriminating check: for each synthetic record
compute the distance to the nearest real record and to the second nearest, and report the distribution of
the ratio. Ratios below about 0.5 for more than a few percent of rows indicate copying, not learning.
Also verify the train/test split from the data itself before interpreting any distance.

**F5 - membership advantage measured with a mismatched reference.** The attack compares model loss on
members to loss on a random sample of non-members drawn from a different period, so the "attack" detects
the period rather than membership. Competing causes: (a) temporal shift between member and non-member
samples; (b) a class imbalance difference; (c) genuine memorisation. Discriminating check: rerun the
attack with non-members drawn from the same period and same class mix. If the advantage collapses to
about zero, the original signal was shift.

**F6 - epsilon budget exhausted by unlogged exploratory queries.** Analysts run ad-hoc counts, each
logged as eps = 0.05, and the dashboard never sums them. Competing causes: (a) no shared accountant;
(b) separate teams with separate budgets over the same table; (c) a cache that serves results without
charging epsilon (a cache hit genuinely costs zero, so this one is legitimate if the cache is keyed on
the full query). Discriminating check: reconstruct the epsilon ledger from query logs and compare to the
declared budget. A cache is legitimate only if a hit returns a bitwise-identical prior answer; a
recomputation at different noise is a new release.

**F7 - deletion verified on the primary store only.** The row is gone from the OLTP database, present in
the analytical warehouse, the object store, and the nightly backup. Competing causes: (a) replication lag
in the pipeline; (b) an immutable backup tier by design; (c) a tombstone that carries the deleted values.
Discriminating check: attempt a restore into an isolated environment and query for the subject's
identifier. If the restore contains the value, the deletion SLA is unmet regardless of the primary
store's state.

**F8 - purpose register that no query can reference.** Queries carry free-text purpose and the register
keys on identifiers, so the join is manual and mostly skipped. Competing causes: (a) client tooling that
does not require the identifier; (b) a register that was never mapped to the tool's enum; (c) analysts
choosing the closest-looking purpose. Discriminating check: sample 50 recent queries and attempt to map
each to a register entry using only the logged fields. A mapping rate below about 0.9 means purpose
limitation is not enforceable, whatever the policy says.

**F9 - coverage measured on aggregated bands.** Coverage is reported per age band, but the task needs the
band-by-region cell. Competing causes: (a) reporting level chosen for readability; (b) too few synthetic
records per cell to report without embarrassment; (c) the generator's own evaluation used bands. Check:
recompute coverage at the cell level the model consumes. Aggregated coverage is always at least as good
as cell coverage, and the gap is the size of the problem.

**F10 - retention window applied at table level.** A single 7-year window covers a table whose fields
have different purposes, so a field with a 30-day purpose lives 7 years. Competing causes: (a) schema
without field-level retention metadata; (b) a table created before field-level policy existed; (c) a
downstream join that needs the long window for one field and inherits it for all. Check: list fields with
min(purpose window) below the table window; each is a finding. Count them; a table with more than a few
is a policy-application failure, not a design choice.

**F11 - utility claim from a single split.** TSTR is reported on one train/test split and the gap looks
small; on a second split the gap doubles. Competing causes: (a) high variance because the synthetic set
is small; (b) a test set that shares records with training; (c) genuine instability of the generator.
Check: repeat TSTR over at least five splits and report the interval, not the point. If the interval
crosses the decision threshold, the utility claim is not established.

**F12 - the "synthetic so it is not personal data" argument.** A release team classifies synthetic rows
as out of scope, then stops measuring. Competing causes: (a) a generator with no privacy mechanism;
(b) a DP generator at a large epsilon; (c) a synthetic set built by sampling with replacement from real
rows and perturbing one field. Check: run a membership attack and a nearest-neighbour test before
classifying. Both are cheap; the classification without them is an assertion, not a finding.

**F13 - composition across a schema migration.** The same records exist in the old and new table, and
each table's releases are accounted separately. Competing causes: (a) migration kept both copies live;
(b) the accountant keys on table name; (c) a backfill duplicated rows. Check: compute the overlap between
the two tables on a stable key and, if non-zero, sum the epsilons rather than taking the max.

**F14 - the aggregation floor applied before the join.** Cells are floored at 5, then joined to a
geography table that splits each cell into smaller units by population share, producing fractional counts
that sum to the floored value. Competing causes: (a) a join performed by the requester; (b) a
disaggregation step in the delivery pipeline; (c) a published lookup table. Check: verify that no
published cell can be split by a public key into parts smaller than the floor. This is a post-release
check, because the requester's joins are outside your pipeline.

## 9. Complexity & Cost Analysis

**Computational complexity of the release pipeline.** Each technique has a different cost driver, and the
driver decides which one you can afford at 50,000 records versus 50 million.

```text
technique              complexity                        driver
full-domain generalisation   O(n * d) per level            n records, d attributes
Mondrian partition           O(n log n) expected           sort and split
optimal k-anonymisation      NP-hard (exact)               never run exactly at scale
suppression search           O(2^d) exact, greedy O(n d)   number of attributes
DP count release             O(cells)                      number of published cells
DP-SGD training              O(epochs * n * params)        model size dominates
non-DP GAN training          O(epochs * n * params)        same, plus discriminator
nearest-neighbour audit      O(m log n) with an index      m synthetic rows
membership attack            O(shadow models * training)   number of shadow models
```

The practical reading: generalisation and DP aggregates are cheap enough to run per request. DP-SGD and
generative training are batch jobs measured in hours, so they belong to a scheduled release, not an
ad-hoc one. The audit costs (nearest neighbour, membership attack) are often skipped because they are
priced like training jobs when they are actually cheap with an index.

**Cost model for generation.** Compute is proportional to steps, and steps are proportional to epochs and
records per batch.

```text
steps = epochs x ceil(N / batch)
DP-SGD example: 30 x ceil(50,000/256) = 30 x 196 = 5,880 steps
row-gradients processed = 5,880 x 256 = 1,505,280  (about 30 passes over 50k rows)
ILLUSTRATIVE cost: 46 GPU-seconds per 1,000 steps -> 5,880 x 46/1,000 = 270 s = 4.5 min
```

Scaling to 5 million records at the same batch size multiplies steps by 100: 588,000 steps, about 7.5
GPU-hours at the same per-step rate. That is the point at which the release becomes a scheduled job with
a queue, and where the epsilon accountant must be re-run because the number of steps is a term in it.

**Cost of the privacy budget as a resource.** Epsilon is the scarcest input in the pipeline, and treating
it as free is the costliest mistake in this section.

```text
budget: eps_total = 4.0 per dataset per year (ILLUSTRATIVE policy)
  one DP synthetic release at 1.7  -> 2.3 remaining
  one DP aggregate family at 1.2   -> 1.1 remaining
  one exploratory query batch at 0.05 x 12 = 0.6 -> 0.5 remaining
  a second synthetic release at 1.7 is refused (would total 4.6)
```

The second synthetic release is refused not because it is unsafe in isolation but because the budget is a
sum. A team that plans a mid-year refresh must reserve its epsilon at the start of the year, exactly as
it would reserve a storage quota.

**Cost of review effort.** The dominant cost in a real minimisation exercise is human attention per
field, and it scales with the number of purposes, not the number of records.

```text
42 requested fields x 15 min per field review = 630 min = 10.5 h
  of which: 20 purpose refusals at 3 min each (clear-cut) = 60 min
            22 necessity reviews at 20 min each           = 440 min
            9 granularity decisions at 15 min each        = 135 min
  total = 635 min (within a few minutes of the first estimate; the split changed, the total did not)
```

Two lessons from the arithmetic. Clear-cut refusals are cheap and should be batched first, because they
remove a third of the work in an hour. Granularity decisions are expensive and should be made once per
attribute, then reused across requests, which is why a generalisation hierarchy library pays for itself.

**Cost of a false release.** The asymmetry between review cost (10 hours) and incident cost is the
argument for the whole discipline. Using a constructed exposure price of 180 USD per record:

```text
row-level release of 50,000 records: 50,000 x 180 = 9,000,000 USD exposure
aggregate release of 336 cells:         336 x 180 =    60,480 USD exposure
ratio: 9,000,000 / 60,480 = 148.8
```

The 149x reduction comes from changing the release unit, not from dropping fields. This is the number to
put in front of a sponsor who asks why the review takes ten hours.

## 10. Key Theorems / Results Reference

The results below are named because a reviewer will be asked to justify a claim by reference. Each
statement includes the condition under which it holds and the practical consequence of misapplying it.

**T1 - Post-processing immunity.** If M is (eps, delta)-DP, then for any randomised function f, f(M(x)) is
(eps, delta)-DP. Consequence: a DP aggregate can be safely rounded, binned, or plotted. Consequence that
is often missed: the immunity runs one way only. A non-DP output cannot be made private by post-processing,
so a synthetic release built without a mechanism cannot be repaired downstream.

**T2 - Basic composition.** The composition of k mechanisms each (eps_j, delta_j)-DP is
(sum eps_j, sum delta_j)-DP. Consequence: every release must be logged. Consequence: it is conservative,
which is why advanced composition exists.

**T3 - Advanced composition (Dwork-Rothblum-Vadhan).** For k mechanisms each (eps, 0)-DP, the composition
is (eps sqrt(2k ln(1/delta)) + k eps (e^eps - 1), delta)-DP. Consequence: the bound improves on basic
composition only for small eps; Block 4.2.1 shows it is *worse* at eps = 0.1 with k = 12, and it can never
help when eps >= ln 2.

**T4 - Group privacy.** If M is (eps, 0)-DP for single-record changes, then for groups of size g it is
(g eps, 0)-DP. Consequence: a release that protects individuals at eps = 0.3 protects a family of four at
1.2. Household-level data therefore needs the group size stated in the accounting, not discovered later.

**T5 - Impossibility of unbounded query release (Dinur-Nissim).** For a database of n records, an
adversary allowed to ask a sufficiently large number of statistical queries and to see answers with error
below about n/sqrt(queries) can reconstruct a constant fraction of the database. Consequence: "we only
publish aggregates" is not a privacy claim unless the number of published aggregates is bounded or noise
is calibrated. A public dashboard that answers arbitrary count queries is an attack surface.

**T6 - k-anonymity provides no formal guarantee.** There is no (eps, delta) statement implied by
k-anonymity, and the guarantee degrades to nothing under background knowledge beyond the QI set. The
homogeneity example in Block 4.3.2 is the standard counterexample: a group of 8 records with 6 positives
satisfies k = 8 and l = 2 while leaking the outcome at 75% confidence.

**T7 - l-diversity is bounded by attribute cardinality.** Entropy l-diversity requires log2(l) <=
log2(cardinality), so l cannot exceed the number of distinct sensitive values. Consequence: for a binary
outcome the maximum is l = 2, and any policy asking for more is unsatisfiable. This is arithmetic, not a
matter of implementation effort.

**T8 - Membership advantage bound under pure DP.** For (eps, 0)-DP, the advantage of any membership test
is at most tanh(eps/2). Consequence: the bound is a certificate you can quote to a requester, and its
inverse is a lower bound you can quote after an attack. The inverse is the more useful direction in an
audit, because the attack is measured and the accountant is asserted.

**T9 - Gaussian mechanism calibration.** For a query of L2 sensitivity D, adding Gaussian noise with
sigma = D sqrt(2 ln(1.25/delta)) / eps gives (eps, delta)-DP. Consequence: the calibration is a formula,
not a tuning exercise, and the delta term is logarithmic, so tightening delta is cheap while tightening
eps is expensive.

**T10 - Amplification by subsampling.** Running a mechanism on a random q-fraction of records and
accounting for the sampling yields a smaller effective epsilon for the same noise. Consequence: this is
what makes DP-SGD practical; without it, the noise needed for eps = 1.7 over 5,880 steps would destroy
the model. Consequence for audits: the amplification claim is only valid if the sampling is actually
uniform and independent, so a sampler that drops records it dislikes voids the accounting.

**T11 - Synthetic data has no free lunch.** For a generator with bounded capacity, improved marginal
fidelity does not imply improved joint fidelity, and joint fidelity at fixed capacity trades against
coverage of rare modes. There is no single scalar that ranks releases, which is why section 7 compares
them on three axes rather than one.

**T12 - Deterministic pseudonymisation preserves linkability.** Any function that maps the same input to
the same output across datasets supports a join. Consequence: a token is only as private as the set of
datasets that share the tokenisation key, and that set is usually wider than the release team believes.

## 11. Interview / Exam Questions

14 questions, ordered from recall through calculation to judgement. At least half require a computation or
a diagnosis. Each answer is stated independently; none is a template.

**Q1. State the difference between the k-anonymity bound and an expected count of re-identifications.**
Answer: the k bound is a per-record probability, 1/k, conditional on the adversary knowing the QI values;
an expected count multiplies a probability by the number of records and implicitly assumes the adversary
attacks every record. Reasoning: the two quantities have different units, probability versus records, and
only the first is implied by the definition. Follow-up: with k = 6 and 49,988 records, why is 8,331 not
the expected number of re-identifications? Because the adversary's attempt budget is set by their external
data, and the bound is conditional on a QI match they may not have.

**Q2. Compute the CDM for 336 cells of mean size 148.81 and for 168 cells of mean size 297.62.**
Answer: 7.44e6 and 1.49e7 records squared. Reasoning: CDM is the sum of squared group sizes, and halving
the cell count doubles the mean size, which quadruples each term and halves the number of terms, so the
total doubles. Check: 336 x 22,144.4 = 7,440,518 and 168 x 88,577.7 = 14,881,053; the ratio is 2.000.
Interpretation: over-generalising from level B to level C doubles cost for no policy benefit.

**Q3. A policy asks for entropy l-diversity with l = 3 on a recorded-complication flag. What is wrong?**
Answer: the requirement is infeasible. Reasoning: a binary attribute has maximum entropy log2(2) = 1.000
bit, and l = 3 needs log2(3) = 1.585 bits, so no group can satisfy it. Discriminating check: compute the
attribute's cardinality when the policy is written; if log2(l) exceeds log2(cardinality), rewrite the
policy. Red flag: an implementation that reports "l = 3 satisfied" for a binary field.

**Q4. Your accountant says eps = 0.1 for each of 12 queries. Which composition bound do you report?**
Answer: the basic sum, 1.2. Reasoning: advanced composition at eps = 0.1 and k = 12 gives
0.1 x (18.209 + 1.262) = 1.947, which is larger than 1.2, and the crossover for this eps is about k = 35.
Reporting the smaller-looking rule without computing both is the error. Follow-up: at eps = 0.05 and 100
queries the advanced bound is 2.885 versus 5.00, so the choice depends on the numbers, not on habit.

**Q5. A model's membership attack reaches advantage 0.36. What does that imply about epsilon?**
Answer: eps >= ln(1.36/0.64) = 0.754, so the release is at least roughly eps = 0.75 equivalent regardless
of what the accountant said. Reasoning: the forward bound advantage <= tanh(eps/2) inverts to a lower
bound on eps. Discriminating check: rerun the attack with a same-period non-member sample to exclude
temporal shift before quoting the number.

**Q6. Noise with sensitivity 1, eps = 1, delta = 1e-5. Give sigma and the relative error on a cell of 154.**
Answer: sigma = 1 x sqrt(2 ln(1.25e5))/1 = sqrt(23.472) = 4.845; sd of the noisy count is 4.845 x 1.4142
= 6.852; relative error is 6.852/154 = 4.45%. Reasoning: the Gaussian calibration plus the per-coordinate
variance. Follow-up: the same cell at eps = 0.1 gives sigma = 48.45 and 44.5% relative error, which is why
small epsilon and small cells cannot both be satisfied.

**Q7. Diagnose: k = 8 is reported, yet a group of 8 records has 6 positives. Which metric catches this?**
Answer: t-closeness. Reasoning: k bounds group size, l-diversity only needs two distinct values, and the
group's positive share of 0.75 versus a global 0.30 gives t = 0.45, which fails a 0.20 threshold.
Follow-up: what if the global rate were 0.75? Then t = 0 and the group is unremarkable, because the group
reveals nothing beyond the base rate.

**Q8. A secure-aggregation deployment uses n = 100, t = 70. Dropout rises from 0.15 to 0.30. Quantify.**
Answer: failure probability rises from about 7e-6 to about 0.457. Reasoning: survivors are Binomial(100,
0.85) with mean 85 and sd 3.571, giving z = -4.34; at dropout 0.30 the mean is exactly 70 and sd is 4.583,
so the threshold sits at the mean. Correction: either lower t to about 60 or add client retries; a
threshold at the mean of the survival distribution is a coin flip by construction.

**Q9. Why is it wrong to call a pseudonymised table anonymised?**
Answer: pseudonymisation is reversible by design, so the token is a direct identifier for retention
purposes, and a deterministic token also preserves joins across datasets. Reasoning: linkability is a
property of the mapping, not of the field name. Discriminating check: ask whether any two tables share
the tokenisation key; if yes, the releases are joinable and compose.

**Q10. Synthetic release: marginal TV 0.031, TSTR retention 0.944, coverage 0.296 on the smallest band.
Approve it for a rare-event audit?**
Answer: no. Reasoning: retention measured on the whole population hides the stratum the audit consumes,
and coverage of 0.296 means the generator under-represents the rare positive cases by a factor of about
3.4. Correction: either raise coverage by stratified generation, or restrict the release's approved
purpose to bands with coverage above 0.8 and record that restriction.

**Q11. A dashboard lets analysts run arbitrary count queries on an aggregated table. Name the theorem
that makes this unsafe and the fix.**
Answer: the Dinur-Nissim reconstruction result: enough statistical queries with small error allow
reconstruction of a constant fraction of the database, so unlimited counting is not protection. Fix:
bound the query interface and calibrate noise per query with a shared accountant, or restrict to a fixed
published set of cells.

**Q12. Field minimisation cuts 42 fields to 9. What does that do to breach exposure, quantitatively?**
Answer: nothing to the record count: 50,000 x 180 = 9,000,000 USD either way under a per-record price.
Reasoning: exposure is priced per record, and dropping fields reduces severity per record rather than the
number of records. Correction: to reduce exposure, reduce the release unit - 336 aggregates gives
336 x 180 = 60,480, a 148.8x reduction.

**Q13. Deletion is confirmed on the primary database. What single test must still pass?**
Answer: a restore-and-query test on a backup replica in an isolated environment. Reasoning: replication
lag, immutable backup tiers, and tombstones that carry the deleted values all leave the value recoverable
after a successful primary-store delete. Follow-up: what makes the tombstone safe? It must contain no
deleted value, only a key and a timestamp.

**Q14. Your generator is DP-SGD at eps = 1.7, and a colleague proposes filtering synthetic rows that are
too close to real rows before release. Is that legitimate?**
Answer: the filtering is legitimate, but it must be described correctly: it is post-processing of a DP
output, so the DP guarantee survives, yet the filtering itself uses real data and can bias the release.
Reasoning: post-processing immunity protects the privacy claim at the same epsilon, but it does not
protect the utility claim, and a nearest-neighbour filter systematically removes the rare modes that
coverage already showed to be weak. Correction: measure coverage before and after the filter and report
both; if coverage drops below the purpose threshold, the filter has made the release useless in exchange
for a guarantee it already had.

## 12. Quick Reference Card

| Item | Value or rule | Why it matters |
|---|---|---|
| Minimisation ratio | retained / requested fields | completeness of review, not quality |
| Cheapest passing generalisation | first lattice node with k >= policy | over-generalising doubles CDM per level |
| k bound | 1/k per record, given QI knowledge | never convert to an expected count |
| Binary l-diversity ceiling | l <= 2 | log2(l) must not exceed log2(cardinality) |
| t-closeness for binary | t = abs(p_group - p_global) | catches homogeneous groups k misses |
| Composition, eps = 0.1 | use the basic sum | advanced bound is worse below k about 35 |
| Composition, adv. threshold | eps < ln 2 = 0.693 | above it advanced composition cannot help |
| Gaussian sigma | D sqrt(2 ln(1.25/delta)) / eps | tightening eps costs 10x noise, delta costs 3% |
| Membership advantage | at most tanh(eps/2); inverse gives eps lower bound | attack result beats accountant claim |
| Synthetic adequacy | TSTR on the target stratum, plus coverage | whole-population retention hides rare modes |
| Exposure | price x release unit count | aggregate release cuts it about 149x here |
| Deletion test | restore a backup and query | primary-store success is not deletion |
