---
title: "Customer Identity Controls"
topic: "Identity resolution, consent-scoped eligibility, and reversible activation for omnichannel retail customer data"
collection: "Omnichannel Customer, Loyalty and Fulfilment — Cheat Sheets"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Customer Identity Controls

All arithmetic in this card is derived and re-checked here; every input population, count, and cost is a
constructed ILLUSTRATIVE value, not a measured retailer result. Named methods (Fellegi–Sunter weighting,
pairwise confusion matrices, cost-weighted thresholds) are published and citable; the numbers fed into them
are not. Nothing in this card is legal advice, and nothing here asserts that a linking technique is lawful
in a given jurisdiction.

## 1. File Header

**Purpose.** This card turns "one customer, one profile" from a slogan into a set of controls: a
named identity decision, a scored candidate set, a threshold justified by cost rather than habit, a
consent gate that filters activation, and a measurement definition that survives a backfill. It exists
because the failure mode in omnichannel retail is rarely "the model is inaccurate". It is almost always
"the number that leadership reads changed meaning when the identity layer changed underneath it".

**The decision this card supports.** For each candidate record pair, choose one of three actions —
auto-link, send to human review, or leave unlinked — and then decide, per linked entity, whether an
outbound or analytic use is permitted. Both decisions are per-unit, reversible, and logged.

**Audience and prerequisites.** You should already be comfortable with ratios, confusion matrices, and
the difference between an event timestamp and a processing timestamp. If logarithms frighten you, work
through §4.2's base case before reading the variants; every weight in this card is a log of a ratio of
two conditional probabilities, and nothing more exotic than that appears.

**Non-goals.** No vendor comparison of identity-resolution platforms by name. No claim about any real
retailer's match rate. No legal determination about consent, retention, or cross-border transfer. No
treatment of fraud identity, bot detection, or account-takeover forensics, which use the same machinery
with an adversarial prior.

**Versioning and evidence status.** Version 1.0, not independently reviewed (see §10's maintenance
note). The derivations are marked DERIVED; the populations that illustrate them are ILLUSTRATIVE. Where
a published result is cited, it is named in place so a reader can check it rather than trust it.

**Notation conventions used throughout.**

- Display mathematics appears only inside fenced `text` blocks; inline prose spells symbols in words.
- All logarithms written `log2` are base 2. A weight of 3.32 means the evidence multiplies odds by ten.
- Currency is written `CU` (constructed unit) so no reader mistakes a constructed number for a real price.
- Times are written as `event time` (when the act happened) or `processing time` (when the row landed).
- Fractions may be written as fractions, decimals, or percentages; each instance states which.

**Symbol register.** Defined once here, used without redefinition later. Units matter: a rate and a
count in the same column is the most common error in identity dashboards.

| Symbol | Meaning | Units | Valid range |
|---|---|---|---|
| `N` | number of source records entering a linkage run | records | integer ≥ 1 |
| `E` | number of true entities (distinct people) behind those records | entities | 1 ≤ E ≤ N |
| `P` | number of candidate pairs produced by blocking | pairs | ≥ 0 |
| `P_all` | number of pairs if every record were compared with every other | pairs | N(N−1)/2 |
| `A` | number of pairs auto-linked | pairs | 0 ≤ A ≤ P |
| `R` | number of pairs routed to human review | pairs | 0 ≤ R ≤ P |
| `T_hat` | number of pairs accepted as duplicates after all stages | pairs | 0 ≤ T_hat ≤ P |
| `T` | number of true duplicate pairs in the evaluation block | pairs | 0 ≤ T |
| `m_i` | probability field `i` agrees, given the pair is a true duplicate | fraction | (0, 1] |
| `u_i` | probability field `i` agrees, given the pair is not a duplicate | fraction | (0, 1) |
| `w_i` | field weight `log2(m_i / u_i)` | bits | real |
| `W` | summed evidence weight over agreeing fields | bits | real |
| `lambda0` | log2 prior odds of duplication inside the candidate set | bits | real |
| `C_FM` | cost of a false merge (two people treated as one) | CU per pair | ≥ 0 |
| `C_FN` | cost of a missed merge (one person split in two) | CU per pair | ≥ 0 |
| `tau` | consent-scoped contactable fraction | fraction | [0, 1] |
| `rho` | blocking recall: share of true pairs surviving candidate generation | fraction | [0, 1] |
| `c` | cross-arm duplication share in an experiment | fraction | [0, 1] |
| `k` | cluster size (records in one resolved entity) | records | ≥ 1 |
| `s_n` | share of a numerator removed by identity repair | fraction | [0, 1] |
| `s_d` | share of a denominator removed by identity repair | fraction | [0, 1] |
| `b` | mean records per blocking bucket | records | ≥ 1 |

**Reading order for a live problem.** If a retention number moved after a pipeline release, jump to §8's
entry "the metric moved and nobody changed the metric definition" before reading formulas. If you are
choosing a threshold for the first time, read §4.2 then §9's break-even argument in that order. If you
are being asked to "just merge the duplicates", read §4.3(c) first: repair can lower a rate.

**Retrieval cues.**

| Situation you are in | Start at |
|---|---|
| Leadership asks why "customers" fell from 2.1M to 1.9M overnight | §8, retroactive-merge entry |
| Match rate is high but outbound complaints are rising | §7 selection rules on household versus person grain |
| Score distribution looks fine but precision on audit is poor | §4.4 independence counterexample |
| Reviewer queue is 20k pairs and nobody can afford it | §9 review economics |
| An A/B test result will not reproduce | §4.3(d) contamination attenuation |
| Legal asks what "contactable" means in a report | §4.3(f) consent-scoped denominator |

**What this card deliberately does not decide.** It does not choose a survivorship rule (which record's
address wins in a merged profile). Survivorship is a business rule about freshness and source authority;
it is a column-level policy that sits downstream of the link decision, and conflating the two is how
teams end up unable to reverse a merge. This card keeps the link ledger and the golden record separate.

## 2. Concept Map / Mental Model

The mental model that makes identity work tractable is a **staged pipeline with an auditable ledger**,
not a profile table that gets overwritten. Seven stages, each with its own failure signature.

```text
stage 1  capture          raw rows + source + ingest time + consent state as received
stage 2  normalisation    casefold, trim, canonical phone, address parse, email local-part rules
stage 3  candidate gen    blocking keys -> candidate pairs P (deliberately lossy, cheap, high recall)
stage 4  scoring          per-field agreement -> weights W -> score
stage 5  decision         auto-link / review / no-link, by cost-implied threshold
stage 6  closure          accepted links -> connected components (an equivalence closure)
stage 7  activation gate  consent purpose x channel x suppression x recency -> permission to use
```

**The single most useful reframing.** Identity is not a column you assign; it is a *time-indexed
equivalence relation you assert*, and every assertion is reversible. A profile table answers "who is
this row about?" with one value and no history. A link ledger answers with "at processing time `t`, we
asserted that record 8 and record 41 denote the same person, under rule version 3, with score 11.4,
reviewed by nobody". Only the second answer can be audited, unwound, or replayed.

**Why the pipeline order is not negotiable.** Normalisation before blocking is not stylistic. If one
source stores `+44 7700 900123` and another stores `07700900123`, a raw-string blocking key splits the
same phone into two buckets and no downstream scoring can recover the pair. Every stage can only lose
information that a later stage cannot reconstruct, which is why the recall budget is spent early and the
precision budget late.

### 2.1 Invariants, stage by stage

**Stage 1 invariant — provenance is never dropped.** Every record carries a source identifier and an
ingest timestamp. Violation signature: two analysts compute different match rates from the same table
because one filtered on `source IS NOT NULL` and the other did not. Check: `count(*)` grouped by source
must sum to the table count.

**Stage 2 invariant — normalisation is idempotent.** Applying the normaliser twice yields the same
string as applying it once. Violation signature: daily re-runs slowly drift as already-normalised values
are re-normalised (`"O'Brien"` → `"o brien"` → `"o  brien"`). Check: assert `norm(norm(x)) == norm(x)`
over a 10k sample before every release.

**Stage 3 invariant — candidate generation is recall-first and cheap.** Blocking is allowed to miss true
pairs; it is not allowed to be expensive. Violation signature: an engineer "improves quality" by making
the blocking key stricter, recall falls silently, and every downstream metric looks clean because the
missed pairs never enter the denominator of the audit.

**Stage 4 invariant — field agreement is computed on the normalised value, not the raw value.** Violation
signature: surname agreement of 0.42 on a dataset where 90% of surnames are already case-consistent.

**Stage 5 invariant — the threshold is a function of costs and calibration, and both are recorded.**
Violation signature: the threshold is a magic `0.85` in three repositories, with three different score
definitions behind it.

**Stage 6 invariant — closure is computed, never assumed.** Link acceptance is pairwise; entity formation
is transitive. Violation signature: two records that a human would call different people land in one
entity because a third record bridges them. Check: for every component of size ≥ 3, recompute the
minimum pairwise score inside the component; a bridging pair near the threshold is the tell.

**Stage 7 invariant — permission is evaluated at use time, not at link time.** A person who withdraws
consent on Tuesday must not receive Wednesday's campaign, even though the link made on Monday is still
valid. Violation signature: a campaign file built Monday and sent Wednesday to an audience that includes
Tuesday's withdrawals.

### 2.2 Two competing architectures, and why this card picks one

| Architecture | What it stores | Strength | Failure it invites |
|---|---|---|---|
| Golden record (overwrite) | one mutable row per entity, best value per column | simple to query, one join | irreversible merges, no audit trail, no replay |
| Link ledger (append-only) | assertions `(record_a, record_b, decision, score, rule_version, decided_at)` | reversible, auditable, replayable at any cut-off | needs a closure step at read time, larger storage |
| Hybrid: ledger + materialised view | ledger as source of truth, profile table rebuilt per run | both properties, at the cost of a rebuild job | view staleness between rebuilds |

The failure that forces the choice is almost always an audit or a complaint: "you mailed this household
twice, show me why". A golden-record architecture cannot answer, because the evidence was overwritten.
A ledger can, because the assertion that caused the duplicate is still a row.

**The time model.** Two clocks matter. `effective time` is when a fact was true in the world (the person
changed address in March). `processing time` is when the system learned it (the March address arrived in
a June batch). A merge decided in June must not be treated as if it had been known in March when you
recompute March's retention, or you will produce a historical number nobody could have acted on.

```text
bitemporal profile read:

  as-of processing time t_p  ->  what the system believed at t_p
  as-of effective time  t_e  ->  what was true in the world at t_e
  both                       ->  the only read that makes a backtest honest
```

**Consequence for measurement.** Any metric that depends on entity count must declare which clock it
uses. "Active customers in March, as believed in June" and "active customers in March, as believed in
March" are different numbers, both correct, and a report that mixes them across periods produces a trend
that is an artifact of release dates.

## 3. Foundational Definitions

Definitions are grouped by the stage they govern. Each row states what the term is, what it is measured
in, and the specific misuse that follows from getting it wrong.

### 3.1 Records, entities, and keys

| Term | Definition | Units / type | Characteristic misuse |
|---|---|---|---|
| Source record | one row as received from one system, with provenance | row | treated as a person by a naive `count(*)` |
| Entity | a set of records asserted to denote one person | set of records | reported as "customer" without saying person or household |
| Deterministic key | a key that declares equality by exact match after normalisation | string | assumed collision-free because it looks unique |
| Probabilistic key | a score comparing several fields, converted to a decision by threshold | real score | compared across vendors whose scores are not commensurable |
| Blocking key | a coarse attribute used to group records into comparable buckets | string or tuple | tightened to reduce cost, silently cutting recall |
| Candidate pair | a record pair inside one block that is scored | pair | counted as a duplicate before it is decided |
| Alias | an alternative identifier known to belong to the same entity | string | conflated with a *possible* alias, which is a hypothesis |
| Pseudonymous identifier | a stable token replacing a direct identifier | token | treated as anonymous when it is merely pseudonymous |
| Household | a group of people at one address or one payment instrument | set of entities | collapsed into a person, breaking per-person metrics |
| Golden record | the materialised best-value view of an entity | row | mistaken for the source of truth about identity |

### 3.2 Matching vocabulary

| Term | Definition | Units / type | Characteristic misuse |
|---|---|---|---|
| Agreement | the event that field `i` matches between two records | boolean | averaged across fields without field-specific base rates |
| `m` probability | P(agree on field `i` given a true duplicate) | fraction | estimated on a sample enriched with duplicates, then applied to a base population |
| `u` probability | P(agree on field `i` given a non-duplicate) | fraction | estimated from a random sample that contains duplicates, biasing it upward |
| Weight | `log2(m/u)` for one agreeing field | bits | summed across correlated fields as if independent |
| Prior odds | duplication odds before field evidence, inside the candidate set | ratio | taken from the full population while the score is applied inside blocks |
| Posterior odds | prior odds multiplied by the likelihood ratio from evidence | ratio | reported as a probability because the reader expects one |
| Calibration | agreement between stated probability and observed frequency | fraction or bits of gap | replaced by a single aggregate precision figure |
| Threshold | score above which a pair is auto-linked | bits | chosen by maximising F1 when costs are asymmetric |
| Review band | score interval routed to humans rather than decided | bits | omitted, forcing every uncertain pair into accept or reject |

### 3.3 Error vocabulary and its costs

| Term | Definition | Units / type | Characteristic misuse |
|---|---|---|---|
| False merge | two distinct people linked into one entity | pair count | counted once, although it corrupts every metric touching either person |
| False split | one person left as two or more entities | pair count | invisible in most dashboards because it adds rows rather than removing them |
| Pair recall | accepted true pairs divided by true pairs | fraction | reported as entity recall, which is smaller once splits are counted |
| Pair precision | accepted true pairs divided by accepted pairs | fraction | reported as "accuracy", hiding the base rate |
| Over-merge cascade | one false link joining two large clusters, corrupting all cross pairs | pair count | underestimated by pair-level precision |
| Link amplification | spurious cross pairs created by one false bridge between clusters of sizes `a` and `b` | pairs | assumed equal to 1 |
| Contamination | the share of units appearing in more than one analysis arm | fraction | ignored when reading experiment results |

### 3.4 Permission vocabulary

| Term | Definition | Units / type | Characteristic misuse |
|---|---|---|---|
| Consent state | the permission a person has given, for a purpose, at a time | categorical + timestamp | stored as a boolean that loses purpose and time |
| Purpose limitation | the rule that permission for one use does not imply another | policy constraint | collapsed so that a transactional opt-in authorises marketing |
| Contactable | consent-valid, channel-verified, and not suppressed, at send time | boolean per entity-channel | equated with "has an email address" |
| Suppression | a standing instruction not to contact, independent of consent | categorical | merged into consent field so a withdrawal cannot be distinguished from a bounce |
| Withdrawal propagation | the process that removes a person from built-but-unsent audiences | job with a lag bound | assumed instantaneous because the API call succeeded |
| Reachable denominator | the population that could lawfully and technically be contacted | entities | replaced by total profiles, inflating every rate |
