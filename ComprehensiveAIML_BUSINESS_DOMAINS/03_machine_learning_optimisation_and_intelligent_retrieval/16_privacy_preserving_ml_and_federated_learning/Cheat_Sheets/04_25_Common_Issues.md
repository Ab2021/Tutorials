---
title: "Common issues: privacy-preserving ML and federated learning"
topic: "Diagnostic catalogue: differential-privacy accounting, federated aggregation, secure aggregation, de-identification"
collection: "Part 03 - Machine Learning, Optimisation and Intelligent Retrieval"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Common issues: privacy-preserving ML

> **Evidence boundary.** Every count, byte figure, probability and epsilon in this file is DERIVED
> arithmetic over one constructed reference system, and every constructed input is labelled
> ILLUSTRATIVE. No vendor benchmark, incident, audited deployment or organisation result is claimed.
> Where a definition comes from a paper, the paper is named in Section 10 as a pointer to the
> definition, not as evidence about your workload. The accounting numbers below follow a published
> method, but they are worked on ILLUSTRATIVE inputs and are not a certified budget for any real run.

## 1. File Header

**Purpose.** Turn the failure symptoms that actually appear in privacy-preserving ML into 30 stable,
individually diagnosable records, each carrying a check that separates two plausible causes before
anyone retunes a noise multiplier, re-samples a cohort, or rewrites a consent record.

**Why a catalogue and not a tutorial.** Privacy failures are almost never announced. They arrive as
"the accuracy is 3 points below the central baseline", "the accountant says 3.05 but legal signed for
1.0", "the release is k-anonymous so it is fine", "the round takes 9 minutes because one phone is
slow". Each of those has two or three causes with similar base rates, and the causes need different
fixes. The catalogue's job is to name the discriminating check.

**Prerequisites.** You must be able to read a per-example gradient budget, distinguish a probability
per record from an expected count over a cohort, follow a logarithm through a sensitivity
calculation, and state the difference between a mechanism's guarantee and a measurement of an
attacker's success. Section 3 defines each term in the order it is used; nothing later assumes a term
before that point.

**Reference system used by every quantitative record.** To keep the numbers comparable, all examples
describe one fictional system. It is a tabular classifier with three weight matrices and no bias
terms: 1,200x1,200 plus 1,200x300 plus 300x40. The parameter count is therefore

```text
1,200 * 1,200 = 1,440,000
1,200 *   300 =   360,000
  300 *    40 =    12,000
P             = 1,812,000 parameters
```

P is a count of scalar weights. It is the only structural fact about the model that any derivation
below uses, which is deliberate: privacy accounting depends on the sensitivity of the update, not on
the layer shapes. A client uploads one dense update of P numbers per round.

**Reference federation.** 60,000 enrolled clients, 1,200 sampled per round without replacement
(sampling rate q = 1,200/60,000 = 0.02), 500 rounds, so each participating client contributes about
500 x 0.02 = 10 times. Local work per round: 250 minibatch steps at minibatch 32. Upload dtype fp32
unless stated. Trust model: honest-but-curious server, no collusion across rounds assumed only where
stated.

**Reference privacy target.** epsilon = 1.0, delta = 1e-5, at record level, for the whole 500-round
run, with a noise multiplier sigma to be chosen. Section 4 derives why that target is not reachable
at the compute budget the run was given, and Section 5 computes what the run actually spends.

**Units used throughout.** P is a scalar count. Bytes are decimal (1 MB = 1e6 B), memory figures are
GiB where named. epsilon and delta are dimensionless; delta is a probability per run. sigma is a
noise multiplier in units of the clipping norm C, so the noise standard deviation is sigma·C. q is a
dimensionless sampling rate. Round time is seconds of wall clock. FLOPs are counted, not measured.

**Outcome if the sheet is applied.** Given a symptom, you can name the two or three causes with the
highest base rate, run one check that separates them, and only then apply a correction that has a
stated bound. You can also reject a guarantee claim that the artifact does not support.

**Exclusions.** This file does not restate the derivations in the three topic cards of this
collection: method selection is card 01, federated topology is card 02, and data minimisation with
synthetic-release limits is card 03. It does not cover legal basis, consent drafting, key ceremony
procedure, or network security except where a privacy claim depends on them. It gives no guarantee
that a listed check is sufficient for a regulator.

**Level and reading order.** Advanced. Read Section 2 for the failure lanes, Section 3 for precise
definitions, Section 4 for the nine results the records depend on, Section 5 for one end-to-end run
priced honestly, then Section 8 for the records themselves. Section 11 rehearses the same material
as questions; if you cannot answer those, the records will not help you under time pressure.

**Retrieval cue.** Numbers live in Section 4 and Section 5. The catalogue is Section 8. Rehearsal is
Section 11. The card is Section 12 and nothing follows it.

**Version.** 1.0, not independently reviewed. Review triggers: a change in the trust model, a change
in the sampling scheme, a change in P or in per-round cohort size, or a change in the accountant
implementation. Each of those four invalidates a different subset of the records.

**What this sheet refuses to claim.** That a small epsilon is always better regardless of utility.
That k-anonymity is a privacy guarantee. That secure aggregation protects against a malicious server
who can also observe metadata. That synthetic data is private because it is fake. That a membership
inference AUC above 0.5 on a small sample demonstrates memorisation. All five appear in the
catalogue as records, because all five are common and all five are wrong for the stated reasons.

**Signature of a competent privacy review.** One row per released artifact naming: the unit of
privacy, the mechanism, the accounting method and version, the achieved (epsilon, delta), the
population the budget covers, the clipping norm and noise multiplier, the composition across
releases, the residual attack evidence with sample size, and the re-identification test that would
have caught a regression. If any of those nine is missing, the review is an opinion.

**Takeaway.** A privacy claim is an accounting statement with a unit, a population, a mechanism and
an auditor; everything else in this file exists to stop a claim from being mistaken for one.

## 2. Concept Map / Mental Model

The first model is three lanes, because the three ways privacy work fails look nothing alike and are
usually diagnosed by different people. A numeric failure is settled by arithmetic. A structural
failure is settled by reading the data flow. A semantic failure is settled by asking what the
guarantee is about.

```text
                     a privacy claim is made
                              |
        +---------------------+---------------------+
        |                     |                     |
   the number is        the flow is           the guarantee
   wrong                wrong                 is about something
        |                     |                else
        v                     v                     v
   epsilon, sigma,      who sees what,       k-anonymity vs DP,
   delta, composition   keys, metadata       DP vs robustness
        |                     |                     |
   Sections 4, 9        Section 8 SF-*        Section 8 CC-*
```

The second model is the guarantee stack. Each layer buys a different statement, and the statements do
not compose upward for free. Reading downward, each step removes an assumption that the step above
silently relied on.

```text
layer 5  released statistic     DP per query family, budget spent
layer 4  aggregated update      secure aggregation, threshold n/2
layer 3  clipped update         sensitivity C, noise sigma*C
layer 2  local training         record-level DP-SGD, per-example grads
layer 1  enrolment             identity, one record per user, consent
```

A claim at layer 4 says nothing about layer 2. A run can be perfectly aggregated and still leak
through an unclipped update: aggregation hides *whose* update it was, not *what* the update says
about the records inside it.

The third model is the accounting funnel, which is where most numeric failures originate. Each stage
is a place to lose the guarantee or to lose the budget, and the order matters because a bound applied
in the wrong order cannot be undone.

```text
threat model and unit of privacy
        |
        v
sensitivity bound (clip norm C)
        |
        v
per-step noise (sigma)
        |
        v
per-step epsilon under sampling (q)
        |
        v
composition over T steps (basic | advanced | RDP | zCDP)
        |
        v
conversion to (epsilon, delta) at the chosen delta
        |
        v
partition of the budget across releases
```

Nine of the thirty records are a stage in this funnel being skipped, applied twice, or applied with
the wrong variable. The funnel is also the reason a "we used DP" statement is unfalsifiable without
the fourth and fifth lines: the mechanism is the easy part.

The fourth model is the federated loop, drawn as a data-flow rather than a sequence, because the
failures are at the edges of the boxes and not in the order of arrows.

```text
clients (60,000 enrolled, 1,200 per round)
   |  local data stays here
   |  local DP-SGD: clip C, noise sigma*C
   v
updates (P floats each, 7.248 MB fp32)
   |  optional: quantise, sparsify, encrypt
   v
aggregator (sums, applies threshold, averages)
   |  metadata visible here: who, when, how often
   v
server model (published as the artifact)
   |  released with an accounting statement
   v
consumers (queries, gradients, or weights)
```

The metadata box is the one most often left outside the privacy analysis and the one every
deployment that has actually been attacked leaked through. It appears in the catalogue as SF-06.

**Why the models are separate.** A single diagram that tried to hold accounting, data flow and
semantics at once would be unusable in a review, because a reviewer at a given meeting is asking one
of the four questions and needs the diagram that answers it. Section 7 gives the selection table that
maps question to mechanism; Sections 4 and 5 give the arithmetic.

**Takeaway.** Diagnose the lane before the fix: almost every wrong correction in this topic is a fix
applied in the right lane for the wrong symptom.

## 3. Foundational Definitions

Each definition states the object, its type, and the mistake it prevents. Definitions are ordered so
that every later one uses only earlier ones.

**Record and unit of privacy.** A record is the smallest unit the guarantee is about: one row, one
user, or one device. Type: a partition of the population. The unit is chosen, not discovered, and the
whole budget is denominated in it. Mistake prevented: claiming record-level DP when a client holds
40 rows, which silently converts epsilon into 40·epsilon through group privacy. See NE-10.

**Adjacent datasets.** Two datasets differing in exactly the unit of privacy: one record changed, or
one record added or removed. Type: a relation on datasets of equal size (bounded) or differing by
one (unbounded). Mistake prevented: mixing bounded and unbounded sensitivity, which changes the clip
norm and therefore sigma.

**Mechanism.** A randomised map from datasets to outputs. Type: a conditional distribution. The
randomness is the guarantee; a deterministic mechanism has infinite epsilon unless its output is
constant.

**(epsilon, delta)-differential privacy.** A mechanism M satisfies it if for all adjacent D, D' and
all output sets S, P[M(D) in S] <= e^epsilon · P[M(D') in S] + delta. Type: a two-parameter bound on
the likelihood ratio, weakened on a set of output mass at most delta. Mistake prevented: reading
delta as "the probability of a leak" — it is the mass of the output region where the epsilon bound is
allowed to fail, which is why a delta of 1e-2 in a 1e6-record dataset is a materially different
object from the same delta in a 60,000-record cohort.

**Sensitivity.** For a query f, the largest change in f's output between adjacent datasets, measured
in the norm the mechanism's noise is calibrated to (L1 for Laplace, L2 for Gaussian). Type: a scalar
with the units of the statistic, or a norm over a vector. Mistake prevented: computing sensitivity on
the mean instead of the sum, or forgetting that a ratio's sensitivity is unbounded near a zero
denominator.

**Clipping norm C.** The per-record bound enforced before aggregation: each per-record gradient is
scaled by min(1, C/||g||). Type: a scalar in gradient units, and simultaneously the L2 sensitivity of
the sum. Mistake prevented: treating C as a tuning constant rather than as the quantity that sets
both the noise and the bias. Small C buys privacy and costs accuracy; the cost is quantified in 4.3.

**Noise multiplier sigma.** The standard deviation of the Gaussian noise added to the clipped sum,
expressed in units of C. Type: dimensionless, sigma > 0. The realised noise standard deviation is
sigma·C. Mistake prevented: quoting sigma without C, which makes the noise unmatchable to the
sensitivity that justifies it.

**Sampling rate q.** The fraction of the eligible population included in one step. Type: dimensionless
in (0,1]. Amplification by subsampling turns q into a reduction in epsilon that is first order in q.
Mistake prevented: accounting at q = 1 for a run that samples, which inflates the budget by a large
factor and pushes teams to over-noise in compensation.

**Composition.** The rule that combines per-step guarantees into a guarantee for the sequence. Type: a
theorem schema with variants: basic (sum), advanced (sum of e^epsilon - 1), RDP (sum of Rényi
divergences, converted), zCDP (sum of rho). Mistake prevented: choosing the variant after seeing which
number is convenient. The variants differ by orders of magnitude on the same run, as 4.2 shows.

**Rényi divergence of order alpha.** D_alpha(P||Q) = log(E_Q[(dP/dQ)^alpha])/(alpha - 1). Type: a
non-negative scalar per alpha. It composes additively and converts to (epsilon, delta) at a cost that
depends on alpha, so the binding alpha matters and the conversion must be minimised over the grid.

**Zero-concentrated DP (rho-zCDP).** A mechanism is rho-zCDP if the Rényi divergence of order
1 + alpha is at most rho·alpha for all alpha > 0. Type: one scalar rho. It converts to (rho +
2·sqrt(rho·log(1/delta)), delta). Mistake prevented: composing rho and then reporting it as epsilon.

**Local DP (LDP).** The guarantee holds for each record before it leaves its owner, so no aggregator
is trusted with raw values. Type: the same (epsilon, delta) object applied at the record boundary.
Mistake prevented: expecting central-DP accuracy from a local mechanism: the variance gap is a factor
of sqrt(n), quantified in 4.3.

**Shuffle model.** Records are locally randomised, then shuffled by a trusted-but-not-curious
shuffler with no access to identities. Type: an intermediate trust point between local and central.
It buys accuracy close to central DP at local trust, but only if the shuffler genuinely cannot see
the sender, which is a systems claim rather than a mathematical one.

**Secure aggregation.** A protocol in which clients secret-share masked updates so the server learns
only the sum above a threshold of surviving contributions. Type: a cryptographic protocol with a
correctness condition (threshold reached) and a privacy condition (no collusion below the threshold).
Mistake prevented: treating secure aggregation as a substitute for clipping and noise; it hides
contributors, not content.

**Threshold.** The minimum number of surviving masked updates for the server to unmask the sum.
Mistake prevented: setting the threshold equal to the sampled cohort, which makes every straggler a
failed round. See SF-03 for the survival arithmetic.

**k-anonymity.** Every released row is indistinguishable from at least k-1 others on the
quasi-identifier set. Type: a combinatorial property of the release, not a bound on an adversary's
success. Mistake prevented: reading k as 1/k privacy. It bounds the *within-table* ambiguity and
nothing else.

**l-diversity.** Each equivalence class contains at least l distinct values of the sensitive
attribute. Type: a per-class count. Mistake prevented: assuming a class with l = 2 values is
protected when the split is 80/20; the posterior on the top value is then 0.8.

**t-closeness.** The distribution of the sensitive attribute within each class is within total
variation t of its distribution in the whole table. Type: a distance per class. It is the property
that actually rules out the 80/20 class, and it costs more utility than l-diversity, which is why it
is used less.

**Membership advantage.** For a distinguisher A, the quantity 2·P[A guesses "in"] - 1 on a balanced
pair of worlds. Type: a scalar in [-1, 1]. Mistake prevented: comparing an advantage to a probability,
or reporting an advantage from 60 records without an interval.

**FedAvg.** Weighted averaging of client updates, with weights proportional to client sample counts.
Type: an estimator. It is not the minimiser of any single objective when client distributions differ,
which is the source of the weighting record CI-01 in the catalogue.

**Staleness.** The number of rounds between the model a client trained against and the model the
update is applied to. Type: an integer per update, bounded by cohort size over participation rate.

**Event time and processing time.** Event time is when a record was created; processing time is when
the privacy budget was spent on it. Type: two timestamps. Mistake prevented: spending one budget
across two releases whose populations overlap in event time, which double-counts the same records.

**Takeaway.** Every term above has a type and a mistake it prevents; if you cannot name the type, you
cannot tell whether two guarantees are the same kind of object.

## 4. Core Formulas with Full Derivation

Each result below uses one anatomy: the formula, the symbols, the validity condition, a numerical
example on the reference system, and the interpretation that a review needs. Arithmetic is
ILLUSTRATIVE and recomputable by hand.

### 4.1 Base case

**C1 — Gaussian mechanism noise scale (classical form).**

**FORMULA**

```text
sigma = Delta * sqrt(2 * ln(1.25 / delta)) / epsilon
```

**WHERE.** Delta is the L2 sensitivity of the released quantity; for a clipped per-record
contribution with clip norm C, Delta = C. epsilon and delta are the target guarantee. sigma
multiplies C to give the noise standard deviation.

**CONSTRAINT.** Valid as a *sufficient* condition for a single Gaussian mechanism when
epsilon < 1 is the regime the constants were fitted to, and only for the one-shot mechanism. It
is not a composition rule and it must not be applied to a composed mechanism.

**NUMERICAL EXAMPLE.** C = 1, delta = 1e-5, epsilon = 1.0.

```text
ln(1.25 / 1e-5)        = ln(125,000)          = 11.7361
sqrt(2 * 11.7361)      = sqrt(23.4722)        = 4.8458
sigma                  = 4.8458 / 1.0         = 4.8458
noise sd               = sigma * C            = 4.8458
relative to signal     = 4.8458 / 1.0         = 4.85x
```

**INTERPRETATION.** At epsilon = 1 a single release of a unit-sensitivity statistic needs noise about
4.85 times its sensitivity. That is the honest price of a strong single-shot guarantee, and it is why
epsilon = 1 for a 500-round training run is not reachable at any useful learning rate.

**C2 — Exact delta achieved by a Gaussian mechanism.**

**FORMULA**

```text
delta(epsilon, sigma, Delta) = Phi(Delta/(2 sigma) - epsilon sigma / Delta)
                             - e^epsilon * Phi(-Delta/(2 sigma) - epsilon sigma / Delta)
```

**WHERE.** Phi is the standard normal CDF. The formula is exact for the Gaussian mechanism with
L2 sensitivity Delta and noise sd sigma·Delta, so it can be inverted numerically for the smallest
sigma that meets a target (epsilon, delta).

**CONSTRAINT.** Requires Delta > 0 and sigma > 0. It describes one Gaussian release. Applying it to a
composed training run is the single most common accounting error in this topic, and it is record
NE-01.

**NUMERICAL EXAMPLE.** epsilon = 8, delta target 1e-5, Delta = 1, sigma from C1 = 0.6056.

```text
term1 = Phi(1/(2*0.6056) - 8*0.6056)   = Phi(0.8256 - 4.8448)
      = Phi(-4.0192)                   = 2.920e-05
term2 = e^8 * Phi(-0.8256 - 4.8448)    = 2980.958 * Phi(-5.6704)
      = 2980.958 * 7.123e-09           = 2.123e-05
delta = 2.920e-05 - 2.123e-05          = 7.97e-06
```

The classic formula returned sigma = 0.6056 and the exact check says 7.97e-06, which satisfies
1e-5. Marginally: 7.97e-06 is 80% of the target, not 1%.

**INTERPRETATION.** C1 is a sufficient condition with slack that shrinks as epsilon grows. At
epsilon = 8 the slack is 20%; at epsilon = 16 it is negative and the formula fails. Section 4.4
locates that boundary.

**C3 — Subsampled-Gaussian RDP at integer alpha.**

**FORMULA**

```text
eps_RDP(alpha) = 1/(alpha-1) * ln( sum_{j=0..alpha} C(alpha,j)
                 * q^j * (1-q)^(alpha-j) * exp(j(j-1)/(2 sigma^2)) )
```

**WHERE.** alpha is an integer >= 2, q the sampling rate, sigma the noise multiplier, C(alpha,j) the
binomial coefficient. The sum is over the number of records j from the sampled batch that are also
in the adjacent dataset.

**CONSTRAINT.** Stated for Poisson sampling; fixed-size sampling without replacement is a different
scheme and needs its own accounting. The bound is loose at large alpha, which is why the conversion
must be minimised over alpha rather than fixed at one value. Compute in log space: the terms overflow
double precision above alpha ~ 20 at small sigma.

**NUMERICAL EXAMPLE.** q = 0.02, sigma = 1.1, alpha = 7, T = 500.

```text
j=0: 1 * 0.98^7 * 1                     = 0.868126
j=1: 7 * 0.02 * 0.98^6 * 1              = 0.124018
j=2: 21 * 0.0004 * 0.98^5 * e^0.8264    = 0.017351
j=3..7: decreasing, total               = 0.004233
sum                                     = 1.013728
eps_RDP(7) = ln(1.013728) / 6           = 0.002272
T * eps_RDP(7)                          = 1.1362 (in RDP units)
```

The check that the sum tends to 1 as sigma grows is visible in the first two terms: with the
exp factors stripped, the binomial weights sum to exactly 1.0000, so the whole excess is the
second term's e^0.8264 inflation.

The arithmetic is stated in RDP units deliberately: 1.136 is not an epsilon and must never be
reported as one. The conversion in C4 is what turns it into a number a reviewer recognises.

**INTERPRETATION.** Per-step RDP at alpha = 7 is 0.177, which looks enormous next to an epsilon
target of 1.0. The conversion divides by (alpha - 1), and the alpha that minimises the converted
value is small, which is why a naive "we compose RDP and then read epsilon off alpha = 64" produces
absurd numbers such as 13,568.

**C4 — RDP to (epsilon, delta) conversion.**

**FORMULA**

```text
epsilon(delta) = min over alpha of [ rho_RDP(alpha) + ln(1/delta)/(alpha-1) ]
```

**WHERE.** rho_RDP(alpha) is the total RDP budget at order alpha, that is T times the per-step value
from C3. delta is the target failure mass. The minimisation is over the same alpha grid used for
composition.

**CONSTRAINT.** The grid must be dense enough near the optimum. At q = 0.02, sigma = 1.1 the optimum
sits at alpha = 7, and a grid of powers of two (8, 16, 32, 64) reports 3.083 instead of 3.055 — a 1%
error that is tolerable, while a grid stopping at alpha = 6 reports 3.216 and one starting at
alpha = 10 reports 13.37.

**NUMERICAL EXAMPLE.** Same run as C3.

```text
alpha=4:  rho=0.5537  + 11.5129/3 = 3.8376 -> 4.3914
alpha=5:  rho=0.7231  + 11.5129/4 = 2.8782 -> 3.6014
alpha=6:  rho=0.9132  + 11.5129/5 = 2.3026 -> 3.2157
alpha=7:  rho=1.1362  + 11.5129/6 = 1.9188 -> 3.0551  <- minimum
alpha=8:  rho=1.4385  + 11.5129/7 = 1.6447 -> 3.0832
alpha=9:  rho=2.2193  + 11.5129/8 = 1.4391 -> 3.6584
alpha=10: rho=12.0936 + 11.5129/9 = 1.2792 -> 13.3728
```

The ln(1/delta) term is 11.5129 = ln(1e5) at every line, which is why a large delta target makes the
conversion cheap and a small one makes it expensive.

**INTERPRETATION.** The 500-round run at sigma = 1.1 spends epsilon = 3.055, not the 1.0 that was
targeted. Bisecting the same accountant for the sigma that does meet epsilon = 1.0 at this q and T
gives 2.401, which is 2.401/1.1 = 2.183 times the noise in every coordinate. Since the
per-coordinate signal-to-noise ratio is proportional to g·n/(sigma·C), that factor cuts it from 2.182
to about 1.000 — the aggregated update then carries no more signal than a single noised step, and the
500-round run learns nothing. The gap between the target and the achievable value is the subject of
records NE-02 and NE-08, and the correction is either a larger sigma or fewer rounds, never a
different accountant.

**C5 — Basic composition.**

**FORMULA**

```text
epsilon_total = sum over t of epsilon_t ,  delta_total = sum over t of delta_t
```

**WHERE.** epsilon_t and delta_t are the per-step guarantees of each mechanism in the sequence. For
identical steps this is T·epsilon_1.

**CONSTRAINT.** Valid always, and nearly always wasteful. It is the right bound only when the steps
are fully adaptive with no sampling amplification and the auditor wants a bound with no constant
hidden in it.

**NUMERICAL EXAMPLE.** Single-round epsilon from C1 with amplification: sigma = 1.1 gives a
full-batch epsilon of 4.4044, and at q = 0.02 the amplified first-order value is

```text
ln(1 + q*(e^4.4044 - 1)) = ln(1 + 0.02*80.8075) = ln(2.6161) = 0.9617
500 * 0.9617                                   = 480.85
```

**INTERPRETATION.** Basic composition spends 480.85 where RDP spends 3.055, a factor of 157. Both are
valid statements about the same run. Reporting the larger number is conservative but useless, and
teams that do it conclude DP is infeasible; the correct move is a tighter accountant, not a smaller
model.

**C6 — Amplification by subsampling (first order).**

**FORMULA**

```text
epsilon_sampled ~ ln(1 + q * (e^epsilon_full - 1))
```

**WHERE.** epsilon_full is the guarantee of the mechanism applied to the whole cohort, q the sampling
rate. This is the small-q expansion that most implementations use to convert a full-batch epsilon
into a per-step value.

**CONSTRAINT.** First order in q, and it degrades quickly: at q = 0.25 the approximation is already
materially optimistic relative to accounting the subsampled mechanism directly. It also assumes the
sampling is uniform and independent of the data, which a stratified or "most valuable clients first"
sampler violates.

**NUMERICAL EXAMPLE.** epsilon_full = 8.0, four sampling rates.

```text
q=0.02  -> ln(1 + 0.02*2980.958) = ln(60.599)  = 4.1043
q=0.05  -> ln(1 + 0.05*2980.958) = ln(149.998) = 5.0106
q=0.25  -> ln(1 + 0.25*2980.958) = ln(745.989) = 6.6147
q=1.00  -> ln(1 + 1.00*2980.958) = ln(2981.958)= 8.0000
```

**INTERPRETATION.** Amplification is worth a factor of 2 at q = 0.25 and a factor of 2.4 at q = 0.02,
measured against the full-batch value. It is a real and large effect, and it is the reason small
cohorts are attractive — which is exactly the tension with straggler and throughput costs in Section
9.

### 4.2 General case

**C7 — Advanced composition.**

**FORMULA**

```text
epsilon_total = sqrt(2 T ln(1/delta')) * epsilon_1 + T * epsilon_1 * (e^epsilon_1 - 1)
delta_total   = delta' + T * delta_1
```

**WHERE.** epsilon_1 is the per-step epsilon (after amplification), T the number of steps, delta' the
slack allocated to the composition rather than to the mechanism.

**CONSTRAINT.** Beats basic composition only when T is large and epsilon_1 small; for epsilon_1 near
1 the second term dominates and it degenerates to a worse bound than basic composition. It is
asymptotic in T and should not be used for T < 10.

**NUMERICAL EXAMPLE.** epsilon_1 = 0.9617, T = 500, delta' = 5e-6.

```text
sqrt(2*500*ln(2e5))      = sqrt(1000*12.2061) = sqrt(12206.1) = 110.48
first term  = 110.48 * 0.9617               = 106.25
second term = 500 * 0.9617 * (e^0.9617 - 1) = 500*0.9617*1.6157 = 777.0
total       = 883.3
```

**INTERPRETATION.** Advanced composition is *worse* than basic composition on this run, 883 against
481, because epsilon_1 is order 1. The lesson is not that advanced composition is broken; it is that
the regimes where each bound is tight are different, and a team that picks the bound by reputation
without checking the regime reports a number that does not help them.

**C8 — Clipping bias (expected shrink).**

**FORMULA**

```text
E[||g_clipped||] / E[||g||] = P(||g|| <= C) + C * E[ 1/||g|| * 1{||g|| > C} ]
```

**WHERE.** ||g|| is the per-record gradient norm, C the clip norm. For lognormally distributed norms
with parameters (m, s), the tail expectation has the closed form
C·e^(-m + s²/2)·Phi((m - s² - ln C)/s).

**CONSTRAINT.** The identity is exact for any norm distribution; the closed form requires lognormal.
The consequence that matters is not the mean but the *direction*: clipping shrinks every contribution
that exceeds C by a different factor, so the clipped sum is not the sum of gradients scaled by a
constant, and the bias does not vanish with more records.

**NUMERICAL EXAMPLE.** C = 1, lognormal norm with median 1.2214 (m = 0.2, s = 0.5).

```text
m = 0.2, s = 0.5, C = 1, median = e^0.2 = 1.2214

P(g <= C)          = Phi((0 - 0.2)/0.5) = Phi(-0.4)      = 0.3446
normaliser         = e^(-0.2 + 0.25/2)  = e^(-0.075)     = 0.9277
normal argument    = (0.2 - 0.25 - 0)/0.5                = -0.10
Phi(-0.10)                                               = 0.4602
E[1/g ; g > C]     = 0.9277 * 0.4602                     = 0.4269
shrink             = 0.3446 + 1.0 * 0.4269               = 0.7715
relative bias      = 1 - 0.7715                          = 0.2285
```

Two checks on that block. First, the tail term 0.4269 must be at most P(g > C) = 1 - 0.3446 = 0.6554,
because 1/g <= 1 everywhere on the tail; it is, so the magnitude is sane. Second, the same formula at
m = 0.5, s = 0.5 gives P(g <= 1) = Phi(-1) = 0.1587, normaliser e^(-0.375) = 0.6873, argument
(0.5 - 0.25)/0.5 = 0.5 so Phi(0.5) = 0.6915, tail 0.4753, shrink 0.6340. A higher median norm with the
same clip produces more bias, not less, because the whole distribution sits further into the clipped
region.

**INTERPRETATION.** A clip norm that sits below the modal gradient norm does not merely add noise; it
changes the objective, and no amount of training recovers the lost component. Records NE-06 and
NE-07 exist because this is routinely diagnosed as "DP makes the model worse".

**C9 — Local versus central variance for a mean.**

**FORMULA**

```text
central: sd = sqrt(2)/ (n * epsilon)      for a sum with L1 sensitivity 1
local:   sd = sqrt(2) / (epsilon * sqrt(n))
ratio:   sd_local / sd_central = sqrt(n)
```

**WHERE.** n is the number of records, epsilon the per-record budget in the local case and the
per-query budget in the central case. Both are for the Laplace mechanism on a count.

**CONSTRAINT.** The sqrt(n) gap is an upper bound on what local randomisation can achieve relative to
central for this query; no local mechanism closes it, and the shuffler model recovers part of it only
by assuming the shuffler cannot see identities.

**NUMERICAL EXAMPLE.** epsilon = 1, three cohort sizes.

```text
n=50:    central sd = 1.4142/50    = 0.02828
         local sd   = 1.4142/7.071 = 0.20000   ratio 7.07
n=512:   central sd = 1.4142/512   = 0.00276
         local sd   = 1.4142/22.63 = 0.06250   ratio 22.63
n=5000:  central sd = 1.4142/5000  = 0.00028
         local sd   = 1.4142/70.71 = 0.02000   ratio 70.71
```

**INTERPRETATION.** At the reference cohort of 5,000 records a local mechanism at epsilon = 1 needs
70 times the records to match central accuracy, or equivalently spends 70·epsilon to match it at the
same n. Any proposal to "just use local DP" for a model update is a proposal to change the cohort by
that factor.

### 4.3 Variants

**C10 — zCDP composition and conversion.**

**FORMULA**

```text
rho_total = sum over t of rho_t
epsilon(delta) = rho_total + 2 * sqrt(rho_total * ln(1/delta))
```

**WHERE.** rho is the zCDP parameter; for a Gaussian mechanism with sensitivity Delta and noise
sigma·Delta, rho = Delta²/(2 sigma²) per release.

**CONSTRAINT.** Valid for all rho > 0 and any delta in (0,1). It is looser than RDP at small rho and
much easier to implement correctly, which makes it the right default when the engineering team owns
the accountant.

**NUMERICAL EXAMPLE.** rho = 0.01 per round, T = 10 and T = 50, delta = 1e-5.

```text
T=10: rho_total = 0.1  -> 0.1 + 2*sqrt(0.1*11.5129)  = 0.1 + 2.1460 = 2.2460
T=50: rho_total = 0.5  -> 0.5 + 2*sqrt(0.5*11.5129)  = 0.5 + 4.7986 = 5.2986
```

**INTERPRETATION.** Doubling the round count from 10 to 50 multiplies epsilon by 2.36, not by 5, and
not by sqrt(5). Composition in zCDP is linear in rho, so the conversion's square root is what softens
the growth: this is the quantitative reason long runs are cheaper per round than short ones.

**C11 — Group privacy.**

**FORMULA**

```text
epsilon_k = k * epsilon ,  delta_k = k * e^((k-1) * epsilon) * delta
```

**WHERE.** k is the number of records that one unit of privacy can influence — for a client holding k
rows, k is that count.

**CONSTRAINT.** The delta term grows exponentially in k and becomes vacuous quickly, which is why
group privacy is a diagnostic rather than a fix: a client with k = 4 at epsilon = 3.055 gets
delta = 0.382, which is no guarantee at all.

**NUMERICAL EXAMPLE.** epsilon = 3.0551, delta = 1e-5.

```text
k=2: eps = 6.110  delta = 2 * e^3.0551 * 1e-5 = 2*21.223*1e-5 = 4.24e-4
k=4: eps = 12.220 delta = 4 * e^9.1654 * 1e-5 = 4*9575.6*1e-5 = 0.3824
```

**INTERPRETATION.** A client with four records destroys the guarantee even though the per-record
claim is intact. The fix is structural — one contribution per user — not numerical, which is why
CI-02 and NE-10 sit in different lanes and share a root cause.

**C12 — Membership advantage from epsilon.**

**FORMULA**

```text
advantage <= (e^epsilon - 1) / (e^epsilon + 1) = tanh(epsilon / 2)
```

**WHERE.** advantage is 2·P[A correct] - 1 for a balanced membership game; epsilon is the guarantee of
the mechanism that produced the released object.

**CONSTRAINT.** The bound applies to *any* distinguisher against the mechanism's output, so it is a
statement about worst-case attackers, not about a particular attack implementation. It saturates:
at epsilon = 8 the bound is 0.9993, so DP at that epsilon offers no useful protection against
membership.

**NUMERICAL EXAMPLE.**

```text
epsilon = 0.5    -> tanh(0.25) = 0.2449
epsilon = 1.0    -> tanh(0.50) = 0.4621
epsilon = 3.0551 -> tanh(1.5276) = 0.9100
epsilon = 8.0    -> tanh(4.0)  = 0.9993
```

**INTERPRETATION.** The reference run's epsilon of 3.055 permits a membership advantage of 0.91 in the
worst case. That is not a defect in the accounting; it is the accounting telling you the run is not a
strong privacy claim, and the honest response is to say so rather than to report "we are
differentially private".

**C13 — FedAvg weighting.**

**FORMULA**

```text
w_i = n_i / sum_j n_j ,  theta_next = sum_i w_i * theta_i
```

**WHERE.** n_i is the number of records client i trained on this round, theta_i its local update.

**CONSTRAINT.** The estimator is unbiased for the pooled risk only when each client's local objective
is the same functional form and the weights are proportional to counts. Under label shift the weights
are wrong by an amount proportional to the shift, and no averaging scheme fixes it.

**NUMERICAL EXAMPLE.** Three clients with 1,200, 800 and 4,000 records, accuracies 0.81, 0.85, 0.79.

```text
weights: 0.2000, 0.1333, 0.6667   (sum 1.0000)
weighted: 0.2*0.81 + 0.1333*0.85 + 0.6667*0.79 = 0.8020
uniform : (0.81 + 0.85 + 0.79)/3              = 0.8167
gap     : 0.8167 - 0.8020                     = 0.0147  (1.47 pp)
```

**INTERPRETATION.** The uniform average is 1.47 percentage points optimistic, and it is optimistic in
the direction of the small clients. Any dashboard that averages per-client metrics without weights
will disagree with the deployed model by about that much, which is CI-01.

**C14 — Threshold survival under dropout.**

**FORMULA**

```text
survivors ~ Binomial(n, 1 - p_drop) ,  P(round succeeds) = P(survivors >= threshold)
```

**WHERE.** n is the number of clients that accepted the round, p_drop the per-client probability of
not completing it, threshold the unmasking requirement.

**CONSTRAINT.** Independent dropout is assumed. Real dropout is correlated — a bad app version, a
single carrier outage, a scheduled maintenance window — and correlated dropout has the same mean and
a much larger variance, so the tail probability computed here is optimistic.

**NUMERICAL EXAMPLE.** n = 1,200, threshold 900.

```text
p_drop=0.10: mean 1080.0  sd 10.39  P(>=900) = 1.000000
p_drop=0.25: mean  900.0  sd 15.00  P(>=900) = 0.515508
p_drop=0.30: mean  840.0  sd 15.87  P(>=900) = 0.000070
```

**INTERPRETATION.** At a 25% dropout rate the round succeeds half the time, which presents as
intermittent stalls with no error message. At 30% it never succeeds, and the failure is total rather
than partial. The threshold must be set from the dropout distribution, not from the sample size.

**C15 — Communication cost per round.**

**FORMULA**

```text
bytes_per_round = n_sampled * P * b ,  total = rounds * bytes_per_round
```

**WHERE.** b is bytes per parameter (4 for fp32, 2 for fp16, 1 for int8), n_sampled the cohort size.

**CONSTRAINT.** Counts one direction only; the server also broadcasts a model of the same size, and
any masking or encryption adds overhead that can exceed the payload (C17).

**NUMERICAL EXAMPLE.** P = 1,812,000, n = 1,200, 500 rounds.

```text
per client fp32 = 1,812,000 * 4      = 7,248,000 B = 7.248 MB
per round       = 1,200 * 7.248 MB   = 8,697.6 MB  = 8.70 GB
500 rounds      = 500 * 8.70 GB      = 4,348.8 GB  = 4.35 TB
per client 10 rounds = 10 * 7.248 MB = 72.48 MB
```

**INTERPRETATION.** 4.35 TB of uplink for a 1.8M-parameter model is the number that kills federated
proposals in review. It is also the number quantisation and sparsification are aimed at, and SF-02
shows why their savings are smaller than advertised.

### 4.4 Limits and counterexamples

**L1 — The classical sigma formula becomes anti-conservative above epsilon ~ 9.** Inverting C2
numerically for delta = 1e-5 gives the following comparison. The classic column is C1; the exact
column is the smallest sigma satisfying C2; the last column is the delta that C1's sigma actually
achieves.

```text
eps    classic   exact    ratio   delta at classic sigma
 1.0    4.8448   3.7306   1.299   4.11e-08   conservative
 2.0    2.4224   1.9938   1.215   1.34e-07   conservative
 4.0    1.2112   1.0812   1.120   6.85e-07   conservative
 8.0    0.6056   0.6002   1.009   7.97e-06   conservative, 20% slack
10.0    0.4845   0.4999   0.969   2.27e-05   VIOLATES the target
16.0    0.3028   0.3442   0.880   3.36e-04   33.6x the target
20.0    0.2422   0.2900   0.835   1.53e-03   153x the target
```

The boundary is between epsilon = 8 and epsilon = 10 at delta = 1e-5. Below it the formula is safe
and loose; above it the formula returns noise that is too small, and the error grows to a factor of
three by epsilon = 20. This is record NE-01, and the discriminating check is C2 evaluated at the
sigma you actually shipped.

**L2 — The bound is about the mechanism, not the data.** C12 says a mechanism at epsilon = 3 permits
advantage 0.91. It does not say an attacker will reach 0.91 on your model, and it does not say that a
measured advantage of 0.6 means the mechanism is broken. Both directions of that inference are wrong:
the bound is worst-case over distinguishers, and a measurement on a finite sample has its own
uncertainty. At 60 records split 30/30, an AUC of 0.5 has a 95% interval of roughly [0.35, 0.65], so
an attack reporting 0.62 has demonstrated nothing. This is CC-03.

**L3 — Composition bounds do not transfer between schemes.** A budget derived under Poisson sampling
does not apply to fixed-size sampling without re-derivation, and a budget composed with basic
composition cannot be "converted" to RDP after the fact. Each row of the accounting statement belongs
to one (scheme, accountant) pair. This is NE-04.

**L4 — Where C8's bias model breaks.** The shrink formula assumes the norm distribution is stable
across steps. Under learning-rate warmup the norms grow by an order of magnitude in the first few
hundred steps, so a C chosen from the steady-state distribution clips almost everything early on. The
counterexample: a run whose steady-state median norm is 1.22 and whose warmup median is 4.0 spends
its first 50 steps at a shrink factor near 0.2, which is enough to put the model in a region it never
recovers from. The discriminating check is per-step clip fraction, not the run-level average.

**L5 — Validity condition for the whole section.** Every result above is a statement about a
mechanism with a stated sensitivity, a stated sampling scheme, and a stated unit of privacy. None of
them is a statement about a *system* that also logs request timestamps, retries failed rounds, or
keeps the previous model version addressable. The system-level claim needs Section 9's cost model and
the metadata record SF-06.

**Takeaway.** The formulas are easy; the limits are where reviews fail, and each limit here is a
record you can look up by name.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE. One federation is priced end to end from enrolment to release so that
every record in Section 8 has a number behind it. No run was performed, no vendor was consulted, and
none of the figures describes a real deployment.

**Setup.** The reference system of Section 1: P = 1,812,000, 60,000 enrolled clients, 1,200 sampled
per round at q = 0.02, 500 rounds, 250 local minibatch steps of 32 per round, clip C = 1.0 in L2 norm,
Gaussian noise at sigma = 1.1 applied client-side, fp32 uploads, target (1.0, 1e-5) at record level.

**Step 1: what the privacy target costs.** Bisecting the accountant of C4 for epsilon = 1.0 at
q = 0.02 and T = 500 gives sigma = 2.401. Compare the two candidates on the only quantity that decides
whether learning happens, the per-coordinate signal-to-noise ratio of the averaged update.

```text
signal per coordinate        g = 0.002 (ILLUSTRATIVE, typical late-run magnitude)
noise sd per coordinate      sigma*C/n = 1.1*1.0/1200   = 9.167e-04
SNR at sigma=1.1            0.002 / 9.167e-04            = 2.182
SNR at sigma=2.401          0.002 / 2.001e-03            = 1.000
cumulative SNR over 500     sqrt(500) * 2.182           = 48.79
cumulative SNR at 2.401     sqrt(500) * 1.000           = 22.35
budget spent at sigma=1.1   3.055
budget spent at sigma=2.401 1.000
```

**Interpretation of step 1.** Honouring epsilon = 1.0 halves the cumulative signal-to-noise ratio and
in exchange buys a factor of 3.055 in the guarantee. That is a real trade, not a defect, and the
number a review needs is the pair (3.055, 48.79) rather than either alone: the run as configured is a
weak-privacy, learnable run, and the compliant variant is a strong-privacy run with less than half
the signal. A team that promised epsilon = 1.0 and shipped sigma = 1.1 has both a compliance problem
and a model-quality problem it will misattribute.

**Step 2: what clipping costs before any noise is added.**

```text
lognormal gradient norms, s = 0.5, C = 1.0

median 0.8187: P(clip)=0.3446  shrink=0.9102  bias=8.98%
median 1.0000: P(clip)=0.5000  shrink=0.8496  bias=15.04%
median 1.2214: P(clip)=0.6554  shrink=0.7715  bias=22.85%
median 1.6487: P(clip)=0.8413  shrink=0.6339  bias=36.61%
```

**Interpretation of step 2.** The four rows are the same mechanism at four operating points, and the
clip fraction rises from 34% to 84% as the median norm moves from 0.82 to 1.65. The bias is not
noise and cannot be averaged away: 22.85% of the update magnitude is lost in the direction it was
pointing. A team that reports "DP cost us 3 points" without this table cannot say whether the 3 points
came from noise (fixable by more data), clipping (fixable by a larger C at a privacy price), or the
interaction.

**Step 3: what the round costs in bytes.**

```text
per client fp32           1,812,000 * 4 B        = 7,248,000 B = 7.248 MB
per round uplink (n=1200) 1,200 * 7,248,000 B    = 8,697.6 MB  = 8.70 GB
500 rounds                500 * 8.70 GB          = 4,348.8 GB  = 4.35 TB
per client (10 rounds)    10 * 7.248 MB          = 72.48 MB
cross-check at fp16       4.3488 / 2             = 2.1744 TB
cross-check at int8       4.3488 / 4             = 1.0872 TB
```

**Interpretation of step 3.** Quantisation is the only lever with a linear effect, and it is a factor
of two or four and nothing more. The popular claim that sparsifying to 1% gives a 100x saving ignores
index overhead: 1% of 7.248 MB is 72.5 kB of values, but at 2 bytes per index for 18,120 indices the
indices alone are 36.2 kB, so the reduction is 97% of payload at best and the indices must be
transmitted in full. SF-02 is the record for teams that quoted the 100x.

**Step 4: what the mask exchange costs relative to the payload.**

```text
mask shares        32 B each, one per ordered pair among 1200 clients
share count        1,200 * 1,199                    = 1,438,800
share bytes        1,438,800 * 32 B                 = 46,041,600 B = 46.04 MB
round payload      8,697.6 MB
share fraction     46.04 / 8,697.6                  = 0.53%
```

**Interpretation of step 4.** At this cohort the masking overhead is negligible, half a percent, which
is why "secure aggregation is too expensive" is usually wrong at 1,200 clients. At 100,000 clients the
share count grows quadratically: 100,000 × 99,999 × 32 B = 320.0 GB per round, which is 37 times the
payload and is the real reason production systems use a tree or a Diffie-Hellman pair only among
neighbours rather than all pairs.

**Step 5: the round clock, which decides how many rounds fit in a night.**

```text
per-client round time, lognormal(m=ln 4, s=0.35)
mean                e^(1.3863 + 0.06125)          = 4.2527 s
99.5th percentile   e^(1.3863 + 0.35*2.5758)      = 9.8536 s
ratio p99.5 / mean  9.8536 / 4.2527               = 2.317
rounds per hour at mean                           = 846.5
rounds per hour at p99.5                          = 365.3
```

**Interpretation of step 5.** A round cannot finish before its slowest survivor, so the throughput that
matters is the tail, not the mean: 365 rounds per hour rather than 846, a factor of 2.317. Five hundred
rounds therefore take 500/365.3 = 1.369 hours of *round* time at the tail, before any
accounting for failed rounds. If the threshold is set at 900 of 1,200 and dropout is 25%, half the
rounds fail outright (C14), and the effective wall-clock cost doubles again.

**Step 6: the budget actually spent, stated three ways.**

```text
accountant          epsilon at delta=1e-5
basic composition   480.85
advanced (C7)       883.37
RDP (C4)            3.0551     at alpha = 7
zCDP equivalent     not applicable without re-deriving rho from sigma
```

**Independent re-check of the RDP figure.** Recompute the per-step RDP at alpha = 7 from C3 with the
terms listed there: ln(1.013728)/6 = 0.002272 per step, times 500 = 1.1362, plus ln(1e5)/6 = 1.9188
gives 3.0551. The same run read at alpha = 8 gives 1.4385 + 1.6447 = 3.0832, which is 0.9% higher, and
at alpha = 6 gives 0.9132 + 2.3026 = 3.2157, which is 5.3% higher. The minimum is genuinely at
alpha = 7 for this configuration, so quoting alpha = 8 costs 0.9% and quoting alpha = 6 costs 5.3% —
the kind of slip that changes a sign-off conversation without changing the run.

**Sensitivity note.** The result is most sensitive to q and sigma, not to T. At sigma = 1.1 and
T = 500, moving q from 0.02 to 0.04 takes epsilon from 3.055 to 6.181, doubling it. Moving T from 500
to 250 takes it to 2.364, a 22.6% reduction. Moving sigma from 1.1 to 1.2 takes it to 2.593, a 15.1%
reduction. A team that wants to improve the number by 20% has one cheap lever (sigma) that costs
model quality and one structural lever (fewer rounds) that costs training time; the sampling rate is
the one that changes fastest and is the one usually left out of the change record.

**Independent arithmetic re-check.** The whole of step 1 to step 6 is re-derivable from three facts:
P = 1,812,000 (from the layer shapes), n = 1,200 and q = 1,200/60,000 = 0.02, and sigma = 1.1. Every byte
figure is a product of P, n and a dtype width. Every epsilon figure is C3 evaluated at that q and
sigma and then composed linearly in T. If someone hands you a budget table that cannot be reproduced
from those three facts, the table was not computed from this configuration.

**Takeaway.** The honest summary of this run is one line: 500 rounds at sigma = 1.1, q = 0.02, spent
epsilon = 3.055 at delta = 1e-5, moved 4.35 TB of uplink, and cost 22.85% of the update magnitude to
clipping. Every record in Section 8 is a way that line is wrong in practice.

## 6. Visual / Geometric Intuition

The numbers above are easier to hold as five pictures. Each picture is a shape, not a diagram, and each
answers a question a reviewer asks out loud.

**6.1 The noise-versus-signal plane.** Two curves cross. Noise standard deviation per coordinate falls
as 1/n when you add clients, while the signal in the averaged update is fixed by the gradient
magnitude. On log-log axes the noise is a line of slope -1 in n.

```text
log(noise sd)
   |\
   | \        sigma=2.401 (eps=1.0): the line sits higher
   |  \
   |   \___   sigma=1.1 (eps=3.055)
   |       \___
   +-----------+---------- log n
       1200      6000
```

Doubling the cohort halves the noise and buys half a bit of epsilon headroom at fixed sigma, because q
doubles and amplification worsens. The picture explains why "just recruit more clients" is not a free
privacy improvement: it improves the *utility* term and simultaneously spends more of the *budget*
term.

**6.2 The clipping ball.** Clipping projects every per-record gradient onto a ball of radius C. The
geometry that matters is that gradients outside the ball keep their direction and lose magnitude,
while gradients inside keep both. The result is not a scaled gradient field: it is a field whose
magnitude has been flattened at the surface, so the sum points slightly differently from the true
sum whenever the outlier directions are correlated.

```text
        directions of large gradients
              \   |   /
               \  |  /
        --------(C)--------   ball of radius C
               /  |  \
              /   |   \
        all vectors here scaled to length C
```

A useful diagnostic image: if outliers concentrate in a subspace (one feature scale dominating), the
clipped field's bias is a rotation toward the rest of the space, and the training loss will plateau
while the gradient norm stays near C. That plateau is CI-07's signature.

**6.3 The epsilon-versus-sigma curve.** On log-log axes epsilon from C4 is close to a straight line of
slope -1 against sigma, with a knee where the binding alpha changes.

```text
eps
 8 |*
 4 |  *
 3 |    *        <- the run lives here (sigma 1.1, eps 3.055)
 2 |      *
 1 |          *
 0 +---+---+---+---+---+--- sigma
   1.0 1.2 1.5 2.0 2.5 3.0
```

The practical reading is the slope: at sigma = 1.1, a 9% increase in sigma (to 1.2) buys 15% of
epsilon. That slope is roughly -2, not -1, in this regime, because both the noise and the sampling
amplification interact. Any planning spreadsheet that assumes a linear trade will misprice the first
20% and get the sign right thereafter.

**6.4 The composition ladder.** Each accounting family sits at a different height for the same run,
and the heights are not a ranking of quality but a map of assumptions.

```text
480.85 |  basic composition    (no amplification, no asymptotic argument)
883.37 |  advanced composition (asymptotic in T, bad here because eps_1 is order 1)
  3.06 |  RDP, minimised over alpha
  2.25 |  zCDP at rho chosen to match sigma (illustrative, same run)
```

The zCDP row is computed by matching rho so that the RDP and zCDP statements describe the same noise,
which is why it lands below RDP: the two conversions are not comparable across different rho. The
lesson is that a number without its accountant is not a number.

**6.5 The k-anonymity lattice.** Equivalence classes as rectangles in QI space, with cell height equal
to class size. The property k bounds the *smallest* rectangle, and says nothing about the distribution
inside it.

```text
  age band
   90+ |  ##  ####  ########
   60  |  ####  ######  ####
   30  |  ######  ########  ##
    0  +-------------------------- zip prefix
        (each # is one record; smallest block is 2 records -> k = 2)
```

The lattice makes two things visible at once. A table can be k = 5 everywhere and still have a class
whose sensitive attribute is 80/20, and it can be k = 2 with a class that is perfectly mixed. The
property is about the block size, not the contents, which is exactly the confusion in CC-01.

**Takeaway.** Five pictures, five different questions: noise against signal for utility, the clipping
ball for bias, epsilon against sigma for planning, the ladder for accounting, the lattice for
de-identification. If a review is stuck, ask which picture is missing.

## 7. Comparison / Selection Table

The tables below select a mechanism, an accountant, and a de-identification method from the stated
constraint. Each row is a decision you can defend in a review; the last column names the record that
fires when the selection is made for the wrong reason.

**7.1 Mechanism selection by trust model.** The trust model is the only thing that determines which
family is admissible. Accuracy and convenience are tie-breakers after it.

| Trust assumption | Admissible mechanism | Utility cost | When it is wrong | Record |
|---|---|---|---|---|
| Server trusted, raw records acceptable | central DP on the aggregate or the model | lowest | the server is honest-but-curious about updates, not records | CC-05 |
| Server not trusted with individual updates, collusion below threshold | secure aggregation plus local clipping and noise | payload overhead 0.53% at n=1200 | the server also sees metadata and can link rounds | SF-06 |
| No server trust at all, including the sum | local DP per record | sd ratio sqrt(n), 70.7x at n=5000 | the model needs the signal from 60k records equally weighted | CC-04 |
| Untrusted server, trusted shuffler without identity access | shuffle model | between local and central | the shuffler is operated by the same party that sees identities | SF-07 |
| Multiple independent analysts querying one dataset | central DP with a budget per analyst and a query ledger | grows with query count | the ledger is per-system rather than per-dataset | NE-09 |

**7.2 Accountant selection by regime.** Pick by T and by the size of the per-step epsilon, not by
reputation. The third column is the number this sheet's reference run produces, so a wrong pick is
visible immediately.

| Regime | Use | Value on the reference run | Why not the alternative | Record |
|---|---|---|---|---|
| T = 1, one release, epsilon < 1 | C1 then verify with C2 | 4.8458 sigma | C2 alone needs a numeric inversion | NE-01 |
| T = 1, epsilon >= 10 | C2, never C1 | 0.3442 exact vs 0.3028 classic | C1 is anti-conservative above ~9 | NE-01 |
| T large, epsilon_1 small (<=0.05) | advanced composition or zCDP | not this run | both are tight only in this regime | NE-03 |
| T large, epsilon_1 order 1 | RDP minimised over alpha | 3.0551 | basic gives 480.85, advanced 883.37 | NE-02 |
| Any T, implementation simplicity valued | zCDP | 2.25-5.30 depending on rho | 1-2 orders looser than RDP when T is large | NE-03 |
| Fixed-size sampling without replacement | a scheme-specific accountant | not computed here | C3 is stated for Poisson sampling | NE-04 |

**7.3 De-identification selection by what must be released.** The right-hand column is the only honest
statement the method supports, which is what a reviewer should demand.

| Release requirement | Method | Supports this claim | Does not support | Record |
|---|---|---|---|---|
| Row-level microdata, published | k-anonymity with k >= 5 plus suppression | within-table ambiguity of at least k | any bound on an attacker with external data | CC-01 |
| Row-level, sensitive attribute categorical | l-diversity with l >= 2 | at least l distinct values per class | protection when the split is 80/20 | CC-02 |
| Row-level, distribution matters | t-closeness with t <= 0.2 | class distribution within TV 0.2 of global | protection of outliers within the class | CC-02 |
| Aggregates for external analysts | central DP per query family with a ledger | (epsilon, delta) per family per dataset | protection across releases without ledger discipline | NE-09 |
| Synthetic training data | DP synthetic with epsilon on the generator | DP of the generator, not of the samples | per-record protection for a record that was memorised | CC-06 |
| Model weights for a partner | DP training plus clipping | (epsilon, delta) for the training population | protection of the partner's queries against the model | CC-05 |

**7.4 Failure-lane selection by symptom.** This is the table a person under time pressure should read
first: symptom on the left, lane in the middle, first check on the right.

| Symptom as reported | Lane | First check | Section |
|---|---|---|---|
| "accuracy is 3 points below central" | clipping bias vs noise | per-step clip fraction (C8) | 8, NE-06 |
| "the accountant says 3.05, legal signed 1.0" | accounting target | recompute sigma for the target (C4) | 8, NE-02 |
| "the release is k-anonymous so it is safe" | semantic | TV distance of the worst class (CC-02) | 8, CC-01 |
| "the round hangs once a day" | structural | dropout distribution against threshold (C14) | 8, SF-03 |
| "the attack shows AUC 0.62 so we leak" | measurement | interval on 60 records (Section 4.4, L2) | 8, CC-03 |
| "synthetic data is private because it is fake" | semantic | nearest-neighbour distance to training rows | 8, CC-06 |

**Selection rule.** Choose the row whose *when it is wrong* condition you can actually test. If you
cannot test any of them, you are not selecting a mechanism; you are picking a name to put in a
document, and the record that fires will be the one for the trust assumption you did not write down.

**Takeaway.** Selection tables earn their space only if the "when it is wrong" column is as specific as
the "when it is right" column; everything else is a menu.

## 8. Failure Modes & Edge Cases

The catalogue. Thirty records, each with a stable ID, a category, a symptom as it is actually
reported, at least two competing causes with different fixes, one discriminating check, a correction
with a bound, a prevention, and a verification. Categories are VIOLATED_ASSUMPTION (VA),
NUMERICAL_EDGE (NE), SCALE_FAILURE (SF) and CONCEPT_CONFUSION (CC).

**Reading the table.** The symptom column is deliberately written in the language a colleague would
use, including the wrong diagnosis they usually attach to it. The discriminating check is the smallest
action that separates the competing causes; run it before the correction, because three of the
corrections below are irreversible for a run.

| ID | Category | Symptom | Competing causes | Discriminating check |
|---|---|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | Per-client dashboard accuracy disagrees with the deployed model by ~1.5 pp | unweighted averaging of client metrics; cohort composition drift; eval set differs from training cohort | recompute both averages from the same per-client table with n_i weights (C13) |
| CI-02 | VIOLATED_ASSUMPTION | Per-record epsilon is satisfied but a client's four records leak | group privacy ignored; records correlated within client; the unit of privacy was set to row but the boundary is client | compute k·epsilon and the group delta (C11) for the largest client |
| CI-03 | NUMERICAL_EDGE | Two releases each at epsilon 0.6 are reported as epsilon 0.6 | composition skipped; budgets kept in separate ledgers; the second release reuses first-release aggregates | list both releases with population and event-time window and sum |
| CI-04 | CONCEPT_CONFUSION | "The model is differentially private" and the release also includes a per-user histogram | DP claim covers the model, not the companion artifact; ledger does not include the histogram | enumerate released artifacts and ask which budget each spent |
| CI-05 | NUMERICAL_EDGE | RDP budget reported as 13,568 and the team concludes DP is infeasible | epsilon read at alpha=64 instead of the minimising alpha; RDP units reported as epsilon; conversion term omitted | recompute argmin over alpha (C4) and show the curve |
| CI-06 | NUMERICAL_EDGE | Accuracy 3 pp below the central baseline immediately after enabling DP | clipping bias; noise variance; reduced learning rate to compensate | per-step clip fraction over the first 200 steps (C8) |
| CI-07 | NUMERICAL_EDGE | Loss plateaus at a value central training never reaches, gradient norm pinned near C | correlated outliers concentrating bias in one subspace; learning rate too low; noise too high | distribution of per-record norms against C at step 100 and 500 |
| CI-08 | NUMERICAL_EDGE | Clipping bias appears only in the first 50 steps of a warmup schedule | norm distribution non-stationary during warmup; C chosen from steady state; LR warmup interacts | per-step median norm for the first 100 steps |
| CI-09 | SCALE_FAILURE | Secure aggregation round never unmasks at 100k clients | quadratic share traffic; threshold above survivors; key exchange timeout | compare share bytes to payload bytes at the actual cohort size |
| CI-10 | SCALE_FAILURE | Round time doubles when the cohort doubles, though per-client time is unchanged | threshold straggler tail; sequential key distribution; server-side unmask cost | measure round time at n=600, 1200, 2400 with the same clients |
| CI-11 | SCALE_FAILURE | One slow client stalls every round; retries do not help | threshold includes the straggler; client is scheduled every round; retry re-enters the same tail | participation histogram per client id and the round's completion order |
| CI-12 | NUMERICAL_EDGE | Masked sums are off by small amounts and grow with round count | integer overflow at the fixed-point scale; quantisation step; clients dropping after masking | bit-width of the accumulator against n·C·scale (C16) |
| CI-13 | NUMERICAL_EDGE | Unmasked sum is correct in staging and wrong in production at 3x the cohort | accumulator width sized for staging cohort; scale chosen per environment | recompute required bits from n, C and scale in the production config |
| CI-14 | CONCEPT_CONFUSION | A membership attack at AUC 0.62 on 60 records is reported as a leak | sampling noise; no interval; unbalanced membership set | recompute the interval for n=60 (Section 4.4, L2) |
| CI-15 | CONCEPT_CONFUSION | Attack AUC is 0.5 but the model memorised a rare record | aggregate AUC hides per-record tail; canary not in the evaluation sample | per-record loss rank for a known canary |
| CI-16 | CONCEPT_CONFUSION | Team replaces DP with "we removed the names" | quasi-identifiers untouched; k-anonymity assumed to equal DP; linkage attack not modelled | count unique QI combinations in the released table |
| CI-17 | CONCEPT_CONFUSION | k = 5 release is claimed privacy-preserving and a linked record is re-identified | k bounds within-table ambiguity only; external dataset links; QI set incomplete | TV distance within the smallest class and a linkage test against a public file |
| CI-18 | CONCEPT_CONFUSION | l = 2 is claimed sufficient and the sensitive value is inferred at 0.8 | 80/20 class satisfies l=2; entropy not checked; t-closeness never computed | per-class frequency histogram of the sensitive attribute |
| CI-19 | CONCEPT_CONFUSION | Synthetic data is released without budget and a training row is reproduced | generator not DP; overfitting of the generator; mode collapse returns real rows | nearest-neighbour distance from each synthetic row to the training set |
| CI-20 | VIOLATED_ASSUMPTION | DP guarantee holds per round but the run's total is never stated | per-round accounting reported as the run total; composition left to the reader | read the artifact's accounting statement for a total |
| CI-21 | VIOLATED_ASSUMPTION | Budget spent on the same records twice across two releases | event-time window overlaps; population overlap not checked; ledger per release not per population | intersect the two populations on event time |
| CI-22 | NUMERICAL_EDGE | Sigma is quoted without C and nobody can reproduce the noise | sigma reported dimensionless while C changed between runs; C stored per-client | find C in the run record and recompute sigma·C |
| CI-23 | CONCEPT_CONFUSION | Local DP proposed to avoid trusting the server, and the model stops learning | sqrt(n) variance gap; cohort too small; epsilon unchanged | compute the local-versus-central sd ratio at the actual n (C9) |
| CI-24 | CONCEPT_CONFUSION | Secure aggregation is claimed to remove the need for clipping and noise | aggregation hides contributors, not content; malicious server can probe a single-client sum | check whether a single-client cohort is admissible in the protocol |
| CI-25 | SCALE_FAILURE | Federated run needs 4.35 TB of uplink and the network team refuses | dense fp32 upload; per-round cohort too large; no quantisation | recompute bytes from P and n and compare with the measured transfer |
| CI-26 | NUMERICAL_EDGE | Int8 quantisation is expected to give 4x and delivers 2.7% accuracy loss instead | quantisation error relative to DP noise; range chosen from the outlier; per-tensor scaling | quantisation sd against sigma·C/n per coordinate |
| CI-27 | NUMERICAL_EDGE | Quantised update passes the privacy audit but the model degrades only after many rounds | error accumulation across rounds; clipping interacts with rounding; no error feedback | per-round quantisation error norm and its sum over 50 rounds |
| CI-28 | VIOLATED_ASSUMPTION | Federated privacy claim assumes one record per user and enrolment shows 40 | unit of privacy inconsistent with the data model; enrolment deduplication missing | count records per user id at enrolment |
| CI-29 | SCALE_FAILURE | Metadata alone identifies participants even though updates are encrypted | participation logs, timing, round membership; encryption covers payload only | join the round logs with any external schedule |
| CI-30 | CONCEPT_CONFUSION | Delta is interpreted as "probability of a breach" and set to 1e-2 for 60,000 records | delta is failure mass, not breach probability; scale relative to 1/N ignored | compute 1/N and compare with delta |

**Detail blocks.** Each record below states the mechanism, the arithmetic where it exists, and the
check that separates the causes. They are ordered as in the table.

### CI-01 — the dashboard that averages wrong
The deployed model is trained on a weighted objective but the dashboard averages per-client
accuracies uniformly. With the three-client example of C13 the gap is 1.47 pp, and it moves with
cohort composition, so it changes between rounds without any model change. Competing causes: cohort
composition drift, which would show the same gap in a weighted average too; and an eval set that
differs from the training cohort, which would move both averages in the same direction. Discriminating
check: recompute the weighted and unweighted means from one per-client table. If they differ by
roughly the C13 magnitude, the dashboard is the bug. Correction: weight by n_i, and publish the weight
formula next to the metric. Prevention: a dashboard test that computes both and asserts they agree
within 0.2 pp when all clients have equal n. Verification: the deployed metric and the dashboard agree
within the stated tolerance, and the weight formula is in the metric definition.

### CI-02 — four records per client
The guarantee was written for rows, but a client uploads a gradient computed on four of its own
records, so the unit that can be distinguished is the client. Group privacy gives epsilon = 12.22 and
delta = 0.382 at k = 4, which is vacuous. Competing causes: correlated records *within* the client,
which produces the same symptom but is fixed by per-record clipping rather than by redefining the
unit; and a client that appears in more than one round, which compounds across rounds instead of
within them. Discriminating check: count records per user id and multiply rounds per client by records
per client to get the true k. Correction: define the unit as the client and clip the client-level
contribution to C, then account at that unit. Prevention: enrolment deduplication with a hard cap of
one contribution per user per round. Verification: the accounting statement names the unit and the
maximum k.

### CI-03 — two releases, one number
Each release spent epsilon 0.6 and both are reported as 0.6. The population overlaps, so the pair
spends 1.2. Competing causes: the releases genuinely target disjoint populations, in which case the
sum is right and the report is wrong anyway; and the second release reuses first-release aggregates,
which spends no new budget on the reused cells but does spend on the new ones. Discriminating check:
list both releases with population and event-time window and intersect. Correction: sum the budgets
and state the composition method. Prevention: a ledger keyed on population, not on release id.
Verification: the total in the ledger equals the sum of the per-release rows and covers the union.

### CI-04 — the model is private, the histogram is not
The DP claim covers the model artifact; the release also contains a per-user activity histogram that
was never accounted. Competing causes: the histogram was derived from the same DP model, in which case
it is post-processing and free; and the histogram was computed from raw records, in which case it
spends its own budget. Discriminating check: read the histogram's producer and see whether it touches
raw data. Correction: either derive the histogram from the DP output or account for it. Prevention:
an artifact manifest that names the budget each artifact spent. Verification: every released artifact
maps to a ledger row, and the ledger has no unmapped rows.

### CI-05 — the budget read at the wrong alpha
A team composed RDP and then read epsilon at the largest alpha in their grid, getting 13,568 and
concluding DP is unusable. The minimum over alpha is 3.055. Competing causes: a genuinely loose
mechanism, which would show up as a min over alpha that is also large; and the conversion term being
omitted, which produces a number that grows with alpha rather than shrinking. Discriminating check:
plot converted epsilon against alpha and look for a minimum. Correction: minimise; store the argmin
with the budget. Prevention: an accountant wrapper that returns (epsilon, alpha) together.
Verification: the stored alpha is the argmin on the stored grid.

### CI-06 — three points, three possible owners
Enabling DP costs 3 pp and everyone blames the noise. The three candidates have different signatures.
Noise variance raises the *variance* of the metric across seeds; clipping bias lowers its *mean*; and
a reduced learning rate chosen "to compensate" does both. Discriminating check: run the same seed with
and without noise at the same sigma and compare the metric against a run with noise but no clipping
(C = infinity is not implementable, but C = 1e6 approximates it). If the no-clip run recovers most of
the gap, the cost is clipping. Correction: raise C, then re-account; do not compensate with the
learning rate, which confounds the diagnosis permanently. Prevention: two ablation arms in the
onboarding run. Verification: the reported gap is decomposed into a bias term and a variance term with
seeds.

### CI-07 — the plateau at the clip surface
Per-record norms sit at C for a large fraction of records and the loss plateaus. If outliers are
correlated in a subspace, the clipped field's bias is a rotation, so the update can keep a healthy
norm while pointing away from the descent direction. Competing causes: a learning rate too low, which
shows a flat but *aligned* gradient (cosine similarity with the unclipped sum near 1); and noise too
high, which shows a gradient norm dominated by noise and uncorrelated with the true sum. Discriminating
check: cosine similarity between the clipped sum and the unclipped sum on the same batch. Below 0.7 is
bias; above 0.9 with a slow loss is the learning rate. Correction: raise C or per-coordinate clipping
in the dominant subspace. Prevention: log the cosine similarity every 50 steps. Verification: loss
resumes descending and the similarity stays above 0.9 for the rest of the run.

### CI-08 — warmup destroys the clip calibration
C was chosen from the median norm at step 1000. During the first 50 steps the median norm is 4.0, so
almost everything clips and the shrink factor sits near 0.2. Competing causes: a genuinely
fast-changing gradient distribution later in training, which would move gradually rather than drop by
an order of magnitude; and a bug in the norm computation over the first batch, which would show a
discontinuity at step 1 rather than a decay. Discriminating check: per-step median norm for the first
100 steps. Correction: raise C during warmup, or delay clipping until the norm distribution stabilises,
then account for the budget over the steps that actually clipped. Prevention: record the norm
distribution at every step during the calibration window and pick C at the quantile you intend.
Verification: the clip fraction at step 1 and at step 1000 lie within a factor of two.

### CI-09 — the round that never unmasks at 100k clients
Share traffic is quadratic: 100,000 × 99,999 × 32 B = 320.0 GB per round, 36.8 times the 8.70 GB
payload at 1,200 clients. Competing causes: a threshold above the survivor count, which fails
deterministically rather than slowly; and a key-exchange timeout, which fails with a timeout in the
log. Discriminating check: compare share bytes with payload bytes at the actual cohort size, then look
for key-exchange errors. Correction: use a tree or neighbour-only masking topology, which reduces the
pair count to O(n log n) or O(n) at the cost of more collusion risk. Prevention: a share-traffic
budget checked in the config review. Verification: share bytes are below the payload bytes at the
target cohort.

### CI-10 — round time grows superlinearly with cohort
Doubling the cohort doubles the payload but round time doubles plus a bit. Competing causes: the
straggler tail, which grows like the maximum of n samples, so a doubling moves the max by
sqrt(2·ln n)/... and is sublinear in expectation for exponential tails but visible; and server-side
unmask cost, which grows linearly with the number of masks. Discriminating check: measure round time at
n = 600, 1,200, 2,400 with the same client mix and decompose into upload, key exchange, and unmask.
Correction: cap the cohort and accept a smaller q, re-accounting since q appears in C6. Prevention:
round-time budget per phase in the config. Verification: the sum of phase times equals the measured
round time within 10%.

### CI-11 — one client, every round
A single slow client is selected every round and its 20-second round sets the pace. Competing causes:
a scheduling policy that favours recently-active clients, which concentrates participation; and a
threshold that includes the slowest survivor by being set at the cohort size. Discriminating check:
participation histogram per client id and the completion order of the last 10 rounds. Correction:
select without replacement across rounds and set the threshold below the cohort. Prevention: a
participation cap per client per night. Verification: no client appears in more than 1 + 10/n of the
rounds over a 500-round window, and the threshold is at most 90% of the sample.

### CI-12 — masked sums drift with round count
The fixed-point accumulator loses correctness. With scale 2^20 and C = 1, the sum over 1,200 clients
reaches 1,258,291,200, which needs 31 bits including sign, so a signed 32-bit accumulator has 1.71x
headroom — fine — but at scale 2^24 the same sum is 20,132,659,200 and needs 35 bits, which silently
overflows. Competing causes: quantisation error accumulating, which grows like sqrt(rounds) and is
small in magnitude; and clients dropping after masking, which produces a wrong sum of a different
shape (a whole missing contribution rather than a low-order error). Discriminating check: recompute
the required bit width from n, C, and the scale in the config. Correction: widen the accumulator or
reduce the scale. Prevention: a startup assertion on required bits. Verification: the assertion passes
at twice the production cohort.

### CI-13 — staging is fine, production is not
Same code, 3x the cohort, wrong sums. This is CI-12 with the config differing per environment.
Competing causes: a different client application build, which shows in the client version telemetry;
and a different scale constant loaded from an environment variable, which shows in the startup log if
it is printed. Discriminating check: print the accumulator width and scale at startup in both
environments. Correction: derive the width from n at runtime, not from a constant. Prevention: a
config that cannot express a cohort larger than the accumulator supports. Verification: the production
startup log states a width that satisfies the assertion at the production n.

### CI-14 — an attack on 60 records
AUC 0.62 on 30 members and 30 non-members is reported as leakage. At n = 30 per arm the standard
error of an AUC of 0.5 is 0.0752, so the 95% interval is [0.353, 0.647] and 0.62 is inside it.
Competing causes: a genuine signal, which would survive a larger sample; and an unbalanced membership
set, which biases the estimate rather than widening it. Discriminating check: recompute the interval
and, if it excludes 0.5, rerun on 10x the records. Correction: report the interval and the sample size
on the same line as the point estimate. Prevention: a minimum sample for any attack claim, set by the
effect size you care about. Verification: the reported interval is attached to the claim.

### CI-15 — the average that hides the canary
A canary record is memorised while the aggregate AUC is 0.5. Competing causes: the canary is not in
the evaluation sample, which shows as a zero count in the per-record table; and the memorisation is
real but confined to a handful of records, which is exactly what the aggregate hides. Discriminating
check: rank all records by the attack's confidence and inspect the top 10 by hand. Correction: report
a tail metric (recall at 1% FPR) alongside the AUC. Prevention: canaries planted before training and
listed in the evaluation manifest. Verification: the canary's rank is reported and its loss is
compared with the nearest non-member.

### CI-16 — names removed is not de-identification
Dropping name fields leaves quasi-identifiers unique. With 60,000 records and 30,000 QI cells the
expected number of singletons is 60,000 × e^(-2) = 8,120, or 13.5% of the table; at 120,000 cells it is
60,000 × e^(-0.5) = 36,392, or 60.7%. Competing causes: the QI set is incomplete, which undercounts
uniqueness and looks *better* than reality; and the population is actually much larger than the release
sample, which reduces uniqueness relative to the sample. Discriminating check: count distinct QI
combinations in the released table and compute the singleton share. Correction: generalise the QI set
until k meets the target, then re-count. Prevention: publish the uniqueness count with the release.
Verification: the singleton share is under the stated threshold and the QI set is enumerated in the
release notes.

### CI-17 — k = 5 and still re-identified
Every class has at least 5 rows, and a linked public file re-identifies one of them. Competing causes:
the link is through a field that was not treated as a QI, which shows in the released schema; and the
class's 5 rows are all the same person's visits, so the effective k is 1. Discriminating check: TV
distance within the smallest class plus a linkage test against the public file. Correction: widen the
QI generalisation and deduplicate by person before computing k. Prevention: k computed per person, not
per row. Verification: a linkage attempt on a held-out public file yields no unique match at the
stated threshold.

### CI-18 — l = 2 that is 80/20
The class has two distinct sensitive values, satisfying l = 2, and the attacker's posterior on the
common one is 0.8. Entropy of an 80/20 split is 0.722 bits. Competing causes: the attacker has a prior
stronger than the table's distribution, which is a modelling assumption you can state; and the class
mix is not what the release says, which is a data bug. Discriminating check: per-class frequency
histogram of the sensitive attribute, and the global distribution for comparison. Correction: move to
t-closeness with t <= 0.2, or suppress the class. Prevention: report both entropy and TV per class.
Verification: the worst class has TV <= t and the report shows both statistics.

### CI-19 — synthetic rows that are real rows
A generator without DP reproduces training rows, and the nearest-neighbour distance for the closest
synthetic row is 0. Competing causes: mode collapse in a DP generator, which produces near-duplicates
that are still noisy; and a copy of the input file in the pipeline, which produces exact matches.
Discriminating check: nearest-neighbour distance distribution from synthetic to training, and an exact
hash comparison. Correction: if exact matches exist, the pipeline is leaking; if only near-duplicates,
the generator needs a lower epsilon or a rejection rule. Prevention: a distance screen on every
synthetic release. Verification: the minimum distance exceeds a stated threshold or the matches are
disclosed and justified.

### CI-20 — the per-round number reported as the total
The artifact says "trained with DP at epsilon 1.1" where 1.1 is the per-round value and the run is 500
rounds. Competing causes: the total genuinely is 1.1 because only one round had data; and the
composition was done but the wrong number was copied into the document. Discriminating check: read the
artifact's accounting statement for a total and the round count. Correction: publish the total with T
and the accountant. Prevention: the accounting statement template has a mandatory total field.
Verification: the total equals the composition of the per-round rows under the stated accountant.

### CI-21 — the same records, twice
Two releases spend budget on overlapping populations. Competing causes: the overlap is in identity but
not in event time, in which case the records differ and the budgets do not compose on the same
individuals; and the overlap is real and the second release was derived from the first's outputs, in
which case it is post-processing. Discriminating check: intersect the populations on event time.
Correction: sum, or state post-processing with the derivation chain. Prevention: a ledger keyed on
(population, event-time window). Verification: the ledger has one row per (population, window) and the
total is the sum.

### CI-22 — sigma without C
Someone quotes sigma = 1.1 and nobody can reproduce the noise because C moved from 1.0 to 4.0 between
runs, making the realised noise 4.4 instead of 1.1. Competing causes: the noise is genuinely different
because sigma changed, which shows in the run record; and the mechanism added noise at a different
place (before averaging vs after), which changes the units. Discriminating check: find C in the run
record and recompute sigma·C per coordinate, then compare with the measured update variance.
Correction: report (sigma, C) together and the realised sd. Prevention: one config field that holds
both and one log line that prints both. Verification: the measured update variance matches sigma²C²/n
within the sampling error.

### CI-23 — local DP for a 5,000-record cohort
Local DP at epsilon = 1 needs 70.7 times the records to match central accuracy for a mean. At n = 5,000
the sd ratio is exactly sqrt(5000) = 70.71. Competing causes: the cohort is genuinely too small, which
shows the same ratio and is fixed by growth; and epsilon is being applied per record while the analyst
needs a per-query budget, which changes the comparison entirely. Discriminating check: compute the
ratio at the actual n and compare the achieved CI width with the central baseline. Correction: use the
shuffle model or central DP with a trusted aggregator. Prevention: an accuracy requirement expressed as
a CI width, checked against the mechanism's variance before the run. Verification: the released
statistic's CI width meets the requirement at the stated n and epsilon.

### CI-24 — secure aggregation as a substitute for noise
Updates are encrypted and summed, and the team drops clipping and noise. Aggregation hides
contributors; a server that can choose the cohort can isolate one client's contribution. Competing
causes: the protocol genuinely enforces a threshold above 1, which reduces but does not remove the
problem; and the server can also see individual unmasked updates through a retry path, which is a
worse and separate bug. Discriminating check: can a cohort of size 1 complete a round in the
implementation? Correction: keep clipping and noise at the client, and require threshold >= 2 with
cohort >= 3. Prevention: a protocol-level minimum cohort. Verification: a single-client round is
rejected by the server and by the client.

### CI-25 — 4.35 TB and a refusal
The network team will not fund 4.35 TB of uplink. Competing causes: the payload is dense fp32, which
quantisation fixes by 2x or 4x; and the cohort is larger than needed, which fewer rounds or a smaller
q fixes at a privacy cost. Discriminating check: recompute bytes from P and n and compare with the
measured transfer; if they match, the model is the problem, not the network. Correction: fp16 upload
with a compensated accumulator, then int8 if the quantisation sd stays below 5% of the DP noise
(4.94% at range 0.04, 24.70% at range 0.2). Prevention: a bytes-per-round budget in the design
review. Verification: measured transfer matches the recomputed figure within 5%.

### CI-26 — int8 that is expected to be free
Quantisation to int8 is expected to be a pure bandwidth win and costs accuracy. At range 0.04 the
quantisation sd per coordinate is 4.53e-05 against a DP noise sd of 9.17e-04, or 4.94%; at range 0.2
it is 2.26e-04, or 24.70%. Competing causes: the DP noise is larger than assumed, which would make the
ratio smaller and the loss invisible; and error accumulates over rounds rather than appearing at once,
which is CI-27. Discriminating check: quantisation sd against sigma·C/n per coordinate at the shipped
range. Correction: choose the range from a quantile of the update distribution, not from the maximum.
Prevention: report the ratio as a config invariant. Verification: the ratio is below 5% and the
per-round metric loss is inside its interval.

### CI-27 — the slow degradation
The quantised run is fine for 50 rounds and then drifts. Competing causes: error feedback missing, so
quantisation errors accumulate as a random walk whose variance grows linearly in rounds; and the
clipping norm interacting with rounding, which biases small updates to zero and shows as an increasing
fraction of zero updates. Discriminating check: per-round quantisation error norm, and the share of
coordinates that round to zero. Correction: add error feedback, which makes the error a martingale
rather than a drift. Prevention: track the error norm as a first-class metric. Verification: the
error norm stays bounded over 500 rounds and the zero-share is stable.

### CI-28 — 40 records per user
The privacy claim says record-level and enrolment shows a mean of 40 records per user. Competing
causes: the claim is really user-level and the document is wrong, which is a documentation fix; and the
data model has duplicate enrolments, which is a deduplication fix that also changes the cohort. Note
that users here are modelled as holding many records, which is the tabular analogue of a device with
many sessions. Discriminating check: count records per user id at enrolment and inspect the top
decile. Correction: re-denominate the guarantee in the true unit and re-account. Prevention: the unit
of privacy is a schema field, not prose. Verification: the schema field matches the accounting
statement and the enrolment cap is enforced.

### CI-29 — the metadata that was never encrypted
Updates are encrypted and memberships are logged with timestamps. Competing causes: the log is
aggregate-only, which shows no per-client rows; and the log is per-client but access-controlled, which
is a policy control rather than a privacy guarantee. Discriminating check: join the round logs against
any external schedule (a shift roster, a device heartbeat) and see whether participation is
predictable. Correction: reduce log granularity, add noise to participation counts, or drop per-round
membership. Prevention: metadata is in the threat model, not outside it. Verification: an attempted
join on a held-out external file produces no more information than the published participation
statistics.

### CI-30 — delta as a breach probability
delta = 1e-2 is justified as "a 1% chance of a breach" for a 60,000-record cohort. delta is the mass
of the output region where the epsilon bound may fail; 1e-2 is 600 records' worth of that mass at
1/N = 1.67e-5. Competing causes: epsilon is small enough that the failure region carries little
information, which is a valid argument but requires the joint statement; and the dataset is genuinely
small enough that 1e-2 is negligible relative to 1/N, which it is not here. Discriminating check:
compute 1/N and compare with delta. Correction: set delta <= 1/N or set it to 1e-6 and re-account,
which costs epsilon. Prevention: a rule that delta must be at most 1/N and at most 1e-5. Verification:
the accounting statement shows delta and N on the same line.

**Takeaway.** Thirty records, four categories, and one shared shape: the symptom is a number or a
sentence that is true under an assumption nobody wrote down, and the check is what makes the
assumption visible.

## 9. Complexity & Cost Analysis

Cost in this topic has five terms, and a proposal that prices only one of them will be surprised by
the other four: compute on the client, communication on the uplink, cryptography in the aggregator,
statistical cost in accuracy, and the budget itself, which is a finite resource that is spent whether
or not the run is useful.

**9.1 Compute scales linearly in records and not at all in privacy.** Per-sample forward plus backward
is about 6P FLOPs, so a minibatch of 32 costs 6 × 1,812,000 × 32 = 347.9 MFLOP and a round of 250 steps
costs 87.0 GFLOP. The whole run is 500 rounds × 250 steps × 32 samples × 6P = 43.49 TFLOP of client
compute, spread across 12,000 client-rounds.

```text
per sample            6 * 1,812,000          = 10,872,000 FLOP  = 10.87 MFLOP
per step (batch 32)   32 * 10.87 MFLOP       = 347.9 MFLOP
per client round      250 * 347.9 MFLOP      = 86.98 GFLOP
total (12,000 rounds) 12,000 * 86.98 GFLOP   = 1.044 PFLOP
check as TFLOP        500*250*32*6*P/1e12    = 43.49 TFLOP of samples
```

The two totals differ by the factor of 1,200 clients per round and are both stated because they answer
different questions: 43.49 TFLOP is the work, 1.044 PFLOP is the aggregate device work when every
client runs its own copy. At an effective 50 GFLOPS on a mid-range phone one client-round takes
1.740 s; at 6 GFLOPS, the realistic figure for a phone in a low-power state across three layers of
interpreter and framework overhead, it takes 14.50 s. That 8.3x gap between nominal and effective is
the single most common surprise in federated cost models.

**9.2 Communication dominates at production scale.**

```text
payload per round, fp32  8.70 GB      (C15)
payload per round, fp16  4.35 GB
payload per round, int8  2.17 GB
mask shares per round    46.04 MB     (0.53% of fp32 payload at n=1200)
mask shares at n=100k    320.0 GB     (36.8x the fp32 payload)
HE ciphertext set        94.52 MiB    (13.04x the fp32 payload, poly 8192)
```

The ordering matters. At n = 1,200 masking is free and encryption costs a factor of 13. At n = 100,000
masking alone is 37 times the payload, which is why production systems do not use all-pairs masking.
The HE row is for a CKKS-style packing of P values into 222 ciphertexts of 436 KiB each; it is
stated to make the point that encrypted aggregation of a dense update costs an order of magnitude in
bytes before it costs anything in compute, so it is viable for sparse or low-dimensional updates and
not for P = 1.8M dense.

**9.3 The statistical cost is the only one that does not shrink with engineering effort.** From C8, the
shrink factor at a median norm of 1.2214 is 0.7715, a 22.85% loss of update magnitude. No amount of
bandwidth or compute recovers it. The only levers are C, the norm distribution, and per-coordinate
clipping.

| Cost term | Scales with | Lever | Lower bound | Recoverable? |
|---|---|---|---|---|
| Compute | records × steps × P | fewer local steps | signal per round | yes, at the cost of rounds |
| Uplink | cohort × P × dtype width | fp16, int8 | 2 bits/param at int8 without loss | yes, to 4x |
| Masking | cohort² | tree or neighbour topology | O(n) pairs | yes, at collusion risk |
| Encryption | P / slots | sparsity, packing | 13x payload | partly |
| Clipping bias | norm distribution vs C | raise C, per-coordinate clip | 0 at C = inf, which is not private | no, only trade |
| Noise | sigma × C / n | larger n, smaller T | budget | only by changing the guarantee |

**9.4 The budget as a resource.** The run has 500 rounds because 500 rounds was the training plan. If
the target is epsilon = 1.0 and the sigma that meets it is 2.401, the noise triples relative to the
shipped configuration in variance terms (2.401²/1.1² = 4.76), and the cumulative SNR falls from 48.79
to 22.35. The alternative is fewer rounds: at T = 250 the sigma for epsilon = 1.0 is 1.845, which is
only 1.68x the shipped noise and keeps more of the signal. This is the cost analysis that actually
matters — the budget is not a constraint on the run but a variable that selects the run's length.

```text
configuration        sigma for eps=1.0   noise variance vs shipped
T=500, q=0.02        2.401               4.76x
T=250, q=0.02        1.845               2.81x
```

**9.5 Cost of getting the accounting wrong.** Using basic composition where RDP applies reports 480.85
instead of 3.055. The direct cost is zero — no compute is spent on accounting — and the indirect cost
is the whole project, because a number 157 times too large either kills a viable proposal or forces a
sigma so high that the model does not learn. This is the only entry in this section where the cheapest
option (spend an engineer-week on the accountant) has the highest return.

**Takeaway.** Price all five terms; the two that are usually omitted, clipping bias and the budget's
effect on run length, are the two that decide whether the project is worth doing.

## 10. Key Theorems / Results Reference

Definitions and results are listed with the conditions that make them usable, because a theorem quoted
without its conditions is a slogan. Each entry names a pointer for the reader who needs the proof; none
of the pointers is evidence about any deployment.

**R1 — Gaussian mechanism (classical calibration).** For a function with L2 sensitivity Delta, adding
N(0, sigma²Delta²) gives (epsilon, delta)-DP when sigma >= sqrt(2 ln(1.25/delta))/epsilon. Conditions:
a single release, and the constants are a sufficient condition fitted for small epsilon. See C1 and
the counterexample in L1: it fails above epsilon ~ 9 at delta = 1e-5. Pointer: Dwork and Roth,
*The Algorithmic Foundations of Differential Privacy*, the Gaussian mechanism section.

**R2 — Exact Gaussian trade-off.** The exact delta for a Gaussian mechanism follows from the
Gaussian CDF difference in C2 (Balle and Wang's analytic calibration). Condition: one Gaussian release
with known L2 sensitivity. It is the right tool for inverting sigma, and it is exactly as wrong as C1
when applied to a composed run.

**R3 — Basic composition.** The sum of (epsilon_i, delta_i) is a valid guarantee for the sequence.
Condition: none beyond each step being valid. Consequence: linear in T, therefore useless for long
runs, which is why C5 exists as a lower bound to compare against.

**R4 — Advanced composition.** For T-fold adaptive composition, epsilon_total is
sqrt(2T ln(1/delta'))·epsilon + T·epsilon·(e^epsilon - 1) with delta_total = delta' + T·delta.
Condition: T large and epsilon small; degenerate otherwise, as the 883.37 figure in C7 shows.

**R5 — Rényi DP and the subsampled Gaussian.** RDP composes additively in alpha, and the subsampled
Gaussian mechanism has the closed-form RDP of C3 (Mironov, Talwar, Zhang). Conditions: Poisson
sampling for the closed form; a dense alpha grid for the conversion. Consequence: the conversion must
be minimised, which is CI-05.

**R6 — RDP to approximate DP.** epsilon = rho_RDP(alpha) + ln(1/delta)/(alpha - 1). Condition: any
alpha > 1, and the minimum over alpha is the reported value. This is the step that turns a
tensor of RDP numbers into one number, and omitting it is CI-05.

**R7 — zCDP composition and conversion.** rho composes additively; rho-zCDP implies
(rho + 2 sqrt(rho ln(1/delta)), delta)-DP. Condition: any rho > 0. It is the easiest accountant to
implement correctly and loses roughly a factor of two against RDP at this run's scale.

**R8 — Group privacy.** A guarantee at k = 1 implies (k·epsilon, k·e^((k-1)epsilon)·delta) at group
size k. Condition: none, and that is the problem: the delta term is vacuous quickly (0.382 at k = 4,
epsilon = 3.055). Consequence: the unit of privacy is a design decision, not a reporting choice.

**R9 — Membership advantage bound.** For any epsilon-DP mechanism the advantage in a balanced
membership game is at most tanh(epsilon/2). Condition: pure epsilon-DP; for approximate DP, add
delta. Consequence: at epsilon = 3.055 the bound is 0.910, which is the honest description of that
run's protection.

**R10 — Amplification by subsampling.** With Poisson sampling at rate q, an epsilon-DP mechanism
becomes roughly ln(1 + q(e^epsilon - 1))-DP per step. Condition: small q and a first-order expansion;
at q = 0.25 the approximation is optimistic relative to exact accounting. Consequence: cohort size is
a privacy parameter, not only a throughput parameter.

**R11 — Post-processing.** Any function of a DP output remains DP at the same parameters. Condition:
the function cannot access the raw data. Consequence: this is what makes a DP-trained model safe to
release, and it is also the claim most often overstated, because the *companion* artifacts (histograms,
per-user logs) are not post-processing of the model.

**R12 — Secure aggregation correctness and privacy.** A masking protocol with threshold t reveals only
the sum when at least t contributions survive and fewer than t parties collude. Conditions: the
threshold is enforced by the protocol, and the server cannot choose a cohort of size below t. This is
a systems theorem, not a statistical one, and CI-24 is what happens when it is mistaken for a privacy
mechanism.

**R13 — k-anonymity and its limits.** k-anonymity bounds within-table ambiguity to 1/k; it does not
bound an adversary's posterior with external data. Conditions: the QI set is complete and k is counted
per person. Consequence: the claim it supports is precisely the narrow one in the 7.3 table.

**R14 — t-closeness.** Bounding the total-variation distance between a class distribution and the
global distribution to t bounds the information gain from membership in that class to t. Conditions:
the global distribution is knowable by the attacker, which is the conservative assumption to make.
Consequence: t-closeness rules out the 80/20 class that l-diversity admits.

**Takeaway.** Fourteen results, and every one of them carries a condition that a summary slide
usually drops; the conditions are the content.

## 11. Interview / Exam Questions

Fifteen questions, mixing recall, calculation, and diagnosis. The answers are written to be checkable:
where a number appears, it is derivable from Sections 4 and 5.

**Q1. A colleague reports sigma = 0.30 for epsilon = 16, delta = 1e-5. What is wrong and what is the
consequence?**
**Answer.** They used C1, the classical calibration, outside its regime. The exact minimal sigma from
C2 is 0.3442, so 0.30 is 13.7% too little noise; the delta actually achieved at sigma = 0.30 is
3.36e-04, which is 33.6 times the 1e-5 target. **Reasoning.** C1's slack shrinks with epsilon and turns
negative above about epsilon = 9 at this delta; the mechanism then violates the claim it is reported
to satisfy. **Trap.** Verifying C1 against itself, which is always self-consistent. **Follow-up.** What
do you check in the shipped artifact? The realised noise scale, sigma times C, against C2 at the
reported epsilon.

**Q2. Your 500-round run at q = 0.02, sigma = 1.1, delta = 1e-5 reports epsilon = 3.055. Legal signed
for 1.0. What are the two honest options?**
**Answer.** Raise sigma to 2.401 and accept a cumulative SNR of 22.35 instead of 48.79, or cut the run
to 250 rounds and use sigma = 1.845, which keeps more signal at 2.81x the shipped noise variance.
**Reasoning.** The budget is a function of (q, sigma, T); fixing q and T leaves sigma as the only
lever, and halving T changes the required sigma more cheaply than keeping T and raising sigma.
**Trap.** Changing the accountant or the delta to make the number smaller. **Follow-up.** Which option
would you choose if the model needed 500 rounds to converge? Then the honest answer is that the
guarantee must be renegotiated, and the memo should say so.

**Q3. The dashboard says 0.8167 and the deployed model scores 0.8020 on the same clients. Which is
right and what is the bug?**
**Answer.** Neither is wrong; they answer different questions. 0.8167 is the unweighted mean over
three clients, 0.8020 is the n-weighted mean with weights 0.2, 0.1333, 0.6667. The model optimises the
weighted objective, so 0.8020 is the number that describes the model. **Reasoning.** C13: the gap is
1.47 pp and it moves with cohort composition, so it will drift between rounds. **Trap.** Concluding
the model is broken because the dashboard disagrees. **Follow-up.** What if all three clients had equal
n? Then the two agree exactly and the bug is invisible, which is why this survives staging.

**Q4. Two releases each report epsilon 0.6 and both cover the same 60,000 people for the same month.
What is the total, and what would change your answer?**
**Answer.** 1.2 under basic composition. It would change if the second release were computed purely
from the first release's DP outputs, in which case it is post-processing at no additional cost, or if
the populations were disjoint in event time. **Reasoning.** Composition is about the same records being
distinguishable twice. **Trap.** Summing epsilons for releases that do not overlap, which is
conservative but hides the real question of ledger keying. **Follow-up.** What must the ledger be keyed
on? (Population, event-time window), not release id.

**Q5. Compute the membership advantage bound for the reference run and say what it means in one
sentence.**
**Answer.** tanh(3.0551/2) = 0.910. It means that against an adversary designed to test membership, the
accounting permits a 0.910 advantage, so the run's protection is weak in the worst case even though it
is a legitimate DP claim. **Reasoning.** The bound is worst-case over distinguishers, which is the only
kind of statement a DP guarantee makes. **Trap.** Comparing 0.910 with a measured attack AUC and
concluding either agreement or violation; the two are different objects. **Follow-up.** At what
epsilon does the bound fall below 0.5? tanh(epsilon/2) = 0.5 at epsilon = 2·atanh(0.5) = 1.0986.

**Q6. A round fails intermittently. The threshold is 900, the cohort is 1,200, dropout is 25%. What is
the failure rate and what is the cheapest fix?**
**Answer.** Survivors are Binomial(1200, 0.75) with mean 900 and sd 15, so success is 0.5155 and the
failure rate is 48.45%. The cheapest fix is to lower the threshold to about 850, where success is
0.99954, provided the protocol's privacy argument tolerates it. **Reasoning.** C14; the threshold
should be set from the dropout distribution, not from the sample size. **Trap.** Raising the cohort,
which costs payload linearly and only moves the mean by 15 per 20 clients. **Follow-up.** Why is
0.99954 the relevant number rather than the mean? Because the failure is a whole-round stall and the
tail is what the operator experiences.

**Q7. Your quantisation is at int8 with range 0.2 and the DP noise is sigma·C/n = 9.17e-04 per
coordinate. Is the quantisation free?**
**Answer.** No. The step is 0.2/255 = 7.84e-04, so the quantisation sd is 7.84e-04/sqrt(12) = 2.26e-04,
which is 24.70% of the DP noise sd. At range 0.04 the sd is 4.53e-05, or 4.94%. **Reasoning.** The
range, not the bit width, sets the error; a range picked from the maximum update wastes most of the
codebook. **Trap.** Assuming int8 is a pure bandwidth win because the bit count fell by four.
**Follow-up.** What range would you choose? One that puts the 99th percentile of the update
distribution near full scale, then check the ratio against 5%.

**Q8. The accountant returns 13,568. What is the most likely bug and how do you confirm it in one
minute?**
**Answer.** epsilon was read at alpha = 64 instead of the minimising alpha, or the conversion term was
dropped. Confirm by tabulating converted epsilon against alpha and looking for the minimum; on the
reference run it sits at alpha = 7 with 3.0551, and at alpha = 64 the same run converts to 13,568.44.
**Reasoning.** C4; the conversion is a minimum, not a lookup. **Trap.** Believing the largest alpha is
the most conservative choice. **Follow-up.** What should the accountant return besides epsilon? The
argmin alpha and the grid it searched.

**Q9. Someone proposes local DP to avoid trusting the aggregator. The cohort is 5,000 records. What is
the accuracy consequence for a mean?**
**Answer.** The sd ratio is sqrt(5000) = 70.71 at the same epsilon, so local DP needs 70.71 times the
records or costs 70.71 times the budget for the same accuracy. **Reasoning.** C9; the gap is exactly
sqrt(n) for the Laplace mean. **Trap.** Comparing the two at different epsilons, which is the most
common way this comparison is made to look better than it is. **Follow-up.** What intermediate option
exists? The shuffle model, which recovers much of the gap at the cost of assuming the shuffler cannot
see identities.

**Q10. A k = 5 release is re-identified through a public file. Is k-anonymity broken?**
**Answer.** No; it was never a guarantee against external data. k bounds within-table ambiguity at
1/k. The failure is in the claim, not the method. **Reasoning.** R13 and the 7.3 table: k-anonymity
supports exactly one narrow statement. **Trap.** Responding by raising k, which changes the number and
not the class of attack. **Follow-up.** What would you add? t-closeness for the sensitive attribute,
plus a linkage test against the public file you can actually obtain.

**Q11. Clipping at C = 1 with a lognormal norm distribution of median 1.2214 and s = 0.5 loses what
fraction of the update, and how does that compare with the noise?**
**Answer.** The shrink is 0.7715, a 22.85% magnitude loss, and it is a bias rather than a variance, so
it does not average out over records. The noise at sigma = 1.1 contributes sd 9.17e-04 per coordinate
against a per-record gradient magnitude of 0.002, an SNR of 2.18. **Reasoning.** C8 for the shrink;
the two effects are different in kind and are routinely added together. **Trap.** Reporting the total
accuracy loss as "DP noise". **Follow-up.** Which is cheaper to fix at fixed epsilon? Clipping, because
raising C is free in accuracy terms until it forces a larger sigma through the sensitivity.

**Q12. Your masked sum is wrong by a small amount that grows with round count. Two candidate causes and
one check.**
**Answer.** Either the fixed-point accumulator overflows at the configured scale (check the required
bit width: 1,200 × 2^20 = 1,258,291,200 needs 31 bits, and 2^24 needs 35), or quantisation errors
accumulate without error feedback (check the per-round error norm). The check that separates them is
whether the error appears within one round or grows monotonically across rounds. **Reasoning.** An
overflow is a single-round catastrophic wrap with a distinctive sign flip; accumulation is a random
walk. **Trap.** Looking at the final sum instead of per-round error. **Follow-up.** What prevents the
overflow permanently? Deriving the accumulator width from cohort size and scale at startup with a
fail-closed assertion.

**Q13. You must cut 4.35 TB to under 1 TB without changing the privacy parameters. What are the options
in order of cost?**
**Answer.** fp16 uploads give 2.17 TB; int8 gives 1.09 TB but adds quantisation noise at 4.94% of the
DP noise at range 0.04 and 24.70% at range 0.2, so the range must be chosen from the update
distribution. Truncated or sparse updates can go further at the cost of a bias that must be accounted.
**Reasoning.** C15 gives the linear scaling in dtype width; nothing else in the payload term has that
property. **Trap.** Sparsifying to 1% and quoting 100x, which ignores 2 bytes of index per value.
**Follow-up.** What breaks first at int8? Small coordinates that round to zero, which is why the
zero-share is a monitored metric.

**Q14. A membership attack reports AUC 0.62 on 60 records. What do you do before telling anyone?**
**Answer.** Compute the interval. At 30 per arm an AUC of 0.5 has a standard error of 0.0752, so the
95% interval is [0.353, 0.647] and 0.62 is inside it. The claim needs either a much larger sample or
a per-record tail metric. **Reasoning.** Section 4.4 L2 and CI-14; an attack is a measurement with its
own uncertainty. **Trap.** Treating a point estimate above 0.5 as evidence of leakage. **Follow-up.**
What sample size would you need for a true AUC of 0.62 to be distinguishable from 0.5? With the same
SE formula, about 5.5 times the records, so roughly 400 in total.

**Q15. When is secure aggregation not enough, and what do you add?**
**Answer.** When the server can choose a cohort of size one, when it can observe per-round membership
and timing, or when the updates are unclipped. Add a protocol-level minimum cohort and threshold,
clip and noise at the client, and treat metadata as part of the threat model. **Reasoning.** R12 is a
systems theorem with a collusion condition; it says nothing about content, so content needs C8-style
clipping and a noise term. **Trap.** Reading "encrypted" as "private". **Follow-up.** Which check
proves the minimum cohort is enforced? An attempt to run a one-client round, which must be rejected by
both client and server.

**Takeaway.** The questions that separate candidates are Q2, Q9, Q11 and Q14, because each requires
choosing between two defensible-looking options and saying which one the arithmetic supports.

## 12. Quick Reference Card

| Signal as reported | Record | First check | Bounded correction |
|---|---|---|---|
| "3 pp below central baseline" | CI-06 | per-step clip fraction, first 200 steps | raise C, re-account, never retune LR |
| "accountant says 3.05, legal signed 1.0" | CI-20 | sigma needed for the target (C4) | sigma 2.401 or T = 250 at 1.845 |
| "budget is 13,568" | CI-05 | converted epsilon vs alpha, find the min | report 3.0551 at alpha = 7 |
| "k = 5 so it is safe" | CI-17 | TV distance in the smallest class | t-closeness at t <= 0.2 |
| "we removed the names" | CI-16 | distinct QI combinations, singleton share | generalise until the share is under target |
| "synthetic so it is private" | CI-19 | nearest-neighbour distance to training rows | DP generator plus distance screen |
| "round hangs once a day" | CI-11 | dropout vs threshold (C14) | threshold 850 for 0.99954 success |
| "AUC 0.62 means we leak" | CI-14 | interval at n = 60: [0.353, 0.647] | report interval, 10x the records |
| "encrypted so it is private" | CI-24 | can a one-client round complete? | threshold >= 2, cohort >= 3 |
| "int8 is free" | CI-26 | quant sd vs sigma·C/n: 4.94% or 24.70% | range from the 99th percentile |
| "4.35 TB, network refuses" | CI-25 | recompute P × n × 4 bytes | fp16 then int8 |

**Five numbers worth remembering.** 2.401 (sigma for epsilon 1.0 at q = 0.02, T = 500), 3.0551 (what
sigma 1.1 actually spends), 0.910 (membership advantage the run permits), 22.85% (update magnitude lost
to clipping at median norm 1.2214 with C = 1), and 4.35 TB (uplink for 500 rounds of a 1.8M-parameter
model at 1,200 clients).

**Three ratios worth remembering.** sqrt(n) = 70.71 (local versus central at n = 5,000), 157.4 (basic
composition versus RDP on the reference run), and 2.317 (tail round time over mean round time).

Nothing follows this card; use Section 8 for the full records and Section 11 for rehearsal.
