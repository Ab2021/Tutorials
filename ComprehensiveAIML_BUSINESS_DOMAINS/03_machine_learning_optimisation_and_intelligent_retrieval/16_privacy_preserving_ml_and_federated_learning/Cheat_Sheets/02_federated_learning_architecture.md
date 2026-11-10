---
title: "Federated learning architecture and secure aggregation"
topic: "Round topology, client sampling, aggregation protocols, threshold masking, and the cost structure of federated training"
collection: "Part 03 - Machine Learning, Optimisation and Intelligent Retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Federated learning architecture and secure aggregation

> **Evidence boundary.** Every client count, model size, latency, byte figure, and privacy parameter in
> this sheet is a CONSTRUCTED teaching value labelled ILLUSTRATIVE. The arithmetic is DERIVED and
> reproducible from the formulas in section 4. Published results are named in section 10 with their
> validity conditions; no deployment, vendor benchmark, accelerator measurement, or organisational
> outcome is asserted. Where a number came from a simulation, the sheet says so.

## 1. File Header

**Purpose of this sheet.** Let an engineer choose and defend a federated architecture: how many clients
participate per round, how those clients are grouped, what the server is allowed to see, how a round
survives clients disappearing, and what the whole thing costs in bandwidth, latency, energy, and privacy
budget. The sheet is written to be used while drawing the system, not after it ships.

**The decision this sheet settles.** Five coupled choices. The *topology*: flat client-to-server,
hierarchical with intermediate aggregators, or peer-to-peer. The *cohort*: how many clients per round and
how they are sampled. The *aggregation*: plain summation, threshold masking, or masking plus differential
privacy. The *synchrony*: synchronous rounds, bounded staleness, or asynchronous updates. The *client
update*: how many local steps before a client reports.

**Why these five and not fewer.** Each one changes the others. Moving to hierarchical aggregation changes
what the server sees, which changes whether secure aggregation is affordable, which changes the cohort
size, which changes the privacy budget. A design that fixes one without the other four is not a design.

**Who this is for.** Four readers. A platform engineer needs sections 5 and 9 for capacity. A privacy
engineer needs sections 4.3 and 8 to check the threshold and the noise. An ML engineer needs sections 4.2
and 8 to understand why local steps change convergence. An operations engineer needs section 8 to know
which alarm corresponds to which cause. Nobody should skip section 4.4.

**Prerequisites.** You must be able to read a weighted average, reason about a stationary variance of a
recursive filter, and understand modular arithmetic well enough to see why masks cancel. You must also
accept the framing that every model update is a *query against the client's data*, which is what makes the
privacy argument necessary rather than decorative.

**Units used throughout.** Model size in megabytes of parameters at a declared precision. Bandwidth in
megabits per second and bytes in decimal units. Latency in seconds per round. Energy in joules per client
round, then kilowatt hours. Privacy parameters are dimensionless. Client counts are counts of distinct
devices or distinct organisations, and the sheet always says which.

**Notation table.**

| Symbol | Meaning | Unit / type | Valid range |
|---|---|---|---|
| `C` | clients selected per round | count | `C >= 2` in practice; `C >= 2` for a sum |
| `N` | eligible population size | count | `N >= C` |
| `q` | sampling rate `C/N` | dimensionless | `0 < q <= 1` |
| `T` | number of rounds | count | `T >= 1` |
| `M` | model size per round in bytes | bytes | `M >= 1` |
| `P` | parameter count | count | `P >= 1` |
| `bpw` | bytes per parameter on the wire | bytes | `1/8 <= bpw <= 4` in practice |
| `E` | local steps per round | count | `E >= 1` |
| `eta` | server or client step size | dimensionless | `0 < eta < 1` for the stability analysis |
| `rho` | retention factor per round, `(1-eta)^E` | dimensionless in (0,1) | `0 < rho < 1` |
| `t_thr` | masking threshold, clients whose seeds are needed | count | `2 <= t_thr <= C` |
| `sig` | DP noise multiplier relative to the clip norm | dimensionless | `sig > 0` |
| `a` | staleness in rounds | count | `a >= 0` |

**Outcome if the sheet is applied correctly.** You can state the per-round payload, the expected round
duration, the wall-clock time to full population coverage, the DP epsilon from a named accountant, the
server's view under the chosen masking rule, and the single condition that would break each of those.

**What a weak architecture looks like.** Five recurring shapes. A flat topology where every client ships
a full-precision model to a single server that could also read raw data, so nothing cryptographic was
gained. A cohort chosen as "whoever is awake", which silently biases the model toward always-charged
devices. A masking protocol with a dropout threshold that the observed dropout rate exceeds, so rounds
abort silently. A DP claim with no amplification argument, which is off by a factor of forty-five in the
noise actually needed. A local-step count picked for latency reasons that triples the sampling noise.

**Exclusions.** This sheet does not cover differential privacy as a mechanism family, which is sheet 01,
nor data minimisation and retention, which is sheet 03. It does not cover model design, feature
engineering, or the training objective. It treats secure aggregation as a protocol with an invariant and a
threshold, not as a cryptographic construction in its own right.

**How to use it under time pressure.** Section 7 selects the topology. Section 4.3 verifies that the
server's view is what you think it is. Section 8 is the pre-launch check. Section 9 is the capacity
conversation. Section 11 tests whether the team has understood why the round and the query are the same
object.

**The naming caution.** "Federated" describes where the data sits. "Secure aggregation" describes what
the server can compute. "Differentially private" describes what an adversary can infer. These are three
independent properties, and a system can have any subset of them. Most systems that claim all three have
one and assume the other two.

## 2. Concept Map / Mental Model

**The round as the unit of everything.** Every cost, every privacy guarantee, and every failure in
federated learning is per round. A round has six stages, and the slowest stage decides the duration.

```text
[server]  select C clients from N            -> cohort
              |
     broadcast current weights W_t (M bytes to each)
              |
[client]  local steps: W <- W - eta * grad, E times on local shard
              |
     clip update to norm S, quantise to bpw bytes, add masks
              |
     upload M bytes to server (or to an aggregator)
              |
[server]  verify threshold t_thr, sum, unmask, optionally add DP noise
              |
     apply update: W_t+1 = W_t + aggregate
              |
     repeat for T rounds
```

**Reading the loop as a query pipeline.** The client's update is a function of the client's own rows, and
the server receives it. That is a disclosure whether or not the rows moved. The masking stage changes
*who* can read the update; the noise stage changes *what* can be inferred from the sum. Neither is
optional if the threat model includes the server.

**Three trust boundaries.** Boundary one, between client and aggregator: masking and transport security
apply. Boundary two, between aggregator and server: hierarchical aggregation means the server sees a
partial sum rather than individual contributions, which is a weaker but real protection. Boundary three,
between the released model and the world: only differential privacy addresses this one, and no masking
protocol does.

**The mental model that survives review.** A federated round is a distributed optimisation step with
three unusual properties. The data distribution seen by each worker changes between rounds because the
cohort changes. The workers are unreliable by default rather than exceptionally. And every message is an
information release that must be accounted. Classical distributed training shares the first property
weakly, the second not at all, and the third never.

**Sampling is part of the algorithm.** The cohort is not "the set of clients who happened to respond".
It is a random variable with a distribution, and that distribution enters the estimator. A uniform draw
without replacement gives an unbiased estimate of the *client-average* objective; the population-weighted
objective needs weights; and a draw that is uniform over clients gives a biased estimate of the
*record-weighted* objective whenever client shard sizes differ. Section 4.1 turns this into arithmetic.

**Synchrony is a budget, not a preference.** A synchronous round waits for the slowest participating
client, so round duration is a maximum, not a mean. That is why an architecture with a 45-second median
client takes 280 seconds per round at 2,000 clients, and why the same system at 100 clients takes 196
seconds. Adding clients costs latency superlinearly in the tail.

**Coverage is a different question from speed.** A round touches `C` of `N` clients. Full coverage needs
`T` such that `(1-q)^T` is small, and the answer is almost always large: at `q = 4e-5`, touching ninety
percent of the population takes 57,563 rounds. Federated systems are therefore trained on a persistent
subset for a long time before the rest sees the model at all.

**Where the model stops being a model.** The pipeline assumes each client's update is a bounded vector
and the aggregate is a sum. That is true for dense weights and false for anything with state that does
not commute across clients: batch-norm statistics, optimiser moments held client-side, and adaptive
learning rates all break the simple sum. Section 8 collects these as architecture failures rather than
numerical ones.

**The two invariants worth memorising.** Masking preserves the sum modulo a bound, so the server learns
the total and nothing else. Sampling preserves the expectation over cohorts, so averaging across rounds
converges to the population objective. Every other architectural claim is downstream of those two.

**What "architecture" means here.** Not the deployment diagram, but the set of decisions that determine
the four quantities in section 9: bytes per round, seconds per round, rounds to coverage, and epsilon per
campaign. Anything that changes none of those four is a configuration detail.

## 3. Foundational Definitions

**Client.** One participant holding local data. In cross-device settings a client is a phone or a
browser tab; in cross-silo settings a client is an organisation with an identity that persists across
rounds. The distinction matters because cross-device clients drop out constantly and cross-silo clients
do not.

**Cohort.** The `C` clients selected for one round. The cohort is a random variable. Reporting it as "the
clients that participated" hides the sampling mechanism, which is exactly the part that affects the
estimator and the privacy accountant.

**Round.** One broadcast, one local training phase, one upload, one aggregation, one model update. Rounds
are the unit of budget consumption in the privacy accountant and the unit of cost in section 9.

**Shard.** A client's local dataset. Shard sizes vary by orders of magnitude in cross-device settings and
are roughly comparable in cross-silo settings, which is why weighting errors are a cross-device problem.

**Non-IID.** Client distributions differ. Two distinct failures hide under this word: covariate shift,
where the feature distribution differs but the conditional does not, and label shift, where the class
prior differs. Label shift is what makes averaging fight itself, because each client's optimum points in
a different direction.

**FedAvg.** Server-side weighted averaging of client updates, with weights proportional to shard size,
after `E` local steps. The weighting is what makes it an estimator of the record-weighted objective rather
than the client-weighted one.

**Local steps (`E`).** The number of gradient steps a client takes before reporting. `E = 1` makes the
round a single large-batch step. Large `E` saves communication and increases client drift, and section
4.2 gives the exact form of the sampling-noise cost.

**Client drift.** The divergence between what a client would do with many local steps and what the global
objective would prefer, caused by local data being unrepresentative. It grows with `E`, with the
heterogeneity of shards, and with the local step size.

**FedProx.** A proximal term added to the local objective, penalising distance from the broadcast weights
with coefficient `mu`. It bounds the drift at the cost of a biased local optimum, and it is the standard
answer when `E` must be large for bandwidth reasons.

**Straggler.** A client whose local compute or network is slow enough to delay the round. Because a
synchronous round waits for the maximum, the straggler sets the duration. The expected maximum of `C`
lognormal durations grows like `sqrt(2 ln C)` in the log-scale, which is why 2,000 clients are much worse
than twenty times 100.

**Dropout.** A selected client that fails to return a valid update. Dropout is a rate, not an exception,
and the architecture must be designed for a rate rather than for zero.

**Secure aggregation.** A protocol in which the server learns only the sum of client updates, not any
individual update. The canonical construction has each pair of clients agree on a random mask, add the
mask when sending up to the other, and subtract it when sending down, so masks cancel in the sum.

**Masking invariant.** For client `i` sending `u_i = v_i + sum over j of m_ij` with `m_ij = -m_ji` and
`m_ii = 0`, the modular sum `sum_i u_i` equals `sum_i v_i` modulo the modulus. Holds exactly, for any
set of masks, provided nobody is missing.

**Threshold (`t_thr`).** The minimum number of clients whose seeds must be available for the server to
recover the surviving masks. If more than `C - t_thr` clients drop out, the server cannot unmask and the
round is lost. The threshold is the price of tolerating dropout.

**Shamir secret sharing.** A `(t, n)` scheme splitting a secret into `n` shares such that any `t` recover
it and any `t-1` reveal nothing. Used to distribute each client's mask seed so that the server can
reconstruct it only when enough neighbours survive.

**Cross-device versus cross-silo.** Cross-device: millions of untrusted, anonymous, intermittently
available clients, each with little data, and no ability to require participation. Cross-silo: tens of
trusted organisations, always available, with contractual participation and much more data each.

**Hierarchical aggregation.** Clients report to intermediate aggregators, which report partial sums to the
server. It reduces the server's fan-in and its bandwidth by the aggregation factor, at the cost of an
extra trust boundary and a smaller masking domain.

**Synchronous versus asynchronous.** Synchronous rounds wait for a fixed cohort, giving a clean estimator
and a clean accountant. Asynchronous updates apply whatever arrives, which removes the straggler but
introduces staleness: an update computed at weights `W_t` is applied at `W_t+a`.

**Staleness (`a`).** The number of rounds between when an update was computed and when it was applied.
Staleness is not merely a convergence nuisance; it also changes the privacy accountant, because an update
computed against old weights is still a function of current client data.

**Participation rate.** The expected fraction of rounds a given client participates in, `T*C/N`. At
`T = 3000`, `C = 2000`, `N = 50e6` it is 0.12 rounds per client, which is the number that makes device
energy negligible per device and enormous in aggregate.

**Coverage.** The fraction of the eligible population that has participated at least once, `1 - (1-q)^T`.
It answers a compliance question ("has every eligible device been used") and not a modelling one.

**Clip norm (`S`).** The L2 bound applied to each client's update before aggregation. Clipping bounds the
sensitivity, which is what makes noise calibration possible, and it also bounds the influence of a single
client, which is a robustness benefit independent of privacy.

**Noise multiplier (`sig`).** The ratio of DP noise standard deviation to the clip norm, per coordinate.
The accountant consumes `sig` and the sampling rate, not the raw epsilon.

**Amplification by sampling.** The privacy cost of a round is smaller than the cost of a mechanism run on
all clients, because each client is included with probability `q`. For small `q` the saving is roughly
quadratic, which is why the sampling rate appears in every federated accountant.

**Quantisation.** Reducing the bytes per parameter on the wire from four to one or to a fraction. It
reduces payload by up to 32 times, introduces a deterministic error bounded by the step size, and
interacts badly with DP noise when the quantisation step is coarser than the noise.

**Error feedback.** Keeping the residual of quantisation locally and adding it to the next update, so the
quantisation error does not accumulate as bias. It costs client-side state equal to one model copy.

**Server momentum.** A recursive filter at the server, `w <- z + rho (w - z)` for aggregate `z`. It
suppresses round-to-round sampling noise, and its stationary variance is derived in section 4.2.

**The definitions that decide most arguments.**

| Term | What it actually constrains | What it does not constrain |
|---|---|---|
| federated | where raw data stays | what the server infers from updates |
| secure aggregation | what the server can compute per client | what the released model reveals |
| differential privacy | worst-case inference about a participant | whether the client's device is compromised |
| local steps | communication frequency | convergence to the global optimum |
| clipping | per-client influence and sensitivity | the accuracy of the final model |
| hierarchical aggregation | server fan-in and WAN bytes | the trust placed in the aggregator |
| dropout threshold | whose masks are recoverable | whether the round's sum is representative |
| quantisation | bytes on the wire | the precision of the learned model |

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**FedAvg as a weighted estimator, and the bias from unweighted averaging.**

```text
FORMULA      z_t = sum over i in S of (n_i / n_S) * v_i,      n_S = sum over i in S of n_i

WHERE        S is the cohort, n_i is client i's shard size,
             v_i is the update computed by client i after E local steps,
             the client-update scale is arbitrary but identical across clients

CONSTRAINT   the weights must be computed from the same shard sizes used to train the update,
             and n_i must be known to the server before the round closes

EXAMPLE      three clients with n = 1000, 100, 50 and updates v = 0, +1, -1
             n_S = 1150
             z = (1000*0 + 100*1 + 50*(-1)) / 1150 = 50 / 1150 = 0.043478

INTERPRETATION  the large client dominates; the two small clients nearly cancel, and the
             server-side aggregate is 4.3 percent of one client's update magnitude
```

**Why weighting is not cosmetic.** The objective being minimised is
`F(w) = sum over i of (n_i/n) F_i(w)`, a record-weighted sum. The gradient is the same weighted sum, so an
unweighted average estimates `(1/|S|) sum F_i`, the client-weighted objective. When shard sizes differ by
an order of magnitude these are different functions with different minima.

**The magnitude of the error.** In the example the unweighted average is `(0 + 1 - 1)/3 = 0`, while the
correct answer is 0.0435. The error is not small relative to the signal; it is the entire signal.

**The sampling bias, derived.** Suppose the cohort is a uniform random subset of size `k` drawn without
replacement from `K` clients. The estimator is the ratio of two random quantities, so it is biased even
though the numerator and denominator are separately unbiased. Enumerating all subsets of size two from the
three clients above:

```text
FORMULA      bias = E_S[ sum_{i in S} n_i v_i / sum_{i in S} n_i ] - sum_i n_i v_i / sum_i n_i

WHERE        S ranges over all C(K, k) subsets of size k, each with probability 1/C(K,k)

CONSTRAINT   the expectation is over cohorts only, with the updates held fixed; it is not a
             convergence statement

EXAMPLE      K = 3, k = 2, subsets of size two:
             {1,2}: (0*1000 + 1*100)/(1100) = 0.090909
             {1,3}: (0*1000 - 1*50)/(1050)  = -0.047619
             {2,3}: (1*100 - 1*50)/(150)    =  0.333333
             E[z] = (0.090909 - 0.047619 + 0.333333)/3 = 0.125541
             bias = 0.125541 - 0.043478 = 0.082063

INTERPRETATION  a cohort of two thirds of the population is biased by 0.082 on a signal of
             0.043, so the bias is nearly twice the quantity being estimated
```

**Sampling with replacement is worse.** Drawing two clients uniformly from three with replacement gives
nine equally likely ordered pairs and `E[z] = 0.083694`, a bias of 0.040216. The direction is the same but
the magnitude differs, which is why the sampling mechanism must be written down rather than assumed: two
plausible implementations of "pick two clients" differ by a factor of two in bias.

**The bias shrinks but does not vanish with `k`.** The ratio estimator's bias is `O(1/k)` in variance
terms, and the practical rule is that cohorts of a few hundred with comparable shard sizes make it
negligible relative to sampling noise, while cohorts of tens with heavy-tailed shard sizes do not.

**Validity condition.** The base case is valid when shard sizes are stable across rounds, the cohort is
drawn by a declared mechanism, and the server receives the sizes with the updates. It is invalid when
clients report updates trained on a different number of samples than the weight they report, which is the
most common silent error in cross-device deployments.

### 4.2 General case

**Server momentum, local steps, and the stationary variance of the aggregate.**

```text
FORMULA      z_bar_t = mean over i in S_t of v_i
             w_t = z_bar_t + rho (w_{t-1} - z_bar_t),    rho = (1 - eta)^E
             Var[w] = (1 - rho)/(1 + rho) * Var[z_bar]

WHERE        z_bar is the cohort mean, w is the server state, rho in (0,1) is the
             per-round retention, eta is the client step size, E is local steps

CONSTRAINT   Var[z_bar] is the cohort-mean variance, Var[v]/k for a cohort of size k;
             the recursion assumes z_bar is iid across rounds with constant variance,
             which holds only if the cohort is re-drawn each round

EXAMPLE      K = 100 eligible, k = 10 sampled, per-client update variance 1.0
             Var[z_bar] = (1/k) * (K - k)/(K - 1) = 0.1 * 90/99 = 0.090909
             E = 1:  rho = 0.99      Var[w] = (0.01/1.99) * 0.090909 = 0.000456830
                     sd[w] = 0.021374
             E = 5:  rho = 0.950990  Var[w] = 0.049010/1.950990 * 0.090909 = 0.002283687
                     sd[w] = 0.047788
             E = 20: rho = 0.817907  Var[w] = 0.182093/1.817907 * 0.090909 = 0.009106030
                     sd[w] = 0.095426

INTERPRETATION  five local steps multiply the steady-state noise by 2.236 and twenty steps by
             4.465, which are almost exactly sqrt(5) and sqrt(20)
```

**Why the finite-population factor appears.** Drawing `k` of `K` clients without replacement gives a
sample mean whose variance is `(Var[v]/k) * (K-k)/(K-1)`. The factor is 0.909 at `k = 10`, `K = 100`, and
it approaches one as `K` grows. Omitting it overstates the noise by ten percent, which is small; using
`Var[v]/k` when `k = K` would claim nonzero noise for a full census, which is wrong by a factor of zero.

**Why the local-step penalty is a square root.** The retention factor `rho = (1-eta)^E` rises with `E`, so
the filter averages over more rounds and the effective number of independent samples grows. The stationary
variance ratio `(1-rho)/(1+rho)` increases approximately linearly in `E*eta` for small `E*eta`, and the
standard deviation therefore grows like the square root of the step count. The exact ratios verify this:
2.235842 against `sqrt(5) = 2.236068`, and 4.464650 against `sqrt(20) = 4.472136`.

**The invariance that saves communication.** If the total number of gradient steps is held fixed, the
stationary noise depends on the total steps, not on how they are split into rounds. At `eta = 0.01` and
`E = 5`, the filter reaches `rho^r = 0.01` at `r = 91.64` rounds, which is 458.2 steps. At `E = 20` it
takes `r = 22.91` rounds, also 458.2 steps. Communication drops by a factor of four and the statistical
efficiency per gradient step is unchanged, which is the actual argument for local steps.

**What breaks the invariance.** Three things. Client drift, because local steps move further from the
global optimum when shards are heterogeneous. Client availability, because a round touches a different
cohort and `E` local steps on a rare client are wasted. And the privacy accountant, because the number of
rounds is the number of queries, so fewer rounds at larger `E` is genuinely cheaper in epsilon for the
same number of gradient steps.

**The staleness bound.** For an asynchronous system, an update computed at `W_t` and applied at `W_t+a`
has an error that is first order in `a*eta*||grad||` and second order in the local Hessian. The practical
rule is `a <= 1/eta` for stability, which at `eta = 0.01` is 100 rounds, a bound almost never met in
practice because arrivals are far more skewed than that.

**Validity condition.** The general case is valid when cohorts are re-drawn independently, update
variance is roughly stationary, and `rho` is a scalar. It is invalid when the server applies per-layer
learning rates, because the recursion then has a different `rho` per layer and a single stationary
variance no longer describes the state.

### 4.3 Variants

**Secure aggregation masking, with the invariant shown numerically.**

```text
FORMULA      client i sends u_i = v_i + sum over j of m_ij   (mod R),   m_ij = -m_ji,  m_ii = 0
             sum over i of u_i = sum over i of v_i   (mod R)

WHERE        R is a modulus larger than the maximum possible absolute sum, so no wrap occurs;
             pairs are formed by a key agreement, so m_ij is known only to i and j

CONSTRAINT   every selected client must contribute masks with every other selected client, so
             the mask count is C(C-1)/2 and the key agreement count is C(C-1)

EXAMPLE      v = [123, 456, 789], R = 1000
             m_12 = 250, m_21 = 750, m_13 = 100, m_31 = 900, m_23 = 400, m_32 = 600
             u_1 = 123 + 250 + 100 = 473
             u_2 = 456 + 750 + 400 = 1606
             u_3 = 789 + 900 + 600 = 2289
             sum u = 4368, and 4368 mod 1000 = 368
             sum v = 1368, and 1368 mod 1000 = 368

INTERPRETATION  the server recovers 368 and cannot recover any v_i, because each u_i is
             uniform given the masks; recovery needs a single missing client's seed
```

**Why the mask is uniform from the server's view.** If client `j` is present, `m_ij` is unknown to the
server and uniform over the modulus, so `u_i` is uniform regardless of `v_i`. The server learns nothing
about `v_i` until it knows one of the masks in the sum. That is exactly what the threshold governs.

**The threshold recovery and its dropout arithmetic.** Each client's seed is split into `C-1` shares, one
per peer. If a set of clients drops out, the server needs the survivors' shares of each dropped client's
seed. A `(t_thr, C-1)` Shamir scheme means the server can reconstruct a dropped client's seed as long as
at least `t_thr` shares are available, that is, as long as at least `t_thr` of the `C-1` peers survive.

```text
FORMULA      P(round aborts) = P(number of survivors < t_thr)     for a single dropped client
             for a dropout rate p and cohort C, survivors ~ Binomial(C-1, 1-p)

WHERE        t_thr is chosen with `2 <= t_thr <= C - 1`; the abort condition is
             shared across all dropped clients, so the worst case governs

CONSTRAINT   the bound treats dropouts as independent, which they are not: a regional
             outage or a client-app release correlates them

EXAMPLE      C = 2000, p = 0.05, so survivors have mean 1900 and sd 9.75
             t_thr = 1300: P(fewer than 1300 survivors) is effectively zero
             t_thr = 1900: P(fewer than 1900 survivors) is about 0.5, so half the rounds abort
             t_thr = 1950: P(fewer than 1950 survivors) exceeds 0.99, so the design is dead

INTERPRETATION  the threshold must sit several standard deviations below the expected
             survivor count, and the cost of doing so is seed availability per dropped client
```

**Why the threshold is a privacy parameter, not just an availability knob.** A low threshold means fewer
clients need to be present for the server to unmask, so a smaller coalition can collude to expose an
update. A threshold of two makes masking nearly decorative. The design tension is between tolerance for
dropout and the size of the coalition that can break the guarantee.

**Shamir reconstruction, checked numerically.** With prime `p = 17`, secret `s = 7`, and a single random
coefficient `a = 3`, the polynomial is `f(x) = (7 + 3x) mod 17`, giving `f(1) = 10`, `f(2) = 13`,
`f(3) = 16`. Any two points interpolate to `f(0) = 7`: from `(1,10)` and `(2,13)` the slope is 3 and the
intercept is 7; from `(1,10)` and `(3,16)` the slope is 3 and the intercept is 7; from `(2,13)` and
`(3,16)` the same. One point alone leaves 17 candidates, which is the security statement.

**Quantisation.** With `b` bits over a symmetric range `[-S, S]`, the step is `2S/(2^b - 1)` and the
rounding error is bounded by half a step, with RMS `step/sqrt(12)` for uniformly distributed residuals.
At `S = 1` and `b = 8` the step is 0.007843 and the RMS error is 0.002264. At `b = 4` the step is 0.133333
and the RMS error is 0.038490, seventeen times larger. Note that a "signed 8-bit" encoder with a sign bit
and 255 magnitudes spans a different grid than an unsigned 256-level encoder; the sheet uses `2^b - 1`
consistently and states it.

**Why quantisation and DP noise interact.** The noise standard deviation per coordinate at `sig = 1.1`
and `C = 2000` is `1.1/2000 = 0.00055`, which is more than fourteen times *smaller* than the 8-bit
quantisation step of 0.007843. Quantising after adding noise therefore collapses the noise into a few grid
points, destroys its distribution, and invalidates the accountant, while quantising before adding noise is
safe because the noise is then the last operation. Order of operations is a correctness property here, not
a performance one.

**Amplification, and the factor it buys.** At `q = 4e-5`, `sig = 1.1`, `T = 3000`, `delta = 1e-6`, a
subsampled-RDP accountant gives `eps = 0.601`. The same `sig` without the amplification argument, using
`rho = 1/(2 sig^2)` per step, gives `eps = 1242.8`. To reach `0.601` without amplification the noise
multiplier must rise to about 50.06, forty-five times larger.

**The consequence of that factor.** A forty-five-fold noise increase on a 4.2-million-parameter model has
an L2 norm of `50.06 * sqrt(4.2e6) = 102,581` per round, against a signal whose per-coordinate magnitude
is around `3e-3` and whose L2 norm is therefore about 6.15. The noise exceeds the signal by four orders of
magnitude and the model does not train. Amplification is not an accounting nicety; it is the difference
between a working system and a noise generator.

### 4.4 Limits and counterexamples

**Limit one: masking does not survive a missing client without extra machinery.** If client `j` drops
after sending `u_j` but the server never learns `m_ij` for the survivors, those survivors' masks never
cancel and the sum is wrong by a residual that is uniform and large. The protocol must therefore either
recover seeds by threshold sharing or exclude the missing client's mask contributions from every
survivor, which requires knowing who dropped before anyone sends.

**Limit two: the sum can be private while the model is not.** Secure aggregation protects each round's
individual update. The model trained from those sums is released, and it memorises training data. A system
with perfect per-round masking and no DP has a membership-inference problem indistinguishable from
centralised training. This is the single most common architecture claim that fails review.

**Limit three: a full census is not a sample.** The finite-population correction vanishes at `C = N`, so
`q = 1` and the amplification factor disappears. A cross-silo deployment with fifteen partners that all
participate in every round gets no amplification benefit; the guarantee is the unamplified one, and the
accountant must be told that explicitly.

**Limit four: correlated dropout breaks the binomial model.** The threshold arithmetic assumes independent
dropouts. A client-app update that crashes on one OS version, or a regional network partition, removes a
correlated block. When thirty percent of a cohort disappears together, the survivor count is not
`Binomial(2000, 0.95)` but a mixture, and the tail is far fatter than 9.75 standard deviations suggests.

**Limit five: local steps with heterogeneous shards are not free.** The variance invariance in section 4.2
holds for the sampling noise of a *fixed* estimator. It says nothing about the bias introduced when each
client solves its own problem for twenty steps. On label-skewed shards the invariance is real for the
noise and irrelevant to the drift, and the model can fail to converge while every variance calculation
looks healthy.

**Limit six: an unbounded update has no usable sensitivity.** If a client can send an arbitrarily large
update, the sum's sensitivity is unbounded and no noise scale is defensible. Clipping is mandatory before
any DP claim, and clipping a client that is systematically larger than the norm converts that client's
contribution into a truncated one, which is a bias the design must acknowledge.

**Limit seven: the modulus must exceed the largest possible sum.** Choosing `R = 2^32` on a model whose
aggregate can exceed `2^32` in a coordinate causes wrap-around, and the server recovers a sum that is
wrong by a multiple of `R`. The check is arithmetic: `C * max|u_i| < R/2`, with the factor of two
reserving room for signed values.

## 5. Worked Numerical Example

**The setting (ILLUSTRATIVE).** A cross-device keyboard model with `P = 4.2e6` parameters at four bytes
each, so `M = 16.8 MB` per model copy. Population `N = 50e6` eligible devices. Cohort `C = 2000` devices
per round, `T = 3000` rounds. Uplink 1 Mbps, downlink 8 Mbps, local compute lognormal with median 45 s
and geometric standard deviation 1.5, aggregation 5 s. Clip norm `S = 1`, noise multiplier `sig = 1.1`,
`delta = 1e-6`.

**Step one: the sampling rate and the amplification regime.**

```text
FORMULA      q = C / N

WHERE        C = 2000, N = 50e6

CONSTRAINT   q is the probability any given eligible device is in a given cohort, and the
             accountant consumes this number, not the cohort size

NUMERICAL EXAMPLE   q = 2000 / 50,000,000 = 4.0e-5

INTERPRETATION  one device in 25,000 is selected per round, which is what makes amplification
             worth forty-five times the noise
```

**Step two: payload, flat versus hierarchical.**

```text
FORMULA      bytes per round (flat) = 2 * C * M
             bytes per round (hier) = 2 * F * M,  with F aggregators of C/F clients each

WHERE        M = 16.8 MB, C = 2000, F = 40, so each aggregator serves 50 clients

CONSTRAINT   the factor of two is upload plus broadcast; a hierarchical design still pays
             the broadcast to every client locally, which the WAN figure excludes

NUMERICAL EXAMPLE   flat: 2 * 2000 * 16.8 MB = 67.2 GB per round
             over T = 3000 rounds: 67.2 GB * 3000 = 201,600 GB = 201.6 TB
             uplink only: 100.8 TB
             hierarchical: 2 * 40 * 16.8 MB = 1.344 GB per round, 4.032 TB over 3000 rounds
             uplink only: 2.016 TB, a factor of 50 below the flat design

INTERPRETATION  the server-side payload is linear in the fan-in, so the aggregation factor is
             the single largest bandwidth lever available
```

**Step three: cost of that payload at a stated egress price.**

```text
FORMULA      cost = TB * 1000 GB/TB * price_per_GB

WHERE        price 0.05 currency units per GB, an ILLUSTRATIVE figure

CONSTRAINT   the price is a teaching value; the arithmetic is what transfers, not the tariff

NUMERICAL EXAMPLE   flat uplink 100.8 TB * 1000 * 0.05 = 5,040
             hierarchical uplink 2.016 TB * 1000 * 0.05 = 100.80
             a 50-fold reduction in bytes and in cost, from one topology change

INTERPRETATION  the topology decision is worth more than any compression scheme, because
             compression multiplies the same baseline while topology divides the fan-in
```

**Step four: round duration, driven by the maximum not the mean.**

```text
FORMULA      t_round = M*8/uplink + M*8/downlink + E[max compute] + t_aggregate

WHERE        M*8 = 134.4 Mb, uplink 1 Mbps, downlink 8 Mbps, aggregation 5 s
             compute is lognormal with median 45 s and geometric sd 1.5, C = 2000 clients

CONSTRAINT   the maximum term is over the selected cohort only; a cohort of 100 has a much
             smaller expected maximum, so round time is a function of cohort size

NUMERICAL EXAMPLE   upload 134.4 * 1 = 134.4 s
             download 134.4 / 8 = 16.8 s
             E[max compute] for 2000 lognormal clients, by numerical integration of the
             survival function of the maximum: 181.18 s
             t_round = 134.4 + 16.8 + 181.18 + 5 = 337.38 s, about 5.62 minutes
             with C = 100 the maximum term falls to about 124.6 s, giving 280.8 s

INTERPRETATION  for a 2000-client cohort the network is 45 percent of the round and the
             straggler tail is 54 percent; neither dominates, so optimising one alone caps
             the gain at about a factor of two
```

**Step five: wall-clock time and coverage.**

```text
FORMULA      T * t_round  for campaign wall clock
             rounds to coverage c:  T(c) = ln(1 - c) / ln(1 - q)

WHERE        q = 4e-5, t_round = 337.38 s

CONSTRAINT   coverage assumes sampling is uniform and independent across rounds, which fails
             if the sampler prefers always-online devices

NUMERICAL EXAMPLE   3000 rounds at 337.38 s = 1,012,140 s = 11.71 days
             T(0.50) = ln(0.5)/ln(1 - 4e-5) = 17,328 rounds = 67.66 days
             T(0.90) = ln(0.1)/ln(1 - 4e-5) = 57,563 rounds = 224.78 days
             T(0.99) = ln(0.01)/ln(1 - 4e-5) = 115,127 rounds = 449.55 days

INTERPRETATION  the training campaign finishes in under two weeks; touching ninety percent
             of eligible devices takes over seven months at the same cadence
```

**Step six: the privacy budget and its amplification dependence.**

```text
FORMULA      eps = min over alpha of [ T * eps_RDP(alpha; q, sig) + ln(1/delta) ] / (alpha - 1)
             eps_RDP(alpha; q, sig) from the subsampled-RDP expansion for a Gaussian mechanism

WHERE        q = 4e-5, sig = 1.1, T = 3000, delta = 1e-6, alpha searched over integers 2..159

CONSTRAINT   valid only if the sampler is uniform and independent of the data, and if every
             round clips to the same norm S; a change to either invalidates the number

NUMERICAL EXAMPLE   eps = 0.601 at delta = 1e-6
             at delta = 1e-5 the same configuration gives 0.51
             without amplification at sig = 1.1 the same T gives eps = 1242.8
             the sig required for eps = 0.601 without amplification is 50.06

INTERPRETATION  the sampler is doing forty-five times more work for privacy than the noise
             multiplier is, and the accountant must be told about the sampler explicitly
```

**Step seven: is the noise small enough to train?**

```text
FORMULA      ||noise||_2 = sig * S * sqrt(P) / C

WHERE        sig = 1.1, S = 1, P = 4.2e6, C = 2000

CONSTRAINT   the per-round noise is added to the aggregate, so dividing by C is correct only
             when the aggregate is a mean rather than a sum; the sheet uses the mean form

NUMERICAL EXAMPLE   1.1 * 1 * sqrt(4,200,000) / 2000 = 1.1 * 2049.39 / 2000 = 1.12716
             a signal with per-coordinate magnitude 3e-3 has L2 norm
             3e-3 * sqrt(4,200,000) = 6.148, so the noise is 18.3 percent of the signal
             at C = 200 the same signal faces noise of 11.27, a ratio of 1.83

INTERPRETATION  the round-level SNR is workable at C = 2000 and inverted at C = 200; the cohort
             floor for SNR one is 366.7 clients at this signal magnitude
```

**Independent arithmetic re-check.** Verify step two from a different direction: the per-round flat
payload is 67.2 GB, and `67.2 GB / 16.8 MB = 4000`, which is exactly `2C`. Verify step four: `134.4 Mb` at
1 Mbps is 134.4 s and `134.4 Mb` at 8 Mbps is 16.8 s, whose sum is 151.2 s, leaving
`337.38 - 151.2 - 5 = 181.18` s for compute, matching the integration. Verify step seven: at `C = 2000`
the per-coordinate noise is `1.1/2000 = 0.00055`, and `0.00055 * sqrt(4.2e6) = 1.1272`, consistent.

**Sensitivity note.** Four levers, each with its own scaling. Doubling the cohort halves the noise but
increases the straggler term by roughly `sqrt(2 ln 2C / ln C)` in log-scale, a few percent. Eight-bit
quantisation cuts payload by four times and costs an RMS error of 0.001132 per unit of clipped range, which
is twice the DP noise at `C = 2000` and therefore requires care about ordering. Adding a hierarchical layer
divides WAN bytes by the aggregation factor and adds a trust boundary. Reducing rounds from 3000 to 1500
halves epsilon roughly linearly and extends wall clock time only if the wall clock is round-bound, which
here it is.

**What the example deliberately excludes.** No claim is made that any keyboard product uses these
parameters, that `N = 50e6` is a real population, or that 0.05 per GB is a real tariff. The arithmetic is
the deliverable; the parameters are chosen so that a reader can re-derive every line with a calculator.

## 6. Visual / Geometric Intuition

**The round as a pipeline with one bottleneck.** Draw the four terms of `t_round` as a horizontal bar:
network out, network in, compute, aggregation. At 2,000 clients the bar is 134.4, 16.8, 181.2, and 5
seconds. Any optimisation that targets only the largest term is capped at that term's share, which is why
"compress the model" and "cap the straggler" must be done together to get more than a factor of two.

**Fan-in as a geometric series.** Flat aggregation has fan-in `C`. Inserting one hierarchical layer gives
fan-in `F` at the server and `C/F` per aggregator. Two layers give `F^2` at the top. The bandwidth
consequence is that the server's bytes scale with the fan-in, but the total bytes across all links scale
with the number of clients, so hierarchy trades server cost against per-aggregator cost rather than
eliminating load. Section 9 makes this concrete.

**The masking matrix.** Picture a `C x C` matrix `M` with `M_ij = -M_ji` and a zero diagonal. Each client
row-sums the matrix and adds it to its own value. The matrix is skew-symmetric, so the column sums are the
negatives of the row sums and the grand total is zero. The server sees the row sums only after the client
values are added, and the skew symmetry is the reason the invariant holds. Any protocol failure that
breaks the symmetry, such as one client adding a mask without its partner subtracting it, leaves a
residual equal to the mask.

**The trust ladder, drawn as three nested boxes.** Innermost: raw data on device. Next: the individual
update. Next: the aggregate. Next: the released model. Secure aggregation removes the server's access to
the second box. Differential privacy bounds inference about the first from the fourth. Nothing in the
architecture removes the first box from the device's own operating system, which is why device compromise
is outside the model.

**Straggler geometry.** Lognormal compute times with median 45 s and geometric standard deviation 1.5
have a 99th percentile of 115.57 s but an expected maximum at 2,000 clients of 181.18 s. The maximum is
not near the 99th percentile of the individual distribution, because the maximum of many draws is not its
own quantile. This is the single most commonly miscalculated number in federated capacity planning.

**Coverage as exponential saturation.** `1 - (1-q)^T` starts linear and flattens. At `q = 4e-5` the first
17,328 rounds cover half the population and the next 40,235 cover another forty percent. The shape is the
reason compliance arguments about "every device eventually participates" are true and unhelpful: the
last ten percent takes almost as long as the first ninety.

**The quantisation ladder.** Four bytes to one byte is a factor of four in bytes and a step of
`2S/255`. One byte to one bit is another factor of eight and a step of the full range, which is only
usable with error feedback and a residual that persists across rounds. Each rung down buys payload and
costs a bounded, non-random error that interacts with any noise applied afterwards.

## 7. Comparison / Selection Table

**The topology matrix.** Latency and byte figures are computed for `P = 4.2e6`, `M = 16.8 MB`, `C = 2000`,
`T = 3000`, and the network parameters of section 5, so the comparison is arithmetic rather than opinion.

| Topology | Server fan-in | WAN bytes over 3000 rounds | Round time driver | Trust boundary added | Dropout tolerance | Choose when |
|---|---|---|---|---|---|---|
| Flat, one server | 2000 | 201.6 TB both directions | 181 s straggler at C = 2000 | none, but a single point of sight | any, via threshold | population is small or the server is trusted |
| One hierarchical layer, F = 40 | 40 | 4.032 TB both directions | 181 s client compute, plus per-agg | the aggregator sees its 50 clients' sums | per-aggregator, not global | 2000 clients is bandwidth-prohibitive |
| Two hierarchical layers, F = 40, F = 40 | 40 | 4.032 TB at the top, plus local | same client compute | two aggregator tiers | localised, harder to reason about | continental deployments with strong locality |
| Peer-to-peer ring | 2 | 2C*M per round, same total | slowest neighbour link twice | every peer sees its neighbours | fragile, a broken link splits the ring | no server is permitted at all |
| Asynchronous flat | 1 | same bytes, spread in time | arrival rate, not a maximum | none | high, no waiting | stragglers dominate and staleness is tolerable |
| Cross-silo federation, C = 15, no sampling | 15 | 1.512 TB both directions | the slowest partner, plus WAN | none, but legal agreements | low, partners are contractual | partners are organisations with stable availability |
| Centralised training on pooled data | not applicable | one copy of the data | not applicable | every party sees every record | not applicable | pooling is lawful and the risk is acceptable |

**Reading the fifth column.** A trust boundary is not a bad thing; it is a thing that must be named. A
hierarchical design that puts an aggregator in a region usually means that region's operator can see the
sum of fifty clients. If the masking domain is the fifty, the aggregator can unmask with a coalition of
two. If the masking domain is all two thousand, the aggregator forwards masked shares and gains nothing,
which changes the protocol, the seed distribution, and the byte count.

**Selection rule one: fix the threat model before the topology.** If the server must not see individual
updates, flat plus threshold masking is the simplest correct design. If the server must not see individual
updates *and* regional operators must not either, the masking domain must span the hierarchy, which
increases seed traffic and threshold arithmetic.

**Selection rule two: cohort size is set by the noise, not by the network.** The noise scales as `1/C`, so
the minimum workable cohort is set by the SNR requirement in section 5 step seven. Once that floor is met,
larger cohorts buy amplification and cost straggler latency and bytes.

**Selection rule three: prefer synchrony unless the straggler is proven dominant.** Asynchronous designs
remove a maximum from the duration but add staleness, complicate the accountant, and make coverage
arguments harder. The measured straggler share in section 5 is 54 percent, which is high but not so high
that asynchrony is obviously right.

**Selection rule four: quantise before adding noise, never after.** This is an ordering constraint from
section 4.3, not a performance preference. A system that quantises after noise must either raise the noise
or accept a broken accountant.

**Selection rule five: budget for the last ten percent of coverage.** If a compliance requirement is
"every eligible device has participated", the campaign length is set by `T(0.99)`, not `T(0.90)`, and
`T(0.99) = ln(0.01)/ln(1-4e-5) = 115,127` rounds, which at 337 s is 449 days. Either the requirement is
softened or the architecture changed.

**Selection rule six: write down what the aggregator sees.** In every topology in the table, the
aggregator's view is the thing that review will ask about, and it is the thing deployment diagrams omit.
One sentence per topology: flat sees nothing individual under masking; hierarchical sees its subtree's
sum; peer-to-peer sees each neighbour's masked value; asynchronous sees updates arriving out of order.

## 8. Failure Modes & Edge Cases

**How to use this catalogue.** Each entry gives a symptom, the competing causes that produce it, the check
that separates them, the correction, and a verification. An entry is only useful if a competent engineer
could plausibly blame the wrong cause.

**F1: rounds abort at a higher rate than the threshold model predicts.**

| Field | Content |
|---|---|
| Symptom | five percent of rounds fail to unmask, while the arithmetic predicted fewer than one in a thousand |
| Competing causes | dropout is correlated, not independent; the threshold was set from the mean rather than the tail; some clients drop after uploading but before seed sharing |
| Discriminating check | plot dropout by time-of-day and by app version; a bimodal or clustered pattern proves correlation, a flat pattern points at the threshold choice |
| Correction | set `t_thr` at least four standard deviations below the expected survivor count, and treat correlated blocks explicitly |
| Prevention | the threshold formula consumes an empirical dropout distribution, not a single rate |
| Verification | observed abort rate sits inside the predicted interval for ten consecutive days |

**F2: the model trains but the aggregate is wrong by a constant.**

| Field | Content |
|---|---|
| Symptom | loss plateaus high and the aggregate magnitude is smaller than any single client's update |
| Competing causes | unweighted averaging over unequal shards; weights computed from a stale shard-size table; clients reporting updates normalised by their own step count |
| Discriminating check | recompute the weighted mean by hand for one round from the logged shard sizes and updates; compare with the server's aggregate |
| Correction | compute weights from the sizes attached to the update, and reject an update whose size tag is missing |
| Prevention | the update message carries `n_i` and a checksum over the size used for training |
| Verification | hand recomputation matches to six significant figures for a sampled round |

**F3: masking is bypassed by the aggregate itself.**

| Field | Content |
|---|---|
| Symptom | the server cannot read individual updates but can recover one client's data from the model |
| Competing causes | no DP anywhere; clipping applied per coordinate rather than per update; a single-client round for a rare device |
| Discriminating check | run a membership-inference test against the released model; a success rate well above chance proves the model, not the protocol, is the leak |
| Correction | add clipping and noise, and forbid cohorts below the minimum size |
| Prevention | the release gate requires both a masking claim and a DP claim, recorded separately |
| Verification | the membership test on a held-out split is near chance, and the accountant number is present in the release record |

**F4: the coordinator has a second copy of the data.**

| Field | Content |
|---|---|
| Symptom | the client's upload includes a raw feature vector or an identifier alongside the update |
| Competing causes | telemetry fields added by a library default; a debugging hook left enabled; a "model card" payload that carries sample rows |
| Discriminating check | inspect the serialised upload at the socket; any field whose length correlates with shard size is suspect |
| Correction | whitelist the message schema; reject unknown fields at the aggregator |
| Prevention | schema validation runs in the client, not only in the server |
| Verification | a synthetic client with a distinctive payload fails the validator in a test |

**F5: staleness is invisible until accuracy collapses.**

| Field | Content |
|---|---|
| Symptom | training curves look healthy, then accuracy drops sharply at a round count |
| Competing causes | asynchronous updates applied at `W_t+a` with `a` growing; a client retry queue replaying old updates; momentum compounding stale gradients |
| Discriminating check | log the age of every applied update; if the 99th percentile exceeds `1/eta`, staleness is the cause |
| Correction | drop updates older than a staleness bound, or reset server momentum when the bound is exceeded |
| Prevention | the aggregator tags updates with the round number they were computed at |
| Verification | the age histogram has a hard ceiling equal to the configured bound |

**F6: local steps silently bias the model on skewed shards.**

| Field | Content |
|---|---|
| Symptom | the model performs well on large shards and poorly on small or rare ones |
| Competing causes | client drift with large `E`; weighting that under-represents small shards; a global class prior that the large shards dominate |
| Discriminating check | train a model with `E = 1` on the same data and compare per-shard accuracy; a gap that closes at `E = 1` is drift |
| Correction | add a proximal term, reduce `E`, or increase the cohort so each round sees more of the distribution |
| Prevention | the configuration records `E` alongside the heterogeneity estimate of the population |
| Verification | per-shard accuracy dispersion is inside a stated band at the chosen `E` |

**F7: the modulus wraps.**

| Field | Content |
|---|---|
| Symptom | a coordinate of the recovered aggregate is wildly wrong and the error is a multiple of a power of two |
| Competing causes | modulus smaller than the maximum sum; signed values encoded without reserving the sign bit; quantisation pushing a value past the reserved range |
| Discriminating check | compute `C * max|u_i|` from one round and compare with `R/2` |
| Correction | raise the modulus to at least `2 * C * max|u_i|`, and re-derive the message width |
| Prevention | the modulus is derived from the clip norm and the cohort size, not hard-coded |
| Verification | a test with adversarial all-maximum updates produces no wrap |

**F8: noise is added on the client and the accountant is wrong.**

| Field | Content |
|---|---|
| Symptom | the claimed epsilon is an order of magnitude smaller than a reviewer's recomputation |
| Competing causes | the accountant assumes a central mechanism while clients add local noise; noise is added per client and then averaged, so it shrinks by `C` but the guarantee does not improve; clipping is applied after noise |
| Discriminating check | find where in the code path the noise is drawn; central noise is applied once to the aggregate, local noise once per client |
| Correction | pick one trust model and account for it; local noise at the client cannot use the amplification argument |
| Prevention | the mechanism and the accountant are selected by the same configuration object |
| Verification | the ledger's `sig` matches the value in the client upload handler |

**F9: the sampler is not uniform.**

| Field | Content |
|---|---|
| Symptom | coverage stalls below the predicted value and certain demographics never appear |
| Competing causes | selection prefers devices that are charging and on wifi; a retry queue re-selects the same clients; the eligibility set is stale |
| Discriminating check | compare the empirical selection frequency distribution with uniform; a Kolmogorov-Smirnov style check on device buckets is enough |
| Correction | sample uniformly from the eligible set and refresh eligibility, or declare a stratified design and account for it |
| Prevention | the sampler is a versioned component with a stated distribution |
| Verification | the selection-frequency histogram is flat across buckets within sampling error |

**F10: quantisation after noise destroys the guarantee.**

| Field | Content |
|---|---|
| Symptom | the DP accountant is satisfied but an attack distinguishes two neighbouring inputs |
| Competing causes | quantisation applied to the noisy aggregate with a step coarser than the noise; a deterministic rounding that leaks the pre-rounding value; a cache of the pre-quantised value |
| Discriminating check | compare the quantisation step with `sig*S/C`; if the step is larger, the pipeline is leaking through rounding |
| Correction | quantise before noise, or make the step at least an order of magnitude finer than the noise |
| Prevention | the ordering is a documented invariant of the aggregation function |
| Verification | a differential test over neighbouring inputs shows output distributions that differ by at most `exp(eps)` |

**F11: the client's optimiser state is silently dropped.**

| Field | Content |
|---|---|
| Symptom | convergence is much slower than a centralised baseline at identical hyperparameters |
| Competing causes | Adam moments reset each round; batch-norm statistics not aggregated; learning-rate schedule applied per round rather than per step |
| Discriminating check | count the state bytes uploaded; a four-byte-per-parameter upload means no optimiser state is travelling |
| Correction | either ship optimiser state (two extra model copies per round) or switch to a stateless optimiser and retune |
| Prevention | the round payload schema names every tensor included |
| Verification | the payload size equals `P * bpw * (number of state tensors + 1)` |

**F12: the campaign's wall-clock time is set by an unmodelled stage.**

| Field | Content |
|---|---|
| Symptom | rounds take twice the predicted duration and no single network or compute metric is high |
| Competing causes | serialisation on the client; key agreement for masks at `C(C-1)` scale; the aggregator's own queueing; retries after a partial failure |
| Discriminating check | instrument each of the six round stages with timestamps and sum them; the residual is the unmodelled stage |
| Correction | pre-compute mask agreements across rounds, or reduce the masking domain |
| Prevention | stage-level timing is a first-class metric, not a debugging aid |
| Verification | the six stage durations sum to the observed round time within two percent |

**Edge case: a cohort of one.** A round with a single client reveals that client's update completely, with
or without masking, because the sum is the value. The architecture needs a hard floor on cohort size, and
the floor must be enforced at the aggregator rather than requested in configuration.

**Edge case: the same client selected twice.** A sampler that draws with replacement can select the same
client twice, which double-counts its data and, worse, makes its two updates addable in a way that
cancels its masks. Sampling must be without replacement within a round, and the check is a set cardinality
assertion before the round opens.

## 9. Complexity & Cost Analysis

**Four cost currencies.** Bandwidth, wall clock, energy, and privacy budget. A design that optimises one
usually spends another, and the exchange rates are computable, which is what this section does.

**Bandwidth as a table.** With `M = 16.8 MB`, `C = 2000`, `T = 3000`, and bytes counted in decimal units.

| Configuration | Bytes per round, both directions | Bytes over 3000 rounds | Relative to flat |
|---|---|---|---|
| Float32, flat | 67.2 GB | 201.6 TB | 1.0 |
| Float16, flat | 33.6 GB | 100.8 TB | 0.5 |
| Int8, flat | 16.8 GB | 50.4 TB | 0.25 |
| 1-bit with error feedback, flat | 2.1 GB | 6.3 TB | 0.031 |
| Int8, one hierarchical layer, F = 40 | 0.336 GB | 1.008 TB | 0.005 |
| Float32, one hierarchical layer, F = 40 | 1.344 GB | 4.032 TB | 0.02 |

**Reading the table correctly.** The last two rows separate the two levers: compression multiplies the
baseline by the ratio of precisions, while hierarchy divides the fan-in. Applying both gives
`4.032 TB / 4 = 1.008 TB`, which is the smallest entry, and neither lever alone reaches it.

**Latency as a table.** Stage durations for the section 5 configuration, with the compute term from the
expected maximum over the cohort.

| Cohort `C` | Upload s | Download s | Expected max compute s | Aggregation s | Round s | Rounds to 90 percent coverage | Campaign days |
|---|---|---|---|---|---|---|---|
| 100 | 134.4 | 16.8 | 124.61 | 5 | 280.81 | 57,563 | 187.1 |
| 500 | 134.4 | 16.8 | 147.08 | 5 | 303.28 | 57,563 | 202.0 |
| 2000 | 134.4 | 16.8 | 181.18 | 5 | 337.38 | 57,563 | 224.8 |
| 20000 | 134.4 | 16.8 | 268.60 | 5 | 424.80 | 57,563 | 283.0 |

**Reading the latency table.** The upload and download terms are per client and do not change with `C`;
only the straggler term does. A fivefold cohort increase costs 8.0 percent more round time, from 280.81 s to
303.28 s, and a twentyfold increase costs 51.3 percent more, because the maximum of a lognormal sample
grows logarithmically in the sample size. Campaign days to ninety percent coverage rise from 187.1 to
283.0 across the same range, a smaller relative increase than the round time suggests, because the
coverage round count is identical in every row.

**The privacy cost curve.** Under the subsampled accountant at `q = 4e-5`, `delta = 1e-6`:

| `sig` | eps at `T = 3000` | Relative noise | Comment |
|---|---|---|---|
| 0.8 | 1.166 | 0.73 | cheapest training, weakest guarantee, about 1.9 times the section 5 epsilon |
| 1.1 | 0.601 | 1.00 | the section 5 setting |
| 1.3 | 0.419 | 1.18 | 30 percent less epsilon for 18 percent more noise |
| 1.5 | 0.314 | 1.36 | roughly half the epsilon for 36 percent more noise |
| 2.0 | 0.174 | 1.82 | 3.45 times less epsilon, if the model tolerates the noise |

**Why the curve is gentle here and brutal elsewhere.** Going from `sig = 1.1` to `sig = 2.0` multiplies
noise by 1.82 and divides epsilon by 3.45, which is better than linear because the subsampled accountant's
dependency on `sig` is close to quadratic in this regime. That favourable exchange exists only because `q`
is tiny; at `q = 1` the same `sig` change buys a factor of only `(2.0/1.1)^2 = 3.31` in rho with no
amplification to compound it, and the practical difference between cross-device and cross-silo projects
is exactly this term.

**Energy as a table.** Compute at 1.5 W for 45 s, `C = 2000`, `T = 3000`.

| Quantity | Value | Derivation |
|---|---|---|
| Energy per client round | 67.5 J | `45 s * 1.5 W` |
| Energy per round, cohort | 135 kJ | `2000 * 67.5 J` |
| Energy per round in kWh | 0.0375 kWh | `135 kJ / 3.6e6` |
| Energy per campaign | 112.5 kWh | `3000 * 0.0375 kWh` |
| Energy per device per campaign | 8.1 J | `0.12 rounds * 67.5 J` |
| Charge per device per campaign | about 5.07 mAh | `67.5 J / 3.6 / 3.7 V` at 0.12 rounds |

**The energy conclusion.** Per device the campaign is negligible, which is what makes cross-device
federated learning acceptable to participants. In aggregate it is a real cost, and the aggregate is what
appears on a sustainability report. Both statements are true and a design review needs both.

**Secure aggregation cost.** Two terms. Mask agreement is `C(C-1)` key agreements per round, which at
`C = 2000` is 3,998,000 agreements, and this is usually the dominant CPU cost of the protocol rather than
the bytes. Seed sharing is `C*(C-1)*32` bytes if every client shares with every peer, which is 127.9 MB per
round, or `C*t_thr*32` bytes with a threshold scheme, which at `t_thr = 131` is 8.384 MB.

**Why the threshold changes the cost by 15 times.** All-pairs sharing means every client sends a share to
every peer, so the byte count is quadratic in `C`. A `(t_thr, C-1)` scheme sends each share to a designated
subset whose size is set by the security parameter, and the count is linear in `C` with a constant of
`t_thr`. At `C = 2000` the ratio is `1999/131 = 15.3`.

**The scale wall.** At `C = 10,000` the all-pairs share volume becomes `10,000 * 9,999 * 32 = 3.2 GB` per
round and the agreement count becomes 99,990,000. Both are prohibitive, which is why large-cohort secure
aggregation uses a hierarchical protocol or a threshold scheme with a small `t_thr`, and why cohort size
is jointly constrained by privacy (wants more) and cryptography (wants fewer).

**Cost of the wrong cohort.** Too small and the noise swamps the signal: at `C = 200` the round noise norm
is 11.27 against a signal norm of 6.48, so the SNR is 0.58 and training stalls. Too large and the
straggler term and the mask agreement cost grow. The window between the two is narrower than most designs
assume: at `sig = 1.1` and a signal norm of 6.48, the cohort must exceed `1.1 * 2049.39 / 6.48 = 348`
clients just to reach SNR one.

**Cost of retries.** A five percent dropout rate on `C = 2000` is 100 missing updates per round. If the
design retries them, it pays `100/2000 = 5` percent extra bytes and, worse, re-selects clients that were
already in the cohort, which perturbs the sampling distribution the accountant assumed. Retrying silently
breaks the amplification argument, and the fix is to treat a retry as a new cohort draw rather than a
repair of the old one.

## 10. Key Theorems / Results Reference

**FedAvg (McMahan, Moore, Ramage, Hampson, Agüera y Arcas, 2017).** Averaging client models after local
SGD steps approximates a single large step, with the approximation degrading as `E` and the heterogeneity
grow. Validity: identical initialisation across clients, full participation in the derivation, and a
convex or well-behaved objective. The paper's own experiments show the degradation, so citing it as a
convergence guarantee misreads it.

**Local SGD convergence (Stich, 2019; Khaled, Mishchenko, Richtárik, 2020).** For smooth convex objectives,
local SGD with periodic averaging matches minibatch SGD up to a communication-dependent factor, and the
gap is governed by the number of local steps and the data heterogeneity. Validity: convexity or a
bounded-heterogeneity assumption, both of which fail on real networks, which is why the practical rule
below matters more than the constant.

**FedProx (Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith, 2020).** Adding `(mu/2)||w - w_global||^2` to the
local objective bounds client drift, and the bound improves with `mu` at the cost of a biased local
solution. Validity: the proximal term is actually applied in the local optimiser, which is where
implementations diverge from the paper.

**Secure aggregation (Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth, 2017).**
The masking protocol reveals only the sum, tolerates dropout up to a threshold, and is secure against a
coalition smaller than the threshold under a semi-honest server. Validity: the threshold must exceed the
coalition size and the key agreement must be authenticated. Security against a malicious server requires
additional verification not present in the basic protocol.

**Shamir secret sharing (Shamir, 1979).** A `(t, n)` threshold scheme with information-theoretic security
against `t-1` shares. Validity: the modulus is prime and larger than the secret, and shares are generated
with independent random coefficients. The scheme is the reason seed recovery costs `O(C*t)` bytes rather
than `O(C^2)`.

**Amplification by subsampling (Kasiviswanathan et al., 2011; Abadi et al., 2016 for the DP-SGD form;
Balle, Barthe, Gaboardi, 2018 for tight constants).** Sampling with rate `q` before a private mechanism
reduces the effective epsilon by roughly a factor of `q^2` in the small-`q` regime. Validity: sampling is
uniform, independent of the data, and the accountant uses the tight constant rather than a heuristic.
Section 5's factor of forty-five is this result, computed.

**The moments accountant (Abadi et al., 2016).** Tracking the log moment generating function of the privacy
loss composes adaptively and gives substantially tighter bounds than basic composition over many steps.
Validity: the mechanism is Gaussian with subsampling, and the accountant is evaluated at the same `q` and
`sig` used in training. RDP is a later reformulation with the same asymptotics.

**Concentration of the maximum of lognormal samples.** The expected maximum of `k` iid lognormal variables
grows approximately as `median * exp(sigma * Phi^-1(1 - 1/k))` for large `k`, which for median 45, `sigma =
ln 1.5`, and `k = 2000` gives about 174 s against an exact numerical value of 181.18 s. Validity: the
approximation is a heuristic; the sheet uses the numerically integrated value and cites the heuristic only
as a sanity check.

**EWMA stationary variance.** For `w_t = z_t + rho(w_{t-1} - z_t)` with iid `z`, the stationary variance is
`Var[z](1-rho)/(1+rho)`, derived by setting `Var[w] = rho^2 Var[w] + (1-rho)^2 Var[z]` and solving. Validity:
`|rho| < 1` and iid `z`, which fails under a persistent cohort or a drifting population.

**Finite-population sampling variance.** A simple random sample of `k` from `K` without replacement has
mean variance `(Var/k)(K-k)/(K-1)`. Validity: sampling without replacement from a fixed population. This is
the correction that makes a full census have zero sampling variance, which is the right answer and a useful
test of any estimator implementation.

**k-anonymity does not appear here, deliberately.** No federated protocol in this sheet produces a
k-anonymous artefact, and no epsilon can be derived from a group size. Design documents that mix the two
vocabularies are usually describing an intent rather than a mechanism.

## 11. Interview / Exam Questions

**Q1. A cohort of 2,000 clients has a 45-second median compute time with geometric standard deviation
1.5. Estimate the round's compute term and explain why it is not the 99th percentile.**

Answer: numerically integrating the survival function of the maximum gives 181.18 s, while the individual
99th percentile is 115.57 s. The maximum of 2,000 draws is not its own quantile: `P(max > 115.57) = 1 -
0.99^2000`, which is essentially one, so the maximum sits far above it. Reasoning: the CDF of the maximum
is `F(t)^C`, and solving `F(t)^C = 0.5` gives a median maximum well above `F^-1(0.5)`. Follow-up: what does
`C = 100` give? About 124.61 s by the same integration, so a twentyfold cohort costs 45 percent more in the
compute term. Red flag: quoting 115.57 s as the round's compute time. Scoring: two points for the value,
two for the tail explanation, one for the cohort comparison.

**Q2. Three clients have shard sizes 1000, 100, 50 and updates 0, +1, -1. What does an unweighted average
report, and what should it report?**

Answer: the unweighted average is `(0 + 1 - 1)/3 = 0`, and the correct weighted value is
`(1000*0 + 100*1 + 50*(-1))/1150 = 50/1150 = 0.043478`. The error equals the entire signal. Reasoning: the
objective is record-weighted, so the gradient is the size-weighted sum. Follow-up: what happens if the
cohort is a uniform 2-of-3 sample? The expectation of the weighted estimator is 0.125541, a bias of 0.082
against a true 0.043. Red flag: assuming the ratio estimator is unbiased because both parts are.
Scoring: two points for the arithmetic, one for the objective argument, two for the sampling bias.

**Q3. A round costs 67.2 GB at 2,000 clients with a 16.8 MB model. Where does the factor of two come from
and what does one hierarchical layer with 40 aggregators save?**

Answer: the factor of two is broadcast plus upload, so each of 2,000 clients receives and sends 16.8 MB.
One hierarchical layer with 40 aggregators of 50 clients each reduces the server's WAN traffic to
`2*40*16.8 MB = 1.344 GB` per round, a factor of 50. Reasoning: the server's bytes scale with fan-in, not
with the number of clients. Follow-up: what is the total traffic across all links? Still roughly
`2*C*M`, so hierarchy moves cost rather than removing it. Red flag: claiming a 50-fold total-traffic
reduction. Scoring: two points for the factor of two, two for the fan-in arithmetic, one for the caveat.

**Q4. Why does a fivefold increase in local steps multiply the steady-state update noise by about
`sqrt(5)`?**

Answer: `rho = (1-eta)^E` rises with `E`, so the filter averages over more rounds; the stationary variance
ratio `(1-rho)/(1+rho)` grows roughly linearly in `E*eta`, hence the standard deviation grows as the square
root. At `eta = 0.01`, `E = 1`, `K = 100`, `k = 10`, the computed standard deviations are 0.021374 and
0.047788, a ratio of 2.2358 against `sqrt(5) = 2.2361`. Reasoning: the effective sample count of an EWMA is
`(1+rho)/(1-rho)`, so variance is inversely proportional to it. Follow-up: at `E = 20`? Ratio 4.4646
against `sqrt(20) = 4.4721`. Red flag: claiming noise is independent of `E`. Scoring: two points for the
mechanism, two for the numbers, one for the effective-sample argument.

**Q5. A masking protocol with masks 250/750, 100/900, 400/600 over values 123, 456, 789 and modulus 1000.
Verify the invariant.**

Answer: `u = [473, 1606, 2289]`, summing to 4368, and `4368 mod 1000 = 368`; the values sum to 1368 and
`1368 mod 1000 = 368`. The server recovers 368 and each `u_i` is uniform to it. Reasoning: the mask matrix
is skew-symmetric, so its total is zero and the sum is preserved. Follow-up: what if client 3 drops? The
server must subtract `m_13` and `m_23` from the survivors, which requires their seeds. Red flag: asserting
the server learns nothing about `v_i` without checking that a mask is unknown to it. Scoring: two points
for the arithmetic, two for the invariant argument, one for the dropout consequence.

**Q6. At the same settings as section 5, the accountant reports 0.601 with amplification and 1242.8
without. Explain the gap and the factor of 45.**

Answer: the amplification argument uses the inclusion probability `q = 4e-5` in every step's RDP, so the
per-step cost scales roughly with `q^2`. Removing it leaves `rho = 1/(2*1.21)` per step over 3,000 steps, a
total `rho` of about 1239.7, which converts to an epsilon in the thousands. Reaching 0.601 without
amplification needs `sig` around 50.06, which is `50.06/1.1 = 45.5` times more noise. Reasoning:
amplification is a statement about a random subset, not about a mechanism. Follow-up: what breaks it? A
non-uniform sampler, or a cross-silo census where `q = 1`. Red flag: treating amplification as a
constant factor to be assumed. Scoring: two points for the mechanism, two for the numbers, one for the
failure conditions.

**Q7. Why must quantisation happen before noise rather than after?**

Answer: at `sig = 1.1` and `C = 2000` the per-coordinate noise is 0.00055, while an 8-bit quantisation step
over a unit range is 0.003922, seven times coarser. Rounding a noisy aggregate to that grid can collapse
distinct outputs to the same value in a data-dependent way and breaks the ratio bound the accountant
proved. Reasoning: quantisation is a deterministic function, so it is free only if it is applied to the
already-private value, and it stops being free when its resolution interacts with the noise structure.
Follow-up: what resolution is safe? A step at least an order of magnitude below `sig*S/C`, which here
means about 16-bit quantisation. Red flag: "rounding is post-processing, so it is free". Scoring: two
points for the ordering, two for the resolution comparison, one for the post-processing nuance.

**Q8. A cross-silo deployment has 15 partners, all of whom participate every round. What does the
accountant look like and why is it worse than the cross-device case?**

Answer: `q = 1`, so there is no amplification: the total `rho` is `T/(2 sig^2)` and epsilon grows linearly
in rounds. At 3,000 rounds reaching 0.6 needs `sig` about 50, versus 1.1 with `q = 4e-5`. Reasoning: the
amplification factor is `q^2`-ish and `q = 1` contributes nothing. Follow-up: what are the options? Fewer
rounds, larger `sig`, a per-partner contribution bound with a smaller clip, or a secure-computation
approach that is not DP at all. Red flag: quoting the cross-device epsilon for a census. Scoring: two
points for `q = 1`, two for the consequence, one for the remedies.

**Q9. Five percent of a 2,000-client cohort drops out. How do you choose the masking threshold?**

Answer: survivors are `Binomial(1999, 0.95)` with mean 1899.05 and standard deviation 9.75, so a threshold
of 1,900 aborts about half the rounds. Set the threshold at least four standard deviations below the mean,
around 1,860, and verify against the empirical dropout distribution rather than the assumed rate.
Reasoning: the abort condition is a lower tail on the survivor count, and the tail is fat when dropout is
correlated. Follow-up: what does a lower threshold cost? A smaller coalition can unmask, so the privacy
guarantee weakens. Red flag: setting the threshold at the mean. Scoring: two points for the tail
arithmetic, two for the correlation caveat, one for the privacy trade.

**Q10. A system reports per-round masking and no DP. What is it protected against, and what is it not?**

Answer: protected against a server that wants to read an individual update in a given round, assuming a
coalition below the threshold. Not protected against inference from the released model, which memorises the
same training data as a centralised model, and not protected against a compromised client or a coalition at
or above the threshold. Reasoning: masking bounds what the server computes, DP bounds what any adversary
infers from the output. Follow-up: what test demonstrates the gap? Membership inference against the
released model, which succeeds at well above chance on a non-private model regardless of masking.
Red flag: describing masking as anonymisation. Scoring: two points for the positive claim, two for the
gap, one for the test.

**Q11. You need to cut payload by 16 times without breaking the privacy argument. Which lever?**

Answer: hierarchical aggregation with an aggregation factor of 16, which divides the server's bytes
exactly and leaves the client-side payload, the clip norm, and the accountant untouched. Compression by 16
would require roughly `bpw = 0.25`, whose quantisation step over a unit clip range is `2/...` — computed as
`2S/(2^0.25 - 1)`, which is far coarser than the noise and would force a re-derivation of the mechanism.
Reasoning: topology changes the fan-in and is privacy-neutral, whereas precision changes the output
distribution. Follow-up: what does the hierarchical layer cost? An extra trust boundary and per-subtree
masking domains. Red flag: choosing compression because it is a client-side change. Scoring: two points
for the topology answer, two for the quantisation objection, one for the trust cost.

**Q12. What single configuration change most improves coverage per unit of wall clock, and what does it
cost?**

Answer: raising the cohort size, because coverage rounds scale as `1/q` while round time grows only
logarithmically in `C` through the straggler term. Going from 2,000 to 20,000 clients reduces
`T(0.90)` from 57,563 to 5,756 rounds while round time rises from 337.4 s to 424.8 s, so coverage wall
clock falls from 224.8 days to 28.3 days. Reasoning: `q` enters exponentially and `C` enters
logarithmically. Follow-up: what does it cost? Ten times the mask agreements, 3.2 GB of seed traffic at
`C = 10,000`, and a larger cohort that may exceed device availability. Red flag: assuming coverage is
round-bound rather than sampling-bound. Scoring: two points for the lever, two for the arithmetic, one
for the cost.

**Q13. A client's update is clipped from norm 4.1 to norm 1.0. What are the consequences beyond privacy?**

Answer: the client's contribution to the aggregate is scaled down by a factor of 4.1, so its influence on
the model is capped regardless of how unusual its data is. That is a robustness benefit against poisoning
and a bias against legitimate outliers, and the bias persists across rounds because the same client stays
large. Reasoning: clipping is a projection onto a ball, applied per update, so it changes the estimator
whenever the norm exceeds the bound. Follow-up: how would you detect systematic clipping? Measure the
fraction of clipped updates per client; a client clipped every round is either misconfigured or
adversarial. Red flag: describing clipping as a tuning parameter with no estimator consequence.
Scoring: two points for the influence cap, two for the bias, one for the detection.

**Q14. The ledger says the campaign is 0.6 epsilon. A reviewer recomputes 1.8. What are the three most
likely causes?**

Answer: the number of accounted steps is the epoch count rather than the optimiser steps; the sampler
used for accounting is uniform while the production sampler prefers charging devices, so the effective `q`
is larger; or the clip norm used in training differs from the one in the accountant, which changes the
sensitivity and therefore `sig`. Reasoning: epsilon depends on `T`, `q`, `sig`, and the accountant
version, and any one of them being wrong in an unlogged way reproduces this symptom. Follow-up: what makes
the recomputation authoritative? A stored ledger with the accountant version and a configuration hash.
Red flag: reconciling by picking the accountant that yields the smaller number. Scoring: two points for
the causes, two for the reconciliation procedure, one for the direction of the discrepancy.

**Q15. When would you choose asynchronous aggregation despite the staleness?**

Answer: when the straggler term dominates the round and the model tolerates approximate gradients, for
example a large cross-device deployment where the 99th percentile client is an order of magnitude slower
than the median. The evidence requirement is a measurement showing that the compute term is more than
half the round and that dropping the slowest percentile does not bias the objective. Reasoning: synchrony
converts a distribution of latencies into its maximum, which is expensive at large `C`. Follow-up: what
does the accountant cost? Each applied update is still a query, so the budget is unchanged, but the
staleness bound must be enforced or the update is a query against stale weights, which complicates the
narrative rather than the number. Red flag: choosing asynchrony to avoid the straggler without measuring
the straggler. Scoring: two points for the condition, two for the evidence requirement, one for the
accounting caveat.

## 12. Quick Reference Card

| Item | Value or rule | Why it matters |
|---|---|---|
| Round payload, flat | `2 * C * M` | the fan-in is the largest bandwidth lever |
| Round payload, hierarchical | `2 * F * M` | divides server bytes by the aggregation factor |
| Round duration | `M*8/up + M*8/down + E[max compute] + agg` | the max, not the mean, sets the duration |
| FedAvg weights | `n_i / n_S` from the sizes used in training | unweighted averaging is a different objective |
| Local-step noise penalty | `sqrt(E)` in steady state | five steps cost 2.236 times the noise |
| Masking invariant | `sum u_i = sum v_i (mod R)` for skew-symmetric masks | the server learns the sum and nothing else |
| Threshold rule | `t_thr <= E[survivors] - 4 sd` | the abort rate is a lower tail on survivors |
| Amplification factor | about 45 at `q = 4e-5`, `sig = 1.1` | without it, epsilon is 1242.8 instead of 0.601 |
| Quantisation ordering | quantise before noise, always | the step must stay well below `sig*S/C` |
| Coverage | `T(c) = ln(1-c)/ln(1-q)` | the last ten percent takes as long as the first ninety |
