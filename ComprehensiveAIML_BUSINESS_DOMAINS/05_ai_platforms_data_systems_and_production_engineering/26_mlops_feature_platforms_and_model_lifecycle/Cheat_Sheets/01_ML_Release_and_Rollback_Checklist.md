---
collection: "mlops-feature-platforms-and-model-lifecycle"
file_id: "01"
topic: "ML release and rollback checklist"
level: 2
status: "draft; unreviewed"
version: "2.0"
created: "2026-11-08"
reviewed: null
review_trigger: "Any change to the release bundle definition, to the SLO used for the error budget, or to the rollout/rollback mechanism named in Sections 4 and 7."
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["release-management", "rollback", "canary", "error-budget", "model-lifecycle", "incident-response"]
source_status: "course-topic draft; visible directory inventory consulted"
owner_role: "ML platform and model lifecycle engineering"
source_index: "INDEX.md"
---

# ML Release and Rollback Checklist

> **Purpose:** Turn "ship the model" into a bounded, auditable decision: what exactly is being released, which evidence authorises promotion, how much harm a bad release can do before it is stopped, and what specifically must be restored when it is stopped.
> **Prerequisites:** Ability to read a metric definition (numerator, denominator, population, window); familiarity with a training/evaluation split; basic arithmetic with rates, percentages, and units of time.
> **Key outcome:** Given a candidate model, its evaluation evidence, an SLO, and a traffic volume, you can compute the error budget, size a canary to a stated detection power, bound the blast radius in requests and money, define a stop condition with a named owner, and prove that rollback restores a coherent bundle rather than only a model alias.
> **This file does NOT cover:** The internals of any particular model registry, feature store, or deployment controller; statistical model validation in depth (see the sibling statistics sheets); nor organisational approval policy, which varies by institution and is not a technical property.
> **Level:** 2 — core methods, with the production diagnostics carried far enough to run a release review.
> **Version / reviewed:** 2.0 / not independently reviewed. Arithmetic in Sections 5 and 9 was recomputed and cross-checked; that is a self-check, not peer review.
> **Evidence:** Invariants are DERIVED from definitions given here. All volumes, rates, prices, and dwell times are ILLUSTRATIVE constructed values chosen for clean arithmetic and labelled as such. No incident, benchmark, vendor capability, or production outcome is claimed anywhere in this file.
> **Retrieval cue:** Bundle contents → Section 3; gate evidence → Sections 4 and 7; canary sizing → Section 4.3 and Section 5; "we rolled back and it is still broken" → Section 8; cost of the pipeline → Section 9.

## 1. File Header

The reference boundary of the previous version is retained: `Train/Templates/directory.txt` is an *inventory*, listing topics that exist in the wider corpus. It is not a body of inspected text, and nothing below reconstructs unseen material. Every claim is either a derivation from a definition stated in this file, a standard result stated with its hypotheses, or an explicitly labelled constructed number. Where a claim would require observing a real system — actual detection latency, actual canary dwell time, actual cost per prediction — the sheet either omits it or marks it as an assumption with a named test that would replace it.

The operational thesis is narrow and, in practice, contested. **A model is never the unit of release.** The unit of release is a *bundle*: a closed set of the artifacts and configurations whose combination determines runtime behaviour, identified by an immutable manifest. Everything awkward about ML releases follows from the gap between the artifact people name ("model v7") and the set of things that actually had to be true for the observed behaviour ("model v7, feature definitions at commit `a91c`, the tokenizer at revision `b3`, the retrieval index snapshot from 02:00 UTC, the prompt template with the new tool description, the guardrail policy with the raised threshold, and runtime image `c7`"). Releasing the first and rolling back to it while the fourth and fifth changed produces the signature failure of this domain: a rollback that restores the alias and does not restore behaviour.

Four decision types recur, and the sheet is organised around them:

1. **Authorise.** Is the evidence sufficient to expose this bundle to users? This is a gates question (Section 4, Section 7), and the answer is a conjunction of independent gates, never a single composite score.
2. **Bound.** How much harm can occur before the release is stopped? This is an exposure question, and it is arithmetic: traffic share times duration times error rate, in requests, in currency, in affected accounts (Section 4.4, Section 5).
3. **Observe.** Will we actually notice? This is a detection question, and it determines both the canary size and the monitoring design (Section 4.3, Section 8's F2 and F6).
4. **Reverse.** What exactly do we restore, and is that restoration coherent? This is the rollback question, and it is the one that is usually asserted rather than tested (Section 4.5, Section 8's F1).

The checklist form is deliberate. A checklist is a *forcing function against omission under time pressure*, and its value comes from items that are cheap to check and expensive to omit. It is not a document that encodes judgment; judgment lives in the gate thresholds and the stop conditions, which are decisions made in advance, in calm conditions, by people who will not be the ones deciding at 03:00. A checklist that contains "ensure the model is good" is worthless because it is unverifiable. A checklist that contains "manifest hash recorded; `kappa` of the eval delta reported; canary cohort excludes the top 1% of accounts by volume; rollback drill executed within 14 days" is worth its maintenance cost because each item can be witnessed by a second person.

Two framing conventions run through the file. First, **the release is bounded by time, not by version**: an SLO window of 30 days, a canary dwell of 15 minutes, a rollback objective of 20 minutes. Time bounds are what make exposure arithmetic possible; version numbers do not bound anything. Second, **every threshold has a unit and a comparator**. "Accuracy must not drop by more than 0.3 percentage points on the frozen evaluation set, measured at the same threshold with the same population definition" is a gate. "Accuracy must not degrade meaningfully" is a wish, and it will be interpreted favourably by whoever is under deadline.

> **Takeaway:** Write the release down as a closed bundle with an immutable manifest, and treat "which artifacts changed" as the first question of every rollback conversation.

## 2. Concept Map / Mental Model

The mental model is a pipeline with three one-way doors and one rehearsal loop. Materials enter on the left (artifact, features, index, prompt, policy, image). Three gates must pass before exposure. Exposure climbs in stages, each with its own stop condition. The rehearsal loop is the rollback drill, which runs *before* the release is approved and is the only thing that converts "we could roll back" from a belief into evidence.

```text
  candidate bundle
  +----------------------------------------------------------+
  | model artifact   feature defs   tokenizer   index snapshot|
  | prompt template  guardrail cfg  runtime img  policy       |
  +----------------------------------------------------------+
                              |
                              v
                   [GATE 1: closure & reproducibility]
                    - manifest hash present?
                    - every runtime dependency pinned?
                    - can the bundle be rebuilt from the manifest?
                     |                              |
                    no                             yes
                     |                              |
              block; the thing being            v
              evaluated is not the thing    [GATE 2: compatibility]
              being released                 - schema producer/consumer
                     ^                       - API contract of scores
                     |                       - expand/contract satisfied
                     |                        |                  |
                     |                       no                 yes
                     |                        |                  |
                     |              expand first, or            v
                     |              version the API        [GATE 3: quality, safety, ops]
                     |                        |            - frozen eval + slices
                     |                        |            - abuse/privacy controls
                     |                        |            - latency, cost, capacity
                     |                        |            - named owner, runbook
                     |                        |             |            |
                     |                        |            no           yes
                     |                        |             |            |
                     |                        +-------------+            v
                     |                                                     |
                     |                                     +---------------+
                     |                                     |
                     |                                     v
                     |                          shadow (no side effects)
                     |                                     |
                     |                                     v
                     |                          canary at declared %
                     |                          (bounded dwell, stop rule)
                     |                                     |
                     |                              +------+------+
                     |                              |             |
                     |                           stop rule     no breach
                     |                            fires          |
                     |                              |             v
                     |                              |     staged expansion
                     |                              |     25% -> 50% -> 100%
                     |                              |             |
                     |                              v             v
                     |                      +-------------------------------+
                     +--------------------->|  ROLLBACK: restore bundle    |
                                            |  or enter degraded mode      |
                                            |  T_total = detect + decide   |
                                            |          + execute           |
                                            +-------------------------------+
                                                        ^
                                                        |
                                            rehearsal: rollback drill
                                            executed before approval
```

Read the map as a claim about where releases fail. Gate 1 failures are embarrassing but cheap: they are caught by inspection. Gate 2 failures are the ones that take an hour to diagnose because the symptom (a consumer crashing on a renamed field, or a score that is now on a different scale) looks like a model problem. Gate 3 failures are the ones the process exists for. And the rollback branch is the one that is most often drawn but least often exercised, which is why it is drawn here as the terminal state with a rehearsal arrow feeding into it.

The second map is a state machine, because the interesting question about a release is not "did it succeed" but "which state is it in, and what are the legal transitions from here". A release that has no legal transition except forward is not a release, it is a commitment.

```text
   PROPOSED --approve--> APPROVED --deploy--> SHADOWING
                          ^                       |
                          |                       | shadow verdict
                     reject (with reason)         v
                          |                    CANARYING
   REJECTED <-------------+                    /        \
                                               |          |
                                    stop rule  |          | no breach
                                     fires     |          |
                                               v          v
                                          ROLLING_BACK  EXPANDING
                                               |          |
                                               |          | 100% + dwell
                                               v          v
                                          ROLLED_BACK   RELEASED
                                               |          |
                                               +----> (superseded)
                                                    |
                                                    v
                                              SUPERSEDED

  Illegal transitions, stated so they can be refused:
    CANARYING  -> APPROVED        (no such edge; you must roll back or expand)
    ROLLED_BACK-> CANARYING       (same bundle twice without a new reason)
    RELEASED   -> SHADOWING       (a live bundle is not re-shadowed in place)
    any        -> PROPOSED        (a new bundle gets a new manifest, not a reset)
```

The third map is the exposure cone, which is the mental model that makes rollout sizing concrete. At each stage the product of traffic share and elapsed time bounds the population that can be harmed. The cone widens because both factors grow, and the design question is where to put the stop rule: early enough that the cone is still narrow, late enough that the signal is statistically readable.

```text
  exposed
  requests
     ^
     |                                              /
     |                                            /
     |                                          /  100%
     |                                        /
     |                                    ___/    50%
     |                              _____/
     |                        _____/              25%
     |                 ______/
     |          ______/                            5%  canary
     |    _____/
     |___/                                          shadow: 0 external effects
     +-----------------------------------------------------> time

  blast radius at a stage = traffic share x duration x error rate x harm/error
  the stop rule must fire inside the narrow part of the cone
```

A fourth orientation point ties the maps to the sections. Every quantity in this sheet belongs to one of four accounting ledgers, and mixing ledgers is the most common analysis error: the **reliability ledger** (error budget in minutes of unavailability), the **quality ledger** (metric deltas with their uncertainty), the **harm ledger** (requests, accounts, currency), and the **cost ledger** (compute-hours, engineer-hours). An error budget in minutes cannot answer "how many customers were affected", and a harm figure in currency cannot answer "did we burn the SLO". Section 9 keeps the four separate on purpose.

> **Takeaway:** Draw the release as a state machine with legal transitions and a rehearsal arrow into rollback; a release whose only transition is forward is a commitment, not a release.

## 3. Foundational Definitions

Each definition names the distinction it prevents. A checklist item that cannot be tied to one of these terms is probably decoration.

| Term | Definition | Unit / range | Distinction that prevents an error |
|---|---|---|---|
| Release bundle | The closed set of artifacts and configuration whose combination determines runtime behaviour | set of pinned references | The unit of release; a model version alone is not a bundle |
| Manifest | Immutable document listing every bundle member with an immutable identifier (hash, revision, snapshot ID) | document | A manifest is a *claim of completeness*; completeness is what gate 1 tests |
| Artifact | A stored, versioned, immutable file or object (model weights, tokenizer, image) | opaque | Artifacts are immutable; *aliases* are mutable pointers to them |
| Alias / pointer | A mutable name that resolves to an artifact ("production", "latest") | name | Rolling back an alias changes routing; it does not change what the alias points at unless the target bundle is coherent |
| Runtime dependency | Anything the serving path reads that can change behaviour without a new artifact (index, cache, config service, remote model) | enumerated list | The most commonly omitted bundle members, because they live outside the registry |
| Feature definition | The declared computation that turns raw inputs into model features, at a specified revision | revision | Training and serving must use the same definition; a mismatch is not a bug in either one |
| Schema contract | The agreed names, types, nullability, and units at a producer/consumer boundary | versioned contract | Compatibility is a property of the contract, not of the values currently flowing |
| Expand/contract | The migration pattern: add the new field, dual-write, migrate readers, then remove the old field | phase | Removes the need to coordinate two deploys in the same minute; skipping it forces an outage |
| Frozen evaluation set | An evaluation dataset with a recorded version, population definition, and exclusion list, not altered between candidates | dataset ID | A moving evaluation set makes two candidates incomparable and every gate meaningless |
| Gate | A pass/fail test with a declared threshold, comparator, population, and window | decision | Gates are conjunctions; averaging them into a score destroys the veto property |
| Slice | A defined subpopulation on which the metric is reported separately | filter expression | Aggregate parity can hide a slice regression, so slices are gate items in their own right |
| Canary | A release stage exposing a bounded traffic share to the new bundle | share in [0,1] | A canary is a *statistical instrument* with a power requirement, not a formality |
| Cohort | The rule that selects which requests enter the canary | predicate | If the cohort does not resemble exposed traffic, the canary measures the wrong population (F2) |
| Shadow / dark launch | Running the new bundle on a copy of live traffic with no external side effects | boolean | Safe only if the no-side-effect property is *enforced*, not intended (F3) |
| Stop condition | A pre-declared rule that aborts the release, with threshold, window, and owner | rule | Deciding to stop during an incident is slower and more biased than reading a pre-declared rule |
| Blast radius | The population or value exposed to a defect before containment | requests, accounts, currency | Makes exposure a computable quantity rather than a phrase |
| SLO | Service-level objective: a target for a measured reliability indicator over a window | target + window | The window is part of the definition; 99.9% "overall" is undefined |
| SLI | The measured indicator (success ratio, latency percentile) | measurement | SLI is what you measure; SLO is what you promised; both are needed |
| Error budget | The unreliability an SLO permits in its window | minutes or failed requests | A budget is *consumed*, not "available capacity"; spend it on releases deliberately |
| Burn rate | Speed of budget consumption relative to the rate that would exhaust it exactly at window end | multiple | Burn rate is the right alarm variable because it is window-normalised |
| MTTD | Mean time to detect: from defect onset to a signal being visible to a responder | minutes | Precedes and is independent of the decision time |
| MTTR | Mean time to restore: from detection to service restored | minutes | Restoration may be rollback, degraded mode, or forward fix; they differ in risk |
| RTO | Recovery time objective: the declared bound on restoration time | minutes | An RTO that was never measured is a hope; the drill measures it |
| Degraded mode | A declared reduced-function state that contains harm without restoring full service | mode name | Legitimate rollback outcome when the previous bundle cannot be restored |
| Rollback | Restoring a coherent previously approved bundle, or entering a declared degraded mode | action | Must restore *behaviour*, not merely an alias (F1) |
| Forward fix | Repairing the defect in a new bundle released forward | action | Often faster, but only legitimate when the fix is understood and testable |
| Release record | The durable evidence bundle for one release decision: manifest, approver, evidence, cohort, stop rules, times | document | Without it, the release cannot be reconstructed or audited |
| Post-release review | Structured review after a release or incident, producing named changes | process | Distinct from a status update; it must produce a change or a recorded decision not to change |
| Dwell time | Time spent at a rollout stage before advancing | minutes | Too short and the signal is unreadable; too long and the release stalls |

Two definitional notes that resolve most disputes before they start. First, **a model version number is not a behaviour identifier.** Two different behaviours can carry the same version if any other bundle member changed, and one behaviour can carry two version numbers if only metadata changed. The immutable manifest is the behaviour identifier, which is why gate 1 refuses to proceed without it. Second, **an SLO is not an average.** "99.9% available" over a 30-day window is a statement about the fraction of good events (or good time) in that window, and a burst of unavailability that consumes the whole budget on day two is fully compliant with the *arithmetic* of the objective while being catastrophic in the *experience* of it. Burn-rate alarms exist to bridge that gap (Section 4.6).

A third note on what counts as evidence. There are four evidentiary states a release item can be in, and they must not be conflated:

| State | Meaning | Permitted claim |
|---|---|---|
| Executed and recorded | The check ran, and its output is stored with the release | "This was verified on this bundle at this time" |
| Executed, not recorded | The check ran in someone's terminal | "Someone believes this was verified" — treat as unverified for a high-risk release |
| Declared | The property is asserted in a document with no test | "This is intended", not "this is true" |
| Unknown | Nobody has looked | Not a pass; blocking for gates 1–3 on a high-risk release |

The checklist's power comes almost entirely from converting "declared" into "executed and recorded". A release process that says "rollback tested" without an artifact showing the drill's timestamp, operator, restored manifest hash, and measured restoration time is operating in the second row, and the second row is where rollback surprises come from.

> **Takeaway:** Define bundle, manifest, gate, cohort, stop condition, and rollback before writing any checklist item, because every vague checklist item is a term that was never defined.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case: One Gate, One Window, No Stages

Start with the smallest useful object: a single binary outcome observed over a fixed window, with a threshold. Everything later generalises this. Let `n` be the number of requests in the window, `k` the number that failed, and let the target be that the failure ratio `p = k/n` must not exceed `p0`. The base question is: given the observed `(k, n)`, is `p > p0`?

**Derivation of the decision rule.** If each request fails independently with probability `p`, then `k` is binomial `(n, p)`. Under the hypothesis `p = p0`, the expected count is `n p0` and the standard deviation is `sqrt(n p0 (1 - p0))`. So the standardised deviation is

```text
z = (k - n p0) / sqrt(n p0 (1 - p0))                [ID: RL-01]
```

and `z > z_alpha` means the observed failure count is above what the SLO permits by more than `z_alpha` standard deviations. The reason this is a *gate* and not a test of "is the model good" is that we are asking a narrow question: is the failure rate demonstrably above the contract, on this population, in this window.

```text
[1] FORMULA: z = (k - n*p0) / sqrt(n*p0*(1-p0))          [ID: RL-01]
[2] WHERE: k = failures observed (count, integer >= 0); n = requests observed
           (count); p0 = permitted failure ratio (dimensionless, 0<p0<1);
           z = standardised deviation (dimensionless). Requests must be the
           same unit that the SLI counts, which is a decision (per HTTP call
           vs per user session), not a given.
[3] CONSTRAINT: independent failures within the window, a fixed n known in
           advance or at least not chosen *after* looking at k, and a window
           long enough that n*p0 >= 5 so the normal approximation to the
           binomial is reasonable. Failures that cluster (one bad host causing
           many failures) violate independence and inflate z spuriously.
[4] NUMERICAL EXAMPLE: n = 20000 requests, k = 34 failures, p0 = 0.001.
           n*p0 = 20, so z = (34 - 20)/sqrt(20*0.999) = 14/4.4708 = 3.131.
           With z_alpha = 1.645 (one-sided 5%), the gate trips: 3.131 > 1.645.
[5] INTERPRETATION: the failure rate is 0.17% against a 0.1% contract, and the
           excess is not plausibly noise at this sample size; the release is
           stopped and the budget arithmetic in 4.2 is consulted to size the
           damage already done.
```

The base case has a symmetry worth noting because it drives the canary design later. `z` grows with `sqrt(n)` for a fixed absolute deviation in the failure ratio. So a *small* canary cannot detect a *small* regression, no matter how long you stare at it. That single sentence explains most canary failures in practice, and Section 4.3 turns it into a sizing rule.

**Derivation of the error budget in minutes.** For an SLO stated as an availability ratio `a` over a window of `W` minutes, the permitted unavailable time is derived by rearranging `a = good_time / W`:

```text
bad_time = W * (1 - a)                               [ID: RL-02]
```

For `W = 30 * 24 * 60 = 43200` minutes and `a = 0.999`, `bad_time = 43200 * 0.001 = 43.2` minutes. The derivation is one rearrangement, and the reason it deserves a formula block is that the *unit* chosen for `W` silently chooses the granularity of the promise. A window of 30 days at 99.9% permits 43.2 minutes; the same 99.9% over a *calendar month* varies between 40.3 and 44.6 minutes depending on the month's length, so a monthly promise without a fixed minute count is ambiguous.

```text
[1] FORMULA: B = W * (1 - a)                             [ID: RL-02]
[2] WHERE: B = error budget (minutes of permitted unavailability);
           W = SLO window (minutes); a = availability target as a ratio
           (0 < a < 1, so 99.9% enters as 0.999, not as 99.9).
[3] CONSTRAINT: the SLI must be time-based availability for this form. If the
           SLI is a request success ratio, the budget is in failed requests
           or in "good requests shortfall", not in minutes, and converting
           requires a traffic model. Both conventions are used in industry and
           mixing them produces budgets that differ by orders of magnitude.
[4] NUMERICAL EXAMPLE: W = 30 days = 43200 min, a = 0.999.
           B = 43200 * 0.001 = 43.2 min. At a = 0.9999: B = 4.32 min.
           At a = 0.99: B = 432 min = 7.2 hours.
[5] INTERPRETATION: an order-of-magnitude step in the target is an
           order-of-magnitude step in the budget, so the target is a cost
           decision, not a quality aspiration; 99.99% buys 38.9 fewer minutes
           per month than 99.9% and typically costs far more than 10x.
```

**Derivation of the request-based budget.** If the SLI is a success ratio over requests, the permitted failure count follows the same rearrangement applied to counts:

```text
F_max = N * (1 - a)                                  [ID: RL-03]
```

for `N` requests in the window. `F_max` is a count, and counts are what convert into harm (Section 4.4). The conversion from `B` minutes to `F_max` requests is *not* `N * (B/W)` unless traffic is perfectly uniform; under a diurnal pattern, an outage at peak consumes the same minutes but far more requests. This is why the two ledger conventions must be declared rather than assumed.

**A boundary worth computing once.** For `a = 0.999` over 30 days and a service handling `N = 20,000,000` requests in that window, `F_max = 20,000` failed requests. The budget is `43.2` minutes *or* `20,000` requests, and these two statements are consistent only if the average request rate during outages equals the average rate overall. In a real system the first 20 minutes of an outage often occur at peak, so a time-based budget can be spent while a request-based budget is over-consumed by a factor of two or more. The checklist item that follows: **declare which convention the release is judged by, before the release starts.**

**Base-case limit.** The `z` computation assumes failures are independent. Consider a canary where a single host serves 40% of the canary's requests and fails them all, while other hosts fail at `p0`. The observed `k` may exceed `n p0` by several standard deviations, but the estimate of `p` has an effective sample size close to the number of *hosts*, not the number of requests. Correct treatment requires clustering by the unit that actually fails (host, shard, model instance), which changes `sqrt(n)` into roughly `sqrt(clusters)`. The practical consequence: an alarm that fires on `z` computed over requests will fire on single-host problems while remaining blind, at small canary sizes, to systemic problems. Both signals are needed, and they must be reported separately.

> **Takeaway:** A gate is a standardised deviation with a declared unit of observation; choosing "request" where "host" is the failing unit inflates significance and misleads the stop decision.

### 4.2 General Case: Bundle Closure, Compatibility, and the Rollback Invariant

The base case handled one number. The general case handles the object that actually ships, and its central result is a closure condition.

**Closure condition (derivation).** Let the runtime behaviour `b` be a function of the bundle's members: `b = f(m_1, ..., m_k)` where each `m_i` is a pinned reference. A release is *closed* iff every argument of `f` appears in the manifest. The derivation is by contradiction: suppose some argument `m_j` used at runtime is absent from the manifest. Then the manifest does not determine `b`, because `m_j` can change while every manifest entry stays fixed. Two releases with the same manifest can then exhibit different behaviour, so the manifest is not a behaviour identifier, and gate 1's premise — that the thing evaluated is the thing released — fails. Therefore closure is necessary for the manifest to serve its purpose.

The condition is easy to state and hard to satisfy because `f` is not written down anywhere. The practical derivation is an enumeration of the *sources* from which `f` reads:

```text
b = f(  model weights          <- registry, immutable by hash
        tokenizer / vocab      <- registry, immutable by revision
        preprocessing code     <- container image, immutable by digest
        feature definitions    <- feature store, VERSIONED BUT MUTABLE BY DEFAULT
        feature values         <- online store, changes every minute
        retrieval index        <- external, snapshot ID required
        prompt / template      <- config service, MUTABLE
        guardrail policy       <- config service, MUTABLE
        thresholds / cutoffs   <- config service, MUTABLE
        remote dependencies    <- external services, versioned by them
     )
```

The four mutable sources are where closure fails. Feature *values* are deliberately outside closure — they are inputs, not configuration — but feature *definitions* must be inside it, and the distinction is exactly the checklist item "pin the definition, not the values". A retrieval index snapshot is inside closure because different snapshots yield different behaviour for the same query; a `latest` pointer to an index is therefore the same class of defect as an unpinned dependency in a software build.

```text
[1] FORMULA: closed(manifest) <=> {m_1..m_k} ⊇ args(f)   [ID: RL-04]
[2] WHERE: args(f) is the set of runtime inputs that can change behaviour
           without a new artifact being published; the manifest is the
           declared subset; k is the number of declared members.
[3] CONSTRAINT: closure is relative to a stated serving path. A batch scoring
           job and an online endpoint have different f and therefore different
           manifests. Closure for one does not imply closure for the other.
[4] NUMERICAL EXAMPLE: a manifest declares 9 members; the serving path reads
           12. Closure fails with 3 undeclared arguments. If those 3 are the
           feature definition, the prompt template and the guardrail policy,
           then any rollback that restores only the 9 declared members leaves
           3 of 12 behaviour inputs at their new values, i.e. the restored
           bundle agrees with the pre-release bundle on 9 of 12 inputs.
[5] INTERPRETATION: "rollback restored the model" is compatible with a
           behaviour that is 25% new; closure is what turns the word
           "rollback" into a claim about behaviour rather than about a pointer.
```

**Compatibility condition (derivation).** Let a producer emit a payload with field set `P` and a consumer require field set `C`. The consumer fails when `C \ P` is non-empty, i.e. it needs a field that is not present. A release that removes a field from the producer is safe only if no live consumer requires it. Since consumers are released on their own schedule, the safe ordering is forced: expand (producer adds the new field and keeps the old), migrate (consumers switch to the new field), contract (producer removes the old field) — and the contract step may proceed only after every consumer's *observed* requirement set has moved. Observed, not declared: a consumer's code may require the field without any documentation saying so, which is why the gate item is a compatibility test that exercises the consumer on a payload without the field.

```text
[1] FORMULA: safe_remove(P, C_1..C_j) <=> ∀i: field ∉ C_i   [ID: RL-05]
[2] WHERE: P is the producer's emitted field set; C_i are the requirement
           sets of the j live consumers, as observed by tests; the removal
           candidate is a single field.
[3] CONSTRAINT: the consumer set must be *enumerated*. If any consumer is
           undiscoverable (an external partner, a notebook, a job whose owner
           left), the set is incomplete and the condition cannot be established;
           in that case the correct action is the deprecation window, not the
           test.
[4] NUMERICAL EXAMPLE: 7 known consumers; 6 pass a payload-without-field test;
           1 crashes. removal not safe; j = 7, violation count 1. If the 7th
           is discovered only in production, the gate has converted a
           controlled migration into an outage whose duration equals the time
           to find and redeploy that consumer.
[5] INTERPRETATION: the compatibility gate's value is measured by how many
           consumers it enumerates before the release, not by how many tests
           it runs after.
```

**Rollback invariant (derivation).** Define the pre-release bundle `B0` and the attempted bundle `B1`, differing in a set of members `Delta`. Define rollback success as "the system's externally observable behaviour returns to what `B0` produced". Rollback that restores a *subset* `R ⊂ B0` succeeds iff `Delta ∩ (args(f) \ R) = ∅` — that is, no member whose change caused an observable difference is left un-restored. The alias-only rollback is the special case `R = {model}`; it succeeds only if `Delta ⊆ {model}`, i.e. only if nothing else in the bundle changed.

```text
[1] FORMULA: rollback_success(R) <=> Delta ∩ (args(f) \ R) = ∅   [ID: RL-06]
[2] WHERE: Delta = set of bundle members that changed between B0 and B1;
           R = set of members actually restored; args(f) \ R = behaviour
           inputs left at their B1 values.
[3] CONSTRAINT: assumes all restored members return to their exact B0 values
           (immutable references make this true; a mutable "rollback to
           latest-stable" does not). Also assumes no irreversible side effect
           occurred during exposure — emails sent, orders placed, records
           written. Side effects are outside this formula and are handled by
           containment and compensation, not by rollback.
[4] NUMERICAL EXAMPLE: Delta = {model, prompt_template, guardrail_policy}.
           R = {model}. Then Delta ∩ (args(f) \ R) = {prompt, policy},
           non-empty, so rollback fails while the deploy tool reports success
           and the alias now points at the previous model.
[5] INTERPRETATION: this is the formal statement of failure mode F1, and it is
           the reason the checklist requires the manifest to be enumerable:
           without enumerating Delta you cannot evaluate the condition at all.
```

**Irreversibility (derivation of the compensation requirement).** Some effects during exposure cannot be undone: a notification already delivered, an offer already accepted, a payment already captured. Model the exposure as producing `E` external effects. A rollback restores *state*; it does not retract effects. The corrective action for an irreversible effect is compensation — a corrective message, a reversal transaction, an apology with a credit — which is a *separate* process with its own cost and its own failure modes. Therefore every release that can emit irreversible effects requires a compensation path *before* approval, not after the incident.

```text
[1] FORMULA: residual_harm = E_irreversible * c_effect - E_compensated * c_recovery
                                                              [ID: RL-07]
[2] WHERE: E_irreversible = count of side effects that occurred and cannot be
           deleted (count); c_effect = cost per unretracted effect (currency
           per effect); E_compensated = count successfully compensated (count);
           c_recovery = value recovered per compensation (currency per effect).
[3] CONSTRAINT: residual_harm is a lower bound on damage when compensation is
           partial; the formula assumes independence between effects. It is an
           accounting identity, not a prediction.
[4] NUMERICAL EXAMPLE: illustrative 1,200 messages already sent at an
           illustrative cost of 0.40 currency units each; compensation reaches
           900 of them at a realised recovery of 0.25 each. residual_harm =
           1200*0.40 - 900*0.25 = 480 - 225 = 255 currency units remaining
           plus the non-monetised trust cost, which the formula does not capture.
[5] INTERPRETATION: the exposure design question is not "can we roll back" but
           "how many irreversible effects can occur before the stop rule
           fires", which is Section 4.4's blast radius applied to effects
           rather than to requests.
```

**Base-case connection.** Section 4.1's `z` and this section's closure and invariant conditions are the same object at different levels: `z` asks whether an *observed* quantity deviates, closure asks whether the *bundle being observed* is the one being judged, and the invariant asks whether the *action taken* addresses the deviation. A release process that gets the statistics right and the bundle wrong will stop the right release and fix the wrong thing.

> **Takeaway:** Write the bundle down and enumerate it; a rollback is only as coherent as the set of members you can list, so the manifest is a rollback control before it is a documentation artifact.

### 4.3 Variants: Canary Sizing, Burn-Rate Alarms, and Staged Exposure

**Canary sizing (derivation).** Suppose the pre-release failure ratio is `p0` and the release would raise it to `p1 = p0 + d`. A canary of `n` requests observes a count `K` and trips the stop rule when `K > n p0 + z_alpha sqrt(n p0 (1-p0))`. The probability of detecting the regression is

```text
power(n) = P( K > threshold | p = p1 ) ≈ 1 - Phi( (threshold - n p1) / sqrt(n p1 (1-p1)) )
```

For small base rates and small `n d`, the threshold is close to `n p0` plus a margin, so the detectable effect at a given power scales as `d ≈ (z_alpha + z_beta) sqrt(p0 (1-p0) / n)`. Rearranging for the sample size:

```text
n_min ≈ (z_alpha + z_beta)^2 * p0 (1 - p0) / d^2      [ID: RL-08]
```

This is the one formula in the sheet that most changes decisions, because it is routinely violated: teams run a canary of a few thousand requests with a 10-minute dwell and believe they would catch a 20% relative degradation.

```text
[1] FORMULA: n_min ≈ (z_alpha + z_beta)^2 * p0(1-p0) / d^2   [ID: RL-08]
[2] WHERE: n_min = requests required in the canary (count); p0 = baseline
           failure ratio (dimensionless); d = absolute increase in failure
           ratio to be detected (dimensionless, i.e. 0.002 for 0.2 percentage
           points); z_alpha, z_beta = normal quantiles for the false-positive
           and false-negative rates.
[3] CONSTRAINT: independent observations, small base rate so the normal
           approximation holds, a single look at the data or a corrected
           alpha for multiple looks, and a stable baseline (an unrelated
           incident during the canary invalidates the comparison). With
           z_alpha = 1.645 (5% one-sided) and z_beta = 0.8416 (80% power),
           z_alpha + z_beta = 2.4866 and its square is 6.1832.
[4] NUMERICAL EXAMPLE: p0 = 0.001, d = 0.0002 (a 20% relative increase).
           n_min = 6.1832 * 0.001*0.999 / (0.0002^2)
                 = 6.1832 * 0.000999 / 4e-8
                 = 6.1770e-3 / 4e-8 = 154,425 requests.
           At 50 requests/second this is 154425/50 = 3,088.5 s = 51.5 minutes.
           For a 2.0x relative increase (d = 0.001): n_min = 6.1832*0.000999/
           1e-6 = 6,177 requests, i.e. 123.5 s at the same rate.
[5] INTERPRETATION: detecting a doubled failure rate takes two minutes; detecting
           a 20% increase at the same base rate takes most of an hour. The
           canary dwell is therefore not a taste variable — it is implied by
           the effect size you claim to catch, and a 10-minute canary at this
           base rate can only catch effects of roughly 45% relative or larger.
```

The interpretation deserves an arithmetic follow-through, because it is the point of the section. Solving for the detectable `d` at a fixed `n = 50 req/s * 600 s = 30,000` requests:

```text
d_min = (z_alpha + z_beta) * sqrt( p0 (1-p0) / n )
      = 2.4866 * sqrt( 0.001*0.999 / 30000 )
      = 2.4866 * sqrt( 3.33e-8 )
      = 2.4866 * 1.8257e-4
      = 4.54e-4
```

So a 10-minute canary at 50 req/s with a 0.1% baseline detects an absolute increase of about 0.045 percentage points, i.e. roughly a 45% relative degradation of the failure rate, at 80% power. Stating that number in the release record converts "canary for 10 minutes" from a ritual into a quantified claim, and it exposes the common situation where the claim the team wants to make (catch a 10% regression) is incompatible with the canary they are willing to run.

**Burn-rate alarm (derivation).** Normalise budget consumption by the rate that would exhaust the budget exactly at the end of the window. At time `t` into a window of length `W`, with budget `B` (in the same unit), the burn rate is

```text
b(t) = (consumed(t) / B) / (t / W) = W * consumed(t) / (B * t)   [ID: RL-09]
```

A burn rate of 1.0 spends the budget at exactly the sustainable pace. A burn rate of 14.4 exhausts a 30-day budget in 50 hours, which is the origin of the familiar "1-hour window at 14.4x burn" multi-window alarm pattern: the number comes from `30 days / 50 hours = 14.4`, and it is a design choice about how quickly you want to know, not a natural constant.

```text
[1] FORMULA: b = W * consumed / (B * t)                  [ID: RL-09]
[2] WHERE: b = burn rate (multiple, dimensionless); W = window length
           (minutes); t = elapsed time in window (minutes); consumed and B in
           the same unit (minutes unavailable, or failed requests).
[3] CONSTRAINT: consumed and B must use the same SLI convention (RL-02 vs
           RL-03). Using minutes for B and requests for consumed produces a
           number with no meaning. Requires t > 0.
[4] NUMERICAL EXAMPLE: W = 43200 min, B = 43.2 min. After t = 60 min, 5.0
           minutes have been consumed: b = 43200*5.0/(43.2*60) = 216000/2592 =
           83.33. The budget would be gone in 43.2/5.0 * 60 = 518 s, i.e.
           under 9 minutes at this rate. A sustained b = 1.0 for the full
           window consumes the budget exactly.
[5] INTERPRETATION: burn rate is the correct alarm variable because it is
           window-normalised; a raw "minutes consumed" threshold means
           something different on hour 1 and hour 700, which is why alerts
           written on raw consumption fire late in the window for no reason.
```

**Staged exposure (derivation of the cumulative bound).** For stages `s = 1..S` with traffic shares `w_s` and dwell times `T_s`, the cumulative exposed request count is `sum_s w_s * T_s * lambda` for an arrival rate `lambda`, and the cumulative exposure *before* stage `s` is the partial sum. The design requirement is that the stop rule's detection time at each stage is shorter than the dwell at that stage:

```text
detect_time(s) < T_s     and     exposure(s) = w_s * T_s * lambda    [ID: RL-10]
```

If `detect_time(s) > T_s`, the stage advances before its own alarm could fire, so the stage contributes exposure without contributing protection. This is the mathematical statement of "we ran a canary but we were going to promote anyway": the schedule and the detection capability were never compared.

```text
[1] FORMULA: exposure(s) = w_s * T_s * lambda             [ID: RL-10]
[2] WHERE: w_s = traffic share at stage s (0..1, dimensionless); T_s = dwell
           at stage s (minutes); lambda = arrival rate (requests/minute);
           exposure in requests. Cumulative exposure is the sum over stages.
[3] CONSTRAINT: constant arrival rate over the stage; independent requests;
           the share is enforced at the routing layer, not just intended. If
           the routing is by user ID, the same users recur at every stage, so
           the *population* exposed differs from the request count, which
           matters for harm accounting (4.4).
[4] NUMERICAL EXAMPLE: lambda = 3000 req/min, stages 5%/15 min, 25%/30 min,
           50%/60 min, 100% (remaining). Exposure: 0.05*15*3000 = 225;
           0.25*30*3000 = 22500; 0.50*60*3000 = 90000; cumulative 112,725
           requests before full traffic. If the defect rate is 1% in that
           population, affected requests ≈ 1,127 before full release.
[5] INTERPRETATION: the first stage is cheap (225 requests) and the third is
           expensive (90,000); almost all the protection must come from the
           early stages, which is why the early stages need the longest dwell,
           not the shortest.
```

**Variant: shadow then canary (derivation of why both).** Shadowing computes the new bundle's output on live inputs without emitting effects, so it can compare outputs on 100% of traffic at zero external harm. Its blind spot is that it observes *outputs* but not *consequences*: latency under real load, downstream behaviour, and user response are absent. The canary supplies consequences but at a small sample. Hence the two stages are complementary rather than redundant, and the correct reading of a clean shadow is "the model computes what we expect on live inputs", not "the model is safe to expose".

> **Takeaway:** Size the canary from the effect you claim to catch; a dwell shorter than the detection time buys exposure without buying protection.

### 4.4 Limits and Counterexamples

**Limit 1 — an average metric cannot license a slice claim.** Suppose the aggregate failure ratio is unchanged at 0.1% while one slice holding 2% of traffic rises from 0.1% to 4.1%. Aggregate contribution from the slice rises by `0.02 * 0.04 = 0.0008`, i.e. 0.08 percentage points, which *would* show up. Now let the slice hold 0.2% of traffic: contribution `0.002 * 0.04 = 0.00008`, i.e. 0.008 percentage points, at the level of noise in a moderate canary. The aggregate gate passes while a small population experiences a 41x failure ratio. The general statement: a slice of share `w` with excess rate `e` moves the aggregate by `w*e`, so any slice with `w*e` below the detection floor is invisible in the aggregate *by construction*, not by bad luck.

**Limit 2 — the previous bundle may no longer be restorable.** Rollback assumes `B0` is still deployable: its image exists, its feature definitions are still served, its index snapshot is still retained, and downstream consumers still accept its schema. Any of these can be false after the contract step of an expand/contract migration. Concretely, if the release also *removed* a field that `B0` requires, then rolling back to `B0` breaks the new producers. This is not a rare edge: it is the normal consequence of a successful migration, and it means **the rollback target must itself be revalidated after every schema migration**, not merely recorded at the time it was released.

**Limit 3 — rollback can be slower than the forward fix.** If the defect is a threshold set in a mutable config, a forward fix is a config write; a full bundle rollback may require re-pointing an index, draining connections, and waiting for caches. The invariant in RL-06 does not say rollback is fastest; it says rollback is the action that restores coherence. A forward fix is legitimate when the defect is understood, the fix is testable in minutes, and the fix's own risk is bounded — but "we'll forward fix" chosen under pressure without those three conditions is how a small incident becomes a long one.

**Limit 4 — the canary cohort can be systematically unrepresentative.** Routing by request ID gives an unbiased sample of requests; routing by user ID gives a sample of *users*, and if the canary uses the first `w` fraction of a hashed ID space, it is unbiased in expectation. Routing by geography, device, or account tier is not a random sample, and the canary then measures a different population from the one that will be exposed at 100%. This is F2, and the limit statement is precise: the canary's estimate of the population effect is valid only for the population the cohort actually represents.

**Limit 5 — the error budget is not a permission slip.** Consuming budget is not evidence of harm containment. A service that is exactly at its budget has, by definition, delivered the unreliability it promised; but the *distribution* of that unreliability matters — 43 minutes in one outage is a different event from 43 minutes spread over 30 days, even though the arithmetic is identical. Burn-rate alarms and multi-window checks exist because the budget alone cannot distinguish them.

**Limit 6 — a clean shadow does not mean a safe release.** The shadow observes the model's outputs on real inputs. It cannot observe the consequences of acting on them. A recommender whose outputs are perfectly computed but systematically exclude a category produces a clean shadow and an unacceptable release.

**Limit 7 — pinning everything freezes the ability to fix things.** Perfect closure requires every mutable source to be pinned, but pinning the index snapshot means an index refresh requires a new release bundle. The operational resolution is to separate *pinned configuration* from *rotating data*: pin the *definition and selection rule* for the index, log the snapshot ID actually used at request time, and treat a snapshot change as a versioned event in the manifest rather than as an invisible mutation. The limit statement is that closure and operability pull in opposite directions, and the resolution is logging rather than prohibition.

**Limit 8 — significance is not harm.** A statistically significant degradation in a metric that does not affect the decision is not a reason to roll back, and a non-significant degradation in a slice that includes every high-value account may be. The gate design in this sheet therefore keeps *statistical* gates (quality deltas with intervals) separate from *harm* gates (blast radius limits), because the two use different thresholds and different units. Any process that runs them through one composite score has destroyed the veto property of each.

**Limit 9 — the checklist itself decays.** Items accumulate, nobody retires them, and after a year the checklist has 60 items of which 12 are load-bearing. A checklist with 60 items under time pressure is executed as a ritual: the boxes get ticked, the evidence does not get read. The countermeasure is structural — each item must name the failure it prevents (the F-numbers in Section 8), items whose failure mode has been eliminated by automation are deleted, and the review at Section 7 includes a mandatory "delete one item" decision.

**Limit 10 — the SLO window can be gamed by choosing the boundary.** Rolling windows and calendar windows give different verdicts for the same event. A 30-day rolling window means a bad day is "forgiven" 30 days later; a calendar month means a bad day on the 1st is punished differently from the same day on the 28th. Neither is wrong, but a release process that does not state which one it uses cannot compute its own budget, and a process that switches between them during a dispute has no budget at all.

```text
[1] FORMULA: aggregate_shift = w * e                      [ID: RL-11]
[2] WHERE: w = slice traffic share (0..1); e = excess failure ratio in the
           slice (dimensionless); aggregate_shift is the change in the
           population failure ratio.
[3] CONSTRAINT: valid for a two-partition decomposition (slice vs rest) and
           small rates; with several slices the shifts add.
[4] NUMERICAL EXAMPLE: w = 0.002, e = 0.04 -> shift = 0.00008 = 0.008
           percentage points. Compare the detection floor from RL-08 at
           n = 30,000 and p0 = 0.001: d_min = 4.54e-4 = 0.045 percentage
           points. The slice shift is 5.7x below the floor, so the aggregate
           gate structurally cannot see it.
[5] INTERPRETATION: slice gates must exist as separate items with their own
           sample-size requirement, because the aggregate is blind to exactly
           the small populations where a regression is most harmful.
```

> **Takeaway:** Every limit above is a case where a correct formula applied to the wrong population, unit, or window produces a confident wrong answer; the countermeasure is declaring the population, unit, and window in the release record.

### 4.5 The Rollback Equation and Its Three Terms

The operational quantity that rollback planning actually needs is time to restore, and it decomposes additively:

```text
T_total = T_detect + T_decide + T_execute + T_verify    [ID: RL-12]
```

Each term has a different owner and a different lever, and conflating them is why rollback targets are usually optimistic. `T_detect` is a monitoring property: how long until the signal crosses the alarm threshold. `T_decide` is a human/process property: how long until someone with authority acts. `T_execute` is a mechanical property: the deploy action's duration. `T_verify` is the property most often set to zero in planning and never zero in reality: the time to confirm that behaviour actually returned.

```text
[1] FORMULA: T_total = T_detect + T_decide + T_execute + T_verify   [ID: RL-12]
[2] WHERE: all four terms in minutes; T_total is the restoration time whose
           objective version is the RTO.
[3] CONSTRAINT: the terms are sequential, which is the pessimistic assumption.
           They can overlap in practice (a responder can start executing while
           detection is still being confirmed), in which case T_total is
           bounded by the sum and may be shorter. Using the sum as the
           planning value is deliberate: it is the value you can defend.
[4] NUMERICAL EXAMPLE: T_detect = 5, T_decide = 8, T_execute = 3,
           T_verify = 6 -> T_total = 22 minutes. Against an RTO of 20 minutes
           the plan fails by 2 minutes, and the largest lever is T_decide
           (8 min) followed by T_verify (6 min), not T_execute (3 min).
[5] INTERPRETATION: teams optimise T_execute because it is the term they own
           technically, while the dominant terms are usually detection and
           decision; the decomposition redirects effort toward the actual
           bottleneck.
```

**Derivation of the exposure bound from the time decomposition.** During `T_total`, at failure ratio `q` and arrival rate `lambda`, the affected request count is `lambda * T_total * q`. Combining with RL-10 for the pre-stop period gives the total blast radius:

```text
affected = lambda * (T_stage + T_total) * q               [ID: RL-13]
```

and with an irreversible-effect fraction `f`, the irreversible count is `affected * f`. This is the formula that answers "how bad can it get" before the release, which is the question a release approval is actually answering.

**Boundary of RL-13.** It assumes `q` stays constant during exposure. In practice `q` often grows — a failing dependency fails harder under retry pressure — so the formula is a lower bound and the release record should say so. It also assumes `lambda` is constant; at peak, the same `T_total` produces proportionally more affected requests, which is why a release during peak traffic has the same duration but a larger blast radius.

**A worked comparison of two rollback strategies.** Suppose the defect is a bad threshold, `lambda = 3000 req/min`, `q = 0.02`, and the forward fix takes 4 minutes total (`T_detect 2 + T_decide 1 + T_execute 0.5 + T_verify 0.5`) while the bundle rollback takes 22 minutes. Forward fix affects `3000 * 4 * 0.02 = 240` requests; rollback affects `3000 * 22 * 0.02 = 1,320`. But the forward fix carries its own risk: if it is wrong, a second cycle starts, and its failure could be worse than the original. The decision rule that follows is not "prefer the faster" but "prefer the faster *when the fix is understood and the second-cycle risk is bounded*, otherwise prefer the coherent restoration". The arithmetic informs; it does not decide.

> **Takeaway:** Decompose restoration time into detect, decide, execute, and verify, then attack the largest term; optimising the execution step while detection and decision dominate is the standard misallocation.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE. The service, traffic, baseline rates, prices, and dwell times below are constructed for clean arithmetic. They are not a benchmark, not a vendor figure, and not an observed incident. All arithmetic was recomputed and cross-checked by a second route where one exists; the *method* is the transferable content, and the numbers must be replaced with measured values before any real decision.

### 5.1 The situation

A ranking model serves a product feed. The team proposes bundle `B1` to replace the live bundle `B0`.

```text
Traffic:            lambda = 3,000 requests/minute (uniform model over the window)
Baseline failure:   p0 = 0.001  (0.1% of requests return an error)
Baseline latency:   p50 = 120 ms, p99 = 400 ms (declared SLO at p99 <= 450 ms)
SLO:                99.9% availability over a 30-day window
Quality metric:     NDCG@10 on a frozen evaluation set of 50,000 queries
Incumbent NDCG:     0.4120
Candidate NDCG:     0.4185  (a +0.0065 absolute improvement)
Proposed canary:    5% for 15 minutes, then 25% for 30 minutes, then 100%
Release window:     a Tuesday, 14:00 local, i.e. peak traffic
```

### 5.2 Step 1 — the error budget in each convention

```text
Window: W = 30 days * 24 h * 60 min = 43,200 minutes
Budget in minutes:  B = 43200 * (1 - 0.999) = 43,200 * 0.001 = 43.2 minutes
Requests in window: N = 3000 req/min * 43,200 min = 129,600,000 requests
Budget in requests: F_max = N * 0.001 = 129,600 failed requests
Consistency check:  43.2 min * 3000 req/min = 129,600 requests. Agrees,
                    because the uniform-rate model was assumed.
```

The consistency check is the point, not the number. It confirms the two conventions agree *under this model*, and it also shows exactly what would break the agreement: any non-uniformity in time. At peak, 43.2 minutes would correspond to more than 129,600 requests; off-peak, fewer.

### 5.3 Step 2 — is the canary large enough?

Proposed stage 1: 5% of 3,000 req/min for 15 minutes.

```text
Arrival rate in canary: 0.05 * 3000 = 150 req/min
Requests in stage 1:    150 * 15 = 2,250 requests
Expected failures at baseline: 2250 * 0.001 = 2.25
```

Detectable absolute increase, from RL-08 rearranged, at 5% one-sided and 80% power:

```text
d_min = (z_alpha + z_beta) * sqrt( p0 (1-p0) / n )
      = 2.4866 * sqrt( 0.001 * 0.999 / 2250 )
      = 2.4866 * sqrt( 4.44e-7 )
      = 2.4866 * 6.6633e-4
      = 1.657e-3
```

So the canary can detect an absolute failure-rate increase of about 0.166 percentage points, i.e. the failure rate rising from 0.1% to roughly 0.27% — a 2.66x relative increase — with 80% power. Anything smaller is invisible.

Now the second stage: 25% for 30 minutes.

```text
Rate: 0.25 * 3000 = 750 req/min;  requests = 750 * 30 = 22,500
d_min = 2.4866 * sqrt(0.001*0.999/22500)
      = 2.4866 * sqrt(4.44e-8)
      = 2.4866 * 2.1066e-4 = 5.238e-4
```

Stage 2 detects an absolute increase of 0.052 percentage points, a 52% relative increase. Both stages are *far* weaker than the team's implied claim (they believe they would notice a bad release). The cumulative exposure before full traffic, by RL-10:

```text
Stage 1: 0.05 * 15 * 3000 = 2,250
Stage 2: 0.25 * 30 * 3000 = 22,500
Cumulative before 100%: 24,750 requests
Expected failures at baseline in that exposure: 24750 * 0.001 = 24.75
```

If the candidate had a defect doubling the failure rate (`q = 0.002`), expected extra failures during the canary would be `24,750 * 0.001 = 24.75`, and the probability that stage 2's gate trips is 80% by construction. So a doubled failure rate is caught about four times in five — acceptable for a reliability regression, and *not* acceptable as the only protection for a quality or harm regression.

### 5.4 Step 3 — blast radius if the stop rule fails to fire

This is the case the release process must survive: the defect is real but below the detection floor, so the canary passes and the release expands.

```text
Suppose the true effect is a quality regression on a slice, not a failure-rate
regression: 2% of traffic (the long-tail catalogue) sees NDCG drop 0.15 absolute.
Aggregate NDCG effect: w * e is not additive for NDCG in general, so use the
slice gate directly rather than the aggregate:
  slice share w = 0.02, slice NDCG drop = 0.15 -> the aggregate moves by at
  most 0.02 * 0.15 = 0.0030 if the metric were a simple average, which is
  below the candidate's measured +0.0065 improvement. The aggregate gate would
  PASS while 2% of traffic degrades sharply.
Exposure per hour at full traffic on the slice:
  3000 req/min * 60 min * 0.02 = 3,600 requests/hour on the affected slice.
```

The arithmetic makes the structural point: an aggregate improvement of +0.0065 can coexist with a slice regression of -0.15, and no aggregate threshold will catch it. The remedy is a slice gate with its own sample size — for a 2% slice at 150 req/min of canary traffic, the slice sees 3 req/min, so a 15-minute stage gives 45 requests, which is hopeless for measuring a quality metric. The honest conclusion is that **the slice gate must be evaluated on the shadow stage, where 100% of traffic is available**, or the slice must be promoted to its own canary cohort.

### 5.5 Step 4 — restoration time and RTO

Declared terms for the rollback drill (illustrative but internally consistent):

```text
T_detect  = 5 min   (p99 latency alarm on a 1-minute window, 3 consecutive)
T_decide  = 8 min   (responder on-call, escalation to release owner)
T_execute = 3 min   (re-point alias, restart, warm)
T_verify  = 6 min   (check p99, failure rate, and 3 sampled responses)
T_total   = 5 + 8 + 3 + 6 = 22 min
RTO target = 20 min   ->  FAILS by 2 min
```

Now compute the blast radius of the rollback itself with RL-13, at the moment the defect is discovered:

```text
q = 0.02 (2% of requests failing during the defect), lambda = 3000 req/min
T_stage = 15 min already elapsed before detection
affected = 3000 * (15 + 22) * 0.02 = 3000 * 37 * 0.02 = 2,220 requests
Budget consumed in the incident, in requests: 2,220 of 129,600 = 1.71%
Budget consumed in minutes (time-based convention, service fully degraded
for the detection+rollback period is NOT assumed here; only the failing
fraction counts as unavailability): 2,220/3000 = 0.74 minutes.
```

That second figure is instructive: a 37-minute incident in which 2% of requests fail consumes 0.74 of the 43.2 available minutes — 1.7% of the budget. A single release incident of this shape is affordable; three per week is not, and that comparison is the real use of the budget arithmetic.

Where should the two minutes be found? The decomposition says `T_decide = 8` is the largest term. A pre-declared stop rule that removes the decision (the on-call is *authorised and required* to roll back on the declared signal without escalation) plausibly converts 8 minutes into 1, which brings `T_total` to 15 and satisfies the RTO with margin. This is the concrete instance of the general claim in Section 4.5: the dominant lever is usually a process term, and it is cheap to change if the stop rule was written before the release.

### 5.6 Step 5 — the gate verdict, computed

Conjunction of gates for `B1`, with the numbers above:

```text
Gate 1 closure:        manifest 9 of 12 members declared  -> FAIL
Gate 2 compatibility:  7 of 7 consumers pass payload test  -> PASS
Gate 3 quality:        NDCG +0.0065 aggregate, no slice gate -> CONDITIONAL
Gate 4 safety:         abuse suite executed, 0 critical     -> PASS
Gate 5 operations:     p99 400 ms vs 450 ms SLO             -> PASS
Gate 6 exposure plan:  stage 1 n=2250 detects only 2.66x     -> FAIL (claim
                       of "we would notice a bad release" not supported)
Gate 7 rollback:       drill not executed in 14 days        -> FAIL
```

Three of seven fail, and the release is blocked — not because the model is bad (it is better on the aggregate metric) but because the *process around it* cannot support the claims being made. This is the single most important worked outcome in the sheet: a candidate that improves the headline metric can still be correctly blocked, because a release approval is a statement about a controlled operation, not a statement about a model.

Resolving the three failures, in order of cost:

```text
Gate 1: enumerate the 3 undeclared members; add them to the manifest.
        Cost: minutes. Blocks everything else, so do it first.
Gate 6: either extend stage 1 from 15 to 60 minutes (n = 0.05*3000*60 = 9,000;
        d_min = 2.4866*sqrt(0.000999/9000) = 2.4866*3.332e-4 = 8.28e-4,
        i.e. it can now detect a 0.083pp increase, a 1.83x relative change),
        or restate the claim honestly as "catches a doubled failure rate".
Gate 7: run the drill. Cost: one engineer-hour plus a shared staging slot;
        produces a timestamped artifact with the restored manifest hash and
        the measured T_total, which also refreshes the RTO evidence.
```

Note that fixing gate 6 by extending the dwell *increases exposure*: `0.05 * 60 * 3000 = 9,000` requests instead of 2,250. This is the central trade in canary design and it cannot be resolved by wishful thinking — either accept more exposure to gain detection, or accept weaker detection to limit exposure, and *say which you chose*.

### 5.7 Step 6 — the money

Assign illustrative costs to make the harm ledger concrete:

```text
Cost per failed request (retry, support, lost conversion): 0.02 currency units
Cost per 1,000 requests of compute for the new bundle:     0.35 units
Expected extra compute for the release week at full traffic:
  129,600,000 requests * 0.35/1000 = 45,360 units per 30 days
  -> 45,360/30*7 = 10,584 units for the release week
Cost of a rollback drill: 1.5 engineer-hours * 80 units/hour = 120 units
Cost of the 2,220 failed requests in 5.5: 2,220 * 0.02 = 44.4 units
Cost of the 8-minute decision delay in failed requests: 8*3000*0.02*0.02
  = 9.6 units
```

The comparison is the decision-relevant one: the drill costs 120 units, and the delay it could remove costs 9.6 units *per incident*. The drill pays for itself after about 12.5 incidents at that magnitude — or after a single incident if the defect fraction were larger. This is why "we cannot afford the drill" is usually an unexamined claim: the drill's cost is fixed and small, while the delay cost scales with the defect, and the defect size is exactly the unknown. Note also that all these unit prices are constructed; the *structure* of the comparison is the transferable part.

### 5.8 What this example establishes and what it does not

It establishes: the budget in both conventions and the condition for their agreement; the detection floor for two canary designs, computed from the sample size rather than assumed; that an aggregate gain can conceal a slice loss and that the aggregate is structurally blind to it; a restoration-time budget that fails its RTO and the identification of the dominant term; a blast radius consistent with a modest share of the error budget; and a full gate conjunction in which a better model is correctly blocked.

It does not establish: that 3,000 req/min is realistic for anything; that 0.1% or 2% are typical; that these prices are real; that a 15-minute dwell is short (it is short *at this traffic and this effect size*, and the same 15 minutes at 30,000 req/min would detect a 3x smaller effect); or that the drill costs 1.5 engineer-hours. Any real release must substitute measured values, and the sheet's method survives that substitution while its numbers do not.

> **Takeaway:** The worked example blocks a better model: release approval is a claim about a controlled operation, and a process that cannot support its own claims is a legitimate reason to refuse a good candidate.

## 6. Visual / Geometric Intuition

### 6.1 The exposure cone, drawn to scale

The cone of Section 2 is not decorative; its slope is the arrival rate and its width at any height is the cumulative exposure. Drawing it to scale for the Section 5 numbers makes the asymmetry visible: the last stage dominates.

```text
 cumulative
 exposed requests
        ^
 112,725|                                              ****
        |                                        *****
        |                                  *****
  90,000|                            *****
        |                      *****
        |                *****
  22,500|          *****
        |      ****
   2,250|  ***
        +--+----+---------+-----------------+-------------------> stage
          sh  5%        25%               50%               100%
          (0) 15min     30min            60min            remainder

  area added by stage:  shadow 0 | 2,250 | 22,500 | 90,000 | rest
  fraction of pre-full exposure added:
    2,250/112,725  = 2.0%
   22,500/112,725  = 20.0%
   90,000/112,725  = 79.8%
```

Read the fractions as a warning. Nearly 80% of the exposure before full traffic comes from the 50% stage. If the process treats all stages as equally "safe because it is not 100%", it has misallocated its attention by a factor of 40 relative to the first stage. The correct design principle, visible in the areas, is that **the earliest stages should have the longest dwells** and the later stages should be short, because the marginal information per request is much higher when the sample is scarce.

### 6.2 The detection floor as a curve

RL-08 says `d_min ∝ 1/sqrt(n)`. Plotted, the consequence is stark: doubling the sample reduces the detectable effect by only about 29%, and a tenfold increase in sample only reduces it by 68%.

```text
 d_min
 (absolute
  failure-rate
  increase)
  1.6e-3 +*        n=2,250
         |  *
  8e-4   |    *    *          n=9,000
         |      *     *
  4e-4   |          *    *    n=30,000
         |              *   *
  2e-4   |                  *  *
         |                        *   *   *
         +---|---|---|---|---|---|---|---|---|---> n
             2.5k  5k  10k 20k  40k  80k 160k 320k

  each doubling of n buys a factor of 1/sqrt(2) = 0.707 in d_min
  to halve d_min you must quadruple n
```

The practical implication is that there are only three honest ways to detect a small regression: run longer, run at a higher share, or use a lower-variance measurement (a paired comparison against the incumbent on the same requests, which removes request-level variance). The third is the most under-used: comparing the new bundle's output to the incumbent's *on the same input* converts an unpaired two-sample problem into a paired one, often reducing the required sample by an order of magnitude. It requires shadow capability, which is why shadowing pays for itself twice — once for correctness inspection, once for statistical power.

### 6.3 The rollback timeline as a bar chart

```text
 T_detect  |=====                                |  5 min
 T_decide  |     ========                        |  8 min   <- largest
 T_execute |             ===                     |  3 min
 T_verify  |                ======               |  6 min
           +----+----+----+----+----+----+----+--> min
           0    5   10   15   20   25   30
                                              ^
                                          RTO 20 fails here

 after rewriting the stop rule to remove escalation:
 T_detect  |=====                                |  5
 T_decide  |=                                   |  1
 T_execute | ===                                 |  3
 T_verify  |    ======                           |  6
           +----+------> total 15 min, RTO met with 5 min margin
```

The picture makes the argument that prose often fails to make: the bar you can shrink by policy is bigger than the bar you can shrink by engineering, and it shrinks in hours rather than sprints.

### 6.4 The bundle as a set diagram

Closure is easiest to see as a Venn diagram between "what the manifest declares" and "what the serving path actually reads".

```text
   +-----------------------------+     +------------------------------+
   |   manifest (declared)       |     |   args(f) (actually read)    |
   |                             |     |                              |
   |  model hash                 |     |  model hash                  |
   |  tokenizer rev              |     |  tokenizer rev               |
   |  image digest               |     |  image digest                |
   |  feature def rev            |     |  feature def rev             |
   |  eval set id                |     |                              |
   |                             |     |  prompt template    <-- UNDECLARED
   |                             |     |  guardrail policy   <-- UNDECLARED
   |                             |     |  index snapshot     <-- UNDECLARED
   |                             |     |  feature values (input, OK)  |
   +-----------------------------+     +------------------------------+
                    ^                              ^
                    |                              |
          the manifest also contains         the shaded difference is the
          non-runtime items (eval set id)    closure gap; it is exactly the
                                             set that rollback will miss
```

The overlap is what the manifest is for; the difference is what the rollback will leave behind. Two observations follow. First, the manifest can legitimately contain members not in `args(f)` (an evaluation set ID, an approval record) — closure requires containment `⊇`, not equality. Second, feature *values* appear in `args(f)` but must not be pinned; the resolution is that the *source* of values is pinned (which store, which definition revision) while the values themselves rotate. That distinction is what keeps closure from freezing the service.

### 6.5 Two failure curves: gradual and cliff

Not all regressions look alike, and the shape determines whether a short canary can see them.

```text
 failure        (a) gradual drift                (b) cliff
  ratio                                          |
  0.03 |                              *          |        *
       |                        *                 |        |
  0.02 |                  *                       |        |
       |            *                             |        |
  0.01 |      *                                   |        |
       |  *                                       |        |
  0.00 +--+---+---+---+---+---+---> time          +--------+---> time
        canary ends here (sees 0.01, under        the entire new population
        any sane threshold -> PASSES)             fails at once (sees it)
```

Curve (a) is the dangerous one for a short canary: the regression is real, grows, and the canary's window samples only the low part of it. Curve (b) is caught immediately, which is why "the canary was clean" is stronger evidence against cliffs than against drifts. The countermeasures differ too: cliffs are caught by stop rules on rates, drifts by comparing against the incumbent over a longer window, ideally with the paired comparison from 6.2.

### 6.6 The error budget as a tank, not a number

```text
   budget 43.2 min
   +--------------------+
   |####################|  100% remaining at window start
   +--------------------+
   |################    |  release incident: -0.74 min (1.7%)
   +--------------------+
   |##############      |  dependency incident: -6.0 min (13.9%)
   +--------------------+
   |####                |  peak-traffic outage: -30.0 min (69.4%)
   +--------------------+
   |                    |  remaining: 6.46 min (15.0%) -> burn-rate alarm
   +--------------------+     arms for the rest of the window
```

The tank picture explains why burn rate rather than level is the alarm variable: at 15% remaining with 60% of the window elapsed, the *rate* is still sustainable, while at 15% remaining with 95% of the window elapsed it is not. The same level means different things depending on where you are in time, which is exactly what RL-09 normalises.

> **Takeaway:** Draw the exposure cone and the detection curve before choosing stages; the areas and slopes, not the stage names, determine where the risk actually sits.

## 7. Comparison / Selection Table

### 7.1 Choosing the rollout shape

| Shape | Traffic exposed before full | Detection capability | Side effects | When to choose | When to refuse |
|---|---|---|---|---|---|
| Shadow only | 0 external | Output comparison on 100% of traffic; no consequence data | Must be structurally zero | Any bundle whose outputs can be computed cheaply and compared; first stage for quality and slice verdicts | When inference cost of doubling compute is unaffordable, or when the output cannot be produced without the side effect |
| Canary only | Small but non-zero | Consequence data, low power at small `n` | Real | Tightly bounded harm per request; fast cliffs | When the claimed detectable effect needs more requests than the harm budget allows |
| Shadow then canary | Small, well-targeted | Both output and consequence | Real, after validation | The default for user-visible ML | When the two stages would measure different populations, invalidating the comparison |
| Blue/green (two full stacks) | 0 then 100, switched | Instant reversal, no statistical protection | Real at cutover | Infrastructure changes where partial traffic is meaningless | Model changes where a bad cutover hits everyone at once |
| Staged percentage ramp | Moderate | Increasing power with share | Real | The general case | When stages are shorter than the detection time (RL-10 violation) |
| Per-account ring | Moderate, by account | Skewed population | Real | Enterprise or B2B where accounts are the unit of harm | Consumer products where rings are not representative |
| Forward-only with feature flag | Configurable | Flag-off is the reversal | Real | Only when the flag alone fully determines behaviour | Any bundle where other members changed too (F1) |

The last row is the most consequential entry: a feature flag is a *routing* mechanism, not a rollback of the bundle. It is a complete answer only if the flag is the sole changed member. If the release also changed a prompt template, the flag-off state is not the pre-release state.

### 7.2 Choosing between rollback, forward fix, and degraded mode

| Option | Time to restore | Risk of the option itself | Reversible? | Choose when | Refuse when |
|---|---|---|---|---|---|
| Bundle rollback | `T_detect + T_decide + T_execute + T_verify` (RL-12) | Low if drilled; high if untested | Yes, if the target bundle still deploys | The defect is in the bundle and `B0` is coherent | The previous bundle is no longer deployable (Limit 2), or a schema contract step already removed what `B0` needs |
| Forward fix | Often shorter for config-level defects | The fix is untested in production; may compound | Only if it works first time | Defect understood, fix testable in minutes, failure blast radius bounded | The fix requires a code change and a build; or the fix's own failure mode is unknown |
| Degraded mode | Declared in the runbook | Loss of function is a product decision | Yes | The defect cannot be isolated and harm must be contained now | No degraded mode exists in the runbook; improvising one during an incident |
| Feature-flag off | Fastest | The flag may not isolate the change | Yes | The flag is the only changed behaviour input | Other bundle members changed (F1) |
| Do nothing, monitor | Zero | Ongoing harm | Not applicable | Evidence of harm is weak and the budget is ample | Any irreversible side effect is occurring |

The table's structure encodes a principle: *the option that restores coherence* takes precedence over *the option that is fastest*, unless the fastest option is provably sufficient. Most post-incident regret comes from choosing "fastest" on an incomplete model of what changed.

### 7.3 Gate strictness by release risk class

| Risk class | Examples | Gate 1 closure | Gate 2 compat | Gate 3 quality | Gate 4 safety | Gate 5 ops | Rollback drill | Approval |
|---|---|---|---|---|---|---|---|---|
| R0 cosmetic | Logging text, comment-only config | Manifest entry | None | None | None | None | Not required | Automated |
| R1 lower risk | Threshold tweak inside an approved range | Manifest entry | Payload test | Offline check only | Existing suite | Latency sanity | Last 90 days | One engineer |
| R2 standard | New model version, same features | Full | Full | Frozen eval + slices | Full | Full | Last 30 days | Engineer + owner |
| R3 elevated | New feature, new index, prompt change | Full | Full + consumer enumeration | Frozen eval + slice gates + shadow | Full + red-team items | Full + capacity test | Last 14 days, timed | Release board |
| R4 critical | Guardrail change, schema contract, irreversible effects | Full + independent audit | Full + deprecation window | Full + shadow + paired comparison | Full + sign-off | Full + load test + degraded mode | Executed on this bundle | Named executive owner |

The right-hand column matters more than it looks. Escalating approval authority for higher risk classes is the mechanism that keeps the *cheap* classes cheap; without it, every release goes through the highest gate set and teams route around the process. The risk class is a decision made at proposal time and recorded in the manifest, so it can be audited after the fact.

### 7.4 Selecting the stop condition

| Stop-rule form | Sensitivity | False-positive behaviour | Best for | Failure it introduces |
|---|---|---|---|---|
| Absolute failure-rate threshold | Low at small `n` | Rare | Known-stable baselines | Cannot see regressions smaller than the threshold |
| Standardised deviation (`z`) | Scales with `sqrt(n)` | Fires on clustered single-host faults | Mature services with modelling | Independence violation inflates `z` (Limit 1 of 4.1) |
| Delta versus incumbent on paired requests | Highest | Fewer, because pairing removes request variance | Shadow and canary with paired capability | Needs the incumbent running alongside; doubles compute |
| Burn rate on an SLI | High for sustained faults | Fires on a pre-existing poor state | Ongoing operations | Confuses "released badly" with "was already bad" |
| Quality metric with a confidence interval | Moderate | Tolerable | Slice and aggregate quality | Interval width may exceed the effect of interest |
| Manual judgment by an operator | Unpredictable | Slow | Nothing; as a supplement only | Bias toward continuing (sunk effort), inconsistent thresholds |
| Two consecutive windows over threshold | Reduced | Fewer, slower | Noisy signals | Two-window delay adds to `T_detect` |

A release normally needs two rules from this table: one on *reliability* (a burn or `z` rule) and one on *quality* (a slice-aware metric rule). They have different units, different sample-size requirements, and different owners, which is why they are never collapsed into one composite.

### 7.5 Choosing what to record

| Release record field | Why it exists | Consequence of omission |
|---|---|---|
| Manifest ID and member list | Makes `Delta` enumerable for RL-06 | Rollback cannot be evaluated, only attempted |
| Risk class and approver | Makes the gate set auditable | Disputes about which gates applied become unresolvable |
| Evidence artifacts (eval output, test logs) | Converts "declared" into "executed and recorded" | The release cannot be reconstructed later |
| Training data window and eval set ID | Bounds what the evaluation claim covers | Silent drift between the data the model saw and the data it is judged on |
| Cohort predicate and stage plan | Defines the exposed population | Harm accounting uses the wrong denominator |
| Stop rules with thresholds and windows | Makes stopping a lookup rather than a debate | Decision time dominates `T_total` |
| Rollback target manifest and drill timestamp | Bounds restoration confidence | The RTO is an unmeasured hope |
| Monitoring queries or dashboard links | Makes verification possible | `T_verify` becomes indefinite |
| Post-release review owner and date | Forces the learning step | The same defect recurs on the next release |
| Known limitations and open questions | Prevents over-claiming | Approvers believe more than the evidence supports |

### 7.6 Gate-by-gate decision guide when evidence is partial

| Situation | Priority | Action | Disqualifier | Fallback |
|---|---|---|---|---|
| Manifest incomplete but everything else passes | Critical | Block; complete the manifest | The gap is a non-runtime item (eval ID) | Record and proceed |
| Quality improves on aggregate, no slice evidence | High | Block or promote to elevated class | Slices are structurally irrelevant (single-segment product) | Run slice eval on shadow |
| Canary sized below the claimed effect | High | Extend dwell or restate the claim | Harm budget forbids more exposure | Restate the claim honestly |
| Rollback drill older than the class threshold | High | Execute the drill before approval | Emergency fix of a live incident | Accept with a recorded exception and a review |
| Baseline already burning budget | Medium | Delay the release | The release is itself the fix | Use a shorter window and a tighter stop rule |
| Consumer count not enumerable | High | Deprecation window instead of removal | Contract change is additive only | Additive release first |
| Cost of rollback untested at peak load | Medium | Add a load-conditioned drill | Peak rollback is architecturally identical | Note the assumption |
| Approver is the release author | Medium | Second approver for R3+ | Class is R1 and automated | Record the exception |

> **Takeaway:** Pick the rollout shape by which harm is bounded and the stop rule by which signal can actually be read at that sample size; the two choices constrain each other and must be made together.

## 8. Failure Modes & Edge Cases

Eight failure modes, each with an observable symptom, at least two competing causes that produce that symptom, a check that discriminates between them, a bounded fix, and a verification that the fix worked. The discriminating check is the part worth memorising: the fix for the wrong cause usually works once.

### F1 — Alias-only rollback

**Symptom.** A rollback completes successfully according to the deployment tool; the alias points at the previous model hash; user-visible behaviour is unchanged or changed in a new way.

**Competing causes.** (a) A genuine closure failure — other bundle members changed and were not restored; (b) the rollback restored a *different* previous bundle than intended because the alias history is ambiguous (two "previous" candidates from a hotfix); (c) a cache or connection pool still serving results from the withdrawn bundle, so restoration is correct in configuration and stale in effect.

**Discriminating check.** Compare the *live* manifest to `B0`'s manifest member by member, not the alias target. If members differ beyond the model, (a) is proven. For (b), check whether the deployment history has more than one candidate at the same timestamp or whether a hotfix was applied outside the process. For (c), issue a request with a unique sentinel input and confirm the response matches the newly restored bundle's expected output, then check cache TTLs and whether the serving process restarted.

**Bounded fix.** For (a), extend the rollback procedure to restore the full set `Delta` and record each restoration. For (b), require a single linear release history with no out-of-band deploys. For (c), make cache invalidation part of the rollback procedure and verify by sentinel rather than by configuration inspection.

**Verification.** A rollback drill whose acceptance criterion is *behavioural*: a fixed set of sentinel requests must return exactly the pre-release responses (byte-identical for deterministic models, within a stated tolerance for stochastic ones with fixed seeds). Configuration equality is necessary and not sufficient.

### F2 — Unrepresentative canary

**Symptom.** The canary passes cleanly at 5% and 25%, then at 50% or 100% a different population shows a sharp regression.

**Competing causes.** (a) The cohort predicate is correlated with a segment that is affected differently (routing by geography, device, or account tier); (b) the canary population differs from the full population in *volume per user*, so a per-request metric looks fine while a per-user metric degrades; (c) the full-traffic stage coincidentally overlaps an unrelated event (a marketing push, a dependency's own deploy).

**Discriminating check.** Compare the canary cohort's distribution on the top five known segment dimensions against the full traffic, using the same window. A large divergence on any dimension proves (a). For (b), recompute the same metric with users as the denominator; if it moves while the per-request version does not, that is the cause. For (c), align the timeline with the release of any dependency and with promotion calendars before concluding.

**Bounded fix.** For (a), route by a hash of a stable identifier so the sample is unbiased, or run stratified stages that deliberately cover each segment. For (b), report both per-request and per-user metrics as gate items. For (c), require a stable window and a dependency-change check before expansion.

**Verification.** A distribution-comparison report included in the release record, produced before each expansion, plus a documented incident of a "canary passed, full failed" if one exists — that record is the strongest evidence the check is real.

### F3 — Shadow with side effects

**Symptom.** After a shadow period, downstream systems show duplicate records, duplicate notifications, or inflated counters.

**Competing causes.** (a) The shadow path genuinely invokes a write; (b) the shadow path is read-only in code but the model's output triggers a downstream consumer that was not considered part of the "write" boundary; (c) both are correct and the duplication comes from a retry loop unrelated to the shadow.

**Discriminating check.** Run the shadow with a unique sentinel identifier and search every downstream store for it. If any store contains an effect keyed by the sentinel, (a) or (b) is present; the distinction is whether the write was in the shadow's own code path or in a callback it triggered. If the sentinel is absent everywhere and duplicates exist, (c). Also compare counters with the shadow enabled and disabled for the identical traffic replay.

**Bounded fix.** For (a), move the side-effecting call behind a guard that is provably closed in shadow mode — not a flag defaulting to false, but an absent code path or a permission the shadow principal does not have. For (b), enumerate the downstream consumers and add each to the shadow's no-effect contract. For (c), fix the retry idempotency separately.

**Verification.** The sentinel test must be part of the shadow stage's own gate, executed every time, because the no-effect property can regress when a consumer is added. "Enforced by permission" is verifiable; "intended by design" is not.

### F4 — Non-reproducible artifact

**Symptom.** A release cannot be rebuilt from its record; or a rebuild produces a numerically different model despite identical metadata.

**Competing causes.** (a) The manifest is genuinely incomplete (missing dependency pin); (b) the build is complete but non-deterministic (unseeded shuffling, non-deterministic GPU kernels, thread-count-dependent reductions); (c) the recorded metadata is correct and the difference is only in bit-level floating point while the behaviour is equivalent within tolerance.

**Discriminating check.** Rebuild in a clean environment from the manifest only. If the build fails on a missing dependency, (a). If it succeeds but the artifact hash differs, compare behaviour on the frozen evaluation set: if metrics are identical and only bits differ, (c); if metrics differ materially, (b) — and then bisect determinism by fixing seeds, thread counts, and kernel selection in turn.

**Bounded fix.** For (a), pin all dependencies including transitive ones and the toolchain version. For (b), either make the build deterministic (seeds, deterministic kernels, fixed reduction order) or accept non-determinism and *validate by behaviour* rather than by hash, updating the gate accordingly. For (c), record the tolerance explicitly.

**Verification.** A rebuild executed by someone other than the original author, from the manifest only, with the result compared by the declared criterion (hash or behavioural equivalence). This is also the only real test of whether the manifest is complete.

### F5 — Alert without an owner

**Symptom.** A stop condition fires; the alert is delivered; nothing happens for tens of minutes. Budget is consumed far beyond the threshold.

**Competing causes.** (a) No named owner and no escalation path, so the alert reaches a channel nobody watches; (b) an owner exists but is asleep or in another timezone with no follow-the-sun rotation; (c) the alert is firing constantly for unrelated reasons, so it is being ignored (alert fatigue).

**Discriminating check.** Count, for the last 30 days, the alerts of this type and the time to first human acknowledgement. A high count with a long acknowledgement time indicates (c); a low count with a long acknowledgement time indicates (a) or (b), separated by checking the rota and the delivery log for the specific timestamp.

**Bounded fix.** For (a), every stop condition carries a named role and an escalation ladder with timeouts. For (b), a documented rotation or an explicit "release only during staffed hours" constraint. For (c), retire or retune the noisy alerts — this is the checklist-decay problem from Limit 9 applied to alerts.

**Verification.** A drill that fires the alert for real and measures time to acknowledgement. The measured value becomes `T_decide` in RL-12; an unrilled `T_decide` is a guess.

### F6 — Detection latency longer than dwell time

**Symptom.** Every stage "passes", but the same defect is found at 100% traffic and the post-mortem shows the signal was already visible during the canary.

**Competing causes.** (a) The dwell was shorter than the alarm's evaluation window plus its confirmation requirement (e.g. a 5-minute dwell with a 10-minute-window alarm that needs 3 consecutive breaches); (b) the alarm existed but the threshold was set above the actual effect size, so it never fired; (c) the signal was visible on a *different* metric than the one being alarmed on.

**Discriminating check.** For each stage, compute the earliest time at which the alarm *could* have fired, and compare it to `T_s`. If `earliest_fire > T_s`, (a). If `earliest_fire <= T_s` but it did not fire, examine the threshold against the realised effect: (b). If another metric moved first, (c), and the alarm set is incomplete.

**Bounded fix.** For (a), either lengthen the dwell or shorten the alarm window — and note that shortening the window raises the false-positive rate, which is a real trade rather than a free improvement. For (b), re-derive the threshold from the smallest effect worth catching, using RL-08 to confirm the sample can support it. For (c), add the metric that moved first to the stop-rule set.

**Verification.** A table in the release record: stage, dwell, alarm earliest-fire time, and whether `earliest_fire <= dwell`. Any stage where the inequality fails is a stage that cannot protect itself and must be re-designed or relabelled as "exposure without protection".

### F7 — Irreversible effects during exposure

**Symptom.** Rollback succeeds, the bundle is coherent, the service is healthy, and yet complaints keep arriving for hours because messages were sent, prices quoted, or applications decided.

**Competing causes.** (a) The release path emits an irreversible effect and the stop rule is based only on service health, so it fires after the effect has been produced many times; (b) the effect is nominally reversible but the reversal process is manual and unowned; (c) the effect is correct for the new bundle's semantics and wrong for the old one, so the rollback itself creates inconsistency.

**Discriminating check.** Count effects emitted between defect onset and the stop decision, using the effect's own log rather than the service's. If the count is large and the stop rule monitored only latency or error rate, (a). If the reversal process exists but had no owner at the time, (b). If effects continue to be produced *after* the rollback with the old bundle's semantics, (c) — and this is the most dangerous because it looks like the rollback caused it.

**Bounded fix.** For (a), add an effect-rate SLI to the stop rules and gate the effect behind a capability the canary cohort controls. For (b), assign the compensation path an owner and a maximum age, and test it. For (c), require a reconciliation step after any rollback that follows an exposure with irreversible effects.

**Verification.** Replay the release in staging with an effect-recording sink and confirm the compensation path processes every recorded effect, and that the count in the sink matches the count in the compensation log. A discrepancy is the residual harm of RL-07.

### F8 — Checklist executed as ritual

**Symptom.** Every box is ticked, every release passes, and defects still reach production through gaps the checklist nominally covers. Alternatively, releases slow to a crawl because the checklist has grown to 60 items.

**Competing causes.** (a) Items are unverifiable, so ticking is indistinguishable from not doing it; (b) items are verifiable but the evidence is not read by the approver; (c) items are stale, covering failure modes that automation has since eliminated, diluting the load-bearing ones.

**Discriminating check.** Sample five recent releases and attempt to reconstruct the evidence for each item from the release record alone. Items that cannot be reconstructed are (a). For items that can be reconstructed, check whether the approver *accessed* the artifact (access logs) — if not, (b). For each item, ask which named failure it prevents; items with no answer are (c).

**Bounded fix.** For (a), convert each item into an artifact-producing check or delete it. For (b), reduce the item count so the load-bearing evidence is actually read; an approver who reads five items is worth more than one who initialises fifty. For (c), delete at least one item per review cycle.

**Verification.** A timed reconstruction exercise: an engineer who was not on the release reconstructs the decision from the record in under a stated time. If they cannot, the record is insufficient regardless of how many boxes were ticked.

### Edge cases that are not failures

- **A canary that is stopped and then restarted.** Not a failure, provided the restart begins at stage 1 with a new observation window; reusing the contaminated window is what makes it a failure.
- **A rollback that restores a better-than-expected state.** If `B0` had a partially-deployed fix, restoring it may include that fix. The record must state which bundle `B0` resolves to, including any hotfix, or the "rollback" is undefined.
- **An RTO miss during a genuine dependency outage.** The release process is not responsible for a third party's incident; the record should distinguish release-caused restoration from external-cause restoration, because mixing them corrupts the `T_total` statistics.
- **A release approved with a recorded exception.** Legitimate when the exception is specific, time-bounded, and has an owner. It becomes illegitimate when exceptions become the default path, which is measurable: exception rate is itself a signal.

> **Takeaway:** Each failure mode has a check that costs minutes and answers the question directly; the ritual failure F8 is the one that disables all the others, which is why it is listed last.

## 9. Complexity & Cost Analysis

### 9.1 Cost model with units

The release process has four cost components, each with its own unit, and they are not comparable without a stated exchange rate. Keeping them separate is the discipline; summing them requires an assumption that should be stated.

| Component | Unit | Driver | Scales with |
|---|---|---|---|
| Duplicate inference | GPU-seconds or CPU-seconds | Shadow and canary run the new bundle alongside the incumbent | traffic × stages × bundle cost per request |
| Rollback drills | engineer-hours + environment-hours | Rehearsing restoration | release frequency × drill duration × class threshold |
| Review and approval | engineer-hours | Gate evidence production and reading | number of items × releases × risk class |
| Incident cost | currency + trust | Defects that escape | escape rate × blast radius × cost per unit of harm |
| Opportunity cost | currency per day | Delay before a beneficial model reaches users | queue time × value per unit of model improvement |

The last two pull in opposite directions, and that tension is the whole economics of release process. A stricter process reduces escape rate and increases delay; a looser one reduces delay and increases escapes. Neither is right in general, and the honest position is to state which side you are trading toward and measure both.

### 9.2 Worked sizing: cost of the shadow stage

Take the Section 5 service: `lambda = 3,000 req/min` at peak, a model costing an illustrative `0.35` currency units per 1,000 requests, and a shadow over 24 hours.

```text
Requests shadowed:     3000 req/min * 60 * 24 = 4,320,000
Additional inference:  4,320,000 / 1000 * 0.35 = 1,512 currency units
Doubled compute peak:  the serving fleet must carry 2x during shadow
                       if the shadow shares the same pool
Storage for outputs:   assume 200 bytes per logged comparison
                       4,320,000 * 200 = 864,000,000 bytes = 823.97 MiB
                       (864e6 / 1048576 = 823.97 MiB)
```

The 823.97 MiB is a *lower bound* on the comparison store: it holds the payload comparison only, not request context or labels. Two observations follow. First, a 24-hour shadow costs about 1,512 currency units in compute at these assumptions, which is the price of measuring on 100% of traffic; a 1-hour shadow costs 63 units, and its sample is `180,000` requests — plenty for failure-rate detection (RL-08 at `n = 180,000`, `p0 = 0.001` gives `d_min = 2.4866*sqrt(0.000999/180000) = 1.85e-4`, an 18.5% relative change), and thin for slice quality on a 2% slice (`3,600` slice requests). The sizing conclusion is that the shadow's *duration* should be set by the smallest slice you intend to judge, not by the aggregate.

### 9.3 Worked sizing: cost of a failed canary

For the 5%/15 min then 25%/30 min plan, with the defect present but undetected:

```text
Stage 1 exposure: 2,250 requests;  stage 2: 22,500 requests
If harm per affected request is 0.02:  stage 1 = 45;  stage 2 = 450
If the defect reaches 100% for 1 hour before a late stop rule fires:
  3000 * 60 * q * 0.02 currency, at q = 0.02:
  3000*60*0.02 = 3,600 affected requests * 0.02 = 72 currency
Total across the incident: 45 + 450 + 72 = 567 currency units (illustrative)
Compare drill cost of 120 units and per-release review cost of ~2 engineer-
hours at 80/hour = 160 units.
```

The structure of this comparison is the transferable result: the *review and drill* costs are per-release and predictable, while the *escape* cost is per-incident and heavy-tailed. A process that budgets only the predictable side will always appear expensive relative to the incidents it prevents in the average case, and will always look cheap after the one incident that mattered. This is why the decision should be made on an explicit acceptable escape rate rather than on a yearly cost comparison that a single large incident can invalidate.

### 9.4 Scaling behaviour of the gate set

Gate cost grows with the number of items and with the number of slices, and both grow faster than linearly in practice because each slice requires its own sample-size check.

```text
evidence work  ~  items * (1 + slices * per-slice-sample-cost)
detection work ~  slices * n_min(slice share, effect size)
                   where n_min ~ (z_a+z_b)^2 p0(1-p0)/d^2  (RL-08)

Consequence: slicing is not free. Judging 20 slices at a 2% share each,
with a 0.05pp effect target and p0=0.001:
  n_min per slice = 6.1832 * 0.000999 / (5e-4)^2 = 24,708 requests
  but a 2% slice at 3000 req/min receives 60 req/min, so 24708/60 = 412 min
  = 6.9 hours per slice at FULL traffic, and much longer at canary share.
```

This is the single most useful scaling number in the sheet, and it is routinely ignored: **a slice gate is a sample-size commitment**, and a release that declares 20 slice gates without a shadow long enough to fill them has declared 20 gates that will be evaluated on noise. The remedies are to restrict slice gates to the few that map to a real decision, to use paired comparisons (which cut the requirement substantially), or to accept a longer shadow.

### 9.5 Where the money actually goes

A four-component view of a mature release process, with the illustrative proportions made explicit so they can be argued with:

```text
  cost component        proportion of total (illustrative)
  --------------------  ----------------------------------
  shadow/duplicate inference     40-60%   <- grows with model cost
  review and approval            10-20%
  drills and rehearsal           5-10%
  incident and escape cost       remainder, heavy-tailed
```

The dominant, controllable item is duplicate inference, and it is the one most often reduced by shortening the shadow. Shortening the shadow trades the first component for the last one — a trade that looks favourable until the first escape. The defensible position is to size the shadow by the smallest slice you intend to judge (Section 9.2) and to state that number in the release record, so the shortening is a visible decision with a named consequence rather than a quiet one.

### 9.6 What is not measured here

No latency, throughput, price, or engineer-hour figure in this section was measured. Every unit rate is a constructed placeholder. The *structure* — which quantities multiply, which are per-release versus per-incident, and where the heavy tail sits — is the transferable content; the coefficients must be replaced with the organisation's own measurements before any budget decision. A release process that adopts these numbers as if they were measurements has substituted arithmetic for evidence, which is the failure this sheet exists to prevent.

> **Takeaway:** Sizing a release process is sizing its sample requirements; slice gates are sample-size commitments, and shrinking the shadow always buys detection at the price of compute.

## 10. Key Theorems / Results Reference

Each result is stated with the conditions that make it true, then what it licenses and what it does not. The pattern to internalise is that every one of these is a *conditional* statement, and the release process fails when a condition is silently dropped.

**T1 — Closure determinism.** If a behaviour function `f` reads only arguments enumerated in a manifest `M`, and every member of `M` is immutable, then equal manifests imply equal behaviour. *Licenses:* treating the manifest as a behaviour identifier, and therefore as the unit of evaluation, release, and rollback. *Does not license:* the converse — equal behaviour does not imply equal manifests, because some members may be behaviourally irrelevant (an evaluation-set ID appears in the manifest and is not read at serving time).

**T2 — Rollback coherence condition.** Given pre-release bundle `B0`, attempted bundle `B1`, `Delta = B0 △ B1` (symmetric difference), and restored set `R`, behaviour returns to `B0` iff `Delta ∩ (args(f) \ R) = ∅` (RL-06). *Licenses:* diagnosing alias-only rollback formally, and deriving the checklist rule that `Delta` must be enumerable. *Does not license:* the claim that restoring behaviour restores *state*; irreversible effects sit outside the theorem entirely (T3).

**T3 — Irreversibility.** There exist effects that cannot be undone by any bundle restoration (delivery of a message, execution of a payment, publication of a decision to a third party). Consequently, rollback is not a complete containment strategy for any release that can emit such effects, and containment requires a capability to prevent emission (gating, capability scoping) rather than a capability to undo it. *Licenses:* the requirement for a compensation path and an effect-rate stop rule before approval. *Does not license:* a numeric bound on harm; that requires RL-13 with a measured `q`.

**T4 — Expand/contract necessity under independent deploy schedules.** If producer and consumer are released at times chosen independently and consumers require a field the producer intends to remove, then no single-instant switch satisfies all consumers; therefore a period in which both the old and new field exist is necessary, and its length is bounded below by the longest consumer migration interval. *Licenses:* the forced ordering (expand, migrate, contract) and the rule that the contract step requires *observed* consumer migration. *Does not license:* a specific window length, which depends on the slowest consumer and must be measured.

**T5 — Binomial detection power (RL-08).** For independent Bernoulli observations with baseline rate `p0` and an effect `d`, the sample required for power `1 - beta` at one-sided level `alpha` is approximately `(z_alpha + z_beta)^2 p0(1-p0)/d^2`. *Licenses:* canary sizing from a claimed effect, and the statement that detection cost grows as `1/d^2`. *Does not license:* its use when failures cluster. Under clustering the effective sample is the number of clusters, which can be orders of magnitude smaller, so the formula overstates power.

**T6 — Error-budget conservation.** Over a window `W`, the total permitted unavailability is `W(1-a)` (RL-02), and the sum of unavailability across incidents in the window cannot exceed it without breaching the objective. *Licenses:* treating budget as a depletable resource and releases as withdrawals. *Does not license:* a claim that an incident within budget was harmless — distribution matters (Limit 5 of 4.4), and a single 40-minute outage inside a 43.2-minute budget is a different event from forty one-minute ones.

**T7 — Burn-rate normalisation (RL-09).** The burn rate `b = W·consumed/(B·t)` is invariant to the choice of time unit and to the scale of the budget, and `b = 1` corresponds to exact exhaustion at window end. *Licenses:* comparing consumption across windows of different lengths, and setting alarms as multiples of `b`. *Does not license:* a single-window alarm design; short windows have high variance and long windows have high latency, which is why multi-window multi-burn-rate alarms exist.

**T8 — Exposure additivity (RL-10, RL-13).** For constant arrival rate and constant failure ratio, exposure is additive across stages and the total affected count is `lambda(T_stage + T_total)q`. *Licenses:* the blast-radius bound used in approval. *Does not license:* linear extrapolation when `q` grows during the incident (retry amplification, cascading dependency failure), which makes the formula a lower bound.

**T9 — Slice blindness (RL-11).** A slice of share `w` with excess rate `e` shifts the aggregate by `we`. Hence any slice with `we` below the aggregate's detection floor is invisible to an aggregate-only gate, by construction rather than by chance. *Licenses:* mandating slice gates with their own sample requirements, and the arithmetic in Section 9.4 showing why that is expensive. *Does not license:* the assumption that all slices are worth gating; the commitment must be justified by decision consequence.

**T10 — Sequential testing inflation.** If a stop rule is evaluated repeatedly during a stage (every minute, say) at a fixed per-look alpha, the family-wise false-positive rate grows with the number of looks and can greatly exceed the nominal alpha. *Licenses:* the requirement either to pre-declare a single analysis point per stage, or to correct alpha across looks, or to use a method designed for continuous monitoring. *Does not license:* comparing a `p`-value from a mid-stage peek to a fixed 5% threshold as if it were a single test — a very common error that produces spurious rollbacks and, worse, spurious confidence when a true effect is missed.

**T11 — Little's law, for capacity reasoning during rollback.** For a stable system, the mean number in the system equals arrival rate times mean time in system: `L = lambda W`. *Licenses:* sizing the drain and warm-up behaviour when a rollback restarts a fleet — the queue that must be absorbed during a restart is `lambda` times the outage duration. *Does not license:* any statement about tail latency; Little's law constrains means.

**T12 — Restoration is a series system.** If `T_total` is the sum of four sequential phases each with its own variability, the distribution of `T_total` has a mean equal to the sum of means but a *variance* equal to the sum of variances, so its coefficient of variation shrinks relative to the individual phases. *Licenses:* using the sum of measured means as the RTO evidence, and expecting restoration times to be more consistent than their worst phase. *Does not license:* using the sum of *best-case* phase durations as an RTO — that is a lower bound which no real restoration achieves.

**T13 — Conjunction structure of gates.** If gates are independent vetoes, the release passes only when all pass; averaging them into a score destroys the property that a single critical failure blocks. *Licenses:* the requirement that safety and closure gates are vetoes, not weighted contributors. *Does not license:* the claim that gates are independent in practice — a bundle that fails closure often also fails reproducibility, so the effective number of independent gates is smaller than the count.

**T14 — The checklists decay result (operational, not mathematical).** For any checklist with items whose failure modes have been automated away, the ratio of load-bearing to total items falls monotonically unless items are retired; and the probability that a reviewer reads any given item falls as the item count grows. *Licenses:* mandatory item retirement per review cycle, and capping the item count for the highest-risk path. *Does not license:* a specific optimum count; that depends on review time available, and it must be measured rather than asserted.

**T15 — Sampling without replacement in cohort assignment.** If the cohort is a deterministic hash function of a stable identifier mapped to `[0,1)` and compared to the share `w`, then the cohort is a uniform sample of identifiers with expected share `w`, and the same identifier is consistently in or out. *Licenses:* cohort stability across requests (important for per-user metrics and for consistent experience) and an unbiased request sample in expectation. *Does not license:* representativeness on dimensions correlated with the hash input — hashing an account ID samples accounts uniformly, not traffic uniformly, and a small number of high-volume accounts can dominate the request mix.

**T16 — The value of a paired comparison.** For two measurements on the same input with per-input variance `sigma^2` and independent errors, the unpaired comparison of means has variance `2sigma^2/n`, while the paired comparison of differences has variance `2sigma^2(1-rho)/n` where `rho` is the correlation between the two measurements on the same input. *Licenses:* the claim in Section 6.2 that pairing can reduce the required sample by an order of magnitude when `rho` is high (typical for two models on the same request). *Does not license:* pairing when the two measurements cannot be made on the same input — for example when acting on the new bundle's output changes the input for the next request.

A note on using this list. The theorems that most change practice are T2 (rollback coherence), T5 (canary power), T9 (slice blindness), and T10 (sequential inflation), because each of them overturns a common belief that is held confidently. The theorems that are easiest to state and hardest to satisfy in practice are T1 (closure) and T4 (expand/contract), because both require an *enumeration* that no tool can produce on your behalf. A release process is, in the end, a set of enumerations kept honest by gates.

> **Takeaway:** Most release failures are conditional statements whose condition was dropped; keep the condition attached to the result and the failure becomes visible before it happens.

## 11. Interview / Exam Questions

Fifteen questions spanning recall, calculation, selection, and diagnosis. Each has a checked answer, the reasoning that produces it, and a red flag naming a specific misconception that would cause a real defect.

**Q1 (recall).** What is the unit of release for an ML system, and why is the model artifact alone insufficient?

*Answer.* The release bundle: a closed set of artifacts and configuration (model, tokenizer, preprocessing image, feature definitions, index snapshot, prompt template, guardrail policy, thresholds) identified by an immutable manifest. The model alone is insufficient because runtime behaviour is a function of all of these, so a change in any one of them alters behaviour without a new model version.
*Reasoning.* Closure: if any behaviour-determining input is outside the manifest, the manifest cannot identify behaviour, so evaluation evidence collected on the model does not transfer to the released system.
*Red flag.* Saying "the model plus the code" and stopping. Code and weights are the two members every registry handles; the four that cause incidents are the ones outside the registry — feature definitions, index snapshots, prompt templates, and policy configuration.

**Q2 (recall).** State the rollback coherence condition and give the smallest counterexample to "rollback means re-pointing the model alias".

*Answer.* Rollback succeeds iff `Delta ∩ (args(f) \ R) = ∅`, i.e. nothing whose change caused the observed difference is left unrestored. Counterexample: the release changed the model and the guardrail threshold; the alias is re-pointed; the threshold remains raised; behaviour differs from pre-release while the tool reports success.
*Reasoning.* RL-06 applied with `Delta = {model, threshold}` and `R = {model}`: the intersection is `{threshold}`, non-empty.
*Red flag.* Describing rollback as "deploy the previous version". That is a description of a deployment action, not of a behavioural restoration, and the gap between them is the most common cause of a failed incident recovery.

**Q3 (calculation).** A service has a 99.95% availability SLO over a 30-day window. Compute the error budget in minutes and in failed requests if it serves 5,000 requests per minute.

*Answer.* `W = 43,200` min. Budget `= 43,200 × 0.0005 = 21.6` minutes. Requests in window `= 5,000 × 43,200 = 216,000,000`. Budget in requests `= 216,000,000 × 0.0005 = 108,000`. Consistency check: `21.6 × 5,000 = 108,000`, which agrees under the uniform-rate assumption.
*Reasoning.* RL-02 and RL-03 are the same rearrangement applied to time and to counts; the agreement holds only for constant traffic.
*Red flag.* Writing `0.95` for 99.95% (confusing the percentage with the complement), or reporting the budget without stating which convention it is in. Both errors are common and both produce budgets that are wrong by orders of magnitude.

**Q4 (calculation).** A canary receives 40 requests/second for 20 minutes. The baseline error rate is 0.2%. What is the smallest absolute increase in error rate it can detect at 5% one-sided alpha and 80% power? Express it also as a relative increase.

*Answer.* `n = 40 × 1,200 = 48,000`. `d_min = 2.4866 × sqrt(0.002 × 0.998 / 48,000) = 2.4866 × sqrt(4.1583e-8) = 2.4866 × 2.0392e-4 = 5.071e-4`. That is 0.0507 percentage points absolute, or `0.000507/0.002 = 25.4%` relative.
*Reasoning.* RL-08 with `z_alpha + z_beta = 1.645 + 0.8416 = 2.4866`. The check is to confirm `n p0 = 96 >= 5`, so the normal approximation is usable.
*Red flag.* Quoting the detectable *relative* change without the baseline, or claiming the canary would catch a 10% regression. A 10% regression at this baseline needs roughly `(25.4/10)^2 ≈ 6.4×` the sample, i.e. about 2.1 hours at the same rate.

**Q5 (calculation).** A 30-day budget is 43.2 minutes. After 6 hours, 4.5 minutes of unavailability have accumulated. Compute the burn rate and the time to exhaustion at that rate.

*Answer.* `b = 43,200 × 4.5 / (43.2 × 360) = 194,400 / 15,552 = 12.5`. Time to exhaustion `= 43.2/4.5 × 6 h = 9.6 × 6 = 57.6 h`. Check: at `b = 12.5` the whole budget is consumed in `30 days / 12.5 = 2.4 days = 57.6 h`. Agrees.
*Reasoning.* RL-09, then the identity that exhaustion time equals `W/b`.
*Red flag.* Comparing 4.5 minutes to 43.2 and concluding "we have used 10% of the budget, so we are fine" — that statement is arithmetically true and operationally wrong, because 10% was consumed in 0.8% of the window. The burn rate is the correct framing and the level is not.

**Q6 (diagnosis).** A canary passes at 5% and 25%; at 100% a regression appears. Rank the possible causes and describe the one check that most efficiently distinguishes them.

*Answer.* Most likely (1) an unrepresentative cohort — the canary sampled a segment that is unaffected; (2) a detection-latency-versus-dwell mismatch, where the signal was present but below the alarm's ability to fire within the dwell; (3) an unrelated event coinciding with expansion. The most efficient single check is to compare the canary cohort's distribution on the main segment dimensions against full traffic *and* recompute the canary-stage metric restricted to the segment that regressed at 100%. If the canary population barely contained that segment, cause (1) is proven and the others are excluded.
*Reasoning.* The restricted recomputation is decisive because it uses data already collected; it converts a speculation into an observation in minutes.
*Red flag.* Concluding "the canary was too short" reflexively. That is cause (2), and if the real cause is (1) then lengthening the canary changes nothing — the cohort still excludes the affected segment.

**Q7 (selection).** You must release a new prompt template and a new guardrail threshold together. What is the minimum safe rollout, and why can a single canary not answer the question you need answered?

*Answer.* A shadow stage to compare outputs on 100% of traffic, then a canary, then a staged ramp, with each member isolated if possible. A single canary of the combined change cannot attribute a failure to either member, so if it fails you cannot roll back selectively, and if it passes you have learned nothing about which change carried the risk. If the two members can be released separately, do so: two releases with one change each attribute cleanly.
*Reasoning.* Attribution requires that `Delta` be a singleton, or that the observation be paired per member. A combined release makes `Delta` have two elements and any observed effect ambiguous.
*Red flag.* Answering "canary at 5% for an hour" without addressing attribution. This is the reasoning that produces incidents where the team rolls back a prompt change that was fine because the threshold change was harmful.

**Q8 (selection).** Under what conditions is a forward fix preferable to a bundle rollback?

*Answer.* Three conditions must hold together: the defect is understood (not merely observed); the fix is testable in minutes (a config value, a threshold, a flag); and the fix's own failure mode is bounded and known. When all three hold, the forward fix has a smaller blast radius because it is faster (shorter `T_total`), as the Section 4.5 arithmetic shows. When any fails, rollback is correct because it restores a known-coherent state.
*Reasoning.* RL-13 makes the trade explicit: exposure is proportional to `T_total`, and the forward fix wins on that term — but it introduces a new, untested state whose failure would start a second cycle.
*Red flag.* "Always roll back first, then investigate" or "always forward fix, rollbacks are slow". Both are policies that fit some systems; stated as absolutes they indicate the trade-off was never analysed.

**Q9 (diagnosis).** Rollback completed; the alias points at the old model; error rates are normal; customer complaints continue for three hours. What is happening and what do you do?

*Answer.* Irreversible effects were emitted during exposure (messages sent, orders accepted, decisions published). The service is healthy because the bundle is coherent; the harm is in the effects, which rollback cannot retract. The action is the compensation path: enumerate effects in the exposure window from the effect log, execute reversals or corrective messages, and record the residual harm. Simultaneously, add an effect-rate stop rule so the next release stops on effects rather than only on health.
*Reasoning.* T3: rollback restores state, not effects; RL-07 accounts for the residual. The signal that distinguishes this from a botched rollback is that service metrics are normal while complaint volume is not.
*Red flag.* Re-running the rollback, or rolling forward again. Both treat a completed action as incomplete and typically make the state less coherent, while the actual harm continues unaddressed.

**Q10 (judgment).** A colleague proposes a composite "release readiness score" that weights reproducibility, compatibility, quality, safety, and operations into a single number with a pass threshold. What is wrong with it?

*Answer.* It destroys the veto property. A safety gate exists to block a release that fails it; in a weighted sum, an excellent quality score can compensate for a failed safety gate, which is precisely the trade the gate was created to forbid. Gates must be conjunctions. A score can be useful as a *reporting* device after the conjunction has been evaluated, never as the decision rule.
*Reasoning.* T13. Also, the weights are unmeasurable in the units that matter: a reproducibility failure and a 1% latency regression do not share a scale, so any weighting encodes an arbitrary trade as if it were a measurement.
*Red flag.* Agreeing because "dashboards need a single number". A dashboard can show five lights; the desire for one number is a desire to avoid the hard veto, and it usually originates with whoever is blocked.

**Q11 (calculation/diagnosis).** You roll back after 15 minutes with a 2% failure rate on 3,000 req/min, and your measured restoration time is 22 minutes. Compute affected requests and the share of a 129.6-million-request monthly budget consumed.

*Answer.* `affected = 3,000 × (15 + 22) × 0.02 = 3,000 × 37 × 0.02 = 2,220`. Budget `= 129,600,000 × 0.001 = 129,600` requests. Share `= 2,220/129,600 = 1.71%`. In time convention (counting only the failing fraction as unavailable): `2,220/3,000 = 0.74` minutes of the 43.2, also 1.71%.
*Reasoning.* RL-13 then RL-03. The two conventions agree here only because the failure ratio was applied uniformly; they would diverge for an outage where all requests fail for 5 minutes.
*Red flag.* Using `15 + 22 = 37` minutes × 3,000 = 111,000 as "affected", which assumes every request failed. That overstates by a factor of 50 and would panic a budget review for no reason.

**Q12 (judgment).** Your organisation releases 40 times a week across teams with a 6-item gate list. A proposal adds 20 more items to cover observed incidents. Evaluate.

*Answer.* The proposal treats each incident as an independent gap and adds an item per incident, which is how checklists reach 60 items and stop being read (Limit 9 of 4.4). The better process: for each incident, ask whether an existing item failed (fix the execution), was absent (add one item), or was unverifiable (convert or delete). Also add a retirement rule: the review cycle deletes at least one item whose failure mode has been automated away, keeping the count bounded. With 40 releases a week, review time per release is the binding constraint, and a 26-item list at 40/week is a ritual by construction.
*Reasoning.* T14: the load-bearing ratio falls unless items are retired, and readership falls with item count. The constraint is the reviewer's minutes, not the list's coverage.
*Red flag.* "More incidents mean we need more checks." Incidents mean the process has a defect too, and often that defect is that items were ticked without evidence, which more items make worse.

**Q13 (diagnosis).** After a release, p99 latency rises from 400 ms to 430 ms, below the 450 ms SLO. The canary saw 405 ms and passed. There is no other signal. What do you do?

*Answer.* This is a case where the correct action depends on the budget and on the trend, not on a single comparison. Compute the burn rate; if latency is trending toward the SLO, the projection matters more than the current value. Check whether the 25 ms difference is within the noise of the p99 estimate at the canary's sample size — p99 estimates are far noisier than means and require large samples. Then either accept with a tighter stop rule and a shorter evaluation interval, or roll back if the trend projects a breach inside the remaining window.
*Reasoning.* A p99 is a quantile estimate, so its sampling error is governed by the density near the quantile and is much larger than for a mean; a 25 ms difference measured on a canary may be entirely sampling noise, which is exactly what a paired comparison would resolve.
*Red flag.* "It's within SLO, ship it" without a trend projection, or rolling back on a difference that has not been shown to exceed its own uncertainty. Both substitute a binary reading for an interval.

**Q14 (judgment).** Explain why a per-account ring rollout can look cleaner than a hash-based canary and still be worse.

*Answer.* Per-account rings make the unit of exposure the account, which matches how harm is often experienced (a business customer notices a degraded experience more than an individual request), and they avoid the same user seeing inconsistent behaviour. But rings ordered by account tier, tenure, or geography are not random samples of traffic, so the ring's metric is an estimate for the ring population, not for the population. A hash-based canary gives an unbiased request sample but can produce inconsistent per-user behaviour and may under-sample small accounts. The choice is between unbiasedness of the request sample and coherence of the user experience, and the correct answer depends on which metric the gate uses.
*Reasoning.* T15 establishes that hashing an identifier samples identifiers, not traffic; ring ordering adds a second layer of selection.
*Red flag.* "Rings are best practice" or "hashed canary is unbiased". Both are half-true, and using either as a universal rule means the cohort's mismatch with the judged population goes unexamined.

**Q15 (judgment).** You are asked to sign a release approval for a bundle whose rollback drill has never been run, arguing that rollback is obviously simple. What is your response, and what would change your mind?

*Answer.* Refuse for any risk class above the lowest, citing that `T_execute` being simple does not make `T_total` acceptable — the drill measures `T_detect`, `T_decide`, and `T_verify` too, and those are usually larger than the mechanical step. What would change the decision: a recorded drill for an architecturally identical bundle within the class threshold, or a documented degraded mode that contains harm without restoration, with its own test. "Rollback is obviously simple" is a claim about one of four terms in RL-12.
*Reasoning.* Section 5.5's decomposition shows a 22-minute `T_total` of which only 3 minutes is execution. The drill is the only instrument that produces the other three numbers.
*Red flag.* Accepting the argument because the alias flip is indeed one command. The command is not the recovery; it is 14% of it.

The distribution: three recall (Q1, Q2, and the definitional half of Q6), five calculation (Q3, Q4, Q5, Q11, and the arithmetic half of Q13), four selection (Q7, Q8, Q14, Q12 partly), and four judgment or diagnosis (Q9, Q10, Q15, Q13 partly). A candidate who answers the calculations and misses Q10 and Q15 has the numeracy without the veto instinct, which is the more dangerous profile for a release owner.

> **Takeaway:** The questions that separate a competent release owner from a numerate one are the ones about coherence, attribution, and vetoes — the places where the arithmetic is easy and the structural claim is what is being tested.

## 12. Quick Reference Card

| Field | Content |
|---|---|
| **Core invariants** | Release = closed bundle with an immutable manifest (RL-04). Rollback succeeds iff `Delta ∩ (args(f) \ R) = ∅` (RL-06) — restoring the model is not restoring behaviour. Gates are conjunctions with veto power; never average them into a score (T13). Closure is containment, not equality: the manifest may hold members the serving path never reads. |
| **Budget and burn arithmetic** | `B = W(1-a)` minutes; `F_max = N(1-a)` requests; the two agree only under uniform traffic. `b = W·consumed/(B·t)`, with `b = 1` meaning exact exhaustion at window end; exhaustion time is `W/b`. Declare the convention (time or requests) before the release, not after a dispute. |
| **Canary sizing** | `n_min ≈ (z_alpha + z_beta)^2 p0(1-p0)/d^2` (RL-08), with `z_alpha + z_beta = 2.4866` at 5% one-sided and 80% power. Detectable `d` falls only as `1/sqrt(n)`, so halving it costs 4x the sample. Require `detect_time(s) <= dwell(s)` at every stage, or relabel the stage as exposure without protection. |
| **Blast radius** | `affected = lambda × (T_stage + T_total) × q` (RL-13), with `T_total = T_detect + T_decide + T_execute + T_verify` (RL-12). Irreversible effects require a compensation path and an effect-rate stop rule, because rollback restores state and not effects (T3). |
| **Failure signatures** | Alias-only rollback → compare live manifest to `B0` member by member. Canary passed, full failed → recompute the stage metric restricted to the regressed segment. Shadow caused effects → sentinel identifier in every downstream store. Drift not caught → check `detect_time` versus dwell and the trend, not just the level. |
| **Selection rules** | Shadow before canary (outputs then consequences); canary before ramp (bounded exposure); ramp with the longest dwells at the lowest shares (they carry only ~2% of pre-full exposure). Prefer rollback over forward fix unless the defect is understood, the fix is minutes away, and its own failure mode is bounded. |
| **Review record minimum** | Manifest ID and member list; risk class and approver; evidence artifacts (not assertions); training and evaluation windows; cohort predicate and stage plan with dwells; stop rules with thresholds, windows, and a named owner; rollback target manifest and drill timestamp; monitoring queries; post-release review owner; known limitations. |
| **Stop rules for the reviewer** | Stop and re-derive if: a member of `args(f)` is undeclared; a stage's dwell is shorter than its alarm's earliest-fire time; slice gates exist without a sample-size commitment; the rollback drill is older than the class threshold; the rollback target was not revalidated after a schema contract step; the reviewer is the author on an R3+ release. |

**Review record.** Version 2.0, drafted 2026-11-08, independently reviewed: no. Evidence: invariants DERIVED from the definitions in Section 3; all traffic rates, error rates, dwell times, prices, and engineer-hour figures are ILLUSTRATIVE constructed values whose arithmetic was recomputed and cross-checked in Sections 5 and 9; no incident, vendor capability, benchmark, or production outcome is claimed. Open items: independent review of Sections 4.2 and 4.3; replacement of the illustrative flop-rate and price assumptions with measured values before any budget decision; a worked example of the paired-comparison variant of RL-08.




