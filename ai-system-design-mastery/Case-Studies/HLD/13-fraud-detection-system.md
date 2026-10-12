# Fraud Detection System — Real-Time Scoring, Label Delay and Adversarial Drift

> **Source note.** Fraud detection is named explicitly in the roadmap video (T1, phase 4: *"ML practice design and Netflix system plus fraud detection system"*). The transcripts name it without designing it, so the design below is **supplementary** — standard industry practice and published payments-industry material, not quoted from the transcripts. Numbers are labelled **[A]** (stated assumption for the exercise) or **[D]** (derived arithmetic). Nothing here is presented as measured telemetry.

> **TL;DR.** A fraud detection system is a **real-time binary classifier with a latency budget, an extreme class imbalance, and a label that arrives weeks later** — and those three properties, not the model choice, determine the architecture. The base rate is roughly 1 fraud per 1,000 transactions, which means a model that sounds excellent — 99.9% specificity — still produces about one false positive for every fraud it catches. So the design is dominated by the **operating point**, not by accuracy: what precision/recall trade you choose, and what you do with the ambiguous middle (challenge, don't block). The two structural hard problems are **label delay** (chargebacks take 30–120 days, so you cannot train on today's truth) and **selective labels** (you only learn the outcome of transactions you *approved*, so your training set is biased by your own past decisions — a feedback loop that quietly degrades the model).

> **60-second interview answer.** "I'd clarify the latency budget and the cost asymmetry first, because they drive everything. Suppose 1B transactions a day — 12,000 TPS average, ~36,000 peak — with about 0.1% fraud. That base rate is the whole problem: at 0.1%, even 99.9% specificity gives me ~1M false positives against 1M frauds, so precision is only ~50% and I've declined a million legitimate payments. So I would not optimise for accuracy; I'd set an explicit operating point on the precision-recall curve, and I'd make the decision three-way — approve, challenge (step-up auth), block — because a challenge converts a false positive into mild friction instead of a lost sale. Architecturally: a rules layer for fast deterministic cases, an ML scorer for the long tail, a feature store that serves online and offline from one definition to avoid train/serve skew, and a case-management queue for analyst review. Then the two things I'd flag as genuinely hard: labels arrive 30–120 days late via chargebacks, and I only ever see labels for what I approved — so the training data is selectively labelled and needs explicit correction."

---

## 1. Definition & mental model

**Definition.** A fraud detection system scores an event — a card authorisation, a login, an account opening, a refund — in real time, and returns a decision that is enforced before the event completes. It is a decisioning system with a classifier inside it, not a classifier with an API bolted on.

| Component | Job | Latency |
|---|---|---|
| **Event ingestion** | Normalise and enrich the transaction stream | ~ms |
| **Feature computation** | Online features (velocity, device, graph) consistent with offline definitions | < 50 ms |
| **Rules layer** | Deterministic, auditable, instant decisions | < 5 ms |
| **ML scorer** | Score the ambiguous middle | < 50 ms |
| **Decision engine** | Map score + policy → approve / challenge / block | < 5 ms |
| **Case management** | Human review of borderline and high-value cases | minutes–days |
| **Learning loop** | Reconcile chargebacks, retrain, replay, deploy | days–weeks |

**Mental model — the airport security queue, not the courtroom.** You cannot investigate everyone, so you triage. Most people walk through (approve). A few are pulled aside for a secondary check (challenge — step-up authentication). A very few are stopped (block). The design question is where to set the two thresholds, and the answer is a business decision about the relative cost of friction versus loss, not a modelling decision.

**What it is *not*.** It is not an accuracy-maximisation problem (accuracy is meaningless at a 0.1% base rate — "approve everything" scores 99.9%). It is not a batch analytics job (the decision must precede the authorisation). And it is not a static model (the adversary adapts, so the model decays by construction).

---

## 2. Why it exists — the problem it solves

**The money.** Card fraud losses run to tens of billions of dollars annually. A detection system exists to move money from the "lost" column to the "declined" column, while not moving so much legitimate traffic into "declined" that you destroy revenue and customer trust.

**The asymmetry that shapes everything.** The two error types are not symmetric:

| Error | What happens | Cost |
|---|---|---|
| **False negative** (fraud approved) | Goods shipped or funds moved; chargeback follows | Direct loss + chargeback fee + possible penalty |
| **False positive** (legit declined) | Customer's payment fails | Lost sale + support cost + abandonment + possible churn |

Neither is free, and the ratio between them is a business input you must ask for, not assume. Typical framing: a false negative costs the transaction value plus a fixed fee; a false positive costs some fraction of the transaction value in lost margin plus a support contact. That ratio determines the operating point.

**The third path is the key design insight.** Because a false positive is expensive, the decision is not binary. **Step-up authentication** (3-D Secure, OTP, biometric) converts a probable-false-positive into a challenge: the legitimate customer confirms and completes the purchase, the fraudster abandons. This single design move is worth more than most modelling improvements, because it lets you operate at a *lower* threshold — catching more fraud — without paying the full false-positive cost.

---

## 3. Internal mechanics

### 3.1 Architecture

```
  Transaction / event stream (card auth, login, transfer)
        │
        ▼
  ┌──────────────────┐
  │  INGESTION &     │  normalise · enrich (merchant category, geo, BIN)
  │  ENRICHMENT      │  assign a decision id  ← idempotency key
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────┐
  │  ONLINE FEATURE STORE  (low-latency KV + streaming)  │
  │   • velocity       : txns on this card in 1m/1h/24h  │
  │   • device         : fingerprint, first-seen age     │
  │   • graph          : shared card/device/address/IP   │
  │   • behavioural    : amount vs member's own baseline │
  │  ── ONE definition, served to both paths ──          │
  └────────┬─────────────────────────────────────────────┘
           │
           ▼
  ┌──────────────┐   pass    ┌──────────────┐  score   ┌────────────────┐
  │ RULES LAYER  │──────────►│  ML SCORER   │─────────►│ DECISION ENGINE│
  │ deterministic│           │ (GBDT/NN)    │          │ 3-way policy   │
  │ auditable    │◄──block───└──────────────┘          │ approve/       │
  │ < 5 ms       │                ▲                    │ challenge/block│
  └──────────────┘                │                    └───────┬────────┘
                                  │                            │
        ┌─────────────────────────┘                            ▼
        │                                          ┌──────────────────────┐
        │                                          │  Enforcement:        │
        │                                          │  allow · 3DS/OTP ·   │
        │                                          │  decline · hold      │
        │                                          └──────────┬───────────┘
        │                                                     │
        │                                          ┌──────────▼───────────┐
        │                                          │  CASE MANAGEMENT     │
        │                                          │  analyst queue, SLA  │
        │                                          └──────────┬───────────┘
        │                                                     │
        │        ┌────────────────────────────────────────────┘
        │        │
  ┌─────▼────────▼─────────────────────────────────────────────────┐
  │  LEARNING LOOP  (asynchronous, days–weeks)                     │
  │   chargeback / dispute feed  →  reconcile labels               │
  │   ↳ SELECTIVE-LABEL CORRECTION (we only see outcomes for       │
  │     what we APPROVED — blocked fraud never charges back)       │
  │   → retrain → replay/shadow on live traffic → ship            │
  └────────────────────────────────────────────────────────────────┘
```

### 3.2 Why the rules layer stays

A pure-ML design is a mistake here, for four reasons:

1. **Auditability.** Regulators and card networks demand that certain decisions be explainable in plain language. "This rule fired" is defensible; a score of 0.83 is not.
2. **Instant, unambiguous cases.** A card on the hot-list, a sanctioned country, a velocity limit breached by 100× — these need no model.
3. **Latency headroom.** Rules resolve the easy majority in under 5 ms, leaving the model budget for genuinely ambiguous traffic.
4. **Organisational velocity.** Analysts can add a rule today in response to an active attack without waiting for a model retrain.

The trade-off: rules accumulate, overlap, and become unmaintainable — so they need ownership, expiry dates and a periodic cull, exactly like alerts.

### 3.3 The feature store is the load-bearing component

This is where most real implementations fail. The model is trained on features computed in a batch warehouse; it is served features computed in a streaming pipeline. If those two computations differ even slightly — a different time window, a different null policy, a different timezone — the model sees a distribution at serving time that it never saw in training. That is **train/serve skew**, and it silently degrades precision in production while offline metrics look fine.

The fix is structural: **one feature definition, two execution engines.** The same transformation logic is expressed once and materialised both offline (for training, point-in-time correct) and online (for serving, low-latency). Point-in-time correctness matters enormously — a training row must see only feature values that were *knowable at that moment*, or you leak the future into training and the model looks brilliant offline and useless live.

### 3.4 The classification problem, stated honestly

| Property | Reality | Consequence |
|---|---|---|
| Base rate | ~0.1% (~1 in 1,000) | Accuracy is useless; use precision/recall at an operating point |
| Class imbalance | ~1:1,000 | Plain logistic regression on raw data predicts "legit" always; needs weighting, resampling, or a ranking objective |
| Label delay | Chargebacks in 30–120 days | Cannot train on recent data with labels; must handle a "matured" window |
| Label bias | Only approved transactions get outcomes | **Selective labels** — the training set is a biased sample of reality |
| Adversary | Adapts to the model | Concept drift by construction; the model decays without retraining |
| Latency | < 100 ms total budget | Limits model complexity and feature richness |

### 3.5 The two genuinely hard problems

**Problem 1 — selective labels (the feedback loop).**
You learn the ground truth only for transactions you *approved*: a fraud you blocked never produces a chargeback, so it appears in your data as a legitimate transaction that you wrongly declined. Train naively on this data and the model learns that anything you blocked was fine — it literally teaches itself to stop blocking. The bias compounds with every retrain.

Mitigations, in order of strength:

- **Randomised exploration.** Approve a small, random fraction of transactions the model would have blocked, so you observe their true outcome. Costly (you eat some fraud) and requires an explicit budget, but it is the only way to get unbiased labels.
- **Reject inference.** Statistically reweight or impute outcomes for the declined population — cheaper, weaker, and always an assumption rather than a measurement.
- **Analyst-review labels.** Cases a human reviews get a human label regardless of the automated decision; analyst throughput caps how much signal this yields.
- **Model the decision separately from the outcome.** Train the fraud model only on approved transactions (where you have honest labels) and treat the *policy* as a separate optimisation. Cleaner, but means the model never learns about blocked patterns.

**Problem 2 — label delay.**
A chargeback arrives 30–120 days after the transaction. So:

- The freshest fully-labelled data is ~4 months old, meaning the model trains on a world that no longer exists (the adversary has moved).
- You need **matured-window training**: train on data old enough to be fully labelled, then blend with a short-horizon signal (analyst-confirmed fraud, immediate customer complaints) to recover recency.
- **Provisional labels** are the pragmatic answer: treat analyst-confirmed fraud as a positive immediately, treat "no chargeback within N days" as a weak negative, and correct when the chargeback finally lands.

---

## 4. Variants & taxonomy

| Variant | How it works | When it applies | Cost |
|---|---|---|---|
| **Rules-only** | Hand-written thresholds | Low volume, high-value, regulated; day-one systems | Brittle, blind to novel patterns, unmaintainable at scale |
| **ML-only** | One model scores everything | Rarely defensible in payments | Unexplainable decisions; no instant-path latency |
| **Rules + ML (standard)** | Rules for deterministic cases, ML for the tail | Most card and platform fraud | Two systems to operate |
| **Supervised batch retrain** | Daily/weekly retrain on matured labels | Stable fraud patterns | Slow to adapt to a new attack |
| **Online learning** | Continuous update from streaming labels | Fast-moving attack surfaces | Instability, feedback-loop amplification, hard to audit |
| **Graph / network** | Score on relationships (shared device, card, address, IP) | Organised fraud, bust-out rings | Expensive to compute and store; entity resolution is hard |
| **Unsupervised anomaly** | Flag deviation from a member's norm | Novel fraud with no labelled precedent | Noisy; high false-positive rate; better as a feature than a decision |
| **Sequence / behavioural** | Model the session or navigation path | Account takeover, bot traffic | Needs session data; heavier feature pipeline |
| **Three-way decision** | approve / challenge / block | **The default good design** | Adds a step-up auth dependency |

---

## 5. Pros / Cons

### Rules + ML (three-way decision)

| Pros | Cons |
|---|---|
| Rules give explainability and instant response to an active attack | Rule sprawl without ownership and expiry |
| ML catches novel patterns rules miss | ML decisions are hard to explain to a regulator |
| Three-way decision lowers the effective cost of a false positive | Depends on a step-up auth channel being available and usable |
| Model and policy are separately tunable | Two subsystems, two on-call surfaces |

### Randomised exploration (for unbiased labels)

| Pros | Cons |
|---|---|
| The only source of unbiased labels | Deliberately approves some fraud — a real, visible loss |
| Keeps the model from collapsing over retrains | Needs an explicit budget and executive buy-in |
| Measures the true cost of the current policy | Politically difficult after any visible fraud incident |

### Graph features

| Pros | Cons |
|---|---|
| Catches organised rings invisible per-transaction | Entity resolution is genuinely hard |
| Strong signal — shared device across 40 cards is damning | Storage and compute cost; latency for multi-hop traversal |
| Detects slow-burn bust-out fraud | Privacy and data-retention obligations |

---

## 6. Exceptions, edge cases & failure modes

| # | Failure | Mechanism | Mitigation |
|---|---|---|---|
| 6.1 | **Selective-label collapse** | Model retrains on outcomes only for approved traffic; learns that blocked traffic was fine | Randomised exploration budget; reject inference; model the policy separately |
| 6.2 | **Train/serve skew** | Batch and streaming feature logic diverge; offline metrics look fine, live precision collapses | One definition, two engines; log served features; monitor for distribution shift |
| 6.3 | **Label leakage** | A training row includes a feature computed after the decision point | Point-in-time-correct feature materialisation; explicit event-time joins |
| 6.4 | **Adversarial adaptation** | Fraudsters probe the boundary, then shift just inside it | Continual retraining; deliberately vary thresholds; monitor the *distribution* of scores, not just the flag rate |
| 6.5 | **Threshold drift / alert fatigue** | Score distribution shifts slowly; the fixed threshold starts flagging too much | Monitor flag rate as an SLI; alert on it; recalibrate on a schedule |
| 6.6 | **Analyst queue saturation** | The queue exceeds review capacity, so cases age past usefulness | Cap queue depth; auto-resolve low-value cases; route by expected loss |
| 6.7 | **Velocity-feature staleness** | A hot key's counter is behind, so a burst looks normal | Bounded staleness budget on streaming features; fail closed for high-risk features |
| 6.8 | **Feature-store outage** | Online feature lookup fails mid-decision | Degrade to rules-only and say so; never block everything by default |
| 6.9 | **Merchant/gateway retry storm** | A client retries a declined authorisation, producing duplicates | Idempotency key per logical payment; dedupe by it |
| 6.10 | **Class-imbalance resampling harm** | Aggressive oversampling distorts the probability calibration | Calibrate after resampling; operate on the calibrated output |
| 6.11 | **Cold start on new entities** | New card, new device, no history | Use entity age as a feature; fall back to rules until enough signal accrues |
| 6.12 | **Seasonal / legitimate spikes** | Black Friday looks like a velocity attack | Calendar and merchant-category features; per-cohort baselines rather than global ones |
| 6.13 | **Friendly fraud** | A legitimate cardholder disputes a real purchase | Not separable at authorisation time; handle in the dispute flow, not the model |
| 6.14 | **Silent model degradation** | Nothing is "down", so uptime monitoring never fires | Quality SLO on precision/recall against a delayed label; model-drift monitoring |
| 6.15 | **Regulatory / explainability challenge** | A declined customer demands a reason; the model cannot produce one | Rules layer and SHAP-style explanations; adverse-action reason codes |

---

## 7. When to use / When NOT to use

**Use real-time ML scoring when:**

- Volume is high enough that manual review cannot touch most traffic.
- The decision must precede the event (authorisation, transfer, login).
- You have (or can acquire) labelled history.
- The loss from fraud exceeds the cost of friction.

**Use rules-only when:**

- Volume is low and the value per transaction is high (manual review is viable).
- The regulatory environment demands fully explainable decisions.
- You have no labels yet — rules are the correct day-one system while you accumulate them.

**Do NOT build a real-time ML scorer when:**

- Your latency budget forbids it (some batch settlement flows do not need it).
- You have no label-acquisition path — a model without labels is guesswork with extra steps.
- Fraud is not the dominant loss (if chargebacks are mostly friendly fraud, a model at authorisation time cannot help; fix the dispute flow instead).

**Decision rule:** *high volume + latency-bound decision + labels exist ⇒ rules + ML + three-way decision. Low volume or no labels ⇒ rules, and invest in label acquisition first.*

---

## 8. Numbers, defaults & tuning

> Assumptions marked **[A]**; arithmetic **[D]**.

| Quantity | Value | Basis |
|---|---|---|
| Transactions / day | 1B | [A] |
| Average TPS | 1B / 86,400 ≈ **12,000 TPS** | [D] |
| Peak TPS | ~36,000 (3× average) | [D] |
| Fraud rate | 0.1% | [A] typical order |
| Fraudulent txns / day | **1,000,000** | [D] |
| Legitimate txns / day | **999,000,000** | [D] |
| Total decision budget | < 100 ms | [A] |
| Rules layer | < 5 ms | [A] |
| Online feature lookup | < 50 ms | [A] |
| Model scoring | < 50 ms | [A] |
| Chargeback label delay | 30–120 days | [A] typical |
| Feature staleness budget | seconds to minutes | [A] |
| Analyst review SLA | minutes (high value) to days | [A] |

### The arithmetic that decides the design

This is the calculation to produce in an interview, because it immediately reframes the problem:

**Specificity 99.9% — a number that sounds excellent:**

```
False positives = 0.1% × 999,000,000 legitimate = 999,000
True positives  = 1,000,000 fraud (assume perfect recall)

Precision = 1,000,000 / (1,000,000 + 999,000) ≈ 50%
```

**Half your alerts are wrong, and you have declined ~1M legitimate transactions.** A model described as "99.9% accurate at rejecting fraud" is, at this base rate, commercially unusable.

**What precision 90% actually requires:**

```
Allowed false positives = 1,000,000 / 0.9 − 1,000,000 ≈ 111,000
Required specificity    = 1 − 111,000 / 999,000,000 ≈ 99.989%
```

So you need **99.989% specificity**, not 99.9% — a difference of ~110× in false-positive rate, and in practice a very different model, feature set and operating point.

**The three-way decision changes the economics.** If a *challenge* converts, say, 80% of false positives into completed legitimate purchases, the effective cost of a false positive falls by roughly 5×, which lets you lower the block threshold and catch more fraud at the same business cost. Quantify that in the interview and you have made the strongest available point about this design.

**Metric set to quote:**

| Metric | Meaning | Typical target framing |
|---|---|---|
| Precision at operating point | Of flagged, how many were fraud | Set by the FP cost budget |
| Recall at operating point | Of fraud, how much caught | Set by loss tolerance |
| Fraud $ caught vs $ lost | Money moved from loss to blocked | The business-facing number |
| False-positive rate | Legit declined / legit total | The friction budget |
| Challenge conversion rate | Share of challenged who complete | Determines the three-way payoff |
| Analyst alerts per day | Queue load | Must fit review capacity |
| Time-to-detect a new attack | Time from attack onset to coverage | The adversarial-adaptation metric |
| Model drift (PSI / KS) | Score-distribution shift | Leading indicator of decay |

---

## 9. Interview Q&A

**Q1. How would you frame this problem before designing?**
I would ask three questions first, and refuse to design until I had approximations. What is the latency budget — is this an authorisation decision with a sub-100 ms bound, or a post-hoc review with minutes? What is the cost ratio between a false negative and a false positive, because that sets the operating point and therefore the whole design? And do I have labels, and how delayed are they? Those three answers change the architecture more than any modelling choice. Then I would state the base rate, because at 0.1% every "accuracy" claim is meaningless.

**Q2. Why is accuracy the wrong metric here?**
Because "approve everything" achieves 99.9% accuracy at a 0.1% base rate. Accuracy is dominated by the majority class and tells you nothing about the only thing you care about. The right framing is precision and recall evaluated at a specific operating point on the curve, plus the business-facing metrics — fraud dollars caught versus lost, and the legitimate-transaction decline rate. You choose a point on the curve, you do not "maximise accuracy."

**Q3. Do the arithmetic: how good does the model actually need to be?**
With 1B transactions and 0.1% fraud, there are 1M frauds and 999M legitimate transactions. At 99.9% specificity — which sounds excellent — I get 999,000 false positives. Against a perfect-recall 1M true positives, precision is only ~50%: half my alerts are wrong and I have declined a million legitimate payments. To reach 90% precision I need false positives under ~111,000, which means specificity of ~99.989%. That is the real bar, and it is why the feature set and the operating point matter far more than squeezing another point of AUC.

**Q4. Why keep rules at all if you have a model?**
Four reasons. Auditability — some decisions must be explainable in plain language to a regulator or a card network, and "rule 42 fired" is defensible where a score is not. Latency — rules resolve the easy majority in under 5 ms, leaving model budget for genuinely ambiguous traffic. Instant response — when an attack is live, an analyst can add a rule today rather than waiting for a retrain cycle. And cold start — with no labels yet, rules *are* the system. The cost is rule sprawl, so they need owners and expiry dates and a periodic cull.

**Q5. What is selective-label bias and why does it matter?**
You only observe the outcome of transactions you approved — a fraud you blocked never produces a chargeback, so in your data it looks like a legitimate transaction you wrongly declined. Train naively and the model learns that everything you blocked was fine; it teaches itself to stop blocking, and the bias compounds every retrain. It is the single most under-appreciated failure mode in this domain. The only fully sound fix is a randomised exploration budget — deliberately approve a small random fraction of what the model would block, so you observe the truth — plus reject inference or analyst labels as weaker alternatives.

**Q6. Labels arrive 120 days late. How do you train?**
Train on a matured window: data old enough that chargebacks have landed. That means the model trains on a world roughly four months stale, so I blend in short-horizon signals — analyst-confirmed fraud and immediate customer complaints — to recover recency. Provisional labels are the pragmatic mechanism: treat analyst-confirmed fraud as an immediate positive, treat "no chargeback within N days" as a weak negative, and correct the label when the chargeback arrives. The trade is that provisional labels are noisier but fresh, and matured labels are clean but old, so you weight accordingly.

**Q7. How do you prevent train/serve skew?**
Structurally, not by discipline: one feature definition, two execution engines. The transformation logic is expressed once and materialised both offline, point-in-time correct, and online, low-latency. Point-in-time correctness is the subtle half — a training row must see only feature values knowable at that instant, or the future leaks into training and the model looks superb offline and fails live. Then I would log the served features and monitor their distribution against the training distribution, because skew that is not monitored is skew you discover from a business metric.

**Q8. Which features actually carry signal?**
Velocity is the strongest family — transactions on this card in the last minute, hour, day, and the ratio to that member's own baseline, since absolute counts mean nothing across different spending habits. Device and entity age — a card used from a device first seen four minutes ago is a different proposition. Graph features — shared device, address, IP, or BIN across many accounts — catch organised rings that are invisible per-transaction. And behavioural deviation: amount versus this member's own distribution, not a global threshold. Notice that almost all of these are *relative to the member*, which is the recurring theme.

**Q9. Why three-way instead of approve/block?**
Because the false-positive cost is what limits how aggressive you can be, and a challenge breaks that link. If a step-up authentication converts most false positives into completed legitimate purchases, the effective cost of a false positive drops by roughly 5×, which lets me lower the block threshold and catch more fraud at the same business cost. It also improves the customer experience: "confirm this payment" is far better than "your card was declined". The dependency is that a usable step-up channel must exist — and if it does not, I am back to a binary decision and a more conservative threshold.

**Q10. How do you know the model is degrading?**
Two monitors, because they fail differently. Uptime monitoring never fires here — nothing is "down" while precision quietly collapses, which is the trap. So I need (a) a quality SLI computed against delayed labels: precision and recall at the operating point, tracked over time; and (b) leading indicators that move before labels arrive — score-distribution drift, feature-distribution drift, flag rate, and challenge conversion rate. Flag rate is the most useful early signal: a threshold that starts flagging 3× more volume is either a real attack or a model whose calibration has drifted, and both need attention now.

**Q11. How would you catch an organised fraud ring?**
Per-transaction features will not do it, because each individual transaction looks unremarkable. You need entity linkage: build a graph where cards, devices, addresses, phone numbers and IPs are nodes and shared usage creates edges, then score on neighbourhood properties — how many distinct cards have used this device, how recently this cluster formed, whether the cluster's transactions share a merchant pattern. The hard part is entity resolution, not the model: deciding when two phone numbers or two addresses are really the same entity. It is also the feature family with the heaviest privacy and retention obligations, so it needs a data-governance story alongside the technical one.

**Q12. Your feature store goes down. What happens?**
Degrade, do not fail closed. If the online feature lookup is unavailable, the decision engine falls back to the rules layer, which is deterministic and needs no feature store, and I mark those decisions as degraded in the logs so they are excluded from model-quality metrics. Failing closed — declining everything — would be far more damaging than the fraud risk. The exception is specific high-risk rules, which can fail closed individually. I would also make the degradation visible: an alert on fallback rate, because silently operating on rules alone means silently losing model coverage.

**Q13. How do you handle a legitimate spike like Black Friday?**
Per-cohort baselines rather than global thresholds, plus calendar and merchant-category features so the model knows what day it is. The global velocity distribution genuinely does shift on Black Friday, so a model that only knows absolute counts will flag enormous volumes. Features that compare the member to themselves, and features that encode time and merchant context, absorb most of it. I would also expect to recalibrate the operating point for the peak period deliberately, accepting a different precision/recall trade for those days rather than pretending the distribution is unchanged.

**Q14. What is friendly fraud and does the model handle it?**
Friendly fraud is a legitimate cardholder disputing a purchase they actually made — either by mistake or deliberately. It is not separable at authorisation time, because at that moment the transaction is indistinguishable from a genuine one; the fraud is in the *dispute*, not the payment. So it belongs in the dispute flow, not the detection model. The practical implication for design is that the fraud rate you can measure from chargebacks overstates the fraud you could ever prevent at authorisation, which matters when you set loss expectations.

**Q15. ⚡ Pushback: "This is just a binary classifier. Why is it a system design question?"**
Because the classifier is the easy part and the surrounding system is where it fails. The base rate means the operating point, not the model, determines the business outcome. The label delay means training data is months stale. The selective-label bias means a naively retrained model teaches itself to stop working. The latency budget constrains which features and models are even available. And the adversary means the model decays by construction, so the *retraining pipeline* and its monitoring are as load-bearing as the model. I would happily spend most of the interview on the feature store, the label pipeline and the drift monitoring, because that is where production fraud systems actually break.

**Q16. ⚡ Pushback: "Why not just use a deep sequence model instead of gradient-boosted trees?"**
Sometimes yes, but the burden of proof is on the heavier model. Gradient-boosted trees on good tabular features give most of the lift, train fast, score in microseconds, and — importantly here — are far easier to explain to an auditor. A sequence model buys you session-level behavioural signal, which matters for account takeover and bot traffic, and it costs latency, explainability, and much more machinery. So I would ship trees first, measure where the errors actually are, and add a sequence model only for the sub-problem where the errors justify it. That is also the honest answer: the choice is empirical, not a matter of fashion.

**Q17. ⚡ Pushback: "Approving random traffic just to get labels means knowingly allowing fraud. Is that defensible?"**
It is a cost, not a failure, and it has to be stated that way to stakeholders. The alternative is a model that silently degrades and produces worse outcomes overall — you pay in accuracy instead of in a small, bounded, measurable exploration cost. The defensible framing is: this is a deliberate, budgeted, capped experiment whose purpose is to keep the decision policy honest, and here is the measured loss it incurs versus the accuracy it buys. It should be a decision made explicitly by whoever owns the fraud loss, not a hidden technical choice. And it is capped — a small fraction of would-be-blocked traffic, not a general amnesty.

---

## 10. Common mistakes candidates make

1. **Optimising accuracy.** At a 0.1% base rate it is meaningless. Reframe on precision/recall at an operating point, immediately.
2. **Never doing the specificity arithmetic.** If you do one calculation in this interview, do the 99.9%-specificity-⇒-50%-precision one. It signals domain understanding instantly.
3. **Skipping the label pipeline.** Treating labels as given. The delay and the selective-label bias are the two hardest parts of the problem.
4. **Binary approve/block.** Missing step-up authentication, which is the single highest-leverage design move available.
5. **Forgetting latency.** Proposing graph traversal or a large model inside a sub-100 ms budget without addressing the cost.
6. **No feature store / no skew discussion.** Train/serve skew is the most common real-world cause of a model that looks good offline and fails live.
7. **No drift story.** Uptime monitoring never catches a decayed model. Name a quality SLI and a leading drift indicator.
8. **Ignoring explainability.** Regulated decisions need reason codes; a score alone will not survive an audit.
9. **Forgetting idempotency.** Clients and gateways retry declines; without a decision idempotency key you double-count and double-decide.
10. **Designing for a static adversary.** Fraudsters probe and adapt; the retraining cadence is part of the architecture.

---

## 11. Cheat-sheet summary

| | |
|---|---|
| **It is** | A real-time decisioning system with a classifier inside it |
| **Base rate** | ~0.1% — the number that dictates everything |
| **The killer arithmetic** | 99.9% specificity ⇒ ~999k FP vs 1M TP ⇒ **~50% precision** |
| **Real bar for 90% precision** | ~**99.989% specificity** |
| **Metric** | Precision/recall at an operating point · fraud $ caught vs lost · legit-decline rate |
| **Never** | Accuracy · AUC alone |
| **Latency budget** | < 100 ms total: rules < 5 ms · features < 50 ms · model < 50 ms |
| **Architecture** | Ingest → online feature store → rules → ML scorer → 3-way decision → case mgmt → learning loop |
| **Best design move** | **Three-way: approve / challenge / block** — step-up auth collapses false-positive cost, letting you catch more fraud |
| **Hard problem 1** | **Label delay** 30–120 days ⇒ matured-window training + provisional labels |
| **Hard problem 2** | **Selective labels** — you only see outcomes for what you approved ⇒ randomised exploration budget |
| **Feature families** | Velocity · device/entity age · graph linkage · amount-vs-own-baseline |
| **Skew** | One definition, two engines; point-in-time-correct joins |
| **Degradation** | Fail *open* to rules-only; never fail closed |
| **Leading drift signal** | Flag rate and score distribution — they move long before labels arrive |
| **The close** | "If the volume dropped so manual review became viable, I'd drop the real-time model and run rules plus analysts, which is more accurate and more explainable at low scale." |

---

## 12. Cross-links

- [Netflix — streaming and recommendation](12-netflix-recommendation-and-streaming.md) — the other ML system named alongside this one in the source roadmap
- [Uber ride-hailing](06-uber-ride-hailing.md) — contains an in-context fraud/payment-risk discussion
- [RAG evaluation & observability](../../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — quality metrics against delayed ground truth
- [SLOs, error budgets & incident response](../../Operations/03-slos-error-budgets-and-incident-response.md) — the quality SLI that uptime monitoring cannot provide
- [Observability & monitoring](../../Operations/02-observability-and-monitoring.md) — drift and distribution monitoring
- [Capacity estimation playbook](../../HLD/01-foundations/05-capacity-estimation-playbook.md) — the TPS and base-rate arithmetic
- [Choosing a database](../../HLD/04-data/05-choosing-a-database-decision-guide.md) — online feature store and graph-store choices
- [Blob & object storage](../../HLD/04-data/04-blob-and-object-storage.md) — feature and label archive, retention policy
