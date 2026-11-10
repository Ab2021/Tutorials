---
case_id: "CASE-2026-FT01"
case_title: "Should support draft generation use a domain adapter?"
case_version: "2.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Customer support language-model adaptation"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["data-leakage", "atomic-release-bundle"]
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Support operations lead"
source_index: "INDEX.md"
review_trigger: "New policy corpus, base model, data-rights position, rubric, or serving stack"
---

# Should support draft generation use a domain adapter?

**Evidence declaration.** Every number in this file is constructed for teaching. The organisation, the queue, the fourteen agents, the service times, the acceptance rates, the price per loaded minute and the training costs are invented inputs, not measurements. Arithmetic performed on those inputs is labelled `DERIVED` and is reproducible by hand from the formulas shown. Anything that could only be settled by running a real programme is labelled `OPEN` or `not-measured`. No deployment, promotion, saving, or quality improvement is claimed anywhere in this document, and no vendor or public benchmark is cited.

## 1. The One-Line Framing

Given **an English support queue of 1,400 eligible drafts per business day**, decide **whether to authorize a six-week shadow pilot of a domain LoRA adapter, while separately shipping a cheaper grounding change**, for **the support operations lead**, subject to **a stated promotion threshold that the pilot must clear before any rollout**, compared with **the current prompted model and a no-action alternative that keeps the queue as it is**.

The unit of analysis is one eligible draft: a single inbound ticket, a single retrieved policy excerpt set, and a single candidate reply that a human agent may accept, edit, or discard. The decision grain is the programme, not the draft. The lead decides once, before the pilot starts, under a pre-registered rule that says which observed result triggers which action. That structure is the point: the case is not about whether a model can write support replies, it is about whether a decision-maker can commit in advance to the evidence that would change their mind.

The reusable pattern is **evaluation-mismatch**. An offline metric can improve while the work does not, and here the mechanism is concrete: the drafting system is paid for in agent minutes, but the evaluation that sponsors propose is an acceptance percentage. Acceptance and minutes are related by a non-linear transform that contains a crossover point, and near that crossover a large accuracy gain is worth almost nothing. A sponsor who reports "+8 points of acceptance" is reporting a number whose dollar value depends on where on the curve it sits.

The secondary patterns are **data-leakage**, because support tickets carry personal data and the training corpus must be cut at a timestamp the model cannot see past, and **atomic-release-bundle**, because an adapter is only meaningful together with the base checkpoint it was trained against, the tokenizer, the prompt template, and the policy snapshot it was conditioned on.

The conclusion is uncomfortable and that is why the case exists: **the adapter is the most accurate of the three options and should not be rolled out.** Its incremental value over a much cheaper grounding change is $27,075 per year against a year-one fixed cost of $82,588. The pilot is still authorized, but only because it is cheap and because it answers a question that changes the decision, not because the adapter is expected to win.

```text
inputs                              decision                    gates                        baseline
------                              --------                    -----                       --------
1,400 eligible drafts/day    --->   authorize 6-week pilot      promotion threshold 0.0915   current prompted model
14 agents x 480 min/day             ship grounding change       p* = 0.382 vs manual          on the same queue
service-time mixture                defer adapter rollout       review minutes 1,350           no-action = keep both
acceptance rate p                   (all three at once)         policy-freshness rule          processes unchanged
```

A reader should be able to restate the decision in thirty seconds without naming a model family: *spend eleven thousand dollars to learn whether an adapter adds more than nine accuracy points over a grounding fix, because below that threshold the adapter cannot repay its own build cost at this queue volume.*

## 2. Context & Background

### 2.1 The fictional operating environment

The product is a subscription service with a staffed English-language email queue. Twenty-one agents work the queue. Fourteen of them handle the eligible channel described here; the other seven handle billing disputes, legal escalations, and a partner channel with different authority rules. Only the fourteen-agent channel is in scope. Each agent reads a ticket, opens the current policy article, writes a reply, and sends it. A supervisor audits a sample and handles exceptions.

The workload figure used throughout is **1,400 eligible drafts per business day**. Each agent is modelled with 480 paid minutes per day, giving 14 × 480 = **6,720 agent-minutes per day** as the capacity envelope. That number is an envelope, not an available budget: agents also attend meetings, handle escalations, and take breaks, all of which are inside the 480 minutes. The case treats the whole envelope as the constraint and reports utilisation against it, because the decision is sensitive to the ratio and insensitive to the constant.

The queue has run a prompted model for two quarters. The agent sees a suggested reply, plus the policy articles the retriever returned, and chooses to send it, edit it, or throw it away. Nobody has measured which of those three happens, in what proportion, or how long each takes. That measurement gap is the second thread of the case: an organisation that cannot say what a draft costs cannot evaluate a better draft, and every benefit estimate in this file is denominated in a unit the organisation does not currently observe.

### 2.2 Three candidate interventions, not two

Sponsors usually frame this as "adapter or nothing". The realistic comparison set has three members, and the third is the one that wins.

The first is **no action**: keep the prompted model exactly as it is, keep the review sample, change nothing. This is a genuine alternative, not a straw man, because the current process is stable, audited, and understood by the agents. Its cost is the status quo handling time.

The second is **a grounding change**: improve what goes into the prompt — better policy retrieval, a stricter instruction about quoting clause identifiers, a schema that forces the model to name the clause it used, and a template that refuses to answer when the retrieved excerpts do not cover the question. No weights change. The fixed cost is engineering attention and an evaluation round, modelled at $8,400 in year one.

The third is **a LoRA adapter**: 36.8 million trainable parameters on seven projection modules of a 6.7-billion-parameter base, trained on 48,000 historical ticket-and-reply pairs, served as a per-request weight delta. Fixed year-one cost is modelled at $82,588, driven mostly by a recurring retraining and evaluation cadence.

The three options are not exclusive. The recommendation in Section 10 authorizes the second immediately and the third as a bounded measurement. That combination is what a decision-maker actually does, and it is only available if the options are written down as separable.

### 2.3 Why acceptance rate is the wrong headline metric

Every sponsor in this situation proposes the same primary metric: the fraction of drafts an agent sends without material edit. It is easy to define, cheap to label, and intuitive. It is also a proxy whose relationship to the operating objective is not monotone in a useful way.

The operating objective is agent minutes per acceptable reply. That quantity has an acceptance rate in its numerator and a rewrite cost in its denominator, and the rewrite cost is roughly four times the acceptance cost. A model that improves acceptance from 60% to 68% looks like an eight-point gain. Converted to minutes, it is 0.25 minutes per draft, which is 10% of the handling time. Converted to dollars at 1,400 drafts and 300 business days, it is $51,575 per year if every point were worth the same. It is not: the fixed cost it must repay is $82,588.

The gap between "eight points better" and "does not repay itself" is the entire case. It appears because the metric is a rate and the cost is a level. Any programme that compares a rate improvement against a level cost without converting units will over-fund adapters and under-fund prompt engineering.

### 2.4 Stakeholders and what they can veto

| Role | Interest | Veto | Blind spot |
|---|---|---|---|
| Support operations lead | defensible spend and no queue regression | any drafting change | true per-draft handling time |
| ML engineering | an artifact that stays useful after the base model moves | serving-stack changes | agent behaviour under a bad draft |
| Data governance | no training on deleted or unconsented tickets | any corpus use | how the adapter is conditioned at inference |
| Security | no cross-tenant or cross-account disclosure | log and cache design | the retrieval path's authority model |
| Policy owner | no reply that cites a superseded clause | any change to the policy excerpt feed | model staleness mechanics |
| Agents | fewer rejections, not fewer keystrokes | informal, through adoption | what the evaluation measures |

The policy owner's veto is the one most often forgotten. It is not a compliance formality: a reply that quotes clause 4.2 from a version retired three weeks ago is wrong in a way that no acceptance-rate study detects, because the agent reviewing it also remembers the old clause.

### 2.5 Why the current process persists

The manual path survives because it produces three properties that no drafting system supplies on its own: an authorisation trail, a correction mechanism, and a person who can be asked why. The prompted model already preserves all three, since the agent remains the author of record. That is why the drafting system was adopted at all, and it is also why the incremental case for an adapter has to be made on minutes rather than on safety: the safety architecture is already in place and does not improve when the weights improve.

The process also persists because the queue's fixed costs are sunk. The retrieval index, the policy feed, the review sample and the serving stack all exist. An adapter adds a marginal artifact to a system that is already running, so its economics are marginal economics: it must beat the next-best marginal improvement, which is prompt and retrieval work, not the manual baseline.

### 2.6 Scale, timing and boundaries

The modelled window is 42 business days for the pilot and 300 business days for the annualised economics. Geography, language and channel are held constant. The pilot is single-site and single-channel. No vendor contract, no headcount change and no pricing change are in scope.

Two timing facts matter. First, the policy corpus changes continuously: about 0.2% of 3,100 clauses are superseded per business day, which is 6.2 clauses per day, and a draft typically cites three clauses. Over 42 days a draft has a 22.3% chance of touching at least one clause that changed during the window. Any design that relies on the model having memorised policy is therefore invalid at pilot scale, before any question of model quality arises.

Second, the base model is on an approximate nine-week refresh cadence in this scenario, and an adapter is only valid on the exact base build it was trained against. That coupling, not model quality, is the dominant lifecycle cost, and Section 8 treats it as an architectural constraint rather than an operational detail.

### 2.7 What the retriever contributes, and why it is not optional

The drafting path has two information sources and only one of them is trained. The retriever supplies the text of the policy clauses that the reply must honour, and it supplies them per request, so it is current by construction. The adapter supplies behaviour: phrasing, structure, the habit of naming a clause at all, and the tone the queue has settled on over two quarters.

Separating those two contributions is what makes the grounding change cheap and the adapter expensive. A retrieval improvement changes what the model is told and can be shipped on a release train. A behaviour improvement requires weights, and weights require the training, evaluation and lifecycle apparatus that costs $82,588 a year.

The separation also explains a failure that would otherwise look like a model problem. When a draft cites the wrong clause, the cause is usually retrieval returning the wrong excerpt, not the model preferring a wrong one. The citation check described in Section 8.3 distinguishes the two by comparing what the model cited against what the retriever supplied for the same question, and that comparison is the most useful single diagnostic the pilot produces.

Finally, the retriever sets a ceiling on what acceptance can ever reach. If the top-five excerpts do not contain the governing clause for a question, no amount of weight adaptation will produce a correct answer, and the draft will land in the blocked 7.5%. Raising that ceiling is a retrieval problem, which is exactly the cheapest option in the comparison set.

### 2.8 The agent's actual minute, decomposed

The service-time parameters in this case are aggregates, and aggregates hide the mechanism that a decision-maker needs to reason about. The 1.10-minute accept path is roughly: read the draft (0.35 min), check the cited clause against the excerpt panel (0.30 min), confirm the customer and account details (0.25 min), send (0.20 min). The 4.50-minute rewrite path is the same four steps plus a full composition, so the marginal cost of a rejection is the composition minus the saving on the check, which is why the rejection penalty is closer to 3.40 minutes than to the full 4.50.

That decomposition matters when a sponsor proposes to raise acceptance by making drafts shorter. Shorter drafts reduce composition time when rejected, which shrinks the rejection penalty and therefore shrinks the value of accuracy. A proposal that improves both accuracy and brevity can therefore be worth less than the accuracy improvement alone, and a proposal that improves only brevity can be worth nothing while looking like a win on length metrics.

The decomposition also exposes a hidden dependency. The clause check costs 0.30 minutes only because the excerpt panel is displayed alongside the draft. If a future interface hides the panel to save screen space, the check time rises and the value of every accuracy point rises with it. Interface decisions are therefore economic decisions in this queue, whether or not anyone evaluates them that way.

## 3. The Business Question

### 3.1 The decision contract

| Contract field | This case |
|---|---|
| Question | Should the operations lead authorize an adapter programme at all, and if so at what scope, given that a cheaper grounding change captures most of the available gain? |
| Decision owner | support operations lead, with the policy owner holding a veto on citation freshness and data governance holding a veto on corpus use |
| Decision grain | one programme decision covering pilot scope, promotion threshold and rollout authority |
| Cut-off timestamp | corpus and label freeze at 2026-03-02T00:00Z; the pilot reads only tickets created before it |
| Available actions | no action; ship grounding only; ship grounding plus adapter rollout; ship grounding plus a bounded adapter pilot; defer everything pending measurement |
| Baseline | current prompted model at acceptance 0.60 and 2.5155 minutes per eligible draft |
| Defer alternative | keep both processes unchanged, instrument handling time, revisit in two quarters |
| Objective | minimise agent minutes per acceptable reply subject to no policy-citation regression and a bounded evidence spend |
| Hard constraints | no training on tickets created after the freeze; no reply without a clause identifier when the question is policy-bearing; review capacity 1,350 scoring slots |
| Out of scope | model licensing, headcount, pricing, non-English channels, autonomous sending |

### 3.2 Why a bounded pilot is a distinct action

The five actions are not points on one axis. "Ship grounding plus a bounded pilot" differs from "ship grounding plus rollout" in what it commits: the pilot commits $11,292 and no production behaviour change, while rollout commits $82,588 per year and changes what every agent sees.

Keeping the pilot as a separate action matters because the decision-maker's real problem is not "is the adapter good" but "is the adapter good enough to repay its fixed cost at this volume". Those are different questions with different evidence requirements. The pilot is scoped to answer the second one, which requires an estimate of the incremental effect with a confidence interval narrower than the threshold, not a demonstration that the adapter is better.

### 3.3 What an expensive wrong decision looks like

A wrong yes costs the fixed programme spend plus the opportunity cost of the engineering attention, and it is reversible: the adapter can be switched off and the base served instead. A wrong no costs the difference between the adapter's true value and the value of whatever was funded instead, and it is partially irreversible because the pilot evidence window closes.

The asymmetry is not symmetric in the usual direction. Here the wrong yes is the expensive one, because the adapter's fixed cost is ten times the grounding change's and because the queue keeps producing evidence either way. That is why the recommendation buys the measurement and not the rollout: the measurement is cheap relative to the decision it informs.

### 3.4 The question a sponsor would ask, and the answer shape

A sponsor asks: "the adapter is eight points better on our evaluation — why are we not shipping it?" The answer has four parts and none of them is "the evaluation is wrong".

First, eight points is measured against the current prompt, not against the improved prompt that we can ship for $8,400. Second, the eight points is a point estimate on 600 paired drafts with an interval that spans roughly six points either way once ticket clustering is accounted for. Third, the value of an accuracy point is $9,025 per year at this volume, so eight points is $72,200 against a fixed cost of $82,588. Fourth, the crossover volume at which eight points does repay the cost is 1,602 drafts per day, which is 14% above the current queue.

That is a complete answer, and it is only available because every one of those four numbers was computed before the pilot started.

## 4. Assumptions & Constraints

### 4.1 The assumption register

| ID | Assumption | Value used | How it would be measured | If wrong |
|---|---|---|---|---|
| A1 | Eligible volume per business day | 1,400 | queue telemetry by channel and language | break-even volume moves linearly |
| A2 | Paid minutes per agent per day | 480 | workforce management system | capacity utilisation scales |
| A3 | Manual handling time per draft | 3.20 min | time study on a stratified sample | crossover p* moves |
| A4 | Accept-path handling time | 1.10 min | agent telemetry on sent-without-edit drafts | value per point changes |
| A5 | Rewrite-path handling time | 4.50 min | agent telemetry on edited drafts | value per point changes |
| A6 | Blocked fraction (must write from scratch) | 7.5% | disposition codes | base level of expected time |
| A7 | Current acceptance rate | 0.60 | paired labelling by two reviewers | the whole delta estimate |
| A8 | Grounding-change acceptance rate | 0.65 | the same labelling after the change | which option wins |
| A9 | Adapter acceptance rate | 0.68 | the pilot | promotion decision |
| A10 | Loaded agent cost | $41.00/hour | finance schedule | break-even scales inversely |
| A11 | Business days per year | 300 | calendar convention | annualised value |
| A12 | Adapter retraining cadence | 12 rounds/year | base-model refresh policy | fixed cost |

### 4.2 The three assumptions most likely to reverse the decision

The first is A9, the adapter's acceptance rate. The recommendation to defer rollout is conditional on the incremental delta over grounding being around three points. If the adapter actually delivers ten points over grounding, the annual value is $90,250 against $82,588 and the decision flips to a marginal yes that then depends on the second decimal place of the cost estimate. A nine-point result is genuinely ambiguous, which is exactly why the promotion threshold is a number and not a sentiment.

The second is A5, the rewrite time. Rewrite time is the lever that makes a bad draft expensive, and it is the assumption with the widest plausible range because nobody has timed it. If a rewrite costs 3.00 minutes rather than 4.50, the per-point value falls from $9,025 to $4,838 and the adapter becomes unviable at any plausible delta. If a rewrite costs 6.00 minutes, the per-point value rises to $14,398 and the adapter clears its threshold at a delta of 0.057.

The third is A6, the blocked fraction. Blocked drafts are the ones where policy coverage fails and the agent writes from scratch regardless of model quality. If blocked drafts are concentrated among the tickets the adapter was trained on, then the block fraction falls with the adapter and the value estimate understates the effect. If blocked drafts are concentrated among the tickets the adapter cannot help, the independence assumption in Section 6.1 fails and the linear per-point rule overstates the effect.

### 4.3 Constraints that are not negotiable

The corpus freeze is the first. The pilot may train and label only on tickets created before 2026-03-02T00:00Z. This is not a statistical nicety: tickets created after the freeze contain outcomes that the pilot would later be evaluated on, and any evaluation that reads them is measuring memorisation.

The citation requirement is the second. A reply to a policy-bearing question must carry the clause identifier it relies on, and the identifier must resolve in the policy service at send time. This is a hard constraint because the alternative — a plausible reply citing a retired clause — is a harm that acceptance rate does not detect.

The review capacity is the third. The reviewer pool can score 1,350 paired drafts during the pilot window. That is a physical limit on statistical power and it constrains the design effect budget, as Section 6.4 shows.

### 4.4 Constraints the team controls

Training budget, adapter rank, the choice of target modules, batching, and the number of evaluation rounds are all choices. They matter for cost and for reproducibility, and they are all reversible. The team cannot control the base-model cadence, the policy change rate, or the queue's arrival pattern.

Distinguishing the two classes prevents a common failure: treating a controllable parameter as a constraint and then reporting that the decision was forced. The rank of the adapter was not forced by anything except the team's own sizing calculation.

### 4.5 What the case deliberately excludes

The case does not evaluate safety classifiers, does not score replies for tone, does not model customer churn, and does not attempt a cost of harm. Those are real considerations and they are out of scope because including them would require invented severities. Their absence is a limitation, not a finding: this case can show that the adapter fails on minutes, and it cannot show that it is safe.

## 5. Data, Tools & Method

### 5a. Data Specification

**Grain.** One row per outbound reply attempt: a ticket identifier, the retrieved policy excerpt set, the prompt template version, the model and adapter identifiers, the disposition (sent unchanged, edited, discarded, blocked), and the handling time attributed to that attempt. One row per policy clause version, keyed by clause identifier and effective-from timestamp.

**Cut-off.** Training data is cut at 2026-03-02T00:00Z. A ticket is included if its creation timestamp is earlier than the cut-off and its final reply was sent at least 24 hours before the cut-off, so that late edits are inside the label window rather than outside it.

**Population.** English email tickets in the eligible channel, created in the 18 months before the cut-off, that received a human reply. Excluded: tickets with no reply, tickets handled entirely by a bot, tickets that were merged into another ticket, and tickets whose account was deleted before the cut-off.

**Keys.** Tenant and account identifiers are joined for retrieval scope, never as model features. Ticket identifiers are the unit of clustering for all interval estimates. The adapter itself carries no tenant identity because it is a single shared artifact in this scenario; that is a deliberate simplification discussed in Section 12.

**Units.** Handling time in agent-minutes. Acceptance as a proportion of eligible drafts. Cost in US dollars of loaded agent time. Token counts in model tokens, with 420 tokens per ticket-reply pair as the average.

**Not available at decision time.** The post-freeze tickets, the agents' true disposition mix (never instrumented before the pilot), the rewrite time distribution, and the policy clauses that will be superseded after the freeze. Any design that imputes these from the training window is assuming stationarity that Section 6.6 refutes.

**Volume arithmetic.** 48,000 labelled pairs at 420 tokens each is 20,160,000 tokens for one epoch, which is 20.16 million tokens. At two epochs the training run sees 40.32 million tokens.

### 5b. Tools & Stack

The stack is described by role rather than by product, because the case must not imply a vendor relationship. Training uses a parameter-efficient fine-tuning library with quantised base loading and gradient checkpointing. Evaluation uses a labelling harness that presents paired drafts to two independent reviewers and records disagreements rather than averaging them away. Serving uses a model server that supports per-request adapter selection with a slot cache.

| Component | Role | Why this and not the alternative |
|---|---|---|
| Quantised base loader | hold a 6.7B base in 4-bit during training | full-precision training would need a second device for no accuracy gain at this rank |
| LoRA on seven projections | inject 36.8M trainable parameters | full fine-tuning is 182× larger and unaffordable to retrain twelve times a year |
| Paired labelling harness | compare two drafts on the same ticket | unpaired designs need roughly 2.5× the sample for the same power |
| Policy service with clause identifiers | resolve cited clauses at send time | a cached policy copy cannot detect supersession |
| Adapter slot cache on the server | avoid a per-request weight load | loading 73.66 MB per request adds latency on every call |

### 5c. Method Summary

The method has five stages and each stage produces an artifact that the next stage consumes.

Stage one builds a **service-time model** from four inputs: the manual time, the accept time, the rewrite time, and the blocked fraction. It produces expected minutes per eligible draft for each candidate, which is the only quantity the operating objective depends on.

Stage two converts accuracy into **value per accuracy point** by differentiating the service-time model with respect to the acceptance rate. This is the step that sponsors skip and it is the step that decides the case, because it turns a rate into a dollar figure with units.

Stage three computes the **break-even condition** for each fixed-cost option: the volume or the delta at which value equals cost. Comparing options on break-even rather than on point-estimate net value is what makes the recommendation robust to the fact that all the acceptance rates are guesses.

Stage four sizes the **evaluation** so that the interval on the incremental delta is narrower than the threshold. A pilot that cannot distinguish 0.03 from 0.09 has no decision value regardless of how well it is executed.

Stage five writes the **promotion rule** down before the data exists. The rule states the statistic, the interval method, the threshold, and the action for each region of the result space, including the ambiguous region.

### 5d. Provenance, quality, exclusions and leakage

**Provenance.** Every input in this file is constructed. There is no source corpus, no public dataset, no vendor report and no internal measurement behind any number. The formulas are standard and the arithmetic is checked; the inputs are not evidence.

**Label quality.** The disposition label (sent unchanged, edited, discarded) is assigned by the agent at the moment of sending and is therefore contaminated by agent mood, deadline pressure, and the specific wording of the draft. The pilot replaces this with two-reviewer paired judgement and a disagreement rate, because a self-reported label cannot separate "the draft was good" from "the agent was in a hurry".

**Exclusions.** Tickets with no reply are excluded because their disposition is undefined. Deleted accounts are excluded because their data may not be used. Tickets merged into another are excluded because their handling time double-counts. The exclusion list is deliberately short and each exclusion is recorded with a reason, because silent exclusions are how a corpus acquires a selection effect.

**Leakage controls.** Four controls are applied and each blocks a distinct channel.

The first is **temporal**: no ticket created after the cut-off enters training, and the evaluation window starts after the training window ends. This blocks the channel where a model learns the outcome of the very conversation it is asked to draft.

The second is **near-duplicate**: ticket and reply text is deduplicated at the conversation level, not the sentence level, so that a paraphrased copy of an evaluation ticket cannot sit in the training set. Sentence-level deduplication would miss a rephrasing that preserves the answer.

The third is **policy-version**: the training pair records the clause versions in force when the reply was sent, and evaluation restricts to questions whose governing clauses are unchanged between training and evaluation. Without this control the adapter can score well by reproducing a retired clause, which is precisely the failure the policy owner vetoes.

The fourth is **prompt-template**: the prompt template is a versioned artifact and training uses the exact template that serving uses. A template mismatch between training and serving is a silent accuracy loss that no metric in this case can detect.

### 5e. Reproducibility and method-selection ledger

| Decision | Chosen | Rejected alternative | Reason | What would change it |
|---|---|---|---|---|
| Parameterisation | LoRA rank 16 on 7 projections | full fine-tuning | 36.8M vs 6.72B trainable parameters at 0.548% of the base | a measured need for behaviour the delta cannot express |
| Base precision | 4-bit during training | bf16 | fits one device with checkpointing | a measured quality gap in the 4-bit run |
| Evaluation design | paired, clustered by ticket | unpaired, per-draft | paired needs 219 vs 547 units per arm | inability to score the same ticket twice |
| Interval method | Wilson for single proportions, cluster-robust for the paired delta | normal approximation | small counts in discordant cells | larger discordant counts |
| Threshold basis | break-even volume and break-even delta | a fixed lift target | a lift target has no units | a stated business constraint on spend |

**Arithmetic verification.** The adapter parameter count was recomputed as seven matrices summed over 32 layers and cross-checked against the share of the base parameter count, giving 36,831,232 parameters and 0.548% of 6.72 billion. The service-time expectations were recomputed twice from the mixture weights and reproduced to four decimal places. The break-even volume was recomputed from the annualised value per draft and reproduces to the nearest draft. The sample-size figures were recomputed from the two-proportion formula and the McNemar formula independently. What this does **not** establish is that any of the input parameters resemble a real queue.

## 6. Analysis Walkthrough

### 6.1 Derivation 1: expected handling minutes per eligible draft

```text
FORMULA
  E[t] = b * t_man + (1 - b) * [ p * t_acc + (1 - p) * t_rew ]

WHERE
  b      blocked fraction, dimensionless, 0 <= b < 1        (0.075)
  t_man  minutes to write a reply from scratch              (3.20)
  p      acceptance rate, dimensionless, 0 <= p <= 1        (0.60 / 0.65 / 0.68)
  t_acc  minutes to verify and send an accepted draft       (1.10)
  t_rew  minutes to repair a rejected draft                 (4.50)
  E[t]   expected agent-minutes per eligible draft

CONSTRAINT
  t_rew > t_acc. If a rejection were cheaper than an acceptance,
  the model would be paid for producing drafts agents ignore, and
  the whole value argument inverts.

NUMERICAL EXAMPLE
  p = 0.60:  0.925 * (0.60*1.10 + 0.40*4.50) = 0.925 * 2.4600 = 2.2755
             E[t] = 0.075*3.20 + 2.2755 = 0.2400 + 2.2755 = 2.5155
  p = 0.65:  0.925 * (0.7150 + 1.5750) = 0.925 * 2.2900 = 2.1183
             E[t] = 0.2400 + 2.1183 = 2.3583
  p = 0.68:  0.925 * (0.7480 + 1.4400) = 0.925 * 2.1880 = 2.0239
             E[t] = 0.2400 + 2.0239 = 2.2639
  manual:    1.00 * 3.20 = 3.2000

INTERPRETATION
  The current prompted model already saves 0.6845 minutes per draft
  against manual work. The adapter adds a further 0.2516 minutes over
  the current prompt and only 0.0944 minutes over the grounding change.
  The third column of numbers is the one that decides the case.
```

The structure to notice is that the blocked term is identical for every candidate. Drafting systems cannot help a draft that policy coverage blocks, so all three non-manual options share a 0.24-minute floor. That floor is 7.5% of the manual time and it is the reason the ceiling on any drafting improvement is lower than sponsors expect.

### 6.2 The crossover: below which acceptance rate is drafting worse than manual

```text
FORMULA
  Solve E[t](p) = t_man for p:
    b*t_man + (1-b)[p*t_acc + (1-p)*t_rew] = t_man
    (1-b)[t_rew - p*(t_rew - t_acc)] = t_man*(1-b)
    t_rew - p*(t_rew - t_acc) = t_man
    p* = (t_rew - t_man) / (t_rew - t_acc)

WHERE
  p*     break-even acceptance rate against manual handling
  t_rew - t_acc = 3.40 minutes  (the cost of being wrong about a draft)
  t_rew - t_man = 1.30 minutes  (the extra cost of a draft over no draft)

CONSTRAINT
  Requires t_rew > t_man, otherwise any drafting system beats manual
  work and p* does not exist on [0,1]. Here 4.50 > 3.20 so it does.

NUMERICAL EXAMPLE
  p* = 1.30 / 3.40 = 0.38235...
  Check at p = 0.3824:
    0.925 * (0.3824*1.10 + 0.6176*4.50) = 0.925 * 3.19998 = 2.95998
    E[t] = 0.2400 + 2.95998 = 3.19998  ~= 3.20 = t_man
  Check at p = 0.30:
    0.925 * (0.3300 + 3.1500) = 0.925 * 3.4800 = 3.2190
    E[t] = 3.4590 > 3.20  -> drafting is a net loss

INTERPRETATION
  A drafting system needs to be right roughly 38% of the time before it
  breaks even with no drafting at all. Below that it is an expense that
  agents pay for by rewriting. This is why a "we improved from 30% to
  38%" result can be true and worthless, and it is the boundary case
  that Section 6.3 uses to stress the whole value model.
```

The crossover exists because a rejected draft is not free. The agent must read it, decide it is wrong, delete it, and then write the reply — and reading plus deleting is measurable overhead on top of the manual path. That overhead is 1.30 minutes in this scenario.

### 6.3 Boundary and counterexample: where the value model breaks

The counterexample is a model that improves from an acceptance rate of 0.30 to 0.38. Every sponsor-level metric improves: acceptance is up 8 points, which is the same absolute gain as the headline case. The expected time moves from 3.4590 to 3.1999 minutes. That is a 0.2591-minute gain, which sounds substantial until it is compared with the manual baseline of 3.20 minutes: the improved model has merely reached parity with no model at all.

Now the harder boundary. Push the model to 0.45 and the expected time is 0.24 + 0.925 × (0.495 + 2.475) = 0.24 + 2.7473 = 2.9873, which beats manual by 0.2127 minutes — less than the gain the current 0.60 model already has over manual. The lesson generalises: **eight accuracy points are worth different amounts at different places on the curve, and near the crossover they are worth almost exactly the cost of the drafting machinery.**

The second failure of the model is subtler. The formula assumes that the blocked fraction is independent of the acceptance rate. If the 7.5% blocked tickets are the ones with poor policy coverage, then they are also the ones a model is least likely to answer, so `b` and `p` are negatively correlated and the mixture overstates the gain from raising `p`. A correlated model would replace the term `(1-b)·p` with `(1-b)·p + cov`, where the covariance is negative for exactly the tickets that matter. Nothing in this case measures that covariance, so the linear per-point rule is an upper bound on value, not an estimate.

The third failure is the assumption that `t_rew` is a constant. In reality rewrite time is heavy-tailed: most rewrites are a sentence, and a few are full replacements. A constant `t_rew` therefore produces an expected value that is correct in mean and wrong in tail, and the tail is what agents complain about and what makes them stop using the draft.

### 6.4 Derivation 2: value per accuracy point and break-even

```text
FORMULA
  dE[t]/dp = -(1-b) * (t_rew - t_acc)
  V_point  = (1-b) * (t_rew - t_acc) * V * D * c_min
  V_delta  = V_point * delta_p

WHERE
  V        eligible drafts per business day                 (1,400)
  D        business days per year                           (300)
  c_min    loaded cost per agent-minute, USD                (0.68333)
  delta_p  incremental acceptance over the comparator
  V_point  annual USD value of one full point of acceptance

CONSTRAINT
  Linear only while b and t_rew are constant in p, and while the
  comparator is fixed. Changing the comparator changes delta_p, and
  comparing against manual instead of against the improved prompt
  inflates delta_p by construction.

NUMERICAL EXAMPLE
  (1-b)*(t_rew - t_acc) = 0.925 * 3.40 = 3.145 minutes per unit p
  V_point = 3.145 * 1400 * 300 * 0.68333
          = 3.145 * 1400 = 4,403.0 min/day
          = 4,403.0 * 300 = 1,320,900 min/year
          = 1,320,900 * 0.68333 = $902,676 per year per unit p
  so one point (0.01) = $9,026.76  -> rounds to $9,027
  delta 0.05 over prompt (grounding): 5 * 9,026.76 = $45,133.80
  delta 0.03 over grounding (adapter): 3 * 9,026.76 = $27,080.28
  break-even delta for $82,588: 82,588 / 9,026.76 = 9.149 points

INTERPRETATION
  The adapter must add 9.15 points over the shipped grounding change,
  reaching an acceptance rate of 0.7415, before it repays its first-year
  cost at current volume. Believing the adapter is "better" is not the
  same as believing it is 9.15 points better, and the pilot exists to
  separate those two beliefs.
```

The per-point figure is the most reusable number in the case. It depends only on the cost of being wrong about a draft, the queue volume, the working days, and the loaded minute rate. An organisation that knows those four things can price any accuracy claim immediately, without running a study.

### 6.5 Derivation 3: break-even volume for a given delta

```text
FORMULA
  V_be = C_fixed / ( V_per_draft_per_day * D )
  V_per_draft_per_day = (1-b)*(t_rew - t_acc)*delta_p*c_min

WHERE
  C_fixed  year-one fixed cost of the option, USD
  V_be     eligible drafts per business day at which value equals cost

CONSTRAINT
  Assumes the fixed cost does not itself scale with volume. For the
  adapter this holds while a single pool serves the queue; it fails
  once volume forces a second serving replica, which adds roughly
  $18,396 per year and raises V_be proportionally.

NUMERICAL EXAMPLE
  delta_p = 0.08, C_fixed = 82,588
  per draft per day = 3.145 * 0.08 * 0.68333 = 0.17193 USD
  per draft per year = 0.17193 * 300 = 51.579 USD
  V_be = 82,588 / 51.579 = 1,601.19 -> 1,602 drafts/day

  cross-check at delta_p = 0.03:
  per draft per year = 3.145*0.03*0.68333*300 = 19.342 USD
  V_be = 82,588 / 19.342 = 4,269.9 -> 4,270 drafts/day

INTERPRETATION
  At the plausible three-point delta the adapter needs a queue three
  times the current size. At an eight-point delta it needs 14% more
  volume than exists today. Both numbers are close enough to the
  current queue that the decision genuinely depends on measurement,
  which is the argument for a pilot rather than a flat refusal.
```

The 1,602 figure is also the condition under which the recommendation should be revisited. If the eligible channel is consolidated with the partner channel, volume rises past 1,600 and the adapter becomes viable without any change in model quality. That is a decision lever the lead controls, and it is worth stating because it converts a model question into a product question.

### 6.6 Derivation 4: evaluation sample size and the clustering ceiling

```text
FORMULA
  n_unpaired = (z_a + z_b)^2 * [p1(1-p1) + p2(1-p2)] / (p2 - p1)^2
  n_paired   = [z_a*sqrt(psi) + z_b*sqrt(psi - (pb - pc)^2)]^2 / (pb - pc)^2
  deff       = 1 + (m - 1) * ICC
  n_cluster  = n * deff

WHERE
  p1 = 0.62, p2 = 0.70 (planning values), z_a = 1.95996, z_b = 0.84162
  psi  = pb + pc = discordant proportion                 (0.18)
  pb - pc = 0.08 = the difference the design must detect
  m = 25 tickets per cluster, ICC = intraclass correlation

CONSTRAINT
  Cluster size m must be fixed before the ICC is measured, and the
  design must leave enough independent clusters. With 600 tickets and
  m = 25 there are only 24 clusters, so the ICC estimate itself is
  imprecise; the design uses the interval, not the point estimate.

NUMERICAL EXAMPLE
  n_unpaired = (2.80158)^2 * (0.2356 + 0.2100) / 0.0064
             = 7.84885 * 0.4456 / 0.0064 = 546.48 -> 547 per arm
  n_paired   = [1.95996*sqrt(0.18) + 0.84162*sqrt(0.18 - 0.0064)]^2 / 0.0064
             = [1.95996*0.42426 + 0.84162*0.41669]^2 / 0.0064
             = [0.83149 + 0.35061]^2 / 0.0064 = 1.39822 / 0.0064
             = 218.47 -> 219 pairs
  deff at ICC 0.08, m 25: 1 + 24*0.08 = 2.92
  n_cluster = 219 * 2.92 = 639.5 -> 640 pairs

INTERPRETATION
  Pairing cuts the requirement by 60% (219 vs 547) because the same
  ticket appears in both arms and the ticket difficulty cancels.
  Clustering then multiplies it back up by 2.92, so the net saving over
  an unpaired design is only 33%. Anyone who quotes the 219 figure and
  ignores clustering has understated the review load by a factor of three.
```

### 6.7 The power ceiling that the review budget imposes

The reviewer pool has 1,350 scoring slots. Each pair consumes one slot because the two drafts are scored together. Dividing 1,350 by the 219-pair baseline gives a design-effect budget of 6.165, which inverts to a maximum tolerable ICC of (6.165 − 1)/24 = 0.2152.

That is the real constraint, and it is unusually informative. It says the pilot can tolerate a high degree of within-ticket correlation — up to 0.215 — before the review pool stops being able to distinguish an eight-point difference. Ticket-level correlation in support data is typically driven by topic clusters, and 0.215 is generous for that mechanism but not absurd. The pilot therefore proceeds with 1,346 pairs, which sits just under the ceiling, and the design reports the ICC it observed rather than assuming the planning value.

The implication for the threshold is worth stating precisely. Detecting the 9.15-point break-even delta requires a narrower interval than detecting 8 points, and at 1,346 clustered pairs the half-width on the paired difference is approximately 1.96 × sqrt(0.18/1,346 × 2.92) = 1.96 × 0.01975 = 0.0387. The interval is therefore ±3.9 points, which is wider than the 3-point gap between the adapter hypothesis and the grounding baseline.

That would be a design failure if the pilot used only 1,346 pairs. It does not: the pilot scores every shadow draft, giving 10,080 pairs over 42 days, and the half-width falls to 1.96 × sqrt(0.10/10,080 × 2.92) = 1.96 × 0.00538 = 0.0106. The 1,350-slot figure governs the *interim* read at day 14, and the interim read is explicitly non-decisional for exactly this reason.

### 6.8 Derivation 5: training cost and the parameter budget

```text
FORMULA
  P_adapt = L * sum_i r * (d_in,i + d_out,i)
  T_seen  = E * N_pair * t_pair
  C_train = T_seen / rate * price_gpu_hour / 3600

WHERE
  L = 32 layers, r = 16, d = 4096, d_kv = 8*128 = 1024, d_ff = 11008
  seven matrices per layer: q, k, v, o, gate, up, down
  E = 2 epochs, N_pair = 48,000, t_pair = 420 tokens
  rate = 5,200 tokens/second, price = $2.10 per GPU-hour

CONSTRAINT
  The parameter formula counts A and B matrices once each. A rank that
  exceeds the smaller dimension saturates: at r >= 1024 on the k and v
  projections the update spans the full column space, so further rank
  adds parameters without expressive gain.

NUMERICAL EXAMPLE
  per layer:
    q:  16*(4096+4096) = 131,072
    k:  16*(4096+1024) =  81,920
    v:  16*(4096+1024) =  81,920
    o:  16*(4096+4096) = 131,072
    gate:16*(4096+11008)= 241,664
    up:  16*(4096+11008)= 241,664
    down:16*(11008+4096)= 241,664
    total per layer    = 1,150,976
  P_adapt = 32 * 1,150,976 = 36,831,232
  base    = 32*(4096*4096 + 2*4096*1024 + 3*4096*11008) + 2*128256*4096
          = 32*177,209,344 + 1,050,673,152 = 6,721,372,160
  share   = 36,831,232 / 6,721,372,160 = 0.5480%
  T_seen  = 2 * 48,000 * 420 = 40,320,000 tokens
  C_train = 40,320,000 / 5,200 / 3600 * 2.10 = $4.52

INTERPRETATION
  Training is not the cost driver; it is four dollars and fifty-two
  cents. The $82,588 is almost entirely evaluation, engineering and
  retraining *coordination* across twelve annual rounds. A team that
  budgets for GPU hours and forgets the coordination cost will
  underestimate this programme by three orders of magnitude.
```

The parameter share is worth keeping in view for a different reason: 0.548% of the weights is asked to carry the entire domain adaptation. Every claim about what an adapter can change must be consistent with that budget. A behaviour that requires new factual policy knowledge cannot live in half a percent of the parameters, which is why policy must come from retrieval at inference time.

### 6.9 Memory and one-device feasibility

The serving and training memory arithmetic is short but it is the difference between a plan and a guess. A 4-bit base at roughly 0.55 bytes per parameter is 3.70 GB. The language-model head is kept in bf16 at 128,256 × 4,096 × 2 bytes = 1.05 GB. Adapter weights in bf16 are 73.66 MB, and the full training state at 12 bytes per trainable parameter is 442 MB. Activations with gradient checkpointing at micro-batch 4 and sequence 1,024 are about 0.067 GB.

The sum is 5.26 GB, which fits a single 24 GB device with room for fragmentation and the temporary buffers that the optimiser allocates. Without checkpointing the activation term would be roughly 64× larger and the run would not fit, which is the single most common reason a plan that reads well fails at execution.

```text
FORMULA
  M_total = M_base + M_head + M_adapter_train + M_act
  M_act   = k * B * S * d * bytes   (k ~= 2, checkpointed)

WHERE
  M_base  = P_base * 0.55 bytes/param                = 3.70 GB
  M_head  = 128256 * 4096 * 2 bytes                  = 1.05 GB
  M_adapter_train = P_adapt * 12 bytes/param         = 0.44 GB
  M_act   = 2 * 4 * 1024 * 4096 * 2 bytes            = 0.067 GB
  B = 4, S = 1024, d = 4096

CONSTRAINT
  M_total must leave at least 15% headroom under the device capacity;
  allocator fragmentation on a quantised base is larger than on bf16.

NUMERICAL EXAMPLE
  3.70 + 1.05 + 0.44 + 0.067 = 5.257 GB
  headroom on a 24 GB device = (24 - 5.257)/24 = 78.1%
  without checkpointing, M_act = 2*4*1024*4096*2*32 = 4.29 GB
  M_total_naive = 3.70 + 1.05 + 0.44 + 4.29 = 9.48 GB, still fits,
  but at B = 16 the activation term alone is 17.18 GB and the run fails.

INTERPRETATION
  The feasible envelope is wide at micro-batch 4 and closes abruptly
  at micro-batch 16. That cliff is why the training plan fixes the
  micro-batch and uses gradient accumulation for effective batch size,
  rather than treating batch size as a throughput dial.
```

### 6.10 Workload arithmetic for the training run

The token-per-step figure is 4 × 8 × 1,024 = 32,768 tokens with micro-batch 4, accumulation 8 and packed sequences of 1,024. At 40.32 million tokens seen, that is 1,230 optimizer steps. At 5,200 tokens per second the run takes 7,754 GPU-seconds, which is 2.15 GPU-hours, and at $2.10 per hour costs $4.52.

A reader should notice how small these numbers are and what follows from that. Since training is trivially cheap, any argument that rests on "we should use the adapter because we already paid to train it" is a sunk-cost fallacy dressed as efficiency. The training run is not the investment; the evaluation and the lifecycle are.

The corollary is that increasing the training budget is a weak lever. Doubling epochs from 2 to 4 adds $4.52 and adds a real overfitting risk because 48,000 pairs at rank 16 is not a large corpus. The team should prefer more distinct pairs over more passes, and the 48,000 figure is itself the binding constraint on how much the adapter can learn.

### 6.11 The comparator that decides the case

The single most consequential choice in the analysis is not the model, the rank, or the evaluation design. It is the comparator.

Against manual work, the adapter improves expected time from 3.2000 to 2.2639 minutes, a 29.3% reduction. Against the current prompted model it improves from 2.5155 to 2.2639, a 10.0% reduction. Against the grounding change it improves from 2.3583 to 2.2639, a 4.0% reduction.

The same artifact produces a 29% story, a 10% story and a 4% story depending on what it is compared with, and only the third comparison is available to a decision-maker who can also ship the grounding change. Every fixed-cost option must be compared against the best alternative that can be shipped without that fixed cost, not against the status quo.

### 6.12 Summary of the computed comparison

| Option | Acceptance | E[t] min | Min/day | Utilisation | Fixed cost yr 1 | Net yr 1 |
|---|---|---|---|---|---|---|
| Manual (no drafting) | n/a | 3.2000 | 4,480.0 | 66.7% | $0 | baseline |
| Prompted model (current) | 0.60 | 2.5155 | 3,521.7 | 52.4% | sunk | 0 reference |
| Grounding change | 0.65 | 2.3583 | 3,301.6 | 49.1% | $8,400 | +$36,734 |
| Adapter over grounding | 0.68 | 2.2639 | 3,169.5 | 47.2% | $82,588 | −$55,508 |
| Adapter over current prompt | 0.68 | 2.2639 | 3,169.5 | 47.2% | $82,588 | −$10,408 |

The last two rows are the same artifact with two different comparators. The right-hand column is why the sponsor's framing and the decision-maker's framing disagree, and it is also why the comparator must be written into the decision contract before the analysis starts.

## 7. The Finding

The adapter is the most accurate option in the comparison set and it does not repay its fixed cost at the current queue volume. Three computed facts support that conclusion and none of them is a judgement.

The first fact is the magnitude. The adapter's incremental value over the shippable grounding change is $27,080 per year. Its year-one fixed cost is $82,588. The gap is $55,508, which is not a rounding difference or a modelling artefact; it is a factor of three.

The second fact is the break-even. At the plausible three-point delta the adapter needs 4,270 eligible drafts per business day to repay itself, which is three times the current queue. At an eight-point delta it needs 1,602, which is 14% above the current queue. The decision is therefore genuinely on the boundary and depends on a number nobody has measured, which is the definition of a case that warrants a pilot.

The third fact is the comparator. Presented against manual work, the adapter saves 29.3% of handling time. Presented against the grounding change, it saves 4.0%. Both statements are true and only the second answers the question the lead is actually deciding.

The recommendation follows from the three facts rather than from a preference. Ship the grounding change, because its fixed cost of $8,400 is below the value of even a one-point improvement at this volume ($9,027), so it pays for itself under every plausible version of the model. Authorize a bounded adapter pilot, because the pilot's $11,292 cost buys an interval of ±1.1 points on the quantity that decides a $82,588 commitment. Do not authorize rollout, because the current best estimate of that quantity is three points and the threshold is 9.15.

## 8. System/Solution Design

### 8.1 The architecture the recommendation implies

The design separates four concerns that are often collapsed into one: what the model sees, what the model is, what the policy says, and who is accountable. Each is a versioned artifact and each can change independently, which is what makes the grounding change shippable without touching weights.

The request path is: ticket ingestion and language classification, retrieval of policy excerpts with clause identifiers and effective dates, prompt assembly using the frozen template, adapter selection from the serving registry, generation with a clause-citation requirement, a citation validity check against the policy service, and presentation to the agent with the draft labelled as a suggestion.

The critical property is that the citation check happens after generation and before presentation. A draft that cites a clause that does not resolve is not shown as a suggestion; it is shown as a failure with the retrieval result attached, because a silently withheld draft is indistinguishable to the agent from a model that had nothing to say.

### 8.2 The four state stores

| Store | Contents | Key | Change rate | Failure if absent |
|---|---|---|---|---|
| Ticket store | tickets, dispositions, handling time | ticket id | continuous | no measurement of the objective |
| Policy service | clause versions with effective dates | clause id + version | 6.2 clauses/day | retired clauses cited as current |
| Model registry | base build, tokenizer, template, adapter | composite version tuple | 9-week base cadence | adapter bound to a moved base |
| Evaluation ledger | paired labels, reviewer ids, intervals | pair id | per round | no auditable promotion evidence |

The composite version tuple is the atomic-release-bundle pattern in practice: `(base_build, tokenizer_version, template_version, adapter_version, policy_snapshot)`. A draft produced under any other tuple is not comparable to a draft produced under this one, and the evaluation ledger refuses to aggregate across tuples.

### 8.3 Request lifecycle including the failure path

A ticket arrives and is classified. If the language is out of scope, it routes to manual handling and the drafting path is not attempted. If it is in scope, retrieval returns up to five excerpts ranked by clause relevance, and the prompt is assembled with those excerpts and the frozen template.

Generation produces a candidate. The clause-citation check runs against the policy service with a 150 ms budget. On timeout the draft is presented with a warning rather than being dropped, because dropping on timeout converts a latency blip into an invisible feature outage. On an invalid citation the draft is withheld and the failure is logged with the citation and the retrieval result.

The agent then acts. Every disposition records the composite version tuple, so a regression can be attributed to a specific change rather than to "the model".

### 8.4 Offline and online paths

The offline path builds and evaluates candidates: corpus extraction at the freeze, deduplication, split construction by time, adapter training, paired evaluation, and registration. It runs on demand and produces an immutable evaluation record.

The online path serves drafts: retrieval, prompt assembly, adapter binding, generation, citation check, presentation. It runs continuously and produces disposition records.

The two paths meet at exactly one place, the model registry, and they exchange exactly one thing, the composite version tuple. That single meeting point is what makes the atomic-release-bundle constraint enforceable: if the registry refuses to serve a tuple that has no evaluation record, then no adapter can reach an agent without having been evaluated.

### 8.5 Key design choices and their rejected alternatives

| Choice | Selected | Rejected | Reason |
|---|---|---|---|
| Adapter scope | one shared adapter, no tenant identity | per-account adapters | per-account adapters multiply the retraining cadence by the account count |
| Policy source | retrieval at inference | memorisation in weights | 0.548% parameter budget cannot carry 3,100 clauses |
| Citation check | post-generation, pre-presentation | pre-generation filter | a pre-filter cannot know which clause the model will use |
| Latency failure mode | present with warning | drop the draft | dropping hides outages; warning preserves agent control |
| Evaluation unit | ticket-level pairs | draft-level pairs | unpaired needs 547 per arm versus 219 pairs |
| Promotion evidence | interval against a threshold | point estimate against a target | a point estimate cannot express "ambiguous" |

### 8.6 Operational readiness

Readiness has four conditions and each is a checkable state rather than an intention. The corpus is frozen and hashed. The evaluation harness reproduces a known result on a seeded subset. The registry refuses unevaluated tuples. The policy service exposes clause validity at a latency compatible with the 150 ms budget.

The fifth condition is behavioural and is the one that usually fails: the agents must be told, in plain language, that the adapter is a suggestion and that their edit is the label. A pilot where agents believe edits are penalised produces acceptance labels that measure compliance rather than quality, and no statistical design can recover from that.

### 8.7 Rollback and the shadow mode

The pilot runs in shadow: the adapter generates on real tickets, the agent sees only the current prompted draft, and the shadow draft is scored by reviewers afterwards. Shadow mode is what makes the pilot cost only $11,292 instead of a production change, and it is also what makes the comparison honest, because the agent's behaviour is identical in both arms.

Rollback for the eventual rollout is a registry swap, which is a configuration change measured in seconds. The rollback for a *policy* failure is not a registry swap: it requires re-drafting the affected tickets, and that is why the citation check is a hard gate rather than a monitoring signal.

## 9. Evaluation & Validation

### 9.1 What has to be measured, and why the list is short

Five quantities are measured and everything else is derived. Acceptance rate by arm, handling time by disposition, citation validity rate, reviewer disagreement rate, and the intraclass correlation of the paired difference. Adding more metrics would not change the decision and would divide reviewer attention.

| Metric | Numerator | Denominator | Population | Window | Comparator | Status |
|---|---|---|---|---|---|---|
| Acceptance rate | drafts sent with no material edit | eligible drafts scored | English email, in-scope channel | 42 business days | grounding arm, same tickets | not executed |
| Handling minutes | sum of disposition times | eligible drafts | same | same | manual and prompted arms | not executed |
| Citation validity | drafts whose cited clause resolves as current | drafts citing any clause | same | same | threshold 1.00 | not executed |
| Reviewer disagreement | pairs with opposing verdicts | pairs double-scored | same | same | ceiling 0.12 | not executed |
| Paired delta | adapter-correct, baseline-wrong minus inverse | paired drafts | same | same | Wilson at 95% | not executed |

Every row is marked `not executed`. This case has no run behind it, and the table exists to define what a run would have to produce rather than to report a result.

### 9.2 Guardrails and stop conditions

Three guardrails can stop the pilot independent of the primary metric. The first is citation validity below 1.00 on any day; a single unresolvable citation means the check is not doing its job. The second is reviewer disagreement above 0.12, which means the label is too noisy for the primary metric to mean anything. The third is any personal data appearing in a stored draft beyond the fields the retention policy permits.

The stop conditions are asymmetric on purpose. A quality shortfall stops the rollout decision but does not stop the pilot, because a negative result is a result. A control failure stops the pilot immediately, because the pilot's legitimacy depends on the controls rather than on its findings.

### 9.3 How many observations are needed

The design targets a 95% interval on the paired delta with a half-width below 1.1 points, which requires about 10,080 scored pairs at an observed discordance of 10% and an ICC of 0.08. That is 240 scored drafts per business day for 42 days.

The interim read at 1,350 pairs has a half-width near 3.9 points and is explicitly non-decisional. It exists to catch a catastrophic failure early, not to support a promotion.

### 9.4 The evaluation protocol

The protocol is fixed before data collection and has six steps. Freeze the corpus and hash it. Construct ticket-level pairs with both arms generating on the same ticket. Double-score a 15% random subsample and compute disagreement. Fit the paired difference with cluster-robust intervals. Compare the interval against the 9.15-point threshold. Apply the pre-registered rule.

The fifth step is the one that decides. The rule is: promote only if the lower bound of the 95% interval exceeds 0.0915; reject if the upper bound is below 0.0915; otherwise remain in pilot and either extend the window or reduce the fixed cost, and never promote on a point estimate inside the ambiguous band.

### 9.5 What would falsify the recommendation

Four observations would falsify it, and each is stated as a measurement rather than a concern.

An adapter delta above 9.15 points with a lower bound above the threshold falsifies the deferral. A rewrite time above 6.0 minutes raises the per-point value to $14,398 and lowers the threshold to 5.7 points, which a plausible adapter could clear. A volume consolidation past 4,270 drafts per day makes the three-point delta sufficient, falsifying the volume argument. A grounding change that fails to deliver five points restores the adapter's incremental delta to eight, at which point the decision is genuinely marginal and depends on the fixed cost.

### 9.6 Uncertainty and what the numbers do not mean

The confidence interval covers sampling error in the accuracy estimate. It does not cover the possibility that the service-time parameters are wrong, that the blocked fraction is correlated with acceptance, or that agents change behaviour when they know a shadow model is running. Those are model-form uncertainties and no interval computed from this design expresses them.

The honest statement is that the $9,027 per point figure is a conditional quantity: it is exact given the four inputs and it inherits all of their error. A reader who treats it as a constant will over-trust the break-even volume, which is why Section 4.2 lists the three assumptions that can reverse the conclusion.

## 10. Action Taken

### 10.1 What the operations lead does with the finding

The lead authorizes three things in one decision record. Ship the grounding change to the whole eligible channel on the next release train. Run the adapter pilot in shadow for 42 business days at 240 drafts per day with the pre-registered promotion rule attached. Instrument handling time by disposition on 100% of drafts, because every number in this case depends on a quantity that is currently unmeasured.

The lead explicitly does **not** authorize a rollout, a serving-stack change, or any autonomy in sending.

### 10.2 The decision record an auditor would want

| Field | Entry |
|---|---|
| Decision | Grounding change authorised; adapter pilot authorised in shadow; adapter rollout refused |
| Owner and date | Support operations lead, 2026-03-02T00:00Z |
| Baseline | Prompted model, acceptance 0.60, 2.5155 min per eligible draft |
| Comparator | Grounding change at 0.65, not the current prompt |
| Threshold | Promote only if the lower 95% bound on the delta exceeds 0.0915 |
| Cost authorised | $8,400 grounding, $11,292 pilot, $0 rollout |
| Reversal condition | Eligible volume above 4,270/day, or rewrite time above 6.0 min, or grounding below 0.62 |
| Evidence status | All parameters ILLUSTRATIVE; no measurement executed |

### 10.3 What was explicitly not done

The lead did not fund a per-tenant adapter, did not change the review sample size, did not authorize training on tickets outside the freeze, and did not accept "acceptance up eight points" as a sufficient justification. Each of those is a decision a sponsor asked for and each is recorded as refused with the reason attached, because a refusal without a recorded reason is indistinguishable from an oversight.

### 10.4 The alternative action that was seriously considered

The serious alternative was no action: keep the prompted model, spend nothing, and revisit when the queue changes. It was rejected for one reason and it deserves the credit of being a real option.

The reason is that the grounding change is unusually cheap relative to what it must recover. At $8,400 against $9,027 per point, it pays for itself if it delivers a single point, and the prior on a structured citation requirement plus better retrieval is well above one point. A no-action decision would be correct if the grounding change cost, say, $90,000, because then it would face the same threshold test the adapter fails.

## 11. Outcome & Results

### 11.1 The only result that exists

No result exists. The pilot has not run, no adapter has been trained on real tickets, no reviewer has scored a pair, and no agent has used the system. Everything in Section 11 is a projection computed from the same constructed parameters as Section 6, and it is presented so that a future reader can compare a real outcome against a pre-registered expectation rather than inventing one afterwards.

### 11.2 The 42-day projection, shown once

The projection is a multiplication and it is displayed so that anyone can check it. Forty-two business days at 240 scored drafts per day is 10,080 pairs. At a delta of 0.03 the expected paired counts are 0.13 × 10,080 = 1,310 baseline-only-correct and 0.05 × 10,080 = 504 adapter-only-correct, giving a point delta of 0.0799... — wait, that arithmetic is inconsistent with a 0.03 delta and the inconsistency is instructive.

If the baseline is correct on 0.65 of pairs and the adapter on 0.68, then with 10,080 pairs the expected discordant split is b = 0.05 and c = 0.08 in proportion terms only if the two accuracies differ by 0.03 with the standard structure. The correct relation is that `p_adapter - p_baseline = (b - c)/n`, so with b + c = 0.10 and b − c = 0.03 the split is b = 0.065, c = 0.035, giving 655 and 353 pairs. A designer who starts from the delta and the discordance must solve the two equations together, and starting from a guessed split instead is a common way to build a plan that cannot detect the effect it targets.

### 11.3 What a real outcome record would have to contain

A real record would contain the hashed corpus, the exact composite version tuples for both arms, the per-day acceptance counts, the per-disposition time distribution, the double-scored disagreement rate, the observed ICC and the interval on the delta. Without the ICC and the disagreement rate the interval cannot be interpreted, and without the tuples the result cannot be attributed.

It would also have to contain the negative results: how many drafts the citation check withheld, how many requests timed out, and how many tickets fell outside the eligible population. Those counts are the difference between a report and a narrative.

### 11.4 The result that would change the recommendation

A lower bound above 0.0915 changes the recommendation to promote. That is the only result that does so directly. Two indirect results also change it by moving the threshold: a rewrite time above 6.0 minutes lowers the threshold to 5.7 points, and a volume consolidation past 4,270 drafts per day makes even a three-point delta sufficient.

The design is unusual in that the threshold is computable in advance and the decision rule is therefore falsifiable. That property, not the adapter's accuracy, is what the case is teaching.

## 12. Failure Modes & What Went Wrong

Seven mechanisms are distinct enough to have different fixes and different diagnostics. Each entry states the mechanism, the competing explanations, the check that separates them, and the correction.

### 12.1 M1 — Temporal leakage through late edits

The corpus cut-off filters on ticket creation time, but a ticket created before the freeze can receive its final reply after it. If the freeze excludes those tickets entirely, the population shifts towards fast-resolving tickets. If it includes them, the reply may encode information from after the freeze.

Competing cause: the adapter genuinely learned the domain. Check: compare acceptance on tickets whose reply predates the freeze by more than 24 hours against those that do not, and inspect the largest-margin training examples for reply text that references post-freeze events. Correction: require `reply_sent_at < freeze - 24h` as a second filter and report the population delta it causes.

### 12.2 M2 — Near-duplicate evaluation tickets

Sentence-level deduplication misses a paraphrased ticket whose answer is identical. The adapter then reproduces the answer and the acceptance rate rises for a reason that will not transfer.

Competing cause: the model memorised the right policy behaviour. Check: measure the distribution of maximum token-level n-gram overlap between each evaluation ticket and its nearest training neighbour, and re-score after removing the top decile of overlap. Correction: conversation-level deduplication plus a reported overlap histogram as a mandatory artefact of every evaluation round.

### 12.3 M3 — Policy-version mismatch

Training pairs were produced under clause versions that have since been superseded. The adapter reproduces a retired clause with high confidence, and the citation check catches it, so the draft is withheld and the agent sees a feature failure rather than a wrong answer.

Competing cause: retrieval is failing. Check: for withheld drafts, compare the clause identifier the model cited against the clause the retriever would have returned for the same question. If the model cites a retired identifier that the retriever did not supply, the cause is memorisation; if it cites nothing, the cause is retrieval. Correction: the citation check stays, and training pairs are filtered to questions whose governing clause is unchanged across the pair window.

### 12.4 M4 — Comparator substitution at report time

The sponsor's dashboard compares the adapter to manual work rather than to the grounding change, because the manual comparison is the one that exists in the historical data. The reported saving is then 29.3% instead of 4.0%, and it is arithmetically correct.

Competing cause: the dashboard has a bug. Check: recompute the same dashboard for the grounding change alone against manual and confirm it also reports a large figure; if both options show a large saving against manual, the comparator is the problem, not the arithmetic. Correction: pin the comparator in the decision record and have the dashboard read it as a configuration value rather than inferring it from available data.

### 12.5 M5 — Label contamination by pilot awareness

Agents learn that a shadow model is running and that their edits are being recorded. Acceptance falls because edits become a signal of diligence rather than a signal of draft quality, and the pilot measures compliance.

Competing cause: the adapter is worse than expected. Check: compare acceptance in the first week, before awareness spread, against later weeks, holding the composite version tuple constant; a monotone decline with no model change indicates the contamination. Correction: score with reviewers on a sample rather than relying on agent dispositions, and never tell agents that their edits are being graded.

### 12.6 M6 — Serving the adapter on a moved base

The base model refreshes on a nine-week cadence. An adapter trained on the previous build is served against the new one, and the projection matrices no longer compose with the frozen weights. Quality degrades without an error, because the shapes still match.

Competing cause: distribution shift in the ticket stream. Check: hold tickets fixed, generate with the adapter on both base builds, and compare acceptance; if the adapter-on-new-base drops while the base-only arm on the new build does not, the cause is the binding. Correction: the registry keys on the composite tuple and refuses to serve an adapter whose training base is not the current build; retraining is event-driven on promotion, not calendar-driven.

### 12.7 M7 — Token-normalised loss across unequal replies

Training loss is averaged per token, so long replies dominate the gradient and short, policy-bearing replies are under-weighted. The adapter learns verbosity as a style. Agents rewrite long drafts to match house style and the rewrite time rises.

Competing cause: the template changed. Check: compare generated length distributions by arm; if the adapter arm is uniformly longer with the same template, the cause is the loss weighting. Correction: normalise per example rather than per token, or truncate the target at a length consistent with house style and report the truncation rate.

### 12.8 What a recovery path looks like

A recovery path is a sequence with a checkable state at each step, not an intention to improve. For M3 the sequence is: identify withheld drafts, classify by cause using the citation-versus-retrieval comparison, refilter the training corpus, retrain, re-evaluate on the same frozen evaluation set, and confirm the withholding rate before resuming. Each step has an artefact and the sequence is abandoned if the withholding rate does not fall.

For M6 the sequence is shorter and the check is a single experiment: the fixed-ticket cross-base comparison. If the adapter-on-new-base arm matches the adapter-on-old-base arm, the binding was fine and the diagnosis moves to M1 or M2.

## 13. Transferable Lesson

### 13.1 The reusable rule

**Price the accuracy claim before you accept it.** Any proposal that improves a rate must be converted into the operating unit before it is compared with a fixed cost, and the conversion factor is a small number of quantities the organisation can usually obtain.

The rule has three parts. Find the unit that the objective is denominated in. Differentiate the objective with respect to the proposed metric to get a per-unit price. Compare that price times the claimed improvement against the fixed cost, and compute the break-even in the metric, the volume, and the cost.

Applying it here takes four lines of arithmetic and changes the answer from "ship it, it is better" to "pilot it, it is not better by enough".

### 13.2 What transfers and what does not

The transferable structure is the crossover. Any human-in-the-loop system where the machine's failure is more expensive than its absence has a break-even accuracy below which the system costs money. The formula `p* = (t_rew - t_man)/(t_rew - t_acc)` depends only on three times and transfers directly to review, coding assistance, triage, and any drafting task with a repair step.

The transferable structure also includes the comparator discipline: the fixed-cost option must be compared against the best option that can be shipped without that cost. This is the single most common analytical error in adaptation proposals, and it is invisible when only one option is on the slide.

What does not transfer is any specific number. The $9,027 per point, the 1,602 break-even volume, the 0.382 crossover and the $82,588 fixed cost are all artefacts of this fictional queue. A reader who quotes them into a different setting has learned the format and not the method.

### 13.3 Compact derivation of the decision logic

The logic compresses to six lines that a reader can reproduce from memory. Expected time is a mixture of three times. Its derivative with respect to acceptance is the negative of the cost of being wrong. Value per point is that derivative times volume, days and minute cost. Break-even volume is fixed cost divided by value per draft. The threshold delta is fixed cost divided by value per point. The decision compares an interval on the delta against the threshold.

Everything else in this file is either an input to those six lines or a consequence of them.

### 13.4 Where the reasoning is thinnest

Two places. The independence assumption between blocking and acceptance is unmeasured and, as Section 6.3 shows, is likely to bias the value estimate upward. And the fixed cost of $82,588 is dominated by a coordination estimate that would be produced by a real team's planning process rather than derived, so its precision is illusory.

A reader should therefore treat the break-even volume as a range near 1,600 to 4,300 depending on the delta, and treat the direction of the conclusion as the durable part.

## 14. Connections & Cross-References

### 14.1 Within this unit

- [Why PEFT and the Fine-Tuning Spectrum](../../8.1_Why_PEFT_and_the_Fine_Tuning_Spectrum.md) — the full-versus-parameter-efficient choice and the parameter accounting reused in Section 6.8.
- [LoRA Parameter-Efficient Fine-Tuning](../Cheat_Sheets/01_LoRA_Parameter_Efficient_Finetuning.md) — the rank and target-module arithmetic that produces 36.8M parameters.
- [QLoRA Quantized Base Adapter Fine-Tuning](../Cheat_Sheets/02_QLoRA_Quantized_Base_Adapter_Finetuning.md) — the 4-bit base loading that makes the one-device memory plan feasible.
- [Data Formatting for Training](../../2.4_Data_Formatting_for_Training.md) — the target masking and per-example versus per-token normalisation behind M7.
- [Model Merging and Weight Averaging](../../8.5_Model_Merging_and_Weight_Averaging.md) — the compatibility gate that generalises the composite version tuple.
- [Customize tenant assistants without crossing isolation boundaries](./02_Customize_Tenant_Assistants_Without_Crossing_Isolation_Boundarie.md) — the multi-tenant variant where the same adapter artifact must not carry tenant identity.

### 14.2 Source boundaries

The mapping from source module titles to these lessons is recorded in the unit [INDEX](../INDEX.md). The original lesson bodies were not available for inspection, so no claim is made that this case reproduces their content. The planning row for this topic is `CASE-2026-FT01` in the collection [catalog](./INDEX.md).

No public benchmark, vendor report, or internal measurement is cited anywhere. Where a reader might expect a citation there is an `OPEN` label, and the case is deliberately constructed so that the argument does not depend on any external source.

### Interview pack

#### Q1. Why does the adapter fail the decision when it has the highest acceptance rate?

**Answer.** Because value is measured against the shippable comparator, not against the status quo. Against the grounding change the adapter adds three points, worth 3 × $9,027 = $27,081 per year, and its year-one cost is $82,588. The gap is $55,508.

**Reasoning.** A fixed-cost option must beat the best option that can be shipped without that cost. Comparing against manual work credits the adapter with value the grounding change also delivers.

**Follow-up.** What if the grounding change is not actually shippable? Then the comparator reverts to the current prompt, the delta becomes eight points, value becomes $72,214, and the decision is marginal rather than negative.

**Red flag.** "It has the highest acceptance rate, therefore ship it." This treats a rate as if it were a level and ignores the fixed cost entirely.

**Scoring.** 0 = picks the highest accuracy; 1 = mentions cost vaguely; 2 = computes the three-point value; 3 = also identifies the comparator as the deciding choice.

#### Q2. Recompute the break-even acceptance rate against manual handling and state its units.

**Answer.** `p* = (4.50 − 3.20)/(4.50 − 1.10) = 1.30/3.40 = 0.3824`. It is a proportion of eligible drafts, dimensionless, valid only because `t_rew > t_man`.

**Reasoning.** Setting the mixture equal to the manual time cancels the blocked term because it is identical on both sides, which is why the blocked fraction does not appear in the result.

**Follow-up.** What happens if `t_rew` falls to 2.00 minutes? Then `t_rew < t_man`, `p*` becomes negative, and drafting is worse than manual at every acceptance rate — the formula's validity condition fails rather than the answer changing.

**Red flag.** Including the 0.075 blocked fraction in the algebra and arriving at a different p*, or quoting 38.24% without units.

**Scoring.** 0 = no formula; 1 = states 38% from memory; 2 = derives 0.3824; 3 = also states the `t_rew > t_man` validity condition.

#### Q3. A model improves acceptance from 0.30 to 0.38. Quantify the value.

**Answer.** Zero against manual work, approximately. `E[t]` falls from 3.4590 to 3.1999 minutes, and manual is 3.2000, so the improved model has reached parity and saves 0.0001 minutes per draft.

**Reasoning.** Near the crossover the marginal value of a point is the same $9,027, but the *level* starts below the manual baseline, so the first eight points are consumed restoring parity.

**Follow-up.** What does the same eight points deliver at 0.60 to 0.68? 0.2516 minutes per draft, or $72,214 per year, because the whole gain is above the crossover.

**Red flag.** Reporting "eight points is eight points" and computing $72,214 for both cases.

**Scoring.** 0 = says it is an eight-point gain; 1 = notes the model is still weak; 2 = computes both expectations; 3 = explains that the gain is real but entirely consumed by parity.

#### Q4. Derive the number of scoring slots the pilot needs and explain why 219 is not the answer.

**Answer.** 219 is the unclustered paired requirement; clustering multiplies it by `deff = 1 + 24 × 0.08 = 2.92`, giving 640 pairs, and the full 42-day pilot needs about 10,080 pairs for a ±1.1-point interval.

**Reasoning.** Tickets arrive in topical clusters, so two drafts on the same ticket are correlated and each pair carries less information than an independent pair would.

**Follow-up.** With 1,350 slots available, what ICC is tolerable? `deff = 1350/219 = 6.165`, so `ICC = (6.165−1)/24 = 0.215`.

**Red flag.** Quoting 219 and stopping, which understates the review load by a factor of three and overstates the power by the same factor.

**Scoring.** 0 = no computation; 1 = gives 547 or 219; 2 = applies the design effect; 3 = also inverts the budget to an ICC ceiling.

#### Q5. Why is the 14-day interim read non-decisional?

**Answer.** At 1,350 pairs the half-width on the paired delta is about `1.96 × sqrt(0.18/1350 × 2.92) = 0.0387`, or ±3.9 points. That interval spans both the three-point and the eight-point hypotheses, so no promotion decision is supportable.

**Reasoning.** An interval wider than the gap between competing hypotheses cannot discriminate between them, regardless of the point estimate.

**Follow-up.** What would make an interim read decisive? A control failure, which is a stop condition rather than a promotion condition.

**Red flag.** "The interim looks positive so we can start rolling out to a small cohort" — that is a promotion decision taken on an interval that cannot support one.

**Scoring.** 0 = agrees the interim can guide rollout; 1 = says it is noisy; 2 = computes a half-width; 3 = ties the width to the hypothesis gap and distinguishes stop from promote.

#### Q6. The sponsor reports a 29% handling-time saving. Is the arithmetic wrong?

**Answer.** No. Against manual work the adapter takes the expectation from 3.2000 to 2.2639 minutes, which is a 29.3% reduction, and that is correct. The error is the comparator, not the division.

**Reasoning.** Manual work is not an available alternative once a prompted model exists, so the 29% credits the adapter with gains that the current system already delivers.

**Follow-up.** Which comparison would a reviewer accept? Adapter against the grounding change, giving 4.0%, because both are available to the lead on the same date.

**Red flag.** Asserting the number is fabricated, which is false, or accepting it because it is arithmetically correct.

**Scoring.** 0 = accepts 29%; 1 = says it is misleading with no reason; 2 = verifies the 29.3% and names the comparator; 3 = also gives the correct comparator's figure.

#### Q7. Diagnose: the adapter's acceptance is high in offline evaluation and drops 12 points in shadow mode within two weeks.

**Answer.** Most likely label contamination or serving-drift, distinguished by whether acceptance falls monotonically across weeks with a constant version tuple. Monotone decline with a constant tuple indicates awareness effects; a step change at a build boundary indicates a base-mismatch binding.

**Reasoning.** Offline evaluation uses reviewers on a sample; shadow mode uses agent dispositions, and those two instruments respond differently to the same draft quality.

**Follow-up.** What single experiment separates them? Hold tickets fixed, cross the adapter with both base builds, and compare; if the drop follows the build, it is a binding failure.

**Red flag.** Concluding "distribution shift" without checking whether the version tuple changed.

**Scoring.** 0 = blames the data; 1 = names one cause; 2 = gives the monotone-versus-step discriminator; 3 = also specifies the cross-base experiment and the fallback correction.

#### Q8. Explain why the citation check cannot be a pre-generation filter.

**Answer.** The filter would have to know which clause the model will cite, and that is only known after generation. A pre-filter can constrain which excerpts are available but cannot verify what the model asserts.

**Reasoning.** The check verifies an assertion against an authoritative source, and the assertion does not exist before inference.

**Follow-up.** What does the check do on timeout? Presents the draft with a warning, because dropping converts a latency blip into an invisible outage.

**Red flag.** Proposing to validate the retrieved excerpts and calling that citation validation, which leaves the model free to cite anything.

**Scoring.** 0 = proposes a pre-filter; 1 = says "check the output"; 2 = identifies that the assertion is post-hoc; 3 = also specifies the timeout behaviour and why.

#### Q9. Compute the training cost and explain why it does not justify the programme.

**Answer.** 40,320,000 tokens at 5,200 tokens per second is 7,754 GPU-seconds, or 2.15 GPU-hours, which is $4.52 at $2.10 per hour. It is four dollars against an $82,588 fixed cost, so it justifies nothing.

**Reasoning.** Sunk cost is not a reason to continue. The investment is the evaluation and the twelve annual retraining cycles, not the gradient steps.

**Follow-up.** What if you doubled the epochs? You add $4.52 and add overfitting risk on 48,000 pairs at rank 16; you should prefer more distinct pairs.

**Red flag.** "We have already trained it" as an argument, or quoting the training cost as the programme cost.

**Scoring.** 0 = no numbers; 1 = quotes a GPU-hour figure; 2 = computes the token count and the cost; 3 = also identifies the real cost driver.

#### Q10. The adapter is trained on 48,000 pairs and reaches 0.548% of the base parameters. What can it not learn?

**Answer.** It cannot reliably carry 3,100 policy clauses. Policy must arrive through retrieval at inference, because the parameter budget is 36.8 million values against a corpus that changes at 6.2 clauses per day.

**Reasoning.** Parameter capacity bounds what can be memorised, and a clause corpus with daily changes would need continuous retraining to stay current even if capacity were sufficient.

**Follow-up.** What would a reader expect to see if memorisation were attempted? Drafts citing retired clause identifiers that the retriever did not supply, which is exactly failure mode M3.

**Red flag.** Claiming the adapter "knows the product" without bounding what 0.548% of the weights can hold.

**Scoring.** 0 = says it knows the domain; 1 = mentions retrieval vaguely; 2 = uses the parameter share; 3 = also predicts the observable symptom and links it to M3.

#### Q11. What single change to the input parameters would reverse the recommendation, and by how much?

**Answer.** Raising rewrite time from 4.50 to 6.00 minutes. Per-point value rises to `0.925 × 4.90 × 1400 × 300 × 0.68333 = $14,398`, the threshold falls to `82,588/14,398 = 5.74` points, and a five-point delta now clears it.

**Reasoning.** Rewrite time is the cost of being wrong about a draft, so it enters every value term linearly and enters the crossover through `t_rew − t_man`.

**Follow-up.** How would you measure it? Time-study a stratified sample of edited drafts with a stopwatch rather than relying on ticket timestamps, which include waiting.

**Red flag.** Naming acceptance rate as the reversing input, which moves both the numerator and the threshold.

**Scoring.** 0 = no parameter; 1 = names rewrite time qualitatively; 2 = computes the new per-point value; 3 = also derives the new threshold and checks the flip.

#### Q12. Why is the blocked fraction identical across all non-manual arms, and what breaks if it is not?

**Answer.** A blocked draft is one where policy coverage fails, so no drafting variant helps and the same 0.24 minutes appears in every arm. If blocking is correlated with acceptance, the mixture overstates the gain because the tickets a model is worst at are the ones already excluded from `p`.

**Reasoning.** The mixture assumes independence between `b` and `p`; negative correlation turns the linear per-point rule into an upper bound.

**Follow-up.** How would you detect the correlation? Cross-tabulate blocked tickets against whether either arm produced an accepted draft; a non-zero concentration is the signal.

**Red flag.** Treating 0.075 as a constant that can be subtracted from the model's gain while also being improved by the model.

**Scoring.** 0 = ignores blocking; 1 = notes it exists; 2 = shows it cancels in the crossover; 3 = also identifies the independence assumption and its direction of bias.

#### Q13. What makes the pilot worth $11,292 when the answer might be "no"?

**Answer.** It narrows the interval on the delta to ±1.1 points, which is narrower than the 9.15-point threshold band, so it resolves a decision worth $82,588 per year. The pilot costs 13.7% of the commitment it governs.

**Reasoning.** Information has value when it changes an action and when the action is expensive relative to the information cost. Both conditions hold here.

**Follow-up.** When would the pilot not be worth it? If the threshold were far from any plausible delta, for example a 30-point threshold, the answer would already be clear and the pilot would be theatre.

**Red flag.** Running the pilot to "learn more" without stating which decision it informs.

**Scoring.** 0 = says learning is good; 1 = notes the cost; 2 = ties the interval to the threshold; 3 = also states the condition under which the pilot would be wasteful.

#### Q14. Distinguish the three leakages this design controls and say what each one would look like if missed.

**Answer.** Temporal leakage shows as an implausibly high offline acceptance that collapses in shadow; near-duplicate leakage shows as a high acceptance concentrated on tickets with high n-gram overlap to training; policy-version leakage shows as withheld drafts citing retired identifiers.

**Reasoning.** Each leakage inflates a different part of the evaluation, so each has a distinct observable signature rather than a shared "too good" symptom.

**Follow-up.** Which is hardest to detect after the fact? Near-duplicate leakage, because overlap statistics are not usually retained and cannot be reconstructed once the corpus is rebuilt.

**Red flag.** Listing "data leakage" as one item and proposing one control, which leaves two channels open.

**Scoring.** 0 = one generic control; 1 = names two; 2 = names three with separate controls; 3 = also gives each an observable signature and ranks detection difficulty.

## 15. Quick Reference Card

| Field | Case reference |
|---|---|
| **Decision** | Ship the grounding change; authorize a 42-day shadow adapter pilot at 240 drafts/day; refuse adapter rollout at current volume. |
| **Owner and grain** | Support operations lead; one programme decision with a pre-registered promotion rule, cut-off 2026-03-02T00:00Z. |
| **Baseline and comparator** | Prompted model at 0.60 acceptance and 2.5155 min/draft; the comparator is the grounding change at 0.65, never manual work. |
| **Core mixture** | `E[t] = 0.075×3.20 + 0.925×[p×1.10 + (1−p)×4.50]`, giving 2.5155 / 2.3583 / 2.2639 minutes. |
| **Crossover** | `p* = (4.50−3.20)/(4.50−1.10) = 0.3824`; below it, drafting costs more than writing from scratch. |
| **Value and threshold** | One point is worth `0.925×3.40×1400×300×0.68333 = $9,027` per year; the adapter must add 9.15 points to repay $82,588. |
| **Break-even volume** | 1,602 drafts/day at an eight-point delta; 4,270 at a three-point delta; current volume is 1,400. |
| **Promotion rule** | Promote only if the lower 95% bound on the delta exceeds 0.0915; reject if the upper bound is below it; otherwise stay in pilot. |
| **Failure checks** | Distinguish monotone label contamination from step-change base mismatch; cross-tabulate blocking against acceptance; never report against manual work. |
| **Evidence status** | All inputs `ILLUSTRATIVE`; arithmetic `DERIVED` and rechecked; no run executed; every evaluation row is `not executed`. |
