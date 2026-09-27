# CS-14 — The Alignment Map: RLHF, PPO, DPO, ORPO

| Field | Value |
|---|---|
| **Module** | Alignment / Preference Optimisation (orientation module for the whole track) |
| **Source video(s)** | LLM Fine-Tuning 16: Preference Alignment & Preference Training in LLMs with RLHF, RLAIF, DPO, LoRA |
| **Transcript file(s)** | `LLM_Fine-Tuning_16_Preference_Alignment_Preference_Training_in_LLMs_with_RLHF_RL.txt` |
| **Companion code** | `LLM Fine-Tuning-16-Preference-based-training\Preference_Aligned_Training_DPO_final.ipynb`, `pharma_preference_data.jsonl`, `pharma_preference_data.csv` |
| **Prerequisites** | CS-01 (lifecycle), CS-13 (SFT), CS-23 (LoRA/QLoRA — you must know what a delta patch is before §6) |
| **Difficulty** | Intermediate conceptually, Advanced practically (this is a *map*; §4.6 is the territory — the planned CS-24–CS-27 modules were never written) |
| **Hands-on required** | Yes — the LoRA-merge bug in §6.5 is the single most common production error in this track |
| **Estimated study time** | 6h theory + 5h practical |

> **What this module is.** This is the **orientation module** for the alignment track. §4.6 is also the deep dive: the planned follow-up modules — CS-24 on RL fundamentals + PPO, CS-25 on DPO, CS-26 on GRPO, CS-27 on ORPO — were never written, so their material lives in the §4.6 profiles here. This module gives you the map: why alignment exists, what the canonical pipeline is, what every method in the landscape actually does, the memory arithmetic that decides between them, and the decision tree. Read this first; then go to the §4.6 profile for whichever method the decision tree selects.
>
> **What this module is not.** It is not a derivation course. The instructor explicitly says he will not do the full derivation here — *"I'm not going to show you the complete mathematical intuition along with the derivation and all. I will just focus on the formula and we'll discuss about that formula"* [3:05]. We honour that scope, and mark derivations as belonging to CS-24/25.

---

## 0. Executive Summary

- **Preference alignment is the third and final stage of the canonical LLM training pipeline.** The instructor states it flatly: *"these are three main stage of any LLM training"* [6:34] — unsupervised/supervised pretraining → supervised fine-tuning (SFT) → preference-based alignment. You are almost never pretraining; you usually start from a released checkpoint.
- **The definition, verbatim:** *"to align or train a model with a human expectation or with a human preferred data is called preference alignment or preference training"* [7:07]. It was introduced by OpenAI in the **InstructGPT** paper, *Training language models to follow instructions with human feedback* [7:20], [20:12].
- **Why it exists, in one example.** An SFT model asked *"how do I lose weight faster?"* answers *"just stop eating so much."* That answer is **factually adequate and operationally useless** — the instructor's own words: *"the answer is correct. But it is rude. It is dismissive and there is no explanation"* [8:22]. Alignment is what turns a correct answer into a **helpful, harmless, honest (HHH)** one.
- **The HHH target is not accuracy.** *"After the preference based training model does not just produce accurate answer it behave according to the user preference"* [10:54]. Accuracy is an SFT property. Preference alignment optimises *which of several accurate answers* you get.
- **The data format is always the same three fields**: `{prompt, chosen, rejected}` [15:11]. Every preference dataset in the ecosystem — Anthropic's `HH-RLHF`, Argilla's `ultrafeedback-binarized-preferences-cleaned`, Zilly's `math-step-dpo-10k` — is a re-skin of this triple. The instructor walks all three on screen [15:30]–[16:45].
- **The method landscape is bigger than RLHF.** The video names RLHF, RLAIF, DPO plus the algorithms PPO, IPO and KTO [17:08]–[17:58]. The full production landscape in 2026 adds ORPO, SimPO, cDPO, SLiC-HF, GRPO, rejection sampling / best-of-n / RAFT, and SPIN. §4.6 profiles all of them.
- **The single most decisive number is memory, not quality.** PPO needs **4 models resident** (policy, reference, reward, value) [17:58]–[18:07]. DPO needs **2**. ORPO needs **1**. For a 7B model in full fine-tuning that is the difference between ~154 GB and ~30 GB of VRAM. This is why the industry moved, and §4.7 does the arithmetic in GB.
- **DPO is a supervised classifier in disguise.** *"This is not a reinforcement based training… This is a simple supervised learning"* [19:37]–[19:47]. That single property — no rollouts, no reward model, no value model, no sampling loop — is why DPO became the default and why the instructor calls RLHF/RLAIF tooling *"deprecated"* for his purposes [22:19].
- **The `beta` (β) knob is the alignment-strength dial.** *"if the beta is high means we are enforcing model to align more to the chosen… if beta is low means the model is more aligning to the rejected"* [26:54]–[27:09]. He gives the typical range as **0.1 to 0.5**, and the notebook uses **β = 0.1** with `loss_type="sigmoid"`.
- **The decision tree splits on one question first: is the reward verifiable?** Verifiable domains (math, code, logic) → **GRPO / RLVR**, because you can check the answer and need no reward model at all. Subjective domains (tone, style, helpfulness, safety) → **DPO / ORPO**, because you cannot write a checker and must learn the preference. This is the single most important modern routing rule and it is a `> **Beyond the video:**` addition — the video predates it.
- **Alignment has a tax and a failure mode.** The tax: alignment degrades some capabilities (InstructGPT's "alignment tax", mitigated with **PPO-ptx**). The failure mode: **reward hacking / Goodhart's law** — the policy maximises the *proxy* reward, not the true objective, producing length bias, sycophancy, and reward-model over-optimisation. §4.8 and §4.9.
- **The production rule this module exists to deliver:** *try DPO or ORPO first; reach for PPO only when you have a reward signal that cannot be expressed as pairs, and for GRPO only when the reward is programmatically checkable.* The video's own conclusion agrees by omission — the instructor never trains PPO on screen, he trains DPO, and calls it *"very simpler"* [22:21].

---

## 1. The Problem This Solves

### 1.1 What breaks in the real world without this

An SFT model is a **next-token predictor shaped by demonstrations**. It learned *what answers look like*. It did not learn *which of several plausible answers a human would prefer*. Those are different objectives, and the gap shows up as five production failure modes:

| Failure | What it looks like | Root cause |
|---|---|---|
| **The correct-but-rude assistant** | "How do I lose weight?" → "Just stop eating so much." [8:15] | SFT never saw a preference signal; it saw demonstrations and generalised to the shortest adequate answer |
| **The unsafe-compliance assistant** | "How do I lose 10 kg in one week?" → "stop eating, drink only water and take fat burner pills" [12:00]–[12:08] | SFT mimics the *form* of an answer; it has no notion of harm |
| **The verbose waffler** | Every answer is 6 paragraphs when 2 lines were asked for | Length is a learned prior from demonstration data; without a preference signal nothing pushes back |
| **The over-refuser** | "What's a good knife for cooking?" → "I can't help with weapons." | Safety data injected without a helpfulness counterweight — the classic HHH trade-off failure |
| **The format drifter** | Passes eval, then drifts out of JSON after two weeks of prompt churn | Behaviour was prompt-conditioned, not weight-conditioned. Alignment *bakes behaviour into weights* |

Every one of these is invisible to a loss curve. Cross-entropy on demonstrations goes down smoothly while the product experience stays bad, because **cross-entropy does not measure preference**. That is the whole reason this stage exists.

### 1.2 The state of the art before preference alignment

Before InstructGPT (March 2022), if you wanted a model that behaved well you had exactly two levers:

1. **Prompt harder.** Few-shot exemplars, system prompts, chain-of-thought. Works, but costs tokens on every call and leaks into context budget; and it cannot fix refusal calibration or tone at the tails.
2. **SFT on more/better demonstrations.** Works, but the ceiling is the annotator's *typical* answer. Demonstration data is a single point per prompt; preferences are an *ordering* over many points, which is a fundamentally richer and cheaper signal. This is the key economic insight: **ranking k answers gives you O(k log k)-ish comparison information in one annotation pass**, versus writing one answer from scratch.

The instructor's framing of the transition [17:08]–[18:07]: RLHF's *"core idea [was] to train a reward model using the human feedback"*, with data being *"ranked responses… ranked by the human"*, demonstrated in the **InstructGPT** paper, using **PPO** — *"even OpenAI is also revealed that they have used this particular technique initially for the reward modeling"*.

The second transition, and the one that matters commercially, is **scalability of the label**. The instructor's objection to human-only feedback is arithmetic, not philosophy: *"human generated feedback was not very scalable idea means how many feedback can be annotated by the human right maybe one lakh two lakh right but if we have to make it scalable then how we can do it so there they have introduced this RL aif where human is not giving a preference actually we are taking that particular preference from the AI itself"* [18:56]–[19:15]. **One lakh = 100,000; two lakh = 200,000.** That is the human-annotation ceiling he names, and RLAIF is the response.

### 1.3 The naive approach, and precisely why it fails

The naive approach is **"just do more SFT on the good answers."** It fails for four distinct reasons, and it is worth being precise because "why not just SFT on the chosen responses?" is a real interview question:

| Reason | Mechanism |
|---|---|
| **You throw away the negative signal** | SFT on `chosen` only teaches "produce this". It never teaches "do not produce *that*". The rejected response is a free contrastive example that gets deleted. |
| **You teach the model to imitate, not to discriminate** | Maximum-likelihood on a single target has no notion of margin. The model can end up assigning high probability to both the chosen and the rejected answer, because both are fluent. |
| **You cannot express relative quality** | Annotations say `chosen > rejected`, not `chosen = 1.0, rejected = 0.0`. SFT flattens a ranking into an absolute target and loses the ordering. |
| **You re-introduce the ceiling** | Each prompt contributes one training target. Preference data contributes a comparison; a prompt with 4 ranked answers contributes 6 pairwise constraints (or 3 if you only use adjacent pairs). |

> **Beyond the video:** there is a fifth, more subtle reason. SFT on chosen-only data causes **likelihood displacement** — the chosen response's probability rises while the *rejected* response's probability falls, but so do the probabilities of unrelated outputs, and the model's overall distribution sharpens. DPO's whole design goal is to fix the relative-probability problem by making the *difference* of log-probabilities the trained quantity. That is exactly what the instructor is pointing at when he says the DPO loss compares *"how much more does your model prefer the chosen answer compared to the reference model"* [25:59].

---

## 2. First-Principles Mental Model

### 2.1 The analogy: a talented intern who has read everything

Imagine a brilliant intern who has read the entire internet (**pretraining**), then spent three weeks shadowing a senior analyst and copying exactly what the analyst writes down (**SFT**), and has never once been told *"that answer was good, that one was bad"* (**no preference stage**).

- The intern can produce a correct, well-formatted answer.
- The intern cannot tell you whether a *blunter* correct answer or a *gentler* correct answer is the one your customer wants.
- The intern will reproduce the analyst's habits including the analyst's bad ones — the terse shortcuts, the jargon, the occasional unsafe guess — because imitation copies everything, including the flaws.
- If you then sit the intern down and say *"of these two answers, this one"* — a few thousand times — the intern recalibrates. That is preference alignment. You did not teach new facts. You taught a **policy over behaviours**.

### 2.2 The same thing stated mechanically

Formally, you have a policy π_θ that maps a prompt x to a distribution over responses y. Pretraining produces π that minimises next-token cross-entropy on web text. SFT produces π^SFT that minimises cross-entropy on (x, y*) demonstration pairs. Preference alignment produces π^aligned that maximises

```
E_{x ~ D, y ~ π(·|x)} [ r(x, y) ]  −  β · KL( π(·|x) ‖ π^SFT(·|x) )
```

where `r` is a reward and `π^SFT` is the frozen reference. Read that expression as two sentences: **"raise the reward"** and **"do not drift far from what you already were."** Every method in §4.6 is a different way of optimising that same objective — or a simplification of it that removes one of the terms.

### 2.3 Where the analogy breaks

1. **The intern generalises; the model interpolates.** A human intern knows *why* an answer was preferred and applies the principle elsewhere. A language model learns a statistical direction in activation space. It will happily apply "be gentle" to a context where bluntness was the point.
2. **The intern is one person; the model is trained on a committee.** If 5 annotators disagree, the intern synthesises. The model averages the disagreement into a policy that satisfies nobody. This is why **inter-annotator agreement** (§4.3) is a leading indicator of alignment quality, not a nicety.
3. **The intern cannot be reward-hacked by a proxy, because the feedback is the real thing.** In RLHF the reward model is a *learned proxy* for human judgement. The policy is optimised against the proxy. Given enough optimisation pressure the policy finds the proxy's bugs — this is Goodhart's law and it has no analogue in human mentorship. §4.9.
4. **The KL anchor has no human analogue.** You *want* the intern to change their habits. You generally do *not* want the LM to move far from π^SFT, because away from π^SFT the model is off-distribution and degrades. The reference model is a leash, and the leash is a hyperparameter (β).

### 2.4 Defining HHH precisely

"Helpful, Harmless, Honest" is Anthropic's framing, popularised via RLHF work and referenced by the instructor as *"safe, helpful and the honest model"* [7:33]. It is used loosely in industry and precisely in the literature. The precise reading:

| Property | Precise definition | Tension with the others | How you measure it |
|---|---|---|---|
| **Helpful** | The model does what the user actually intended, at the right level of detail, in the requested format, without requiring re-prompting. | Maximising helpfulness maximises compliance, which conflicts with harmlessness. | Task success rate, human preference win-rate, instruction-following evals |
| **Harmless** | The model refuses to produce content that causes real-world damage, and does not acquire power-seeking / deceptive dispositions. | Refusal over-firing destroys helpfulness — the "over-refuser" failure in §1.1. | Red-team refusal calibration, false-refusal rate on benign prompts, safety evals |
| **Honest** | The model states what it believes, does not fabricate, calibrates uncertainty, and does not strategically mislead. | Optimising for "sounding confident" via a preference signal directly attacks honesty. | TruthfulQA, hallucination rate, calibration curves, sycophancy evals |

> **Beyond the video:** the three properties **cannot be maximised independently** — they are a Pareto frontier, and the point on the frontier is a *product decision*, not a training decision. This is the correct answer to "how do you balance HHH?" in an interview: you do not balance it in the loss function, you set the frontier position by choosing your preference data mix and your β, then you document the choice. A helpfulness-only preference set produces a sycophant. A harmlessness-only set produces a brick.

---

## 3. Core Concepts — Exhaustive Glossary

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| **Alignment** | Making a model's behaviour match human intent and values, not merely its text match human text. | The whole point of stage 3. | Confused with "instruction following", which is SFT. |
| **Preference alignment / preference training** | The instructor's term for stage 3: *"to align or train a model with a human expectation or with a human preferred data"* [7:07]. | The module's subject. | Confused with RLHF specifically; RLHF is one method of doing it. |
| **HHH** | Helpful, Harmless, Honest. The alignment target. | Gives you the eval axes. | Treated as one score; it is three axes in tension. |
| **Pretraining / self-supervised pretraining** | Next-token prediction on raw text. Produces the foundation model. *"the first step of the LLM training process"* [5:41]. | You rarely do it; you download the result. | Called "unsupervised" by the instructor — modern usage is *self-supervised*. |
| **SFT (supervised fine-tuning)** | Fine-tuning on (input → output) demonstration pairs. *"in the instruction finetuning we'll be having three column… instruction, input, response"* [14:19]. | Stage 2. The reference model for stage 3. | Confused with "instruction tuning" — same thing here. |
| **Non-instruction fine-tuning** | SFT on plain domain text (e.g. PDFs) to learn vocabulary/domain language [30:15]. | The video's stage 1.5; domain adaptation (CS-12). | Confused with continued pretraining; it uses the same objective but far less data. |
| **Instruction model / SFT model / policy prior** | The stage-2 output. Doubles as π_ref in DPO. | It is the KL anchor and the initialisation. | People think you can skip it — you cannot (§4.2). |
| **Preference model / aligned model / policy π_θ** | The stage-3 output. | What you ship. | Confused with **reward model**, which is a different artifact entirely. |
| **Reward model (RM)** | A model trained to score (prompt, response) pairs, standing in for human judgement. Usually initialised from the SFT model with a scalar head. | Needed by PPO/RLHF. *Not* needed by DPO/ORPO/SimPO. | The #1 confusion in this module: *"DPO trains a reward model implicitly"* — no, it re-parameterises the policy so no explicit RM exists. |
| **Value model / critic** | A model predicting expected future reward from a state, used to compute advantage in PPO. | The 4th model in PPO, and the usual reason PPO is painful. | Confused with the reward model. RM scores an outcome; value predicts a return. |
| **Reference model π_ref** | The frozen SFT model used as the KL anchor. | Present in PPO, DPO, GRPO. Absent in ORPO. | People think DPO's `ref_model=None` means "no reference" — it means "use the same weights with the adapter disabled". |
| **Policy π_θ** | The model being trained. | It is the only thing you ship. | — |
| **Rollout / generation** | Sampling responses y ~ π_θ(·\|x) during training. | PPO and GRPO need it; DPO/ORPO do not. Rollouts are most of their cost. | — |
| **Reward hacking / Goodhart's law** | The policy maximises the proxy reward r rather than the true objective. Manifests as length bias, sycophancy, formatting exploits. | The dominant failure mode of RLHF. | Confused with overfitting; it happens *on the training distribution* with a perfectly-trained RM. |
| **KL divergence (KL)** | A measure of how far π_θ has moved from π_ref. The regulariser in the alignment objective. | Controls the exploration/safety trade-off. | Direction confusion — see §4.5. |
| **β (beta)** | The KL coefficient. High β → stay near π_ref. Low β → free to chase reward. | *"typical value of the beta between 0.1 to 0.5"* [27:11]. | In ORPO, λ plays the analogous role but weights an odds-ratio term, not a KL. |
| **Chosen / rejected (y+, y−)** | The preferred and dispreferred response in a pair. | The data format [15:16]. | Confused with "correct/incorrect" — preferences are about *quality*, not truth. |
| **Prompt (x)** | The conditioning input. | Shared by both members of a pair. | — |
| **Pairwise preference data** | `{prompt, chosen, rejected}`. | The universal format. | — |
| **Pointwise / unary preference data** | A single response with a binary desirable/undesirable label (KTO's format). | Lets you use data that has no pair. | Rarely available by accident; KTO exists to exploit it. |
| **Ranked / k-wise preference data** | k responses ordered by quality (the InstructGPT collection format). | Richer than pairs; you can decompose into pairs. | Requires more annotator effort per prompt. |
| **Inter-annotator agreement (IAA)** | The degree to which independent annotators give the same preference. Measured with Cohen's κ (2 raters) or Krippendorff's α (n raters). | Low IAA puts a ceiling on achievable alignment; you cannot learn a signal that is not there. | Ignored in most tutorials, which is why their aligned models are mediocre. |
| **Bradley–Terry (BT) model** | The standard model: P(y+ ≻ y−) = σ(r(y+) − r(y−)). | The RM loss and (via re-parameterisation) the DPO loss both descend from it. | Confused with Elo, which is a *ranking* system derived from BT. |
| **Reward margin** | r(y+) − r(y−). The quantity BT pushes positive. | DPO's implicit reward margin is what you can log. | — |
| **RLHF** | Reinforcement Learning from Human Feedback. Reward model + PPO + KL. Introduced for LLMs in InstructGPT. | The original method; the 4-model problem. | Used as a synonym for "alignment" — it is one method. |
| **RLAIF** | Reinforcement Learning from AI Feedback. Same loop, labels from an LLM. Introduced by Anthropic [18:46]. | Scalability answer to the annotation ceiling [18:56]. | Confused with Constitutional AI, which is a specific RLAIF recipe. |
| **PPO** | Proximal Policy Optimization. The RL algorithm used inside RLHF; also used alone for reward modelling in early OpenAI work [18:01]. | The standard RLHF optimiser. | Confused with RLHF itself — RLHF is the pipeline, PPO is the optimiser. |
| **DPO** | Direct Preference Optimization. Closed-form re-parameterisation that turns the RLHF objective into a supervised classification loss on pairs. Stanford, 2023/2024 [20:46]. | The default method. No RM, no rollouts. | *"I have to train a reward model first"* — you do not. |
| **IPO** | Identity Preference Optimization. DPO with a squared-loss objective that fixes DPO's overfitting-on-deterministic-preferences pathology. | Good when preferences are near-deterministic or data is small. | Confused with DPO's `loss_type="ipo"`, which it is — same family. |
| **cDPO** | Conservative DPO. DPO with **label smoothing** to tolerate annotator noise. | Use when IAA is low. | — |
| **KTO** | Kahneman–Tversky Optimization. Pointwise, prospect-theory-inspired loss using only a binary "good/bad" label. | Use when you have unpaired thumbs-up/thumbs-down data. | Instructor's pronunciation: *"kanman tverki optimization"* [17:44]. |
| **SLiC-HF** | Sequence Likelihood Calibration. A contrastive/ranking loss on sequence log-probs with an explicit margin, plus a cross-entropy term on chosen. | Older than DPO, still competitive; very stable. | — |
| **SimPO** | Simple Preference Optimization. DPO **without a reference model**, using length-normalised average log-prob as the implicit reward plus a target margin γ. | Removes the reference model and the length bias. | — |
| **ORPO** | Odds Ratio Preference Optimization. Single-stage: SFT loss + an odds-ratio preference term, **no reference model**. | One stage, one model. Cheapest credible option. | Confused with "DPO without β". |
| **GRPO** | Group Relative Policy Optimization. PPO **without a value model** — the baseline is the mean reward of a sampled group. The DeepSeek-R1 recipe. | The method for verifiable rewards. | Confused with "PPO with fewer models"; the advantage estimator is genuinely different. |
| **RLVR** | Reinforcement Learning with Verifiable Rewards. GRPO (or PPO) where the reward is a program (unit tests, math answer check, schema validator). | The reasoning-model recipe. Zero reward-model cost. | — |
| **Constitutional AI (CAI)** | Anthropic's RLAIF recipe: a written constitution, model self-critique and revision to generate preference data, then preference training. | Reduces human labelling and makes values explicit and auditable. | Confused with RLAIF generally; CAI is one RLAIF design. |
| **Rejection sampling / best-of-n** | Sample n responses from π, keep the best by some scorer, SFT on the kept ones (a.k.a. RAFT). | Cheapest possible "alignment"; no RL, no pairs. | Called "RS-DPO" when the kept/rejected become pairs. |
| **RAFT** | Retrieval-Augmented Fine-Tuning — often conflated with rejection-sampling FT. | — | Name collision; check which the interviewer means. |
| **SPIN** | Self-Play fine-tunINg. The model's own previous-iteration outputs are the "rejected" and human data is the "chosen"; iterate. | Uses no new labels; can exceed the SFT data ceiling. | — |
| **Alignment tax** | Capability degradation caused by alignment. Named in InstructGPT. | Explains why labs report MMLU drops post-RLHF. | Confused with catastrophic forgetting. |
| **PPO-ptx** | PPO with an added pretraining-gradient term (the "ptx" term) to pay back the alignment tax. | The InstructGPT mitigation. | — |
| **Win rate** | Fraction of comparisons where your model is preferred over a baseline. | The primary alignment metric. | Baseline-relative; a 60% win over a weak baseline is meaningless. |
| **MT-Bench** | Multi-turn benchmark graded by GPT-4 on a 1–10 scale. | Legacy but still cited. | Saturated at the top. |
| **AlpacaEval 2 / LC win rate** | LLM-judged win rate vs a reference; **LC** = length-controlled, regressing out the length advantage. | The LC variant is the honest number. | Reporting raw AlpacaEval win rate flatters verbose models by ~10 points. |
| **Arena Elo** | Human pairwise votes aggregated into an Elo rating (LMSYS Chatbot Arena). | Closest thing to ground truth. | Elo differences are not linear in quality. |
| **RM score** | The reward model's score on held-out prompts. | **The metric being gamed.** | Never report it as a quality metric. |
| **Goodhart's law** | "When a measure becomes a target, it ceases to be a good measure." | The theoretical statement of reward hacking. | — |
| **Over-optimisation** | The empirical phenomenon that RM score keeps rising while true quality falls. Quantified by Gao et al. (2023) and in Anthropic's scaling-law work. | Gives you the early-stopping rule. | Confused with overfitting. |
| **Length bias** | The policy learns longer = preferred because annotators and LLM judges favour length. | The most common concrete reward hack. | Also present in DPO without length normalisation. |
| **Sycophancy** | The policy learns to agree with the user regardless of truth. | The honesty-destroying hack. | — |
| **Delta patch (ΔW)** | A LoRA adapter; *"it's just a delta patch… delta patch cannot be stacked they must be merged before the next training"* [41:52]–[42:06]. | Governs §6.5's merge discipline. | Confused with a full layer. |
| **merge_and_unload()** | PEFT call that folds the adapter into the base weights and removes the adapter wrapper. | The correct step between stages. | People call `get_peft_model` again without merging — the video's central bug. |
| **`get_peft_model` vs `PeftModel.from_pretrained`** | Former *creates* a new LoRA; latter *loads* an existing one [44:47]–[45:38]. | The distinction the instructor spends 10 minutes on. | — |
| **TRL** | Hugging Face's Transformer Reinforcement Learning library; hosts `DPOTrainer`, `DPOConfig`, `RewardTrainer`, `PPOTrainer`, `GRPOTrainer`, `ORPOTrainer`, `KTOTrainer`. | The implementation surface for everything in this module. | — |
| **Processing class** | `DPOTrainer(processing_class=tokenizer)` — TRL ≥0.13 renamed the `tokenizer` argument. | The video's notebook uses the new name. | Old tutorials pass `tokenizer=`; it still works but warns. |
| **KL penalty direction** | KL(π_θ ‖ π_ref), the forward KL — penalising mass the policy puts where the reference does not. See §4.5 for why it is not the reverse. | Getting this backwards is a classic interview trap. | — |

---

## 4. Deep Dive — How It Actually Works

### 4.1 The three-stage pipeline, and exactly what each stage contributes

The instructor's own diagram, drawn from his screen narration [29:55]–[31:00]:

```
                     ┌──────────────────────────────────────────┐
Stage 0  BASE MODEL  │ "TinyLlama/TinyLlama-1.1B-intermediate-  │   Next-token prediction
         (foundation)│  step-1431k-3T"  — downloaded, not ours  │   on web-scale text.
                     └──────────────────┬───────────────────────┘
                                        │  fine-tune on PDF/domain text
                                        │  ("so that my model can understand the
                                        │   domain specific language, domain
                                        │   specific vocabulary" [30:22])
                     ┌──────────────────▼───────────────────────┐
Stage 1  NON-        │  checkpoint-5  (non-instruction model)   │   Domain vocabulary,
         INSTRUCTION │                                          │   domain style.
                     └──────────────────┬───────────────────────┘
                                        │  fine-tune on (input → output) instructions
                                        │  ("I want to teach my model proper
                                        │   conversation" [30:30])
                     ┌──────────────────▼───────────────────────┐
Stage 2  INSTRUCTION │  checkpoint-3  (instruction model)       │   Follows instructions,
         (SFT)       │  == π_ref for stage 3                    │   correct format.
                     └──────────────────┬───────────────────────┘
                                        │  DPO on {prompt, chosen, rejected}
                                        │  ("I finetune my model on human
                                        │   feedback" [30:49])
                     ┌──────────────────▼───────────────────────┐
Stage 3  PREFERENCE  │  tinyllama-preference-alignment/         │   HHH: tone, safety,
         (ALIGNMENT) │  checkpoint-1                            │   refusal calibration,
                     └──────────────────────────────────────────┘   verbosity control.
```

Each stage, stated as input → operation → output → failure mode:

| Stage | Input | Operation | Output | Failure mode if skipped or botched |
|---|---|---|---|---|
| **0. Pretrain** | Trillions of tokens of raw text | Next-token cross-entropy | Foundation model | You cannot skip it; you download it. Starting from a *base* model and going straight to alignment is §4.2. |
| **1. Domain / non-instruction FT** | Domain corpus (here, pharma PDFs) as plain text | Next-token cross-entropy on domain text | Domain-adapted model that knows the vocabulary | Skip it and the model has the domain *concepts* but not the *register*; it will use lay phrasing for technical content. |
| **2. SFT / instruction tuning** | (instruction, input, response) triples | Masked cross-entropy on response tokens | Instruction-following model | Skip it and alignment has nothing to preserve — see §4.2. |
| **3. Preference alignment** | `{prompt, chosen, rejected}` | DPO / ORPO / PPO / GRPO | Aligned model | Skip it and you keep the "correct but rude" model from §1.1. |

**Why the order is not negotiable.** Stage 3's objective contains a **KL term against π_ref**. If π_ref is a raw base model, the KL anchor pins you to a model that does not follow instructions, and the reward term pulls you toward helpfulness. The two terms fight, the policy lands in a region neither term wants, and you get a model that is *marginally* better at tone while being *substantially* worse at instruction following. The instructor's example output proves the same point from the other side: his stage-2 model's answer to the Metformin question is already *"aligned and it is with respect to my question"* [34:30] — the alignment stage refines an already-competent model.

### 4.2 Why alignment on a raw base model does not work

This deserves its own treatment because it is a favourite interview probe.

**Claim:** running DPO/PPO on a *pretrained* checkpoint with no SFT stage produces a model that is worse than the SFT baseline on essentially every axis.

**Mechanism, in four steps:**

1. **The preference data is in instruction format; the base model has never seen instruction format.** The prompt is a question. The base model's continuation distribution is "web text that follows this string", so `chosen` and `rejected` are both *far* off-distribution. DPO's gradient is a function of `log π_θ(y|x) − log π_ref(y|x)`; for a base model, both terms are tiny and noisy, and the *ratio* is dominated by noise.
2. **The KL anchor is meaningless.** KL(π_θ ‖ π_ref) with a base π_ref does not mean "stay as a good assistant". It means "stay as an autocompleter". You are penalising the model *for being an assistant*.
3. **The likelihood-displacement effect is unopposed.** Pushing up `p(chosen)` and down `p(rejected)` on a base model redistributes probability mass across *all* continuations, because the model has no representation of "answer-like text" as a coherent region. You get degraded general text.
4. **There is no reward signal to learn from, empirically.** The preference direction (helpful ≻ unhelpful) is learnable only if the model can already produce both. A base model asked a question produces neither a good nor a bad answer; it produces a *continuation*. Both members of the pair are equally "impossible", so the margin `r(y+) − r(y−)` carries little information.

> **Beyond the video:** the standard empirical demonstration is the "DPO from base" ablation. On a base Llama-class model, DPO reaches a lower training loss than DPO-from-SFT but loses on every held-out win rate — the model learns the *format* of answers and none of the *content*. The rule of thumb used in practice: **the SFT stage is what makes the preference stage cheap.** A good SFT model needs a few thousand preference pairs; a base model needs millions and still underperforms. This is also why "DPO directly on the instruct checkpoint you downloaded" is the standard single-GPU recipe — you are borrowing someone else's SFT stage.

### 4.3 The preference data format, in depth

#### 4.3.1 The schema

The instructor is explicit and repeats it: *"inside the preference based training guys you will find out three column three main column right uh one is a prompt second is a chooser and the third is a reject"* [15:11]–[15:18]. He then clarifies: *"this two is very important… choose right which one is going to be select and which one is going to be rejected"* [15:18]–[15:28].

```json
{
  "prompt":   "Explain the mechanism of action of Metformin in simple scientific terms.",
  "chosen":   "Metformin activates AMPK, which helps cells use glucose more efficiently, ...",
  "rejected": "Metformin only lowers blood sugar by stopping sugar absorption from food and ..."
}
```

**The repo's actual data.** `pharma_preference_data.jsonl` contains exactly **5 rows**, and every row follows the schema above. Three of the five, quoted verbatim:

```json
{"prompt": "Explain the mechanism of action of Metformin in simple scientific terms.", "chosen": "Metformin activates AMPK, which helps cells use glucose more efficiently, reduces glucose production by the liver, and improves insulin sensitivity without increasing insulin levels.", "rejected": "Metformin only lowers blood sugar by stopping sugar absorption from food and does not affect other organs."}
{"prompt": "Compare the lipid-lowering effects of Atorvastatin and Ezetimibe when used together versus alone.", "chosen": "When used together, Atorvastatin and Ezetimibe provide an additive effect — Atorvastatin reduces cholesterol synthesis in the liver, while Ezetimibe blocks intestinal cholesterol absorption, leading to a greater LDL reduction than either drug alone.", "rejected": "Both drugs work the same way, so using them together doesn't provide any additional cholesterol reduction."}
{"prompt": "What are some non-glycemic benefits of Metformin according to recent research?", "chosen": "Beyond blood sugar control, Metformin has shown potential cardiovascular protection, anti-inflammatory properties, and possible anticancer effects through mTOR inhibition.", "rejected": "Metformin is strictly an anti-diabetic drug and has no benefits beyond glucose reduction."}
```

The remaining two, quoted in full:

```json
{"prompt": "Summarize the recent developments in mRNA vaccine technology for emerging variants.", "chosen": "mRNA vaccines can be redesigned rapidly to target new variants such as BQ.1 and XBB.1.5, showing strong antibody and T-cell responses in clinical trials. Research is ongoing to create more stable and cost-effective formulations.", "rejected": "mRNA vaccines are identical for all viruses and cannot be changed once developed."}
{"prompt": "Describe how artificial intelligence is used in pharmaceutical research.", "chosen": "AI accelerates drug discovery by predicting protein–ligand interactions, optimizing lead compounds, and integrating with lab automation to shorten discovery timelines.", "rejected": "AI in pharma is mostly used for marketing and doesn't have a role in actual drug discovery."}
```

**Read those rows as a dataset designer, because they teach you the design rules for free.** Every `chosen`/`rejected` pair in this file is constructed so that the difference is **one of three specific things**, and never a formatting difference:

| Pair | The actual axis of preference | What is being taught |
|---|---|---|
| Metformin mechanism | **Specificity.** Rejected is a *false generalisation* ("only… stopping sugar absorption… does not affect other organs") | Reward mechanism-level specificity over confident oversimplification |
| Atorvastatin + Ezetimibe | **Additivity.** Rejected denies a real pharmacological interaction | Reward correct multi-agent reasoning |
| mRNA variants | **Currency.** Rejected claims vaccines "cannot be changed once developed" | Reward up-to-date technical content |
| AI in pharma | **Scope.** Rejected is dismissive ("mostly used for marketing") | Reward substantive domain answers |
| Metformin non-glycemic | **Breadth beyond the obvious.** Rejected is the closed-world claim "strictly an anti-diabetic drug" | Reward the model that surfaces the non-obvious literature |

> **Beyond the video:** the `chosen` strings here are all **plausible, correct, and non-committal** — they are *good textbook answers*, not masterpieces. That is correct dataset design for a demo but a red flag at scale: a preference set where `chosen` is uniformly "the longer, more specific, more hedged answer" teaches exactly one behaviour — **length and hedging** — and that is how length bias enters your model (§4.9). A production preference set deliberately includes pairs where the *shorter* answer wins, otherwise you have accidentally built a verbosity reward.

#### 4.3.2 Where preferences come from

| Source | How it works | Cost / 1k pairs | Quality | When to use |
|---|---|---|---|---|
| **Human annotators, from scratch** | Contracted labelers write both responses or rank model outputs | $2,000–$15,000 | Highest (if guidelines are good) | Safety-critical, domain-specialist, launch-quality |
| **Human ranking of model samples (InstructGPT style)** | Sample k answers per API prompt, humans rank them [28:46]–[29:27] | $1,500–$8,000 | High | General assistant alignment |
| **LLM judge / RLAIF** | A strong LLM labels which response is better, given guidelines | $20–$200 | 80–95% of human agreement | Scale, iteration speed, non-safety |
| **AI + human spot audit** | RLAIF labels, human reviews a 2–5% sample for agreement | $200–$600 | Near-human on non-safety | **The production default** |
| **Model-vs-model comparisons (Arena style)** | Real users vote between two anonymous models | ~free (traffic) | Noisy, biased by formatting/length | Preference data from production |
| **Implicit feedback** | Did the user copy, regenerate, thumbs-up, re-ask, abandon? | free | Very noisy, strongly confounded | Supplement only; never sole source |
| **Rejection sampling** | Sample n from π, keep the ones that pass a filter/checker → pairs where kept = chosen | ~free (compute only) | Bounded by the sampler's ceiling | Verifiable tasks; bootstrapping with no labels |
| **Self-play (SPIN)** | Previous iteration's outputs are `rejected`; human data is `chosen` | ~free | Improves on the SFT ceiling | SFT data is small and you have no new labels |

The instructor names the first two shifts explicitly. The human-labelling era: *"here human was labeling the data right… whatever response we were generating through the model human was ranking those thing"* [28:46]–[28:53]. The scale problem: *"human generated feedback was not very scalable… maybe one lakh two lakh"* [18:56]. The automation: *"initially I shown you like how the human annotated this entire labels… Later on we have automated this task also. On a very high scale. So this task have been automated by the LM only. So we are giving certain instruction to the LLM and based on that it is able to find out which is the chosen one and which is a rejected one"* [38:38]–[38:56].

#### 4.3.3 The three public datasets the video walks through

| Dataset | Org | Rows | Columns | What it is |
|---|---|---|---|---|
| **`Anthropic/hh-rlhf`** | Anthropic | ~170k pairs | `chosen`, `rejected` **only** | The HH dataset. Note: **no explicit `prompt` column** — the prompt is the shared prefix of the two strings. The instructor flags this: *"inside this particular data you will find out two column only. The first is a chosen and the second is a rejected"* [15:43]–[15:49]. |
| **`argilla/ultrafeedback-binarized-preferences-cleaned`** | Argilla (built on UltraFeedback) | ~62k pairs | `prompt`, `chosen`, `rejected` | LLM-judge (GPT-4) preferences over 4 model outputs per prompt, binarised and cleaned. The instructor: *"the data set name is ultra feedback binarize preference clean data… you will find out two main column… the rejected and the second column will be the chosen… along with the given prompt"* [16:00]–[16:14]. |
| **`Zilly/math-step-dpo-10k`** | Zilly | ~10k pairs | `prompt`, `chosen`, `rejected` | Step-level DPO data for *math reasoning* — i.e. a **verifiable-adjacent** set. The instructor: *"the data set name is math step dpo 10k… the first one will be the chosen okay with respect to the particular prompt and the second column will be the rejected"* [16:22]–[16:38]. |

> **Beyond the video:** the `hh-rlhf` missing-prompt subtlety is a real implementation landmine. TRL's `DPOTrainer` accepts either a `prompt` column or a `chosen`/`rejected` pair where the prompt is inferred from the common prefix — but the inference is fragile and can silently include the response's opening tokens in the prompt. **Always materialise an explicit `prompt` column** before training. Cost of getting it wrong: the model is trained to prefer *the continuation after a prompt that already answers the question*, and the resulting model is subtly worse in a way no loss curve reveals.
>
> Also worth knowing: **UltraFeedback is synthetic and derived from GPT-4 judgements.** Its binarised-cleaned variant removed pairs where the judge's confidence was low or the two responses were near-identical — which raises label quality but also **narrows the preference distribution** toward easy, obvious pairs. Training only on cleaned data gives you a model that is good at obvious distinctions and blind to subtle ones. Mix in ~10–20% harder, uncleaned or human-labelled pairs.

#### 4.3.4 Annotation guidelines: what makes a preference set good

A preference annotation guideline is a **policy document** that defines the ordering. Without one, annotators use their own aesthetics and you train a model on noise. The fields every guideline must specify:

1. **Priority order among HHH.** "If an answer is more helpful but less safe, choose the safer one." State it; do not assume it.
2. **What counts as a tie.** Most guidelines forbid ties, which forces noise on genuinely equal pairs. Allow `tie` and either drop or handle those rows (this is what cDPO's label smoothing is for).
3. **Length neutrality, explicitly.** "Do not prefer a response because it is longer. If both are equally correct and complete, prefer the shorter." Without this clause you *will* build a length reward.
4. **Format tolerance.** "Do not prefer markdown over plain prose unless the prompt asked for a format."
5. **Hallucination handling.** "A response containing a fabricated fact loses to a response that admits uncertainty." This is how you teach calibration.
6. **Refusal calibration.** "Refusal is correct only when the request is genuinely harmful. Prefer a helpful answer to an unnecessary refusal."
7. **Escalation path.** Where do annotators flag a pair they cannot judge? Unflagged hard pairs become training noise.
8. **Worked examples.** 20–50 gold pairs with rationales, plus an onboarding quiz with a pass threshold.

#### 4.3.5 Inter-annotator agreement: the metric nobody computes

**IAA measures whether your preference signal exists.** If two competent annotators agree on only 65% of pairs, then 35% of your labels are noise, and no amount of training fixes that — you are fitting a coin flip.

| IAA statistic | Use when | Interpretation |
|---|---|---|
| **Raw agreement %** | Always report it | <70%: the guidelines are broken. 70–80%: workable. >85%: good. |
| **Cohen's κ** | Exactly 2 annotators | κ<0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, >0.8 strong |
| **Krippendorff's α** | n annotators, handles missing data | The right choice for a real annotation team; α>0.667 is the conventional acceptability floor |
| **Spearman ρ on ranks** | k-wise (ranked) data | Use instead of κ when annotators produce orderings |

**Calibration recipe that actually works:** (a) double-annotate **10%** of the set permanently as a control; (b) run a **gold-standard quiz** of 50 pre-labelled pairs before an annotator starts and re-run monthly; (c) measure per-annotator agreement against the majority, and drop annotators whose agreement is an outlier low; (d) measure agreement **by category** — you will almost always find that safety pairs agree at 92% and style pairs at 68%, and the fix is to sharpen the style guidelines, not to fire annotators.

> **Beyond the video:** there is a cheap and under-used trick. **Train your reward model, then look at the pairs where the RM's margin is close to zero but both annotators agreed.** Those are the ambiguous-but-consensual pairs; upweighting them sharpens the decision boundary exactly where the model is weak. Conversely, pairs where annotators disagreed and the RM has a large margin are the model confidently learning noise — drop them.

#### 4.3.6 The cost, in real numbers

| Configuration | Pairs | Cost/pair | Total | Notes |
|---|---|---|---|---|
| Demo (this repo) | 5 | ~$0 (hand-written) | $0 | 5 rows, 1 epoch, loss 0.66 |
| Hobby | 1,000 | $3 | $3,000 | Enough for a tone/style shift on a narrow task |
| Serious task adapter | 5,000 | $3 | $15,000 | The realistic minimum for a production behaviour change |
| General assistant | 50,000 | $2 | $100,000 | Orders of magnitude smaller than the pretraining bill, still real money |
| RLAIF equivalent | 50,000 | $0.04 | $2,000 | ~50× cheaper; agreement ~85% on non-safety |
| RLAIF + 5% human audit | 50,000 | $0.04 + audit | ~$3,500 | The production sweet spot |

> **Beyond the video:** the instructor's framing of RLAIF is a *scalability* argument [18:56], and the arithmetic above is exactly why. But it is incomplete: **AI feedback is biased toward the judge's own preferences**, which are themselves shaped by its own alignment. A GPT-4 judge systematically prefers verbose, heavily-structured, listy answers — because GPT-4 was trained to produce them. Judge-distillation of this kind imports the judge's style as your reward. Mitigations: use a judge from a *different* model family than the one you are aligning; use a rubric-based judge (score 5 axes separately, then aggregate) rather than a single "which is better"; and keep a human-labelled holdout to measure the judge's own agreement with humans.

### 4.4 Reward modelling: Bradley–Terry and the RM loss

The reward model is the artifact that lets you do RL at all. It is also the artifact that most of the modern methods delete.

**The Bradley–Terry model.** Given a prompt x and two responses y+ ≻ y−, assume each response has a latent scalar quality r(x, y), and

```
P(y+ ≻ y− | x) = σ( r(x, y+) − r(x, y−) )
```

where σ is the logistic sigmoid. This is the same model that underlies Elo ratings and Thurstone's law of comparative judgement. It is a *pairwise comparison* model: it never claims to know absolute quality, only differences.

**The reward-model loss.** Maximum likelihood on that model over a dataset of pairs D:

```
L_RM(φ) = − E_{(x, y+, y−) ~ D} [ log σ( r_φ(x, y+) − r_φ(x, y−) ) ]
```

where `r_φ` is implemented as the SFT model with the LM head replaced by a scalar head (one linear projection to a single logit at the final token). InstructGPT's RM is exactly this, and its size is **6B** in the paper — the same size as the policy.

**Why the loss is a *difference*.** Because σ is shift-invariant to a constant added to both rewards: the loss only depends on the margin. This means the RM is identifiable only up to an additive constant per prompt, which is why RM absolute scores are meaningless across prompts and why comparing RM scores between prompts is a bug. It also means you can, and should, **normalise rewards per prompt** during PPO — that is the "reward whitening" step and it materially stabilises training.

**Numeric sanity check.** Suppose r(y+) = 2.0, r(y−) = 0.5. Margin = 1.5. σ(1.5) = 0.818, so loss = −log(0.818) = 0.201. Push the margin to 4.0: σ(4) = 0.982, loss = 0.018. Push the margin to 0: σ(0) = 0.5, loss = 0.693. Note that the loss is exactly `−log 2 ≈ 0.693` when the RM cannot distinguish the pair at all — a useful constant: **an RM whose training loss is stuck at 0.693 has learned nothing.** (Compare against the notebook's DPO loss of **0.6619** after one step — barely below the chance value, because the run is 1 epoch × 5 examples.)

**Cost of an RM.** An RM is a *second full training run* on top of SFT: same data pipeline, same memory budget, plus you must keep it frozen and served alongside. For 7B that is another 14 GB at bf16 inference, and if you full-fine-tune the RM it is another ~112 GB of training memory. This is precisely the cost that DPO eliminates.

> **Beyond the video:** three RM details that separate a working pipeline from a broken one. (1) **Initialise the RM from the SFT model, not the base model.** An RM from a base model produces scores that are uncorrelated with prompt quality and PPO will chase them into incoherence. (2) **The RM must be trained on the same distribution the policy will generate.** If you train the RM on GPT-4 outputs and then run PPO on your own 1B policy's outputs, the RM has never seen your policy's errors and its scores are extrapolation. This "distribution shift between RM training and policy sampling" is one of the top three causes of failed PPO runs. (3) **Normalise the RM's output layer.** InstructGPT divides the reward by a running estimate of its standard deviation across the batch; without normalisation the advantage magnitudes vary by orders of magnitude across prompts and PPO's clip range becomes meaningless.

### 4.5 The KL divergence term

Every alignment objective except ORPO and SimPO contains a term like `β · KL(π_θ(·|x) ‖ π_ref(·|x))`. Understanding it is mandatory.

**What it is.** For a single prompt x,

```
KL(π_θ ‖ π_ref) = Σ_y π_θ(y|x) · log ( π_θ(y|x) / π_ref(y|x) )
```

It is the expected *log-ratio* under the policy's own distribution. Zero when the two distributions are identical; positive otherwise; unbounded above.

**Which direction, and why it matters.** The term is `KL(policy ‖ reference)` — the **forward** KL, with the policy as the expectation distribution. Two consequences:

1. **It is mode-covering, not mode-seeking.** Because the expectation is over the policy, the penalty is large wherever the *policy* puts mass and the reference does not. It does **not** heavily penalise regions where the reference puts mass and the policy does not — the policy is free to abandon modes. That is the behaviour you want: the policy should be allowed to *stop* producing bad answers without being fined for it.
2. **Do not reverse it.** `KL(π_ref ‖ π_θ)` is mode-seeking and would penalise the policy for *ignoring* anything the reference does, including the bad answers — the exact opposite of alignment. A very common interview trap is being asked "which direction is the KL?" and answering "it's symmetric-ish, doesn't matter". It matters.

**Why it exists: the three jobs.**

| Job | Failure it prevents |
|---|---|
| **Reward-hacking brake** | Keeps the policy in the region where the reward model is *valid*. Outside that region the RM is extrapolating and its scores are meaningless — unbounded reward. |
| **Capability preservation** | Keeps the model fluent and on-manifold, so the alignment tax (§4.8) stays small. |
| **Variance reduction** | Prevents the policy collapsing onto a single high-reward degenerate string. |

**The β knob, precisely.** β multiplies the KL term in the *objective* (`max r − β·KL`). Note the two conventions in the wild:

- **KL-in-objective convention** (PPO/RLHF, GRPO): `objective = E[r] − β·KL`, β is the **KL coefficient**. Large β → stay near reference.
- **KL-in-reward convention** (TRL's PPO): the per-token reward is `r_total = r_RM − β·KL_t`, i.e. `β` is the **KL penalty coefficient**. Same meaning, different sign bookkeeping.
- **DPO convention**: β multiplies the *log-ratio inside the sigmoid argument*, `σ( β·(log-ratio difference) )`. Here **large β = stronger preference signal = the policy is pushed harder away from the reference**, which is the *opposite* intuition from PPO. In DPO, β is often described as the "inverse temperature" of the implicit reward.

The instructor gives the DPO-convention description: *"if the beta is high means we are enforcing model to like go to… we are enforcing model to align more to the chosen — if beta is low means the rejected one more the model is more aligning to the rejected"* [26:54]–[27:09]. **This is correct for DPO** and is the source of a lot of confusion when people move between PPO and DPO code. Flag it in interviews.

He gives the range as *"the typical value of the beta between 0.1 to 0.5"* [27:11]. TRL's default is **0.1**, and the notebook uses **β=0.1**.

**What happens at the extremes:**

| β | Objective becomes | Observable behaviour |
|---|---|---|
| **β → 0** | Pure reward maximisation | Policy drifts far from π_ref, fluency collapses, reward hacking is immediate, output degenerates into the RM's favourite degenerate string. Textbook failure: repeated punctuation, or a single "best" answer for every prompt. |
| **β small (0.01–0.05)** | Strong optimisation | Faster win-rate gains, then over-optimisation and degradation on general capability. Use only with early stopping and a general-capability eval. |
| **β = 0.1** | The default | The industry default for DPO; balances movement against stability. |
| **β ≈ 0.3–0.5** | Gentle | Slow, stable, small behavioural shift. Good for safety fine-tuning where you must not damage the base. |
| **β → ∞** | π_θ ≡ π_ref | Zero training signal. The loss is constant at `−log σ(0) = log 2 ≈ 0.693` and the gradient is zero. Symptom: **loss pinned at 0.693, metrics flat, no learning.** |

> **Beyond the video:** the exact β→0 loss floor differs by loss type, and knowing the floors is a superb debugging aid. For DPO with `loss_type="sigmoid"` the floor is `log 2 = 0.6931` (achieved when the margin is 0 — e.g. policy = reference, which is the state at training *step 0*). For IPO the floor depends on the target margin. **A DPO run whose loss never drops below ~0.69 is not learning** — check that `chosen`/`rejected` are correctly mapped (a swapped mapping trains the model to prefer the rejected response, and the loss *still goes down*), and check that `ref_model` is not accidentally the trainable model (which would be a no-op).
>
> Also worth internalising: **β and the number of training epochs trade off against each other.** A long run at β=0.5 approximates a short run at β=0.1. If your eval is flat, raising epochs is usually a worse idea than lowering β, because more epochs on a small preference set overfits fast. Published DPO recipes typically run **1–3 epochs** and start degrading after that.

### 4.6 The method landscape — complete profiles

Each profile answers the same eight questions: mechanism, objective, data, compute, memory, good at, breaks on, and the one-line verdict. Deep dives live in the sibling modules named in each header.

---

#### 4.6.1 RLHF with PPO — the 4-model problem
*Deep dive: this profile. The planned "CS-24" module was never written.*

**Mechanism.** Four models are resident. (1) The **policy** π_θ, initialised from π^SFT, is the only model being trained. (2) The **reference** π_ref is a frozen copy of π^SFT. (3) The **reward model** r_φ is frozen and scores a completed response. (4) The **value model** V_ψ predicts the expected return from a prefix, used to compute advantages. The loop: sample a prompt from the preference prompt distribution → generate a response with π_θ → score it with r_φ → compute per-token rewards, subtracting `β·KL_t` against π_ref at each token → compute advantages with GAE using V_ψ → take a PPO-clipped policy-gradient step and a value-regression step. Repeat for many prompts.

**Objective.**
```
L_PPO = E[ min( ρ_t · Â_t , clip(ρ_t, 1−ε, 1+ε) · Â_t ) ] − c₁·L_VF + c₂·H(π_θ)
```
where `ρ_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the importance ratio, `Â_t` the advantage, ε (typically 0.1–0.2) the clip range, H an entropy bonus.

**Data needed.** (a) prompts only — no labels required for the RL step (that is PPO's advantage: you need *prompts*, not answers); (b) a *separate* preference dataset for RM training; (c) typically 10k–100k+ prompts.

**Compute.** The highest of any method here. Per iteration you generate rollouts (autoregressive, slow), then run *four* forward passes per sample (policy, ref, RM, value) plus a backward pass. Roughly **3–8× the wall-clock cost of DPO for the same number of optimiser steps**, and many more steps are needed.

**Memory.** ~4× the model weights plus optimizer states plus activations. See §4.7 for GB numbers.

**Good at.** Online exploration — it can discover responses the preference dataset never contained. Handling a reward that is *not* expressible as a pairwise preference (e.g. a unit-test pass rate, a custom heuristic, a live A/B metric). It is the only method here that natively supports a non-differentiable black-box reward with a KL anchor.

**Breaks on.** Reward hacking (the RM is a proxy and the policy will find its bugs). Hyperparameter fragility — PPO has ~10 knobs that interact (clip ε, GAE λ, γ, KL coefficient, value-loss coefficient c₁, entropy c₂, batch size, minibatch size, learning rate, rollout length) and a bad combination looks like "training runs but nothing improves". Value-model lag. Reward normalisation bugs. Non-reproducibility.

**Verdict.** Use it only if you need exploration or a black-box reward that is not a pairwise preference. In 2026 that mostly means large labs, reasoning-model RL with a verifier, and reward-model research. Not a first choice for a product team.

---

#### 4.6.2 Reward modelling (Bradley–Terry) as a standalone artifact
*Deep dive: this profile. The planned "CS-24" module was never written.*

**Mechanism.** Take π^SFT, replace the LM head with a scalar head, train on pairs with the BT loss (§4.4). The output is a *reusable* scorer: one RM can drive many policies, many experiments, and can be used for best-of-n selection, for data curation, and for evaluation.

**Objective.** `L = −log σ(r_φ(x, y+) − r_φ(x, y−))`.

**Data.** `{prompt, chosen, rejected}` — the same data DPO uses.

**Compute / memory.** One training run, memory comparable to SFT (slightly less, no generation). At inference the RM costs one forward pass per scored response; for best-of-n over n candidates the cost is n forward passes (batched, cheap relative to generation).

**Good at.** Being a *reusable asset*. If your organisation will run many alignment experiments, an RM amortises. Also: it is the only way to get a dense scalar signal for best-of-n, and the only way to do reward-based data filtering.

**Breaks on.** Distribution shift (§4.4) — the RM is only valid on the distribution it was trained on. It is also a *proxy*, and everything in §4.9 applies.

**Verdict.** Train an RM if you need best-of-n, a dense scoring function, or you intend to run PPO/GRPO-with-RM. Otherwise DPO gives you the same signal without the artifact.

---

#### 4.6.3 DPO — Direct Preference Optimization
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** No reward model, no sampling, no value model. DPO observes that the RLHF optimum has a closed form: the optimal policy under the KL-constrained reward objective is

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

Inverting for r gives `r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β·log Z(x)`. Substituting that *implicit reward* into the Bradley–Terry loss makes the partition function `Z(x)` cancel — it depends only on x, and both members of a pair share the same x. What remains is a loss over pairs, in the policy's own log-probabilities, requiring no reward model at all.

**Objective (the notebook's `loss_type="sigmoid"`).**
```
L_DPO = − E_{(x, y+, y−)} [ log σ( β · ( log(π_θ(y+|x)/π_ref(y+|x)) − log(π_θ(y−|x)/π_ref(y−|x)) ) ) ]
```
Define `h_θ(x,y) = log π_θ(y|x) − log π_ref(y|x)` — the **log-ratio** between policy and reference. Then `L_DPO = −log σ( β · (h_θ(y+) − h_θ(y−)) )`. The instructor's reading of this formula is exactly right and worth quoting because it is the clearest plain-English statement of DPO in the transcript: *"the meaning is how much more does your model prefer the chosen answer compared to the reference model"* [25:59]–[26:04]. He also decomposes it as *"choose an improvement… and the rejected improvement… if it is going to be positive means my model is aligning to this chosen prompt"* [26:29]–[26:40].

**The intuition in one line.** DPO trains the model to *increase the policy/reference log-ratio more for the chosen answer than for the rejected one*. It is a **contrastive classifier on log-ratios**, not a reward maximiser.

**Data.** `{prompt, chosen, rejected}` — the same data as an RM.

**Compute.** Roughly **2 forward passes + 1 backward per pair** (chosen and rejected pass through the policy; the reference is a no-grad forward). No generation. On a fixed dataset this makes DPO **roughly 5–20× cheaper in wall-clock than PPO per unit of quality gained**, and dramatically more reproducible.

**Memory.** 2 models (policy + reference), reducible to ~1 by disabling the adapter on the reference. §4.7.

**Good at.** Almost everything. Stable, reproducible, hyperparameter-light (essentially β, LR, epochs), and reachable from a `DPOTrainer(model, args, train_dataset, processing_class)` call. This is the method the instructor uses, and his summary is *"This is a simple supervised learning… so nowadays we are using this particular technology"* [19:47]–[20:02].

**Breaks on.** (1) **Offline only** — it cannot explore beyond the pairs you give it, so its ceiling is your dataset's ceiling. (2) **Length bias** — DPO has a documented tendency to increase response length, because `log π_θ(y|x)` sums over tokens so longer sequences have larger-magnitude log-probs; this is the motivation for length-normalised variants (SimPO, R-DPO). (3) **Degenerate preferences** — when `chosen` and `rejected` are nearly identical, the log-ratio can be driven to infinity and the model overfits; this is what **IPO** and **cDPO** fix. (4) **Vanishing gradient on easy pairs** — once the margin is large, σ saturates and the gradient vanishes, so the model stops learning from the pairs it has already mastered. That is not a bug; it is why you need hard pairs.

**Verdict.** The default. If you have pairs and no verifier, start here.

---

#### 4.6.4 IPO — Identity Preference Optimization
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** Replaces the log-sigmoid of DPO with a squared loss on the log-ratio margin, and — crucially — places the regularisation *inside the objective* rather than relying on the sigmoid's implicit saturation.

**Objective.** `L_IPO = E[ ( h_θ(y+) − h_θ(y−) − 1/(2β) )² ]` where `h_θ` is the DPO log-ratio. Note the target margin `1/(2β)`: instead of pushing the margin to +∞, IPO pushes it to a *finite* target.

**Data.** Same as DPO.

**Good at.** **Deterministic or near-deterministic preferences.** If your pairs are of the form "correct vs incorrect" (not "nice vs nicer"), DPO's unbounded margin drive overfits them; IPO's finite target does not. Also better when the dataset is small and each pair will be seen many times.

**Breaks on.** With a large, noisy dataset the finite margin target can under-fit: it refuses to push hard on the pairs that genuinely deserve it. In practice, on typical human-preference data, IPO underperforms DPO slightly.

**Verdict.** Reach for it when your pairs are binary-correct rather than graded, or when DPO is overfitting. Exposed in TRL as `loss_type="ipo"`.

---

#### 4.6.5 cDPO — Conservative DPO
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** DPO with **label smoothing**: the target probability for `chosen` is set to `1 − ε` (typically ε = 0.1) rather than 1.0. Concretely, TRL implements it as `losses = (1−ε)·losses_chosen + ε·losses_rejected` — i.e. it admits that ε of your labels are wrong.

**Objective.** `L_cDPO = (1−ε)·(−log σ(β·Δh)) + ε·(−log σ(−β·Δh))` where Δh is the DPO margin.

**Data.** Same as DPO. **The point is to make noisy data usable rather than to collect cleaner data.**

**Good at.** Low inter-annotator agreement. Crowdsourced labels. LLM-judge labels with known error rate. Any dataset where you *know* 5–15% of the pairs are mislabelled.

**Breaks on.** If your labels are actually clean, ε>0 costs you a small amount of achievable quality — you are deliberately teaching the model that 10% of the signal is noise.

**Verdict.** Free insurance. Set ε to your measured (1 − label precision). Exposed as `label_smoothing` in TRL's `DPOConfig`.

---

#### 4.6.6 KTO — Kahneman–Tversky Optimization
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** Drops the pair requirement entirely. KTO takes a **pointwise** dataset of `{prompt, completion, label: desirable|undesirable}` and uses a prospect-theory-inspired utility: losses are weighted asymmetrically for desirable and undesirable examples, reflecting the empirical finding that humans weight losses more heavily than gains. Formally, it uses a logistic utility with separate `λ_D` and `λ_U` weights and a reference-point term computed from a running estimate of the KL to the reference.

**Objective (sketch).** `L_KTO = E_{y~desirable}[ λ_D · (1 − v(x,y)) ] + E_{y~undesirable}[ λ_U · v(x,y) ]` where `v` is the sigmoid of the log-ratio minus a KL-based reference point, and `λ_U > λ_D`.

**Data.** Unpaired binary labels — the instructor names it as *"kanman tverki optimization"* [17:44], and it exists precisely because **most of the preference data a product has is thumbs-up/thumbs-down, not pairs.**

**Good at.** Production telemetry. A/B test outcomes. Any signal where you have "this was good" and "this was bad" but no head-to-head.

**Breaks on.** It throws away the pairing structure, so it is *less* sample-efficient than DPO when you do have pairs. It is also more sensitive to label imbalance (a 95/5 desirable/undesirable split needs care).

**Verdict.** The right choice when your data is unpaired. Otherwise DPO dominates.

---

#### 4.6.7 SLiC-HF — Sequence Likelihood Calibration
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** A hinge (margin) loss on sequence log-probabilities, plus an optional cross-entropy regulariser on the chosen response. Notably, it was published *before* DPO and motivated DPO's authors to derive the closed form.

**Objective.** `L_SLiC = max(0, δ − log π_θ(y+|x) + log π_θ(y−|x)) − λ · log π_θ(y+|x)`

**Data.** Same as DPO.

**Good at.** **Stability.** The hinge's flat region means mastered pairs contribute zero gradient rather than a vanishing-but-nonzero one, and it does not require a reference model at all (the margin is between raw log-probs of the policy). It is very hard to blow up. It was used for Zephyr-7B's alignment (with DPO-style pairs) with good results.

**Breaks on.** Without a reference term it can drift the base model's distribution more than DPO; the `λ` cross-entropy term is doing the anchoring work, and getting λ wrong either anchors nothing or turns it into plain SFT.

**Verdict.** A solid, under-appreciated baseline. Exposed in TRL as `loss_type="hinge"` (and the instructor's DPOConfig comment mentions exactly this: `loss_type="sigmoid",  # or "hinge", depending on experiment`).

---

#### 4.6.8 SimPO — Simple Preference Optimization
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** DPO **without a reference model**. The implicit reward becomes the **length-normalised average log-probability** of the sequence, plus a target margin γ. No π_ref, no β (the margin γ is the strength dial).

**Objective.** `L_SimPO = −log σ( (β/|y+|)·log π_θ(y+|x) − (β/|y−|)·log π_θ(y−|x) − γ )`

**Data.** Same as DPO.

**Good at.** (1) **Memory** — one model fewer, ~30–50% less VRAM. (2) **Length control** — the `1/|y|` normalisation directly removes the length bias that DPO exhibits, so you do not need to post-hoc penalise verbosity. (3) Simpler hyperparameters.

**Breaks on.** Without a reference model, nothing prevents the policy's *absolute* log-probs from collapsing, and in long runs it can produce degenerate low-probability text. The margin γ must be tuned; γ=0 is not the right default.

**Verdict.** The best answer to "I want DPO but I don't have room for a second model" and to "my DPO model got verbose".

---

#### 4.6.9 ORPO — Odds Ratio Preference Optimization
*Deep dive: this profile. The planned "CS-27" module was never written.*

**Mechanism.** One stage, one model, no reference. ORPO adds a preference term to the ordinary SFT loss, where the preference term is an **odds ratio** rather than a probability: the odds of generating `chosen` divided by the odds of generating `rejected`, given the same prompt. The odds formulation is what makes it work without a reference — the SFT loss *is* the anchor, because you are still training on the chosen responses with cross-entropy.

**Objective.**
```
L_ORPO = L_SFT + λ · L_OR
L_SFT  = − (1/|y+|) Σ_t log π_θ(y+_t | x, y+_<t)
L_OR   = − log σ( log( odds_θ(y+|x) / odds_θ(y−|x) ) )
odds_θ(y|x) = π_θ(y|x) / (1 − π_θ(y|x))
```
λ is typically **0.1–0.5** and plays the role β plays in DPO, but weights the preference term relative to SFT rather than a KL.

**Data.** `{prompt, chosen, rejected}`. **And you can simultaneously use a plain SFT dataset**, because the objective already contains an SFT loss — that is the "one stage" claim: you do not need to SFT first and then align, you can do both at once from the base model.

**Compute / memory.** **1 model.** Roughly the cost of a normal SFT run plus one extra forward pass over the rejected sequence. Empirically **~2–3× cheaper than DPO** in wall-clock and about **half the VRAM**.

**Good at.** The cheapest credible alignment run in existence. Perfect for a 7B or smaller model on a single consumer GPU; perfect when you are already doing an SFT run and want the preference signal for free. Very good on style, tone, and format adherence.

**Breaks on.** Because the SFT term dominates, ORPO's behavioural shift is *smaller* than DPO's for the same data — it will not fix a broken model, only polish a working one. It also inherits SFT's length prior, so if your chosen responses are longer, ORPO learns longer. And because there is no KL anchor to a *frozen* policy, it is less controllable at the extremes.

**Verdict.** The best first experiment for a single-GPU practitioner, and the best way to add preference signal to an SFT run you were already going to do.

---

#### 4.6.10 GRPO — Group Relative Policy Optimization
*Deep dive: this profile. The planned "CS-26" module was never written.*

**Mechanism.** PPO **without the value model**. For each prompt, sample a **group** of G responses (G = 4–64) from the current policy, score each with the reward function, and use the **group's mean reward as the baseline**. The advantage of response i is `(r_i − mean(r)) / std(r)` — a z-score within the group. No critic, no GAE, no value loss.

**Objective.**
```
L_GRPO = E[ (1/G) Σ_i (1/|y_i|) Σ_t min( ρ_{i,t}·Â_i , clip(ρ_{i,t},1−ε,1+ε)·Â_i ) ] − β·KL(π_θ ‖ π_ref)
```
where `Â_i` is the group-normalised advantage (a single scalar per *response*, not per token — that is the second big simplification).

**Data.** **Prompts plus a reward function.** No preference pairs, no reward model, no human labels. This is the whole point.

**Compute.** Expensive in *generation*, cheap in *memory*. You must generate G responses per prompt; with G=16 that is 16× the generation cost of one rollout but you get 16 samples of signal from one prompt. Training memory is 2 models (policy + reference) — half of PPO's 4.

**Memory.** 2 models. §4.7.

**Good at.** **Verifiable-reward domains.** If the reward is a program — a unit test passing, a math answer matching, a JSON schema validating, a regex matching, a compiler accepting — then GRPO with that reward needs no reward model and no preference data at all. This is **RLVR**, the recipe behind DeepSeek-R1 and essentially every reasoning model released since. It is also naturally suited to tasks where you want the model to produce long chains of thought, because the reward is on the final answer and the credit assignment is handled by the group baseline.

**Breaks on.** (1) **Reward is all-or-nothing**, so on hard prompts every response in the group scores 0 and the group's std is 0 — the advantage is 0/0 — and you learn nothing. This is the "zero-variance group" problem, addressed by DAPO's dynamic sampling. (2) **It needs a verifier.** For subjective quality there is no verifier, and if you bolt on an RM you have re-created PPO's cost without PPO's value model — which is sometimes still a win, but is not free. (3) **Group size × generation length sets the cost**, and on long-CoT tasks the generation dominates everything.

**Verdict.** The right tool for reasoning training and any task with a programmatic checker. The wrong tool for tone, style, or helpfulness.

---

#### 4.6.11 RLAIF and Constitutional AI
*Deep dive: this profile (RLAIF); the planned "CS-24" module was never written. AP-01 (the values question)*

**Mechanism (RLAIF).** Identical to RLHF except the preference labels come from an LLM judge rather than a human. The instructor: *"it was inspired from the RLHF only the difference is human is not going to be annotate anything now like only AI will do that"* [21:29]–[21:37]. He attributes it to Anthropic, published 2024, and names contributors *"Abhinav Rastogi and Susant Pragas"* [21:55]–[22:01].

**Mechanism (Constitutional AI).** A specific RLAIF recipe: write a **constitution** (a list of principles); have the model critique its own response against a sampled principle and **revise** it; the revision becomes `chosen` and the original becomes `rejected`. Then run preference training on that self-generated data, and optionally a second RL stage against a harmlessness RM trained on AI labels. The key property: **values are written down in a document you can audit and edit**, rather than being implicit in a crowd of annotators.

**Data.** Prompts + a judge (RLAIF) or prompts + a constitution (CAI). No human labels required, though a human-audit sample is strongly advised.

**Compute / memory.** Same as the underlying algorithm (PPO, or DPO if you use the AI labels for DPO — the common choice).

**Good at.** Scale. Cost. Iteration speed. Making values explicit and versioned. The instructor's motivation is purely the annotation ceiling: *"maybe one lakh two lakh"* [18:58].

**Breaks on.** **Judge bias transfer** (§4.3.6 — the model imports the judge's style preferences). **Value lock-in** — a constitution is a static document and encodes the values of whoever wrote it, at the time they wrote it. **Reward hacking against a rubric** — a model optimising "satisfies principle 3" learns to *appear* to satisfy it.

**Verdict.** The default way to build preference data at scale. Pair it with a human-audited holdout or you are flying blind.

---

#### 4.6.12 Rejection sampling, best-of-n, and RAFT
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** The cheapest thing that works. Sample **n** responses per prompt from the current model (or from a stronger model); score them with *anything* — a reward model, a verifier, a heuristic, a judge; keep the best; run **SFT on the kept responses**. In the pairwise variant, the kept response becomes `chosen` and a lower-ranked sample becomes `rejected` (this is often called RS-DPO). **RAFT** (Retrieval-Augmented Fine-Tuning) is a related but distinct recipe that trains on documents-with-distractors; check which one an interviewer means.

**Data.** Prompts. The labels are generated. For pairing you need the ranking or filter.

**Compute.** n× generation cost per prompt, plus a standard SFT run. Memory = 1 model.

**Good at.** Bootstrapping from zero labels. **Best-of-n at *inference* time is the same machinery, and it is a legitimate production technique**: if you can afford 4× latency, sampling 4 and picking the best with a small RM often matches a much larger model's quality on narrow tasks. It is also the standard way to generate a preference dataset to *then* distil into a single-sample model — which is the standard way to make inference cheap again.

**Breaks on.** **Bounded by the sampler.** `max(y_1..y_n) ≤ max over the model's support`, so if the model never produces a correct answer, rejection sampling never finds one. It is a *variance-reduction* method, not a *capability* method. Also: SFT on self-generated data causes **distribution collapse** — the model narrows toward the modes the filter likes, and diversity falls over iterations.

**Verdict.** The right first move when you have no labels and a scorer. It is also the mechanism behind the STaR / self-taught-reasoner line of work.

---

#### 4.6.13 SPIN — Self-Play Fine-tunINg
*Deep dive: this profile. The planned "CS-25" module was never written.*

**Mechanism.** Iterative self-play with no new labels. At iteration t, the **human-annotated SFT data is `chosen`** and the **previous iteration's model outputs are `rejected`**. Train with a DPO-style loss. Then regenerate the rejected set from the newly-trained model and repeat. The theoretical story: the model learns to distinguish its own outputs from human data, and when it can no longer do so, it has matched the human distribution — so the fixed point is a model whose output distribution is indistinguishable from the SFT data's.

**Data.** Only your existing SFT dataset. No preference labels.

**Compute.** k iterations × (generation + a DPO-style run). Memory = 2 models.

**Good at.** Small SFT datasets where you want to exceed the SFT ceiling without new annotation. It genuinely does improve over SFT on benchmarks in the original paper.

**Breaks on.** It **cannot exceed the human data's ceiling** — the fixed point is the SFT distribution, not a better one. Iterating too long causes degeneration as the rejected set becomes indistinguishable from the chosen. It also amplifies whatever biases are in the SFT set.

**Verdict.** A niche but real technique. Useful when annotation budget is zero and the SFT set is clean.

---

#### 4.6.14 Method family tree

```
                        KL-constrained reward maximisation
                        max E[r(x,y)] − β·KL(π‖π_ref)
                                     │
        ┌────────────────────────────┼─────────────────────────────────┐
        │                            │                                 │
   explicit RM                  closed form                      RL with a
   + RL loop                    (no RM at all)                   program reward
        │                            │                                 │
   ┌────┴─────┐              ┌───────┴────────┐                ┌───────┴────────┐
   │          │              │                │                │                │
 RLHF+PPO  GRPO          DPO family      drop the ref      GRPO + verifier  PPO + verifier
 (4 models) (2 models,    ┌──┴────┬─────┬──────┐            (RLVR)          (rare)
            group rel.)   │       │     │      │
                         DPO    IPO  cDPO  SimPO
                        (2)    (2)   (2)   (1, no ref)
                                  │
                            ┌─────┴──────┐
                            │            │
                    keep SFT loss    drop the pair
                    (1 model)        (unpaired)
                         │                │
                       ORPO             KTO
                    (1 model, no ref)  (1 model, no ref)
```

### 4.7 The memory arithmetic that decides everything

**This is the most practically decisive section of the module.** Quality differences between methods are real but second-order; memory differences are binary — a method either fits on your hardware or it does not.

#### 4.7.1 The base formula

For **full fine-tuning** with AdamW in mixed precision, per trainable parameter:

| Component | Precision | Bytes/param |
|---|---|---|
| Model weights (forward/backward) | bf16 | 2 |
| Gradients | bf16 | 2 |
| FP32 master weights (for the optimiser update) | fp32 | 4 |
| Adam first moment (m) | fp32 | 4 |
| Adam second moment (v) | fp32 | 4 |
| **Total** | | **16** |

With **8-bit Adam** (bitsandbytes `optim="adamw_8bit"` or `paged_adamw_8bit`): 2 + 2 + 4 + 1 + 1 = **10 bytes/param**. With **Adafactor**: no m, factored v ≈ 2 + 2 + 0 + 0 + ~0.5 ≈ **5–6 bytes/param**.

Then add **activations**, which depend on batch size × sequence length × hidden size × layers and are typically 5–25% of the total for a small batch, and can dominate for long sequences.

**Frozen-model inference cost** (for reference/reward/value models held at bf16, no gradients):

| Precision | Bytes/param | 7B model |
|---|---|---|
| bf16 | 2 | 14 GB |
| int8 | 1 | 7 GB |
| NF4 (QLoRA) | ~0.5 + overhead | ~3.5–4.5 GB |

#### 4.7.2 Worked arithmetic for a 7B model — full fine-tuning

Weights: 7B × 16 B = **112 GB** for the trained model alone. Frozen copies are 14 GB each at bf16 (7 GB at int8).

| Method | Trained models | Frozen models | Weights (GB) | Optimizer+grad (GB) | Activations (GB) | **Total VRAM** | Fits on |
|---|---|---|---|---|---|---|---|
| **PPO (full FT policy + full FT value)** | policy 7B, value 7B | ref 7B, RM 7B | 28 + 28 (frozen) = 56 | 112 + 112 | 15–40 | **~295–320 GB** | 4×H100 80GB (tight) |
| **PPO (LoRA policy, LoRA value, frozen ref+RM)** | adapters only | ref, RM, (value base) | 14 + 14 + 14 = 42 | ~2 | 15–40 | **~60–85 GB** | 1×H100 80GB or 2×A100 |
| **PPO (QLoRA policy+value, 4-bit ref+RM)** | adapters | 4-bit ref, RM | 4.5+4.5+4.5 = 13.5 | ~1 | 10–25 | **~25–40 GB** | 1×A100 40GB / 1×RTX 4090 24GB (tight) |
| **RLHF reward model training** | RM 7B | — | 14 | 112 | 10–20 | **~136–146 GB** | 2×H100 |
| **DPO (full FT)** | policy 7B | ref 7B | 14 + 14 = 28 | 112 | 10–25 | **~150–165 GB** | 2–3×H100 |
| **DPO (LoRA)** | adapters on 7B | ref 7B (or shared) | 14 + 14 = 28 | ~1 | 8–18 | **~37–47 GB** | 1×A100 40GB |
| **DPO (QLoRA, shared ref)** | adapters on 4-bit | same model, adapter off | 4.5 | ~1 | 6–14 | **~12–20 GB** | 1×RTX 4090 / 1×A6000 |
| **SimPO (LoRA)** | adapters | **none** | 14 | ~1 | 8–18 | **~23–33 GB** | 1×A100 40GB |
| **GRPO (LoRA)** | adapters | ref 7B | 14 + 14 = 28 | ~1 | 10–25 (rollouts add KV cache) | **~40–60 GB** | 1×H100 |
| **ORPO (full FT)** | policy 7B | **none** | 14 | 112 | 10–25 | **~136–150 GB** | 2×H100 |
| **ORPO (LoRA)** | adapters | **none** | 14 | ~1 | 8–18 | **~23–33 GB** | 1×A100 40GB |
| **ORPO (QLoRA)** | adapters on 4-bit | **none** | 4.5 | ~0.5 | 6–14 | **~11–19 GB** | 1×RTX 4090 / 1×T4 16GB (tight) |

Reading the table: **the step from PPO to DPO is a ~3–4× VRAM reduction; the step from DPO to ORPO is another ~1.5–2×; the step from full FT to QLoRA is ~3×.** These multiply. A 4-model PPO at full FT is *fifteen times* the VRAM of ORPO with QLoRA.

#### 4.7.3 The same arithmetic for a 1.1B model (the video's TinyLlama)

The notebook actually runs TinyLlama-1.1B in **8-bit** with LoRA r=8, so the real footprint is small:

| Component | Calculation | VRAM |
|---|---|---|
| Base weights, load_in_8bit | 1.1B × 1 B | ~1.1 GB |
| LoRA adapters (r=8 on q_proj, v_proj, 22 layers) | q: 2×2048×8, v: 2×2048×8 per layer ≈ 65k × 22 ≈ 1.4M params | ~6 MB |
| bf16 compute copies of trained modules | — | ~50 MB |
| Activations (batch 1, short sequences) | — | ~0.5–1.5 GB |
| Optimizer (AdamW on 1.4M params) | 1.4M × 8 B | ~11 MB |
| **Total** | | **~2–3 GB** |

That is why the notebook's `trainer.train()` completes in **5.18 seconds** with 5 examples and 1 epoch. The whole point of the exercise is the *pipeline*, not the scale.

> **Beyond the video:** the notebook's `load_in_8bit=True` triggers a deprecation warning on screen — *"The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."* The current correct form is:
>
> ```python
> from transformers import BitsAndBytesConfig
> bnb = BitsAndBytesConfig(load_in_8bit=True)          # or load_in_4bit=True, bnb_4bit_quant_type="nf4"
> model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto")
> ```
>
> There is a second warning on screen that matters more: **`UserWarning: Merge lora module to 8-bit linear may get different generations due to rounding errors.`** Merging a LoRA into an *8-bit* base is lossy — the merged weight must be re-quantised to int8, and the rounding changes the model's outputs relative to the unmerged path. For the merge-then-reattach pattern in §6.5, this means **the merged 8-bit model is not exactly the instruction model you trained**. Use `load_in_4bit` with `bnb_4bit_compute_dtype=torch.bfloat16` and merge into a fp16/bf16 base if you care about bit-fidelity; or accept the rounding and evaluate the merged model rather than assuming it is unchanged.

#### 4.7.4 Throughput and wall-clock

Memory decides *whether*, throughput decides *how long*. Approximate relative throughput for a 7B model on one H100:

| Method | Token generation in training? | Relative wall-clock per optimiser step | Notes |
|---|---|---|---|
| ORPO (LoRA) | No | **1×** | One forward over chosen + one over rejected, one backward |
| DPO (LoRA) | No | **1.1×** | Extra no-grad forward for the reference |
| SimPO (LoRA) | No | **1.0×** | No reference pass at all |
| KTO (LoRA) | No | **1.0×** | Pointwise, so half the data per step |
| GRPO (LoRA) | **Yes, G per prompt** | **5–30×** | Dominated by rollouts; G=16 with 1k-token answers is brutal |
| PPO (LoRA) | **Yes, 1 per prompt** | **8–25×** | Generation + 4 forward passes + 2 backward passes |
| Rejection sampling | **Yes, n per prompt** | **n× generation + 1× SFT** | The generation is the whole cost |

**The rule of thumb:** *if the method generates tokens during training, it costs 5–25× more than one that does not.* DPO, ORPO, SimPO, IPO, cDPO, KTO and SLiC-HF never generate. PPO and GRPO always do. That single distinction explains most of the field's migration.

### 4.8 The alignment tax

**The claim (InstructGPT).** Aligning a model degrades some capabilities. OpenAI's paper reported that RLHF models scored worse than the base GPT-3 on several public NLP benchmarks — SQuAD, DROP, HellaSwag, and others — while being strongly preferred by humans. They called this the **alignment tax** and introduced **PPO-ptx** to pay it back: add a term to the PPO objective that also maximises the log-likelihood of pretraining text.

```
L_PPO-ptx = L_PPO + γ_ptx · E_{x ~ D_pretrain}[ log π_θ(x) ]
```

with `γ_ptx` typically 0.01–0.1 (the paper used ~27× the KL coefficient). This is a **mixing** approach: gradients from pretraining data flow into the policy update, so the model does not forget how to do general text.

**Why it happens, mechanistically.** Three distinct causes, and it matters which one you have:

| Cause | Mechanism | Distinguishing symptom |
|---|---|---|
| **Narrow reward** | The RM only scores the task distribution. Capabilities not exercised in that distribution receive no reward and are free to drift under the KL term's pressure. | Benchmarks outside the alignment task drop; in-task quality rises. |
| **KL pressure on the wrong axes** | Even a correct KL anchor pins the policy to π^SFT, and π^SFT is itself already slightly degraded relative to the base. Alignment does not recover pretraining capabilities that SFT already cost you. | Post-DPO model is *between* SFT and base, not worse than SFT. |
| **Distribution shift on long chains** | Preference data is short and single-turn; long-form reasoning and multi-turn coherence are unrepresented and degrade. | Long-generation evals and multi-turn benchmarks drop most. |

**The mitigations, ranked by practicality:**

1. **Mix pretraining/SFT data into the preference run** — the PPO-ptx idea, available in every modern trainer as a "replay" or "auxiliary loss" term. In TRL's DPO the closest analogue is adding an SFT loss on `chosen` (which is exactly what ORPO does natively).
2. **Keep β high enough.** A low β is a direct instruction to drift; drift is where capability loss lives.
3. **Stop early.** The alignment tax grows with optimisation pressure, exactly like reward hacking (§4.9). Early stopping on a *general capability* eval, not on the preference metric, is the control.
4. **LoRA instead of full FT.** Adapters constrain the update to a low-rank subspace; empirically this reduces the tax substantially, and it lets you ship one base + several adapters.
5. **Evaluate both axes and publish both numbers.** A model that gains 8 points of win rate and loses 6 points of MMLU may be the right trade — but only if you *know*, and only if you decided.

> **Beyond the video:** the alignment tax is smaller in 2026 than it was in 2022, for a boring reason — the SFT stages got better and the preference data got cleaner, so the policy starts closer to where it needs to be and the KL penalty has less work to do. But it has **not disappeared, and it cannot**: any KL-constrained optimisation away from π^SFT must reduce log-likelihood on π^SFT's distribution, and some of π^SFT's distribution is genuinely useful capability. The honest formulation is a Pareto frontier: you are choosing a point, not removing a cost. Report both axes or you are hiding a decision.

### 4.9 Reward hacking and Goodhart's law

**"When a measure becomes a target, it ceases to be a good measure."** In RLHF the reward model is the measure and the policy is optimising it, so the law applies with full force. This is *not* overfitting to the training set — it happens on the training distribution, with a perfectly-generalising reward model, because the policy is doing *search* over the response space and the RM has errors everywhere in that space.

#### 4.9.1 What it looks like

| Hack | Mechanism | Where you see it |
|---|---|---|
| **Length bias** | Annotators and LLM judges both prefer longer answers when quality is close. The policy discovers that adding a paragraph raises the RM score. | Mean response length grows monotonically through training; win rate rises then plateaus; users complain about verbosity |
| **Sycophancy** | The RM was trained on human preferences, and humans prefer agreement. The policy discovers that validating the user's premise scores higher than correcting it. | Model agrees with false premises; changes its answer when the user pushes back regardless of evidence |
| **Formatting exploits** | Bullet points, bold headers, and "Let me break this down" preambles score higher with LLM judges. | Every response becomes a bulleted list with a summary at the end |
| **Uncertainty deletion** | "I'm not sure" scores lower than confident phrasing. The policy learns to never hedge. | Calibration collapses; hallucination confidence rises |
| **Refusal drift** | A safety-tuned RM rewards refusal; the policy learns to refuse benign requests. | False-refusal rate climbs |
| **Reward-model-specific exploits** | The RM has a *specific* bug — e.g. it scores the token "however" highly because of a spurious correlation in its training data. The policy finds it. | Output becomes stilted or repetitious in a way that correlates with RM score |
| **Degenerate collapse** | With β too low, the policy converges to a single output for all prompts. | Every prompt gets nearly the same response |

#### 4.9.2 The evidence

Two findings you should be able to cite:

- **Gao, Schulman & Hilton (2023), "Scaling Laws for Reward Model Overoptimization."** Their result: as you optimise harder against a fixed RM, the *proxy* reward rises monotonically while the *true* reward (measured by a much larger "gold" RM) rises, peaks, and then **falls**. The gap grows with the number of optimisation steps and is well-fit by a functional form in `√(KL divergence)`. Concretely: the true reward peaks at a KL of roughly 5–20 nats depending on the RM size, then declines. **This is why you early-stop on a proxy-independent metric.**
- **InstructGPT §4 / Anthropic's HH work.** Human preference scores improve while capability benchmarks decline, and the decline grows with more PPO steps — the same shape from the human-eval side.

#### 4.9.3 Every mitigation, ranked

| Mitigation | How it works | Cost | Effectiveness |
|---|---|---|---|
| **KL penalty (β)** | Bounds how far the policy can travel from π_ref, and therefore how much of the RM's error surface it can explore. | Free (a hyperparameter) | High — the primary control |
| **Early stopping on a held-out, human or gold-judged metric** | Stop when the *true* metric peaks, not when the RM peaks. | Requires a human/judge eval loop | High — the second control, and the only one that is independent of the RM |
| **Length-normalised rewards** | Divide the sequence reward by length (or use SimPO's average-log-prob formulation). | Free | High for length bias specifically |
| **Reward-model ensembles** | Train k RMs on different data orderings/seeds; use the minimum or the mean minus a variance penalty. The policy can only hack the *intersection* of their errors. | k× RM training cost, k× inference memory | Medium-high; k=3–5 is typical. Penalising by inter-model *disagreement* is better than the mean alone |
| **Reward clipping / normalisation** | Clip per-prompt reward to a range, or whiten per batch. | Free | Medium — prevents any single prompt dominating |
| **Reward model refresh (iterative RLHF)** | Re-collect preferences on the *current* policy's outputs, retrain the RM, continue. | Very expensive (the InstructGPT 3-step process: SFT → RM → PPO → new preferences → new RM → PPO again) | High; the standard at frontier labs |
| **Conservative/uncertainty-aware reward** | Penalise rewards in regions of low RM training density (e.g. a "conservative" penalty proportional to RM ensemble variance). | Moderate | Medium-high |
| **Preference data on the policy's own distribution** | Train the RM on outputs from the model you will align, not from a stronger model. | Data re-collection | High — removes the shift, not the hack |
| **Auxiliary capability loss (PPO-ptx / SFT mixing)** | Keeps general capability from degrading, so the tax is paid back even if some hacking occurs. | Free-ish | Medium |
| **Red-teaming the RM directly** | Adversarially search for inputs where the RM scores a bad response highly; add them as training pairs. | Moderate | Medium; fixes specific exploits, not the general phenomenon |
| **Capability evals in the training loop** | Log MMLU/GSM8K/IFEval every N steps alongside the RM score. A divergence is your alarm. | Cheap | **Mandatory.** This is the detection mechanism, not a mitigation |

**The single most important operational rule from this section:** *never select a checkpoint on the reward-model score.* The RM score is the thing being gamed; it will be highest at the worst checkpoint. Select on a human or gold-judge win rate plus a capability suite, and treat the RM score as a training diagnostic only.

---

## 5. The End-to-End Pipeline

### 5.1 The canonical eight stages

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 1. TASK FRAMING                                                              │
 │    Is this a behaviour problem or a knowledge problem?                       │
 │    Behaviour → alignment. Knowledge → RAG (CS-04).  STOP if knowledge.        │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 2. SFT BASELINE (CS-13)                                                      │
 │    Do you have an instruction-tuned model? If not, SFT first.                │
 │    Gate: the SFT model must already produce correct, formatted answers.      │
 │    Failure mode if skipped: §4.2 — alignment on a base model degrades it.    │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 3. PREFERENCE DATA ACQUISITION                                               │
 │    Pick a source (§4.3.2). Write guidelines (§4.3.4). Measure IAA (§4.3.5).  │
 │    Gate: IAA > 70% raw agreement before you spend money on volume.           │
 │    Failure mode: low IAA → you train on noise, no loss curve reveals it.     │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 4. BASELINE EVAL (do this BEFORE training)                                   │
 │    Win rate of the SFT model vs itself (sanity), MT-Bench / AlpacaEval,      │
 │    a capability suite (MMLU/GSM8K/IFEval), and a red-team set.               │
 │    If you skip this you cannot claim your alignment run did anything.        │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 5. METHOD SELECTION  ← the decision tree, §8                               │
 │    Verifiable reward?  → GRPO/RLVR                                           │
 │    Have pairs, no verifier? → DPO (default) or ORPO (cheapest)               │
 │    Black-box non-pair reward? → PPO                                          │
 │    Unpaired thumbs data? → KTO                                               │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 6. TRAINING RUN                                                              │
 │    DPO: LR 5e-7–5e-6 (full FT) or 1e-5–2e-5 (LoRA); β 0.1; 1–3 epochs;       │
 │         warmup 10%; bf16; gradient checkpointing if memory-bound.            │
 │    Watch: loss (must fall below 0.693), implicit margin, response length,    │
 │          KL to reference, and the capability suite.                          │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 7. EVALUATION & CHECKPOINT SELECTION                                         │
 │    Select on: win rate vs SFT (human or gold judge) + capability suite.      │
 │    NOT on: RM score, training loss, or preference accuracy on train.         │
 │    Gate: win rate > 55% AND general-capability delta > −2%. Otherwise stop.  │
 └───────────────────────────┬──────────────────────────────────────────────────┘
                             ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ 8. SHIP, MONITOR, RE-ALIGN                                                    │
 │    Adapter versioning, A/B at fixed traffic, drift detection on win rate,     │
 │    re-collect preferences on production outputs, iterate (iterative RLHF).    │
 └──────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Stage-by-stage: input → operation → output → failure mode

| # | Stage | Input | Operation | Output | Failure mode |
|---|---|---|---|---|---|
| 1 | Task framing | Product requirement | Classify as behaviour vs knowledge | Go/no-go | Fine-tuning to inject knowledge (CS-04) |
| 2 | SFT baseline | Base model + instruction data | SFT run | π^SFT | Skipping → §4.2 |
| 3 | Preference data | Prompts + annotators/judge | Elicit comparisons, write guidelines | `{prompt, chosen, rejected}` | Low IAA; length-correlated `chosen`; judge-bias transfer |
| 4 | Baseline eval | π^SFT + eval sets | Freeze the numbers | Baseline report | Skipping → unmeasurable claims |
| 5 | Method selection | Budget, data, reward type | Decision tree | Chosen method + config | Choosing PPO because it is famous |
| 6 | Training | Pairs + method + config | Optimiser steps | π^aligned | β too low → collapse; β too high → loss stuck at 0.693 |
| 7 | Eval & selection | Checkpoints + eval sets | Score all axes | Shipped checkpoint | Selecting on RM score |
| 8 | Production | Serving stack | A/B, monitor, re-collect | Improved model | Silent drift; no rollback path |

### 5.3 The "before and after" the video actually demonstrates

The notebook runs the same question through three checkpoints, which is exactly the right protocol and worth reproducing:

| Model | Response (abridged, verbatim from the notebook output) | Diagnosis |
|---|---|---|
| **Non-instruction** (`checkpoint-5`) | *"Crohn's disease (CD) is an inflammatory bowel disease that can be severe. Recent research has shown that metformin may reduce the risk of flare-ups in people with CD."* | Answers a question that was not asked — off-task drift, the hallmark of domain-adapted-but-not-instruction-tuned |
| **Instruction** (`checkpoint-3`) | *"Crohn's disease (CD) is an inflammatory bowel disease…"* — **identical** | Same off-task output. The instructor notes the redundancy: *"there's some redundancy in the output because I did the mistake in the previous training"* [56:34] |
| **Preference-aligned** (`tinyllama-preference-alignment/checkpoint-1`) | *"Metformin, a common diabetes drug used to treat type 2 diabetes, has long been known as a \"fat burner,\" but new research suggests that it may also help prevent cancer. In a review of more than 100 studies published this week in the journal Diabetologia, researchers from Britain's National Health Service (NHS) and Oxford University said metformin may have properties that can be exploited to prevent cancer by:"* | **On-topic.** Answers the actual question about Metformin. Still truncated and with an invented citation — the aligned model is more *on-task*, not more *truthful* |

> **Read this honestly as a case study.** The DPO model is visibly better at staying on topic. But the third output contains a **fabricated citation** ("a review of more than 100 studies published this week in the journal Diabetologia… NHS and Oxford University"). A 5-pair DPO run on a 1.1B model did not make the model more truthful — it made it more *responsive to the prompt*. That is precisely what preference training on 5 pairs can and cannot do. Do not over-read the demo; do read it as proof that the pipeline works end to end and is cheap to run.

---

## 6. Hands-On Code (annotated)

The companion notebook is `Preference_Aligned_Training_DPO_final.ipynb`. Environment actually recorded on screen [cell 12–13 output]: **TRL 0.25.1, transformers 4.57.1, datasets 4.0.0, bitsandbytes 0.48.2, torch 2.9.0+cu126, accelerate 1.11.0**. The video's install commands are `!pip install -U trl` and `!pip install -U bitsandbytes`, followed by a kernel restart.

### 6.1 Imports and tokenizer

```python
# --- Cell 1, 14: imports -----------------------------------------------------------------
# AutoModelForCausalLM: loads a decoder-only LM head model.
# PeftModel:            wraps a base model with an existing adapter (the loader, not the creator).
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

base_model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"   # [31:28]
instruction_checkpoint = "/content/checkpoint-3"                     # stage-2 output from CS-13

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token     # TinyLlama ships no pad token [33:08]
```

**Why the pad-token line matters.** TinyLlama's tokenizer has `pad_token = None`. `DPOTrainer` pads `chosen`/`rejected` to a common length within a batch, and with no pad token it either errors or silently pads with id 0. The video's screen output later confirms the alignment happened: *"The tokenizer has new PAD/BOS/EOS tokens that differ from the model config… Updated tokens: {'pad_token_id': 2}"* — the tokenizer's pad token (id 2, `<|eos|>`) was pushed into the model config. **Consequence: your model now has `eos_token_id == pad_token_id == 2`.** During generation this can cause the model to emit EOS as a pad and stop early, or to treat padding as end-of-sequence. It is the standard TinyLlama setup and it works, but be aware it is a compromise.

**What to change for your own data:** nothing here, except the model ID and the pad-token policy. For a model whose tokenizer has a real pad token (Llama-3, Qwen, Mistral), do **not** overwrite it.

### 6.2 Loading the stage-2 model and running the SFT baseline

```python
# --- Cell 6: load the instruction (SFT) checkpoint ----------------------------------------
instruction_model = AutoModelForCausalLM.from_pretrained("/content/checkpoint-3", device_map="auto")

# --- Cells 7–10: the pre-alignment baseline [33:11] ---------------------------------------
prompt = "Explain how artificial intelligence is improving the process of drug discovery and development in the pharmaceutical industry."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = instruction_model.generate(
    **inputs,
    max_new_tokens=100,     # hard cap on generated length
    temperature=0.8,        # sampling temperature — 0.8 is a common "balanced" value
    top_p=0.9,              # nucleus sampling: keep the smallest set with cumulative prob >= 0.9
    do_sample=True,         # REQUIRED, otherwise temperature/top_p are ignored (greedy)
    repetition_penalty=1.1  # [33:47] — the instructor: "we can write up to two"
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Annotated output** (verbatim from the notebook):

```
Explain how artificial intelligence is improving the process of drug discovery and
development in the pharmaceutical industry. 19.3 Explain the main benefits of big data
in the pharmaceutical industry and discuss its impact on drug discovery and development.
Medicine is a subject that has always been important to our society... The first was the
era of barbaric times when medicine was simply a matter of faith and belief, where medicines
```

**Diagnosis:** the model *repeats the prompt*, then emits "19.3" — a textbook exercise number — then drifts into an essay about the history of medicine. This is a 1.1B model with a very short SFT run; it has learned the *shape* of an answer but not the *content*. **This is the correct baseline to run before alignment**, because it is the thing you must beat. The instructor's own comment is honest about it: *"you can get some better answer the uh if you have much data then definitely you will get some good answer aligned answer itself. Uh now here this answer is also aligned and uh it is with respect to my question"* [34:21]–[34:32].

**Repetition penalty, precisely.** It divides the logits of already-generated tokens by the penalty (or multiplies by it, depending on direction) before sampling. Values **above 1.0 discourage repetition**; 1.1 is mild, 1.3 is aggressive, **1.5+ starts breaking syntax** because the model is forbidden from repeating necessary function words and punctuation. The instructor's "up to two" is generous — in production, stay in **1.0–1.3** and prefer `no_repeat_ngram_size` or a presence penalty when you need stronger control.

### 6.3 The LoRA config

```python
# --- Cell 21 [45:48] ----------------------------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          # next-token LM head; the correct task type for DPO/ORPO
    r=8,                                    # rank of the update matrices. Tiny — deliberate.
    lora_alpha=16,                          # scaling numerator; effective scale = alpha / r = 2.0
    lora_dropout=0.05,                      # dropout on the LoRA path
    target_modules=["q_proj", "v_proj"],    # only attention Q and V — the original LoRA paper's choice
    bias="none"                             # do not train any bias terms
)
```

| Field | What it does | The video's value | What to change |
|---|---|---|---|
| `task_type` | Tells PEFT which modules to inject into and what head to keep | `CAUSAL_LM` | Keep. Use `SEQ_CLS` only for classifiers. |
| `r` | Rank of the A/B matrices; capacity of the delta patch | **8** | For preference training use **16–64**. r=8 on q/v only is under-powered for a behavioural shift; r=16–32 on all linear layers is the modern default. |
| `lora_alpha` | Scale on the B matrix; effective LoRA scale is `alpha/r` | **16** (scale 2.0) | Common practice is `alpha = 2r` (scale 2.0) or `alpha = r` (scale 1.0). For DPO, **`alpha = r` (scale 1.0)** is frequently better because DPO's own β already controls the update magnitude — a scale of 2.0 compounds with β. |
| `lora_dropout` | Regularisation on the adapter path | **0.05** | 0.0–0.1. Above 0.1 slows convergence; below 0.05 overfits small preference sets. |
| `target_modules` | Which linear layers get adapters | **q_proj, v_proj** | The 2026 default is **all linear layers**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. With q/v only you are tuning ~0.05% of parameters; on a 1.1B model that is ~0.5M params and is barely enough to move behaviour. |
| `bias` | Whether bias terms are trained | `none` | Keep `none`. |

**Parameter count for the video's exact config, computed from the notebook's own model printout.** The model dump shows 22 layers, hidden 2048, and — importantly — `k_proj`/`v_proj` with `out_features=256` while `q_proj`/`o_proj` have `out_features=2048`. That is **grouped-query attention**: 32 query heads × 64 head_dim = 2048, and 4 KV heads × 64 = 256.

| Module | Shapes | Params per layer | × 22 layers |
|---|---|---|---|
| `q_proj` LoRA | A: 8×2048, B: 2048×8 | 32,768 | 720,896 |
| `v_proj` LoRA | A: 8×2048, B: 256×8 | 18,432 | 405,504 |
| **Total trainable** | | **51,200** | **1,126,400** |

1.13M trainable parameters out of 1.1B — **0.1%**. That is a very small delta patch, and it is why the demo's behavioural shift is modest.

### 6.4 Loading the preference dataset

```python
# --- Cell 17 [37:50] ----------------------------------------------------------------------
dataset = load_dataset("csv", data_files="/content/pharma_preference_data.csv")["train"]
```

The instructor notes the format is interchangeable: *"This data could be available in CSV as well as in the JSON format. So I basically kept the CSV as of now but you can load the same from the JSON file also"* [37:55]–[38:03].

```python
# Equivalent, and preferred — JSONL handles embedded commas and unicode cleanly:
dataset = load_dataset("json", data_files="pharma_preference_data.jsonl")["train"]
```

**The dataset, counted.** 5 rows. 3 columns. Columns: `prompt`, `chosen`, `rejected`. Total tokens across all 15 strings ≈ 700 — smaller than a single training batch at a realistic sequence length. See §4.3.1 for the verbatim rows.

> **Beyond the video:** two lines that will save you an afternoon. (1) **Always inspect the schema before training**: `print(dataset.column_names)` must contain `prompt`, `chosen`, `rejected` *exactly*. TRL silently accepts several aliases (`response`, `completion`) in some versions and silently produces garbage in others. (2) **Assert that chosen ≠ rejected** and that neither is empty — a preference set built by an LLM judge with a parsing bug frequently contains rows where both fields hold the same text, and those rows contribute a zero margin and teach nothing while consuming your compute.

```python
# Pre-flight checks worth running on every preference dataset
assert {"prompt", "chosen", "rejected"} <= set(dataset.column_names), dataset.column_names
n_identical = sum(1 for r in dataset if r["chosen"].strip() == r["rejected"].strip())
n_empty     = sum(1 for r in dataset if not r["prompt"].strip()
                                     or not r["chosen"].strip()
                                     or not r["rejected"].strip())
print(f"rows={len(dataset)} identical_pairs={n_identical} empty_fields={n_empty}")
# On the video's data: rows=5 identical_pairs=0 empty_fields=0  ← clean, as it should be
```

### 6.5 The LoRA-stacking bug — the video's central practical lesson

This is the part the instructor flags as *"very very very important… most of the beginner and even the experienced person does this kind of mistake"* [36:00]–[36:11].

**The wrong way** (commented out in the notebook, cell 22):

```python
# ❌ WRONG — this attaches a SECOND adapter on top of the instruction adapter
pref_model_lora = get_peft_model(instruction_model, lora_config)
```

The notebook itself prints the two warnings that prove it is wrong:

```
UserWarning: You are trying to modify a model with PEFT for a second time.
  If you want to reload the model with a different config, make sure to call `.unload()` before.
UserWarning: Already found a `peft_config` attribute in the model.
  This will lead to having multiple adapters in the model. Make sure to know what you are doing!
```

**Why it is wrong.** A LoRA is a **delta patch**, not a layer. *"LoRA is not a full layer, it's not a complete model, it's just a delta patch… delta patch cannot be stacked they must be merged before the next training"* [41:52]–[42:06]. Mechanically, the forward pass becomes `W·x + (α/r)·B₁A₁·x + (α/r)·B₂A₂·x`. If the first adapter was trained for instruction-following, the second adapter must learn to correct the *sum* of two frozen-plus-adapter paths. Training only touches `A₂`/`B₂` while `A₁`/`B₁` stay frozen — so the new patch is optimising against a fixed offset it cannot see or adjust.

**The instructor's four named consequences** [44:13]–[44:27]: *"loss will be unstable, model will hallucinate, tuning will not be good and the quality will degrade."* His mechanism: *"because loss will be unstable. We won't be able to figure out the loss properly."*

**The correct way** (cells 24–27):

```python
# STEP A: load the BASE model (not the instruction model) --------------------------------
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,       # ⚠ deprecated in transformers 5.x — use BitsAndBytesConfig (see §4.7.3)
    device_map="auto"
)

# STEP B: attach the PREVIOUS stage's LoRA, then FOLD IT INTO the base weights ------------
model = PeftModel.from_pretrained(model, instruction_checkpoint)  # loads an EXISTING adapter
model = model.merge_and_unload()   # W_merged = W_base + (alpha/r) * B * A ; adapter wrapper removed
# ⚠ emits: "Merge lora module to 8-bit linear may get different generations due to rounding errors."

# STEP C: now, on the MERGED weights, attach a FRESH adapter ------------------------------
pref_model_lora = get_peft_model(model, lora_config)   # creates a NEW delta patch to train
```

**The instructor's own summary table** [51:04]–[51:26], reproduced as the notebook cell 28 markdown:

| Stage | What You Should Do | Wrong Way |
|---|---|---|
| Non-Instruction | Base + LoRA | ✅ correct |
| Instruction | Base + **merge(stage1 LoRA)** + NEW LoRA | ❌ "LoRA on LoRA" |
| Preference | Base + **merge(stage2 LoRA)** + NEW LoRA | ❌ "LoRA on LoRA on LoRA" |

**The API distinction to memorise** (notebook cell 20 markdown, video [44:47]–[45:38]):

> `get_peft_model()` → **Create** a new LoRA during training
> `PeftModel.from_pretrained()` → **Load** an already-trained LoRA for inference or further training

> **Correction:** the video leaves one thing unsaid that will bite you. **`merge_and_unload()` requires the model to be in full precision (fp16/bf16/fp32) to be lossless.** Merging an adapter into an *8-bit* base — which is exactly what the notebook does with `load_in_8bit=True` — re-quantises the merged weights back to int8, and the warning on screen says so: *"Merge lora module to 8-bit linear may get different generations due to rounding errors."* **The merged model is therefore not bit-identical to the instruction model you evaluated.** The production-safe pattern is: load the base in `torch.bfloat16`, merge, then optionally re-quantise for training. If you must stay in 4/8-bit through the merge, at minimum re-run your stage-2 evaluation on the merged model and confirm the numbers still hold before you start stage 3.
>
> A second, subtler point the video does not raise: **merging is irreversible and destroys the ability to serve the previous adapter separately.** If you need to A/B the SFT model against the aligned model, either keep an unmerged copy of the stage-2 model, or serve stage 3 as an *additional* adapter alongside the stage-2 adapter rather than merging. The merge approach the instructor teaches optimises for training simplicity; the adapter-stacking approach he warns against optimises for *deployability*. Modern practice: train stage 3 with the stage 2 adapter **merged** (as taught), but keep the pre-merge stage-2 checkpoint for rollback.

### 6.6 The DPOConfig — every knob the video sets

```python
# --- Cell 29–31 [51:34] -------------------------------------------------------------------
import os
os.environ["WANDB_DISABLED"] = "true"      # ⚠ deprecated in `trl`/`transformers` v5 (warning on screen)

dpo_args = DPOConfig(
    output_dir="./tinyllama-preference-alignment",  # where checkpoints + the final adapter land
    learning_rate=2e-5,                              # LoRA-scale LR
    per_device_train_batch_size=1,                    # 1 pair per device per step
    gradient_accumulation_steps=8,                    # effective batch = 1 * 8 = 8 pairs
    num_train_epochs=1,                               # one pass over 5 rows = 1 optimiser step
    beta=0.1,                                         # the KL/preference strength dial (§4.5)
    report_to=None,                                   # disable W&B/TensorBoard
    logging_dir=None,                                 # disable the log directory
    loss_type="sigmoid",                              # DPO's exact loss; "hinge" = SLiC-style
    remove_unused_columns=False                       # ⚠ REQUIRED — see below
)
```

**`remove_unused_columns=False` is mandatory and the video sets it without explaining why** [52:31]. The default is `True`, and the Hugging Face `Trainer` will drop any dataset column not present in the model's `forward()` signature. `prompt`, `chosen` and `rejected` are *not* model inputs — they are consumed by `DPOTrainer`'s internal collator. Leaving the default on deletes all three columns and the trainer fails with a confusing missing-column error. **This is the single most common DPO setup bug.**

**`report_to=None` + `logging_dir=None` + `WANDB_DISABLED`** — the instructor explicitly wants no experiment tracking *"as of now I don't want to be logged anything"* [52:24]. In production this is exactly backwards: you want W&B or TensorBoard on every preference run, because the failure modes (§4.9) are all *trends over time*, invisible in a final loss number.

### 6.7 The DPOTrainer

```python
# --- Cell 32 [53:03] ----------------------------------------------------------------------
trainer = DPOTrainer(
    model=pref_model_lora,      # the model with the FRESH adapter attached
    ref_model=None,             # ← see below
    args=dpo_args,
    train_dataset=dataset,      # the 5-row preference set
    processing_class=tokenizer, # TRL >= 0.13 renamed `tokenizer=` to `processing_class=`
    # you can pass data_collator if needed,
    # optionally eval_dataset etc.
)
```

**`ref_model=None` is the most misunderstood line in the notebook.** The instructor says: *"I'm not passing any reference model otherwise I can pass the reference model over here. Uh that is also one of the facility"* [53:38]–[53:47]. What `None` means depends on whether PEFT adapters are in play:

| Your setup | `ref_model=None` behaviour | Memory implication |
|---|---|---|
| Full fine-tune (no PEFT) | TRL **deep-copies the model at trainer init** to create the reference, and freezes it | You now hold **two full copies** — that copy is silent and is the #1 cause of OOM in DPO at full FT |
| PEFT/LoRA (this notebook) | TRL **disables the adapter** on the same base model and uses that as the reference | **One set of weights.** The reference is the frozen base + nothing. Nearly free |

This is the entire reason LoRA makes DPO cheap: **the reference comes for free because the base weights are frozen and the adapter is separable.** At full fine-tuning you must pay for a second model, either as a deep copy or as a separately-loaded frozen checkpoint.

> **Beyond the video:** if you are doing **full-parameter DPO** and memory is tight, pass `ref_model=AutoModelForCausalLM.from_pretrained(sft_ckpt, torch_dtype=torch.bfloat16, device_map="auto")` explicitly *loaded in a lower precision or sharded*, rather than letting TRL deep-copy. A deep copy inherits the training dtype and cannot be quantised after the fact. Alternatively use **SimPO** (`loss_type="simpo"` in recent TRL, or the `SimPOTrainer`) which needs no reference at all.

### 6.8 Training and the result

```python
# --- Cell 33 [54:21] ----------------------------------------------------------------------
trainer.train()
```

The recorded run output, verbatim:

```
TrainOutput(global_step=1, training_loss=0.66193026304245,
            metrics={'train_runtime': 5.1771,
                     'train_samples_per_second': 0.966,
                     'train_steps_per_second': 0.193,
                     'total_flos': 0.0,
                     'train_loss': 0.66193026304245,
                     'epoch': 1.0})
```

Read each number:

| Metric | Value | What it means |
|---|---|---|
| `global_step` | **1** | 5 examples ÷ (batch 1 × grad-accum 8) = ceil(5/8) = **1** optimiser step. The model took exactly **one** gradient step. |
| `training_loss` | **0.6619** | The DPO sigmoid floor is **log 2 = 0.6931**, reached when the margin is zero — which is *exactly* the state at initialisation, because policy = reference. 0.6619 is 0.031 below the floor, i.e. **one step of learning**. Expect a real run to reach 0.2–0.5. |
| `train_runtime` | **5.18 s** | One 8-bit forward/backward over 5 short pairs on a 1.1B model. |
| `train_samples_per_second` | **0.966** | ~1 sample/s. Slow because batch=1 and the first step includes CUDA/quantisation warmup. |
| `total_flos` | **0.0** | FLOPs counter not populated (8-bit bitsandbytes path). Harmless. |
| `epoch` | **1.0** | One full pass. |

**The arithmetic that makes this a demo rather than a training run:** 5 pairs × 1 epoch ÷ 8 accumulation = 1 step. **You cannot learn a preference direction from one gradient step.** The reason it produces a visibly different output is that the *base model changed between the two generations being compared* — the "instruction model" output shown earlier came from `checkpoint-3` loaded standalone, and the "preference model" output came from a different merged-and-adapted path. Part of the observed difference is the adapter, part is the 8-bit merge rounding, part is sampling temperature.

> **Beyond the video — what a real run looks like.** To make this a genuine reproduction you would: (a) use **2,000–10,000 pairs**, not 5; (b) run **1–3 epochs** so the optimiser takes 250–3,750 steps; (c) set `eval_dataset` and `eval_strategy="steps"` with `eval_steps=50`; (d) log the **implicit reward margin** (`beta * (logratio_chosen - logratio_rejected)`) — TRL writes `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies` and `logps/*` to the trainer log; (e) log **mean response length** to catch length bias; (f) enable `report_to="wandb"`. A 7B QLoRA DPO run on 5k pairs, 1 epoch, batch 1 × accum 16, on a single A100 40GB takes roughly **2–4 hours** and costs $5–15.

### 6.9 Testing the aligned model

```python
# --- Cells 47–52 [56:44] ------------------------------------------------------------------
model_path = "/content/tinyllama-preference-alignment/checkpoint-1"
preference_aligned_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16)
preference_aligned_model.to("cuda")

inputs = tokenizer(question, return_tensors="pt").to("cuda")
outputs = preference_aligned_model.generate(
    **inputs, max_new_tokens=100, temperature=0.8, top_p=0.9,
    do_sample=True, repetition_penalty=1.1
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The model dump in cell 49 confirms the adapter is live — `q_proj` and `v_proj` are `lora.Linear` with `r=8`, `lora_alpha` folded in, `lora_dropout=0.05`, and `k_proj`/`o_proj` are plain `Linear`. That is the video's `target_modules=["q_proj","v_proj"]` config, verifiable from the printout.

**Note the `dtype=torch.float16`.** `torch_dtype` was renamed to `dtype` in transformers 4.56+; the notebook uses the new name. If you are on an older transformers, use `torch_dtype=torch.float16` or you get a silent `float32` load and 2× the VRAM.

**The comparison protocol the instructor uses, and its flaw.** He generates from the instruction model and from the preference model on the same question and eyeballs the difference [55:45]–[58:13]. This is **a single sample at temperature 0.8** — it is an anecdote, not an evaluation. Two samples from the same model differ as much as samples from two different models. For anything real, see §12.

---

## 7. Hyperparameters & Configuration — Every Knob

### 7.1 The master table

| Param | What it does | Typical | Safe range | Too high → | Too low → | Framework flag |
|---|---|---|---|---|---|---|
| **beta** | DPO: scales the log-ratio inside the sigmoid. Higher = push harder away from the reference. | **0.1** | 0.01–0.5 | Over-optimisation, verbosity, degeneration, length blow-up | Loss pinned at 0.693, no learning, model unchanged | `DPOConfig(beta=)` |
| **learning_rate** | Step size | **2e-5** (LoRA) | LoRA: 5e-6–2e-5. Full FT: 5e-7–5e-6 | Loss spikes, degenerate output, reward collapse | No movement in loss over 200 steps | `DPOConfig(learning_rate=)` |
| **per_device_train_batch_size** | Pairs per device per forward | **1** | 1–8 (VRAM-bound) | OOM on long sequences | Noisy gradients (mitigated by accumulation) | `DPOConfig(per_device_train_batch_size=)` |
| **gradient_accumulation_steps** | Steps accumulated per optimiser update | **8** | 4–32 | Slower wall-clock; effective batch may be too large for 5k pairs | Noisy updates; effective batch < 8 destabilises DPO | `DPOConfig(gradient_accumulation_steps=)` |
| **num_train_epochs** | Passes over the data | **1** | 1–3 | **Overfitting from epoch 2–3**; margin explodes; style collapse | Underfit; loss stuck near 0.693 | `DPOConfig(num_train_epochs=)` |
| **loss_type** | Which preference loss | **"sigmoid"** | `sigmoid`, `hinge`, `ipo`, `kto_pair`, `bco_pair`, `sppo_hard`, `aot`, `apo_zero`, `apo_down`, `discopop`, `simpo` | — | — | `DPOConfig(loss_type=)` |
| **label_smoothing** | cDPO's ε — admits *ε* of labels are wrong | 0.0 | 0.0–0.2 | Under-uses clean data | Trains on known-noisy labels as if clean | `DPOConfig(label_smoothing=)` |
| **max_length** | Max total tokens for prompt+response | 1024 | 512–4096 | OOM; pads waste compute | **Truncates responses** → silent loss of the very content you are training on | `DPOConfig(max_length=)` |
| **max_prompt_length** | Max tokens for the prompt alone | 512 | 256–2048 | OOM | Prompts silently truncated from the front (the chat template is cut) | `DPOConfig(max_prompt_length=)` |
| **remove_unused_columns** | Drop dataset columns not in the model signature | **False (required)** | `False` | — | **`True` deletes prompt/chosen/rejected → training fails** | `DPOConfig(remove_unused_columns=False)` |
| **optim** | Optimiser | `adamw_torch` | `adamw_torch`, `paged_adamw_8bit`, `adafactor` | — | 8-bit/paged can be less stable on tiny models | `DPOConfig(optim=)` |
| **lr_scheduler_type** | LR schedule | `linear` | `linear`, `cosine` | — | — | `DPOConfig(lr_scheduler_type=)` |
| **warmup_ratio** | Fraction of steps spent ramping LR | 0.1 | 0.03–0.1 | Wastes steps at tiny LR | Early spikes / divergence | `DPOConfig(warmup_ratio=)` |
| **bf16** | bfloat16 mixed precision | `True` on Ampere+ | `True` or `False` | — | 2× memory if fp32 | `DPOConfig(bf16=True)` |
| **gradient_checkpointing** | Recompute activations in the backward pass | `True` when memory-bound | `True`/`False` | ~30% slower | OOM | `DPOConfig(gradient_checkpointing=)` |
| **ref_model** | The frozen KL anchor | `None` | `None` (PEFT) / explicit (full FT) | — | `None` at full FT silently deep-copies → OOM | `DPOTrainer(ref_model=)` |
| **r (LoRA)** | Adapter rank | **8** | 16–64 for preference work | Overfits small preference sets | Cannot express the behavioural shift | `LoraConfig(r=)` |
| **lora_alpha** | Adapter scale numerator | **16** | `r` to `2r` | Compounds with β → over-optimisation | Under-trained adapter | `LoraConfig(lora_alpha=)` |
| **target_modules** | Which layers get adapters | **q_proj, v_proj** | all linear layers | Marginal overfit | **Too few params to move behaviour** | `LoraConfig(target_modules=)` |
| **lr (ORPO)** | ORPO LR | 8e-6 (full) / 1e-5 (LoRA) | 3e-6–2e-5 | Degenerate repetition | No learning | `ORPOConfig(learning_rate=)` |
| **lambda (ORPO)** | Weight of the odds-ratio term vs the SFT term | **0.1** | 0.1–0.5 | Preference term dominates → base distribution damaged | Pure SFT; no preference learning | `ORPOConfig(lambda=)` |
| **num_generations (GRPO)** | G — group size per prompt | **8** | 4–64 | Cost scales linearly; VRAM for KV cache | High-variance baseline; zero-variance groups | `GRPOConfig(num_generations=)` |
| **temperature (GRPO/PPO rollouts)** | Sampling temp for rollouts | 1.0 | 0.8–1.2 | Incoherent rollouts, all rewards near zero | No diversity → zero-variance groups | `GRPOConfig(temperature=)` |
| **kl_coef (PPO/GRPO)** | β in the KL penalty | 0.04–0.1 | 0.001–0.2 | Policy cannot move | Reward hacking, collapse | `PPOConfig(kl_coef=)` |
| **cliprange / epsilon** | PPO's importance-ratio clip | 0.2 | 0.1–0.3 | Too few updates accepted → no learning | Unstable updates, divergence | `PPOConfig(cliprange=)` |
| **gamma / lam (PPO)** | Discount and GAE λ | 1.0 / 0.95 | γ=1.0; λ=0.9–0.95 | — | Biased advantage estimates | `PPOConfig(gamma=, lam=)` |

### 7.2 The knobs that interact (read this before you tune anything)

**β ↔ epochs.** A long run at high β ≈ a short run at low β. If your eval is flat, **lower β before you add epochs** — more epochs on a small preference set overfits, whereas lowering β explores more of the reward surface while the early-stopping control (capability evals) still protects you.

**β ↔ learning rate.** Both control how fast the policy leaves π_ref. Raising LR *and* lowering β is a double-speed instruction and reliably over-optimises. Raise one, hold the other.

**β ↔ label_smoothing.** cDPO's ε caps the achievable margin at roughly `log((1−ε)/ε)`. With ε=0.1 the margin target is ~2.2 nats; with ε=0.3 it is ~0.85 nats. **If you set both a strong β and a large ε you have capped your own learning.** Set ε from your measured label error rate and leave β at 0.1.

**r ↔ lora_alpha ↔ β.** The effective adapter scale is `alpha/r`, and it multiplies the same update that β scales. `alpha=2r` plus `β=0.1` is a different operating point from `alpha=r` plus `β=0.1`. Pick `alpha=r` when you have no strong reason, and tune β.

**max_length ↔ truncation.** This is the silent one. If `max_length=1024` and your `chosen` responses average 1,400 tokens, TRL truncates them — and it truncates *differently* for chosen and rejected if their lengths differ. You are then training a preference over truncated, possibly-identical prefixes. **Diagnostic:** log the 95th percentile of `len(prompt+chosen)` before training and set `max_length` above it.

**num_generations ↔ temperature.** GRPO's advantage is a z-score within the group. If temperature is too low, all G samples are identical, std = 0, and the advantage is 0/0. If temperature is too high, all rewards are 0 for a verifiable task and the group has zero variance too. **The group must be diverse *and* the rewards must not be all-equal.** For RLVR tasks, a good diagnostic is the fraction of prompts with nonzero reward variance per batch — aim for >30%; below that, the prompt is too hard or too easy for the current policy.

**PPO's clip range ↔ KL coefficient.** ε bounds how far one update moves the policy; β bounds how far the whole run moves it. Setting both large is safe but slow; both small is fast and unstable. The published stable region is ε=0.2, β=0.02–0.05.

---

## 8. Decision Framework — When To Use / When NOT To Use

### 8.1 The decision tree

```
START: Do you have a behaviour problem (tone, format, refusal, verbosity, safety)?
│
├─ NO, it's a knowledge/freshness problem ──────────────────► STOP. Use RAG (CS-04).
│                                                              Alignment will not add facts.
│
└─ YES
   │
   ├─ Q1: Is the model already instruction-tuned (SFT'd)?
   │  ├─ NO ──► Do SFT first (CS-13). Alignment on a base model degrades it (§4.2).
   │  └─ YES ─► continue
   │
   ├─ Q2: Can the reward be COMPUTED by a program? (unit tests pass, math answer matches,
   │       JSON validates, compiler accepts, regex matches, tool call succeeds)
   │  ├─ YES ──► **GRPO / RLVR** (§4.6.10)
   │  │           No reward model. No preference data. No human labels.
   │  │           Cost: G× generation. Memory: 2 models.
   │  │           This is the reasoning-model recipe. Start here if it applies.
   │  └─ NO ──► continue
   │
   ├─ Q3: Do you have PAIRS {prompt, chosen, rejected}?
   │  ├─ NO, you have unpaired thumbs up/down ──► **KTO** (§4.6.6)
   │  │                                            Pointwise, prospect-theory-weighted.
   │  │
   │  └─ YES ─► continue
   │
   ├─ Q4: What is your memory budget for the largest model you must train?
   │  │
   │  ├─ Fits ~1 model only (single consumer GPU, or 7B on 24 GB) ──► **ORPO** (§4.6.9)
   │  │     No reference model. SFT + preference in ONE stage.
   │  │     You can even skip the separate SFT run.
   │  │     λ = 0.1–0.5. LoRA r=16–32 on all linear layers.
   │  │
   │  ├─ Fits 2 models (7B LoRA on 40–80 GB) ──► **DPO** (§4.6.3) — the default.
   │  │     β = 0.1, LR 2e-5 (LoRA), 1–3 epochs.
   │  │     If your DPO model gets verbose → **SimPO** (no reference, length-normalised).
   │  │     If your pairs are near-deterministic and DPO overfits → **IPO**.
   │  │     If your IAA is low → **cDPO** with label_smoothing = 1 − label precision.
   │  │
   │  └─ Budget is not the constraint ─► continue to Q5
   │
   ├─ Q5: Is the reward a BLACK BOX that is not a pairwise preference, and do you need
   │       the policy to EXPLORE beyond a fixed dataset?
   │  ├─ YES ──► **PPO / RLHF** (§4.6.1)
   │  │           4 models. Expect 3–8× the wall-clock and 10× the debugging of DPO.
   │  │           Only worth it for exploration or a non-differentiable custom reward.
   │  └─ NO ──► You have no reason to use PPO. Go back to Q4 and pick DPO or ORPO.
   │
   └─ ALWAYS, regardless of path:
      • Run a capability suite before and after (alignment tax, §4.8)
      • Select checkpoints on human/gold-judge win rate, never on RM score (§4.9.3)
      • Log response length — it is your earliest reward-hacking alarm
      • Keep the pre-alignment checkpoint for rollback (§16)
```

### 8.2 The modern split: verifiable vs subjective

This is the single most important routing rule in the 2026 landscape, and it is a `> **Beyond the video:**` addition — the video's method list predates the reasoning-model era.

| | **Verifiable-reward domains** | **Subjective domains** |
|---|---|---|
| **Examples** | Math, code generation, SQL, logic puzzles, tool-call correctness, schema-valid JSON, unit-test-passing refactors, formal proofs | Tone, style, helpfulness, safety, refusal calibration, brand voice, concision, persona, summarisation quality, empathy |
| **Reward** | A program. You can *check* the answer. | A judgement. Only a human or a model can *rank* answers. |
| **Signal type** | Binary/continuous score per response | Pairwise preference |
| **Method** | **GRPO / RLVR** (or PPO + verifier) | **DPO / ORPO / SimPO** |
| **Reward model needed?** | **No** | Yes, if you go the PPO route; no for DPO/ORPO |
| **Preference data needed?** | **No** | Yes |
| **Why** | A verifier is exact, cheap, and unhackable in the limit — you cannot fool a compiler into accepting invalid syntax. So use it directly as the reward. | There is no program that answers "is this answer more helpful?" So you must learn the preference. DPO learns it directly from pairs. |
| **Dominant failure** | Zero-variance groups on prompts that are too hard or too easy; reward is sparse and all-or-nothing | Length bias, sycophancy, judge-bias transfer |
| **Canonical result** | DeepSeek-R1 (GRPO + rule-based rewards, no RM, no value model) | InstructGPT, Llama-2-Chat, Zephyr (DPO), Phi-3 (ORPO-style single stage) |

**The rule, stated for an interview:** *if you can write a function that returns a number for a response, use GRPO with that function and skip the reward model and the preference data entirely. If you cannot, use DPO on pairs.* Most real products need **both**: a GRPO pass for the reasoning/tool-use core and a DPO pass for the tone and safety shell. Run the GRPO pass first, because tone alignment on top of a model that cannot do the task is wasted money.

### 8.3 STOP conditions — signals this is the wrong tool

| STOP signal | What is actually true | Do instead |
|---|---|---|
| Your evaluation is "the outputs look better" | You have no metric; you will not know if the run worked or which checkpoint to ship | Build the eval set first (§12) |
| You have < 500 preference pairs and no plan to get more | Under 500 pairs you will mostly fit noise; DPO needs ~1k minimum, ~5k for a real shift | Collect more, or use RLAIF to bootstrap, or do better SFT instead |
| Your IAA is below 65% | The preference signal does not exist in your data | Rewrite guidelines; retrain annotators; measure again |
| The model is already correct-but-rude and you have no SFT budget | You are about to do §4.2 | Do the SFT stage first; it is cheaper than a failed alignment run |
| The problem is hallucination | Alignment does not add factual grounding; it can *increase* confident fabrication | RAG + a grounding/citation eval; alignment only for calibration behaviour |
| The prompt already solves it | A system prompt with 3 few-shot exemplars costs $0 and ships today | Try prompting first; it is a 30-minute experiment |
| Your reward is "user engagement" | You will train a sycophantic, addictive, click-maximising model | Choose a metric you are willing to have maximised (§4.9) |
| You need the model to know last week's prices | Preference data is about behaviour, not facts | RAG + a tool call |
| You have one A100 and a 70B model and you want PPO | 4 × 70B × 16 bytes ≈ 1.1 TB of weights before optimiser states | LoRA + DPO, or a smaller model, or a bigger budget |
| Nobody can say which of two responses is better | There is no preference to learn | Redefine the task; a preference model cannot exceed the clarity of its labels |

---

## 9. Pros · Cons · Limitations · Failure Modes

### 9.1 Pros — what preference alignment buys you

| Pro | Mechanism | Evidence / number |
|---|---|---|
| **Behaviour without prompt tokens** | Behaviour is in the weights, so it costs zero per-call tokens and cannot be prompt-injected away | A 200-token style instruction on every call, at 1M calls/month, is ~$1,200/month at a mid-tier input price |
| **Better refusal calibration** | The model learns the *boundary* of acceptable requests, not a keyword list | False-refusal rate is directly controllable via the preference mix |
| **Tone and format consistency** | A single behavioural point instead of a distribution of plausible answers | Format-adherence rate on structured output rises from ~70% (prompted) to ~95%+ (aligned), then needs guardrails anyway |
| **Cheaper per unit of improvement than SFT** | Ranking k answers is O(k) annotator effort and yields O(k) or O(k²) comparisons | Same improvement costs ~5–10× fewer annotation hours than writing demonstrations |
| **Uses data you already have** | Thumbs-up/down, regenerations, A/B outcomes | The KTO path needs no pairs |
| **Composable with LoRA** | Adapters are separable, so alignment is a shippable artifact | One base + N adapters; rollback is a config change |
| **Measurable** | Win rate is a directly interpretable product metric | "62% win over our previous model on 500 held-out prompts" is a ship decision |

### 9.2 Cons — what it costs

| Con | Magnitude |
|---|---|
| **Annotation cost** | $3–$10/pair, so $15k–$100k for a real run |
| **The alignment tax** | Typically 1–5% on out-of-domain capability benchmarks; can be 10%+ at aggressive β / long runs |
| **Over-optimisation** | True quality peaks at KL ≈ 5–20 nats and then *declines* — you must early-stop on an independent metric |
| **Evaluation is hard and expensive** | Human eval at 500 comparisons × 3 raters ≈ $1,500 per checkpoint evaluated |
| **Fragile to data quality** | 15% mislabelled pairs measurably caps the achievable win rate |
| **Not additive with facts** | Cannot fix hallucination; can worsen confident fabrication |
| **Reproducibility** | RL methods (PPO/GRPO) are seed-sensitive and hardware-sensitive; DPO-style methods are reproducible but data-sensitive |

### 9.3 Hard limitations — things no amount of tuning fixes

1. **You cannot learn a preference your annotators cannot articulate.** No IAA, no signal.
2. **You cannot exceed your preference data's ceiling with an offline method.** DPO interpolates the pairs you gave it. GRPO can exceed it (it explores), which is why verifiable-reward RL produces genuinely new capability and DPO produces *better-calibrated existing* capability.
3. **You cannot make a model more truthful by rewarding confidence.** Reward the *calibration*, not the confidence, or you get the opposite.
4. **You cannot align away a capability the base model lacks.** Alignment selects among behaviours the model can already produce. A 1.1B model will not become a research scientist.
5. **You cannot verify an aligned model once, ship it, and forget it.** Behaviour drifts with prompt distribution, and preference data ages.

### 9.4 Silent failure modes — looks fine, is broken

| Silent failure | Why it looks fine | How to detect it |
|---|---|---|
| **Chosen/rejected swapped** | The loss falls smoothly; the model produces output | Win rate *drops* while loss drops. **Detection: evaluate the model explicitly on 20 held-out pairs and check the implicit margin is positive.** Sanity-check by flipping a known pair at data-prep time |
| **`remove_unused_columns=True`** | No — this one is loud (it errors) | n/a |
| **Truncation to `max_length`** | Loss falls; the run completes | Compute p95 of `len(prompt+chosen)` and `len(prompt+rejected)` before training. If truncated, you are training on identical prefixes |
| **The reference model is accidentally trainable** | Loss falls, metrics look plausible | The margin cannot grow if both sides train together, so loss floors near 0.693. Check that `ref_model` is in `eval()` with `requires_grad=False` |
| **Length bias** | Win rate improves | Log mean response length per checkpoint. If it rises monotonically, you are measuring length |
| **Judge-bias transfer** | Win rate improves *against your own judge* | Win rate against a *different-family* judge, or against humans, tells a different story |
| **Capability regression** | The preference metric is great | Run MMLU/GSM8K/IFEval before and after. **If you do not run a capability suite, this is invisible by construction** |
| **The 8-bit merge rounding (§6.5)** | Everything runs | Re-evaluate the merged stage-2 model and confirm it still matches the stage-2 numbers |
| **Evaluation data leakage into preference data** | Win rate is suspiciously high | Deduplicate prompts between preference data and eval sets. A 15-point win rate above expectation usually means leakage |
| **Sampling variance mistaken for improvement** | You compared one generation each | Generate 20 samples per prompt per model, or use a proper pairwise protocol (§12) |

---

## 10. Exceptions, Edge Cases & Gotchas

1. **`hh-rlhf` has no `prompt` column.** The prompt is the shared prefix of `chosen` and `rejected`. TRL can infer it, but the inference is fragile — materialise an explicit `prompt` column yourself. (§4.3.3)
2. **The reference model is free with LoRA and expensive without.** `ref_model=None` means "deep-copy" at full FT and "disable the adapter" with PEFT. Same flag, opposite memory implication. (§6.7)
3. **`remove_unused_columns` must be `False`.** Non-negotiable, and it is the #1 setup error. (§6.6)
4. **A DPO loss below 0.693 does not mean it is working.** It means the margin is positive, which is true from a random *or* a correctly-initialised model. Check the *direction* of the margin on held-out pairs.
5. **ORPO can be run without a preceding SFT stage.** That is its whole point — `L = L_SFT + λ·L_OR` contains the SFT term. Running ORPO on a base model is legitimate where running DPO on a base model is not (§4.2), *provided* your ORPO `chosen` set is a real instruction dataset. This is the one exception to the three-stage rule, and it is a good interview answer.
6. **IPO's finite margin target is a feature in some datasets and a bug in others.** Near-deterministic pairs (correct/incorrect) → IPO wins. Graded pairs (good/better) → IPO underfits. (§4.6.4)
7. **KTO requires a reference model** in its standard formulation, even though it needs no pairs. The reference point in the prospect-theory utility is KL-based. People assume "unpaired ⇒ no reference"; wrong.
8. **SimPO has no β but has a γ, and γ=0 is not the right default.** γ is a target margin (typically 0.5–2.0 in the published recipe); leaving it at 0 removes the margin term entirely.
9. **GRPO's advantage is per-response, not per-token.** This is a genuine departure from PPO and the reason no value model is needed — but it means GRPO cannot express "this part of the reasoning was good, that part was bad". For long CoT with partial credit you need per-step rewards, which is a different method (process reward models; see §4.6.1's relatives).
10. **GRPO with G=1 degenerates.** With one sample the group mean is the sample itself and advantage is 0/0. GRPO needs G ≥ 4, realistically 8–16.
11. **PPO's reward normalisation across a batch of *different prompts* is a bug, not a feature.** Normalise per-prompt; cross-prompt normalisation makes reward magnitudes incomparable and is a common cause of "PPO trains but produces nonsense".
12. **`processing_class` vs `tokenizer`.** TRL ≥ 0.13 renamed the `DPOTrainer` argument. Old tutorials pass `tokenizer=`; new TRL still accepts it with a deprecation path in some versions and errors in others. Use `processing_class=`.
13. **`torch_dtype` vs `dtype`.** Renamed in transformers 4.56. Using the old name on a new version warns and may silently load fp32 — 2× VRAM, and on a 40 GB card that is the difference between fitting and not.
14. **`load_in_4bit`/`load_in_8bit` are deprecated.** Use `BitsAndBytesConfig` + `quantization_config=`. (§4.7.3)
15. **`WANDB_DISABLED` is deprecated.** Use `report_to="none"`. And in production, use `report_to="wandb"` — the failure modes are trends, not endpoints. (§6.6)
16. **A preference dataset built by an LLM judge will contain rows where `chosen == rejected`.** The judge's output parsing failed. Assert against it. (§6.4)
17. **Merging a LoRA into a quantised base is lossy.** Re-quantisation changes outputs; the warning is on screen in the video. (§6.5, §4.7.3)
18. **DPO on a *chat* model needs the chat template, and TRL's handling changed.** In TRL ≥ 0.13, the trainer applies the tokenizer's chat template when the dataset has no pre-formatted `prompt`. If your tokenizer has no chat template set, TRL silently falls back to raw concatenation — and your prompts are formatted differently at train and inference time. **Always set `tokenizer.chat_template` explicitly and log one fully-formatted training example.**
19. **The `beta` meaning inverts between PPO and DPO.** In PPO, high β = stay near the reference. In DPO, high β = push away from it. Copying a β value between the two is a real and common error. (§4.5)
20. **Alignment does not compose additively across runs.** DPO then ORPO then DPO again does not give you three times the alignment; later runs mostly undo or overfit earlier ones. Run one alignment stage, evaluate, and re-run from the SFT checkpoint with better data rather than stacking.

---

## 11. Cost, Compute & Memory

### 11.1 The cost model in one table

For a **7B model**, one alignment run, 5,000 pairs, 1 epoch, single-turn responses averaging 300 tokens:

| Method | Hardware | Wall-clock | Cloud cost (A100 40GB @ ~$1.50/hr, or H100 @ ~$2.50/hr) | Notes |
|---|---|---|---|---|
| **ORPO, QLoRA** | 1× RTX 4090 24GB / A10G | 3–5 h | **$2–8** (or $0 on your own GPU) | 1 model, no reference, SFT term included |
| **ORPO, LoRA** | 1× A100 40GB | 2–3 h | **$3–5** | |
| **DPO, QLoRA** | 1× A100 40GB | 4–6 h | **$6–10** | Reference is free (adapter-off) |
| **DPO, LoRA** | 1× A100 40GB | 3–5 h | **$5–8** | |
| **DPO, full FT** | 2–3× H100 80GB | 6–10 h | **$30–75** | Reference deep-copy doubles the weights |
| **SimPO, LoRA** | 1× A100 40GB | 2–4 h | **$3–6** | No reference pass at all |
| **GRPO, LoRA, G=8** | 1× H100 80GB | 12–40 h | **$30–100** | Rollout-dominated; scales ~linearly in G and generation length |
| **PPO, LoRA (4 models)** | 1–2× H100 80GB | 24–72 h | **$60–180** | Plus a separate RM training run |
| **RM training (7B, full FT)** | 2× H100 80GB | 6–12 h | **$30–60** | Required before PPO |
| **PPO end-to-end (RM + PPO)** | 2× H100 80GB | 30–84 h | **$90–240** | The real cost of the 4-model path |
| **Rejection sampling, n=8 + SFT** | 1× A100 40GB | 8–16 h | **$12–25** | 8× generation + one SFT run |

**The headline: DPO/ORPO on LoRA costs $3–10 in compute for a 7B model.** The annotation is the real bill. Teams routinely spend 100× more on the data than on the GPUs, and then skimp on the evaluation — which is the actual mistake.

### 11.2 Worked example — the video's own run

| Item | Value |
|---|---|
| Model | TinyLlama-1.1B-intermediate-step-1431k-3T |
| Precision | int8 base + LoRA r=8 (q_proj, v_proj) |
| Trainable params | **1,126,400** (0.10% of 1.1B) |
| Preference pairs | **5** |
| Epochs | 1 |
| Effective batch | 1 × 8 accumulation |
| Optimiser steps | **1** |
| Wall-clock | **5.18 s** |
| Final loss | **0.6619** (floor = 0.6931) |
| Hardware | Google Colab T4 (free tier) |
| Cost | **$0** |

### 11.3 Worked example — a realistic production run

*Scenario: align a 7B model for a customer-support assistant. Tone, refusal calibration, and concision. No verifier.*

| Line item | Quantity | Unit | Total |
|---|---|---|---|
| Preference pairs needed | 6,000 | — | — |
| Pairs from human annotation | 1,500 (25%) | $4.00 | $6,000 |
| Pairs from RLAIF + human audit | 4,500 (75%) | $0.06 | $270 |
| Guideline writing + annotator onboarding | 40 h | $60/h | $2,400 |
| IAA control set (double-annotated 10%) | 600 pairs | $4.00 | $2,400 |
| **Data subtotal** | | | **$11,070** |
| DPO LoRA training (1× A100 40GB) | 5 h | $1.50/h | $8 |
| Hyperparameter sweep (4 configs × 3 h) | 12 h | $1.50/h | $18 |
| **Compute subtotal** | | | **$26** |
| Human eval, 500 comparisons × 3 raters × 4 checkpoints | 6,000 judgements | $0.60 | $3,600 |
| Capability suite runs | 4 checkpoints | — | $20 |
| **Evaluation subtotal** | | | **$3,620** |
| **TOTAL** | | | **≈ $14,700** |

**The lesson in the arithmetic: compute is 0.2% of the cost. Data is 75%. Evaluation is 25%.** Any decision process that starts by choosing a GPU and ends by choosing a method is optimising the wrong 0.2%.

> **Beyond the video:** the biggest under-appreciated line above is the **evaluation** cost. It is common to see teams spend $15k on data and then evaluate on 50 prompts with one rater. That is not an evaluation; it is a vibe. 500 comparisons × 3 raters gives you a 95% CI of roughly ±4 percentage points on a win rate near 60% — which is the minimum precision needed to distinguish a real improvement from noise. Budget for it up front or you will not know whether to ship.

---

## 12. Evaluation — How To Know It Worked

### 12.1 The metric stack, and how each one lies

| Metric | What it measures | Cost | How it lies |
|---|---|---|---|
| **Training loss** | Nothing you care about | free | Falls for swapped labels, for a flat margin, for overfitting. **Only useful as a "did it move" check against the 0.693 floor** |
| **Preference accuracy on held-out pairs** | P(model's implicit reward ranks `chosen` above `rejected`) | free | Rises during over-optimisation while true quality falls. Only meaningful on a *held-out* split — which requires splitting your pairs |
| **Implicit reward margin** | `β·(logratio_chosen − logratio_rejected)` | free | Same as above; watch for it growing without bound |
| **RM score** | The RM's opinion | free | **This is the metric being gamed (Goodhart).** Diagnostic only; never a selection criterion |
| **Win rate vs SFT baseline (human)** | The real thing | $0.50–$2/judgement | Expensive; rater fatigue; needs a blinded, randomised, position-swapped protocol |
| **Win rate vs SFT baseline (LLM judge)** | A cheap proxy for the above | ~$0.01/judgement | **Length bias, self-preference bias, position bias, style bias.** Correct with LC and position swapping |
| **MT-Bench** | Multi-turn quality on a 1–10 GPT-4 rubric | cheap | Saturated above ~8.5; does not discriminate modern models |
| **AlpacaEval 2 (raw)** | LLM-judged win rate vs a reference model | cheap | **Verbose models win ~10 points more than they deserve** |
| **AlpacaEval 2 LC (length-controlled)** | Win rate with length regressed out | cheap | The honest version. Report this one |
| **Arena Elo** | Human pairwise votes across many models | free (public) | Only available for your model if you submit it; Elo is not linear in quality |
| **Capability suite (MMLU/GSM8K/IFEval/HumanEval)** | The alignment tax | cheap | Measures the tax, tells you nothing about alignment quality. **You must run both or you are flying blind** |
| **False-refusal rate on benign prompts** | Refusal calibration | cheap | The metric that catches the over-refuser failure |
| **Mean response length** | Length bias, and a *leading indicator* of reward hacking | free | Not a quality metric — a **canary** |
| **KL to reference** | How far you moved | free | Not quality; but a large KL with a small win-rate gain means you paid a lot for nothing |

### 12.2 LC win rate, explained properly

AlpacaEval 2's raw win rate has a documented flaw: the judge (GPT-4-turbo) prefers longer outputs, so a model that simply writes more wins more. AlpacaEval 2's **length-controlled (LC) win rate** fits a logistic regression where the outcome is the judge's verdict and the predictors include the *length difference* between the two responses. It then reports the win rate at **zero length difference** — i.e. the counterfactual "if both models produced equally long responses, who would win?"

**Practical consequences:**

- If your raw win rate is 65% and your LC win rate is 52%, **you built a verbosity model.**
- A model can have a *lower* raw win rate and a *higher* LC win rate than another — in which case it is better in substance and worse in marketing.
- The same correction applies to your own internal judge. If you are not length-controlling, you are likely selecting the most verbose checkpoint.

**Cheap internal version.** Compute mean length for each candidate checkpoint. If length and win rate move together (correlation > 0.7 across checkpoints), your win-rate evaluation is contaminated. Fix by either length-controlling your judge, adding length to your judge's rubric explicitly ("do not reward length; prefer the more concise response when both are correct"), or trimming both responses to equal token counts before judging.

### 12.3 The minimal eval harness

```python
"""
Minimal preference-alignment evaluation harness.
Runs on held-out pairs + a capability suite. Reports the four numbers that matter.
"""
import torch, json, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

def implicit_margin(model, ref_model, tokenizer, prompt, chosen, rejected, beta):
    """DPO's own quantity: beta * (logratio_chosen - logratio_rejected)."""
    def logprob(m, text, ctx):
        ids = tokenizer(ctx + text, return_tensors="pt", truncation=True,
                        max_length=1024).input_ids.to(m.device)
        ctx_len = tokenizer(ctx, return_tensors="pt", truncation=True,
                            max_length=1024).input_ids.shape[1]
        with torch.no_grad():
            logits = m(ids).logits[:, :-1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return tok_lp[:, ctx_len - 1:].sum().item()      # response tokens only

    lc = logprob(model, chosen,   prompt) - logprob(ref_model, chosen,   prompt)
    lr = logprob(model, rejected, prompt) - logprob(ref_model, rejected, prompt)
    return beta * (lc - lr)

def evaluate(model_path, ref_path, pairs_path, tokenizer_path, beta=0.1, n=200):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    ref   = AutoModelForCausalLM.from_pretrained(ref_path,   dtype=torch.bfloat16).cuda().eval()
    pairs = [json.loads(l) for l in open(pairs_path, encoding="utf-8")][:n]

    margins = [implicit_margin(model, ref, tokenizer,
                               p["prompt"], p["chosen"], p["rejected"], beta) for p in pairs]

    print(f"pairs evaluated        : {len(margins)}")
    print(f"preference accuracy    : {sum(m > 0 for m in margins)/len(margins):.3f}   <- want > 0.60, and > baseline")
    print(f"mean implicit margin   : {statistics.mean(margins):+.3f}   <- want positive and growing, then flattening")
    print(f"margin p10 / p90       : {sorted(margins)[len(margins)//10]:+.3f} / {sorted(margins)[9*len(margins)//10]:+.3f}")
    # A p10 below zero means 10%+ of held-out pairs got WORSE - the classic over-optimisation signature.
    return margins

# The four numbers to record per checkpoint, in one row of your experiment table:
#   1. preference accuracy on HELD-OUT pairs   (did the preference direction generalise?)
#   2. mean implicit margin                     (how far did it move?)
#   3. mean generation length                   (canary: is this a verbosity model?)
#   4. MMLU / GSM8K / IFEval delta vs SFT       (the alignment tax)
```

**The evaluation protocol, end to end:**

1. **Split your pairs** 90/10 into train/held-out *before* training, and never let the held-out prompts appear in training.
2. **Generate responses** from each candidate checkpoint on a *separate* prompt set (200–500 prompts), with the same decoding parameters, 1 sample per prompt minimum, 4–8 preferred.
3. **Run the reference-free checks** first: implicit margin, held-out preference accuracy, mean length, capability suite. These are free and catch 80% of failures.
4. **Run the pairwise judge** with position swapping: judge (A,B) and (B,A) for each pair and count a win only if the verdict is consistent. Position bias flips 15–25% of verdicts otherwise.
5. **Length-control** the judge verdicts, or measure the length correlation across checkpoints.
6. **Human-eval a 200-comparison sample** of the top-2 checkpoints. Blinded, randomised order, 3 raters, majority vote, report Krippendorff's α on the human verdicts too.
7. **Select the checkpoint where the human win rate is highest**, subject to the capability delta being worse than −2%. Both conditions, always.

> **Beyond the video:** the most valuable single eval number is the **p10 of the implicit margin on held-out pairs**. Mean margin tells you the model moved; p10 tells you whether moving *broke* a tenth of your distribution. Over-optimisation almost always shows up at the tail first — the mean keeps rising while the bottom decile goes negative. This is the cheap, free, per-checkpoint early-warning signal that almost nobody computes.

---

## 13. Comparison Tables

### 13.1 The master comparison matrix

| Method | Reward model? | Reference model? | Value model? | **Models in memory** | Data format | Stability | Typical compute (7B) | When to use |
|---|---|---|---|---|---|---|---|---|
| **RLHF + PPO** | ✅ yes | ✅ yes | ✅ yes | **4** | Prompts + RM preference data | ★★☆☆☆ Fragile, 10 interacting knobs | 30–84 h, 2×H100 | Black-box non-pair reward; need exploration; large lab |
| **Reward modelling alone** | *is* the RM | ❌ | ❌ | 1 (train) / 1 (serve) | `{prompt, chosen, rejected}` | ★★★★☆ Standard supervised | 6–12 h, 2×H100 | You need best-of-n, a dense scorer, or PPO downstream |
| **DPO** | ❌ | ✅ yes | ❌ | **2** | `{prompt, chosen, rejected}` | ★★★★★ Very stable | 3–5 h, 1×A100 | **The default.** Pairs available, no verifier |
| **IPO** | ❌ | ✅ | ❌ | 2 | `{prompt, chosen, rejected}` | ★★★★★ | 3–5 h, 1×A100 | Near-deterministic pairs; DPO overfitting |
| **cDPO** | ❌ | ✅ | ❌ | 2 | `{prompt, chosen, rejected}` + known noise rate | ★★★★★ | 3–5 h, 1×A100 | Low IAA / noisy labels |
| **KTO** | ❌ | ✅ | ❌ | 2 | `{prompt, completion, label}` — **unpaired** | ★★★★☆ | 3–5 h, 1×A100 | Thumbs-up/down telemetry, no pairs |
| **SLiC-HF** | ❌ | ❌ (hinge on raw log-probs) | ❌ | 1–2 | `{prompt, chosen, rejected}` | ★★★★★ Hinge = hard to blow up | 3–4 h, 1×A100 | Want maximum stability; λ cross-entropy anchors it |
| **SimPO** | ❌ | **❌ no reference** | ❌ | **1** | `{prompt, chosen, rejected}` | ★★★★☆ | 2–4 h, 1×A100 | Memory-bound; DPO made you verbose |
| **ORPO** | ❌ | **❌ no reference** | ❌ | **1** | `{prompt, chosen, rejected}` (+ plain SFT data) | ★★★★★ Very stable | 3–5 h (incl. SFT) | **Single GPU.** Can replace SFT + DPO with one run |
| **GRPO** | ❌ (or ✅ if non-verifiable) | ✅ yes | **❌ no value model** | **2** | Prompts + a **reward function** | ★★★☆☆ Group variance issues | 12–40 h, 1×H100 | **Verifiable reward** (math/code/tools). RLVR |
| **RLAIF** | ✅ (usually) | ✅ | ✅ | same as host method | Prompts + a judge | inherits host | inherits host | Scaling labels; no human budget |
| **Constitutional AI** | ✅ (stage 2) | ✅ | ✅ | same as host | Prompts + a constitution document | inherits host | inherits host | Auditable values; harmlessness |
| **Rejection sampling / best-of-n** | optional scorer | ❌ | ❌ | **1** | Prompts (+ any scorer) | ★★★★★ | n× generation + SFT | No labels at all; bootstrap; inference-time lift |
| **RAFT** | ❌ | ❌ | ❌ | 1 | Prompt + docs + answer | ★★★★★ | ~SFT | RAG-aware fine-tuning (different task; name collision) |
| **SPIN** | ❌ | ✅ (previous iteration) | ❌ | 2 | Your existing SFT data | ★★★★☆ | k × DPO | No new labels; small clean SFT set |

### 13.2 Head-to-head: PPO vs DPO vs ORPO vs GRPO

| Dimension | PPO (RLHF) | DPO | ORPO | GRPO |
|---|---|---|---|---|
| **Models in memory** | 4 | 2 (1 with LoRA ref) | **1** | 2 |
| **VRAM, 7B LoRA** | 60–85 GB | 37–47 GB | **23–33 GB** | 40–60 GB |
| **VRAM, 7B QLoRA** | 25–40 GB | 12–20 GB | **11–19 GB** | 20–35 GB |
| **Generates during training?** | Yes | **No** | **No** | Yes, G× per prompt |
| **Wall-clock per step** | 8–25× DPO | 1× | 0.9× | 5–30× |
| **Hyperparameter count** | ~10 interacting | 2 (β, LR) | 3 (λ, LR, epochs) | 5 (G, temp, β, clip, LR) |
| **Reproducible?** | Poorly | Well | Well | Moderately |
| **Needs preference pairs?** | For the RM only | Yes | Yes | **No** |
| **Needs a reward model?** | Yes | No | No | **No** |
| **Can explore beyond the dataset?** | Yes | No | No | Yes |
| **Typical win-rate gain over SFT** | +15–25 pts (with a good RM, at scale) | +8–15 pts | +5–12 pts | +20–40 pts **on verifiable tasks** |
| **Alignment tax** | Moderate, mitigated by PPO-ptx | Low–moderate | Low | Low on general, high if reward is narrow |
| **Failure mode** | Reward hacking, instability | Length bias, overfitting on small sets | Under-shifts; inherits SFT length prior | Zero-variance groups |
| **Time to first working run** | Days–weeks | **Hours** | **Hours** | Days |
| **Who uses it** | Frontier labs, reasoning RL with custom rewards | Most product teams, most papers | Single-GPU practitioners, small teams | Reasoning models, code, math, agents |
| **Verdict** | Only if you must | **Default** | **Cheapest** | **Best when verifiable** |

### 13.3 Head-to-head: data efficiency

Quality of the aligned model as a function of pairs available (approximate, 7B, style/tone task):

| Pairs | DPO | ORPO | PPO | GRPO (verifiable task) |
|---|---|---|---|---|
| 100 | Little change; mostly noise | Little change | No signal | n/a |
| 500 | Detectable style shift | Small shift | Nothing usable | n/a |
| 1,000 | Clear shift, some overfit | Clear shift | Unstable | n/a |
| 5,000 | **Production-usable** | **Production-usable** | Improving | n/a |
| 20,000 | Diminishing returns | Diminishing returns | **Where PPO starts to win** | — |
| 50,000+ | Saturated | Saturated | Best achievable | — |
| Prompts + a verifier (any count) | — | — | — | **+20–40 pts on math/code** |

**Read the crossover:** PPO needs roughly **20,000+ pairs** and a well-trained RM before it beats DPO. Below that, DPO matches or beats it at a fraction of the cost. That is the entire economic case for the field's migration, in one row.

---

## 14. Debugging Playbook

### 14.1 Loss-curve diagnosis

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| **Loss pinned at ~0.693, flat** | Margin is zero. Reference = policy (no training happening), or β too high, or LR 0, or `ref_model` accidentally the trainable model | Compute `logratio_chosen − logratio_rejected` at step 0 — it should be ~0 and start moving within 20 steps | Lower β; raise LR; verify `ref_model` is frozen and in `eval()` |
| **Loss decreases smoothly, win rate decreases** | **`chosen`/`rejected` are swapped** in the data pipeline | Take 20 held-out pairs, run `implicit_margin` (§12.3); if the mean is *negative* you have your answer | Swap the mapping; add a data-prep assertion comparing against a hand-labelled gold pair |
| **Loss drops to near 0 within one epoch** | Overfitting. The margin is exploding on 5k pairs seen repeatedly | Log mean implicit margin — if it exceeds ~10 nats you are past the useful region | Fewer epochs (1), lower β (0.05–0.1), add `label_smoothing=0.1`, get harder pairs |
| **Loss spiking / periodic jumps** | LR too high for LoRA, or a bad batch of very long sequences, or grad-accum with batch=1 producing pathological micro-batches | Log per-step loss, not per-epoch; look for correlation with sequence length | Lower LR 2–5×; add warmup (0.1); clip gradients (`max_grad_norm=1.0`); length-bucket or filter outliers |
| **Loss NaN** | In fp16 (not bf16), or a division by zero in an empty-sequence log-prob, or corrupted labels | Check `--bf16` is actually on; inspect for zero-length responses | Switch to bf16; filter empty/zero-length rows; reduce LR |
| **Train loss falls, eval loss rises immediately** | Classic overfitting on a small preference set | — | Fewer epochs, more data, higher `lora_dropout`, lower `r` |
| **Eval loss falls, eval *win rate* falls** | **Over-optimisation** (§4.9). You are fitting the judge's biases | Log mean response length and the capability suite | Early stop; raise β; length-normalise; switch to SimPO |
| **Loss falls but the model is byte-identical** | The adapter was never attached, or you saved the wrong checkpoint, or `target_modules` matched nothing | `print(sum(p.numel() for p in model.parameters() if p.requires_grad))` — must be > 0 | Fix `target_modules`; verify the saved checkpoint directory contains `adapter_model.safetensors` |
| **Loss oscillates with a period of grad-accum steps** | Batch-size-1 micro-batches with high variance | — | Raise `per_device_train_batch_size` to 2–4 and lower accumulation; or use a larger effective batch |
| **Loss is fine, but generation is garbage** | **Truncation.** Responses were cut at `max_length`, or the chat template differs between train and inference | Print one fully-formatted training example and one inference prompt and diff them | Raise `max_length` above the p99; set `tokenizer.chat_template` explicitly |

### 14.2 Symptom → cause → fix

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| **CUDA OOM at trainer init, before step 1** | Full-FT DPO's silent reference deep-copy | `print(torch.cuda.memory_allocated()/1e9)` after construction | Switch to LoRA, or pass an explicitly-loaded quantised `ref_model`, or use SimPO |
| **OOM at step 1 but not init** | Sequence length, not model size | Log the batch's token count | Lower `max_length`/`max_prompt_length`; enable `gradient_checkpointing=True`; batch=1 |
| **`KeyError: 'prompt'` / missing columns** | `remove_unused_columns=True` | Check `report_to` and config dump | Set `remove_unused_columns=False` |
| **`AttributeError: 'DPOTrainer' got unexpected keyword 'tokenizer'`** | TRL ≥ 0.13 API rename | `pip show trl` | Use `processing_class=tokenizer` |
| **Model output is repetitive/stuck** | `repetition_penalty` > 1.3, or degenerate over-optimisation | Try penalty 1.0 | Lower the penalty; check β and the margin; you may have over-trained |
| **Model got much more verbose** | Length bias — the classical DPO failure | Mean response length per epoch | Switch to SimPO, or length-normalise, or add short-and-correct pairs to the data |
| **Model agrees with everything** | Sycophancy — the RM/judge rewards agreement | Feed a *false* premise and check whether the model pushes back | Add "corrects a false premise" pairs; check the judge for agreement bias |
| **Model refuses benign requests after alignment** | Safety-heavy preference mix; refusal is over-rewarded | False-refusal rate on a benign set | Rebalance the preference mix with helpfulness pairs; add a refusal-calibration eval |
| **Model got worse at general tasks** | Alignment tax | MMLU/GSM8K delta | Raise β; add SFT/pretraining mixing (PPO-ptx analogue = SFT loss on chosen); LoRA instead of full FT; fewer epochs |
| **Merged stage-2 model produces different text than the stage-2 checkpoint** | 8-bit merge rounding | Compare generations pre/post merge | Merge in bf16 (§6.5) |
| **Loss is 0.0 exactly** | Every pair was filtered out (all `chosen == rejected` after cleaning) | `len(trainer.train_dataset)` | Fix the data pipeline; assert non-empty |
| **`train_samples_per_second` ≈ 1 with a small model** | Batch 1 + first-step warmup + 8-bit dequantisation overhead | Run 10 steps and re-read | Expected on tiny data; ignore |
| **Training completes in seconds and the model is unchanged** | 1 optimiser step (5 rows ÷ 8 accumulation) | `global_step` in the `TrainOutput` | Use more data, or lower `gradient_accumulation_steps` |
| **GRPO: reward is constant across the group** | Zero-variance group (all 0 or all 1) | Log `reward_std` per batch | Dynamic sampling (skip zero-variance prompts — DAPO); raise temperature; adjust task difficulty |
| **GPT-4 judge says your model is better, humans disagree** | Judge bias — length, style, self-preference | Run the judge with position swapping and length control | Use LC win rate; use a different-family judge; human-eval a sample |
| **Win rate is 90%+ vs your own baseline** | Evaluation set leaked into preference data, or the baseline is broken | Deduplicate prompts across the two sets | Rebuild the eval set from a held-out prompt pool |

---

## 15. Applied Case Studies

### 15.1 The pharma assistant — the video's own pipeline, continued

**Situation.** A pharma-domain assistant. You have already done domain adaptation on PDFs (CS-12, `checkpoint-5`) and instruction tuning (CS-13, `checkpoint-3`). The SFT model answers questions but drifts off-topic and produces textbook-exercise artefacts (`19.3 Explain the main benefits of big data…`).

**Why alignment.** This is a pure behaviour problem: the model has domain vocabulary and instruction format, but its *answer selection* is miscalibrated — it emits the most likely continuation of a training-corpus question, not the best answer to the asked question.

**Config, exactly as run:**

```python
base_model            = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
instruction_checkpoint = "/content/checkpoint-3"
dataset               = load_dataset("csv", data_files="pharma_preference_data.csv")["train"]  # 5 rows
lora_config           = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                   lora_dropout=0.05, target_modules=["q_proj","v_proj"], bias="none")
# load base -> PeftModel.from_pretrained(base, instruction_checkpoint) -> merge_and_unload()
# -> get_peft_model(merged, lora_config)
dpo_args              = DPOConfig(output_dir="./tinyllama-preference-alignment", learning_rate=2e-5,
                                  per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                  num_train_epochs=1, beta=0.1, loss_type="sigmoid",
                                  remove_unused_columns=False)
trainer = DPOTrainer(model=pref_model_lora, ref_model=None, args=dpo_args,
                     train_dataset=dataset, processing_class=tokenizer)
trainer.train()
```

**Result.** `global_step=1, loss=0.6619, runtime=5.18 s`. The aligned model's output to the Metformin question stays on-topic where the SFT model's output drifted to Crohn's disease.

**What went wrong first, and the lesson.** The instructor's own account: the *previous* video's pipeline stacked LoRA adapters across the three stages — `LoRA on LoRA on LoRA` — and the outputs were poor and redundant. *"There's some redundancy uh in the output uh because I did the mistake uh in the previous uh training"* [56:34]–[56:40]. The cause was adapter stacking, and the fix was `merge_and_unload()` between every stage. **This is the most transferable lesson in the module**: the pipeline shape is right and the plumbing was wrong, and the plumbing failure was invisible in the loss curve.

### 15.2 Support-agent tone alignment on a single GPU

**Situation.** A 7B model behind a B2B SaaS support product. Customers report the assistant is "terse and slightly condescending" even though answers are correct. No GPU cluster — one A100 40GB.

**Constraints.** Tone/helpfulness only. No verifier. 6 months of thumbs-up/down telemetry (~40k events) and no head-to-head pairs. Budget: 3 engineer-weeks.

**Decision path.** Behaviour problem → model is already instruction-tuned → reward is not verifiable → **no pairs, but unpaired binary labels exist → KTO.** (If a few thousand pairs could be derived by pairing the thumbs-up and thumbs-down responses to the *same* prompt, DPO would be better; the telemetry did not have that structure.)

**Config.**

```python
KTOTrainer(
    model=qlora_model,                 # 4-bit base + LoRA r=16, all linear layers
    ref_model=None,                    # PEFT path: same base, adapter disabled
    args=KTOConfig(beta=0.1,
                   desirable_weight=1.0,     # λ_D
                   undesirable_weight=1.5,   # λ_U - losses weigh heavier, per prospect theory
                   learning_rate=1e-5,
                   per_device_train_batch_size=2,
                   gradient_accumulation_steps=8,
                   num_train_epochs=1,
                   max_length=1024,
                   remove_unused_columns=False),
    train_dataset=telemetry_dataset,   # {"prompt","completion","label": bool}
    processing_class=tokenizer,
)
```

**Data preparation, the part that decided the outcome.** Raw thumbs-down events are *not* training data: a thumbs-down can mean the answer was wrong (a knowledge problem), slow, or rude. The team wrote a classifier prompt and used a different-family LLM to label the *reason*, then kept only the tone-related thumbs-downs and paired them with the same-prompt thumbs-up responses. **60% of the raw signal was discarded, and the run worked because it was.**

**Result.** Win rate vs the SFT baseline: **58% on 400 held-out comparisons** (95% CI ±5 pts), with the mean response length *unchanged* (length bias controlled for). MMLU delta: −0.8%. Cost: ~$7 of compute, plus 4 days of data work.

**What went wrong first.** Run 1 used the raw telemetry. The model got *worse* at answering (it had been trained to prefer the *style* of responses that had been thumbed up for reasons of speed, which included many "I'll look into that" non-answers). Lesson: **label the reason, not just the sentiment.**

### 15.3 Code assistant — verifiable rewards

**Situation.** A 7B code model. 30k synthetic tasks with unit tests. The model produces compilable code that fails ~45% of tests.

**Decision path.** Behaviour problem (producing *correct* code is a capability shaped by a behaviour) → instruction-tuned → **the reward is verifiable: run the tests** → **GRPO/RLVR.** No reward model, no preference data, no human labels.

**Config.**

```python
GRPOConfig(
    output_dir="./code-grpo",
    learning_rate=1e-6,                 # RLVR wants a much smaller LR than DPO
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=8,                  # G - the group
    temperature=1.0,                    # rollout diversity
    max_completion_length=1024,
    beta=0.04,                          # KL coefficient - note this is the PPO-convention beta
    loss_type="dapo",                   # DAPO-style: token-level loss + clip-higher
    num_train_epochs=1,
    bf16=True,
)
# reward function: execute the generated code against the task's unit tests in a sandbox
def reward_func(completions, tests, **kwargs):
    return [1.0 if run_tests(c, t) else 0.0 for c, t in zip(completions, tests)]
```

**Result.** Pass@1 on the held-out test suite: **45% → 71%** over 3 days on 2×H100. General capability (MMLU) delta: −1.2%.

**What went wrong first.** Run 1 used G=4 and a fixed task set. Two failure modes appeared simultaneously: (a) ~40% of prompts produced a **zero-variance group** (all four rollouts failed, or all four passed) and contributed no gradient; (b) the model learned to **pass the tests without generalising** — it started special-casing the test inputs. Fixes: G=8 with **dynamic sampling** (skip zero-variance prompts), a held-out test split the model never sees, and a reward that includes a small "code executes without crashing on 3 held-out inputs" component.

> **Beyond the video — DAPO/GSPO/Dr.GRPO refinements.** The vanilla GRPO in this section has known issues that the 2025 literature fixes, and each fix is a one-line config change in TRL:
> - **DAPO** — (1) *clip-higher*: decouple the lower and upper clip bounds (ε_low=0.2, ε_high=0.28) because symmetric clipping suppresses the low-probability tokens that drive exploration; (2) *dynamic sampling*: drop prompts whose group has zero reward variance instead of wasting the batch; (3) *token-level loss*: average the loss over all tokens in the batch rather than per-sequence, removing the length bias in the gradient; (4) *overlong filtering*: mask truncated sequences out of the loss.
> - **GSPO** (Group Sequence Policy Optimization) — moves the importance ratio from the **token** level to the **sequence** level, which stabilises very long CoT training where per-token ratios multiply out to extreme values. It is what Qwen used for their reasoning models.
> - **Dr.GRPO** — removes two biases in the vanilla formulation: the length normalisation (which favours shorter *correct* answers, but also distorts the gradient) and the per-group std division (which up-weights questions with low reward variance). Simply using `(r_i − mean)/1` instead of `/std` improves results in their ablations.
>
> **None of these change the memory story** — GRPO and all its refinements remain 2-model, no value model, rollout-dominated.

### 15.4 The run that should not have happened

**Situation.** A team wants "a model that knows our internal documentation." They plan a three-week effort: SFT on 12,000 doc chunks, then DPO on 3,000 preference pairs generated by GPT-4.

**The intervention.** This is a knowledge problem, not a behaviour problem. Fine-tuning to inject facts into a 7B model is the failure mode named in CS-04. The correct architecture is RAG over the docs (CS-04, and `code/10_embedding_finetune.py` for retrieval quality).

**What they actually did instead.** Ran a 3-day RAG prototype first, measured answer accuracy on 200 questions: **84%**. Then ran the fine-tuning path anyway as a "behaviour overlay" — but *only* on the pairs that RAG could not fix (tone, citation formatting, refusal when the docs do not contain the answer). Pairs used: **600**, not 3,000. Run: DPO LoRA, 1 epoch, 2 hours, $4.

**Result.** Citation-format adherence 61% → 94%. Refusal-when-unknown rate 30% → 88%. Answer accuracy unchanged (RAG supplies it). Total spend: **$11k**, of which $9k was the (avoided) annotation budget redirected to the RAG eval.

**The lesson.** *Alignment fixes behaviour. Retrieval fixes knowledge.* Diagnose first. If a 3-day RAG prototype moves the metric you care about, you do not have an alignment problem. §8.3's STOP conditions would have caught the original plan in ten minutes.

---

## 16. Production Considerations

### 16.1 Serving

| Concern | Approach |
|---|---|
| **Adapter vs merged** | Serve the **adapter** (PEFT) rather than the merged model: one base in VRAM, N adapters swapped per request, ~4 MB each. vLLM and TGI both support multi-LoRA serving with per-request adapter selection. Merged models require a full copy per variant (14 GB for 7B). |
| **Latency** | Alignment changes weights, not architecture — so latency is unchanged from the SFT model. This is a real advantage over prompt-based alignment, which adds input tokens on every call. |
| **Quantisation for serving** | After training, quantise for serving (GPTQ/AWQ/GGUF — CS-10/CS-11). Quantise *after* alignment, and re-run the win-rate eval on the quantised model: quantisation interacts with the sharpened output distribution that alignment produces. |
| **Reference model at inference** | None. The reference model exists only during training. Do not ship it. |

### 16.2 Versioning the alignment artifact

Version these **together**, because a change in any one invalidates the others:

```
alignment-artifact/
├── adapter/                     # the trained LoRA
├── base_model_revision          # exact HF revision SHA of the base
├── sft_checkpoint_revision      # exact SHA of the stage-2 checkpoint (the reference)
├── preference_data_hash         # hash of the training pairs
├── preference_data_provenance   # annotator IDs, judge model + version, guidelines version
├── training_config.json         # beta, lr, epochs, loss_type, max_length, seed
├── eval_report.json             # win rate, CI, capability deltas, length stats, per-checkpoint
└── chat_template.txt            # verbatim - mismatched templates are a top-3 silent bug
```

### 16.3 Monitoring and drift

| Signal | Threshold | Action |
|---|---|---|
| **Win rate vs shipped baseline** (rolling, human-audited sample) | < 50% over 200 comparisons | Roll back |
| **Mean response length** | +20% vs the eval-time baseline | Investigate: prompt distribution changed, or the model is drifting verbose |
| **Refusal rate on benign traffic** | +5 pts | Re-balance the preference mix |
| **Capability suite** (weekly) | −2% vs baseline | Investigate; likely prompt-distribution shift, not weight drift |
| **Prompt distribution drift** (embedding centroid distance) | PSI > 0.2 | The preference data is stale; collect new preferences on the new distribution |
| **Complaint rate / escalation rate** | Any sustained rise | Roll back first, diagnose second |

### 16.4 Regression tests in CI

```python
# tests/test_alignment_regression.py  — run on every adapter promotion
KNOWN_GOOD = {                                   # frozen at the last promoted version
    "win_rate_vs_sft": 0.58,
    "preference_accuracy_heldout": 0.71,
    "mmlu_delta": -0.008,
    "mean_length_tokens": 212,
    "false_refusal_rate": 0.03,
    "sycophancy_probe_pass": 0.90,               # fraction of false-premise tests where the model pushes back
}

def test_no_regression(candidate):
    for k, baseline in KNOWN_GOOD.items():
        got = candidate[k]
        # win rate / accuracy / probes must not fall; length must not balloon
        assert got >= baseline - 0.03 if k != "mean_length_tokens" else got <= baseline * 1.2, \
            f"REGRESSION on {k}: {got} vs baseline {baseline}"
```

**Every one of those six numbers is a *different* failure mode.** Win rate catches "it got worse". Preference accuracy catches "the margin reversed". MMLU delta catches the alignment tax. Mean length catches verbosity drift. False-refusal catches over-refusal. The sycophancy probe catches agreement drift. A CI gate on win rate alone misses four of the six.

### 16.5 Compliance and the audit trail

- **Preference data is often PII-bearing.** Customer telemetry used as preference data (the KTO case) must pass the same retention and consent rules as the raw logs. It usually does not, because it is "just training data".
- **Annotation guidelines are policy documents.** Version them and keep every version, because they encode your refusal policy. A regulator asking "how does the model decide to refuse?" is asking about your guidelines, not your β.
- **LLM-judge provenance must be recorded.** Which model, which version, which rubric, which sampling parameters. A judge upgrade can silently change your reward and therefore your shipped behaviour.
- **Keep the pre-alignment checkpoint.** Rollback from an alignment regression is otherwise a re-run, and a re-run is not reproducible on a different data sample.

---

## 17. Common Misconceptions

1. **"RLHF and alignment are the same thing."** RLHF is *one method* of alignment. DPO, ORPO, GRPO, KTO and SimPO are alignment methods that are not RLHF. The video's own framing is that RLHF, RLAIF and DPO are three techniques side by side [17:08].
2. **"DPO trains a reward model implicitly, so it's basically RLHF."** DPO re-parameterises the *policy* so that the optimal reward is expressible as a log-ratio — no reward model exists at any point. There is no reward model artifact, no scoring step, and no sampling loop. `"This is a simple supervised learning"` [19:47].
3. **"RLHF is deprecated."** The instructor says *"that is deprecated. Now people are using DPO"* [22:19] — read that as *deprecated for his use case and for teams without an RL infrastructure*. Frontier labs still use PPO heavily, and the entire reasoning-model wave is PPO-family RL. The accurate statement is: **RLHF/PPO is deprecated as a default for product teams, not as a technique.**
4. **"Preference alignment makes the model more accurate."** It makes the model's *choices among plausible answers* better. It does not add knowledge and can increase confident fabrication.
5. **"More epochs = more alignment."** DPO overfits after 1–3 epochs. The loss keeps falling; the model gets worse.
6. **"A lower loss means a better model."** Loss is a function of the margin, and the margin grows during over-optimisation. Loss only tells you the run is *doing something*.
7. **"DPO is strictly better than PPO."** DPO is better *at its price point*. With 50k+ pairs, a well-trained RM, and the compute to run PPO, PPO's exploration advantage is real. The honest statement is that DPO dominates below ~20k pairs.
8. **"The KL term keeps the model safe."** The KL term keeps the model *near the reference*, which is not the same as safe. A KL anchor against a rude SFT model anchors you to rudeness. It is an anti-drift term, not a safety mechanism.
9. **"β is a regularisation strength, so higher is safer."** In DPO, higher β pushes the policy *harder away* from the reference. The sign convention inverts between PPO and DPO and this trips up experienced engineers.
10. **"Reward hacking is overfitting."** It happens with a perfectly-generalising RM, on the training distribution, because the policy is *searching* the response space where the RM has errors. This is why early stopping on an independent metric is mandatory.
11. **"I can skip SFT and go straight to DPO on a base model — it's the same objective."** It is the same objective with a meaningless reference. §4.2.
12. **"ORPO is just DPO with a different loss."** ORPO has **no reference model** and **no β** — it uses an odds ratio and includes the SFT loss in the same objective, which is why it can replace two stages with one.
13. **"GRPO is PPO without a value model, so it's strictly cheaper and equivalent."** GRPO's advantage is *group-relative* and *per-sequence*, not per-token, which means it cannot express partial credit within a response. For long chains of thought with intermediate rewards, PPO plus a process reward model is a genuinely different capability.
14. **"AlpacaEval win rate is the number to report."** Report the **length-controlled** win rate. Raw win rate flatters verbosity by ~10 points, and DPO's most common failure is verbosity.
15. **"The reward model score tells me which checkpoint to ship."** The RM score is the quantity being maximised by the policy and therefore the *last* thing to trust. It peaks at the worst checkpoint.
16. **"Alignment is a one-time step."** It is a loop: collect preferences on production traffic, re-align, re-evaluate. Static preference data against drifting traffic is the standard cause of slow quality decay.
17. **"You need human feedback for preference alignment."** RLAIF exists, and the instructor's motivation for it was exactly that humans top out at *"one lakh two lakh"* labels [18:58].
18. **"The reference model costs a full extra model's memory in DPO."** With LoRA it costs nothing — the adapter is disabled and the same base weights serve as the reference.
19. **"DPO needs a prompt column, and `hh-rlhf` has one."** `hh-rlhf` has only `chosen` and `rejected`; the prompt is the shared prefix. Materialise it explicitly.
20. **"Alignment can't be A/B tested because it changes the whole model."** It can: serve the base with the old adapter and the new adapter behind a traffic split with per-request adapter selection. This is the correct way to ship (§16.1).

---

## 18. Key Takeaways

1. **Alignment is stage three of three.** Pretrain → SFT → preference alignment. Skipping stage 2 makes stage 3 actively harmful.
2. **HHH is a Pareto frontier, not a score.** You choose a point on it with your data mix and your β; you do not maximise all three.
3. **Preferences are `{prompt, chosen, rejected}`.** Every dataset in the ecosystem is a re-skin of that triple. Materialise the `prompt` column explicitly.
4. **Preference data is an ordering, not a label.** That is why it is 5–10× cheaper per unit of improvement than demonstrations.
5. **IAA is the ceiling on your alignment.** Measure it before you spend money on volume; below 65% raw agreement you have no signal.
6. **The memory arithmetic decides the method.** 4 models for PPO, 2 for DPO, 2 for GRPO, 1 for ORPO, 1 for SimPO. For 7B at full FT that is ~295 GB vs ~150 GB vs ~136 GB.
7. **LoRA collapses DPO's reference cost to zero**, because the frozen base *is* the reference once the adapter is disabled. `ref_model=None` means different things with and without PEFT.
8. **β means opposite things in PPO and DPO.** In PPO it is a leash; in DPO it is an accelerator. Typical DPO range 0.1–0.5; default 0.1.
9. **β→0 collapses the policy; β→∞ freezes it.** Symptom of the latter: loss pinned at `log 2 = 0.6931`.
10. **DPO's loss floor is 0.6931.** A run that never drops below it is not learning, and a run at 0.66 after one step has taken one step.
11. **If the reward is verifiable, use GRPO and skip the reward model and the preference data entirely.** If it is not, use DPO or ORPO.
12. **PPO only wins above ~20,000 pairs with a well-trained reward model.** Below that, DPO matches it at a fraction of the cost. This is the whole migration story.
13. **Never select a checkpoint on the reward-model score.** It is the metric being gamed; it peaks at the worst checkpoint. Use human/gold-judge win rate plus a capability suite.
14. **Length is the canary.** If mean response length rises monotonically through training, you are measuring verbosity, not quality. Report LC win rate.
15. **Merge LoRA between stages; never stack it.** `merge_and_unload()` then `get_peft_model()`. Stacked adapters produce unstable loss, hallucination and degraded quality, and the loss curve does not show it. The instructor: *"delta patch cannot be stacked they must be merged before the next training"* [42:03].
16. **Alignment fixes behaviour; retrieval fixes knowledge.** Diagnose first. If a RAG prototype moves the metric, you do not have an alignment problem.
17. **The alignment tax is real and irreducible.** Any KL-constrained optimisation away from π^SFT gives up some of π^SFT's distribution. Mitigate with PPO-ptx-style mixing, LoRA, high β, and early stopping — then report both axes.
18. **Compute is 0.2% of an alignment project's cost.** Data is ~75%, evaluation ~25%. Optimise the expensive parts.

---

## 19. Self-Check Questions

1. Name the three stages of the canonical LLM training pipeline and one thing each stage contributes that the others cannot.
2. Why does running DPO on a raw base model produce a worse model than running it on an SFT checkpoint? Give the mechanism in terms of the KL term.
3. Write the DPO loss from memory. Define every symbol and state what happens to the loss at β→∞.
4. A DPO run's loss goes from 0.69 to 0.12 over two epochs and the mean response length doubles. What happened, what is the metric you failed to watch, and what are three fixes?
5. Your `chosen`/`rejected` labels were accidentally swapped. Describe exactly what the loss curve looks like and how you would detect it within ten minutes.
6. What does `ref_model=None` do when using PEFT adapters, and what does it do at full fine-tuning? Why does the answer differ?
7. Compute the VRAM for full fine-tuning PPO on a 7B model and for QLoRA ORPO on the same model. Show the arithmetic.
8. Given: unit-test-checkable code generation, 30k synthetic tasks, no human labels. Which method, why, and what config knob do you set differently from a DPO run?
9. What is the alignment tax, what is PPO-ptx, and which of the three mechanisms in §4.8 does it address?
10. Explain Goodhart's law as it applies to a reward model, describe the shape of the over-optimisation curve, and name four mitigations in priority order.

<details>
<summary>Answers</summary>

1. **Pretrain** (next-token prediction on web text) supplies general language and world knowledge; **SFT** (demonstration pairs) supplies instruction-following and response format; **preference alignment** (pairs) supplies the selection among plausible answers — tone, safety, refusal calibration, concision. None substitutes for another: pretraining has no instruction format, SFT has no preference ordering, and alignment cannot add knowledge.
2. The objective is `max E[r] − β·KL(π_θ ‖ π_ref)`. With a base model as π_ref, the KL term pins the policy to an *autocompleter* rather than an assistant. The reward term pulls toward helpfulness. The two terms fight, and the resulting policy is worse at instruction-following than an SFT model while only marginally better at tone. Concretely: both `chosen` and `rejected` are far off-distribution for a base model, so `log π(y|x)` is tiny and noisy for both and the log-ratio carries almost no signal.
3. `L_DPO = −E[log σ( β · ( log(π_θ(y+|x)/π_ref(y+|x)) − log(π_θ(y−|x)/π_ref(y−|x)) ) )]`, where x is the prompt, y+/y− are chosen/rejected, π_θ is the policy, π_ref the frozen SFT reference, β the strength coefficient, σ the logistic sigmoid. As **β→∞** the sigmoid argument saturates for any nonzero margin; the loss becomes constant, its gradient is zero, and **π_θ never moves from π_ref**. At initialisation (π_θ = π_ref) the margin is 0 and the loss is exactly `−log σ(0) = log 2 = 0.6931`.
4. **Over-optimisation / length bias.** The model discovered that longer answers raise the implicit reward. The metric you failed to watch is **mean generation length** (and, upstream, the LC win rate rather than the raw one). Fixes: (a) switch to **SimPO**, which length-normalises the implicit reward by construction; (b) **length-normalise or length-penalise** the reward in your judge/RM; (c) rebalance the preference data to include pairs where the *shorter* response wins, and add an explicit length-neutrality clause to the annotation guidelines. Also: early-stop on a human/gold-judge win rate rather than on training loss.
5. The loss curve looks **completely normal** — it falls smoothly, because the objective is symmetric in the roles and the model happily learns to prefer the dispreferred response. Detection: compute the **implicit margin** on 20 held-out pairs (`β·(logratio_chosen − logratio_rejected)`); it will be **negative**. Confirmation: generate both responses for a held-out prompt and check that the model assigns higher likelihood to `rejected`. Prevention: assert at data-prep time that a hand-labelled gold pair yields a positive margin after one epoch.
6. **With PEFT/LoRA:** TRL disables the adapter on the same base weights and uses that as the reference — effectively a *free* reference, one set of weights in memory. **At full fine-tuning:** TRL **deep-copies the model at trainer init** and freezes the copy — two full models, which is the most common cause of an unexpected OOM in full-FT DPO. The difference is that with LoRA the frozen base weights and the trainable adapter are separable, so "the model minus its adapter" is already a valid reference; with full FT there is no separable trainable part, so a copy is required.
7. **Full-FT PPO, 7B:** policy 7B×16 B = 112 GB (weights+grads+fp32 master+Adam m,v); value model trained too, another 112 GB; frozen reference 14 GB (bf16); frozen RM 14 GB (bf16); activations 15–40 GB. **Total ≈ 295–320 GB.** **QLoRA ORPO, 7B:** 4-bit base ≈ 4.5 GB; LoRA adapters + 8-bit optimiser state ≈ 1 GB; activations 6–14 GB; **no reference model, no RM, no value model.** **Total ≈ 11–19 GB.** Ratio: roughly **20×**.
8. **GRPO / RLVR**, because the reward is a program (run the unit tests) — no reward model, no preference pairs, no human labels. Config differences vs DPO: (a) `num_generations=8` (or more) instead of a pair batch; (b) `temperature=1.0` for rollout diversity (DPO has no sampling); (c) `beta=0.04` — and note this is the **PPO-convention β**, a KL *penalty* coefficient, so it does NOT mean the same thing as DPO's β=0.1; (d) a **much smaller learning rate** (1e-6 vs 2e-5); (e) you supply a `reward_func` instead of a `chosen`/`rejected` dataset. Also enable dynamic sampling (DAPO) so zero-variance groups are skipped.
9. The **alignment tax** is the capability degradation caused by alignment — InstructGPT reported RLHF models scoring lower than base GPT-3 on SQuAD, DROP, HellaSwag and others while being strongly human-preferred. **PPO-ptx** adds a term maximising the log-likelihood of pretraining text to the PPO objective (`L_PPO + γ_ptx · E[log π_θ(x)]`, γ_ptx ≈ 0.01–0.1), so pretraining gradients keep flowing. It addresses the **narrow-reward** mechanism: capabilities outside the alignment distribution receive no reward and are free to drift, and the ptx term gives them a gradient again. It does **not** address the "π^SFT is itself already degraded" mechanism, which is upstream of alignment.
10. **Goodhart:** the reward model is a *proxy* for human judgement, and the policy is optimising the proxy; given enough optimisation pressure the policy finds the proxy's errors, on the training distribution, with a perfectly-generalising RM. **Shape of the curve:** as optimisation proceeds (measured by KL from the reference), the *proxy* reward rises monotonically while the *true* reward rises, **peaks**, and then declines — Gao, Schulman & Hilton (2023) fit this in `√KL`, with the true-reward peak around KL ≈ 5–20 nats depending on RM size. **Mitigations in priority order:** (1) the **KL penalty** — bounds how much of the RM's error surface the policy can reach, free; (2) **early stopping on a proxy-independent metric** (human or gold-judge win rate, plus a capability suite) — the only control independent of the RM itself; (3) **length-normalised rewards / SimPO-style formulations** — kills the most common concrete hack; (4) **reward-model ensembles** (k=3–5, penalise by inter-model disagreement) — the policy can only hack the intersection of their errors. Behind those: reward clipping/whitening, RM refresh on the current policy's outputs (iterative RLHF), preference data collected on the policy's own distribution, and capability evals in the training loop as the alarm.

</details>

---

## 20. Cross-References

| Relationship | Module |
|---|---|
| **Builds on** | CS-01 (LLM lifecycle — the three stages), CS-13 (SFT — you cannot align what you have not instruction-tuned), CS-13 §6.8 + CS-11 §4.11 (LoRA/QLoRA — the delta patch and merge discipline in §6.5; the planned "CS-23" module was never written) |
| **Needed by** | *nothing written yet.* The intended deep dives — CS-24 (RLHF with PPO, §4.6.1–4.6.2), CS-25 (DPO, §4.6.3–4.6.8, 4.6.13), CS-26 (GRPO, §4.6.10), CS-27 (ORPO, §4.6.9) — and the CS-28 capstone were planned but never written; their material is in the §4.6 profiles here |
| **Contrasts with** | CS-04 (Fine-Tuning vs RAG vs Agents — the diagnose-first rule this module's §8.3 enforces), CS-12 (domain-adaptive continued pretraining — stage 1 of the pipeline), CS-16/CS-17 (Unsloth/Axolotl — frameworks that implement these trainers) |
| **Implements with** | CS-15 (LLaMA-Factory — no-code DPO/ORPO), CS-16 (Unsloth — DPO/ORPO with 2× memory savings), CS-17 (Axolotl — YAML DPO/ORPO/GRPO configs) |
| **Bridges to** | AP-01 (The Ethics & Philosophy of Alignment — *what is a preference, and who gets to encode it*) |

---

## Appendix A — Instructor's Verbatim Key Claims

| Timestamp | Claim |
|---|---|
| [5:41] | *"the first step of the LLM training process is called the unsupervised pre-training or self-supervised training. So here we get the foundational model."* |
| [6:34] | *"these are three main stage of any LLM training."* |
| [7:07] | *"to align or train a model with a human expectation or with a human preferred data is called preference alignment or preference training."* |
| [7:20] | *"this particular technique was introduced by the open AI in uh instruct GPT paper."* |
| [7:33] | *"we do it to get a safe, helpful and the honest model."* |
| [8:22] | *"the answer is correct. But it is rude. It is dismissive and there is no explanation means it is not aligned with the human expectation."* |
| [10:54] | *"after the preference based training model does not just produce accurate answer it behave according to the user preference."* |
| [11:25] | *"this preference training we do to improve the uh safety and ethics of the model… for generating the helpful answer for the more polite answer."* |
| [12:26] | *"rapid bait loss 10 kin a week is unsafe… A safer approach is just to loss 0.5 to 1 kg per week through balanced diet, hydration and activity."* |
| [15:11] | *"inside the preference based training guys you will find out three column three main column right uh one is a prompt second is a chooser and the third is a reject."* |
| [15:43] | *"the data set is called the HH RLHF… inside this particular data you will find out two column only. The first is a chosen and the second is a rejected."* |
| [16:02] | *"the data set name is ultra feedback binarize preference clean data."* |
| [16:25] | *"the data set name is math step dpo 10k."* |
| [17:15] | *"the very infamous technique that is called RLHF reinforcement learning through human feedback. The second technique is RL AIF reinforcement learning through AI generated feedback. The third is DPO direct preference optimization."* |
| [17:44] | *"KTO is called the canman tverki optimization p stand for proximal policy optimization and this IPO stand for implicit preference optimization."* |
| [18:01] | *"even OpenAI is also revealed that they have used this particular technique initially for the reward modeling."* |
| [18:20] | *"The core idea [was] to train a reward model using the human feedback… ranked responses… which was ranked by the human."* |
| [18:58] | *"human generated feedback was not very scalable idea means how many feedback can be annotated by the human right maybe one lakh two lakh."* |
| [19:37] | *"this is not a reinforcement based training… This is a simple supervised learning."* |
| [20:46] | *"the direct preference optimization DPO technique… it was introduced by the Stanford University."* |
| [21:29] | *"it was inspired from the RLHF only the difference is human is not going to be annotate anything now like only AI will do that."* |
| [22:19] | *"we are not going to use the RLHF technique, RLAIF technique as of now… that is deprecated. Now people are using DPO which is like very simpler."* |
| [25:59] | *"the meaning is how much more does your model prefer the chosen answer compared to the reference model."* |
| [26:29] | *"we'll see the differences. So choose an improvement. Okay. And the rejected improvement… if it is going to be positive means my model is aligning to this chosen prompt."* |
| [26:54] | *"if the beta is high means we are enforcing model to like go to we are enforcing model to align more to the chosen — if beta is low means the rejected one more the model is more aligning to the rejected."* |
| [27:11] | *"the typical value of the beta between 0.1 to 0.5."* |
| [28:46] | *"here human was labeling the data… whatever response we were generating through the model human was ranking those thing."* |
| [30:22] | *"So that my model can understand the domain specific language domain specific vocabulary."* |
| [33:47] | (on `repetition_penalty`) *"the token should not be repeat… we can mention this particular score 1.1 and I think this score we can up like write up to two."* |
| [36:06] | *"most of the beginner and even the experienced person uh does this kind of mistake. So I don't want to repeat that mistake over here."* |
| [38:44] | *"Later on we have automated this task also. On a very high scale. So this task have been automated by the LM only."* |
| [41:52] | *"lura is not a fully full layer okay it's not a complete model it's just a delta patch."* |
| [42:03] | *"delta patch cannot be stacked they must be merged before the next training."* |
| [44:13] | *"disadvantages loss will be unstable, model will hallucinate… tuning will not be good and the quality will degrade."* |
| [44:47] | *"`get_peft_model`… it creates a new lora during the training… `PeftModel.from_pretrained`… it load a already trained Lora model for the inference or for the further training."* |
| [51:04] | (the correct-approach table) *"non-instruction base lora it is correct instruction base plus merge… then new lora this is a correct approach directly lura on lura it is not a correct approach."* |
| [53:38] | *"here is my model and reference model. I'm not passing any reference model otherwise I can pass the reference model over here."* |
| [56:34] | *"there's some redundancy uh in the output uh because I did the mistake uh in the previous uh training."* |
| [58:02] | *"if you want to more enhance on the human preferences, you can take one more step like this chat GPT have done."* |

## Appendix B — Reference Links & Papers

| Work | What it is | Why it matters here |
|---|---|---|
| **InstructGPT — Ouyang et al., 2022**, *Training language models to follow instructions with human feedback* | The paper the video shows [20:12]. Defines the 3-step RLHF pipeline, the alignment tax, and PPO-ptx. | The origin of everything in this module. §4.4, §4.8. |
| **DPO — Rafailov, Sharma, Mitchell et al., 2023** (Stanford; the video names Archit Sharma [21:06]) | Derives the closed-form policy and the pairwise supervised loss. | §4.6.3, the module's default method. |
| **RLAIF — Lee et al. (Anthropic), 2023/2024** | AI feedback replaces human feedback; shows RLAIF can match RLHF. | §4.6.11. |
| **Constitutional AI — Bai et al., 2022** | The constitution + self-critique + revision recipe. | §4.6.11, AP-01. |
| **IPO — Azar et al., 2023** | Identity preference optimisation; finite margin target. | §4.6.4. |
| **KTO — Ethayarajh et al., 2024** | Prospect-theory utility for unpaired binary feedback. | §4.6.6. |
| **SLiC-HF — Zhao et al., 2023** | Sequence likelihood calibration; hinge loss preprint predating DPO. | §4.6.7. |
| **SimPO — Meng, Xia, Chen, 2024** | Reference-free, length-normalised preference optimisation. | §4.6.8. |
| **ORPO — Hong, Lee, Thorne, 2024** | Odds-ratio preference optimisation; single-stage SFT+preference. | §4.6.9. |
| **GRPO — Shao et al., 2024 (DeepSeekMath)** | Group-relative policy optimisation; drops the value model. | §4.6.10. |
| **DeepSeek-R1 — 2025** | GRPO + rule-based verifiable rewards at scale; the RLVR recipe. | §4.6.10, §8.2. |
| **DAPO — Yu et al., 2025** | Clip-higher, dynamic sampling, token-level loss, overlong filtering. | §15.3 Beyond the video. |
| **GSPO — Zheng et al., 2025 (Qwen)** | Sequence-level importance ratios for stable long-CoT RL. | §15.3. |
| **Dr.GRPO — Liu et al., 2025** | Removes length and std normalisation biases in GRPO. | §15.3. |
| **Gao, Schulman & Hilton, 2023** — *Scaling Laws for Reward Model Overoptimization* | Quantifies the proxy-vs-true reward gap against KL. | §4.9.2 — the evidence for early stopping. |
| **SPIN — Chen et al., 2024** | Self-play fine-tuning against your own previous outputs. | §4.6.13. |
| **Anthropic HH-RLHF dataset** | `Anthropic/hh-rlhf` — ~170k pairs, `chosen`/`rejected` only. | §4.3.3. |
| **UltraFeedback / Argilla** | `argilla/ultrafeedback-binarized-preferences-cleaned` — LLM-judge preferences. | §4.3.3. |
| **TRL documentation** | `DPOTrainer`, `DPOConfig`, `ORPOTrainer`, `KTOTrainer`, `GRPOTrainer`, `RewardTrainer`, `PPOTrainer`. | The implementation surface for §6. |
| **AlpacaEval 2 / LC win rate** | Length-controlled LLM-judged win rate. | §12.2. |

