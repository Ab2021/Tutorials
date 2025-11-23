# Day 30: RLHF Fundamentals
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why do we need RLHF? Why isn't SFT enough?

**Answer:**
- **SFT** teaches the model to mimic the training data. If the data contains both good and bad examples, the model learns both.
- **RLHF** teaches the model to optimize for *quality* as judged by humans. It learns to distinguish between a mediocre answer and a great answer.
- **Example:** SFT on "Explain gravity" might produce technically correct but boring text. RLHF optimizes for clarity, engagement, and helpfulness.

#### Q2: What is the Reward Model in RLHF?

**Answer:**
- A neural network (usually the same architecture as the base LLM) trained to predict human preferences.
- **Input:** (Prompt, Response).
- **Output:** Scalar score representing quality.
- **Training:** Pairwise comparisons. Given (Prompt, Response A, Response B, Winner), train to predict $P(A > B)$.

#### Q3: Explain the KL penalty in PPO. Why is it necessary?

**Answer:**
- **Problem:** If we only maximize the Reward Model's score, the policy might drift into a degenerate mode (e.g., generating nonsense that happens to score high due to RM bugs).
- **KL Penalty:** $-\beta \cdot KL(\pi || \pi_{ref})$ penalizes the new policy for diverging too much from the reference policy (SFT model).
- **Effect:** Keeps the model "grounded" in the original distribution while still improving quality.

#### Q4: What is "Reward Hacking"?

**Answer:**
- The model exploits weaknesses in the Reward Model to get high scores without actually being good.
- **Example:** RM prefers longer answers. Model generates 10,000-word rambling nonsense.
- **Example:** RM was trained on English. Model switches to French to confuse the RM (gets neutral score, which is higher than a bad English score).

#### Q5: How does RLHF differ from DPO (Direct Preference Optimization)?

**Answer:**
- **RLHF:** Two-stage. Train a Reward Model, then use PPO to optimize the policy.
- **DPO:** One-stage. Directly optimize the policy on preference data without training a separate RM.
- **Benefit of DPO:** Simpler, more stable, no reward hacking.
- **Benefit of RLHF:** Reward Model can be reused, inspected, and debugged.

---

### Production Challenges

#### Challenge 1: Reward Model Overfitting

**Scenario:** Your RM achieves 95% accuracy on the validation set, but the PPO-trained model is worse than SFT.
**Root Cause:** The RM overfitted to the specific phrasing/style of the preference data. It doesn't generalize to the distribution shift caused by PPO.
**Solution:**
- **Ensemble RMs:** Train 3 RMs with different seeds. Average their scores.
- **Regularization:** Add dropout, weight decay to the RM.
- **More Data:** Collect 10x more preference pairs.

#### Challenge 2: PPO is Unstable

**Scenario:** Loss is spiking. KL divergence explodes. Model outputs garbage.
**Root Cause:** PPO is notoriously hard to tune. Learning rate, KL coefficient, batch size all matter.
**Solution:**
- **Lower LR:** Use 1e-6 instead of 1e-5.
- **Adaptive KL:** Increase $\beta$ if KL > threshold.
- **Gradient Clipping:** Clip to norm 1.0.
- **Use DPO:** If PPO is too unstable, switch to DPO.

#### Challenge 3: Human Labeling is Expensive

**Scenario:** You need 100k preference pairs. At $1 per comparison, that's $100k.
**Solution:**
- **AI Labeling:** Use GPT-4 as a judge to generate synthetic preferences. (RLAIF - RL from AI Feedback).
- **Active Learning:** Train an initial RM on 10k pairs. Use it to select the most uncertain pairs for human labeling.
- **Crowdsourcing:** Use platforms like Scale AI, Surge, or MTurk.

#### Challenge 4: Length Bias in Preferences

**Scenario:** Humans consistently prefer longer responses, even if they are verbose.
**Result:** The model learns to be unnecessarily wordy.
**Solution:**
- **Instruction to Labelers:** "Prefer concise answers. Penalize verbosity."
- **Length Normalization:** $R_{norm} = R / \sqrt{length}$.
- **Controlled Generation:** Add a "Be concise" instruction during PPO sampling.

### Summary Checklist for Production
- [ ] **Data:** Collect **10k+** preference pairs.
- [ ] **RM:** Train with **Ensemble** (3 models).
- [ ] **PPO:** Use **Adaptive KL** penalty.
- [ ] **Monitor:** Track **KL Divergence** and **Reward**.
- [ ] **Fallback:** If PPO fails, use **DPO**.
