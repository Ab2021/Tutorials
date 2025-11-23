# Day 31: DPO (Direct Preference Optimization)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the main advantage of DPO over RLHF?

**Answer:**
- **Simplicity:** DPO eliminates the need for a separate Reward Model and the complex PPO algorithm. It's just supervised learning on preference data.
- **Stability:** PPO is notoriously unstable and requires careful hyperparameter tuning. DPO is as stable as standard fine-tuning.
- **No Reward Hacking:** Since there's no explicit Reward Model, the policy cannot exploit RM bugs.
- **Efficiency:** DPO trains in 1-3 epochs. RLHF requires thousands of PPO iterations.

#### Q2: How does DPO avoid training a Reward Model?

**Answer:**
- DPO uses a mathematical reparameterization of the RLHF objective.
- It derives that the optimal policy has a closed-form relationship to the reward function.
- By substituting this relationship into the Bradley-Terry preference model, the reward function cancels out.
- The result is a loss function that directly optimizes the policy on preference data without needing an explicit reward.

#### Q3: What is the role of the reference policy in DPO?

**Answer:**
- The reference policy $\pi_{ref}$ (usually the SFT model) is frozen and used to compute log probability ratios.
- The ratio $\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ acts as an implicit KL penalty.
- This prevents the policy from drifting too far from the reference, avoiding mode collapse.
- Without the reference, the model could collapse to always outputting the same "safe" response.

#### Q4: Can you extract a reward function from a DPO-trained model?

**Answer:**
- Yes. The implicit reward is:
$$ R_{DPO}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} $$
- This reward is always "grounded" in the reference policy.
- It cannot be arbitrarily large (unlike an explicit RM that might give infinite reward to adversarial inputs).

#### Q5: When would you choose RLHF over DPO?

**Answer:**
- **Iterative Improvement:** If you plan to collect new preference data on the improved model and iterate (like GPT-4), RLHF's explicit RM is easier to retrain.
- **Reward Inspection:** If you need to inspect, debug, or modify the reward function, having an explicit RM is valuable.
- **Online Learning:** If you want to collect preferences during deployment and continuously update, RLHF's RL framework is more natural.

---

### Production Challenges

#### Challenge 1: Length Bias in Preferences

**Scenario:** Your DPO-trained model always generates very long responses, even for simple questions.
**Root Cause:** The preference dataset has a bias: humans prefer longer, more detailed answers.
**Analysis:** $\log \pi(y|x) = \sum_{t=1}^T \log \pi(y_t | y_{<t}, x)$. Longer sequences have more terms.
**Solution:**
- **Length Normalization:** Divide log probs by sequence length.
- **Instruction:** Add "Be concise" to the system prompt during generation.
- **Data Curation:** Ensure the preference dataset includes examples where shorter is better.

#### Challenge 2: DPO Overfitting

**Scenario:** DPO training loss goes to zero, but the model performs worse on held-out prompts.
**Root Cause:** The model memorized the specific (prompt, chosen, rejected) triplets.
**Solution:**
- **Early Stopping:** Monitor accuracy on a validation set. Stop when it plateaus.
- **Fewer Epochs:** Train for 1 epoch instead of 3.
- **Regularization:** Add weight decay (1e-4).

#### Challenge 3: Reference Model Mismatch

**Scenario:** You trained SFT on Dataset A, but your preference data is from Dataset B. DPO performs poorly.
**Root Cause:** The reference policy $\pi_{ref}$ is out-of-distribution for the preference data.
**Solution:**
- **Unified SFT:** Ensure the SFT model is trained on a superset of the domains covered in the preference data.
- **Adaptive $\beta$:** Increase $\beta$ to rely more on the reference (more conservative).

#### Challenge 4: Debugging "Why is DPO not improving?"

**Scenario:** DPO loss decreases, but MT-Bench score doesn't improve.
**Diagnosis:**
1. **Check Accuracy:** Is the model correctly predicting preferences on the validation set? If accuracy is ~50%, the data might be noisy.
2. **Check KL:** Compute $D_{KL}(\pi_\theta || \pi_{ref})$. If it's too small (<0.01), increase $\beta$ or train longer. If it's too large (>1.0), the model drifted too far.
3. **Check Data Quality:** Are the preferences high-quality? Use GPT-4 to audit a sample.
4. **Check Evaluation:** Is MT-Bench the right metric? Try AlpacaEval or human eval.

#### Challenge 5: Combining DPO with SFT

**Scenario:** You want to do both SFT (on demonstrations) and DPO (on preferences) simultaneously.
**Solution:**
- **ORPO (Odds Ratio Preference Optimization):** A variant that combines SFT and DPO into a single loss.
$$ L = L_{SFT} + \lambda L_{DPO} $$
- **Sequential:** Do SFT first (1 epoch), then DPO (1 epoch). Repeat.

### Summary Checklist for Production
- [ ] **Data:** Ensure **10k+** high-quality preferences.
- [ ] **Reference:** Use the **SFT model** as $\pi_{ref}$.
- [ ] **Beta:** Start with **$\beta = 0.1$**.
- [ ] **Epochs:** Train for **1-2 epochs** (avoid overfitting).
- [ ] **Monitor:** Track **Accuracy** and **KL Divergence**.
- [ ] **Eval:** Use **MT-Bench** or **AlpacaEval**.
