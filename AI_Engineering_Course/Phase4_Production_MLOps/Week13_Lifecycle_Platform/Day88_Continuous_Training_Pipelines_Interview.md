# Day 67: Continuous Training (CT) Pipelines
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is Catastrophic Forgetting and how do you prevent it in CT?

**Answer:**
- **Problem:** When a neural network is trained on new data, it tends to overwrite weights optimized for old data, losing previously learned capabilities.
- **Prevention:**
  - **Replay Buffer:** Mix a subset of old data with new data during training.
  - **Parameter Efficient Tuning (LoRA):** Train new adapters for new data, keeping the base model frozen. Or train one adapter on mixed data.
  - **Regularization:** Use EWC (Elastic Weight Consolidation) to penalize changes to important weights.

#### Q2: What is the difference between Model Drift and Data Drift?

**Answer:**
- **Data Drift (Covariate Shift):** The distribution of input data $P(X)$ changes. (e.g., Users start asking about "GPT-5" instead of "GPT-4").
- **Model Drift (Concept Drift):** The relationship between input and output $P(Y|X)$ changes. (e.g., The definition of "spam" changes over time).
- **Impact:** Both lead to performance degradation.

#### Q3: Why use Shadow Deployment instead of Canary?

**Answer:**
- **Shadow:** Runs the new model on real traffic *without* returning the response to the user. Zero risk to user experience. Good for verifying stability and latency.
- **Canary:** Returns response to a small % of users. Low risk, but non-zero. Good for verifying business metrics (conversion rate).
- **Order:** Usually Shadow -> Canary -> Full Rollout.

#### Q4: How do you automate the "Evaluation" step in a CT pipeline?

**Answer:**
- **Golden Set:** Run the new model on a curated set of Q&A pairs.
- **LLM-as-a-Judge:** Use GPT-4 to grade the new model's responses against the Golden Set or against the previous model's responses.
- **Metrics:** Check exact match, semantic similarity, or custom criteria (e.g., "politeness").

#### Q5: When should you *not* use Continuous Training?

**Answer:**
- **Stable Domain:** If the data distribution doesn't change (e.g., OCR for standard fonts).
- **High Cost:** If training is too expensive compared to the marginal gain.
- **Risk:** If the cost of a bad model deployment is catastrophic (e.g., medical diagnosis) and automated eval is unreliable.

---

### Production Challenges

#### Challenge 1: Feedback Loop Bias

**Scenario:** Model predicts "A". User clicks "A". We train on this. Model becomes confident in "A".
**Root Cause:** Self-reinforcing loop. The model only sees feedback on its own predictions, never on "B".
**Solution:**
- **Exploration:** Occasionally show random/different options (Epsilon-Greedy) to gather data on other paths.
- **Importance Sampling:** Weight the training data to correct for bias.

#### Challenge 2: Training Pipeline Flakiness

**Scenario:** CT pipeline fails 50% of the time due to OOM or network issues.
**Root Cause:** Fragile infrastructure.
**Solution:**
- **Checkpointing:** Save intermediate states.
- **Retries:** Configure automatic retries in Airflow.
- **Resource Isolation:** Use dedicated nodes for training.

#### Challenge 3: "Poisoned" Data entering CT

**Scenario:** A malicious user spams the bot with bad data. CT picks it up and ruins the model.
**Root Cause:** Lack of data sanitation.
**Solution:**
- **Outlier Detection:** Remove data points with high loss or unusual embedding distance.
- **Rate Limiting:** Cap the contribution of any single user to the training set.

#### Challenge 4: Version Explosion

**Scenario:** CT runs daily. After a year, you have 365 LoRA adapters. Storage and management become a mess.
**Root Cause:** No cleanup policy.
**Solution:**
- **Pruning:** Keep only weekly/monthly snapshots.
- **Merging:** Periodically merge the LoRA adapter into the base model to reset.

#### Challenge 5: Evaluation Latency

**Scenario:** Training takes 1 hour. Evaluation (using GPT-4 judge) takes 10 hours and costs $500.
**Root Cause:** Expensive evaluation step.
**Solution:**
- **Tiered Eval:** Run cheap metrics (BLEU/ROUGE) first. If pass, run small subset of GPT-4 eval. If pass, run full eval.
- **Smaller Judge:** Use a fine-tuned 7B model as a judge instead of GPT-4.

### System Design Scenario: News Summarizer CT

**Requirement:** Summarizer must stay up-to-date with new names/events.
**Design:**
1.  **Ingest:** Daily news articles.
2.  **Dataset:** Create (Article, Summary) pairs using a larger model (Distillation).
3.  **Train:** Fine-tune Llama-7B (LoRA) nightly.
4.  **Eval:** Check ROUGE score on "Yesterday's News" Golden Set.
5.  **Deploy:** Shadow deploy for 1 hour, then swap.

### Summary Checklist for Production
- [ ] **Trigger:** Automate based on **Drift** or **Schedule**.
- [ ] **Data:** Use **Replay Buffer** to prevent forgetting.
- [ ] **Eval:** Implement **Automated Gates** (LLM-as-Judge).
- [ ] **Safety:** Use **Shadow Deployment**.
- [ ] **Monitoring:** Alert on **Pipeline Failures**.
