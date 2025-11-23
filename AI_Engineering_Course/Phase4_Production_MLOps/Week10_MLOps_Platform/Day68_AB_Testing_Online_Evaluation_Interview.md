# Day 68: A/B Testing & Online Evaluation
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between Offline and Online Evaluation?

**Answer:**
- **Offline:** Performed on a static test set (Golden Set). Metrics: Accuracy, BLEU, ROUGE. Fast, cheap, safe. Does not capture real user behavior or drift.
- **Online:** Performed on live traffic. Metrics: Click-through rate, Acceptance rate, Retention. Slow, risky, expensive. The ultimate ground truth.

#### Q2: Explain the "Explore-Exploit" trade-off in Multi-Armed Bandits.

**Answer:**
- **Explore:** Send traffic to unknown/new models to gather data on their performance. (Risk: User sees bad model).
- **Exploit:** Send traffic to the best-known model to maximize reward. (Risk: Miss out on a potentially better new model).
- **MAB Algorithms:** Thompson Sampling and UCB (Upper Confidence Bound) mathematically balance this trade-off to minimize regret.

#### Q3: How do you measure "Success" for a Chatbot?

**Answer:**
- **Hard:** There is no "click" to measure.
- **Metrics:**
  - **Sentiment:** Analyze user's tone in follow-up.
  - **Resolution Rate:** Did the user ask "Talk to human"?
  - **Turn Count:** Successful support chats are short. Successful entertainment chats are long. Context matters.
  - **Thumbs Up:** Explicit feedback (gold standard but sparse).

#### Q4: What is Interleaving and why is it better than A/B testing for ranking?

**Answer:**
- **A/B:** Split users. User Group A sees List A. User Group B sees List B. High variance because users are different. Requires large sample size.
- **Interleaving:** Every user sees a mixed list (A1, B1, A2, B2...). We measure which items are clicked.
- **Benefit:** Removes user variance. Much more sensitive (requires 10x fewer samples to detect difference).

#### Q5: What is Sample Ratio Mismatch (SRM)?

**Answer:**
- A common bug in A/B testing.
- You intended 50/50 split, but got 40/60.
- **Cause:** Bug in router, or one variant is crashing/slow and requests are dropped/timed out.
- **Result:** The test is invalid. You cannot trust the metrics if the sample ratio is wrong.

---

### Production Challenges

#### Challenge 1: Novelty Effect

**Scenario:** New model shows +10% engagement in the first week, then drops to -5%.
**Root Cause:** Users were curious about the change (e.g., new writing style), but the novelty wore off.
**Solution:**
- **Longer Tests:** Run A/B tests for at least 2-4 weeks (full business cycles).
- **Cohort Analysis:** Analyze "New Users" vs "Returning Users" separately.

#### Challenge 2: Sparse Feedback

**Scenario:** Only 0.1% of users click "Thumbs Up". Not enough data for significance.
**Root Cause:** UX friction.
**Solution:**
- **Implicit Signals:** Use "Copy to Clipboard" or "Insert Code" as proxies for positive feedback.
- **Active Learning:** Ask the user "Was this helpful?" only when the model is uncertain.

#### Challenge 3: Latency Confounding

**Scenario:** Model B is smarter but slower (500ms vs 200ms). Users prefer Model A because it's fast.
**Root Cause:** Latency impacts UX metrics.
**Solution:**
- **Latency Matching:** Artificially slow down Model A to match Model B during the test (controversial but scientifically accurate).
- **Metric:** Measure "Quality per Latency" or accept that speed is a feature.

#### Challenge 4: Carryover Effects

**Scenario:** User sees Model A today, Model B tomorrow. Their experience with A affects their behavior with B.
**Root Cause:** User memory/learning.
**Solution:**
- **Sticky Hashing:** Ensure a user stays in the same bucket (A or B) for the duration of the experiment.

#### Challenge 5: Cost of Bandits

**Scenario:** Bandit algorithm routes 90% traffic to GPT-4 (Best) and 10% to Llama-7B (Cheap). Cost explodes.
**Root Cause:** Bandit optimizes for *Reward* (Quality), ignoring *Cost*.
**Solution:**
- **Cost-Aware Bandit:** Modify reward function: `Reward = Quality - (Cost * Alpha)`.
- **Budget Constraint:** Cap the traffic to expensive models.

### System Design Scenario: A/B Testing Platform for LLMs

**Requirement:** Build a system to test prompts and models.
**Design:**
1.  **Config:** JSON config defining experiments (`{"exp_id": "prompt_v2", "variants": ["A", "B"]}`).
2.  **Router:** SDK in the application (Sidecar) that hashes UserID and fetches config.
3.  **Logger:** Async log stream (Kafka) capturing `(exp_id, variant, user_id, metric)`.
4.  **Analysis:** Data warehouse (Snowflake) job calculating P-values nightly.
5.  **Dashboard:** UI showing "Win Probability".

### Summary Checklist for Production
- [ ] **Routing:** Use **Sticky Hashing** (User ID).
- [ ] **Metrics:** Define **Primary** (Business) and **Guardrail** (Latency) metrics.
- [ ] **Safety:** Run **Canary** before A/B.
- [ ] **Significance:** Wait for **P-value < 0.05**.
- [ ] **Feedback:** Capture **Implicit Signals** (Copy/Edit).
