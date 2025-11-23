# Day 68: A/B Testing & Online Evaluation
## Core Concepts & Theory

### The Truth is in Production

**Offline vs Online:**
- **Offline Eval:** Test set metrics (BLEU, Accuracy). Good for sanity check.
- **Online Eval:** Real user metrics (Click-through, Acceptance Rate, Dwell Time). The ultimate truth.

**Why A/B Test?**
- Offline metrics often don't correlate with user preference.
- A model with higher perplexity might actually write better emails.

### 1. A/B Testing Fundamentals

**Concept:**
- Split traffic into buckets (e.g., Control 50%, Treatment 50%).
- **Control (A):** Current Production Model.
- **Treatment (B):** New Candidate Model.
- Compare metrics after statistical significance is reached.

**Routing:**
- **User-based:** Hash(UserID) % 100. Ensures consistent experience.
- **Session-based:** Hash(SessionID). Good for anonymous users.

### 2. Online Metrics for LLMs

**Explicit Feedback:**
- **Thumbs Up/Down:** Strong signal, but sparse (few users click).
- **Rewrite:** User edits the generated text. (Strong negative signal).
- **Copy/Paste:** User copies the text. (Strong positive signal).

**Implicit Feedback:**
- **Acceptance Rate:** (Coding Copilot) Did user press Tab?
- **Conversation Length:** Longer might be better (engagement) or worse (confusion).
- **Sentiment Analysis:** Analyze user's follow-up message ("That's wrong", "Thanks").

### 3. Multi-Armed Bandits (MAB)

**Concept:**
- Adaptive A/B testing.
- Instead of fixed 50/50 split, dynamically route more traffic to the winning model.
- **Explore-Exploit:** Explore new models, Exploit the best one.
- **Algorithms:** Thompson Sampling, Epsilon-Greedy.

**Benefit:**
- Minimizes "regret" (users exposed to bad model).
- Faster convergence.

### 4. Interleaving

**Concept:**
- Show results from Model A and Model B in the *same* list (e.g., Search results, RAG citations).
- See which one the user clicks.
- **Benefit:** Removes user bias. Very powerful for ranking/retrieval.

### 5. LLM-Specific Online Eval

**Model-Based Evaluation:**
- Sample 1% of production logs.
- Send (Prompt, Response A, Response B) to GPT-4.
- Ask "Which response is better?".
- **Pros:** Scalable, detailed feedback.
- **Cons:** Cost, bias of the judge model.

### 6. Statistical Significance

**The Math:**
- You need enough samples to prove the difference isn't noise.
- **P-value:** Probability that the difference is random chance. Target < 0.05.
- **Power:** Probability of detecting a real difference.

### 7. Canary Deployments

**Safety First:**
- Before A/B test, do a Canary.
- **Step 1:** Route 1% traffic to New Model.
- **Step 2:** Monitor Errors/Latency.
- **Step 3:** If healthy, proceed to 50/50 A/B test.

### 8. Feature Flags

**Tooling:**
- Use Feature Flags (LaunchDarkly, Split) to control routing.
- Decouples deployment (code push) from release (traffic shift).

### 9. Shadow Mode (Revisited)

- Run Model B on 100% traffic but *discard* output.
- Compare latency and errors.
- Compare output similarity with Model A.

### 10. Summary

**Evaluation Strategy:**
1.  **Offline:** Pass Golden Set.
2.  **Shadow:** Verify stability.
3.  **Canary:** Verify safety (1%).
4.  **A/B Test:** Verify business value (50%).
5.  **Bandit:** Optimize long-term.

### Next Steps
In the Deep Dive, we will implement a Traffic Router for A/B testing, a Bandit algorithm, and an Online Metric Logger.
