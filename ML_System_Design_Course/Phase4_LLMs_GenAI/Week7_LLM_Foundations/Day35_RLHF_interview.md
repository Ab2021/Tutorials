# Day 35: RLHF - Interview Questions

> **Topic**: Alignment
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is RLHF (Reinforcement Learning from Human Feedback)?
**Answer:**
*   Method to align LLMs with human intent (Helpful, Honest, Harmless).
*   Steps: SFT -> Reward Model -> PPO.

### 2. Explain the 3 steps of RLHF (InstructGPT).
**Answer:**
1.  **SFT (Supervised Fine-Tuning)**: Train on human demonstrations.
2.  **RM (Reward Model)**: Train model to rank outputs based on human preference.
3.  **RL (PPO)**: Optimize Policy to maximize Reward.

### 3. Why not just use SFT? Why do we need RL?
**Answer:**
*   SFT suffers from **Exposure Bias**.
*   Humans are better at *judging* (Ranking) than *writing* (Demonstration).
*   RL allows exploring the space of answers to find what humans like best.

### 4. What is a Reward Model?
**Answer:**
*   A BERT/GPT model that takes `(Prompt, Response)` and outputs a scalar score.
*   Trained on comparison data ($A > B$).
*   Loss: $- \log(\sigma(r_A - r_B))$.

### 5. Explain PPO (Proximal Policy Optimization) in context of LLMs.
**Answer:**
*   Update policy $\pi$ to maximize reward.
*   **Constraint**: Don't drift too far from original SFT model (KL Divergence).
*   Prevents "Reward Hacking".

### 6. What is Reward Hacking?
**Answer:**
*   Model finds a loophole to get high reward without being helpful.
*   Example: Repeating "I love you" if the reward model likes positive sentiment.
*   Fixed by KL penalty.

### 7. What is DPO (Direct Preference Optimization)?
**Answer:**
*   Alternative to RLHF. No separate Reward Model. No PPO.
*   Optimizes policy directly on preference data using a simple classification loss.
*   More stable and efficient.

### 8. What is the KL Divergence penalty?
**Answer:**
*   Term added to Reward: $R(x, y) - \beta \log(\frac{\pi(y|x)}{\pi_{ref}(y|x)})$.
*   Penalizes if current policy $\pi$ produces text that is very unlikely under reference policy $\pi_{ref}$.

### 9. How do you collect Human Feedback data?
**Answer:**
*   Show labelers two model outputs for same prompt.
*   Ask: "Which is better?".
*   Rankings are converted to pairwise comparisons.

### 10. What is "Constitutional AI" (Anthropic)?
**Answer:**
*   RLAIF (RL from AI Feedback).
*   Use a strong model (Constitution) to critique and revise its own outputs.
*   Train Reward Model on these AI-generated preferences.
*   Scalable alignment.

### 11. What are the challenges of RLHF?
**Answer:**
*   **Instability**: PPO is hard to tune.
*   **Cost**: Human labeling is expensive.
*   **Ambiguity**: Humans disagree on what is "better".

### 12. What is "Rejection Sampling" (Best-of-N)?
**Answer:**
*   Generate N outputs.
*   Score them with Reward Model.
*   Pick the best one.
*   Simple alternative to PPO.

### 13. What is the difference between "Helpful" and "Harmless"?
**Answer:**
*   **Helpful**: Follows instruction.
*   **Harmless**: Does not generate toxic/dangerous content.
*   Often a trade-off (The "Waluigi Effect").

### 14. Explain the Bradley-Terry model.
**Answer:**
*   Probabilistic model for pairwise comparisons.
*   $P(A > B) = \frac{e^{r_A}}{e^{r_A} + e^{r_B}}$.
*   Used to train Reward Model.

### 15. What is "Mode Collapse" in RLHF?
**Answer:**
*   Model starts generating the same generic "safe" response to everything.
*   Due to excessive KL penalty or poor reward model.

### 16. How does DPO compare to PPO?
**Answer:**
*   **DPO**: Simpler, less memory, stable.
*   **PPO**: Theoretically more general, can use non-differentiable rewards.

### 17. What is "Offline RL"?
**Answer:**
*   Learning from a fixed dataset of interactions without interacting with environment.
*   DPO is essentially offline RL.

### 18. Can you use RLHF for coding tasks?
**Answer:**
*   Yes. Reward can be "Does code compile?" + "Does it pass unit tests?".
*   Objective ground truth makes it easier than chat.

### 19. What is the "Alignment Tax"?
**Answer:**
*   Performance on standard benchmarks (math, logic) might drop after RLHF (alignment).
*   Model becomes "dumber" but "nicer".

### 20. How do you evaluate an RLHF model?
**Answer:**
*   **Elo Rating**: Chatbot Arena (Humans vote).
*   **GPT-4 as Judge**: Ask GPT-4 to rate responses.
