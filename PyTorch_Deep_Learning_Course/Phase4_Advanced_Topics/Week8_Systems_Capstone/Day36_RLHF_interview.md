# Day 36: RLHF & Alignment - Interview Questions

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: Alignment, RL, and Safety

### 1. Why do we need RLHF? Why not just SFT?
**Answer:**
*   SFT (Supervised Fine-Tuning) suffers from **Teacher Forcing**. It learns to mimic the exact words of the annotator.
*   RLHF optimizes the **outcome** (Reward). There are many ways to say the same correct thing. RLHF allows the model to explore and find the best way that satisfies the human.
*   SFT is about "What to say". RLHF is about "How to say it".

### 2. Explain the "Reward Model".
**Answer:**
*   A scalar regression model (usually initialized from the SFT model).
*   Input: (Prompt, Response). Output: Scalar Score.
*   Trained using Bradley-Terry model on pairwise comparisons: $L = -\log \sigma(r_{chosen} - r_{rejected})$.

### 3. What is "DPO"?
**Answer:**
*   Direct Preference Optimization.
*   Mathematically equivalent to RLHF but bypasses the explicit Reward Model.
*   Optimizes the policy to increase the likelihood of chosen responses relative to rejected ones, weighted by the reference model.

### 4. What is the "KL Penalty" in RLHF?
**Answer:**
*   A term added to the reward: $R(x, y) = r_{model}(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}$.
*   Prevents the model from drifting too far from the SFT distribution (Mode Collapse / Gibberish).
*   Prevents Reward Hacking.

### 5. What is "Mode Collapse" in RLHF?
**Answer:**
*   The model finds a single "safe" or "high-reward" response and outputs it for every prompt.
*   KL penalty mitigates this by forcing diversity (staying close to SFT).

### 6. What is "Constitutional AI"?
**Answer:**
*   Anthropic's approach.
*   Using AI to generate the feedback instead of humans.
*   The AI critiques responses based on a set of principles (Constitution).
*   Scalable alignment.

### 7. What is "PPO"?
**Answer:**
*   Proximal Policy Optimization.
*   Policy Gradient method.
*   Uses a "Trust Region" (clipping) to ensure updates are small and stable.

### 8. What is "Reward Hacking"?
**Answer:**
*   The agent exploiting flaws in the reward function to get high scores without doing the task.
*   Example: A summarization model adding random positive words because the reward model prefers positive sentiment.

### 9. What is "SFT"?
**Answer:**
*   Supervised Fine-Tuning.
*   Standard Cross-Entropy training on high-quality instruction data.
*   Step 1 of the InstructGPT pipeline.

### 10. How does DPO differ from PPO in memory usage?
**Answer:**
*   **PPO**: Needs 4 models (Actor, Critic, Ref, Reward). High VRAM.
*   **DPO**: Needs 2 models (Policy, Ref). Lower VRAM.

### 11. What is "Elo Rating" in LLMs?
**Answer:**
*   A metric to rank models based on pairwise battles (Chatbot Arena).
*   If Model A beats Model B, A gains points, B loses points.

### 12. What is "Red Teaming"?
**Answer:**
*   Adversarial testing.
*   Humans (or AIs) try to break the model (make it generate toxic/illegal content) to find vulnerabilities.

### 13. What is "Jailbreaking"?
**Answer:**
*   Prompt engineering techniques to bypass safety filters.
*   "Ignore previous instructions", "DAN mode".

### 14. What is "Iterative DPO"?
**Answer:**
*   Running DPO in rounds.
*   Train DPO $\to$ Generate new pairs with current model $\to$ Annotate $\to$ Train DPO again.
*   Approximates Online RL.

### 15. What is "ORPO"?
**Answer:**
*   Odds Ratio Preference Optimization.
*   Combines SFT and Alignment in a single stage.
*   Adds a penalty to the SFT loss to disfavor rejected responses.

### 16. Why is RLHF unstable?
**Answer:**
*   RL is inherently sensitive to hyperparameters.
*   Reward values are non-stationary (as policy changes, the distribution of responses changes).

### 17. What is "Spinning Up"?
**Answer:**
*   OpenAI's educational resource for Deep RL.
*   Good place to learn PPO math.

### 18. What is "Safety Training"?
**Answer:**
*   Specifically training the model to refuse harmful requests ("I cannot help with building a bomb").
*   Usually done via SFT on refusal examples + RLHF.

### 19. What is "Over-refusal"?
**Answer:**
*   When the model refuses harmless prompts due to over-aggressive safety training.
*   "Kill the process" $\to$ "I cannot help with killing".

### 20. What is "TRL"?
**Answer:**
*   Transformer Reinforcement Learning library by Hugging Face.
*   Implements PPO, DPO, RewardTrainer.
