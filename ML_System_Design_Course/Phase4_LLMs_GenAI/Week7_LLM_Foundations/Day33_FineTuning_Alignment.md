# Day 33: Fine-Tuning & Alignment

> **Phase**: 4 - LLMs & GenAI
> **Week**: 7 - LLM Foundations
> **Focus**: Making Models Helpful
> **Reading Time**: 50 mins

---

## 1. The Alignment Pipeline

A pretrained model (Base Model) is a "document completer." It will complete "The capital of France is" with "a beautiful city." We want it to answer "Paris."

### 1.1 Supervised Fine-Tuning (SFT)
*   **Data**: (Instruction, Response) pairs.
*   **Process**: Train the model to predict the Response given the Instruction.
*   **Result**: An "Instruction Tuned" model.

### 1.2 RLHF (Reinforcement Learning from Human Feedback)
*   **Step 1**: Train a **Reward Model (RM)**. Humans rank 2 responses (A vs B). RM learns to predict the score.
*   **Step 2**: Optimize the LLM (Policy) to maximize the Reward using PPO (Proximal Policy Optimization).
*   **Goal**: Align with human preferences (Helpfulness, Safety) that are hard to define mathematically.

---

## 2. Modern Alternatives: DPO

### 2.1 Direct Preference Optimization (DPO)
*   **Idea**: RLHF is unstable and complex (requires training a separate Reward Model and PPO).
*   **DPO**: Optimizes the policy *directly* on the preference data (A > B).
*   **Math**: Derives a loss function that implicitly maximizes the reward without needing an explicit Reward Model.
*   **Status (2025)**: The default standard. Used by Llama 3, Mistral.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: The Alignment Tax
**Scenario**: After RLHF, the model refuses to answer innocent questions ("How to kill a process in Linux?") or its coding ability degrades.
**Solution**:
*   **Mix SFT Data**: Keep high-quality coding/reasoning data in the RLHF training mix to prevent catastrophic forgetting.

### Challenge 2: Reward Hacking
**Scenario**: The model learns that using long, polite sentences gets higher rewards, even if the answer is wrong.
**Solution**:
*   **KL Penalty**: Add a penalty term to keep the RLHF model close to the SFT model. Don't let it drift too far.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Explain the difference between SFT and RLHF.**
> **Answer**:
> *   **SFT**: Learning to *mimic* a demonstration. "Write like this."
> *   **RLHF**: Learning to *maximize* a score. "Write something that gets a high rating." RLHF often leads to better performance because it's easier for humans to rate an answer than to write a perfect one.

**Q2: Why is DPO preferred over PPO in 2025?**
> **Answer**: DPO is mathematically equivalent to PPO (under certain assumptions) but is much simpler to implement and more stable to train. It removes the need for a separate Reward Model and the complex hyperparameter tuning of PPO.

**Q3: What is "Catastrophic Forgetting" in Fine-Tuning?**
> **Answer**: When fine-tuning on a specific task (e.g., Medical), the model forgets general knowledge (e.g., Python coding). We mitigate this by using LoRA (freezing weights) or mixing general data into the fine-tuning dataset.

---

## 5. Further Reading
- [Direct Preference Optimization (DPO) Paper](https://arxiv.org/abs/2305.18290)
- [Illustrating RLHF (Hugging Face)](https://huggingface.co/blog/rlhf)
