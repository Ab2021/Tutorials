# Day 36 Interview Questions: Transformers in RL

## Q1: How does the Decision Transformer differ from standard RL?
**Answer:**
*   **Standard RL (DQN, PPO):** Learns value functions or policies via bootstrapping and Bellman updates.
*   **Decision Transformer:** Treats RL as **supervised sequence modeling**. Predicts actions given (return-to-go, state, action) sequences.

No value functions, no temporal difference learning. Just autoregressive prediction.

## Q2: What is return conditioning in Decision Transformers?
**Answer:**
At test time, we can **condition** the model on a desired return $R^*$ to achieve that level of performance.
*   High $R^*$: Model generates expert-level actions.
*   Low $R^*$: Model generates suboptimal actions.

This allows controlling the agent's behavior at inference time without retraining.

## Q3: What is GATO?
**Answer:**
**GATO** (DeepMind, 2022) is a single Transformer trained on diverse tasks:
*   Atari games, image captioning, chat, robot control.
*   All inputs/outputs are tokenized sequences.
*   Demonstrates the feasibility of **generalist agents** that can handle multiple modalities and tasks.

## Q4: What is in-context reinforcement learning?
**Answer:**
Large Transformers can adapt to new RL tasks **in-context** without gradient updates:
*   Provide task demonstrations in the prompt: $(s, a, r)$ tuples.
*   The model infers the optimal policy from the context.
*   **Algorithm Distillation:** Train on many RL learning curves, model learns to "run RL algorithms" in-context.

## Q5: What are the advantages of Transformers for RL?
**Answer:**
*   **Long Context:** Model entire episode histories (thousands of steps).
*   **Multi-Modal:** Easily combine vision, language, proprioception.
*   **Transfer Learning:** Pretrain on large diverse datasets, fine-tune on specific tasks.
*   **In-Context Adaptation:** Few-shot learning via prompting.
*   **No Bootstrapping:** Avoids extrapolation errors in offline settings.

## Q6: What are the limitations?
**Answer:**
*   **Computational Cost:** Transformers are expensive (quadratic in sequence length).
*   **Sample Efficiency:** Still needs large offline datasets for pretraining.
*   **Context Window:** Limited to 2k-32k tokens (long episodes may not fit).
*   **Lack of Planning:** Pure sequence modeling doesn't explicitly plan ahead like MCTS.
