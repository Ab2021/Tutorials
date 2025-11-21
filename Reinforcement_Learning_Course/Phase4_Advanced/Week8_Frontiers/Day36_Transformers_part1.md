# Day 36 Deep Dive: In-Context RL and Prompting

## 1. In-Context Reinforcement Learning
Large Transformers can adapt to new tasks **in-context** (without gradient updates):
*   Provide examples in the prompt: $(s, a, r)$ tuples.
*   The model infers the task and acts optimally.
*   **Algorithm Distillation:** Train on many RL algorithm rollouts, model learns to "run RL in-context".

## 2. Prompt Engineering for RL
*   **Return Conditioning:** "Achieve return $R^*$".
*   **Natural Language Goals:** "Pick up the red cube".
*   **Few-Shot Demonstrations:** Show 3-5 examples of the task.

## 3. Multi-Task Pretraining
Train on diverse RL datasets (Atari, robotics, games):
*   Shared representations emerge.
*   Transfer to new tasks via fine-tuning or prompting.
*   **Video Pretraining:** Pretrain on YouTube videos, fine-tune on RL tasks.

## 4. Limitations
*   **Computational Cost:** Transformers are expensive for long sequences.
*   **Context Length:** Limited to 2k-32k tokens (episode length).
*   **Sample Efficiency:** Still needs large offline datasets.
