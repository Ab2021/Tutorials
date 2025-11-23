# Day 102: Robotics & Embodied AI
## Core Concepts & Theory

### From Chatbots to Robots

Embodied AI is AI that interacts with the physical world.
*   **VLA (Vision-Language-Action):** Models that output *motor commands* instead of text.
*   **Moravec's Paradox:** High-level reasoning (Chess) is easy. Low-level sensorimotor skills (Walking) are hard.

### 1. RT-2 (Robotic Transformer 2)

Google's VLA model.
*   **Input:** Image + Text Command ("Pick up the apple").
*   **Output:** Tokenized Actions (`<open_gripper>`, `<move_x_10>`).
*   **Training:** Internet text/images + Robot demonstration data.
*   **Result:** Generalization. It can pick up a "dinosaur toy" even if it never saw one in the robot data, because it knows what a dinosaur is from the web data.

### 2. Sim-to-Real Transfer

Training robots in the real world is slow and dangerous.
*   **Simulation:** Train in Isaac Gym / MuJoCo (millions of FPS).
*   **Domain Randomization:** Randomize colors, friction, and lighting in Sim so the Real world looks like just another variation.

### 3. Foundation Models for Control

*   **PaLM-E:** Embodied multimodal language model.
*   **Eureka:** Using LLMs to write Reward Functions for Reinforcement Learning.

### Summary

Robotics is the ultimate test of Generalization. An LLM can hallucinate a fact, but a Robot cannot hallucinate a grasp (it will drop the cup).
