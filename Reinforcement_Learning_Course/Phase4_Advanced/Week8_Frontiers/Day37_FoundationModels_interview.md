# Day 37 Interview Questions: Foundation Models for RL

## Q1: How can LLMs be used in RL?
**Answer:**
1. **Planning:** Generate high-level task plans (SayCan).
2. **Reward Design:** Generate reward functions (Eureka).
3. **Code Generation:** Write control policies as code (Code as Policies).
4. **Reasoning:** Explain failures and suggest fixes (Inner Monologue).
5. **Natural Language Interface:** Accept instructions in natural language.

## Q2: What is a Vision-Language-Action (VLA) model?
**Answer:**
A model that jointly processes vision, language, and outputs actions:
*   **Input:** Images + natural language instructions.
*   **Output:** Robot actions (joint angles, gripper commands).
*   **Example:** RT-2 (Robotic Transformer 2) from Google.

VLAs enable robots to follow complex instructions like "Pick up the apple and put it in the bowl."

## Q3: What is PaLM-E?
**Answer:**
**PaLM-E** (Google, 2023) is a 588B parameter embodied multimodal LLM:
*   Trained on vision, language, and robotics data.
*   Can control robots, answer questions about the environment, and plan.
*   Demonstrates the power of scaling up multimodal foundation models.

## Q4: How does SayCan work?
**Answer:**
**SayCan** combines LLM planning with RL execution:
1. **LLM:** Generates high-level task plan from natural language ("Clean the table").
2. **Affordance Function (RL):** Checks which skills are feasible given the current state.
3. **Execute:** Run RL policies for each skill in the plan.
4. **Replan:** If a skill fails, ask LLM to replan.

## Q5: What is Eureka (NVIDIA)?
**Answer:**
**Eureka** uses LLMs to automatically generate reward functions for RL:
*   Input: Task description (natural language or code skeleton).
*   LLM generates candidate reward functions.
*   Evaluate rewards via RL training in simulation.
*   Iterate and refine.

Achieves better performance than hand-designed rewards on complex tasks (e.g., pen spinning for robot hand).

## Q6: What are the challenges of using Foundation Models for robotics?
**Answer:**
*   **Computational Cost:** LLMs are slow, not suitable for real-time low-level control.
*   **Data Requirements:** VLA models need large robot datasets.
*   **Hallucinations:** LLMs can generate unsafe or infeasible plans.
*   **Sim-to-Real Gap:** Models trained on internet data may not transfer well.
*   **Interpretability:** Hard to debug failures in large blackbox models.
