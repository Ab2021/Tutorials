# Day 38 Interview Questions: Emerging Topics

## Q1: What is Unsupervised RL?
**Answer:**
Learning useful skills **without explicit rewards**:
*   **DIAYN:** Maximize mutual information $I(S; Z)$ between skills $Z$ and states $S$.
*   **Curiosity:** Use prediction error as intrinsic reward.
*   **Empowerment:** Learn actions that maximize control over future states.

**Goal:** Discover diverse, reusable skills that can later be composed for downstream tasks.

## Q2: What is Causal RL?
**Answer:**
Using causal inference to improve RL:
*   **Causal Models:** Explicitly model cause-effect relationships.
*   **Counterfactual Reasoning:** "What if I had taken action $a'$ instead?"
*   **Off-Policy Evaluation:** Estimate policy performance from logs without deployment.
*   **Transfer:** Causal structure helps generalize across environments.

## Q3: What is Continual Learning in RL?
**Answer:**
Learning **multiple tasks sequentially** without forgetting previous tasks.
**Challenge:** Catastrophic forgetting (new tasks overwrite old knowledge).
**Solutions:**
*   **EWC:** Penalize changes to important parameters.
*   **Progressive Networks:** Add new modules for each task.
*   **PackNet:** Partition network capacity across tasks.

## Q4: How are Graph Neural Networks used in RL?
**Answer:**
GNNs model relational structure:
*   **Multi-Agent RL:** Model interactions as graphs (agents = nodes, communication = edges).
*   **Combinatorial Optimization:** TSP, scheduling (cities/tasks = nodes).
*   **Molecular Design:** RL for drug discovery (atoms = nodes, bonds = edges).

GNNs enable generalization to graphs of different sizes.

## Q5: What are some recent breakthroughs in RL?
**Answer:**
*   **AlphaDev (2023):** RL discovers faster sorting algorithms than human-designed ones.
*   **Voyager (2023):** LLM-powered Minecraft agent that autonomously learns and explores.
*   **RT-2 (2023):** Vision-language-action model for robotic manipulation.
*   **Eureka (2023):** LLM generates reward functions for complex RL tasks.

## Q6: What are the biggest unsolved problems in RL?
**Answer:**
1. **Sample Efficiency:** Match human learning speed (few examples, not millions).
2. **Generalization:** Transfer across tasks, domains, and embodiments.
3. **Safety:** Ensure alignment and avoid catastrophic failures.
4. **Scalability:** Handle real-world complexity (huge state/action spaces, POMDP).
5. **Interpretability:** Understand why agents make decisions.
