# Day 35 Interview Questions: Real-World RL

## Q1: What are the main challenges in deploying RL in the real world?
**Answer:**
1. **Sim-to-Real Gap:** Trained policies fail when transferred from simulation to reality due to discrepancies.
2. **Sample Efficiency:** Real-world interaction is expensive and slow.
3. **Safety:** Cannot explore dangerous actions (crashes, injuries).
4. **Robustness:** Must handle distribution shift, adversarial inputs.
5. **Interpretability:** Need to understand why the agent makes decisions.

## Q2: How do you bridge the sim-to-real gap?
**Answer:**
*   **Domain Randomization:** Train on diverse simulated environments to improve generalization.
*   **Realistic Simulation:** Use high-fidelity physics and rendering.
*   **Sim-to-Real Transfer Learning:** Fine-tune in the real world after sim pretraining.
*   **System Identification:** Learn simulator parameters from real data.
*   **Adversarial Training:** Make the policy robust to model errors.

## Q3: What is the standard pipeline for real-world RL deployment?
**Answer:**
1. **Simulation:** Train the initial policy in a simulator.
2. **Offline Pretraining:** Use historical data if available (offline RL).
3. **Online Fine-Tuning:** Carefully explore in the real system with safety constraints.
4. **Monitoring:** Track performance and constraint violations.
5. **Gradual Rollout:** A/B testing, staged deployment.
6. **Iteration:** Continuously collect data and improve.

## Q4: Give examples of successful RL deployments.
**Answer:**
*   **Google Data Centers:** 40% reduction in cooling energy using RL.
*   **AlphaStar:** Grandmaster level in StarCraft II.
*   **OpenAI Rubik's Cube:** Dexterous manipulation with a robot hand.
*   **Recommender Systems:** YouTube, Netflix use RL for recommendations.
*   **Waymo:** Autonomous driving trained in simulation and real-world testing.

## Q5: How do you ensure safety in real-world RL?
**Answer:**
*   **Constrained RL:** Enforce safety constraints (CPO).
*   **Offline Pretraining:** Learn from safe historical data first.
*   **Conservative Updates:** Use small policy steps (PPO, TRPO).
*   **Safety Layer:** Filter unsafe actions before execution.
*   **Human-in-the-Loop:** Allow human override and monitoring.
*   **Gradual Deployment:** Start in low-stakes scenarios, scale up gradually.

## Q6: What role does offline RL play in real-world deployment?
**Answer:**
Offline RL is critical for real-world applications:
*   **Leverage Historical Data:** Use existing logs and demonstrations.
*   **Safe Pretraining:** Learn a reasonable policy before real-world interaction.
*   **Sample Efficiency:** Avoid expensive and dangerous exploration.
*   **Fine-Tuning:** Use offline pretraining + online fine-tuning for best results.
