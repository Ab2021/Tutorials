# Day 15 Interview Questions: Rainbow DQN

## Q1: List the 7 components of Rainbow DQN.
**Answer:**
1.  **DQN:** Base algorithm.
2.  **Double DQN:** Fixes overestimation.
3.  **Prioritized Experience Replay (PER):** Focuses on hard samples.
4.  **Dueling Networks:** Generalizes across actions.
5.  **Multi-Step Learning (N-step):** Faster reward propagation.
6.  **Distributional RL (C51):** Learns return distributions.
7.  **Noisy Nets:** Better exploration.

## Q2: According to the ablation studies, which component is the most important?
**Answer:**
**Prioritized Experience Replay** and **Multi-step Learning** were found to be the most crucial. Removing them caused the most significant performance drops. Distributional RL was also very important.

## Q3: Why might Double DQN be less important in Rainbow than in vanilla DQN?
**Answer:**
Distributional RL (C51) already mitigates overestimation bias. By learning the full distribution, the "max" operator behaves differently (operating on distributions rather than scalar means), which naturally stabilizes the value estimates. Therefore, the specific mechanism of Double DQN becomes redundant.

## Q4: What is the main downside of Rainbow DQN?
**Answer:**
**Complexity.** Implementing and tuning 7 interacting components is difficult. It introduces many hyperparameters and makes the code harder to debug and maintain. In research, simpler baselines (like PPO or SAC) are often preferred for their ease of use.

## Q5: How does Multi-Step Learning (N-step) help in Deep RL?
**Answer:**
It bridges the gap between MC and TD.
By looking $n$ steps ahead ($R_{t+1} + \gamma R_{t+2} + ... + \gamma^n \max Q(S_{t+n}, a)$), we propagate reward information faster. This is especially helpful in environments with sparse or delayed rewards, as the signal reaches the start state in fewer updates.
