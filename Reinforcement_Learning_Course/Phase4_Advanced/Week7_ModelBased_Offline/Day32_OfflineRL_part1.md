# Day 32 Deep Dive: Offline RL Algorithms

## 1. Decision Transformer
Treats RL as **sequence modeling**.
*   Input: $(R_t, s_t, a_t, R_{t+1}, s_{t+1}, a_{t+1}, ...)$ where $R_t$ is return-to-go.
*   Use a Transformer to predict $a_t$ given the sequence.
*   At test time, condition on desired return $R^*$ to generate optimal behavior.

**Advantages:**
*   No value function bootstrapping (avoids extrapolation error).
*   Leverages powerful Transformer architecture.

## 2. AWAC: Advantage-Weighted Actor-Critic
Combines offline and online learning:
*   Policy update weighted by advantage:
    $$ \pi \leftarrow \arg\max_\pi \mathbb{E}_{(s,a) \sim \mathcal{D}} [\log \pi(a|s) \cdot \exp(A(s, a) / \beta)] $$
*   Actions with high advantage are upweighted.
*   Can fine-tune online after offline pretraining.

## 3. Behavior Regularization
Constrain the learned policy to stay close to the behavior policy:
$$ J(\pi) = \mathbb{E}[r] - \alpha \cdot D_{KL}(\pi || \pi_\beta) $$
*   Prevents the policy from drifting into OOD regions.
*   Examples: AWR, BRAC, TD3+BC.

## 4. Dataset Quality Matters
*   **Expert Data:** High performance, but BC may suffice.
*   **Random/Suboptimal Data:** Offline RL can improve, but harder.
*   **Mixed Data:** Most realistic. Offline RL excels here.
