# Day 8 Interview Questions: Multi-Armed Bandits

## Q1: What is the fundamental difference between a Multi-Armed Bandit and a full MDP?
**Answer:**
*   **Bandit:** No state transitions. The action affects the immediate reward but does not influence the next situation the agent faces. (Or, there is only 1 state).
*   **MDP:** Actions influence both the immediate reward AND the next state. The agent must consider long-term consequences (delayed rewards).

## Q2: What is "Regret" in the context of Bandits?
**Answer:**
Regret is the difference between the total reward the agent *could* have gotten (if it always chose the optimal arm) and the total reward it *actually* got.
$$ \text{Regret}_T = \sum_{t=1}^T (q_* - q(A_t)) $$
The goal of a bandit algorithm is to minimize the cumulative regret over time (ideally logarithmic growth).

## Q3: Why is UCB often better than $\epsilon$-Greedy?
**Answer:**
$\epsilon$-Greedy explores indiscriminately. It will pull the worst arm with probability $\epsilon/k$ forever, even if it knows it's bad.
UCB explores based on **uncertainty**. As it learns more about an arm, the confidence interval shrinks, and it stops exploring sub-optimal arms, leading to lower regret in the long run.

## Q4: Explain the concept of "Contextual Bandits".
**Answer:**
It's a middle ground between Bandits and MDPs.
The agent observes a state (context) $S_t$ (e.g., user features), takes an action $A_t$ (e.g., show ad), and gets a reward $R_t$.
However, the action does not affect the next context $S_{t+1}$ (which is usually drawn from a distribution or determined by the next user arriving).
It's widely used in personalization and ad targeting.

## Q5: What is the "Cold Start" problem?
**Answer:**
It occurs when a new arm (or new user in contextual bandits) is introduced. We have no data ($N=0$), so estimates are undefined.
Algorithms like UCB or Thompson Sampling handle this naturally by assigning high uncertainty (or high prior probability) to new items, encouraging immediate exploration.
