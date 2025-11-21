# Day 6: N-Step TD & Eligibility Traces

## 1. Bridging TD and MC
*   **TD(0):** Updates based on 1 step ($R_{t+1} + \gamma V(S_{t+1})$). Low variance, high bias.
*   **Monte Carlo:** Updates based on full episode ($G_t$). Zero bias, high variance.
*   **N-Step TD:** Updates based on n steps.
    $$ G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$
    $$ V(S_t) \leftarrow V(S_t) + \alpha [G_{t:t+n} - V(S_t)] $$
    This allows us to tune the bias-variance tradeoff.

## 2. Eligibility Traces ($\lambda$)
N-step TD is computationally annoying because we have to wait $n$ steps to make an update.
**Eligibility Traces** allow us to achieve the same effect (averaging over all n-step returns) with an incremental, per-step update.
*   **TD($\lambda$):** Averages all n-step returns weighted by $\lambda^{n-1}$.
*   If $\lambda=0$: We get TD(0).
*   If $\lambda=1$: We get Monte Carlo.

## 3. Backward View (Mechanism)
Instead of looking forward, we keep a "trace" $E_t(s)$ for every state.
*   When we visit state $s$, we bump up its trace: $E_t(s) \leftarrow E_t(s) + 1$ (Accumulating).
*   At every step, all traces decay: $E_{t+1}(s) \leftarrow \gamma \lambda E_t(s)$.
*   We update *all* states based on the current TD error $\delta_t$:
    $$ V(s) \leftarrow V(s) + \alpha \delta_t E_t(s) $$
This means "credit" for a reward is distributed back to recently visited states.

## 4. Code Example: TD($\lambda$) on Random Walk
1D Random Walk with 5 states.

```python
import numpy as np

# 5 states: 0, 1, 2, 3, 4. Start at 2.
# Left of 0 is Terminal (Reward 0). Right of 4 is Terminal (Reward 1).

def run_td_lambda(episodes=100, lam=0.5, alpha=0.1):
    V = np.zeros(5) + 0.5 # Init values
    
    for _ in range(episodes):
        E = np.zeros(5) # Eligibility Traces
        state = 2 # Start
        
        while True:
            # Random Walk Policy
            action = np.random.choice([-1, 1])
            next_pos = state + action
            
            reward = 0
            done = False
            
            if next_pos < 0: # Left terminal
                next_state = None
                reward = 0
                done = True
            elif next_pos > 4: # Right terminal
                next_state = None
                reward = 1
                done = True
            else:
                next_state = next_pos
                
            # TD Error
            target = reward
            if not done: target += V[next_state]
            delta = target - V[state]
            
            # Update Trace
            E[state] += 1
            
            # Update Values and Decay Traces
            for s in range(5):
                V[s] += alpha * delta * E[s]
                E[s] *= lam # Decay (gamma=1 here)
                
            if done: break
            state = next_state
            
    return V

print("TD(0.5) Values:", run_td_lambda(lam=0.5))
print("TD(0) Values:", run_td_lambda(lam=0.0)) # Should be slower
print("TD(1) Values:", run_td_lambda(lam=1.0)) # Like MC
```

### Key Takeaways
*   $\lambda$ controls the credit assignment decay.
*   Traces allow efficient backward updates.
*   Connects TD and MC smoothly.
