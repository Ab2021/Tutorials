# Day 7: Function Approximation

## 1. The Problem with Tables
So far, we used **Tabular Methods** (one entry per state).
*   **Limitation:** If $|S|$ is large (e.g., Go: $10^{170}$) or continuous (e.g., Robot joint angles), a table is impossible.
*   **Solution:** Approximate the value function using a parameterized function $V(s, \mathbf{w})$.
    $$ V(s) \approx \hat{v}(s, \mathbf{w}) $$
    where $\mathbf{w}$ is a weight vector with $d \ll |S|$ components.

## 2. Linear Function Approximation
The simplest form is a linear combination of features:
$$ \hat{v}(s, \mathbf{w}) = \mathbf{w}^T \phi(s) = \sum_{i=1}^d w_i \phi_i(s) $$
*   $\phi(s)$: Feature vector (e.g., position, velocity, polynomial terms).
*   $\mathbf{w}$: Weights to learn.

## 3. Gradient Descent
We want to minimize the Mean Squared Error (MSE) between the true value $v_{\pi}(s)$ and our approximation $\hat{v}(s, \mathbf{w})$.
$$ J(\mathbf{w}) = \mathbb{E}_{\pi} [ (v_{\pi}(s) - \hat{v}(s, \mathbf{w}))^2 ] $$
Update rule (Stochastic Gradient Descent):
$$ \mathbf{w} \leftarrow \mathbf{w} + \alpha [Target - \hat{v}(s, \mathbf{w})] \nabla \hat{v}(s, \mathbf{w}) $$
For linear FA, $\nabla \hat{v}(s, \mathbf{w}) = \phi(s)$.

## 4. Code Example: Linear FA for 1D State
Approximating a value function for a 1D random walk using polynomial features.

```python
import numpy as np
import matplotlib.pyplot as plt

# True Value Function (Unknown to agent)
def true_value(s):
    return np.sin(s) + 0.5 * s

# Feature Construction (Polynomials)
def get_features(s, order=3):
    # s is in range [-2, 2]
    return np.array([s**i for i in range(order + 1)])

# Linear FA Agent
class LinearAgent:
    def __init__(self, n_features, alpha=0.01):
        self.weights = np.zeros(n_features)
        self.alpha = alpha
        
    def predict(self, s):
        features = get_features(s)
        return np.dot(self.weights, features)
    
    def update(self, s, target):
        features = get_features(s)
        prediction = np.dot(self.weights, features)
        error = target - prediction
        self.weights += self.alpha * error * features

# Training Loop
agent = LinearAgent(n_features=4) # Cubic polynomial
states = np.linspace(-2, 2, 100)

for _ in range(1000): # 1000 samples
    s = np.random.choice(states)
    target = true_value(s) # Supervised setting for demo
    agent.update(s, target)

# Visualization
predictions = [agent.predict(s) for s in states]
targets = [true_value(s) for s in states]

print("Weights:", agent.weights)
# In a real notebook, we would plot this:
# plt.plot(states, targets, label='True')
# plt.plot(states, predictions, label='Approx')
```

### Key Takeaways
*   FA allows generalization to unseen states.
*   Linear FA is simple and has convergence guarantees (for on-policy).
*   Feature engineering ($\phi(s)$) is critical.
