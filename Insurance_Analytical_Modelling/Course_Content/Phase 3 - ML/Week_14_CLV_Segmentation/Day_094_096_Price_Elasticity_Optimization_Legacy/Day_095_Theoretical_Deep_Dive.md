# Price Elasticity & Optimization (Part 2) - Optimization Algorithms - Theoretical Deep Dive

## Overview
"Finding the sweet spot."
Once we have a Risk Model (Cost) and a Demand Model (Elasticity), we have the ingredients.
Now we need the **Recipe**: How to combine them to maximize profit?
This day focuses on **Optimization Algorithms**: Linear Programming, Genetic Algorithms, and the "Efficient Frontier" of Pricing.

---

## 1. Conceptual Foundation

### 1.1 The Objective Function

*   **Maximize Profit:** $\sum (Price_i - Cost_i) \times Prob(Buy_i | Price_i)$.
*   **Maximize Revenue:** $\sum Price_i \times Prob(Buy_i | Price_i)$.
*   **Maximize Volume:** $\sum Prob(Buy_i | Price_i)$.
*   **Trade-off:** You cannot maximize all three.

### 1.2 Constraints

1.  **Profit Constraint:** Total Profit $\ge$ Target.
2.  **Volume Constraint:** Total Policies $\ge$ Target (to cover fixed costs).
3.  **Rate Capping:** Individual rate change $\in [-10\%, +10\%]$ (Regulatory/Customer impact).
4.  **Relativity Constraints:** "Gold Plan" must cost more than "Silver Plan".

---

## 2. Mathematical Framework

### 2.1 Non-Linear Programming (NLP)

*   **Problem:** The Profit function is non-linear (because $Prob(Buy)$ is Logistic).
*   **Solver:** Gradient Descent (SLSQP) or Newton-Raphson.
*   **Goal:** Find the vector of price adjustments $\Delta P$ that maximizes the objective.

### 2.2 Genetic Algorithms (Evolutionary)

*   **Why?** The profit landscape is "bumpy" (non-convex) due to complex constraints. Gradient descent gets stuck in local optima.
*   **Method:**
    1.  **Population:** Generate 100 random pricing strategies.
    2.  **Fitness:** Calculate Profit for each.
    3.  **Selection:** Keep the best 20.
    4.  **Crossover/Mutation:** Mix them to create new strategies.
    5.  **Repeat.**

---

## 3. Theoretical Properties

### 3.1 The Efficient Frontier

*   **Plot:** X-axis = Volume, Y-axis = Profit.
*   **Curve:** The set of optimal pricing strategies.
*   **Insight:** Any point *below* the curve is suboptimal (you could get more profit for the same volume).
*   **Decision:** Management picks a point on the curve (e.g., "Aggressive Growth" vs. "Harvesting Profit").

### 3.2 Global vs. Local Optimization

*   **Individual Optimization:** Optimize price for *each* customer independently. (Ideal but risky).
*   **Segment Optimization:** Optimize Base Rates for *groups* (e.g., Young Drivers). (More stable).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Simple Optimization (Scipy)

```python
from scipy.optimize import minimize
import numpy as np

# Inputs
costs = np.array([100, 200, 300])
elasticities = np.array([-2.0, -1.5, -1.0])
current_prices = np.array([120, 240, 350])

def demand(prices):
    # Simplified demand function
    return np.exp(elasticities * (prices - current_prices) / current_prices)

def objective(prices):
    # Negative Profit (for minimization)
    q = demand(prices)
    profit = np.sum((prices - costs) * q)
    return -profit

# Constraints: Prices > Costs
cons = ({'type': 'ineq', 'fun': lambda x: x - costs})

# Solve
res = minimize(objective, current_prices, constraints=cons)
print("Optimal Prices:", res.x)
```

### 4.2 Genetic Algorithm (DEAP)

*   **Library:** `deap` (Distributed Evolutionary Algorithms in Python).
*   **Use:** When you have complex "If-Then" constraints (e.g., "If Age < 25, Price must be < $500").

---

## 5. Evaluation & Validation

### 5.1 Impact Analysis

*   **Report:** Before deploying, generate an "Impact Report".
    *   "How many customers will see a >10% increase?"
    *   "How much volume will we lose?"
*   **Dislocation:** Visualizing the shift in the book of business.

### 5.2 Scenario Testing

*   **Stress Test:** "What if Competitor X drops their rates by 5%?"
*   **Robustness:** Does our optimal strategy collapse?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Spiral of Death"

*   **Scenario:** You optimize for Profit.
*   **Result:** You raise rates on risky segments. They leave.
*   **Next Cycle:** Your fixed costs are spread over fewer customers. You raise rates again.
*   **End Game:** You have 0 customers and infinite price.
*   **Fix:** Always include a **Volume Constraint**.

### 6.2 Overfitting to Elasticity

*   **Issue:** The Elasticity model has high variance.
*   **Result:** The optimizer exploits "noise" (e.g., charging \$1M to a segment the model wrongly thinks is perfectly inelastic).
*   **Fix:** Smooth the elasticity surface. Apply bounds.

---

## 7. Advanced Topics & Extensions

### 7.1 Dynamic Pricing (Real-Time)

*   **Context:** Travel Insurance or Usage-Based Auto.
*   **Method:** Contextual Bandits (Day 90).
*   **Update:** Adjust price every hour based on conversion rates.

### 7.2 Competitive Game Theory

*   **Concept:** If I lower my price, my competitor will react.
*   **Nash Equilibrium:** Finding the price where neither side wants to move.
*   **Simulation:** Agent-Based Modeling (ABM) of the market.

---

## 8. Regulatory & Governance Considerations

### 8.1 Rate Capping (Dislocation)

*   **Rule:** "No customer shall see a rate increase > 15% in one renewal."
*   **Implementation:**
    *   Calculate Optimal Price ($P^*$).
    *   Capped Price = $\min(P^*, P_{current} \times 1.15)$.
    *   **Transition:** Move to $P^*$ over 3 years.

---

## 9. Practical Example

### 9.1 The "Young Driver" Strategy

**Scenario:** Young drivers have high loss ratios (110%).
**Optimization:**
1.  **Risk Model:** Says raise rates 30%.
2.  **Demand Model:** Says they are highly elastic (will leave).
3.  **Constraint:** We need them because they become "Old Drivers" (Profitable) later.
4.  **Result:** Optimizer suggests raising rates only 10% (taking a loss now) to maximize **Lifetime Value** (Day 91).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Objective Function** defines success.
2.  **Constraints** define reality.
3.  **Efficient Frontier** visualizes the trade-off.

### 10.2 When to Use This Knowledge
*   **Product Management:** Designing new rate plans.
*   **Actuarial:** Filing rates.

### 10.3 Critical Success Factors
1.  **Constraint Management:** The art is in defining the constraints, not the objective.
2.  **Execution:** Can the IT system actually handle "Customer-Level Pricing"?

### 10.4 Further Reading
*   **Boyd & Vandenberghe:** "Convex Optimization".
*   **Towers Watson:** "Price Optimization in P&C Insurance".

---

## Appendix

### A. Glossary
*   **Dislocation:** The magnitude of rate changes experienced by the current book.
*   **Off-Balance:** The net revenue change after a rate revision.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Lagrangian** | $L(x, \lambda) = f(x) - \lambda g(x)$ | Constrained Optimization |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
