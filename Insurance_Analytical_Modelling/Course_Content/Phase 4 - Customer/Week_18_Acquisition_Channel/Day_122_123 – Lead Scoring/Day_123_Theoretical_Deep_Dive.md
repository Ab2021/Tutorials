# Acquisition Funnel & Lead Scoring (Part 3) - Case Study & Implementation - Theoretical Deep Dive

## Overview
"The right offer, to the right person, at the right time."
We have the theory (Lead Scoring, Attribution). Now we build the **System**.
This Case Study focuses on a Multi-Line Insurer implementing a **Real-Time Next Best Action (NBA)** engine.

---

## 1. Conceptual Foundation

### 1.1 The Business Scenario

*   **Company:** "Global Insure Co."
*   **Problem:**
    *   Silos: Auto team sells Auto. Home team sells Home. No cross-talk.
    *   Waste: Customers get 5 conflicting emails a week.
    *   Churn: Customers leave because "you don't know me."
*   **Goal:** A Unified "Customer Decision Hub" that arbitrates all interactions.

### 1.2 The "Next Best Action" (NBA) Paradigm

*   **Old Way:** Product-Centric. "We need to sell 5000 Life policies. Blast the email list."
*   **New Way:** Customer-Centric. "What does John Doe need right now?"
    *   *Options:* Sell Auto? Sell Life? Service Claim? Do Nothing?
    *   *Arbitration:* Calculate $P(\text{Response}) \times \text{Value}$ for each option and pick the winner.

---

## 2. Mathematical Framework

### 2.1 The Arbitration Logic

For Customer $i$, we have a set of Actions $A = \{a_1, a_2, ..., a_k\}$.
We select the optimal action $a^*$ that maximizes Expected Value:

$$ a^* = \arg\max_{a \in A} \left( P(\text{Response}|i, a) \times \text{LTV}(i, a) - \text{Cost}(a) \right) $$

*   $P(\text{Response})$: Propensity Model (GBM).
*   $\text{LTV}$: Lifetime Value of the product.
*   $\text{Cost}$: Cost of the channel (Email = \$0.01, Call = \$5.00).

### 2.2 Contextual Bandits

*   **Problem:** Propensity models are static. They don't learn from today's feedback.
*   **Solution:** Thompson Sampling (Multi-Armed Bandit).
    *   Explore: Try a new offer on 10% of traffic.
    *   Exploit: Show the winner to 90% of traffic.

---

## 3. Theoretical Properties

### 3.1 The "360-Degree" View

*   **Inputs:**
    *   **Static:** Age, Income, Current Policies.
    *   **Real-Time:** "Just visited the 'Claims' page."
    *   **Context:** "It is raining in their Zip Code."
*   *Insight:* Context changes the NBA. If they are filing a claim, the NBA is *not* "Sell Life Insurance". It is "Help with Claim".

### 3.2 Latency Constraints

*   **Web Channel:** Decision must be made in < 200ms (before the page loads).
*   **Call Center:** Decision must be made in < 1s (before the agent says "Hello").

---

## 4. Modeling Artifacts & Implementation

### 4.1 System Architecture

1.  **Event Stream (Kafka):** "User clicked 'Quote'".
2.  **Feature Store (Feast):** Retrieve `user_history`.
3.  **Model Serving (Seldon/MLflow):** Score 5 Propensity Models in parallel.
4.  **Arbitration Engine (Rules):**
    *   *Rule 1:* If `claim_open == True`, suppress all Sales offers.
    *   *Rule 2:* If `credit_score < 600`, suppress "Premium Auto".
5.  **Response Capture:** Did they click? Feed back to Kafka.

### 4.2 Python "Toy" Arbitrator

```python
import pandas as pd

# 1. Candidate Offers
offers = [
    {"name": "Auto_Upsell", "value": 500, "cost": 1},
    {"name": "Life_CrossSell", "value": 2000, "cost": 5},
    {"name": "Retention_Discount", "value": 1000, "cost": 100}
]

# 2. Score Propensities (Mock Models)
def get_propensities(customer_id):
    return {
        "Auto_Upsell": 0.05,
        "Life_CrossSell": 0.01,
        "Retention_Discount": 0.20
    }

# 3. Arbitrate
def get_nba(customer_id):
    probs = get_propensities(customer_id)
    results = []
    for offer in offers:
        ev = (probs[offer["name"]] * offer["value"]) - offer["cost"]
        results.append((offer["name"], ev))
    
    # Sort by Expected Value
    return sorted(results, key=lambda x: x[1], reverse=True)[0]

print(get_nba("Cust_123"))
# Output: ('Retention_Discount', 100.0) -> (0.2 * 1000) - 100
```

---

## 5. Evaluation & Validation

### 5.1 The "Holdout" Group

*   **Global Control Group:** 5% of customers receive *random* offers (or no offers).
*   **Purpose:** Measure the *Incremental* Revenue of the NBA Engine.
*   **Metric:** $\text{Revenue}_{\text{NBA}} - \text{Revenue}_{\text{Random}}$.

### 5.2 Interaction History Analysis

*   **Fatigue Rules:**
    *   "Do not show the same offer more than 3 times in 7 days."
    *   *Validation:* Check if the engine respects this rule.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Creepy" Factor**
    *   *Scenario:* You know they just had a baby (from external data). You offer Life Insurance immediately.
    *   *Reaction:* Customer feels stalked.
    *   *Fix:* Soften the messaging. "Planning for the future?" instead of "Congrats on the baby!"

2.  **Trap: Channel Conflict**
    *   *Scenario:* Agent calls to sell Auto. Website offers a discount on Auto.
    *   *Result:* Agent looks stupid.
    *   *Fix:* Omni-channel State Management. The Agent must see what the Website offered.

---

## 7. Advanced Topics & Extensions

### 7.1 Real-Time Event Processing (Flink)

*   Detecting "Life Events" in real-time.
*   *Pattern:* `Change of Address` + `New Car Purchase` = High Propensity for "Umbrella Policy".

### 7.2 Reinforcement Learning (DQN)

*   Instead of separate Propensity Models, train a single Agent (DQN) to maximize Long-Term Reward (CLV).
*   The Agent learns the optimal *sequence* of actions.

---

## 8. Regulatory & Governance Considerations

### 8.1 Fairness in Marketing

*   **Risk:** The model only offers "Gold Plans" to White neighborhoods.
*   **Regulation:** Unfair Deceptive Acts and Practices (UDAP).
*   **Mitigation:** Test the NBA distribution across demographic groups.

---

## 9. Practical Example

### 9.1 Worked Example: The "Save" Team

**Scenario:**
*   **Trigger:** Customer visits "Cancellation" page.
*   **Real-Time Action:**
    1.  Event sent to NBA Engine.
    2.  Engine checks `churn_risk` (High) and `customer_value` (High).
    3.  Engine checks `budget` (Available).
    4.  **Decision:** "Pop up a Chat Window with a Senior Agent."
*   **Outcome:** Agent saves the customer.
*   **Alternative:** If `customer_value` was Low, decision might be "Let them go" (Automated Form).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Arbitration** chooses the winner.
2.  **Context** is King.
3.  **Latency** determines feasibility.

### 10.2 When to Use This Knowledge
*   **Enterprise Architecture:** Designing the Marketing Stack.
*   **Strategy:** Moving from "Push" to "Pull" marketing.

### 10.3 Critical Success Factors
1.  **Unified Data:** If you don't know they called yesterday, you fail.
2.  **Executive Buy-In:** Product heads hate losing control of "their" campaigns.

### 10.4 Further Reading
*   **Pega Systems:** "The Next Best Action paradigm".

---

## Appendix

### A. Glossary
*   **NBA:** Next Best Action.
*   **NBO:** Next Best Offer.
*   **CDP:** Customer Data Platform.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Expected Value** | $P \times V - C$ | Arbitration |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
