# Day 72: Negotiation & Contract Nets
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the "Winner's Curse" in Auctions?

**Answer:**
In a common-value auction (e.g., bidding for an oil field), the winner is often the one who *overestimated* the value the most.
*   **Result:** They win the auction but lose money.
*   **Agent Mitigation:** Agents should shade their bids (bid less than their estimated value) to account for uncertainty.

#### Q2: Why use a "Vickrey Auction" (Second-Price) for Agents?

**Answer:**
In a First-Price auction, agents try to "snipe" or guess the other bids to bid *just* slightly higher. This is computationally expensive and unstable.
In a Second-Price auction, the dominant strategy is to **bid your true value**.
*   If you bid true ($100) and win at $80 (second price), you keep $20 surplus.
*   If you bid lower ($90) and lose to someone bidding $95, you lost a profitable opportunity.
*   This simplifies the agent's logic significantly.

#### Q3: How do you prevent "Collusion" among agents?

**Answer:**
Agents might coordinate: "You bid low on Task A, I'll bid low on Task B."
*   **Anonymity:** Hide agent identities during bidding.
*   **Randomness:** Occasionally award the contract to the 2nd best bidder to keep them in the game.

#### Q4: What is "Expediting" in CNP?

**Answer:**
After the award, the Contractor must actually *do* the work.
Expediting is the monitoring phase.
*   Manager: "Status?"
*   Contractor: "50% done."
If the Contractor fails or is too slow, the Manager might cancel and re-award to the runner-up.

### Production Challenges

#### Challenge 1: The "Lying" Contractor

**Scenario:** Agent bids low ($10) to win, then claims "Unforeseen complications" and asks for $50.
**Root Cause:** Information Asymmetry.
**Solution:**
*   **Reputation System:** Track final cost vs bid price. If variance is high, penalize the agent's future bids.
*   **Escrow:** Lock the bid amount. No renegotiation allowed.

#### Challenge 2: Bid Spam

**Scenario:** Manager broadcasts a task. 10,000 agents send bids. Manager crashes processing them.
**Root Cause:** Open Broadcast.
**Solution:**
*   **Targeted Announcement:** Only invite Top-N trusted agents.
*   **Bid Fee:** Require a micro-payment to submit a bid.

#### Challenge 3: Deadlocks in Negotiation

**Scenario:** Agent A wants $100. Agent B offers $90. Neither budges.
**Root Cause:** Hard constraints.
**Solution:**
*   **Time-Dependent Concession:** "If time > 10s, lower ask by $1."
*   **Mediator:** Introduce a 3rd party to propose a split ($95).

### System Design Scenario: Ad Bidding Agent

**Requirement:** Real-time Bidding (RTB) for Ad slots.
**Design:**
1.  **Constraint:** < 100ms latency.
2.  **Logic:**
    *   Value = `p(click) * value_of_click`.
    *   Bid = `Value * strategic_factor`.
3.  **Budget:** "Spend max $100/day".
4.  **Pacing:** Don't blow the budget in the first hour. Use a PID controller to pace bids throughout the day.

### Summary Checklist for Production
*   [ ] **Timeouts:** Bidding window must close strictly.
*   [ ] **Tie-Breaking:** Define rule (Random? First received?).
*   [ ] **Reputation:** Use history to weight bids.
*   [ ] **Fallback:** What if no one bids? (Do it yourself or raise price).
