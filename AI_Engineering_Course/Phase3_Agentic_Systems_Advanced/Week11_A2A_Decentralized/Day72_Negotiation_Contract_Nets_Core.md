# Day 72: Negotiation & Contract Nets
## Core Concepts & Theory

### Why Negotiate?

In a centralized system, the controller decides: "Truck A goes to Location B."
In a decentralized system, Truck A is an independent agent. It might refuse because it's low on fuel or wants more money.
**Negotiation** is the process of reaching an agreement on terms (Price, Time, Quality).

### 1. The Contract Net Protocol (CNP)

The gold standard for task allocation.
*   **Manager:** Has a task to outsource.
*   **Contractor:** Can execute the task.
*   **Phases:**
    1.  **Announcement:** Manager broadcasts task spec.
    2.  **Bidding:** Contractors evaluate task and send bids.
    3.  **Awarding:** Manager evaluates bids and awards contract.
    4.  **Expediting:** Contractor performs task and reports result.

### 2. Auctions

*   **English Auction:** Ascending price. "Do I hear $10? $20?" (Open cry).
*   **Dutch Auction:** Descending price. "Price is $100... $90... $80." First to accept wins.
*   **Vickrey Auction (Second-Price):** Sealed bids. Winner pays the price of the *second-highest* bid. Encourages truthful bidding.

### 3. Bargaining (1-on-1)

*   **Alternating Offers Protocol:**
    *   Agent A: "I offer $100."
    *   Agent B: "Reject. I counter with $120."
    *   Agent A: "Accept."
*   **Strategy:** Agents need a "Reserve Price" (Walk-away point) and a "Concession Strategy" (How fast to lower demands).

### 4. Nash Equilibrium

In Game Theory, a state where no agent can improve their outcome by unilaterally changing their strategy.
Agents are usually "Self-Interested" (maximizing their own utility), not "Benevolent". The protocol must be designed so that the Nash Equilibrium aligns with the Global Good (Mechanism Design).

### Summary

Negotiation allows agents to resolve conflicts and allocate resources efficiently without a central dictator. It turns the system into a **Market Economy**.
