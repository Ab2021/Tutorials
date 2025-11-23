# Day 77: Capstone: Building an Autonomous Trading Swarm
## Core Concepts & Theory

### The Ultimate A2A Challenge

We will combine **Communication**, **Negotiation**, **Decentralization**, and **Commerce** into a single system.
**Goal:** A swarm of agents that trade synthetic assets to make a profit.

### 1. The Roles

*   **Market Maker (MM):** Provides liquidity. "I buy at 99, sell at 101."
*   **Arbitrageur (Arb):** Looks for price discrepancies. "Buy from MM A at 100, sell to MM B at 102."
*   **Analyst:** Reads news (simulated) and predicts price movement. Sells signals to Traders.
*   **Trader:** Buys/Sells based on signals and prices.

### 2. The Infrastructure

*   **Registry:** Agents find each other.
*   **Communication:** FIPA-ACL (Request/Propose/Accept).
*   **Settlement:** Simulated Ledger (or Testnet Crypto).
*   **Environment:** A "Stock Exchange" server that maintains the Order Book.

### 3. The Workflow

1.  **Discovery:** Arb Agent queries Registry for "Market Makers".
2.  **Subscription:** Trader Agent pays Analyst Agent for "Signals".
3.  **Negotiation:** Trader requests quote from MM. MM proposes price. Trader accepts/rejects.
4.  **Execution:** Trade is recorded. Balances update.

### 4. Emergent Behavior

We expect to see:
*   **Efficiency:** Spreads tighten as Arbs compete.
*   **Specialization:** Analysts get better at predicting; MMs get better at pricing.
*   **Crash:** If all Analysts hallucinate the same signal, the market might crash (Flash Crash).

### Summary

This Capstone demonstrates the power of **Agent Economies**. It's not just about one agent doing a task; it's about a society of agents interacting to form a complex adaptive system.
