# Day 77: Capstone: Building an Autonomous Trading Swarm
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you ensure "Fairness" in a decentralized exchange?

**Answer:**
*   **Front-Running:** A fast agent sees a large Buy Order and buys before it to sell higher.
*   **Solution:** Commit-Reveal Schemes. Agents submit a hashed order (Commit). Once all orders are in, they reveal the key. Or use a "Batch Auction" where all orders in a 1-second window are executed at the same clearing price.

#### Q2: What happens if the Ledger desynchronizes?

**Answer:**
*   **Double Spend:** Agent A spends the same coin twice.
*   **Solution:** This is why we need a Blockchain (Consensus). The Ledger must be the single source of truth, secured by Proof of Work/Stake. In a centralized simulation, a mutex lock on the `Ledger` class is enough.

#### Q3: How do you handle "Rogue Agents"?

**Answer:**
An agent that spams orders or exploits a bug in the MM logic.
*   **Circuit Breakers:** If price moves > 10% in 1 minute, halt trading.
*   **Rate Limits:** Max 10 orders per second per Agent ID.

#### Q4: Explain "Backtesting" for Agents.

**Answer:**
Running the swarm on historical data.
*   **Challenge:** In history, the price was X. But if your Agent bought huge volume, the price *would have changed* (Market Impact).
*   **Solution:** You need a "Market Simulator" that models impact, not just a "Data Replayer".

### Production Challenges

#### Challenge 1: Latency Arbitrage

**Scenario:** Agent A is co-located with the server (1ms). Agent B is on WiFi (50ms). Agent A wins every trade.
**Root Cause:** Physics.
**Solution:**
*   **Speed Bumps:** Artificially delay all orders to match the slowest (or a fixed 100ms delay).
*   **Colocation Services:** Charge for proximity.

#### Challenge 2: Debugging Emergent Bugs

**Scenario:** The market crashes every Tuesday at 10 AM.
**Root Cause:** Complex interaction of 50 agents.
**Solution:**
*   **Agent Replay:** Record the seed and message log of every agent. Replay the exact sequence to reproduce the crash.

#### Challenge 3: Economic Exploits

**Scenario:** Agents learn to "Wash Trade" (buy/sell to themselves) to generate fake volume and earn rewards.
**Root Cause:** Bad incentive design.
**Solution:**
*   **Fee:** Charge a fee per trade. Wash trading becomes expensive.

### System Design Scenario: High-Frequency Trading (HFT) Agent

**Requirement:** Execute trades in microseconds.
**Design:**
1.  **Language:** C++ or Rust (Python is too slow).
2.  **Hardware:** FPGA (Field Programmable Gate Array) for logic.
3.  **Strategy:** Simple logic (if price > X buy). No LLMs in the hot path.
4.  **LLM Role:** The LLM runs "Offline" to analyze data and update the C++ parameters every hour.

### Summary Checklist for Production
*   [ ] **Audit:** Check smart contracts for re-entrancy bugs.
*   [ ] **Limits:** Set max position size.
*   [ ] **Kill Switch:** A master button to stop all agents.
*   [ ] **Logs:** Record every tick and message for compliance.
