# Day 76: Agent Commerce & Payments
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why can't Agents use Stripe?

**Answer:**
*   **Legal:** Stripe requires a legal entity (Human/Company) + SSN/EIN. Agents are software.
*   **Risk:** Agents might trigger fraud detection algorithms with high-frequency micro-transactions.
*   **Currency:** Stripe deals in Fiat ($). Agents prefer programmable money (Crypto) for smart contract integration.

#### Q2: What is "Gas" and how do Agents handle it?

**Answer:**
Gas is the fee to execute a transaction on blockchain.
*   **Problem:** If Agent runs out of Gas (ETH), it freezes.
*   **Solution:**
    *   **Gas Station Network:** User pays gas for the agent (Meta-Transactions).
    *   **Auto-Topup:** A monitoring service sends ETH to the agent when balance < 0.01.

#### Q3: Explain "Atomic Swap" in Agent Trading.

**Answer:**
Agent A wants Data. Agent B wants Money.
*   **Risk:** A pays, B doesn't send Data.
*   **Atomic Swap:**
    *   A locks Money in contract.
    *   B reveals Data Key to unlock Money.
    *   The operation happens "Atomically" (All or Nothing). If B doesn't reveal Key, Money is refunded to A.

#### Q4: How do you tax an Agent?

**Answer:**
Unsolved legal problem.
*   **Current View:** The owner of the agent is liable for taxes on the agent's income.
*   **Future View:** DAO (Decentralized Autonomous Organization) structures where the agent pays its own taxes via smart contract logic.

### Production Challenges

#### Challenge 1: Volatility

**Scenario:** Agent prices service at 0.01 ETH. ETH crashes 20%. Agent loses money on server costs (USD).
**Root Cause:** Crypto volatility.
**Solution:**
*   **Stablecoins:** Use USDC/USDT for all pricing and settlement.
*   **Real-time Oracle:** Adjust ETH price dynamically based on current USD exchange rate.

#### Challenge 2: Private Key Security

**Scenario:** Agent is hacked. Wallet drained.
**Root Cause:** Hot Wallet on server.
**Solution:**
*   **Spending Limits:** The hot wallet only holds $10. The rest is in a Cold Wallet.
*   **Multi-Sig:** Large transactions require Human approval.

#### Challenge 3: Invoice Management

**Scenario:** Agent makes 10,000 payments. Accounting team goes insane.
**Root Cause:** Granularity.
**Solution:**
*   **Aggregation:** Group payments by vendor and settle once a day.
*   **Crypto Accounting Software:** Use tools like Bitwave/Cryptio to ingest on-chain data and export to QuickBooks.

### System Design Scenario: API Monetization Gateway

**Requirement:** Sell access to your Fine-Tuned Model via L402.
**Design:**
1.  **Gateway:** Reverse Proxy (Nginx + L402 Plugin).
2.  **Lightning Node:** LND (Lightning Network Daemon).
3.  **Flow:**
    *   Request comes in.
    *   Gateway checks for Preimage.
    *   If missing, generate Invoice (1 satoshi per token). Return 402.
    *   If valid, proxy to Model Server.
4.  **No Database:** The "Account" is the Preimage. No user registration needed.

### Summary Checklist for Production
*   [ ] **Stablecoins:** Avoid volatility risk.
*   [ ] **Limits:** Hard cap on daily spending.
*   [ ] **L2s:** Use Polygon/Arbitrum/Solana for low fees.
*   [ ] **Monitoring:** Alert on low balance.
