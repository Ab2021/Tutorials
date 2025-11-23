# Day 73: Decentralized Agent Networks (uAgents)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does a Decentralized Agent differ from a Microservice?

**Answer:**
*   **Autonomy:** A microservice does what it's told (passive). An agent decides *whether* to do it (active).
*   **State:** Agents usually maintain their own state and wallet.
*   **Discovery:** Agents find each other dynamically via Registry; Microservices usually have hardcoded or DNS-based discovery.
*   **Protocol:** Agents speak high-level negotiation protocols; Microservices speak REST/gRPC.

#### Q2: What is the role of the "Wallet" in an AEA?

**Answer:**
*   **Identity:** The Wallet Address is the Agent's ID (DID).
*   **Signing:** Every message is signed by the wallet's private key to prove authenticity.
*   **Transacting:** The agent can hold funds (FET, ETH) to pay for API usage or receive payment for services.

#### Q3: How do you handle "Offline Agents"?

**Answer:**
If Alice sends a message to Bob, but Bob is offline (laptop closed).
*   **Store-and-Forward:** The Agentverse (or a Mail server) buffers the message.
*   **Push Notifications:** Wake up the agent via a webhook.
*   **TTL:** If Bob doesn't wake up in 24h, drop the message.

#### Q4: What is "Oracle" in the context of Agents?

**Answer:**
An Oracle connects the blockchain to the real world.
An Agent can act as an Oracle.
*   *Task:* "What is the temperature in NY?"
*   *Agent:* Calls Weather API -> Signs result -> Pushes to Smart Contract.

### Production Challenges

#### Challenge 1: Key Management

**Scenario:** You deploy 1000 agents. Managing 1000 seed phrases is a nightmare.
**Root Cause:** Decentralization complexity.
**Solution:**
*   **HD Wallets (Hierarchical Deterministic):** Generate 1000 keys from a single Master Seed.
*   **Vault:** Store seeds in HashiCorp Vault and inject them as env vars.

#### Challenge 2: Sybil Attacks

**Scenario:** A spammer registers 1M agents to clog the Almanac.
**Root Cause:** Low barrier to entry.
**Solution:**
*   **Registration Fee:** The Almanac contract charges a small fee (gas) to register. This makes spamming expensive.

#### Challenge 3: Interoperability

**Scenario:** uAgents want to talk to AutoGen agents.
**Root Cause:** Different protocols.
**Solution:**
*   **Gateway Agent:** Build a special uAgent that acts as a bridge. It receives uAgent messages and forwards them to the AutoGen group chat (and vice versa).

### System Design Scenario: Decentralized Compute Market

**Requirement:** Users want to rent GPU time from anyone.
**Design:**
1.  **Provider Agent:** Runs on a machine with a GPU. Monitors idle time. Registers "I have H100".
2.  **User Agent:** "I need H100 for 1 hour."
3.  **Match:** User finds Provider via Almanac.
4.  **Negotiation:** Agree on price ($2/hr).
5.  **Payment:** User locks $2 in a Smart Contract.
6.  **Work:** Provider runs the Docker container.
7.  **Verification:** Provider submits result + Proof of Computation.
8.  **Release:** Contract releases funds.

### Summary Checklist for Production
*   [ ] **Seed Backup:** Don't lose the keys.
*   [ ] **Monitoring:** Track agent uptime.
*   [ ] **Gas Management:** Ensure agents have enough crypto for transaction fees.
*   [ ] **Security:** Sandboxing for executing external tasks.
