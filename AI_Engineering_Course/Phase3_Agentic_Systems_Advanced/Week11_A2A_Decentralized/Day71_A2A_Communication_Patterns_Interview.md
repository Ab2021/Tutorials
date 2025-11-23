# Day 71: Agent-to-Agent (A2A) Communication Patterns
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why use FIPA-ACL instead of just REST APIs?

**Answer:**
*   **REST:** Resource-oriented (`POST /orders`). Good for dumb clients controlling smart servers.
*   **ACL:** Intent-oriented (`REQUEST buy_stock`). Good for peer-to-peer negotiation.
*   **State:** ACL messages imply a conversation state (Initiator, Participant). REST is stateless.
*   **Flexibility:** ACL allows for `REFUSE` (I don't want to), whereas REST usually implies obedience (or 500 Error).

#### Q2: What is the "Ontology Problem" in Multi-Agent Systems?

**Answer:**
Different agents use different vocabularies.
*   Agent A: `{"temp": 20}` (Celsius).
*   Agent B: `{"temp": 20}` (Fahrenheit).
*   Result: Catastrophe.
*   **Solution:** Explicit Ontologies (Shared Schema) or LLM-based translation/validation layer.

#### Q3: How do you handle "Dead Letters" (Undeliverable Messages)?

**Answer:**
*   **TTL:** Message expires after 1 hour.
*   **DLQ (Dead Letter Queue):** Failed messages go to a special queue for manual inspection.
*   **Ack:** Sender waits for an `ACK`. If no `ACK`, retry with exponential backoff.

#### Q4: Explain "Broadcasting" vs "Multicasting" in Agents.

**Answer:**
*   **Broadcast:** Send to *everyone* on the network. (Expensive, noisy).
*   **Multicast:** Send to a specific *group* (e.g., "All Finance Agents"). (Efficient).
*   **Mechanism:** Pub/Sub topics (`agents/finance/#`).

### Production Challenges

#### Challenge 1: Message Storms

**Scenario:** Agent A sends a message to B. B replies. A replies. They get into an infinite loop of "I didn't understand" -> "Please clarify".
**Root Cause:** Automated error responses.
**Solution:**
*   **Loop Detection:** Check `In-Reply-To` chain depth. Stop after 5.
*   **Circuit Breaker:** If Agent B sends 10 errors in 1 second, stop talking to it.

#### Challenge 2: Latency in P2P Networks

**Scenario:** Agents are on different continents. Negotiation takes too long.
**Root Cause:** Network RTT.
**Solution:**
*   **Co-location:** Move agents to the same region/cluster.
*   **Batching:** Send `REQUEST_BATCH` (Buy 10 stocks) instead of 10 individual requests.

#### Challenge 3: Debugging Distributed State

**Scenario:** The system is stuck. Agent A thinks it's waiting for B. Agent B thinks it's waiting for A.
**Root Cause:** State desynchronization.
**Solution:**
*   **Global Sequence Numbers:** Order messages globally.
*   **Vector Clocks:** Track causality.
*   **Centralized Logger:** Even if agents are P2P, they should async log to a central observer.

### System Design Scenario: Smart Grid Energy Trading

**Requirement:** Solar panels (Agents) sell excess energy to Neighbors (Agents).
**Design:**
1.  **Protocol:** Contract Net.
2.  **Trigger:** Solar Agent detects excess battery. Broadcasts `CFP` (Call for Proposal) "Who wants 1kWh?".
3.  **Bidding:** Neighbor Agents check their battery. If low, send `PROPOSE $0.10`.
4.  **Award:** Solar Agent picks highest bid. Sends `ACCEPT`.
5.  **Settlement:** Energy transfer happens. Payment via Crypto.

### Summary Checklist for Production
*   [ ] **Schema:** Define strict JSON schemas for content.
*   [ ] **Timeouts:** Every request needs a timeout.
*   [ ] **Idempotency:** Handling the same message twice shouldn't break things.
*   [ ] **Logging:** Log the `performative` and `sender` for every message.
