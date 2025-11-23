# Day 69: Agent Discovery & Registry
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between "Service Discovery" (Kubernetes) and "Agent Discovery"?

**Answer:**
*   **Service Discovery:** "Where is the `payment-service`?" (Exact name match). Used for internal microservices.
*   **Agent Discovery:** "Who can help me `calculate taxes`?" (Semantic match). Used for open, dynamic ecosystems where you don't know the service name in advance.

#### Q2: How do you prevent "Agent Spam" in a public registry?

**Answer:**
*   **Staking:** Require a deposit (crypto) to register. If the agent spams or lies about capabilities, the stake is slashed.
*   **Web-of-Trust:** Only index agents whose DIDs are signed by a trusted entity (e.g., Verisign for Agents).

#### Q3: Explain "Capability Negotiation".

**Answer:**
After discovery, the agents must agree on a protocol.
*   Agent A: "I speak Agent Protocol v1."
*   Agent B: "I speak Agent Protocol v2."
*   Negotiation: They agree to downgrade to v1 or use a translator.
*   This happens during the initial handshake request.

#### Q4: What is a "Broker Agent"?

**Answer:**
A Broker is a middleman.
*   User -> Broker: "Plan my wedding."
*   Broker -> Registry: "Find florist", "Find caterer".
*   Broker -> Sub-Agents: Coordinates the work.
The Broker adds value by knowing *how* to combine the discovered agents.

### Production Challenges

#### Challenge 1: Stale Entries

**Scenario:** `weather_bot` goes offline, but the Registry still returns it. The calling agent fails.
**Root Cause:** No health checks.
**Solution:**
*   **Active Probing:** The Registry periodically pings registered endpoints.
*   **TTL:** Registrations expire after 1 hour unless renewed (Heartbeat).

#### Challenge 2: Malicious Capabilities

**Scenario:** An agent registers with "I can calculate taxes" but actually steals data.
**Root Cause:** Unverified metadata.
**Solution:**
*   **Sandboxed Test:** The Registry runs a "Test Suite" against the agent before listing it. "Calculate 2+2". If it fails or acts suspiciously, reject registration.

#### Challenge 3: Privacy Leaks in Discovery

**Scenario:** You search for "Help me hide assets". The Registry logs this query.
**Root Cause:** Centralized Registry.
**Solution:**
*   **Private Information Retrieval (PIR):** Cryptographic techniques to query a DB without the DB knowing *what* you queried.
*   **Local Registry:** Download the index (compressed) and search locally.

### System Design Scenario: Decentralized Uber

**Requirement:** Riders find Drivers without a central server.
**Design:**
1.  **Drivers:** Run an Agent on their phone. Register "I am at location X, price Y" in a geospatial registry (DHT - Distributed Hash Table).
2.  **Riders:** Run an Agent. Query the DHT for "Drivers near X".
3.  **Matching:** Rider Agent negotiates price directly with Driver Agent.
4.  **Payment:** Lightning Network.
5.  **No Middleman:** No 30% fee.

### Summary Checklist for Production
*   [ ] **Health Checks:** Remove dead agents.
*   [ ] **Semantic Search:** Use embeddings, not keywords.
*   [ ] **Verification:** Verify DIDs.
*   [ ] **Caching:** Cache discovery results to reduce latency.
