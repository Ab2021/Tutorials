# Day 70: Capstone: Building a Universal Agent Interface
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you handle "Recursive Delegation Loops"?

**Answer:**
Agent A delegates to Agent B. Agent B delegates back to Agent A. Infinite loop.
**Solution:**
*   **Call Stack Trace:** Pass a `trace_id` and a list of `visited_agents` in the request headers.
*   **Check:** If `my_id` is in `visited_agents`, reject the request with `400 Loop Detected`.

#### Q2: What is the "Context Window" challenge in a UAI?

**Answer:**
The UAI aggregates tools from 10 MCP servers and 5 Remote Agents. The system prompt becomes huge.
**Solution:**
*   **RAG for Tools:** Index the tool descriptions. Only retrieve the top-5 most relevant tools for the current user query.
*   **Dynamic Loading:** Only load the "Writer Agent" tool definition if the user mentions "writing".

#### Q3: How do you debug a distributed agent transaction?

**Answer:**
*   **Distributed Tracing (OpenTelemetry):**
    *   UAI starts a span.
    *   MCP Client adds a child span.
    *   Agent Protocol Client adds a child span and propagates the `traceparent` header.
    *   Remote Agent extracts the header and continues the trace.
*   **Visualization:** View the entire waterfall in Jaeger/Zipkin.

#### Q4: Explain the "Standardization vs Innovation" trade-off.

**Answer:**
*   **Standardization (MCP/AP):** Ensures compatibility. Everyone speaks the same language. Good for ecosystem growth.
*   **Innovation:** Limits you to the "lowest common denominator". If you invent a cool new feature (e.g., "Telepathic Communication"), the standard doesn't support it yet.
*   **Strategy:** Use standards for the 80% core, use "Extensions" or "Custom Metadata" fields for the 20% innovation.

### Production Challenges

#### Challenge 1: Network Latency

**Scenario:** The UAI is fast, but the Remote Agent is in a different region and takes 200ms round-trip.
**Root Cause:** Physics.
**Solution:**
*   **Edge Deployment:** Deploy agents close to each other.
*   **Asynchronous Delegation:** Don't block. Fire the delegation, continue with other subtasks, and check back later (Future/Promise pattern).

#### Challenge 2: Payment Settlement

**Scenario:** Remote Agent does the work but UAI refuses to pay.
**Root Cause:** Lack of atomic swap.
**Solution:**
*   **Escrow:** Put funds in a smart contract. Release only when the UAI signs a "Receipt of Satisfaction".
*   **Micropayments:** Stream money (per second/token) as the work is being done. Stop if work stops.

#### Challenge 3: Version Mismatch

**Scenario:** UAI speaks Agent Protocol v1. Remote Agent speaks v2.
**Root Cause:** Fast-moving standards.
**Solution:**
*   **Content Negotiation:** Use HTTP `Accept` headers. `Accept: application/vnd.agent.v1+json`.
*   **Adapters:** Middleware that translates v1 requests to v2.

### System Design Scenario: The "Meta-Agent"

**Requirement:** An agent that manages your entire digital life (Email, Calendar, Bank, Health).
**Design:**
1.  **Privacy:** Runs locally (Local LLM + MCP).
2.  **Connectors:** MCP Servers for Gmail, GCal, Plaid.
3.  **Delegation:** Delegates complex tasks (e.g., "Analyze my genome") to specialized cloud agents via Agent Protocol, sending *only* the necessary data (anonymized), not the whole context.
4.  **Identity:** Uses a DID stored in the phone's Secure Enclave to sign transactions.

### Summary Checklist for Production
*   [ ] **Tracing:** Implement OpenTelemetry.
*   [ ] **Loop Detection:** Prevent recursion.
*   [ ] **Timeouts:** Fail fast.
*   [ ] **Security:** Verify signatures on every remote call.
