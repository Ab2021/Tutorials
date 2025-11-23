# Day 78: Enterprise Agent Patterns
## Core Concepts & Theory

### The Enterprise Gap

**Hobbyist Agent:** "Write a poem."
**Enterprise Agent:** "Access the HR database, find employees with >5 years tenure, calculate bonus, and email them."
**Requirements:** Security, Reliability, Auditability, Control.

### 1. Human-in-the-Loop (HITL)

**Concept:**
- Agents should not execute high-stakes actions (Delete DB, Send Email) without approval.
- **Patterns:**
  - **Approval:** Agent proposes action -> Human approves -> Agent executes.
  - **Edit:** Human edits the proposed email -> Agent sends.
  - **Feedback:** Human gives feedback -> Agent retries.

### 2. Role-Based Access Control (RBAC)

**Problem:** Agent has "Admin" access to tools. User asks "Delete all users". Agent obeys.
**Solution:**
- **Identity Propagation:** The agent should act *on behalf of* the user.
- **Token Exchange:** Pass the user's JWT token to the downstream API.
- **Scope:** If User is "Viewer", Agent cannot call `delete_user`.

### 3. Audit Logging

**Requirement:**
- Every thought, tool call, and result must be logged.
- **Compliance:** SOC2 / HIPAA requires tracing who did what.
- **Structure:** `[Timestamp, UserID, AgentID, Step, Input, Output]`.

### 4. Deterministic Guardrails

**Concept:**
- Don't rely on LLM to be safe. Use code.
- **Input Guard:** Regex to block PII.
- **Tool Guard:** Whitelist allowed arguments.
- **Output Guard:** JSON Schema validation.

### 5. Multi-Tenant Agents

**Architecture:**
- **Shared Model:** One GPT-4 deployment.
- **Isolated Memory:** Vector DB partitioned by `tenant_id`.
- **Isolated Tools:** Tenant A's agent cannot access Tenant B's API keys.

### 6. Service-Oriented Agents (SOA)

**Concept:**
- Instead of one Monolithic Agent, build micro-agents.
- **HR Agent:** Handles HR tasks.
- **IT Agent:** Handles IT tasks.
- **Router:** Routes user to correct agent.
- **Benefit:** Modular, easier to secure.

### 7. RAG for Agents (Knowledge Injection)

**Concept:**
- Agents need access to enterprise docs (SOPs).
- **Tool:** `search_knowledge_base(query)`.
- **Context:** Inject relevant SOPs into system prompt dynamically.

### 8. Error Handling & Recovery

**Patterns:**
- **Retry:** Transient errors (Network).
- **Fallback:** If Tool A fails, try Tool B.
- **Escalation:** If Agent fails 3 times, hand off to Human Support.

### 9. Summary

**Enterprise Strategy:**
1.  **Security:** Implement **RBAC** and **Identity Propagation**.
2.  **Control:** Use **HITL** for sensitive actions.
3.  **Compliance:** Log everything to **Audit Logs**.
4.  **Architecture:** Use **Micro-Agents** instead of monoliths.
5.  **Reliability:** Implement **Escalation** paths.

### Next Steps
In the Deep Dive, we will implement an RBAC-aware Tool wrapper, an Audit Logger, and a Human-in-the-Loop workflow.
