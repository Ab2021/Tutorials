# Day 78: Enterprise Agent Patterns
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is "Identity Propagation" in the context of Agents?

**Answer:**
- The practice of passing the end-user's identity (Token/Claims) through the agent to the downstream services.
- **Why:** Ensures the agent operates with the *least privilege* (the user's permissions), rather than "Super Admin" permissions.
- **Prevents:** Privilege Escalation attacks.

#### Q2: How do you implement Human-in-the-Loop (HITL) for agents?

**Answer:**
- **Interrupt Pattern:** The agent workflow pauses before sensitive steps.
- **State Persistence:** The state is saved to a DB.
- **UI:** A human reviews the pending action in a UI and clicks "Approve" or "Reject".
- **Resume:** The workflow resumes from the saved state.

#### Q3: What is the difference between "Stateless" and "Stateful" agents in enterprise?

**Answer:**
- **Stateless:** REST API style. Request -> Response. No memory of past requests. Easier to scale.
- **Stateful:** Maintains conversation history and context over days/weeks. Requires a database (Redis/Postgres) to persist state. Essential for complex workflows.

#### Q4: Why is "Audit Logging" critical for GenAI?

**Answer:**
- **Debugging:** "Why did the agent refund $1000?" -> Check the log to see the reasoning.
- **Compliance:** GDPR/SOC2 requires knowing who accessed what data.
- **Liability:** If the agent hallucinates legal advice, you need a record of exactly what was generated.

#### Q5: Explain the "Micro-Agent" architecture.

**Answer:**
- Decomposing a complex task into smaller, specialized agents.
- **Example:** "Trip Planner" -> "Flight Agent", "Hotel Agent", "Car Rental Agent".
- **Benefit:** Easier to test, secure, and maintain each micro-agent independently.

---

### Production Challenges

#### Challenge 1: The "Super Admin" Agent

**Scenario:** To make things work, devs give the agent an API Key with `admin` access. Attacker prompts "Delete all users". Agent succeeds.
**Root Cause:** Over-privileged agent.
**Solution:**
- **RBAC:** Give the agent a Service Account with minimal scopes.
- **Identity Propagation:** Use the user's token.

#### Challenge 2: Audit Log Noise

**Scenario:** Logging every token generates TBs of logs. Expensive and hard to search.
**Root Cause:** Logging raw LLM streams.
**Solution:**
- **Structured Logging:** Log only the *Prompt* and *Final Response* and *Tool Calls*. Don't log the intermediate thinking tokens unless debugging.
- **Sampling:** Log 100% of sensitive actions, 1% of chat.

#### Challenge 3: Approval Fatigue

**Scenario:** HITL is implemented for *every* email. Users stop using the agent because it's too slow.
**Root Cause:** Too much friction.
**Solution:**
- **Confidence Thresholds:** Auto-approve if Confidence > 99%.
- **Risk-Based:** Only require approval for external emails, not internal ones.

#### Challenge 4: Token Expiry during Long Tasks

**Scenario:** Agent starts a task. User's token expires in 1 hour. Task takes 2 hours. Agent fails.
**Root Cause:** Short-lived tokens.
**Solution:**
- **Refresh Tokens:** Agent needs logic to refresh the user's token.
- **Offline Access:** Request "Offline Access" scope for long-running jobs.

#### Challenge 5: Multi-Tenant Data Leak

**Scenario:** Tenant A asks "Summarize recent tickets". Agent retrieves Tenant B's tickets.
**Root Cause:** Missing metadata filter in Vector Search.
**Solution:**
- **Enforced Filters:** Wrap the Vector DB client to *always* inject `tenant_id` filter based on the authenticated user. Never trust the client to pass it.

### System Design Scenario: Enterprise HR Bot

**Requirement:** Bot can answer policy questions (Public) and update salary (Private/Admin).
**Design:**
1.  **Router:** Classifies intent.
2.  **Policy Agent:** Access to Vector DB (Public Docs). No Auth required.
3.  **Admin Agent:** Access to HRIS API. Requires `hr_admin` role.
4.  **Auth:** User logs in via SSO. Token passed to Admin Agent.
5.  **Gate:** Admin Agent checks `if 'hr_admin' in token.roles`.
6.  **Audit:** Log "User X updated salary of Y".

### Summary Checklist for Production
- [ ] **Auth:** Use **Identity Propagation**.
- [ ] **Logs:** Implement **Structured Audit Logs**.
- [ ] **Approval:** Use **HITL** for high-risk actions.
- [ ] **Isolation:** Enforce **Tenant Filters** in Vector DB.
- [ ] **Scope:** Give Agent **Least Privilege**.
