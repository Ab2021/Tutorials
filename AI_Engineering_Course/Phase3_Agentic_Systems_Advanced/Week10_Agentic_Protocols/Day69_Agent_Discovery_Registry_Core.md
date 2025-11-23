# Day 69: Agent Discovery & Registry
## Core Concepts & Theory

### The Discovery Problem

In a decentralized world, how does my "Travel Agent" find your "Hotel Agent"?
*   **Web 2.0:** Google Search (Centralized).
*   **Web 3.0 / Agentic Web:** We need a DNS for Agents.

### 1. The Agent Registry

A Registry is a phonebook.
*   **Entry:** `AgentID` -> `Capabilities` + `Endpoint`.
*   **Example:** `did:agent:123` -> `["book_hotel", "check_availability"]` @ `https://hotel-agent.com/api`.

### 2. Semantic Discovery

Agents don't search by keywords ("Hotel"); they search by **Intent**.
*   *Query:* "I need to book a room in Paris."
*   *Mechanism:* The Registry uses Vector Search. It embeds the capabilities of all registered agents. It finds the agent whose capability description ("I can book rooms in France") is semantically closest to the query.

### 3. The Almanac (Fetch.ai)

Fetch.ai's Almanac is a smart contract-based registry.
*   Agents register themselves on-chain (or on a sidechain).
*   They pay a small fee (preventing spam).
*   They update their endpoint if their IP changes.

### 4. Model Context Discovery (MCP)

MCP has a local discovery mechanism.
*   **mcp-config.json:** Lists available servers.
*   **Dynamic Discovery:** A "Mother Server" can return a list of other servers based on the user's prompt. "User asked about SQL -> Load the SQL Server."

### 5. Trust & Reputation

Discovery is useless without Trust.
*   **Verified Agents:** The Registry checks if the Agent owns the domain `hotel-agent.com`.
*   **Feedback Loops:** After an interaction, the calling agent rates the service. "This agent was slow." The Registry updates the reputation score.

### Summary

Agent Discovery transforms the ecosystem from a set of isolated silos into a **Dynamic Marketplace**. It allows agents to form ad-hoc teams to solve problems the user never explicitly anticipated.
