# Day 68: Agent Identity & Authentication
## Core Concepts & Theory

### The Identity Crisis

If an Agent books a flight for you, how does the Airline know it's *your* agent?
If an Agent posts on Twitter, how do we know it's a bot and who owns it?
**Identity** is the missing layer in the Agentic Web.

### 1. Traditional Auth (Web 2.0)

*   **API Keys:** You give the agent your Stripe Secret Key.
    *   *Risk:* If the agent is hacked or goes rogue, it drains your account.
*   **OAuth:** You authorize the agent via "Login with Google".
    *   *Limit:* The agent acts *as* you. It has full access to your scope.

### 2. Decentralized Identity (DID)

**DIDs (Decentralized Identifiers)** are a W3C standard.
`did:web:example.com:agent:123`
*   **Self-Sovereign:** The agent creates its own public/private key pair.
*   **Verifiable:** The agent signs every message with its private key. The receiver verifies it with the public key (stored on a blockchain or web endpoint).

### 3. Verifiable Credentials (VCs)

A VC is a digital "Badge" issued to an agent.
*   *Issuer:* Bank of America.
*   *Credential:* "This agent is authorized to spend up to $50/day on behalf of Alice."
*   *Holder:* The Agent.
*   *Verifier:* The E-commerce Site.

When the Agent tries to buy something, it presents the VC. The site verifies the Bank's signature. The site *doesn't* need to know who Alice is, just that the Bank trusts this agent.

### 4. OIDC for Agents

**OpenID Connect** is evolving to support "Machine" actors.
*   **Workload Identity:** Similar to how Kubernetes pods authenticate to AWS. The Agent exchanges a signed JWT (JSON Web Token) for a short-lived access token.

### Why this matters

1.  **Liability:** If an agent commits a crime, the cryptographic signature proves who owns it.
2.  **Delegation:** You can give an agent *scoped* authority (spend $50) without giving it your credit card number.
3.  **Reputation:** Agents can build a reputation score. "This agent has successfully completed 1000 coding tasks."

### Summary

Agent Identity moves us from "Scripts with API Keys" to "Digital Legal Entities". It enables **Trust** in a zero-trust environment.
