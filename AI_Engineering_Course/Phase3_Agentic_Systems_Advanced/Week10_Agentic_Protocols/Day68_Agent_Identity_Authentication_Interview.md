# Day 68: Agent Identity & Authentication
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the risk of giving an Agent an API Key vs. a Verifiable Credential?

**Answer:**
*   **API Key:** Bearer token. Anyone who has it *is* the user. If the agent leaks it (prompt injection), the attacker has full access. Hard to revoke without breaking everything.
*   **VC:** Bound to the Agent's DID. The attacker needs the Agent's *Private Key* to use it, not just the token. Can be scoped (limit $50) and expired/revoked individually without changing the user's master key.

#### Q2: How do you recover a "Lost Agent"?

**Answer:**
If the Agent loses its private key (container deleted), its Identity is gone.
**Solution:**
*   **Controller DID:** The User's DID is set as the "Controller" of the Agent's DID.
*   **Key Rotation:** The User can sign a message saying "Rotate Agent 123's key to NewKeyXYZ".
*   **Backup:** Encrypt the Agent's private key with the User's key and store it in a DB.

#### Q3: Explain "Attestation" in the context of Agents.

**Answer:**
Attestation proves *what code* the agent is running.
*   **Secure Enclave (AWS Nitro / SGX):** The hardware generates a hash of the running code and signs it.
*   **Verification:** The user checks the hash. "I am authorizing this specific version of `trader.py`. If the code changes (hack), the hash changes, and the authorization fails."

#### Q4: What is a "Sybil Attack" in Agent Networks?

**Answer:**
An attacker spins up 1,000,000 fake agents to flood the network or rig a vote.
**Mitigation:**
*   **Proof of Work/Stake:** Require a cost to create an identity.
*   **Web of Trust:** Only trust agents endorsed by reputable issuers (e.g., "Verified by Coinbase").

### Production Challenges

#### Challenge 1: Key Management in Stateless Containers

**Scenario:** Agents run in ephemeral Docker containers. Where do they store their Private Key?
**Root Cause:** Security vs Persistence trade-off.
**Solution:**
*   **Vault (HashiCorp):** The container authenticates to Vault (via AWS IAM) on startup, fetches its Private Key into memory, and never writes it to disk.

#### Challenge 2: Revocation

**Scenario:** An agent goes rogue. You need to stop it NOW.
**Root Cause:** Distributed systems are eventually consistent.
**Solution:**
*   **Short TTLs:** Tokens expire every 5 minutes. The agent must refresh them.
*   **Revocation Registry:** A global "Blacklist" on a fast K/V store (Redis) that all Service Providers check before accepting a request.

#### Challenge 3: User UX

**Scenario:** "Please sign this JSON blob to authorize Agent X." Users are scared.
**Root Cause:** Crypto complexity.
**Solution:**
*   **Abstraction:** "Click 'Approve' to let ShoppingBot spend $50." The app handles the signing in the background (Passkeys/FaceID).

### System Design Scenario: Secure Agent Marketplace

**Requirement:** A platform where devs upload agents, and users rent them.
**Design:**
1.  **Upload:** Dev uploads code. Platform builds container, generates DID, and signs an Attestation ("This is version 1.0").
2.  **Rent:** User pays. User issues a UCAN delegating access to their data *to the Agent's DID*.
3.  **Run:** Agent runs in a sandbox. It uses the UCAN to access User data.
4.  **Audit:** Every API call is signed by the Agent. If data leaks, the signature proves which agent did it.

### Summary Checklist for Production
*   [ ] **Scoped Access:** Never give master keys. Use UCANs/Macaroons.
*   [ ] **Key Rotation:** Automate it.
*   [ ] **Attestation:** Verify the code hasn't been tampered with.
*   [ ] **Audit Logs:** Log the DID of the caller, not just the User ID.
