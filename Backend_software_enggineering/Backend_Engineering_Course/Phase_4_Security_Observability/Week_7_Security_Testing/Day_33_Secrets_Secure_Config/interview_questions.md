# Day 33: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is storing secrets in Environment Variables considered "less secure" than using a Secret Manager?
**Answer:**
1.  **Leakage**: Env vars are often dumped in crash logs, debugging tools, or child processes.
2.  **No Audit**: You don't know *who* read the env var.
3.  **No Rotation**: Changing an env var usually requires a redeploy/restart. Secret Managers support dynamic rotation without restart.

### Q2: What are "Dynamic Secrets" in Vault?
**Answer:**
*   **Concept**: Instead of a static password, Vault generates a *temporary* username/password for the DB on the fly.
*   **Flow**: App asks Vault -> Vault creates user `app-123` in Postgres -> Returns creds to App.
*   **TTL**: The user `app-123` is automatically deleted after 1 hour.
*   **Benefit**: Even if leaked, the credential expires quickly.

### Q3: How do you detect if a secret was committed to Git?
**Answer:**
*   **Tools**: `gitleaks`, `trufflehog`, GitHub Secret Scanning.
*   **Process**: Run these tools in CI/CD. If a high-entropy string (looks like a key) is found, block the merge.

---

## Scenario-Based Questions

### Q4: You accidentally committed an AWS Access Key to a public GitHub repo. What do you do?
**Answer:**
1.  **Revoke**: Immediately disable/delete the key in AWS IAM.
2.  **Assess**: Check CloudTrail logs to see if the key was used.
3.  **Rotate**: Generate a new key for the application.
4.  **Scrub**: Rewrite Git history (BFG Repo-Cleaner) to remove the commit (though assume it's already scraped).

### Q5: How does a Kubernetes Pod authenticate to HashiCorp Vault?
**Answer:**
*   **Kubernetes Auth Method**:
    1.  Pod sends its **Service Account Token** (JWT) to Vault.
    2.  Vault verifies the JWT with the K8s API Server.
    3.  If valid and authorized, Vault returns a **Vault Token**.
    4.  Pod uses Vault Token to read secrets.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to encrypt secrets in the Git repo using `git-crypt` or `SOPS`. Is this okay?
**Answer:**
*   **It's a valid middle ground**.
*   **Pros**: GitOps friendly. Versioned with code.
*   **Cons**: Key management is hard (who has the decryption key?). No audit logs of access.
*   **Verdict**: Okay for small teams. For Enterprise/Compliance, use a dedicated Secret Manager (Vault/AWS SM).
