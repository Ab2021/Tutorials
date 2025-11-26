# Day 33: Secrets & Secure Config

## 1. The "Secret" Problem

*   **Scenario**: You commit `DB_PASSWORD=secret` to GitHub.
*   **Result**: Bots scan GitHub every second. Your DB is hacked in 5 minutes.
*   **Rule #1**: **NEVER** commit secrets to Git. Not even in private repos.

---

## 2. Environment Variables (The Basics)

*   **12-Factor App**: Store config in the environment.
*   **Local**: `.env` file (added to `.gitignore`).
*   **Production**: Inject via K8s ConfigMap/Secret or CI/CD pipeline.
*   *Weakness*: Env vars can leak in logs or debugging screens (`phpinfo()`, `printenv`).

---

## 3. Secret Managers (The Enterprise Way)

Tools dedicated to storing secrets.
*   **HashiCorp Vault**: The gold standard. Cloud-agnostic.
*   **AWS Secrets Manager / GCP Secret Manager**: Cloud-native.

### 3.1 How it works
1.  **App Starts**: App has an IAM Role (Identity).
2.  **Auth**: App authenticates to Vault using IAM Role.
3.  **Fetch**: App requests `secret/data/db-pass`.
4.  **Vault**: Checks policy ("Does this Role have read access?"). Returns secret.
5.  **Memory**: App keeps secret in RAM. Never writes to disk.

---

## 4. Key Rotation

Passwords leak. You must change them regularly.
*   **Manual**: Painful. Downtime.
*   **Automated**:
    1.  Secrets Manager generates new password.
    2.  Updates the Database User.
    3.  Updates the Secret in the Manager.
    4.  App fetches new secret (or restarts).

---

## 5. Summary

Today we hid the keys.
*   **Git**: No secrets allowed.
*   **Env Vars**: Good start.
*   **Vault**: Best practice.
*   **Rotation**: Change keys often.

**Tomorrow (Day 34)**: We shift gears to **Quality Assurance**. How to write tests that actually catch bugs (Unit, Integration, E2E).
