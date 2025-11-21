# Day 28: Security Design

## 1. CIA Triad
*   **Confidentiality:** Only authorized users can see data. (Encryption).
*   **Integrity:** Data is not tampered with. (HMAC, Signatures).
*   **Availability:** System is up. (DDoS protection).

## 2. Authentication (AuthN) vs Authorization (AuthZ)
*   **AuthN:** "Who are you?" (Login).
    *   Passwords (Salt + Hash with Argon2/Bcrypt).
    *   MFA (TOTP).
    *   SSO (OIDC/SAML).
*   **AuthZ:** "What can you do?" (Permissions).
    *   **RBAC:** Role Based (Admin, User).
    *   **ABAC:** Attribute Based (Can view document if `dept=HR` and `time<5pm`).

## 3. HTTPS / TLS
*   **Handshake:**
    1.  Client sends `ClientHello` (Supported ciphers).
    2.  Server sends `ServerHello` + Certificate (Public Key).
    3.  Client verifies Cert with CA (Certificate Authority).
    4.  Key Exchange (Diffie-Hellman) to generate Session Key.
    5.  Encrypted communication using Session Key (Symmetric).

## 4. Common Attacks
*   **SQL Injection:** `USER input: ' OR 1=1 --`. Use Prepared Statements.
*   **XSS (Cross Site Scripting):** Inject JS into page. Sanitize inputs/CSP.
*   **CSRF (Cross Site Request Forgery):** Trick user into clicking link. Use CSRF Tokens / SameSite Cookies.
