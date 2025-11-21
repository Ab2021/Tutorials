# Day 28 Interview Prep: Security

## Q1: How to store passwords securely?
**Answer:**
*   **Never** store plain text.
*   **Never** use fast hashes (MD5, SHA256).
*   **Use:** Slow, salted hashes (Bcrypt, Argon2, PBKDF2).
*   **Salt:** Random string added to password to prevent Rainbow Table attacks.

## Q2: Session vs JWT?
**Answer:**
*   **Session:**
    *   **Stateful:** Server stores session ID in Redis/DB.
    *   **Pros:** Easy to revoke (delete from Redis).
    *   **Cons:** Server lookup overhead.
*   **JWT:**
    *   **Stateless:** Server verifies signature.
    *   **Pros:** Fast. Scalable.
    *   **Cons:** Hard to revoke. Token size is larger.

## Q3: Explain a MITM (Man In The Middle) attack.
**Answer:**
*   Attacker sits between Client and Server.
*   Intercepts traffic.
*   **Prevention:** HTTPS (TLS). The Certificate ensures you are talking to the real server, not the attacker.

## Q4: What is CORS?
**Answer:**
*   **Cross-Origin Resource Sharing.**
*   Browser security feature. Prevents `evil.com` from making AJAX requests to `bank.com`.
*   Server must send header `Access-Control-Allow-Origin: evil.com` (which it won't) to allow it.
