# Day 32: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between XSS and CSRF?
**Answer:**
*   **XSS (Cross-Site Scripting)**: The attacker runs *their* code (JS) in *your* browser. (Steals cookies, modifies page).
*   **CSRF (Cross-Site Request Forgery)**: The attacker tricks *your* browser into sending a request to *another* site where you are logged in. (Transfers money, changes password).

### Q2: How does `SameSite=Strict` prevent CSRF?
**Answer:**
*   **Mechanism**: It tells the browser "Only send this cookie if the request originates from the *same site* that set the cookie".
*   **Effect**: If `evil.com` tries to POST to `bank.com`, the browser sees the domain mismatch and *withholds* the session cookie. `bank.com` rejects the request as unauthenticated.

### Q3: What is SSRF (Server-Side Request Forgery)?
**Answer:**
*   **Scenario**: App takes a URL from user and fetches it. `GET /fetch?url=http://google.com`.
*   **Attack**: User sends `GET /fetch?url=http://localhost:8080/admin`. The *Server* fetches its own internal admin page and returns it to the user.
*   **Fix**: Whitelist allowed domains. Block internal IP ranges (127.0.0.1, 192.168.x.x).

---

## Scenario-Based Questions

### Q4: You found a SQL Injection vulnerability in a legacy app. How do you fix it?
**Answer:**
*   **Identify**: Look for string concatenation: `query = "SELECT * FROM users WHERE id = " + id`.
*   **Remediate**: Switch to Parameterized Queries.
    *   *Bad*: `cursor.execute(f"SELECT ... {id}")`
    *   *Good*: `cursor.execute("SELECT ... %s", (id,))`
*   **Validate**: Use `sqlmap` to verify the fix.

### Q5: A penetration tester reports that your API is missing "Security Headers". Which ones should you add?
**Answer:**
1.  `Strict-Transport-Security`: max-age=31536000; includeSubDomains.
2.  `X-Content-Type-Options`: nosniff.
3.  `X-Frame-Options`: DENY.
4.  `Content-Security-Policy`: default-src 'self'. (Requires testing to ensure it doesn't break app).

---

## Behavioral / Role-Specific Questions

### Q6: A developer says "We don't need HTTPS for internal microservices". Do you agree?
**Answer:**
*   **No (Zero Trust)**.
*   **Risk**: If an attacker gets into the network (e.g., via a compromised laptop), they can sniff all internal traffic (passwords, data) if it's HTTP.
*   **Standard**: Use **mTLS** (Mutual TLS) or at least HTTPS for all internal traffic.
