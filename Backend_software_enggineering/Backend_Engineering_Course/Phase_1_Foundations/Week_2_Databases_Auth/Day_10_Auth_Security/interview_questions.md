# Day 10: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between Session-based and Token-based Authentication?
**Answer:**
*   **Session-based**:
    *   Server stores session data in memory/DB (Stateful).
    *   Client gets a `session_id` cookie.
    *   *Pros*: Easy revocation (delete session).
    *   *Cons*: Harder to scale (sticky sessions or shared Redis).
*   **Token-based (JWT)**:
    *   Server is stateless. Token contains the data.
    *   Client stores token (LocalStorage/Cookie).
    *   *Pros*: Scalable, works for Mobile/API.
    *   *Cons*: Hard to revoke.

### Q2: Why is `HS256` (Symmetric) different from `RS256` (Asymmetric) for JWTs?
**Answer:**
*   **HS256**: Uses a single Secret Key to Sign and Verify. Both Auth Service and API Service need the same secret. (Risk: If API Service is compromised, they can forge tokens).
*   **RS256**: Uses a Private Key to Sign (Auth Service) and Public Key to Verify (API Service). API Service cannot forge tokens. Safer for microservices.

### Q3: Why shouldn't you store JWTs in LocalStorage?
**Answer:**
*   **XSS Risk**: If an attacker injects JS into your site (XSS), they can read `localStorage` and steal the token.
*   **Solution**: Store in an **HttpOnly Cookie**. JS cannot read it, but the browser sends it automatically. (Introduces CSRF risk, which needs mitigation).

---

## Scenario-Based Questions

### Q4: You discover that your JWT Secret Key has been leaked. What do you do?
**Answer:**
1.  **Rotate Key**: Generate a new Secret Key immediately.
2.  **Invalidate**: All existing tokens will fail verification (users logged out).
3.  **Deploy**: Update all services with the new key.
4.  **Investigate**: Find how it leaked (committed to git? env var exposed?).

### Q5: A user forgot their password. Design the "Forgot Password" flow securely.
**Answer:**
1.  User enters email.
2.  Generate a high-entropy random token (e.g., UUID).
3.  Hash the token and store in DB with `user_id` and `expiration` (e.g., 15 mins).
4.  Email the *raw* token link to user: `app.com/reset?token=xyz`.
5.  User clicks link.
6.  Server hashes the incoming token and compares with DB.
7.  If match + not expired: Allow password reset. Delete token.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use MD5 for hashing passwords because "it's fast". How do you explain why this is bad?
**Answer:**
*   **Speed is the Enemy**: Hashing *should* be slow. A modern GPU can calculate billions of MD5 hashes per second.
*   **Rainbow Tables**: Pre-computed tables exist for MD5.
*   **Demonstration**: I would show them an online MD5 cracker that reverses their hash in milliseconds.
*   **Recommendation**: Use `bcrypt` with a work factor (cost) of 10 or 12.
