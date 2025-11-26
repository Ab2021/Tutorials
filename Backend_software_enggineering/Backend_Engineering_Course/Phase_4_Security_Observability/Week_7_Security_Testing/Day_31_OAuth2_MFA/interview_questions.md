# Day 31: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between an Access Token and an ID Token?
**Answer:**
*   **Access Token (OAuth2)**: The "Key". Used to call the API. The Client shouldn't read it (it's opaque or JWT for the API).
*   **ID Token (OIDC)**: The "Badge". Contains user info (name, email). Intended for the Client to read and display.

### Q2: Why is the "Implicit Flow" deprecated?
**Answer:**
*   **Mechanism**: It returns the Access Token directly in the URL fragment (`#access_token=...`).
*   **Risk**:
    *   Browser History logs the token.
    *   Malicious JS can steal it easily.
*   **Replacement**: Authorization Code Flow with **PKCE** (Proof Key for Code Exchange).

### Q3: Where should you store tokens in a Browser (SPA)?
**Answer:**
*   **LocalStorage**: Vulnerable to XSS. If an attacker runs JS, they steal the token.
*   **HttpOnly Cookie**: Safe from XSS (JS can't read it). Vulnerable to CSRF (but CSRF is easier to mitigate with SameSite=Strict).
*   **Verdict**: **HttpOnly Cookie** is the recommended best practice for high security.

---

## Scenario-Based Questions

### Q4: A user complains they have to log in every 15 minutes. How do you fix this without compromising security?
**Answer:**
*   **Short Access Token**: Keep Access Token short-lived (15 mins) for security.
*   **Refresh Token**: Issue a long-lived Refresh Token (e.g., 7 days).
*   **Flow**: When Access Token expires (401), the Client uses the Refresh Token to get a new Access Token silently. User is not interrupted.

### Q5: You need to implement "Log out from all devices". How?
**Answer:**
*   **Stateless JWT**: You can't revoke a JWT easily.
*   **Solution**:
    1.  **Blacklist**: Store revoked JTI (JWT ID) in Redis with TTL. Middleware checks Redis on every request.
    2.  **Token Version**: Add `token_version: 1` to User DB and JWT. On logout, increment User's version to 2. Middleware checks `jwt.version == user.version`.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to build their own Auth system because "OAuth is too complex". What do you say?
**Answer:**
*   **"Don't roll your own crypto/auth."**
*   **Risks**: Hashing passwords wrong, missing edge cases (reset password flow, email verification), maintenance burden.
*   **Advice**: Use a library (Passport.js, Authlib) or a provider (Auth0, Firebase Auth, AWS Cognito). It saves months of work and is more secure.
