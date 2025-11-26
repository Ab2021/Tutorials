# Day 10: Auth & Security Basics

## 1. AuthN vs AuthZ

*   **Authentication (AuthN)**: "Who are you?"
    *   *Methods*: Password, OTP, Biometric, API Key.
*   **Authorization (AuthZ)**: "What are you allowed to do?"
    *   *Methods*: RBAC (Role-Based), ABAC (Attribute-Based).

---

## 2. Password Storage (The Cardinal Sin)

**NEVER STORE PASSWORDS IN PLAIN TEXT.**

### 2.1 Hashing
A hash function is one-way. `Hash(password) -> digest`. You cannot reverse it.
*   *Bad*: MD5, SHA1 (Too fast, vulnerable to collisions).
*   *Good*: SHA-256 (Better, but still too fast for GPUs).
*   *Best*: **Bcrypt**, **Argon2**, **Scrypt**. These are "slow hashes" designed to resist brute-force attacks.

### 2.2 Salting
*   **Problem**: If two users have the password "password123", they get the same hash. Rainbow Tables can reverse this.
*   **Solution**: Add a random string (Salt) to the password before hashing. `Hash(password + salt)`.
*   *Note*: Modern libraries (like Python's `bcrypt`) handle salting automatically.

---

## 3. JWT (JSON Web Tokens)

JWT is the standard for stateless authentication.

### 3.1 Structure
`Header.Payload.Signature`
1.  **Header**: Algorithm (`HS256`).
2.  **Payload (Claims)**: Data (`user_id`, `exp`, `role`).
3.  **Signature**: `Hash(Header + Payload + Secret)`.

### 3.2 The Flow
1.  **Login**: Client sends User/Pass.
2.  **Verify**: Server checks DB. If valid, generates a JWT signed with a **Secret Key**.
3.  **Response**: Server sends JWT to Client.
4.  **Request**: Client sends `Authorization: Bearer <token>` in header.
5.  **Validate**: Server decodes token, verifies signature using the Secret Key. **No DB lookup needed!**

### 3.3 Pros & Cons
*   *Pros*: Stateless (Scales well), contains user info.
*   *Cons*: Cannot be revoked easily (need a Blacklist or short TTL).

---

## 4. HTTPS & TLS

*   **Encryption in Transit**: Prevents Man-in-the-Middle attacks.
*   **Certificates**: Issued by a CA (Certificate Authority like Let's Encrypt).
*   **Handshake**:
    1.  Client Hello.
    2.  Server sends Cert (Public Key).
    3.  Client verifies Cert.
    4.  Client generates Session Key, encrypts with Public Key.
    5.  Server decrypts with Private Key.
    6.  Communication continues using Session Key (Symmetric).

---

## 5. Summary

Today we secured our application.
*   **Passwords**: Hash with Salt (Bcrypt).
*   **Tokens**: Use JWTs for stateless auth.
*   **Transport**: Always use HTTPS.

**Week 2 Wrap-Up**:
We have covered:
1.  Relational DBs (SQL, Normalization).
2.  NoSQL (Mongo, Redis).
3.  Vector DBs (Embeddings).
4.  Security (AuthN/AuthZ).

**Next Week (Week 3)**: We move to **Architecture**. We will break the Monolith into **Microservices**, learn about **Event-Driven Architecture**, and explore the **API Gateway** pattern.
