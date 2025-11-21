# Day 28 Deep Dive: OAuth 2.0 & OIDC

## 1. The Problem
*   "Log in with Google".
*   You don't want to give your Google Password to `RandomApp.com`.

## 2. OAuth 2.0 (Authorization)
*   **Roles:**
    *   **Resource Owner:** You.
    *   **Client:** `RandomApp.com`.
    *   **Authorization Server:** Google.
    *   **Resource Server:** Gmail API.
*   **Flow (Authorization Code):**
    1.  App redirects you to Google.
    2.  You login and consent.
    3.  Google redirects back to App with `code`.
    4.  App exchanges `code` for `access_token` (Back channel).
    5.  App uses `access_token` to fetch data.

## 3. OIDC (OpenID Connect) - Authentication
*   OAuth is for *Access*, not *Identity*. (The key card gives access, doesn't say who holds it).
*   **OIDC:** Layer on top of OAuth.
*   **ID Token:** JWT (JSON Web Token) containing user info (`sub`, `email`, `name`).
*   **Flow:** Same as OAuth, but returns `access_token` AND `id_token`.

## 4. JWT Anatomy
*   `Header.Payload.Signature`
*   **Header:** Algo (`HS256`).
*   **Payload:** Claims (`user_id: 123`, `exp: 17000000`).
*   **Signature:** `HMAC(Header + Payload, Secret)`.
*   **Pros:** Stateless. Server doesn't need to check DB.
*   **Cons:** Hard to revoke (Need blocklist or short TTL).
