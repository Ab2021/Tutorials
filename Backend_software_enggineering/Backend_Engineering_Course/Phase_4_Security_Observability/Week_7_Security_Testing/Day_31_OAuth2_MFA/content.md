# Day 31: Advanced Auth with OAuth2 & MFA

## 1. Beyond Basic Auth

Username/Password is not enough.
*   **Delegated Access**: "Let this app access my Google Photos" (without giving it my Google password).
*   **Security**: "Something you know" + "Something you have" (MFA).

---

## 2. OAuth 2.0 (The Standard)

It's an **Authorization** framework, not Authentication (though OIDC adds AuthN).

### 2.1 The Roles
1.  **Resource Owner**: You (the user).
2.  **Client**: The App (e.g., Spotify).
3.  **Authorization Server**: The Guard (e.g., Google Accounts).
4.  **Resource Server**: The Vault (e.g., Google Photos API).

### 2.2 The Dance (Authorization Code Flow)
1.  **User** clicks "Login with Google".
2.  **Client** redirects User to Google (`/authorize`).
3.  **User** logs in and approves permissions ("Scopes").
4.  **Google** redirects User back to Client with a `code`.
5.  **Client** swaps `code` for an `access_token` (Back-channel).
6.  **Client** uses `access_token` to call API.

### 2.3 Grant Types
*   **Authorization Code**: For server-side apps (Most secure).
*   **Client Credentials**: Machine-to-Machine (Service A calls Service B). No user involved.
*   **Implicit**: Deprecated (Don't use).
*   **PKCE**: For Mobile/SPA apps (Adds security to Code flow).

---

## 3. OpenID Connect (OIDC)

OAuth 2.0 is for *Access* (Keys). OIDC is for *Identity* (Badge).
*   **ID Token**: A JWT that says "This is Alice".
*   **UserInfo Endpoint**: Returns `{ "name": "Alice", "email": "..." }`.

---

## 4. Multi-Factor Authentication (MFA)

### 4.1 TOTP (Time-based One-Time Password)
*   **Algorithm**: `HMAC(Secret, Time)`.
*   **Setup**: Server generates a Secret (QR Code). User scans it into Google Authenticator.
*   **Login**: User enters the 6-digit code. Server checks if it matches.

### 4.2 WebAuthn (Passkeys)
*   **Future**: Biometrics (FaceID/TouchID) replace passwords entirely.
*   **Mechanism**: Public Key Cryptography. Private key stays on the device.

---

## 5. Summary

Today we secured the front door.
*   **OAuth2**: Don't handle passwords. Delegate.
*   **OIDC**: Know who the user is.
*   **MFA**: Add a second layer of defense.

**Tomorrow (Day 32)**: We will think like a hacker. We will explore the **OWASP Top 10** and how to inject SQL into our own apps.
