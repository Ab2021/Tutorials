# Day 10: Authentication & Security Foundations

## Table of Contents
1. [Authentication vs Authorization](#1-authentication-vs-authorization)
2. [Password Security](#2-password-security)
3. [Session-based Auth](#3-session-based-auth)
4. [Token-based Auth (JWT)](#4-token-based-auth-jwt)
5. [OAuth 2.0 Deep Dive](#5-oauth-20-deep-dive)
6. [Multi-Factor Authentication](#6-multi-factor-authentication)
7. [Security Best Practices](#7-security-best-practices)
8. [OWASP Top 10](#8-owasp-top-10)
9. [API Security](#9-api-security)
10. [Summary](#10-summary)

---

## 1. Authentication vs Authorization

### 1.1 Definitions

**Authentication** (AuthN): "Who are you?"
- Proving identity
- Examples: Password, fingerprint, 2FA

**Authorization** (AuthZ): "What can you do?"
- Permissions & access control
- Examples: Admin role, read-only access

### 1.2 Real-World Analogy

```
Airline Security:
┌─────────────────────────────────────┐
│ Show ID → Authentication            │
│ Board plane → Authorization         │
│   (Business class vs Economy)       │
└─────────────────────────────────────┘
```

### 1.3 Common Mistakes

❌ **Confusing the two**:
```python
# Bad: HTTP 401 used incorrectly
if not user.has_permission('delete_users'):
    return 401  # Should be 403!

# Good:
# 401 Unauthorized = Not authenticated (missing/invalid token)
# 403 Forbidden = Authenticated but not authorized
```

---

## 2. Password Security

### 2.1 Why Plain Text is Dangerous

**Scenario**: Database breach
```sql
users:
id | email              | password
1  | alice@example.com  | password123  ← Exposed!
```

**Impact**: All users compromised.

### 2.2 Hashing (One-Way Function)

```python
import hashlib

password = "password123"
hash = hashlib.sha256(password.encode()).hexdigest()
# Result: ef92b778bafe771e89...
```

**Problem**: Rainbow table attacks.

```
SHA256("password123") = ef92b778...
Attacker pre-computes: "password123" → ef92b778...
```

### 2.3 Salting

```python
import os
import hashlib

def hash_password(password):
    salt = os.urandom(32)  # Random 32 bytes
    hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt + hash  # Store salt with hash

def verify_password(password, stored):
    salt = stored[:32]
    hash = stored[32:]
    new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return new_hash == hash
```

**Benefit**: Same password → different hashes (unique salt).

### 2.4 Modern Approach: bcrypt/Argon2

**bcrypt** (Industry standard):
```python
import bcrypt

# Hash password
password = "password123"
salt = bcrypt.gensalt(rounds=12)  # Cost factor (higher = slower)
hashed = bcrypt.hashpw(password.encode(), salt)

# Verify
is_valid = bcrypt.checkpw(password.encode(), hashed)
```

**Argon2** (Winner of Password Hashing Competition):
```python
from argon2 import PasswordHasher

ph = PasswordHasher()
hash = ph.hash("password123")
# $ argon2id$v=19$m=65536,t=3,p=4$...

ph.verify(hash, "password123")  # Returns True or raises exception
```

**Why slow hashing is good**: Brute-forcing 1 billion passwords takes years instead of seconds.

### 2.5 Password Requirements

✅ **Enforce**:
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, symbols
- Ban common passwords ("password123", "qwerty")

❌ **Avoid**:
- Storing password hints
- Emailing passwords
- Limiting max length (< 64 chars is suspicious)

---

## 3. Session-based Auth

### 3.1 How Sessions Work

```mermaid
sequenceDiagram
    User->>Server: POST /login {email, password}
    Server->>DB: Verify credentials
    Server->>Redis: Create session {user_id: 123}
    Server->>User: Set-Cookie: session_id=abc123
    User->>Server: GET /profile (Cookie: session_id=abc123)
    Server->>Redis: Lookup session abc123
    Redis-->>Server: {user_id: 123}
    Server->>User: Profile data
```

### 3.2 Implementation (Python/Flask)

```python
from flask import Flask, session, request
import redis

app = Flask(__name__)
app.secret_key = "super-secret-key"
redis_client = redis.Redis()

@app.route('/login', methods=['POST'])
def login():
    email = request.json['email']
    password = request.json['password']
    
    # Verify credentials (omitted for brevity)
    user = db.query("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not bcrypt.checkpw(password.encode(), user['password_hash']):
        return {"error": "Invalid credentials"}, 401
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    redis_client.setex(
        f"session:{session_id}",
        3600,  # 1 hour TTL
        json.dumps({"user_id": user['id']})
    )
    
    # Set cookie
    response = {"message": "Logged in"}
    response.set_cookie(
        'session_id',
        session_id,
        httponly=True,  # XSS protection
        secure=True,     # HTTPS only
        samesite='Strict'  # CSRF protection
    )
    return response

@app.route('/profile')
def profile():
    session_id = request.cookies.get('session_id')
    if not session_id:
        return {"error": "Not authenticated"}, 401
    
    session_data = redis_client.get(f"session:{session_id}")
    if not session_data:
        return {"error": "Session expired"}, 401
    
    user_data = json.loads(session_data)
    user = db.query("SELECT * FROM users WHERE id = ?", (user_data['user_id'],)).fetchone()
    return {"user": user}
```

### 3.3 Pros & Cons

**Pros**:
- ✅ Server-side revocation (delete session key)
- ✅ Stateful (server knows who's logged in)

**Cons**:
- ❌ Requires session store (Redis/DB)
- ❌ Horizontal scaling tricky (session affinity or shared Redis)
- ❌ Mobile apps awkward (cookies don't work well)

---

## 4. Token-based Auth (JWT)

### 4.1 JWT Structure

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMiLCJuYW1lIjoiQWxpY2UiLCJleHAiOjE3MTY4NDgwMDB9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
└────────────────┬────────────────┘ └─────────────────┬─────────────────┘ └───────────────┬───────────────┘
         Header                                 Payload                                  Signature
```

**Header**:
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**Payload**:
```json
{
  "sub": "123",  // Subject (user ID)
  "name": "Alice",
  "exp": 1716848000  // Expiration timestamp
}
```

**Signature** = HMAC-SHA256(base64(header) + "." + base64(payload), secret)

### 4.2 Implementation

**Generating JWT**:
```python
import jwt
import datetime

secret = "your-256-bit-secret"

def generate_token(user_id):
    payload = {
        "sub": user_id,
        "iat": datetime.datetime.utcnow(),  // Issued At
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token

token = generate_token(123)
```

**Verifying JWT**:
```python
def verify_token(token):
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

@app.route('/profile')
def profile():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return {"error": "No token provided"}, 401
    
    token = auth_header.replace("Bearer ", "")
    payload = verify_token(token)
    
    if not payload:
        return {"error": "Invalid or expired token"}, 401
    
    user_id = payload['sub']
    user = db.query("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"user": user}
```

### 4.3 Pros & Cons

**Pros**:
- ✅ Stateless (no database lookup)
- ✅ Works across domains (CORS-friendly)
- ✅ Mobile-friendly (pass in Authorization header)

**Cons**:
- ❌ Cannot revoke before expiry (must blacklist)
- ❌ Token size larger than session ID (cookies)
- ❌ Payload visible (base64-encoded, not encrypted)

### 4.4 Refresh Tokens

**Problem**: Short-lived access tokens (15 min) → frequent re-login.

**Solution**: Refresh token (long-lived, 7 days).

```python
def login():
    # ...
    access_token = generate_token(user_id, exp=timedelta(minutes=15))
    refresh_token = generate_token(user_id, exp=timedelta(days=7))
    
    # Store refresh token in DB (for revocation)
    db.execute(
        "INSERT INTO refresh_tokens (user_id, token) VALUES (?, ?)",
        (user_id, refresh_token)
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token
    }

@app.route('/refresh', methods=['POST'])
def refresh():
    refresh_token = request.json['refresh_token']
    
    # Verify refresh token
    payload = verify_token(refresh_token)
    if not payload:
        return {"error": "Invalid refresh token"}, 401
    
    # Check if still valid in DB
    db_token = db.query(
        "SELECT * FROM refresh_tokens WHERE token = ?",
        (refresh_token,)
    ).fetchone()
    
    if not db_token:
        return {"error": "Refresh token revoked"}, 401
    
    # Issue new access token
    new_access_token = generate_token(payload['sub'], exp=timedelta(minutes=15))
    return {"access_token": new_access_token}
```

---

## 5. OAuth 2.0 Deep Dive

### 5.1 OAuth Roles

1. **Resource Owner**: User (owns their data)
2. **Client**: Your app (wants access to user's data)
3. **Authorization Server**: Google/GitHub (issues tokens)
4. **Resource Server**: Google Drive API (serves data)

### 5.2 Authorization Code Flow

```mermaid
sequenceDiagram
    User->>Client: Click "Login with Google"
    Client->>AuthServer: Redirect to /authorize
    User->>AuthServer: Login & consent
    AuthServer->>Client: Redirect with code=abc123
    Client->>AuthServer: POST /token (code + client_secret)
    AuthServer->>Client: access_token + refresh_token
    Client->>ResourceServer: API request (Bearer token)
    ResourceServer->>Client: User data
```

### 5.3 Implementation (GitHub OAuth)

**Step 1**: Redirect to GitHub
```python
@app.route('/login')
def login():
    github_auth_url = (
        "https://github.com/login/oauth/authorize?"
        f"client_id={GITHUB_CLIENT_ID}&"
        f"redirect_uri=http://localhost:5000/callback&"
        f"scope=user:email"
    )
    return redirect(github_auth_url)
```

**Step 2**: Handle callback
```python
@app.route('/callback')
def callback():
    code = request.args.get('code')
    
    # Exchange code for token
    response = requests.post(
        "https://github.com/login/oauth/access_token",
        data={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code
        },
        headers={"Accept": "application/json"}
    )
    
    access_token = response.json()['access_token']
    
    # Fetch user info
    user_response = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    user_data = user_response.json()
    # Create user session or JWT
    return {"user": user_data}
```

### 5.4 PKCE (Proof Key for Code Exchange)

**Problem**: Mobile apps can't securely store `client_secret`.

**Solution**: Use code verifier & code challenge.

```python
import hashlib
import base64
import secrets

# Step 1: Generate code verifier
code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=')

# Step 2: Create code challenge
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier).digest()
).rstrip(b'=')

# Step 3: Auth request with code_challenge
auth_url = (
    f"https://auth.example.com/authorize?"
    f"code_challenge={code_challenge}&"
    f"code_challenge_method=S256"
)

# Step 4: Token request with code_verifier
token_response = requests.post(
    "https://auth.example.com/token",
    data={
        "code": code,
        "code_verifier": code_verifier  # Proves you initiated the request
    }
)
```

---

## 6. Multi-Factor Authentication

### 6.1 TOTP (Time-Based One-Time Password)

**How it works**:
1. Server generates secret
2. User scans QR code (adds to Google Authenticator)
3. App generates 6-digit code every 30 seconds
4. User enters code to login

**Implementation**:
```python
import pyotp
import qrcode

def enable_2fa(user_id):
    # Generate secret
    secret = pyotp.random_base32()
    
    # Store secret in DB
    db.execute("UPDATE users SET totp_secret = ? WHERE id = ?", (secret, user_id))
    
    # Generate QR code
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user_email,
        issuer_name="MyApp"
    )
    
    qr = qrcode.make(totp_uri)
    qr.save(f"qr_{user_id}.png")
    
    return {"secret": secret, "qr_code_url": f"/qr/{user_id}.png"}

def verify_2fa(user_id, code):
    user = db.query("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    totp = pyotp.TOTP(user['totp_secret'])
    
    return totp.verify(code, valid_window=1)  # Allow 1 time step drift (30s)
```

### 6.2 SMS-based (Not Recommended)

**Why not SMS**:
- SIM swapping attacks
- SMS spoofing
- Telco vulnerabilities

**Better alternatives**: TOTP, WebAuthn (hardware keys).

---

## 7. Security Best Practices

### 7.1 Secure Cookie Flags

```python
response.set_cookie(
    'session_id',
    value=session_id,
    httponly=True,   # JS cannot access (XSS protection)
    secure=True,      # HTTPS only
    samesite='Strict' # CSRF protection
)
```

### 7.2 CSRF Protection

**Attack**: Malicious site executes action on your behalf.

```html
<!-- Attacker's page -->
<img src="https://bank.com/transfer?to=attacker&amount=1000">
<!-- If user is logged in, this executes! -->
```

**Defense**: CSRF Tokens

```python
from flask_wtf.csrf import CSRFProtect

csrf = CSRFProtect(app)

# Automatically adds csrf_token to forms
<form method="POST">
  <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
</form>
```

### 7.3 Rate Limiting

**Prevent brute-force**:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")  # Max 5 login attempts/minute
def login():
    # ...
```

### 7.4 Input Validation

❌ **Trusting user input**:
```python
# Vulnerable to SQL injection
query = f"SELECT * FROM users WHERE email = '{email}'"
```

✅ **Parameterized queries**:
```python
query = "SELECT * FROM users WHERE email = ?"
cursor.execute(query, (email,))
```

✅ **Schema validation (Pydantic)**:
```python
from pydantic import BaseModel, EmailStr

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@app.post('/login')
def login(user: UserLogin):  # Auto-validates email format
    # ...
```

---

## 8. OWASP Top 10

### 8.1 A01: Broken Access Control

**Vulnerability**: Users access unauthorized resources.

```python
# Bad: No authorization check
@app.route('/admin/delete_user/<user_id>')
def delete_user(user_id):
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return {"message": "User deleted"}

# Good: Check permissions
@app.route('/admin/delete_user/<user_id>')
@require_role('admin')
def delete_user(user_id):
    # ...
```

### 8.2 A02: Cryptographic Failures

❌ **Storing passwords plain text**  
❌ **Using MD5/SHA1 for passwords**  
✅ **Use bcrypt/Argon2**

### 8.3 A03: Injection

**SQL Injection**:
```python
# Vulnerable
email = request.args.get('email')
query = f"SELECT * FROM users WHERE email = '{email}'"
# Attacker: email = "' OR '1'='1"
# Result: SELECT * FROM users WHERE email = '' OR '1'='1' (returns all users!)

# Safe
query = "SELECT * FROM users WHERE email = ?"
cursor.execute(query, (email,))
```

### 8.4 A07: Identification & Authentication Failures

- ❌ Weak passwords allowed
- ❌ Session fixation
- ❌ Credentials in URLs (`/reset-password?token=...`)

---

## 9. API Security

### 9.1 API Keys

```python
@app.route('/api/data')
def get_data():
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        return {"error": "API key required"}, 401
    
    # Validate API key
    key_record = db.query("SELECT * FROM api_keys WHERE key = ?", (api_key,)).fetchone()
    
    if not key_record:
        return {"error": "Invalid API key"}, 401
    
    # Rate limit by API key
    # ...
```

### 9.2 CORS (Cross-Origin Resource Sharing)

```python
from flask_cors import CORS

# Allow specific origin
CORS(app, resources={r"/api/*": {"origins": "https://myapp.com"}})

# Or configure manually
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://myapp.com'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response
```

### 9.3 Secrets Management

❌ **Hardcoding secrets**:
```python
DB_PASSWORD = "password123"  # Committed to Git!
```

✅ **Environment variables**:
```python
import os
DB_PASSWORD = os.getenv('DB_PASSWORD')
```

✅ **Secrets manager** (AWS Secrets Manager, HashiCorp Vault):
```python
import boto3

secrets_client = boto3.client('secretsmanager')
response = secrets_client.get_secret_value(SecretId='myapp/db_password')
DB_PASSWORD = response['SecretString']
```

---

## 10. Summary

### 10.1 Key Takeaways

 1. ✅ **Password hashing** - Use bcrypt or Argon2
2. ✅ **Session vs JWT** - Sessions for web apps, JWT for APIs
3. ✅ **OAuth 2.0** - Delegate authentication to trusted providers
4. ✅ **2FA/MFA** - Add extra layer of security
5. ✅ **OWASP Top 10** - Know common vulnerabilities
6. ✅ **Rate limiting** - Prevent brute-force attacks

### 10.2 Auth Decision Tree

```mermaid
graph TD
    Start[Choose Auth Method] --> Type{App Type?}
    Type -->|Web App| Session[Session-based]
    Type -->|API/Mobile| Token[Token-based JWT]
    Type -->|Third-party| OAuth[OAuth 2.0]
    
    Token --> Stateless{Need Revocation?}
    Stateless -->|No| JWT[Short-lived JWT]
    Stateless -->|Yes| Refresh[JWT + Refresh Token]
```

### 10.3 Tomorrow (Day 11): Monolith vs Microservices

Phase 2 begins! We'll cover:
- **Monolithic Architecture**: When it's the right choice
- **Microservices**: Benefits & challenges
- **Service Communication**: REST, gRPC, Message Queues
- **Database per Service**: Pattern & trade-offs
- **Migration Strategies**: Strangler Fig pattern

See you tomorrow! 🚀

---

**File Statistics**: ~1100 lines | Authentication & Security mastered ✅
