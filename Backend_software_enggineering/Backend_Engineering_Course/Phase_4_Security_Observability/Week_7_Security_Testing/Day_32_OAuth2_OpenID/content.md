# Day 32: OAuth 2.0 & OpenID Connect - Modern Authentication

## Table of Contents
1. [OAuth 2.0 Overview](#1-oauth-20-overview)
2. [OAuth 2.0 Flows](#2-oauth-20-flows)
3. [OpenID Connect](#3-openid-connect)
4. [Social Login Integration](#4-social-login-integration)
5. [JWT Deep Dive](#5-jwt-deep-dive)
6. [Token Refresh](#6-token-refresh)
7. [PKCE for SPAs](#7-pkce-for-spas)
8. [Security Considerations](#8-security-considerations)
9. [Production Patterns](#9-production-patterns)
10. [Summary](#10-summary)

---

## 1. OAuth 2.0 Overview

### 1.1 What is OAuth 2.0?

**OAuth 2.0**: Authorization framework (not authentication!).

**Use case**: Allow third-party app to access user's data without sharing password.

**Example**:
```
You (user) → Grant "PhotoApp" access to your Google Photos
PhotoApp can now read your photos (no password needed)
```

### 1.2 Roles

- **Resource Owner**: User (owns the data)
- **Resource Server**: API server (stores data, e.g., Google Photos API)
- **Client**: Third-party app (e.g., PhotoApp)
- **Authorization Server**: Issues tokens (e.g., accounts.google.com)

---

## 2. OAuth 2.0 Flows

### 2.1 Authorization Code Flow

**Use case**: Web applications with backend.

**Flow**:
```
1. User clicks "Login with Google"
2. Client redirects to Authorization Server:
   GET https://accounts.google.com/oauth/authorize?
       response_type=code&
       client_id=YOUR_CLIENT_ID&
       redirect_uri=https://yourapp.com/callback&
       scope=openid email profile

3. User logs in & consents
4. Authorization Server redirects back with code:
   https://yourapp.com/callback?code=AUTH_CODE

5. Client exchanges code for token (backend):
   POST https://accounts.google.com/token
   {
     "grant_type": "authorization_code",
     "code": "AUTH_CODE",
     "client_id": "YOUR_CLIENT_ID",
     "client_secret": "YOUR_SECRET",
     "redirect_uri": "https://yourapp.com/callback"
   }

6. Response:
   {
     "access_token": "ya29.a0AfH6...",
     "refresh_token": "1//0gZ...",
     "expires_in": 3600
   }
```

**Implementation (FastAPI)**:
```python
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()
oauth.register(
    name='google',
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_SECRET',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    # Store user_info in session
    return {"email": user_info['email']}
```

### 2.2 Client Credentials Flow

**Use case**: Machine-to-machine (no user involved).

**Flow**:
```python
import requests

response = requests.post('https://oauth.example.com/token', data={
    'grant_type': 'client_credentials',
    'client_id': 'YOUR_CLIENT_ID',
    'client_secret': 'YOUR_SECRET',
    'scope': 'api:read'
})

token = response.json()['access_token']

# Use token to call API
api_response = requests.get(
    'https://api.example.com/data',
    headers={'Authorization': f'Bearer {token}'}
)
```

### 2.3 Implicit Flow (Deprecated)

**Don't use!** Tokens exposed in URL (insecure).

**Use Authorization Code + PKCE instead** (see Day 32).

---

## 3. OpenID Connect

### 3.1 OAuth vs OIDC

**OAuth 2.0**: Authorization (access delegation)
**OpenID Connect (OIDC)**: Authentication (identity layer on top of OAuth)

**OIDC adds**:
- ID Token (JWT with user identity)
- UserInfo endpoint

### 3.2 ID Token

```json
{
  "iss": "https://accounts.google.com",
  "sub": "10123456789",
  "aud": "YOUR_CLIENT_ID",
  "exp": 1672531200,
  "iat": 1672527600,
  "email": "user@example.com",
  "name": "Alice Smith"
}
```

**Verification**:
```python
import jwt
from cryptography.hazmat.primitives import serialization

# Get public key from .well-known/jwks.json
public_key = get_public_key_from_jwks()

try:
    decoded = jwt.decode(
        id_token,
        public_key,
        algorithms=['RS256'],
        audience='YOUR_CLIENT_ID',
        issuer='https://accounts.google.com'
    )
    
    email = decoded['email']
except jwt.ExpiredSignatureError:
    # Token expired
except jwt.InvalidTokenError:
    # Invalid token
```

### 3.3 UserInfo Endpoint

```python
# Get additional user info
response = requests.get(
    'https://www.googleapis.com/oauth2/v3/userinfo',
    headers={'Authorization': f'Bearer {access_token}'}
)

user_info = response.json()
# {"sub": "123", "email": "...", "picture": "..."}
```

---

## 4. Social Login Integration

### 4.1 Google Login

```python
from authlib.integrations.starlette_client import OAuth

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get("/login/google")
async def google_login(request: Request):
    redirect_uri = request.url_for('google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get('userinfo')
    
    # Create or update user in DB
    user = get_or_create_user(email=user_info['email'], name=user_info['name'])
    
    # Create session
    session_token = create_session(user.id)
    
    response = RedirectResponse(url='/')
    response.set_cookie('session', session_token, httponly=True, secure=True)
    return response
```

### 4.2 GitHub Login

```python
oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    authorize_url='https://github.com/login/oauth/authorize',
    access_token_url='https://github.com/login/oauth/access_token',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'}
)

@app.get("/login/github")
async def github_login(request: Request):
    redirect_uri = request.url_for('github_callback')
    return await oauth.github.authorize_redirect(request, redirect_uri)

@app.get("/auth/github/callback")
async def github_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)
    
    # Get user info from GitHub API
    response = await oauth.github.get('user', token=token)
    user_info = response.json()
    
    # Create session
    user = get_or_create_user(email=user_info['email'], name=user_info['login'])
    session_token = create_session(user.id)
    
    response = RedirectResponse(url='/')
    response.set_cookie('session', session_token)
    return response
```

---

## 5. JWT Deep Dive

### 5.1 JWT Structure

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Header.Payload.Signature
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
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022
}
```

**Signature**:
```
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)
```

### 5.2 Creating JWTs

```python
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"

def create_access_token(user_id: int):
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# Usage
token = create_access_token(user_id=123)
```

### 5.3 Verifying JWTs

```python
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload['sub'])
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Middleware
@app.middleware("http")
async def authenticate(request: Request, call_next):
    auth_header = request.headers.get('Authorization')
    
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.replace('Bearer ', '')
        user_id = verify_token(token)
        request.state.user_id = user_id
    
    response = await call_next(request)
    return response
```

---

## 6. Token Refresh

### 6.1 Access vs Refresh Tokens

**Access Token**:
- Short-lived (1 hour)
- Sent with every API request
- If stolen, only valid for 1 hour

**Refresh Token**:
- Long-lived (30 days)
- Stored securely (httpOnly cookie)
- Used to get new access token

### 6.2 Implementation

```python
def create_tokens(user_id: int):
    # Access token (1 hour)
    access_token = jwt.encode({
        "sub": str(user_id),
        "type": "access",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }, SECRET_KEY, algorithm="HS256")
    
    # Refresh token (30 days)
    refresh_token = jwt.encode({
        "sub": str(user_id),
        "type": "refresh",
        "exp": datetime.utcnow() + timedelta(days=30)
    }, SECRET_KEY, algorithm="HS256")
    
    return access_token, refresh_token

@app.post("/login")
def login(username: str, password: str, response: Response):
    user = authenticate_user(username, password)
    
    access_token, refresh_token = create_tokens(user.id)
    
    # Store refresh token in httpOnly cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=30*24*60*60  # 30 days
    )
    
    return {"access_token": access_token}

@app.post("/refresh")
def refresh(refresh_token: str = Cookie(None)):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=["HS256"])
        
        if payload.get('type') != 'refresh':
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = int(payload['sub'])
        
        # Create new access token
        new_access_token = jwt.encode({
            "sub": str(user_id),
            "type": "access",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }, SECRET_KEY, algorithm="HS256")
        
        return {"access_token": new_access_token}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
```

---

## 7. PKCE for SPAs

### 7.1 What is PKCE?

**PKCE** (Proof Key for Code Exchange): Secure OAuth for SPAs (no client secret).

**Problem**: SPAs can't securely store client secret (JavaScript visible to users).

**Solution**: PKCE uses dynamically generated secret per request.

### 7.2 Flow

```javascript
// 1. Generate code verifier & challenge
const codeVerifier = generateRandomString(128);
const codeChallenge = base64URLEncode(sha256(codeVerifier));

// Store codeVerifier in session storage
sessionStorage.setItem('code_verifier', codeVerifier);

// 2. Authorization request
window.location.href = `https://accounts.google.com/oauth/authorize?` +
  `response_type=code&` +
  `client_id=YOUR_CLIENT_ID&` +
  `redirect_uri=${encodeURIComponent('https://yourapp.com/callback')}&` +
  `code_challenge=${codeChallenge}&` +
  `code_challenge_method=S256`;

// 3. Callback
// (User redirected back with code)
const code = new URLSearchParams(window.location.search).get('code');
const codeVerifier = sessionStorage.getItem('code_verifier');

// 4. Exchange code for token
fetch('https://accounts.google.com/token', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    grant_type: 'authorization_code',
    code: code,
    client_id: 'YOUR_CLIENT_ID',
    redirect_uri: 'https://yourapp.com/callback',
    code_verifier: codeVerifier  // No client_secret needed!
  })
})
.then(res => res.json())
.then(data => {
  const accessToken = data.access_token;
  // Store in memory (NOT localStorage!)
});
```

---

## 8. Security Considerations

### 8.1 Token Storage

❌ **Bad**: localStorage (vulnerable to XSS)
```javascript
localStorage.setItem('access_token', token);  // NEVER!
```

✅ **Good**: httpOnly cookie (backend-only) or memory (SPA)
```python
response.set_cookie('access_token', token, httponly=True, secure=True)
```

### 8.2 Validate Redirect URI

```python
ALLOWED_REDIRECT_URIS = [
    'https://yourapp.com/callback',
    'https://yourapp.com/oauth/callback'
]

@app.get("/oauth/authorize")
def authorize(redirect_uri: str):
    if redirect_uri not in ALLOWED_REDIRECT_URIS:
        raise HTTPException(status_code=400, detail="Invalid redirect_uri")
    
    # Continue authorization
```

### 8.3 State Parameter (CSRF Protection)

```python
import secrets

# Generate random state
state = secrets.token_urlsafe(32)
session['oauth_state'] = state

# Redirect to OAuth provider
return redirect(f'https://oauth.example.com/authorize?state={state}')

# Callback
@app.get("/callback")
def callback(state: str):
    if state != session.get('oauth_state'):
        raise HTTPException(status_code=400, detail="Invalid state")
    
    # Continue
```

---

## 9. Production Patterns

### 9.1 Token Rotation

```python
# Rotate refresh tokens on use
@app.post("/refresh")
def refresh(old_refresh_token: str):
    # Verify old refresh token
    user_id = verify_refresh_token(old_refresh_token)
    
    # Invalidate old token
    blacklist_token(old_refresh_token)
    
    # Issue new access & refresh tokens
    new_access_token, new_refresh_token = create_tokens(user_id)
    
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token
    }
```

### 9.2 Token Blacklisting

```python
import redis

r = redis.Redis()

def blacklist_token(token: str):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"], options={"verify_exp": False})
    exp = payload['exp']
    ttl = exp - int(datetime.utcnow().timestamp())
    
    if ttl > 0:
        r.setex(f"blacklist:{token}", ttl, "1")

def is_blacklisted(token: str):
    return r.exists(f"blacklist:{token}")

# Middleware
@app.middleware("http")
async def check_blacklist(request: Request, call_next):
    token = extract_token(request)
    
    if token and is_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token revoked")
    
    response = await call_next(request)
    return response
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **OAuth 2.0** - Authorization (access delegation)
2. ✅ **OIDC** - Authentication (identity layer)
3. ✅ **Authorization Code** - Most secure flow (web apps)
4. ✅ **PKCE** - Secure OAuth for SPAs
5. ✅ **JWT** - Stateless tokens (verify signature)
6. ✅ **Refresh Tokens** - Long-lived, httpOnly cookie
7. ✅ **Token Storage** - httpOnly cookies or memory (not localStorage)

### 10.2 OAuth Flow Comparison

| Flow | Use Case | Client Secret | Security |
|:-----|:---------|:--------------|:---------|
| **Authorization Code** | Web app (backend) | Yes | High |
| **Authorization Code + PKCE** | SPA/Mobile | No | High |
| **Client Credentials** | Machine-to-machine | Yes | Medium |
| **Implicit** | Deprecated | No | Low (don't use!) |

### 10.3 Tomorrow (Day 33): RBAC & ABAC Authorization

- **RBAC**: Role-Based Access Control (Admin, User, Guest)
- **ABAC**: Attribute-Based Access Control (context-aware)
- **Policy engines**: Open Policy Agent (OPA), Casbin
- **Permission models**: Hierarchical roles, policies
- **Production patterns**: Caching permissions, audit logs

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | OAuth 2.0 & OpenID Connect mastered ✅
