# Day 31: OWASP Security Best Practices - Protecting Your Applications

## Table of Contents
1. [OWASP Top 10 Overview](#1-owasp-top-10-overview)
2. [SQL Injection Prevention](#2-sql-injection-prevention)
3. [XSS Prevention](#3-xss-prevention)
4. [CSRF Protection](#4-csrf-protection)
5. [Authentication Security](#5-authentication-security)
6. [Authorization Best Practices](#6-authorization-best-practices)
7. [Sensitive Data Exposure](#7-sensitive-data-exposure)
8. [Security Misconfiguration](#8-security-misconfiguration)
9. [Security Headers](#9-security-headers)
10. [Summary](#10-summary)

---

## 1. OWASP Top 10 Overview

### 1.1 OWASP Top 10 (2021)

1. **Broken Access Control** - Users access unauthorized resources
2. **Cryptographic Failures** - Sensitive data exposure
3. **Injection** - SQL, NoSQL, LDAP, OS command injection
4. **Insecure Design** - Missing/ineffective security controls
5. **Security Misconfiguration** - Default configs, verbose errors
6. **Vulnerable Components** - Outdated libraries
7. **Authentication Failures** - Weak passwords, no MFA
8. **Data Integrity Failures** - Unverified data/updates
9. **Logging Failures** - Insufficient monitoring
10. **Server-Side Request Forgery (SSRF)** - Fetching remote resources

---

## 2. SQL Injection Prevention

### 2.1 The Vulnerability

```python
# ❌ VULNERABLE CODE
user_id = request.args.get('id')
query = f"SELECT * FROM users WHERE id = {user_id}"
db.execute(query)

# Attack: ?id=1 OR 1=1
# Executed query: SELECT * FROM users WHERE id = 1 OR 1=1
# Returns ALL users!
```

### 2.2 Prevention: Parameterized Queries

```python
# ✅ SAFE: Parameterized query
user_id = request.args.get('id')
query = "SELECT * FROM users WHERE id = ?"
db.execute(query, (user_id,))

# Attack: ?id=1 OR 1=1
# Treated as literal string "1 OR 1=1", not executable code
```

### 2.3 ORM (SQLAlchemy)

```python
# ✅ SAFE: ORM automatically parameterizes
user_id = request.args.get('id')
user = db.query(User).filter(User.id == user_id).first()

# Even if user_id is malicious, ORM sanitizes it
```

### 2.4 NoSQL Injection

```python
# ❌ VULNERABLE (MongoDB)
username = request.json['username']
password = request.json['password']

user = db.users.find_one({
    "username": username,
    "password": password
})

# Attack: {"username": {"$ne": null}, "password": {"$ne": null}}
# Returns first user (bypasses authentication!)

# ✅ SAFE: Validate types
if not isinstance(username, str) or not isinstance(password, str):
    return {"error": "Invalid input"}, 400

user = db.users.find_one({
    "username": username,
    "password": hashlib.sha256(password.encode()).hexdigest()
})
```

---

## 3. XSS Prevention

### 3.1 Reflected XSS

```python
# ❌ VULNERABLE
@app.get("/search")
def search(query: str):
    return f"<h1>Results for: {query}</h1>"

# Attack: ?query=<script>alert(document.cookie)</script>
# Injected script executes in user's browser!
```

### 3.2 Prevention: HTML Escaping

```python
# ✅ SAFE
from html import escape

@app.get("/search")
def search(query: str):
    return f"<h1>Results for: {escape(query)}</h1>"

# Attack: ?query=<script>alert(1)</script>
# Rendered as: &lt;script&gt;alert(1)&lt;/script&gt; (harmless text)
```

### 3.3 Stored XSS

```python
# ❌ VULNERABLE
@app.post("/comments")
def create_comment(content: str):
    db.execute("INSERT INTO comments (content) VALUES (?)", (content,))
    return {"status": "success"}

@app.get("/comments")
def get_comments():
    comments = db.execute("SELECT content FROM comments").fetchall()
    return render_template("comments.html", comments=comments)

# comments.html
# {% for comment in comments %}
#   <p>{{ comment.content | safe }}</p>  # ❌ DANGEROUS!
# {% endfor %}
```

### 3.4 Prevention: Sanitize Output

```python
# ✅ SAFE
# comments.html
{% for comment in comments %}
  <p>{{ comment.content }}</p>  # Auto-escaped by Jinja2
{% endfor %}

# Or use DOMPurify (JavaScript)
<script src="https://cdn.jsdelivr.net/npm/dompurify@2.4.0/dist/purify.min.js"></script>
<script>
const clean = DOMPurify.sanitize(userInput);
document.getElementById('output').innerHTML = clean;
</script>
```

### 3.5 Content Security Policy (CSP)

```python
from fastapi import Response

@app.get("/")
def index(response: Response):
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'"
    )
    return render_template("index.html")

# Blocks inline scripts, only allows scripts from 'self' and cdn.jsdelivr.net
```

---

## 4. CSRF Protection

### 4.1 The Vulnerability

```html
<!-- Attacker's site -->
<form action="https://bank.com/transfer" method="POST">
  <input type="hidden" name="to" value="attacker_account">
  <input type="hidden" name="amount" value="10000">
  <input type="submit" value="Click for free prize!">
</form>

<!-- User clicks button while logged into bank.com
     → Transfer executes using user's session cookie! -->
```

### 4.2 Prevention: CSRF Tokens

```python
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
csrf = CSRFProtect(app)

@app.route("/transfer", methods=["POST"])
@csrf.exempt  # Only if explicitly needed
def transfer():
    # CSRF token automatically validated by Flask-WTF
    to_account = request.form['to']
    amount = request.form['amount']
    # Process transfer
```

**HTML Form**:
```html
<form method="POST" action="/transfer">
  {{ csrf_token() }}  <!-- Hidden CSRF token field -->
  <input name="to" value="">
  <input name="amount" value="">
  <button type="submit">Transfer</button>
</form>
```

### 4.3 SameSite Cookies

```python
from fastapi import Response

@app.post("/login")
def login(response: Response):
    # Set session cookie with SameSite=Strict
    response.set_cookie(
        key="session_id",
        value=session_token,
        httponly=True,  # Prevents JavaScript access
        secure=True,    # HTTPS only
        samesite="strict"  # Blocks cross-site requests
    )
    return {"status": "logged in"}

# SameSite=Strict → Cookie NOT sent on cross-site requests
# Prevents CSRF!
```

---

## 5. Authentication Security

### 5.1 Password Storage

```python
# ❌ NEVER store plaintext
user.password = request.json['password']  # NEVER!

# ❌ NEVER use MD5/SHA1 (too fast, crackable)
user.password = hashlib.md5(request.json['password'].encode()).hexdigest()

# ✅ GOOD: bcrypt
import bcrypt

password = request.json['password']
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
user.password = hashed

# Verify
if bcrypt.checkpw(password.encode(), user.password):
    # Correct password
```

### 5.2 Multi-Factor Authentication (MFA)

```python
import pyotp

# Generate secret for user
secret = pyotp.random_base32()
user.mfa_secret = secret

# User scans QR code with Google Authenticator
totp = pyotp.TOTP(secret)
qr_uri = totp.provisioning_uri(user.email, issuer_name="MyApp")

# Verify MFA code
@app.post("/verify-mfa")
def verify_mfa(code: str):
    totp = pyotp.TOTP(user.mfa_secret)
    
    if totp.verify(code):
        # MFA verified
        return {"status": "success"}
    else:
        return {"status": "invalid"}, 401
```

### 5.3 Rate Limiting Login Attempts

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/login")
@limiter.limit("5/minute")  # Max 5 attempts per minute
def login(username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    
    if not user or not bcrypt.checkpw(password.encode(), user.password):
        return {"error": "Invalid credentials"}, 401
    
    return {"token": generate_jwt(user)}
```

---

## 6. Authorization Best Practices

### 6.1 Broken Access Control Example

```python
# ❌ VULNERABLE: No authorization check
@app.get("/users/{user_id}/profile")
def get_profile(user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# Attack: User 123 can access /users/456/profile (not their own!)
```

### 6.2 Prevention: Ownership Check

```python
# ✅ SAFE: Verify ownership
@app.get("/users/{user_id}/profile")
def get_profile(user_id: int, current_user: User):
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    return db.query(User).filter(User.id == user_id).first()
```

### 6.3 Indirect Object Reference

```python
# ❌ VULNERABLE: Direct object reference
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, current_user: User):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    
    # Missing ownership check!
    db.delete(doc)
    db.commit()

# ✅ SAFE: Verify ownership
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, current_user: User):
    doc = db.query(Document).filter(
        Document.id == doc_id,
        Document.owner_id == current_user.id  # Owner check
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404)
    
    db.delete(doc)
    db.commit()
```

---

## 7. Sensitive Data Exposure

### 7.1 HTTPS Enforcement

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)

# All HTTP requests → redirect to HTTPS
```

### 7.2 Encryption at Rest

```python
from cryptography.fernet import Fernet

# Generate key (store securely!)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt
plaintext = "sensitive data"
encrypted = cipher.encrypt(plaintext.encode())
db.execute("INSERT INTO secrets (data) VALUES (?)", (encrypted,))

# Decrypt
encrypted_from_db = db.execute("SELECT data FROM secrets WHERE id = 1").fetchone()[0]
decrypted = cipher.decrypt(encrypted_from_db).decode()
```

### 7.3 Secure API Responses

```python
# ❌ VULNERABLE: Exposing sensitive fields
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# Returns: {"id": 123, "email": "...", "password": "hashed", "ssn": "..."}

# ✅ SAFE: Filter sensitive fields
class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    # Password, SSN excluded

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# Returns only: {"id": 123, "email": "...", "name": "..."}
```

---

## 8. Security Misconfiguration

### 8.1 Disable Debug Mode in Production

```python
# ❌ NEVER in production
app = FastAPI(debug=True)

# ✅ Production
app = FastAPI(debug=False)
```

### 8.2 Remove Default Credentials

```python
# ❌ Default admin credentials
DEFAULT_ADMIN = {"username": "admin", "password": "admin"}

# ✅ Force password change on first login
if user.is_first_login:
    return {"message": "Please change your password"}, 403
```

### 8.3 Verbose Error Messages

```python
# ❌ Exposes stack trace
@app.exception_handler(Exception)
def handle_exception(request, exc):
    return {"error": str(exc), "traceback": traceback.format_exc()}, 500

# ✅ Generic error message
@app.exception_handler(Exception)
def handle_exception(request, exc):
    # Log detailed error server-side
    logger.error(f"Error: {exc}", exc_info=True)
    
    # Return generic message to client
    return {"error": "Internal server error"}, 500
```

---

## 9. Security Headers

### 9.1 Essential Headers

```python
from fastapi import Response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions policy
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
    
    # HSTS (force HTTPS)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response
```

### 9.2 Content Security Policy

```python
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.example.com; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **SQL Injection** - Use parameterized queries/ORMs
2. ✅ **XSS** - Escape output, CSP headers
3. ✅ **CSRF** - CSRF tokens, SameSite cookies
4. ✅ **Passwords** - bcrypt, never plaintext/MD5
5. ✅ **MFA** - TOTP with pyotp
6. ✅ **Authorization** - Verify ownership, not just authentication
7. ✅ **HTTPS** - Enforce TLS, HSTS header
8. ✅ **Security Headers** - X-Frame-Options, CSP, etc.

### 10.2 Security Checklist

- [ ] All inputs validated/sanitized
- [ ] Parameterized queries (no string concatenation)
- [ ] Output escaped (HTML, JSON)
- [ ] CSRF protection enabled
- [ ] Passwords hashed with bcrypt
- [ ] MFA implemented for sensitive actions
- [ ] Authorization checks on all endpoints
- [ ] HTTPS enforced (HSTS header)
- [ ] Security headers configured
- [ ] Debug mode disabled in production
- [ ] Secrets in environment variables (not code)
- [ ] Rate limiting on authentication
- [ ] Regular security audits

### 10.3 Tomorrow (Day 32): OAuth 2.0 & OpenID Connect

- **OAuth 2.0 flows**: Authorization Code, Client Credentials, PKCE
- **OpenID Connect**: ID tokens, UserInfo endpoint
- **Social login**: Google, GitHub integration
- **JWT validation**: Signature verification, claims validation
- **Token refresh**: Access/refresh token patterns
- **Security considerations**: Token storage, PKCE for SPAs

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | OWASP Security Best Practices mastered ✅
