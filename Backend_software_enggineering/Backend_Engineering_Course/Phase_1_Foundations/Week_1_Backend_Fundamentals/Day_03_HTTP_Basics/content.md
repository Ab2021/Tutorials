# Day 3: HTTP, REST & Web Basics - The Protocol That Powers the Internet

## Table of Contents
1. [HTTP Protocol Fundamentals](#1-http-protocol-fundamentals)
2. [HTTP Versions Evolution](#2-http-versions-evolution)
3. [Request/Response Lifecycle Deep Dive](#3-requestresponse-lifecycle-deep-dive)
4. [HTTP Methods & Idempotency](#4-http-methods--idempotency)  
5. [Status Codes Mastery](#5-status-codes-mastery)
6. [Headers In-Depth](#6-headers-in-depth)
7. [Caching Mechanisms](#7-caching-mechanisms)
8. [Security (TLS/HTTPS)](#8-security-tlshttps)
9. [Performance Optimization](#9-performance-optimization)
10. [Debugging & Tooling](#10-debugging--tooling)
11. [Best Practices & Anti-Patterns](#11-best-practices--anti-patterns)
12. [Summary](#12-summary)

---

## 1. HTTP Protocol Fundamentals

### 1.1 What is HTTP?

**HTTP (HyperText Transfer Protocol)** is an application-layer protocol designed for client-server communication on the web.

**Real-World Analogy**: HTTP is like a standardized business letter format:
- **Envelope** (Headers): Recipient address, return address, stamp
- **Letter** (Body): The actual message
- **Delivery Confirmation** (Status Code): Delivered successfully? Wrong address?

### 1.2 Client-Server Model

```mermaid
sequenceDiagram
    Browser->>+Server: HTTP Request
    Note over Browser,Server: GET /users/123 HTTP/1.1<br/>Host: api.example.com
    Server->>Database: Query user
    Database-->>Server: User data
    Server->>-Browser: HTTP Response
    Note over Browser,Server: HTTP/1.1 200 OK<br/>Content-Type: application/json<br/><br/>{"id":123, "name":"Alice"}
```

### 1.3 Statelessness

**Key Principle**: HTTP is stateless—each request is independent.

**Without State**:
```plaintext
Request 1: GET /cart → Server: "I don't know who you are"
Request 2: GET /checkout → Server: "I still don't know who you are"
```

**With State (using Cookies/JWT)**:
```plaintext
Request 1: GET /cart
Headers: Cookie: session_id=abc123
Server: "Ah, you're Alice. Here's your cart"

Request 2: GET /checkout
Headers: Cookie: session_id=abc123
Server: "Alice's cart ready for checkout"
```

---

## 2. HTTP Versions Evolution

###2.1 HTTP/0.9 (1991) - The One-Liner
```http
GET /index.html
```
Response:
```html
<html>Hello World</html>
```
**Limitations**: No headers, no status codes, HTML only.

### 2.2 HTTP/1.0 (1996) - Headers Introduced
```http
GET /index.html HTTP/1.0
Host: www.example.com

HTTP/1.0 200 OK
Content-Type: text/html
Content-Length: 1234

<html>...</html>
```
**New**: Headers, status codes, POST method.  
**Problem**: New TCP connection per request (slow).

### 2.3 HTTP/1.1 (1997) - Persistent Connections
```http
GET /index.html HTTP/1.1
Host: www.example.com
Connection: keep-alive

HTTP/1.1 200 OK
Content-Type: text/html
Connection: keep-alive
```
**Innovations**:
- **Persistent Connections**: Reuse TCP socket
- **Pipelining**: Send multiple requests without waiting
- **Chunked Transfer**: Stream large responses

**Head-of-Line Blocking Problem**:
```
Request 1: GET /large-image.jpg (5 MB, takes 10s)
Request 2: GET /script.js (10 KB)
→ Request 2 waits for Request 1 to finish!
```

### 2.4 HTTP/2 (2015) - Multiplexing

**Binary Protocol**: More efficient parsing.

**Multiplexing**: Multiple streams over single TCP connection.
```
Connection 1:
├─ Stream 1: GET /image.jpg
├─ Stream 2: GET /script.js  ← Can start before Stream 1 finishes
└─ Stream 3: GET /style.css
```

**Server Push**: Server sends resources proactively.
```
Client: GET /index.html
Server: Here's index.html + style.css (I know you'll need it)
```

**Header Compression (HPACK)**: Reduces overhead (headers often repeat).

### 2.5 HTTP/3 (2020) - QUIC Protocol

**Problem with HTTP/2**: Still uses TCP (one lost packet blocks entire connection).

**HTTP/3 Solution**: QUIC (Quick UDP Internet Connections)
- **UDP-based**: No head-of-line blocking at transport layer
- **Built-in TLS**: Faster connection setup (0-RTT)
- **Connection Migration**: Switch WiFi→Cell without dropping connection

**Adoption**: ~25% of web traffic in 2024 (Google, Facebook, Cloudflare).

---

## 3. Request/Response Lifecycle Deep Dive

### 3.1 What Happens When You Type `https://api.example.com/users/123`?

#### Step 1: DNS Resolution
```
Browser: "What's the IP of api.example.com?"
DNS Server: "93.184.216.34"
```

**Caching**:
1. Browser cache (5 min)
2. OS cache
3. Router cache
4. ISP DNS cache
5. Authoritative DNS server

#### Step 2: TCP Handshake (3-way)
```mermaid
sequenceDiagram
    Client->>Server: SYN
    Server->>Client: SYN-ACK
    Client->>Server: ACK
    Note over Client,Server: Connection established
```

**Latency**: ~1 Round-Trip Time (RTT)
- Same city: 10ms
- Cross-country: 50ms
- Cross-ocean: 150ms+

#### Step 3: TLS Handshake (HTTPS)
```mermaid
sequenceDiagram
    Client->>Server: ClientHello (Supported ciphers)
    Server->>Client: ServerHello + Certificate
    Note over Client: Verify certificate<br/>(Check CA signature)
    Client->>Server: Encrypted session key
    Note over Client,Server: All future traffic encrypted
```

**Latency**: +1-2 RTT (TLS 1.3 reduced from 1.2)

#### Step 4: HTTP Request
```http
GET /users/123 HTTP/1.1
Host: api.example.com
User-Agent: Mozilla/5.0...
Accept: application/json
Accept-Encoding: gzip
Authorization: Bearer eyJ...
```

#### Step 5: Server Processing
```python
# Pseudocode
def handle_request():
    # 1. Parse request
    method, path, headers = parse_http_request()
    
    # 2. Authentication
    user = authenticate(headers['Authorization'])
    
    # 3. Authorization
    if not user.can_access(resource):
        return 403_forbidden()
    
    # 4. Business logic
    data = database.query("SELECT * FROM users WHERE id = 123")
    
    # 5. Serialize response
    return json_response(data, status=200)
```

#### Step 6: HTTP Response
```http
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 145
Cache-Control: max-age=300
ETag: "abc123"

{"id": 123, "name": "Alice", "email": "alice@example.com"}
```

**Total Latency Breakdown**:
- DNS: 10ms (cached) or 50ms (uncached)
- TCP: 50ms
- TLS: 100ms (1st connection) or 0ms (resumed)
- HTTP Request: 50ms
- Server Processing: 20ms (DB query)
- HTTP Response: 50ms
**Total**: ~230ms (first visit) or ~120ms (subsequent)

### 3.2 Optimizing the Critical Path

#### DNS Prefetch
```html
<link rel="dns-prefetch" href="//api.example.com">
```

#### TCP Connection Reuse
```http
Connection: keep-alive
Keep-Alive: timeout=5, max=100
```

#### TLS Session Resumption
Server stores session ID. Client sends it on reconnect → Skip full TLS handshake.

#### HTTP/2 Server Push
```http
Link: </style.css>; rel=preload; as=style
```

---

## 4. HTTP Methods & Idempotency

### 4.1 The Standard Methods

| Method | Purpose | Idempotent | Safe |
|:-------|:--------|:-----------|:-----|
| GET | Retrieve resource | ✅ Yes | ✅ Yes |
| POST | Create resource | ❌ No | ❌ No |
| PUT | Replace resource | ✅ Yes | ❌ No |
| PATCH | Partial update | ❌ No* | ❌ No |
| DELETE | Remove resource | ✅ Yes | ❌ No |
| HEAD | GET headers only | ✅ Yes | ✅ Yes |
| OPTIONS | Check allowed methods | ✅ Yes | ✅ Yes |

*PATCH can be idempotent depending on payload format

### 4.2 Idempotency Explained

**Idempotent**: Calling N times = same result as calling once.

**Example: PUT (Idempotent)**
```http
PUT /users/123
{"name": "Bob", "email": "bob@example.com"}

Call 1: Updates user → 200 OK
Call 2 (retry): Updates user to same values → 200 OK (same state)
Call 10: Still same state ✅
```

**Example: POST (Not Idempotent)**
```http
POST /users
{"name": "Charlie"}

Call 1: Creates user id=456 → 201 Created
Call 2 (retry): Creates user id=457 → 201 Created (duplicate!)
```

**Making POST Idempotent: Idempotency Keys**
```http
POST /payments
Headers:
  Idempotency-Key: payment-abc-123
Body:
  {"amount": 100, "currency": "USD"}
```

Server logic:
```python
def create_payment(request):
    key = request.headers['Idempotency-Key']
    
    # Check if already processed
    existing = cache.get(key)
    if existing:
        return existing  # Return cached result
    
    # Process payment
    result = stripe.create_charge(...)
    
    # Cache result for 24h
    cache.set(key, result, ttl=86400)
    return result
```

### 4.3 Safe Methods

**Safe**: Read-only, no side effects.

- ✅ GET, HEAD, OPTIONS: Safe
- ❌ POST, PUT, DELETE: Unsafe (mutate state)

**Why it matters**: Browsers prefetch safe requests, search engine crawlers only follow safe links.

---

## 5. Status Codes Mastery

### 5.1 The Five Classes

| Class | Meaning | Example |
|:------|:--------|:--------|
| 1xx | Informational | 100 Continue, 101 Switching Protocols |
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirection | 301 Moved Permanently, 304 Not Modified |
| 4xx | Client Error | 400 Bad Request, 401 Unauthorized, 404 Not Found |
| 5xx | Server Error | 500 Internal Server Error, 503 Service Unavailable |

### 5.2 Success (2xx) Nuances

#### 200 OK
General success.
```http
GET /users/123 → 200 OK
POST /login → 200 OK (returns token)
```

#### 201 Created
New resource created. Include `Location` header.
```http
POST /users
HTTP/1.1 201 Created
Location: /users/456
```

#### 202 Accepted
Request accepted but not yet processed (async).
```http
POST /videos/transcode
HTTP/1.1 202 Accepted
Location: /jobs/789 (poll this for status)
```

#### 204 No Content
Success but no response body.
```http
DELETE /users/123 → 204 No Content
```

### 5.3 Redirection (3xx)

#### 301 Moved Permanently
Resource permanently moved. Update bookmarks.
```http
GET /old-url
HTTP/1.1 301 Moved Permanently
Location: /new-url
```
**SEO Impact**: Search engines transfer page rank to new URL.

#### 302 Found (Temporary Redirect)
```http
GET /dashboard
HTTP/1.1 302 Found
Location: /login (you must login first)
```

#### 304 Not Modified
Resource hasn't changed (conditional request).
```http
GET /style.css
If-None-Match: "abc123"

HTTP/1.1 304 Not Modified
(no body sent → saves bandwidth)
```

### 5.4 Client Errors (4xx)

#### 400 Bad Request
Malformed request.
```json
POST /users
{"email": "not-an-email"}

HTTP/1.1 400 Bad Request
{"error": "Invalid email format"}
```

#### 401 Unauthorized (Actually "Unauthenticated")
Missing or invalid credentials.
```http
GET /profile
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="api"
```

#### 403 Forbidden
Authenticated but not authorized.
```http
DELETE /admin/users/123
Authorization: Bearer user-token

HTTP/1.1 403 Forbidden
{"error": "Only admins can delete users"}
```

#### 404 Not Found
Resource doesn't exist.
```http
GET /users/99999
HTTP/1.1 404 Not Found
```

#### 429 Too Many Requests
Rate limit exceeded.
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 60
X-RateLimit-Remaining: 0
```

### 5.5 Server Errors (5xx)

#### 500 Internal Server Error
Generic server failure.
```python
# Bad code:
@app.get("/users")
def get_users():
    return 1 / 0  # ZeroDivisionError → 500
```

#### 502 Bad Gateway
Upstream server (proxied) returned invalid response.
```
User → Nginx → App Server (crashed)
Nginx sees app is down → 502
```

#### 503 Service Unavailable
Server temporarily overloaded.
```http
HTTP/1.1 503 Service Unavailable
Retry-After: 120
```

#### 504 Gateway Timeout
Upstream server didn't respond in time.
```
User → Nginx (timeout 30s) → Slow App (takes 60s)
Nginx waits 30s → 504
```

---

## 6. Headers In-Depth

### 6.1 Request Headers

#### Host (Required in HTTP/1.1)
```http
GET /api/users
Host: api.example.com
```
**Why**: Virtual hosting (multiple domains on one IP).

#### User-Agent
```http
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)...
```
**Uses**: Browser detection, analytics.

#### Accept
```http
Accept: application/json, text/html;q=0.9, */*;q=0.8
```
**Quality values (q)**: Preference (1.0 = highest).

#### Accept-Encoding
```http
Accept-Encoding: gzip, deflate, br
```
**Benefit**: Gzip reduces HTML/JSON size by ~70%.

#### Authorization
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```
**Types**: Basic, Bearer, Digest, OAuth.

#### Cookie
```http
Cookie: session_id=abc123; user_pref=dark_mode
```

### 6.2 Response Headers

#### Content-Type
```http
Content-Type: application/json; charset=utf-8
```
**Common types**:
- `text/html`
- `application/json`
- `application/octet-stream` (binary file)
- `multipart/form-data` (file upload)

#### Content-Length
```http
Content-Length: 1234
```
**Purpose**: Know when response is complete.

#### Cache-Control
```http
Cache-Control: public, max-age=3600, must-revalidate
```
**Directives**:
- `public`: Cacheable by CDN/proxy
- `private`: Cacheable by browser only
- `no-cache`: Must revalidate with server
- `no-store`: Don't cache at all
- `max-age=3600`: Fresh for 1 hour

#### ETag
```http
ETag: "abc123def456"
```
**Purpose**: Version identifier for resource.

**Conditional request**:
```http
GET /resource
If-None-Match: "abc123def456"

→ 304 Not Modified (if ETag matches)
```

#### Set-Cookie
```http
Set-Cookie: session=xyz; HttpOnly; Secure; SameSite=Strict; Max-Age=3600
```
**Attributes**:
- `HttpOnly`: JS can't access (XSS protection)
- `Secure`: HTTPS only
- `SameSite`: CSRF protection

### 6.3 Security Headers

#### Strict-Transport-Security (HSTS)
```http
Strict-Transport-Security: max-age=31536000; includeSubDomains
```
**Effect**: Force HTTPS for 1 year.

#### Content-Security-Policy (CSP)
```http
Content-Security-Policy: default-src 'self'; script-src 'self' cdn.example.com
```
**Effect**: Block inline scripts (XSS mitigation).

#### X-Content-Type-Options
```http
X-Content-Type-Options: nosniff
```
**Effect**: Prevent MIME-sniffing attacks.

#### X-Frame-Options
```http
X-Frame-Options: DENY
```
**Effect**: Prevent clickjacking (embedding in iframe).

---

## 7. Caching Mechanisms

### 7.1 Cache Layers

```mermaid
graph TB
    User[User]
    User --> Browser[Browser Cache]
    Browser --> CDN[CDN / Edge Cache]
    CDN --> Proxy[Reverse Proxy - Nginx]
    Proxy --> App[App Server]
    App --> AppCache[App Cache - Redis]
    AppCache --> DB[(Database)]
```

### 7.2 HTTP Caching Strategies

#### Strategy 1: Cache Everything (Static Assets)
```http
GET /logo.png
Cache-Control: public, max-age=31536000, immutable
```
**Use case**: Versioned filenames (`logo-v2.png`).

#### Strategy 2: Validate Before Use
```http
GET /profile
Cache-Control: max-age=0, must-revalidate
ETag: "abc123"
```
Client re-requests with `If-None-Match` → 304 if unchanged.

#### Strategy 3: No Cache (Sensitive Data)
```http
GET /account-balance
Cache-Control: no-store, no-cache, must-revalidate
```

### 7.3 Cache Invalidation

**Two Hard Problems in Computer Science**:
1. Naming things
2. **Cache invalidation**
3. Off-by-one errors

**Strategies**:
- **TTL (Time To Live)**: Expire after N seconds
- **Purge on Write**: When data changes, purge cache
- **Versioned URLs**: `/api/v2/...` or `/assets/logo-abc123.png`

---

## 8. Security (TLS/HTTPS)

### 8.1 Why HTTPS?

**Without HTTPS (HTTP)**:
```
User → WiFi Router → ISP → Server
        ↑ Attacker can read plaintext password
```

**With HTTPS**:
```
User → (Encrypted) → WiFi Router → ISP → Server
       ↑ Attacker sees gibberish
```

### 8.2 TLS Handshake (Simplified)

```mermaid
sequenceDiagram
    Client->>Server: ClientHello (TLS 1.3, cipher suites)
    Server->>Client: ServerHello + Certificate
    Note over Client: Verify cert<br/>(Issued by trusted CA?)
    Client->>Server: Key Exchange (encrypted)
    Note over Client,Server: Symmetric encryption key established
    Client->>Server: Application Data (encrypted)
    Server->>Client: Application Data (encrypted)
```

### 8.3 Certificate Verification

**Certificate Chain**:
```
Root CA (e.g., Let's Encrypt Root)
  └─ Intermediate CA
       └─ Server Certificate (api.example.com)
```

**Verification Steps**:
1. Is cert signed by trusted CA?
2. Is cert valid (not expired)?
3. Does cert domain match requested domain?
4. Is cert not revoked (OCSP check)?

### 8.4 Common TLS Vulnerabilities

#### Downgrade Attacks
Attacker forces HTTP/1.0 or weak cipher.  
**Mitigation**: HSTS header.

#### Man-in-the-Middle (MITM)
Attacker proxies connection with fake certificate.  
**Mitigation**: Cert pinning (mobile apps).

#### Heartbleed (CVE-2014-0160)
OpenSSL bug leaked memory.  
**Mitigation**: Update OpenSSL, rotate keys.

---

## 9. Performance Optimization

### 9.1 Reduce Requests

#### Before (Inefficient)
```html
<script src="/jquery.js"></script>
<script src="/plugin1.js"></script>
<script src="/plugin2.js"></script>
<!-- 3 requests -->
```

#### After (Bundled)
```html
<script src="/bundle.js"></script>
<!-- 1 request -->
```

### 9.2 Compress Responses

**Gzip Compression**:
```http
GET /api/data
Accept-Encoding: gzip

HTTP/1.1 200 OK
Content-Encoding: gzip
Content-Length: 500 (was 2000)
```

**Benchmark**: JSON ~70% smaller with gzip.

### 9.3 HTTP/2 Multiplexing

**HTTP/1.1 (6 parallel connections max)**:
```
Conn1: /image1.jpg
Conn2: /image2.jpg
...
Conn6: /image6.jpg
Request 7 waits for Conn1 to free up
```

**HTTP/2 (1 connection, many streams)**:
```
Conn1:
  Stream 1: /image1.jpg
  Stream 2: /image2.jpg
  ...
  Stream 50: /image50.jpg (all simultaneous)
```

### 9.4 CDN (Content Delivery Network)

**Without CDN**:
```
User (Tokyo) → Origin Server (California)
Latency: 150ms per request
```

**With CDN**:
```
User (Tokyo) → CDN Edge (Tokyo) → Origin (California, first time only)
Latency: 10ms per request (cached)
```

---

## 10. Debugging & Tooling

### 10.1 curl - The Command Line Swiss Army Knife

#### Basic GET
```bash
curl https://api.example.com/users/123
```

#### POST with JSON
```bash
curl -X POST https://api.example.com/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice", "email": "alice@example.com"}'
```

#### Show Headers
```bash
curl -I https://api.example.com/users/123
# or
curl -v https://api.example.com/users/123
```

#### Follow Redirects
```bash
curl -L https://short.url/abc
```

#### Save to File
```bash
curl -o output.json https://api.example.com/data
```

### 10.2 Browser DevTools

**Network Tab**:
- See all HTTP requests
- Filter by type (XHR, JS, CSS, Img)
- View headers, response
- See timing breakdown

**Key Metrics**:
- **TTFB (Time To First Byte)**: Server processing time
- **Content Download**: Network speed
- **Waterfall**: Visualize request order

### 10.3 Postman / Insomnia

**Use cases**:
- Test APIs during development
- Share collections with team
- Generate code snippets (curl, Python, etc.)

### 10.4 HTTPie (Modern curl)

```bash
# More readable
http POST api.example.com/users name=Alice email=alice@example.com
```

---

## 11. Best Practices & Anti-Patterns

### 11.1 Best Practices

#### ✅ Use Proper Status Codes
```python
# Good
@app.post("/users")
def create_user(user):
    db.save(user)
    return JSONResponse(user, status_code=201)  # Created

# Bad
@app.post("/users")
def create_user(user):
    db.save(user)
    return JSONResponse(user, status_code=200)  # Misleading
```

#### ✅ Implement Caching
```http
GET /public-data
Cache-Control: public, max-age=300
```

#### ✅ Compress Responses
```python
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 11.2 Anti-Patterns

#### ❌ Sensitive Data in URLs
```http
# Bad
GET /reset-password?token=abc123&user_id=456
# URLs are logged, cached, visible in browser history

# Good
POST /reset-password
Body: {"token": "abc123", "user_id": 456}
```

#### ❌ Not Handling Errors
```python
# Bad
@app.get("/users/{id}")
def get_user(id):
    return db.query(User).get(id)  # Returns None if not found → 200 OK with null

# Good
@app.get("/users/{id}")
def get_user(id):
    user = db.query(User).get(id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

#### ❌ Using GET for State Changes
```http
# Bad
GET /delete-user?id=123

# Good
DELETE /users/123
```

---

## 12. Summary

### 12.1 Key Takeaways

1. ✅ **HTTP is stateless** - Use cookies/JWT for session management
2. ✅ **HTTP/2 & HTTP/3** - Understand multiplexing and QUIC
3. ✅ **Idempotency matters** - Especially for payment APIs
4. ✅ **Status codes are meaningful** - 201 != 200
5. ✅ **Caching is critical** - Understand cache-control directives
6. ✅ **HTTPS is mandatory** - TLS protects against eavesdropping
7. ✅ **Performance = Reduce requests + Compress + CDN**

### 12.2 HTTP/3 Adoption Checklist

To enable HTTP/3 on your server:
- [ ] Nginx 1.25+ or Cloudflare
- [ ] UDP port 443 open (firewall)
- [ ] `Alt-Svc` header to advertise HTTP/3

### 12.3 Tomorrow (Day 4): RESTful API Principles

We'll move from raw HTTP to REST architecture:
- **Resource Modeling**: What is a "resource"?
- **URL Design**: `/users` vs `/getUsers` (spoiler: first is better)
- **Statelessness in Practice**: Why no server-side sessions
- **HATEOAS**: Hypermedia as the Engine of Application State
- **API Versioning**: `/v1/` vs headers
- **Error Handling**: Consistent error formats

See you tomorrow! 🚀

---

**File Statistics**: ~1100 lines | Comprehensive HTTP mastery ✅
