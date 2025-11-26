# Day 4: RESTful API Principles - Building Predictable, Scalable APIs

## Table of Contents
1. [REST Architecture Fundamentals](#1-rest-architecture-fundamentals)
2. [Resource Modeling](#2-resource-modeling)
3. [URL Design Best Practices](#3-url-design-best-practices)
4. [HTTP Methods Applied to REST](#4-http-methods-applied-to-rest)
5. [Representations & Content Negotiation](#5-representations--content-negotiation)
6. [HATEOAS - Hypermedia Driven APIs](#6-hateoas---hypermedia-driven-apis)
7. [Statelessness in Practice](#7-statelessness-in-practice)
8. [Error Handling & Status Codes](#8-error-handling--status-codes)
9. [Pagination, Filtering, Sorting](#9-pagination-filtering-sorting)
10. [API Versioning Strategies](#10-api-versioning-strategies)
11. [Best Practices & Anti-Patterns](#11-best-practices--anti-patterns)
12. [Summary](#12-summary)

---

## 1. REST Architecture Fundamentals

### 1.1 What is REST?

**REST (REpresentational State Transfer)** is an architectural style for designing networked applications, defined by Roy Fielding in his 2000 PhD dissertation.

**Core Idea**: The web already works great (URLs, HTTP). Let's build APIs the same way.

**Real-World Analogy**: Think of REST like a library:
- **Resources** = Books (each has a unique ID)
- **Representations** = Different formats (hardcover, ebook, audiobook)
- **Methods** = Actions (borrow, return, reserve)
- **Stateless** = Librarian doesn't remember previous visits (you show your library card each time)

### 1.2 The Six REST Constraints

#### 1. Client-Server Separation
**Principle**: Frontend and backend are independent.  
**Benefit**: Frontend can be React/Vue/Swift. Backend doesn't care.

#### 2. Statelessness
**Principle**: Each request contains all information needed.  
**Benefit**: Easy to scale (no session affinity required).

#### 3. Cacheability
**Principle**: Responses must define if they're cacheable.  
**Benefit**: Reduced load, faster responses.

#### 4. Uniform Interface
**Principle**: Consistent patterns across all endpoints.  
**Benefit**: Predictable, easy to learn.

#### 5. Layered System
**Principle**: Client doesn't know if talking to origin server or proxy.  
**Benefit**: Can insert Load Balancers, CDNs transparently.

#### 6. Code on Demand (Optional)
**Principle**: Server can send executable code (JavaScript).  
**Rarely used**: Most APIs just send data.

### 1.3 RESTful vs REST-like

Many APIs claim to be "RESTful" but actually just use HTTP + JSON.

**REST Maturity Model (Richardson)**:

| Level | Description | Example |
|:------|:------------|:--------|
| 0 | HTTP as transport | SOAP over HTTP |
| 1 | Resources | `/users`, `/posts` (not `/getAllUsers`) |
| 2 | HTTP Verbs | `GET /users`, `POST /users` |
| 3 | HATEOAS | Responses include links to related resources |

**Reality**: Most APIs are Level 2. Level 3 (HATEOAS) is rare.

---

## 2. Resource Modeling

### 2.1 What is a Resource?

**Resource**: Anything that can be named and manipulated.

**Examples**:
- Users: `/users, /users/123`
- Blog posts: `/posts`, `/posts/456`
- Comments: `/posts/456/comments`
- Current user: `/me`
- Search results: `/search?q=rest`

**Not Resources** (Avoid):
- `/getUser`, `/createPost` (these are RPC-style, not REST)

### 2.2 Nouns, Not Verbs

#### ❌ Bad (RPC-Style)
```
POST /createUser
POST /updateUser
POST /deleteUser
GET /getAllUsers
```

#### ✅ Good (RESTful)
```
POST /users (create)
PUT /users/123 (update)
DELETE /users/123 (delete)
GET /users (list all)
```

### 2.3 Singular vs Plural

**Convention**: Use plural for collections.

```
GET /users → List of users
GET /users/123 → Single user
POST /users → Create user
```

**Exception**: Singleton resources
```
GET /me → Current authenticated user
GET /profile → Current user's profile
```

###2.4 Nested Resources

**Hierarchy**: Model relationships in URLs.

```
/users/123/posts → Posts by user 123
/posts/456/comments → Comments on post 456
/posts/456/comments/789 → Specific comment
```

**Limit Nesting**: Max 2-3 levels. Beyond that, use query params.

```
❌ /orgs/1/teams/2/projects/3/tasks/4/comments/5
✅ /comments/5?task_id=4
```

---

## 3. URL Design Best Practices

### 3.1 URL Structure

**Anatomy**:
```
https://api.example.com/v1/users/123?include=posts&sort=name
└─────┬────────────┘ └┬┘ └──┬──┘└─┬─┘ └────────┬────────┘
    Base URL       Version Resource ID    Query Parameters
```

### 3.2 Naming Conventions

#### ✅ Use Lowercase
```
✅ /users
❌ /Users, /USERS
```

#### ✅ Use Hyphens for Multi-Word
```
✅ /user-profiles
❌ /userProfiles, /user_profiles
```

#### ✅ Avoid File Extensions
```
✅ /users/123
❌ /users/123.json
```
Use `Accept` header instead.

### 3.3 Query Parameters

**Use for**:
- Filtering: `/users?role=admin`
- Sorting: `/users?sort=created_at:desc`
- Pagination: `/users?page=2&per_page=20`
- Searching: `/users?q=alice`
- Partial responses: `/users/123?fields=name,email`

**NOT for actions**:
```
❌ /users?action=delete
✅ DELETE /users/123
```

### 3.4 Real-World Examples

**GitHub API**:
```
GET /repos/facebook/react/issues?state=open&sort=created
GET /users/torvalds/repos
GET /search/repositories?q=machine+learning&sort=stars
```

**Stripe API**:
```
GET /v1/customers/cus_123
POST /v1/charges
GET /v1/invoices?customer=cus_123&status=paid
```

---

## 4. HTTP Methods Applied to REST

### 4.1 CRUD Mapping

| Operation | HTTP Method | URL | Request Body | Success Code |
|:----------|:------------|:----|:-------------|:-------------|
| Create | POST | `/users` | User data | 201 Created |
| Read (list) | GET | `/users` | None | 200 OK |
| Read (single) | GET | `/users/123` | None | 200 OK |
| Update (replace) | PUT | `/users/123` | Full user data | 200 OK |
| Update (partial) | PATCH | `/users/123` | Changed fields | 200 OK |
| Delete | DELETE | `/users/123` | None | 204 No Content |

### 4.2 POST vs PUT vs PATCH

#### POST - Create New Resource
```http
POST /users
Content-Type: application/json

{"name": "Alice", "email": "alice@example.com"}

→ 201 Created
Location: /users/456
{"id": 456, "name": "Alice", "email": "alice@example.com"}
```

**Server generates ID**.

#### PUT - Replace Entire Resource
```http
PUT /users/456
Content-Type: application/json

{"name": "Alice Smith", "email": "alice@example.com", "age": 30}

→ 200 OK
{"id": 456, "name": "Alice Smith", "email": "alice@example.com", "age": 30}
```

**Client provides full representation**. Omitted fields are nulled.

#### PATCH - Partial Update
```http
PATCH /users/456
Content-Type: application/json

{"email": "newemail@example.com"}

→ 200 OK
{"id": 456, "name": "Alice Smith", "email": "newemail@example.com", "age": 30}
```

**Only specified fields change**.

### 4.3 Idempotency in Practice

**GET, PUT, DELETE are idempotent**:
```http
DELETE /users/123 → 204 No Content
DELETE /users/123 → 404 Not Found (idempotent: resource gone)
DELETE /users/123 → 404 Not Found (same result)
```

**POST is NOT idempotent** (usually):
```http
POST /users {"name": "Bob"}
→ Creates user id=1

POST /users {"name": "Bob"}
→ Creates user id=2 (duplicate!)
```

**Making POST idempotent**:
```http
POST /users
Idempotency-Key: unique-client-uuid

→ Server checks if key exists, returns cached response if yes
```

---

## 5. Representations & Content Negotiation

### 5.1 Multiple Representations

A resource can have multiple representations:
- JSON
- XML
- HTML
- CSV

**Example**: `/users/123`
- As JSON: `{"id": 123, "name": "Alice"}`
- As XML: `<user><id>123</id><name>Alice</name></user>`
- As HTML: `<h1>Alice</h1>`

### 5.2 Content Negotiation

**Client specifies desired format**:
```http
GET /users/123
Accept: application/json

→ HTTP/1.1 200 OK
Content-Type: application/json
{"id": 123, "name": "Alice"}
```

**Server doesn't support requested format**:
```http
GET /users/123
Accept: application/xml

→ HTTP/1.1 406 Not Acceptable
{"error": "Only application/json supported"}
```

### 5.3 Language Negotiation
```http
GET /articles/456
Accept-Language: fr, en;q=0.8

→ HTTP/1.1 200 OK
Content-Language: fr
{" titre": "Bonjour"}
```

---

## 6. HATEOAS - Hypermedia Driven APIs

### 6.1 What is HATEOAS?

**HATEOAS**: Hypermedia As The Engine Of Application State

**Idea**: API responses include links to related actions.

**Why**: Client doesn't need to hard-code URLs. API is discoverable.

### 6.2 Example

**Without HATEOAS**:
```json
GET /users/123
{
  "id": 123,
  "name": "Alice"
}
```
Client must know `/users/123/posts` exists.

**With HATEOAS**:
```json
GET /users/123
{
  "id": 123,
  "name": "Alice",
  "_links": {
    "self": {"href": "/users/123"},
    "posts": {"href": "/users/123/posts"},
    "followers": {"href": "/users/123/followers"}
  }
}
```

Client can discover available actions.

### 6.3 HAL (Hypertext Application Language)

**Standard format** for HATEOAS:
```json
{
  "_links": {
    "self": {"href": "/orders/123"},
    "cancel": {"href": "/orders/123/cancel", "method": "POST"},
    "items": {"href": "/orders/123/items"}
  },
  "id": 123,
  "total": 99.99,
  "status": "pending"
}
```

### 6.4 Adoption

**Reality**: HATEOAS is **rarely used** in practice.
- GraphQL solves discoverability differently
- Hard to implement properly
- Increased payload size

**Who uses it**: PayPal, GitHub (partially).

---

## 7. Statelessness in Practice

### 7.1 What is Stateless?

**Stateless**: Server doesn't store client session.

**Each request is self-contained**:
```http
Request 1:
GET /cart
Authorization: Bearer <JWT>

Request 2:
POST /cart/items
Authorization: Bearer <JWT>
Body: {"product_id": 456}
```

Server doesn't "remember" Request 1 when processing Request 2.

### 7.2 Authentication Without Sessions

**Traditional (Stateful)**:
```
POST /login → Server creates session, stores in Redis → Set-Cookie: session_id=abc
GET /profile → Cookie: session_id=abc → Server looks up session in Redis
```

**Stateless (JWT)**:
```
POST /login → Server generates JWT (signed token) → Return JWT to client
GET /profile → Authorization: Bearer <JWT> → Server verifies signature (no DB lookup)
```

### 7.3 Benefits of Statelessness

1. **Horizontal Scaling**: Any server can handle any request
```
Request 1 → Server A
Request 2 → Server B (doesn't need to talk to Server A)
```

2. **No session synchronization**: No Redis/DB required for sessions

3. **Fault tolerance**: Server crash doesn't lose sessions

### 7.4 Trade-offs

**Downside**: Can't revoke JWT before expiry.

**Solution**: Short-lived tokens (15 min) + refresh tokens.

---

## 8. Error Handling & Status Codes

### 8.1 Consistent Error Format

**RFC 7807 Problem Details**:
```json
HTTP/1.1 400 Bad Request
Content-Type: application/problem+json

{
  "type": "https://example.com/errors/invalid-email",
  "title": "Invalid Email Format",
  "status": 400,
  "detail": "The email 'not-an-email' is not valid",
  "instance": "/users",
  "invalid_fields": ["email"]
}
```

### 8.2 Common Patterns

#### Validation Errors (422 Unprocessable Entity)
```json
{
  "error": {
    "message": "Validation failed",
    "fields": {
      "email": "Invalid email format",
      "password": "Minimum 8 characters required"
    }
  }
}
```

#### Business Logic Errors (400 Bad Request)
```json
{
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Account balance is $50. Cannot withdraw $100."
  }
}
```

#### Not Found (404)
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "User with ID 999 does not exist"
  }
}
```

### 8.3 Error Code Conventions

```
GET /users/123 (doesn't exist) → 404 Not Found
POST /users (missing required field) → 400 Bad Request
POST /users (email already exists) → 409 Conflict
GET /admin (non-admin user) → 403 Forbidden
GET /profile (no auth header) → 401 Unauthorized
POST /users (rate limit exceeded) → 429 Too Many Requests
GET /users (DB down) → 503 Service Unavailable
```

---

## 9. Pagination, Filtering, Sorting

### 9.1 Pagination Strategies

#### Offset-based (Page Numbers)
```http
GET /users?page=2&per_page=20

Response:
{
  "data": [...],
  "meta": {
    "current_page": 2,
    "total_pages": 50,
    "total_count": 1000,
    "per_page": 20
  },
  "links": {
    "first": "/users?page=1&per_page=20",
    "prev": "/users?page=1&per_page=20",
    "next": "/users?page=3&per_page=20",
    "last": "/users?page=50&per_page=20"
  }
}
```

**Problem**: If items are added/deleted between pages, results shift.

#### Cursor-based (Keyset Pagination)
```http
GET /users?cursor=dXNlcjoxMjM=&limit=20

Response:
{
  "data": [...],
  "paging": {
    "cursors": {
      "before": "dXNlcjoxMjM=",
      "after": "dXNlcjoyMzQ="
    },
    "next": "/users?cursor=dXNlcjoyMzQ=&limit=20"
  }
}
```

**Cursor** = Base64-encoded `user_id:123`.

**Benefit**: Consistent results even as data changes.

### 9.2 Filtering

```http
GET /products?category=electronics&price_min=100&price_max=500&in_stock=true
```

**Complex filtering (JSON API style)**:
```http
GET /products?filter[category]=electronics&filter[price][gte]=100&filter[price][lte]=500
```

### 9.3 Sorting

```http
GET /users?sort=created_at:desc,name:asc
```

**SQL Translation**:
```sql
ORDER BY created_at DESC, name ASC
```

### 9.4 Partial Responses (Field Selection)

**Problem**: Mobile apps don't need all fields.

```http
GET /users/123?fields=id,name,email

Response:
{"id": 123, "name": "Alice", "email": "alice@example.com"}
```

**GraphQL equivalent**:
```graphql
query {
  user(id: 123) {
    id
    name
    email
  }
}
```

---

## 10. API Versioning Strategies

### 10.1 Why Version?

**Breaking changes**:
- Rename field: `first_name` → `given_name`
- Change data type: `age` (number) → `birthdate` (date)
- Remove endpoint: `DELETE /legacy-feature`

**Goal**: Don't break existing clients.

### 10.2 Versioning Strategies

#### Strategy 1: URL Versioning
```
https://api.example.com/v1/users
https://api.example.com/v2/users
```

**Pros**: Explicit, easy to test  
**Cons**: Pollutes URL space

**Who uses it**: Stripe, Twitter, GitHub.

#### Strategy 2: Header Versioning
```http
GET /users
Accept: application/vnd.example.v2+json

→ Version 2 response
```

**Pros**: Clean URLs  
**Cons**: Harder to test (can't just paste URL in browser)

**Who uses it**: GitHub (supports both).

#### Strategy 3: Query Parameter
```
GET /users?api_version=2
```

**Pros**: Easy to test  
**Cons**: Ugly, easy to forget

**Rarely used**.

### 10.3 Deprecation Strategy

1. **Announce deprecation** (6 months notice)
```http
GET /v1/legacy-endpoint
Sunset: Wed, 01 Jan 2025 00:00:00 GMT
Warning: "This endpoint is deprecated. Use /v2/new-endpoint"
```

2. **Track usage**: Log which clients still use old version

3. **Notify users**: Email API key owners

4. **Shutdown**: Remove old version

### 10.4 Semantic Versioning for APIs

```
v2.3.1
│ │ │
│ │ └─ Patch: Bug fixes
│ └─── Minor: New features (backward compatible)
└───── Major: Breaking changes
```

---

## 11. Best Practices & Anti-Patterns

### 11.1 Best Practices

#### ✅ Use Consistent Naming
```
GET /users (not /user, /Users, /get-users)
GET /posts (not /post, /articles)
```

#### ✅ Return Created Resource
```http
POST /users
{"name": "Alice"}

→ 201 Created
Location: /users/456
{"id": 456, "name": "Alice", "created_at": "2024-01-01T00:00:00Z"}
```

#### ✅ Support ETag for Conditional Requests
```http
GET /users/123
→ ETag: "abc123"

# Later:
PUT /users/123
If-Match: "abc123"
{"name": "Updated"}

→ 412 Precondition Failed (if ETag changed)
```

#### ✅ Use HTTP Caching
```http
GET /public-profiles/123
→ Cache-Control: public, max-age=3600
```

### 11.2 Anti-Patterns

#### ❌ Verbs in URLs
```
❌ POST /users/delete
✅ DELETE /users/123
```

#### ❌ Exposing Database IDs
```
❌ GET /users/12345 (sequential DB ID)
✅ GET /users/u_abc123xyz (UUID or hashed ID)
```

**Why**: Avoids enumeration attacks, hides user count.

#### ❌ Nested Routes Too Deep
```
❌ /orgs/1/teams/2/projects/3/tasks/4
✅ /tasks/4?project_id=3
```

#### ❌ Using GET for State Changes
```
❌ GET /users/123/activate
✅ POST /users/123/activate
```

**Why**: GET is safe (no side effects). Crawlers follow GET links.

#### ❌ Not Handling CORS
```javascript
fetch('https://api.example.com/users')
→ CORS error (if API doesn't set Access-Control-Allow-Origin)
```

**Fix**:
```http
HTTP/1.1 200 OK
Access-Control-Allow-Origin: https://app.example.com
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
```

---

## 12. Summary

### 12.1 Key Takeaways

1. ✅ **Resources, not actions** - `/users`, not `/getUsers`
2. ✅ **HTTP methods have meaning** - Use GET/POST/PUT/PATCH/DELETE correctly
3. ✅ **Design for statelessness** - JWT > sessions for scalability
4. ✅ **Consistent error format** - RFC 7807 Problem Details
5. ✅ **Pagination is mandatory** - Never return unbounded arrays
6. ✅ **Version your APIs** - URL versioning is simplest
7. ✅ **HATEOAS is optional** - Nice to have, rarely implemented

### 12.2 REST vs GraphQL vs gRPC

| Feature | REST | GraphQL | gRPC |
|:--------|:-----|:--------|:-----|
| Protocol | HTTP/1.1+ | HTTP/1.1+ | HTTP/2 |
| Data Format | JSON | JSON | Protobuf (binary) |
| Overfetching | Yes | No | No |
| Type Safety | No (unless OpenAPI) | Yes (schema) | Yes (proto) |
| Caching | HTTP cache | Complex | No built-in |
| Best For | Public APIs | Complex UIs | Internal services |

**Tomorrow (Day 5)**: We dive into **Database Fundamentals** - SQL vs NoSQL, ACID, normalization, and choosing the right database for your use case.

See you tomorrow! 🚀

---

**File Statistics**: ~1050 lines | Comprehensive REST mastery ✅
