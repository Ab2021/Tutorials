# Day 21: REST API Best Practices - Building Production-Grade APIs

## Table of Contents
1. [Richardson Maturity Model](#1-richardson-maturity-model)
2. [Resource Modeling](#2-resource-modeling)
3. [HTTP Methods Deep Dive](#3-http-methods-deep-dive)
4. [Status Codes Mastery](#4-status-codes-mastery)
5. [Pagination Strategies](#5-pagination-strategies)
6. [Filtering & Sorting](#6-filtering--sorting)
7. [Error Handling](#7-error-handling)
8. [HATEOAS](#8-hateoas)
9. [API Security](#9-api-security)
10. [Summary](#10-summary)

---

## 1. Richardson Maturity Model

### 1.1 Level 0: The Swamp of POX

**Plain Old XML** (or JSON):
```http
POST /api HTTP/1.1

{
  "action": "getUser",
  "userId": 123
}

Response:
{
  "user": {"id": 123, "name": "Alice"}
}
```

**Problem**: Single endpoint, everything is POST.

### 1.2 Level 1: Resources

**Introduce resource URLs**:
```http
GET /api/users/123
POST /api/users
DELETE /api/users/123
```

**Better**: Different URLs for different resources.

### 1.3 Level 2: HTTP Verbs

**Use correct HTTP methods**:
```
GET /users/123       - Read
POST /users          - Create
PUT /users/123       - Replace
PATCH /users/123     - Update
DELETE /users/123    - Delete
```

### 1.4 Level 3: Hypermedia Controls (HATEOAS)

**Include links in responses**:
```json
{
  "id": 123,
  "name": "Alice",
  "_links": {
    "self": {"href": "/users/123"},
    "orders": {"href": "/users/123/orders"},
    "edit": {"href": "/users/123", "method": "PATCH"}
  }
}
```

**Benefit**: Client discovers available actions dynamically.

---

## 2. Resource Modeling

### 2.1 RESTful Resource Hierarchy

```
/users                  - Collection
/users/123              - Individual user
/users/123/orders       - User's orders (sub-collection)
/users/123/orders/456   - Specific order
```

### 2.2 Nouns, Not Verbs

❌ **Bad**:
```
POST /createUser
POST /deleteUser/123
GET /getOrders
```

✅ **Good**:
```
POST /users
DELETE /users/123
GET /orders
```

### 2.3 Singular vs Plural

✅ **Use plural** (consistent):
```
/users
/orders
/products
```

❌ **Avoid mixing**:
```
/user
/orders
/product
```

### 2.4 Complex Relationships

**Option 1: Nested resources**
```
GET /users/123/orders
```

**Option 2: Query parameters**
```
GET /orders?userId=123
```

**Choose based on**: Nested if strong ownership, query params if filtering.

---

## 3. HTTP Methods Deep Dive

### 3.1 GET (Safe & Idempotent)

**Safe**: No side effects
**Idempotent**: Multiple calls = same result

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### 3.2 POST (Not Idempotent)

**Create new resource**:
```python
@app.post("/users", status_code=201)
def create_user(user: UserCreate):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {
        **db_user.dict(),
        "_links": {"self": f"/users/{db_user.id}"}
    }
```

**Return**: `201 Created` + `Location` header

```http
HTTP/1.1 201 Created
Location: /users/123
```

### 3.3 PUT (Idempotent, Full Replace)

```python
@app.put("/users/{user_id}")
def replace_user(user_id: int, user: UserUpdate):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404)
    
    # Replace ALL fields
    for field, value in user.dict().items():
        setattr(db_user, field, value)
    
    db.commit()
    return db_user
```

**Key**: Must provide ALL fields (replaces entire resource).

### 3.4 PATCH (Idempotent, Partial Update)

```python
@app.patch("/users/{user_id}")
def update_user(user_id: int, user: UserPartialUpdate):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404)
    
    # Update ONLY provided fields
    for field, value in user.dict(exclude_unset=True).items():
        setattr(db_user, field, value)
    
    db.commit()
    return db_user
```

**Request**:
```json
PATCH /users/123
{
  "email": "newemail@example.com"
}
```

**Only email updated**, other fields unchanged.

### 3.5 DELETE (Idempotent)

```python
@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404)
    
    db.delete(db_user)
    db.commit()
    return  # 204 No Content (empty body)
```

**Idempotent**: Deleting twice → same result (resource gone).

---

## 4. Status Codes Mastery

### 4.1 2xx Success

- **200 OK**: GET, PATCH, PUT success
- **201 Created**: POST success (new resource)
- **202 Accepted**: Async processing started
- **204 No Content**: DELETE success (no body)

### 4.2 3xx Redirection

- **301 Moved Permanently**: Resource moved, update bookmarks
- **302 Found**: Temporary redirect
- **304 Not Modified**: Cached version still valid

### 4.3 4xx Client Errors

- **400 Bad Request**: Invalid syntax/validation error
- **401 Unauthorized**: Not authenticated (missing/invalid token)
- **403 Forbidden**: Authenticated but not authorized
- **404 Not Found**: Resource doesn't exist
- **409 Conflict**: Resource state conflict
- **422 Unprocessable Entity**: Validation error
- **429 Too Many Requests**: Rate limited

### 4.4 5xx Server Errors

- **500 Internal Server Error**: Generic server error
- **502 Bad Gateway**: Upstream server error
- **503 Service Unavailable**: Server overloaded/maintenance
- **504 Gateway Timeout**: Upstream timeout

### 4.5 Correct Usage

✅ **Good**:
```python
# Missing auth token
return 401

# Valid token, but user lacks permission
return 403

# Invalid email format
return 422
```

❌ **Bad**:
```python
# Using 200 for errors
return {"error": "User not found"}, 200  # NO!

# Using 500 for client errors
return 500  # For validation error (should be 422)
```

---

## 5. Pagination Strategies

### 5.1 Offset-Based Pagination

```python
@app.get("/users")
def list_users(skip: int = 0, limit: int = 20):
    users = db.query(User).offset(skip).limit(limit).all()
    total = db.query(User).count()
    
    return {
        "data": users,
        "pagination": {
            "skip": skip,
            "limit": limit,
            "total": total
        }
    }
```

**Request**:
```
GET /users?skip=0&limit=20   # Page 1
GET /users?skip=20&limit=20  # Page 2
```

**Pros**: Simple, can jump to any page
**Cons**: Slow for large offsets, misses/duplicates if data changes

### 5.2 Cursor-Based Pagination

```python
@app.get("/users")
def list_users(cursor: Optional[str] = None, limit: int = 20):
    query = db.query(User).order_by(User.id)
    
    if cursor:
        # Decode cursor (base64 encoded last ID)
        last_id = int(base64.b64decode(cursor))
        query = query.filter(User.id > last_id)
    
    users = query.limit(limit + 1).all()
    
    has_more = len(users) > limit
    users = users[:limit]
    
    next_cursor = None
    if has_more:
        next_cursor = base64.b64encode(str(users[-1].id).encode()).decode()
    
    return {
        "data": users,
        "pagination": {
            "next_cursor": next_cursor,
            "has_more": has_more
        }
    }
```

**Request**:
```
GET /users?limit=20                      # Page 1
GET /users?cursor=abc123&limit=20       # Page 2
```

**Pros**: Fast, no duplicates/misses
**Cons**: Can't jump to arbitrary page

### 5.3 Link Headers (GitHub Style)

```python
from urllib.parse import urlencode

@app.get("/users")
def list_users(page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    users = db.query(User).offset(offset).limit(per_page).all()
    total = db.query(User).count()
    total_pages = (total + per_page - 1) // per_page
    
    # Build Link header
    links = []
    if page < total_pages:
        links.append(f'</users?{urlencode({"page": page + 1, "per_page": per_page})}>; rel="next"')
    if page > 1:
        links.append(f'</users?{urlencode({"page": page - 1, "per_page": per_page})}>; rel="prev"')
    links.append(f'</users?{urlencode({"page": 1, "per_page": per_page})}>; rel="first"')
    links.append(f'</users?{urlencode({"page": total_pages, "per_page": per_page})}>; rel="last"')
    
    headers = {"Link": ", ".join(links)}
    
    return Response(content=json.dumps({"data": users}), headers=headers)
```

---

## 6. Filtering & Sorting

### 6.1 Filtering

```python
@app.get("/users")
def list_users(
    email: Optional[str] = None,
    status: Optional[str] = None,
    created_after: Optional[datetime] = None
):
    query = db.query(User)
    
    if email:
        query = query.filter(User.email.contains(email))
    if status:
        query = query.filter(User.status == status)
    if created_after:
        query = query.filter(User.created_at > created_after)
    
    return query.all()
```

**Request**:
```
GET /users?status=active&created_after=2024-01-01
```

### 6.2 Sorting

```python
@app.get("/users")
def list_users(sort_by: str = "id", order: str = "asc"):
    allowed_fields = ["id", "email", "created_at"]
    
    if sort_by not in allowed_fields:
        raise HTTPException(status_code=400, detail="Invalid sort field")
    
    query = db.query(User)
    
    if order == "desc":
        query = query.order_by(desc(getattr(User, sort_by)))
    else:
        query = query.order_by(asc(getattr(User, sort_by)))
    
    return query.all()
```

**Request**:
```
GET /users?sort_by=created_at&order=desc
```

### 6.3 Advanced: Filter DSL

```python
# URL: /users?filter=status:active,created_after:2024-01-01

@app.get("/users")
def list_users(filter: Optional[str] = None):
    query = db.query(User)
    
    if filter:
        for condition in filter.split(","):
            field, value = condition.split(":")
            
            if field == "status":
                query = query.filter(User.status == value)
            elif field == "created_after":
                query = query.filter(User.created_at > datetime.fromisoformat(value))
    
    return query.all()
```

---

## 7. Error Handling

### 7.1 RFC 7807 Problem Details

```json
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "Email format is invalid",
  "instance": "/users/123",
  "errors": [
    {
      "field": "email",
      "message": "Must be a valid email address"
    }
  ]
}
```

### 7.2 Implementation

```python
from fastapi import HTTPException
from pydantic import BaseModel

class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str
    errors: Optional[List[dict]] = None

@app.exception_handler(ValidationError)
def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ProblemDetail(
            type="https://api.example.com/errors/validation",
            title="Validation Error",
            status=422,
            detail="Request validation failed",
            instance=str(request.url),
            errors=[{"field": err["loc"][-1], "message": err["msg"]} for err in exc.errors()]
        ).dict()
    )
```

### 7.3 Consistent Error Format

✅ **Good** (consistent):
```json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User with ID 123 not found"
  }
}
```

❌ **Bad** (inconsistent):
```json
// Sometimes:
{"error": "Not found"}

// Other times:
{"message": "User doesn't exist", "status": 404}
```

---

## 8. HATEOAS

### 8.1 What is HATEOAS?

**Hypermedia As The Engine Of Application State**

**Goal**: Client doesn't hardcode URLs, discovers them dynamically.

### 8.2 Example

```json
GET /users/123

{
  "id": 123,
  "name": "Alice",
  "email": "alice@example.com",
  "_links": {
    "self": {
      "href": "/users/123"
    },
    "orders": {
      "href": "/users/123/orders"
    },
    "edit": {
      "href": "/users/123",
      "method": "PATCH"
    },
    "delete": {
      "href": "/users/123",
      "method": "DELETE"
    }
  }
}
```

**Client code**:
```python
user = requests.get("/users/123").json()

# Don't hardcode URL
orders_url = user["_links"]["orders"]["href"]
orders = requests.get(orders_url).json()
```

### 8.3 HAL (Hypertext Application Language)

```json
{
  "_links": {
    "self": {"href": "/orders/456"},
    "customer": {"href": "/users/123"}
  },
  "id": 456,
  "total": 99.99,
  "status": "pending",
  "_embedded": {
    "items": [
      {"id": 1, "name": "Product A", "price": 49.99},
      {"id": 2, "name": "Product B", "price": 50.00}
    ]
  }
}
```

---

## 9. API Security

### 9.1 Authentication

**Bearer Token** (JWT):
```http
GET /users/123
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
```

### 9.2 Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/users")
@limiter.limit("100/minute")
def list_users():
    return db.query(User).all()
```

### 9.3 Input Validation

```python
from pydantic import BaseModel, EmailStr, constr

class UserCreate(BaseModel):
    email: EmailStr
    password: constr(min_length=8, max_length=100)
    name: constr(min_length=1, max_length=100)

@app.post("/users")
def create_user(user: UserCreate):
    # user is already validated by Pydantic
    ...
```

### 9.4 SQL Injection Prevention

✅ **Good** (parameterized):
```python
user = db.query(User).filter(User.email == email).first()
```

❌ **Bad** (vulnerable):
```python
query = f"SELECT * FROM users WHERE email = '{email}'"
db.execute(query)
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **Richardson Level 2+** - Use proper HTTP verbs
2. ✅ **Resource URLs** - Nouns, plural, hierarchical
3. ✅ **Idempotency** - GET, PUT, DELETE, PATCH
4. ✅ **Status Codes** - 2xx success, 4xx client, 5xx server
5. ✅ **Cursor Pagination** - Better than offset for large datasets
6. ✅ **Filtering & Sorting** - Query parameters
7. ✅ **RFC 7807** - Consistent error format
8. ✅ **HATEOAS** - Discoverability via links

### 10.2 REST API Checklist

- [ ] Use plural nouns (`/users`, not `/user`)
- [ ] Correct HTTP methods (GET read, POST create, etc.)
- [ ] Proper status codes (404 not found, 422 validation)
- [ ] Pagination (cursor or offset)
- [ ] Filtering & sorting support
- [ ] Consistent error format (RFC 7807)
- [ ] Rate limiting
- [ ] Input validation (Pydantic)
- [ ] Authentication (JWT)
- [ ] Documentation (OpenAPI/Swagger)

### 10.3 Tomorrow (Day 22): API Versioning Strategies

- **URL versioning**: `/v1/users`, `/v2/users`
- **Header versioning**: `Accept: application/vnd.api.v2+json`
- **Query parameter**: `/users?version=2`
- **Content negotiation**: `Accept` header
- **Deprecation strategy**: Sunset header
- **Breaking vs non-breaking changes**

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | REST API Best Practices mastered ✅
