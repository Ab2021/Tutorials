# Day 25: API Documentation - OpenAPI, Swagger & Beyond

## Table of Contents
1. [Why Documentation Matters](#1-why-documentation-matters)
2. [OpenAPI Specification](#2-openapi-specification)
3. [Swagger Tools](#3-swagger-tools)
4. [FastAPI Auto-Documentation](#4-fastapi-auto-documentation)
5. [Postman Collections](#5-postman-collections)
6. [API Blueprint](#6-api-blueprint)
7. [AsyncAPI](#7-asyncapi)
8. [Documentation Best Practices](#8-documentation-best-practices)
9. [Interactive Documentation](#9-interactive-documentation)
10. [Summary](#10-summary)

---

## 1. Why Documentation Matters

### 1.1 The Problem

**Undocumented API**:
```
Developer: "How do I create a user?"
Support: "Check the code... maybe POST to /users?"
Developer: "What fields are required?"
Support: "¯\_(ツ)_/¯"
```

**Result**: Slow onboarding, support overhead, frustrated developers.

### 1.2 Good Documentation Benefits

✅ **Self-service**: Developers answer own questions  
✅ **Faster integration**: Clear examples  
✅ **Fewer support tickets**: Complete reference  
✅ **Better adoption**: Easy to get started

---

## 2. OpenAPI Specification

### 2.1 What is OpenAPI?

**OpenAPI** (formerly Swagger): Standard for describing REST APIs.

**Format**: YAML or JSON

### 2.2 Basic OpenAPI Document

```yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: API for managing users

servers:
  - url: https://api.example.com/v1

paths:
  /users:
    get:
      summary: List all users
      operationId: listUsers
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/User'
    post:
      summary: Create a user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          description: Invalid input

  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found

components:
  schemas:
    User:
      type: object
      required:
        - id
        - name
        - email
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
          format: email
        createdAt:
          type: string
          format: date-time
    
    CreateUserRequest:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        password:
          type: string
          format: password
          minLength: 8

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - BearerAuth: []
```

### 2.3 Example Responses

```yaml
paths:
  /users/{id}:
    get:
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
              examples:
                success:
                  value:
                    id: 123
                    name: "Alice"
                    email: "alice@example.com"
                    createdAt: "2024-01-01T00:00:00Z"
```

---

## 3. Swagger Tools

### 3.1 Swagger UI

**Interactive documentation** from OpenAPI spec.

```html
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({
      url: '/openapi.yaml',
      dom_id: '#swagger-ui'
    })
  </script>
</body>
</html>
```

**Features**:
- Read API documentation
- Try API calls (interactive)
- See response examples

### 3.2 Swagger Editor

**Online editor**: https://editor.swagger.io

- Write OpenAPI spec
- Real-time validation
- Preview Swagger UI

### 3.3 Swagger Codegen

**Generate client libraries**:

```bash
# Generate Python client
swagger-codegen generate \
  -i openapi.yaml \
  -l python \
  -o client/

# Usage
from client import UsersApi

api = UsersApi()
users = api.list_users(limit=10)
```

---

## 4. FastAPI Auto-Documentation

### 4.1 Automatic OpenAPI Generation

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel, EmailStr

app = FastAPI(
    title="User API",
    description="API for managing users",
    version="1.0.0"
)

class User(BaseModel):
    id: int
    name: str
    email: EmailStr

class CreateUserRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

@app.get(
    "/users",
    response_model=list[User],
    summary="List all users",
    description="Retrieve a paginated list of users",
    tags=["users"]
)
def list_users(
    limit: int = Query(20, description="Maximum number of users to return", ge=1, le=100)
):
    """
    List all users with pagination.
    
    - **limit**: Number of users to return (1-100)
    
    Returns a list of user objects.
    """
    return db.query(User).limit(limit).all()

@app.post(
    "/users",
    response_model=User,
    status_code=201,
    tags=["users"],
    responses={
        201: {"description": "User successfully created"},
        400: {"description": "Invalid input"}
    }
)
def create_user(user: CreateUserRequest):
    """
    Create a new user.
    
    Example request body:
    ```json
    {
      "name": "Alice",
      "email": "alice@example.com",
      "password": "secret123"
    }
    ```
    """
    new_user = User(id=generate_id(), name=user.name, email=user.email)
    db.add(new_user)
    db.commit()
    return new_user
```

**Access docs**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### 4.2 Custom OpenAPI Schema

```python
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="User API",
        version="1.0.0",
        description="API for managing users",
        routes=app.routes,
    )
    
    # Add custom fields
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

## 5. Postman Collections

### 5.1 Creating a Collection

```json
{
  "info": {
    "name": "User API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Users",
      "item": [
        {
          "name": "List Users",
          "request": {
            "method": "GET",
            "header": [],
            "url": {
              "raw": "{{baseUrl}}/users?limit=20",
              "host": ["{{baseUrl}}"],
              "path": ["users"],
              "query": [
                {
                  "key": "limit",
                  "value": "20"
                }
              ]
            }
          },
          "response": []
        },
        {
          "name": "Create User",
          "request": {
            "method": "POST",
            "header": [
              {
                "key": "Content-Type",
                "value": "application/json"
              }
            ],
            "body": {
              "mode": "raw",
              "raw": "{\n  \"name\": \"Alice\",\n  \"email\": \"alice@example.com\",\n  \"password\": \"secret123\"\n}"
            },
            "url": {
              "raw": "{{baseUrl}}/users",
              "host": ["{{baseUrl}}"],
              "path": ["users"]
            }
          }
        }
      ]
    }
  ],
  "variable": [
    {
      "key": "baseUrl",
      "value": "https://api.example.com/v1"
    }
  ]
}
```

### 5.2 Environment Variables

```json
{
  "name": "Production",
  "values": [
    {
      "key": "baseUrl",
      "value": "https://api.example.com/v1"
    },
    {
      "key": "authToken",
      "value": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    }
  ]
}
```

### 5.3 Import from OpenAPI

**In Postman**:
1. Import → Link
2. Paste OpenAPI URL: `https://api.example.com/openapi.json`
3. Auto-generates collection!

---

## 6. API Blueprint

### 6.1 Markdown-Based Documentation

```markdown
FORMAT: 1A

# User API

API for managing users.

# Group Users

## Users Collection [/users]

### List All Users [GET /users{?limit}]

+ Parameters
    + limit (number, optional) - Maximum number of users
        + Default: 20

+ Response 200 (application/json)
    + Attributes (array[User])

+ Response 401 (application/json)
    + Attributes (Error)

### Create a User [POST]

+ Request (application/json)
    + Attributes (CreateUserRequest)

+ Response 201 (application/json)
    + Attributes (User)

## User [/users/{id}]

+ Parameters
    + id (number) - User ID

### Get User [GET]

+ Response 200 (application/json)
    + Attributes (User)

+ Response 404 (application/json)
    + Attributes (Error)

# Data Structures

## User (object)
+ id: 123 (number, required)
+ name: Alice (string, required)
+ email: alice@example.com (string, required)
+ createdAt: `2024-01-01T00:00:00Z` (string)

## CreateUserRequest (object)
+ name: Alice (string, required)
+ email: alice@example.com (string, required)
+ password: secret123 (string, required)

## Error (object)
+ error: `USER_NOT_FOUND` (string)
+ message: User with ID 123 not found (string)
```

**Render**:
```bash
aglio -i api.apib -o api.html
```

---

## 7. AsyncAPI

### 7.1 Event-Driven API Documentation

```yaml
asyncapi: 2.6.0
info:
  title: Order Events API
  version: 1.0.0
  description: Kafka events for order processing

servers:
  production:
    url: kafka://kafka.example.com:9092
    protocol: kafka

channels:
  orders.created:
    description: Order creation events
    subscribe:
      summary: Listen for new orders
      message:
        $ref: '#/components/messages/OrderCreated'
  
  orders.completed:
    description: Order completion events
    publish:
      summary: Publish order completion
      message:
        $ref: '#/components/messages/OrderCompleted'

components:
  messages:
    OrderCreated:
      payload:
        type: object
        properties:
          orderId:
            type: string
          userId:
            type: string
          total:
            type: number
          items:
            type: array
            items:
              type: object
              properties:
                productId:
                  type: string
                quantity:
                  type: integer
    
    OrderCompleted:
      payload:
        type: object
        properties:
          orderId:
            type: string
          completedAt:
            type: string
            format: date-time
```

---

## 8. Documentation Best Practices

### 8.1 Include Examples

✅ **Good**:
```yaml
examples:
  success:
    value:
      id: 123
      name: "Alice"
      email: "alice@example.com"
  
  error:
    value:
      error: "USER_NOT_FOUND"
      message: "User with ID 123 not found"
```

❌ **Bad**: No examples, just schema.

### 8.2 Document Error Responses

```yaml
responses:
  '400':
    description: Invalid input
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/ValidationError'
        example:
          error: "VALIDATION_ERROR"
          details:
            - field: "email"
              message: "Invalid email format"
  
  '401':
    description: Not authenticated
  
  '403':
    description: Not authorized
  
  '404':
    description: User not found
  
  '500':
    description: Internal server error
```

### 8.3 Versioned Documentation

**Separate docs per version**:
```
/docs/v1  → v1 documentation
/docs/v2  → v2 documentation
```

### 8.4 Changelog

```markdown
# Changelog

## v2.0.0 (2024-01-01)

**BREAKING CHANGES**:
- `name` field split into `firstName` and `lastName`

**New Features**:
- Added `/users/search` endpoint

## v1.1.0 (2023-06-01)

**New Features**:
- Added `email` field to User
```

---

## 9. Interactive Documentation

### 9.1 Try-It-Out Features

**Swagger UI** allows testing APIs directly:
```
1. Click "Try it out"
2. Fill in parameters
3. Click "Execute"
4. See response
```

### 9.2 Code Samples

```yaml
x-code-samples:
  - lang: Python
    source: |
      import requests
      
      response = requests.get(
          'https://api.example.com/v1/users/123',
          headers={'Authorization': 'Bearer TOKEN'}
      )
      print(response.json())
  
  - lang: curl
    source: |
      curl -X GET \
        'https://api.example.com/v1/users/123' \
        -H 'Authorization: Bearer TOKEN'
  
  - lang: JavaScript
    source: |
      fetch('https://api.example.com/v1/users/123', {
        headers: {
          'Authorization': 'Bearer TOKEN'
        }
      })
      .then(res => res.json())
      .then(data => console.log(data))
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **OpenAPI** - Standard for REST API docs
2. ✅ **Swagger UI** - Interactive documentation
3. ✅ **FastAPI** - Auto-generates OpenAPI
4. ✅ **Postman** - Shareable collections
5. ✅ **API Blueprint** - Markdown-based docs
6. ✅ **AsyncAPI** - Event-driven API docs
7. ✅ **Examples** - Include request/response samples

### 10.2 Documentation Checklist

- [ ] OpenAPI spec (YAML/JSON)
- [ ] Interactive UI (Swagger/ReDoc)
- [ ] Request/response examples
- [ ] Error responses documented
- [ ] Authentication documented
- [ ] Rate limiting documented
- [ ] Versioned docs
- [ ] Changelog
- [ ] Code samples (multiple languages)
- [ ] Try-it-out functionality

### 10.3 Tomorrow (Day 26): Advanced Caching Strategies

- **Caching layers**: Client, CDN, server, database
- **Redis patterns**: Cache-aside, write-through, write-behind
- **Cache invalidation**: TTL, manual, event-driven
- **Distributed caching**: Consistency, partitioning
- **HTTP caching**: ETag, Last-Modified, Cache-Control
- **Production patterns**: Cache warming, stampede prevention

See you tomorrow! 🚀

---

**File Statistics**: ~1000 lines | API Documentation mastered ✅
