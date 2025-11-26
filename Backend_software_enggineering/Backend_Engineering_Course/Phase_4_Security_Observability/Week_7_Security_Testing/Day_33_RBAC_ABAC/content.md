# Day 33: RBAC & ABAC - Advanced Authorization

## Table of Contents
1. [Authorization Fundamentals](#1-authorization-fundamentals)
2. [RBAC (Role-Based Access Control)](#2-rbac-role-based-access-control)
3. [ABAC (Attribute-Based Access Control)](#3-abac-attribute-based-access-control)
4. [Policy Engines](#4-policy-engines)
5. [Permission Models](#5-permission-models)
6. [Casbin Deep Dive](#6-casbin-deep-dive)
7. [Open Policy Agent (OPA)](#7-open-policy-agent-opa)
8. [Production Patterns](#8-production-patterns)
9. [Audit Logging](#9-audit-logging)
10. [Summary](#10-summary)

---

## 1. Authorization Fundamentals

### 1.1 Authentication vs Authorization

**Authentication**: Who are you?
```python
# User proves identity (login)
user = authenticate(username, password)
```

**Authorization**: What can you do?
```python
# Check if user has permission
if user.has_permission('delete_post'):
    delete_post()
```

### 1.2 Access Control Models

1. **DAC** (Discretionary): Resource owner decides access
2. **MAC** (Mandatory): System enforces policies (military)
3. **RBAC** (Role-Based): Permissions assigned to roles
4. **ABAC** (Attribute-Based): Context-aware decisions

---

## 2. RBAC (Role-Based Access Control)

### 2.1 Basic RBAC

```python
# Database schema
class User:
    id: int
    role: str  # 'admin', 'editor', 'viewer'

class Permission:
    role: str
    resource: str  # 'posts', 'users'
    action: str    # 'create', 'read', 'update', 'delete'

# Permissions table
permissions = [
    {'role': 'admin', 'resource': '*', 'action': '*'},
    {'role': 'editor', 'resource': 'posts', 'action': 'create'},
    {'role': 'editor', 'resource': 'posts', 'action': 'update'},
    {'role': 'viewer', 'resource': 'posts', 'action': 'read'}
]

def has_permission(user, resource, action):
    for perm in permissions:
        if perm['role'] == user.role:
            if (perm['resource'] == '*' or perm['resource'] == resource) and \
               (perm['action'] == '*' or perm['action'] == action):
                return True
    return False

# Usage
@app.delete("/posts/{post_id}")
def delete_post(post_id: int, current_user: User):
    if not has_permission(current_user, 'posts', 'delete'):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Delete post
```

### 2.2 Hierarchical Roles

```python
role_hierarchy = {
    'admin': ['editor', 'viewer'],
    'editor': ['viewer'],
    'viewer': []
}

def get_all_roles(user_role):
    """Get role + all inherited roles"""
    roles = {user_role}
    for inherited in role_hierarchy.get(user_role, []):
        roles.update(get_all_roles(inherited))
    return roles

def has_permission(user, resource, action):
    user_roles = get_all_roles(user.role)
    
    for perm in permissions:
        if perm['role'] in user_roles:
            if (perm['resource'] == '*' or perm['resource'] == resource) and \
               (perm['action'] == '*' or perm['action'] == action):
                return True
    return False

# Admin inherits editor & viewer permissions automatically
```

### 2.3 Multiple Roles per User

```python
class UserRole:
    user_id: int
    role: str

# User can have multiple roles
user_roles = db.query(UserRole).filter(UserRole.user_id == user.id).all()

def has_permission(user, resource, action):
    for user_role in user_roles:
        all_roles = get_all_roles(user_role.role)
        
        for perm in permissions:
            if perm['role'] in all_roles:
                if matches(perm, resource, action):
                    return True
    return False
```

---

## 3. ABAC (Attribute-Based Access Control)

### 3.1 What is ABAC?

**ABAC**: Decisions based on attributes (user, resource, environment).

**Example**:
```
User can edit document IF:
- User is document owner OR
- User is in document's shared_with list OR
- User is admin AND
- Document is not locked AND
- Current time is within business hours
```

### 3.2 ABAC Implementation

```python
def can_edit_document(user, document, context):
    # User attributes
    if user.id == document.owner_id:
        return True  # Owner can always edit
    
    if user.role == 'admin' and not document.is_locked:
        return True  # Admin can edit unlocked docs
    
    # Resource attributes
    if user.id in document.shared_with and document.editable:
        return True  # Shared users can edit if document is editable
    
    # Environment attributes
    current_hour = context['current_time'].hour
    if current_hour < 9 or current_hour > 17:
        return False  # No editing outside business hours
    
    return False

# Usage
@app.patch("/documents/{doc_id}")
def update_document(doc_id: int, current_user: User):
    document = db.query(Document).filter(Document.id == doc_id).first()
    context = {'current_time': datetime.now()}
    
    if not can_edit_document(current_user, document, context):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Update document
```

---

## 4. Policy Engines

### 4.1 Why Policy Engines?

**Problem**: Authorization logic scattered across codebase.

**Solution**: Centralize policies in policy engine.

**Benefits**:
- ✅ Single source of truth
- ✅ Easy to audit
- ✅ Change policies without code changes

---

## 5. Permission Models

### 5.1 ACL (Access Control List)

```python
# Per-resource permissions
class DocumentACL:
    document_id: int
    user_id: int
    permissions: List[str]  # ['read', 'write', 'delete']

# Check permission
acl = db.query(DocumentACL).filter(
    DocumentACL.document_id == doc_id,
    DocumentACL.user_id == user.id
).first()

if 'write' not in acl.permissions:
    raise HTTPException(status_code=403)
```

### 5.2 ReBAC (Relationship-Based)

```python
# Google Docs-style permissions
class DocumentRelationship:
    document_id: int
    user_id: int
    relationship: str  # 'owner', 'editor', 'viewer'

def can_edit(user, document):
    rel = db.query(DocumentRelationship).filter(
        DocumentRelationship.document_id == document.id,
        DocumentRelationship.user_id == user.id
    ).first()
    
    return rel and rel.relationship in ['owner', 'editor']
```

---

## 6. Casbin Deep Dive

### 6.1 Installation & Setup

```bash
pip install casbin
```

**Model file** (`model.conf`):
```ini
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
```

**Policy file** (`policy.csv`):
```csv
p, admin, *, *
p, editor, posts, write
p, viewer, posts, read

g, alice, admin
g, bob, editor
g, charlie, viewer
```

### 6.2 Using Casbin

```python
import casbin

enforcer = casbin.Enforcer('model.conf', 'policy.csv')

# Check permission
if enforcer.enforce('alice', 'posts', 'delete'):
    # Alice (admin) can delete posts
    delete_post()

if enforcer.enforce('bob', 'posts', 'write'):
    # Bob (editor) can write posts
    create_post()

if not enforcer.enforce('charlie', 'posts', 'write'):
    # Charlie (viewer) cannot write posts
    raise HTTPException(status_code=403)
```

### 6.3 Dynamic Policies

```python
# Add policy at runtime
enforcer.add_policy('bob', 'users', 'read')

# Remove policy
enforcer.remove_policy('bob', 'users', 'read')

# Add role
enforcer.add_role_for_user('dave', 'editor')

# Get roles
roles = enforcer.get_roles_for_user('alice')  # ['admin']
```

### 6.4 ABAC with Casbin

**Model** (`abac_model.conf`):
```ini
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub_rule, obj_rule, act

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = eval(p.sub_rule) && eval(p.obj_rule) && r.act == p.act
```

**Policy**:
```csv
p, r.sub.role == "admin", r.obj.owner == r.sub.id, write
p, r.sub.role == "editor", r.obj.published == false, write
```

**Usage**:
```python
# Pass attributes
sub = {'id': 123, 'role': 'admin'}
obj = {'id': 456, 'owner': 123, 'published': True}

if enforcer.enforce(sub, obj, 'write'):
    # Admin can write (owner check passes)
    update_document()
```

---

## 7. Open Policy Agent (OPA)

### 7.1 What is OPA?

**OPA**: Policy engine using Rego language.

**Use cases**:
- API authorization
- Kubernetes admission control
- Microservice policies

### 7.2 Rego Policy

**Policy file** (`policy.rego`):
```rego
package authz

default allow = false

# Admins can do anything
allow {
    input.user.role == "admin"
}

# Users can edit their own posts
allow {
    input.method == "PATCH"
    input.resource.type == "post"
    input.resource.owner == input.user.id
}

# Editors can create posts
allow {
    input.method == "POST"
    input.resource.type == "post"
    input.user.role == "editor"
}
```

### 7.3 Using OPA

```python
import requests

def check_permission(user, method, resource):
    # Call OPA API
    response = requests.post('http://localhost:8181/v1/data/authz/allow', json={
        'input': {
            'user': {'id': user.id, 'role': user.role},
            'method': method,
            'resource': resource
        }
    })
    
    return response.json()['result']

# Usage
@app.patch("/posts/{post_id}")
def update_post(post_id: int, current_user: User):
    post = db.query(Post).filter(Post.id == post_id).first()
    
    if not check_permission(
        user=current_user,
        method='PATCH',
        resource={'type': 'post', 'owner': post.owner_id}
    ):
        raise HTTPException(status_code=403)
    
    # Update post
```

---

## 8. Production Patterns

### 8.1 Permission Caching

```python
import redis

r = redis.Redis()

def has_permission_cached(user, resource, action):
    cache_key = f"perm:{user.id}:{resource}:{action}"
    
    # Check cache
    cached = r.get(cache_key)
    if cached is not None:
        return cached == b'1'
    
    # Cache miss → check permission
    allowed = has_permission(user, resource, action)
    
    # Cache for 5 minutes
    r.setex(cache_key, 300, '1' if allowed else '0')
    
    return allowed
```

### 8.2 Permission Preloading

```python
def get_user_permissions(user_id):
    """Load all user permissions at login"""
    user_roles = db.query(UserRole).filter(UserRole.user_id == user_id).all()
    
    all_permissions = []
    for user_role in user_roles:
        perms = db.query(Permission).filter(Permission.role == user_role.role).all()
        all_permissions.extend(perms)
    
    # Store in session/JWT
    return all_permissions

# At login
permissions = get_user_permissions(user.id)
session['permissions'] = permissions

# At authorization
def has_permission(resource, action):
    for perm in session['permissions']:
        if matches(perm, resource, action):
            return True
    return False
```

---

## 9. Audit Logging

### 9.1 Authorization Audit

```python
class AuthorizationAuditLog:
    timestamp: datetime
    user_id: int
    resource: str
    action: str
    allowed: bool
    reason: str  # Why allowed/denied

def audit_authorization(user, resource, action, allowed, reason):
    log = AuthorizationAuditLog(
        timestamp=datetime.utcnow(),
        user_id=user.id,
        resource=resource,
        action=action,
        allowed=allowed,
        reason=reason
    )
    db.add(log)
    db.commit()

# Usage
def check_permission_with_audit(user, resource, action):
    allowed = has_permission(user, resource, action)
    reason = "Has required role" if allowed else "Missing permission"
    
    audit_authorization(user, resource, action, allowed, reason)
    
    return allowed
```

### 9.2 Query Audit Logs

```python
@app.get("/audit/authorization")
def get_authorization_audit(
    user_id: Optional[int] = None,
    resource: Optional[str] = None,
    start_date: Optional[datetime] = None
):
    query = db.query(AuthorizationAuditLog)
    
    if user_id:
        query = query.filter(AuthorizationAuditLog.user_id == user_id)
    if resource:
        query = query.filter(AuthorizationAuditLog.resource == resource)
    if start_date:
        query = query.filter(AuthorizationAuditLog.timestamp >= start_date)
    
    return query.order_by(AuthorizationAuditLog.timestamp.desc()).limit(100).all()
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **RBAC** - Role-based permissions (simple, hierarchical)
2. ✅ **ABAC** - Attribute-based (context-aware)
3. ✅ **Casbin** - Policy engine (model + policy files)
4. ✅ **OPA** - Policy engine (Rego language)
5. ✅ **Permission Caching** - Redis for performance
6. ✅ **Audit Logging** - Track all authz decisions

### 10.2 Model Comparison

| Model | Complexity | Flexibility | Use Case |
|:------|:-----------|:------------|:---------|
| **RBAC** | Low | Low | Simple apps |
| **Hierarchical RBAC** | Medium | Medium | Most apps |
| **ABAC** | High | High | Complex requirements |
| **ReBAC** | Medium | High | Social apps (Google Docs) |

### 10.3 Tomorrow (Day 34): Encryption & TLS

- **Encryption at rest**: AES-256, field-level encryption
- **Encryption in transit**: TLS 1.3, certificate management
- **Key management**: Rotation, HSM
- **Hashing**: bcrypt, Argon2, PBKDF2
- **Digital signatures**: RSA, ECDSA
- **Production patterns**: Key derivation, envelope encryption

See you tomorrow! 🚀

---

**File Statistics**: ~950 lines | RBAC & ABAC Authorization mastered ✅
