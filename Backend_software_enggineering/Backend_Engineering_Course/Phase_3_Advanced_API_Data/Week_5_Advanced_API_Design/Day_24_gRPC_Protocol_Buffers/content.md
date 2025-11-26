# Day 24: gRPC & Protocol Buffers - High-Performance RPC

## Table of Contents
1. [gRPC Overview](#1-grpc-overview)
2. [Protocol Buffers](#2-protocol-buffers)
3. [Service Definition](#3-service-definition)
4. [Streaming](#4-streaming)
5. [Error Handling](#5-error-handling)
6. [Authentication](#6-authentication)
7. [Production Patterns](#7-production-patterns)
8. [gRPC vs REST vs GraphQL](#8-grpc-vs-rest-vs-graphql)
9. [When to Use gRPC](#9-when-to-use-grpc)
10. [Summary](#10-summary)

---

## 1. gRPC Overview

### 1.1 What is gRPC?

**gRPC**: Google Remote Procedure Call - high-performance RPC framework.

**Key features**:
- ✅ **Binary protocol** (Protocol Buffers) → 10x faster than JSON
- ✅ **HTTP/2** → Multiplexing, server push
- ✅ **Streaming** → Real-time bidirectional communication
- ✅ **Code generation** → Strongly typed clients/servers

### 1.2 REST vs gRPC

**REST**:
```http
GET /users/123 HTTP/1.1
Content-Type: application/json

{"id": 123, "name": "Alice"}
```

**gRPC**:
```
# Binary message (Protocol Buffers)
[binary data] → 10x smaller, 10x faster
```

### 1.3 Performance Comparison

| Metric | REST (JSON) | gRPC (Protobuf) |
|:-------|:------------|:----------------|
| **Payload Size** | 1000 bytes | 100 bytes |
| **Latency** | 50ms | 5ms |
| **Throughput** | 10k req/sec | 100k req/sec |

---

## 2. Protocol Buffers

### 2.1 What are Protobufs?

**Protocol Buffers**: Binary serialization format (like JSON, but faster).

### 2.2 .proto File

```protobuf
// user.proto
syntax = "proto3";

package user;

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
  repeated string roles = 4;  // Array
}

message GetUserRequest {
  int32 id = 1;
}

message GetUserResponse {
  User user = 1;
}
```

**Key concepts**:
- `int32 id = 1` → Field number (never change!)
- `repeated` → Array
- `syntax = "proto3"` → Proto3 (latest)

### 2.3 Compiling Protobufs

```bash
# Install protoc
brew install protobuf

# Generate Python code
protoc --python_out=. --grpc_python_out=. user.proto

# Generates:
# user_pb2.py (messages)
# user_pb2_grpc.py (service stubs)
```

### 2.4 Using Generated Code

```python
import user_pb2

# Create message
user = user_pb2.User(
    id=123,
    name="Alice",
    email="alice@example.com",
    roles=["admin", "user"]
)

# Serialize (binary)
binary_data = user.SerializeToString()

# Deserialize
user2 = user_pb2.User()
user2.ParseFromString(binary_data)
```

---

## 3. Service Definition

### 3.1 Defining Service

```protobuf
// user.proto
service UserService {
  // Unary RPC (request → response)
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  
  // Server streaming
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Client streaming
  rpc CreateUsers(stream CreateUserRequest) returns (CreateUsersResponse);
  
  // Bidirectional streaming
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

### 3.2 Server Implementation (Python)

```python
import grpc
from concurrent import futures
import user_pb2
import user_pb2_grpc

class UserServiceServicer(user_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        # Fetch user from database
        user = db.query(User).filter(User.id == request.id).first()
        
        if not user:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("User not found")
            return user_pb2.GetUserResponse()
        
        return user_pb2.GetUserResponse(
            user=user_pb2.User(
                id=user.id,
                name=user.name,
                email=user.email
            )
        )

# Start server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    user_pb2_grpc.add_UserServiceServicer_to_server(
        UserServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

### 3.3 Client Usage

```python
import grpc
import user_pb2
import user_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = user_pb2_grpc.UserServiceStub(channel)

# Call RPC
request = user_pb2.GetUserRequest(id=123)
response = stub.GetUser(request)

print(response.user.name)  # "Alice"
```

---

## 4. Streaming

### 4.1 Server Streaming

**Use case**: Download large dataset

```protobuf
service UserService {
  rpc ListUsers(ListUsersRequest) returns (stream User);
}
```

**Server**:
```python
def ListUsers(self, request, context):
    users = db.query(User).all()
    
    for user in users:
        yield user_pb2.User(
            id=user.id,
            name=user.name,
            email=user.email
        )
```

**Client**:
```python
request = user_pb2.ListUsersRequest()

# Stream responses
for user in stub.ListUsers(request):
    print(user.name)
```

### 4.2 Client Streaming

**Use case**: Upload large dataset

```protobuf
service UserService {
  rpc CreateUsers(stream CreateUserRequest) returns (CreateUsersResponse);
}
```

**Client**:
```python
def user_generator():
    for i in range(1000):
        yield user_pb2.CreateUserRequest(
            name=f"User{i}",
            email=f"user{i}@example.com"
        )

response = stub.CreateUsers(user_generator())
print(f"Created {response.count} users")
```

**Server**:
```python
def CreateUsers(self, request_iterator, context):
    count = 0
    
    for user_request in request_iterator:
        user = User(name=user_request.name, email=user_request.email)
        db.add(user)
        count += 1
    
    db.commit()
    return user_pb2.CreateUsersResponse(count=count)
```

### 4.3 Bidirectional Streaming

**Use case**: Chat application

```protobuf
service ChatService {
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

**Server**:
```python
def Chat(self, request_iterator, context):
    for message in request_iterator:
        # Broadcast to all connected clients
        for client in connected_clients:
            yield user_pb2.ChatMessage(
                user=message.user,
                text=message.text
            )
```

---

## 5. Error Handling

### 5.1 Status Codes

```python
from grpc import StatusCode

# Server
def GetUser(self, request, context):
    if request.id < 0:
        context.set_code(StatusCode.INVALID_ARGUMENT)
        context.set_details("ID must be positive")
        return user_pb2.GetUserResponse()
    
    user = db.query(User).filter(User.id == request.id).first()
    
    if not user:
        context.set_code(StatusCode.NOT_FOUND)
        context.set_details(f"User {request.id} not found")
        return user_pb2.GetUserResponse()
    
    return user_pb2.GetUserResponse(user=user_pb2.User(...))

# Client
try:
    response = stub.GetUser(request)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.NOT_FOUND:
        print("User not found")
    elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        print("Invalid request")
```

### 5.2 Common Status Codes

- `OK`: Success
- `CANCELLED`: Cancelled by client
- `INVALID_ARGUMENT`: Invalid input
- `NOT_FOUND`: Resource not found
- `PERMISSION_DENIED`: Not authorized
- `UNAUTHENTICATED`: Not authenticated
- `UNAVAILABLE`: Service unavailable
- `INTERNAL`: Server error

---

## 6. Authentication

### 6.1 Token-Based Auth

```python
# Client sends token in metadata
metadata = [('authorization', f'Bearer {jwt_token}')]
response = stub.GetUser(request, metadata=metadata)

# Server validates token
def GetUser(self, request, context):
    # Get metadata
    metadata = dict(context.invocation_metadata())
    auth_header = metadata.get('authorization', '')
    
    if not auth_header.startswith('Bearer '):
        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
        context.set_details("Missing token")
        return user_pb2.GetUserResponse()
    
    token = auth_header.replace('Bearer ', '')
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        context.set_code(grpc.StatusCode.UNAUTHENTICATED)
        context.set_details("Token expired")
        return user_pb2.GetUserResponse()
    
    # Continue with authenticated user
    ...
```

### 6.2 mTLS (Mutual TLS)

```python
import grpc

# Server with TLS
server_credentials = grpc.ssl_server_credentials(
    [(private_key, certificate)]
)
server = grpc.server(futures.ThreadPoolExecutor())
server.add_secure_port('[::]:50051', server_credentials)

# Client with TLS
channel_credentials = grpc.ssl_channel_credentials(
    root_certificates=ca_cert
)
channel = grpc.secure_channel('localhost:50051', channel_credentials)
stub = user_pb2_grpc.UserServiceStub(channel)
```

---

## 7. Production Patterns

### 7.1 Load Balancing

**Client-side load balancing**:
```python
# DNS-based
channel = grpc.insecure_channel(
    'user-service:50051',
    options=[('grpc.lb_policy_name', 'round_robin')]
)
```

**Server-side (Envoy proxy)**:
```yaml
# envoy.yaml
static_resources:
  listeners:
  - address:
      socket_address:
        address: 0.0.0.0
        port_value: 9211
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          route_config:
            virtual_hosts:
            - name: users
              routes:
              - match: {prefix: "/user.UserService"}
                route:
                  cluster: user-service
  clusters:
  - name: user-service
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: user-service-1
                port_value: 50051
        - endpoint:
            address:
              socket_address:
                address: user-service-2
                port_value: 50051
```

### 7.2 Health Checks

```protobuf
service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
  }
  ServingStatus status = 1;
}
```

### 7.3 Timeouts & Retries

```python
# Client timeout
response = stub.GetUser(
    request,
    timeout=5  # 5 seconds
)

# Retry policy
channel = grpc.insecure_channel(
    'localhost:50051',
    options=[
        ('grpc.enable_retries', 1),
        ('grpc.service_config', json.dumps({
            "methodConfig": [{
                "name": [{"service": "user.UserService"}],
                "retryPolicy": {
                    "maxAttempts": 3,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "1s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE"]
                }
            }]
        }))
    ]
)
```

---

## 8. gRPC vs REST vs GraphQL

| Feature | REST | GraphQL | gRPC |
|:--------|:-----|:--------|:-----|
| **Protocol** | HTTP/1.1 | HTTP/1.1 | HTTP/2 |
| **Payload** | JSON (text) | JSON (text) | Protobuf (binary) |
| **Performance** | Baseline | Same as REST | 10x faster |
| **Streaming** | No | Subscriptions | Native |
| **Browser Support** | Yes | Yes | Limited (needs proxy) |
| **Contract** | OpenAPI | Schema | .proto |
| **Code Generation** | Optional | Optional | Required |
| **Use Case** | Public APIs | Flexible queries | Microservices |

---

## 9. When to Use gRPC

✅ **Use gRPC when**:
- **Microservices** (internal communication)
- **High performance** critical
- **Streaming** needed
- **Strong typing** preferred
- **Polyglot** teams (code generation)

❌ **Don't use gRPC when**:
- **Browser clients** (needs gRPC-Web proxy)
- **Public APIs** (REST more familiar)
- **Human-readable** payloads needed

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **gRPC** - High-performance RPC framework
2. ✅ **Protocol Buffers** - Binary serialization (10x faster)
3. ✅ **HTTP/2** - Multiplexing, streaming
4. ✅ **Streaming** - Unary, server, client, bidirectional
5. ✅ **Status Codes** - Standardized error handling
6. ✅ **mTLS** - Mutual authentication
7. ✅ **Microservices** - Primary use case

### 10.2 gRPC Best Practices

- [ ] Use Protocol Buffers for schema
- [ ] Never change field numbers
- [ ] Implement health checks
- [ ] Use load balancing (client or server-side)
- [ ] Set timeouts & retries
- [ ] Use mTLS for authentication
- [ ] Monitor with OpenTelemetry

### 10.3 Tomorrow (Day 25): API Documentation

- **OpenAPI/Swagger**: Auto-generate docs
- **Postman Collections**: Share examples
- **API Blueprint**: Markdown-based
- **AsyncAPI**: Event-driven APIs
- **Interactive Docs**: Try-it-out features
- **Versioned Docs**: Per API version

See you tomorrow! 🚀

---

**File Statistics**: ~900 lines | gRPC & Protocol Buffers mastered ✅
