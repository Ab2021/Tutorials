# Day 24: gRPC & Protocol Buffers

## 1. The Need for Speed

JSON is great for humans, but slow for machines.
*   **Text-based**: Parsing JSON takes CPU cycles.
*   **Verbose**: `{"name": "Alice"}` repeats the key "name" every time.
*   **Loose**: No strict types.

**gRPC** (Google RPC) uses **Protobuf** (Protocol Buffers) to solve this.

### 1.1 What is Protobuf?
A binary serialization format.
*   **Schema**: Defined in `.proto` files.
*   **Compact**: `name = 1` (Field ID 1). The wire format just sends `08 05 41 6C 69 63 65` (Field 1, Length 5, "Alice"). No keys sent.
*   **Fast**: 10x faster serialization than JSON.

---

## 2. gRPC Concepts

### 2.1 Service Definition
```protobuf
syntax = "proto3";

service UserService {
  rpc GetUser (UserRequest) returns (UserResponse);
}

message UserRequest {
  int32 id = 1;
}

message UserResponse {
  int32 id = 1;
  string name = 2;
}
```

### 2.2 Code Generation
You run `protoc`. It generates Python/Go/Java code.
*   **Client Stub**: `client.GetUser(req)`
*   **Server Base**: `class UserService(UserServiceBase): ...`

### 2.3 HTTP/2
gRPC runs on HTTP/2.
*   **Multiplexing**: Multiple requests over one TCP connection.
*   **Streaming**: Native support for streams.

---

## 3. Streaming Patterns

1.  **Unary**: Request -> Response. (Like REST).
2.  **Server Streaming**: Request -> Stream of Responses. (e.g., "Download File", "Stock Ticker").
3.  **Client Streaming**: Stream of Requests -> Response. (e.g., "Upload File").
4.  **Bidirectional Streaming**: Stream <-> Stream. (e.g., "Chat App", "Live Game").

---

## 4. gRPC vs REST

| Feature | REST | gRPC |
| :--- | :--- | :--- |
| **Payload** | JSON (Text) | Protobuf (Binary) |
| **Contract** | OpenAPI (Optional) | .proto (Required) |
| **Protocol** | HTTP/1.1 or 2 | HTTP/2 |
| **Browser** | Native Support | Requires gRPC-Web Proxy |
| **Use Case** | Public APIs | Internal Microservices |

---

## 5. Summary

Today we optimized our internal traffic.
*   **Protobuf**: Small, fast, typed.
*   **gRPC**: The standard for microservice communication.
*   **Streaming**: Real-time powers.

**Tomorrow (Day 25)**: We will learn how to document all this. Whether it's REST or gRPC, if you don't document it, it doesn't exist.
