# Day 2 Deep Dive: gRPC & Protocol Buffers

## 1. What is gRPC?
gRPC (Google Remote Procedure Call) is a high-performance RPC framework.
*   **Transport:** HTTP/2 (Multiplexing, Streaming).
*   **Format:** Protocol Buffers (Binary, not JSON).
*   **Speed:** 7-10x faster than REST+JSON.

## 2. Protocol Buffers (Protobuf)
*   **Schema-first:** You define `.proto` files.
*   **Binary:** Compact. No field names in payload (uses tags).
*   **Type-safe:** Generated code for Python, Go, Java, C++.

### Example `.proto`
```protobuf
syntax = "proto3";

service UserService {
  rpc GetUser (UserRequest) returns (UserResponse);
}

message UserRequest {
  int32 id = 1;
}

message UserResponse {
  string name = 1;
  string email = 2;
}
```

## 3. REST vs gRPC
| Feature | REST | gRPC |
| :--- | :--- | :--- |
| **Protocol** | HTTP/1.1 (usually) | HTTP/2 |
| **Format** | JSON (Text) | Protobuf (Binary) |
| **Readability** | Human readable | Not readable |
| **Browser Support** | Native | Requires gRPC-Web proxy |
| **Use Case** | Public APIs | Internal Microservices |

## 4. Head-of-Line Blocking (HOL)
*   **HTTP/1.1:** If Request 1 is slow, Request 2 waits. (TCP HOL + HTTP HOL)
*   **HTTP/2:** Solves HTTP HOL (Multiplexing). Still suffers TCP HOL (if one packet drops, all streams wait).
*   **HTTP/3 (QUIC):** Solves TCP HOL by using UDP. Streams are independent.

## 5. When to use what?
*   **Public API:** REST (Ease of use).
*   **Internal Microservices:** gRPC (Performance, Type safety).
*   **Real-time:** WebSockets.
