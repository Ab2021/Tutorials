# Day 24: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is gRPC faster than REST?
**Answer:**
1.  **Binary Format (Protobuf)**: Smaller payload size (no field names) and faster serialization/deserialization (CPU efficient).
2.  **HTTP/2**: Multiplexing (no Head-of-Line blocking), Header Compression (HPACK), and persistent connections.
3.  **Strict Contract**: Code generation avoids runtime type checking overhead.

### Q2: How do you handle Backward Compatibility in Protobuf?
**Answer:**
*   **Rule 1**: Never change the **Tag Number** of an existing field. (e.g., `string name = 1;` -> `string full_name = 1;` is fine, but don't change `1` to `2`).
*   **Rule 2**: Never remove a required field (though `proto3` removed `required` keyword for this reason).
*   **Rule 3**: You can add new fields. Old clients will ignore them. New clients will see default values (0/empty) if reading old data.

### Q3: Why is Load Balancing gRPC harder than REST?
**Answer:**
*   **Persistent Connections**: gRPC uses HTTP/2, which keeps a single TCP connection open for a long time.
*   **L4 LB Issue**: A Layer 4 Load Balancer sees 1 connection and sends it to 1 server. Even if you send 1000 requests, they all go to that same server (imbalanced).
*   **Fix**: You need a **Layer 7 Load Balancer** (Envoy, Linkerd) or Client-side Load Balancing that understands gRPC frames and distributes individual requests, not just connections.

---

## Scenario-Based Questions

### Q4: You need to expose your gRPC service to a Browser (React App). How?
**Answer:**
*   **Problem**: Browsers don't support raw HTTP/2 frames required by gRPC.
*   **Solution**: Use **gRPC-Web**.
*   **Proxy**: Run an Envoy Proxy sidecar. The browser sends `gRPC-Web` (HTTP/1.1 text-encoded). Envoy translates it to `gRPC` (HTTP/2 binary) and forwards to the backend.

### Q5: When would you choose REST over gRPC for internal microservices?
**Answer:**
*   **Simplicity**: If the team knows JSON/HTTP and doesn't want the complexity of `.proto` files and codegen.
*   **Tooling**: Debugging JSON is easier (just `curl` it). Debugging binary Protobuf requires tools like `grpcurl`.
*   **Loose Coupling**: If you need ad-hoc JSON structures that change dynamically (no fixed schema).

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to use gRPC for a file upload service (uploading 1GB videos). Is this good?
**Answer:**
*   **Yes, but...**
*   **Client Streaming**: gRPC supports client streaming, which is perfect for uploading chunks.
*   **Memory**: Ensure you don't load the whole 1GB into RAM. Process chunks as they arrive.
*   **Alternative**: For massive files, Signed URLs to S3 (Direct Upload) is usually better to offload traffic from your servers.
