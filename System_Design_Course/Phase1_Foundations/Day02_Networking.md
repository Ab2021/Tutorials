# Day 2: Networking Internals

## 1. The OSI Model (Simplified)
*   **L4 (Transport):** TCP, UDP.
*   **L7 (Application):** HTTP, FTP, SMTP, gRPC.

## 2. TCP vs UDP
### TCP (Transmission Control Protocol)
*   **Features:** Reliable, Ordered, Error-checked.
*   **Mechanism:** 3-Way Handshake (SYN, SYN-ACK, ACK).
*   **Use Case:** Web (HTTP), Email, File Transfer.
*   **Cons:** Slower (handshake + head-of-line blocking).

### UDP (User Datagram Protocol)
*   **Features:** Unreliable, Unordered, Fast (Fire and Forget).
*   **Use Case:** Video Streaming, Gaming, DNS.
*   **Cons:** Packet loss is expected.

## 3. HTTP Evolution
*   **HTTP/1.1:** Text-based. Keep-Alive. Head-of-Line Blocking (one request per connection).
*   **HTTP/2:** Binary. Multiplexing (multiple requests over one connection). Header Compression (HPACK).
*   **HTTP/3 (QUIC):** Built on UDP. Solves TCP Head-of-Line blocking. Faster handshake.

## 4. WebSockets
*   **Concept:** Full-duplex communication over a single TCP connection.
*   **Use Case:** Chat apps, Real-time notifications.
*   **Handshake:** Starts as HTTP, upgrades to WebSocket.

## 5. Code: Simple TCP Server (Python)
```python
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9999))
server.listen(5)

print("Listening on 9999...")

while True:
    client, addr = server.accept()
    print(f"Accepted connection from {addr}")
    client.send(b"Hello from TCP Server!\n")
    client.close()
```
