# Module 24: Networking

## Overview
Network programming with modern C++, covering socket programming, asynchronous I/O, and protocol implementation.

## Learning Objectives
By the end of this module, you will be able to:
- Implement TCP/UDP socket programming
- Use asynchronous I/O with Asio
- Build HTTP clients and servers
- Implement custom protocols
- Handle network security basics
- Create scalable network applications

## Key Concepts

### 1. Socket Programming
Low-level network communication.

```cpp
#include <sys/socket.h>
#include <netinet/in.h>

int sockfd = socket(AF_INET, SOCK_STREAM, 0);
sockaddr_in addr{};
addr.sin_family = AF_INET;
addr.sin_port = htons(8080);
bind(sockfd, (sockaddr*)&addr, sizeof(addr));
listen(sockfd, 10);
```

### 2. Asynchronous I/O with Asio
Non-blocking network operations.

```cpp
#include <boost/asio.hpp>

boost::asio::io_context io;
boost::asio::ip::tcp::socket socket(io);

socket.async_read_some(boost::asio::buffer(data),
    [](boost::system::error_code ec, size_t bytes) {
        // Handle read
    });
```

### 3. HTTP Protocol
Building HTTP clients and servers.

```cpp
class HttpServer {
    void handleRequest(const std::string& request) {
        std::string response = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "\r\n"
            "<html><body>Hello</body></html>";
        send(socket, response);
    }
};
```

### 4. Protocol Implementation
Custom binary protocols.

```cpp
struct Message {
    uint32_t length;
    uint16_t type;
    std::vector<char> payload;
    
    void serialize(std::vector<char>& buffer) {
        // Serialize to network byte order
    }
};
```

### 5. Network Security
TLS/SSL and secure communication.

```cpp
#include <openssl/ssl.h>

SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
SSL* ssl = SSL_new(ctx);
SSL_set_fd(ssl, sockfd);
SSL_connect(ssl);
```

## Rust Comparison

### Async Networking
**C++:**
```cpp
socket.async_read_some(buffer, handler);
```

**Rust:**
```rust
use tokio::net::TcpStream;
let stream = TcpStream::connect("127.0.0.1:8080").await?;
```

### HTTP Server
**C++:**
```cpp
// Using Boost.Beast or custom
```

**Rust:**
```rust
use actix_web::{web, App, HttpServer};
```

## Labs

1. **Lab 24.1**: TCP Socket Basics
2. **Lab 24.2**: UDP Communication
3. **Lab 24.3**: Asio Introduction
4. **Lab 24.4**: Async TCP Server
5. **Lab 24.5**: HTTP Client
6. **Lab 24.6**: HTTP Server
7. **Lab 24.7**: WebSocket Protocol
8. **Lab 24.8**: Custom Protocol
9. **Lab 24.9**: TLS/SSL Integration
10. **Lab 24.10**: Chat Server (Capstone)

## Additional Resources
- Boost.Asio documentation
- "Unix Network Programming" by Stevens
- RFC 2616 (HTTP/1.1)

## Next Module
After completing this module, proceed to **Module 25: Serialization**.
