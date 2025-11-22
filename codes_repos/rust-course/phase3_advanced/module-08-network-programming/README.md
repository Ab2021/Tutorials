# Module 08: Network Programming

## 🎯 Learning Objectives

- Work with TCP/UDP sockets
- Implement network protocols
- Build client-server applications
- Handle network security
- Create distributed systems

---

## 📖 Core Concepts

### TCP Server

```rust
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;
    
    loop {
        let (mut socket, _) = listener.accept().await?;
        
        tokio::spawn(async move {
            let mut buf = [0; 1024];
            socket.read(&mut buf).await.unwrap();
            socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await.unwrap();
        });
    }
}
```

### UDP Socket

```rust
use tokio::net::UdpSocket;

let socket = UdpSocket::bind("0.0.0.0:8080").await?;
let mut buf = [0; 1024];

loop {
    let (len, addr) = socket.recv_from(&mut buf).await?;
    socket.send_to(&buf[..len], addr).await?;
}
```

### HTTP Client

```rust
use reqwest;

let response = reqwest::get("https://api.example.com/data")
    .await?
    .json::<MyStruct>()
    .await?;
```

---

## 🔑 Key Takeaways

1. **TCP** for reliable connections
2. **UDP** for fast, connectionless communication
3. **Async** for handling many connections
4. **Tokio** for async networking
5. **Security** is critical

Complete 10 labs, then proceed to Module 09: Systems Programming
