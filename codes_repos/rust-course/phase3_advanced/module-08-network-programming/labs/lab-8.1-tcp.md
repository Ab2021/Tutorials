# Lab 8.1: TCP Server and Client

## Objective
Build TCP server and client applications.

## Exercises

### Exercise 1: TCP Server
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

### Exercise 2: TCP Client
```rust
use tokio::net::TcpStream;

let mut stream = TcpStream::connect("127.0.0.1:8080").await?;
stream.write_all(b"Hello").await?;
```

## Success Criteria
✅ Create TCP server  
✅ Handle connections  
✅ Build TCP client

## Next Steps
Lab 8.2: UDP Communication
