# Lab 6.2: API Design Principles

## Objective
Design clean, usable Rust APIs.

## Exercises

### Exercise 1: Builder Pattern
```rust
pub struct Config {
    host: String,
    port: u16,
}

pub struct ConfigBuilder {
    host: Option<String>,
    port: Option<u16>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        ConfigBuilder {
            host: None,
            port: None,
        }
    }
    
    pub fn host(mut self, host: String) -> Self {
        self.host = Some(host);
        self
    }
    
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }
    
    pub fn build(self) -> Result<Config, String> {
        Ok(Config {
            host: self.host.ok_or("host required")?,
            port: self.port.unwrap_or(8080),
        })
    }
}
```

### Exercise 2: Ergonomic Error Types
```rust
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parse(String),
}
```

## Success Criteria
✅ Design ergonomic APIs  
✅ Use builder pattern  
✅ Create type aliases

## Next Steps
Lab 6.3: Publishing to crates.io
