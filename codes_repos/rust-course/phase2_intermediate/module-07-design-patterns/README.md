# Module 07: Design Patterns in Rust

## 🎯 Learning Objectives

- Implement common design patterns in Rust
- Understand Rust-specific patterns
- Apply patterns to real problems
- Know when to use each pattern
- Build maintainable code

---

## 📖 Core Patterns

### Builder Pattern

```rust
pub struct Config {
    host: String,
    port: u16,
    timeout: u64,
}

pub struct ConfigBuilder {
    host: Option<String>,
    port: Option<u16>,
    timeout: Option<u64>,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        ConfigBuilder {
            host: None,
            port: None,
            timeout: None,
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
            timeout: self.timeout.unwrap_or(30),
        })
    }
}
```

### Strategy Pattern

```rust
trait CompressionStrategy {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
}

struct GzipCompression;
impl CompressionStrategy for GzipCompression {
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        // Gzip compression
        vec![]
    }
}

struct Compressor {
    strategy: Box<dyn CompressionStrategy>,
}

impl Compressor {
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        self.strategy.compress(data)
    }
}
```

### Newtype Pattern

```rust
struct UserId(u64);
struct ProductId(u64);

// Can't accidentally mix them up!
fn get_user(id: UserId) -> User { /* ... */ }
```

### Type State Pattern

```rust
struct Locked;
struct Unlocked;

struct Door<State> {
    _state: PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> {
        Door { _state: PhantomData }
    }
}

impl Door<Unlocked> {
    fn open(&self) {
        println!("Door opened");
    }
}
```

---

## 🔑 Key Takeaways

1. **Builder** for complex construction
2. **Strategy** for interchangeable algorithms
3. **Newtype** for type safety
4. **Type State** for compile-time state machines
5. **Patterns adapt** to Rust's ownership system

Complete 10 labs, then proceed to Module 08: Concurrency Fundamentals
