# Lab 7.2: Strategy Pattern

## Objective
Implement the Strategy pattern using traits.

## Exercises

### Exercise 1: Strategy with Traits
```rust
trait CompressionStrategy {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
}

struct GzipCompression;
impl CompressionStrategy for GzipCompression {
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        // Gzip compression logic
        data.to_vec()
    }
}

struct ZipCompression;
impl CompressionStrategy for ZipCompression {
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        // Zip compression logic
        data.to_vec()
    }
}

struct Compressor {
    strategy: Box<dyn CompressionStrategy>,
}

impl Compressor {
    fn new(strategy: Box<dyn CompressionStrategy>) -> Self {
        Compressor { strategy }
    }
    
    fn compress(&self, data: &[u8]) -> Vec<u8> {
        self.strategy.compress(data)
    }
}
```

## Success Criteria
✅ Define strategy trait  
✅ Implement multiple strategies  
✅ Use strategy pattern

## Next Steps
Lab 7.3: Observer Pattern
