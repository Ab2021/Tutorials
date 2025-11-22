# Lab 2.2: Working with Streams

## Objective
Process asynchronous streams of data.

## Exercises

### Exercise 1: Basic Stream
```rust
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    let mut stream = tokio_stream::iter(vec![1, 2, 3]);
    
    while let Some(value) = stream.next().await {
        println!("{}", value);
    }
}
```

### Exercise 2: Stream Transformations
```rust
let stream = tokio_stream::iter(vec![1, 2, 3, 4, 5])
    .map(|x| x * 2)
    .filter(|x| x % 4 == 0);
```

## Success Criteria
✅ Create streams  
✅ Transform streams  
✅ Consume streams

## Next Steps
Lab 2.3: Async Error Handling
