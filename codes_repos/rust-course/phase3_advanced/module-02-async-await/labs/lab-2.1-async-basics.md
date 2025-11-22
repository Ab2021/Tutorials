# Lab 2.1: Async/Await Basics with Tokio

## Objective
Write asynchronous code using async/await syntax.

## Exercises

### Exercise 1: Basic Async Function
```rust
use tokio;

async fn say_hello() {
    println!("Hello, async world!");
}

#[tokio::main]
async fn main() {
    say_hello().await;
}
```

### Exercise 2: Concurrent Tasks
```rust
#[tokio::main]
async fn main() {
    let task1 = tokio::spawn(async {
        println!("Task 1");
    });
    
    let task2 = tokio::spawn(async {
        println!("Task 2");
    });
    
    task1.await.unwrap();
    task2.await.unwrap();
}
```

### Exercise 3: Async HTTP Request
```rust
use reqwest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::get("https://api.github.com")
        .await?
        .text()
        .await?;
    
    println!("{}", response);
    Ok(())
}
```

## Success Criteria
✅ Write async functions  
✅ Use .await  
✅ Spawn concurrent tasks

## Next Steps
Lab 2.2: Futures and Streams
