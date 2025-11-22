# Lab 6.1: Building REST APIs with Axum

## Objective
Create REST APIs using the Axum framework.

## Exercises

### Exercise 1: Basic Server
```rust
use axum::{Router, routing::get};

async fn hello() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(hello));
    
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### Exercise 2: JSON Responses
```rust
use axum::Json;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct User {
    id: u64,
    name: String,
}

async fn get_user() -> Json<User> {
    Json(User {
        id: 1,
        name: "Alice".to_string(),
    })
}
```

## Success Criteria
✅ Create HTTP server  
✅ Handle routes  
✅ Return JSON responses

## Next Steps
Lab 6.2: Database Integration
