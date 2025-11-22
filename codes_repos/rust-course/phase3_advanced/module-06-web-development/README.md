# Module 06: Web Development with Rust

## 🎯 Learning Objectives

- Build REST APIs with Actix-web/Axum
- Handle HTTP requests and responses
- Implement middleware
- Add authentication
- Deploy web services

---

## 📖 Core Concepts

### Actix-web Basics

```rust
use actix_web::{web, App, HttpResponse, HttpServer};

async fn index() -> HttpResponse {
    HttpResponse::Ok().body("Hello, world!")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### Axum Framework

```rust
use axum::{Router, routing::get};

async fn handler() -> &'static str {
    "Hello, World!"
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(handler));
    
    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### JSON APIs

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
}

async fn create_user(user: web::Json<User>) -> HttpResponse {
    HttpResponse::Ok().json(user.into_inner())
}
```

### Middleware

```rust
use actix_web::middleware::Logger;

App::new()
    .wrap(Logger::default())
    .route("/", web::get().to(index))
```

---

## 🔑 Key Takeaways

1. **Actix-web** or **Axum** for web frameworks
2. **Async** for high performance
3. **Serde** for JSON handling
4. **Middleware** for cross-cutting concerns
5. **Type safety** in routes

Complete 10 labs, then proceed to Module 07: Database Integration
