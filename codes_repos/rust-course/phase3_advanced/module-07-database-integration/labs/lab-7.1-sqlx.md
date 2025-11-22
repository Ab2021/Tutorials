# Lab 7.1: SQLx Basics

## Objective
Use SQLx for async database operations.

## Exercises

### Exercise 1: Connecting to Database
```rust
use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect("postgres://user:pass@localhost/db").await?;
    
    Ok(())
}
```

### Exercise 2: Queries
```rust
#[derive(sqlx::FromRow)]
struct User {
    id: i64,
    name: String,
}

let users = sqlx::query_as::<_, User>("SELECT * FROM users")
    .fetch_all(&pool)
    .await?;
```

## Success Criteria
✅ Connect to database  
✅ Execute queries  
✅ Map results to structs

## Next Steps
Lab 7.2: Migrations
