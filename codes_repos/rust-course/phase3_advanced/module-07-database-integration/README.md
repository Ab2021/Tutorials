# Module 07: Database Integration

## 🎯 Learning Objectives

- Use Diesel ORM
- Work with SQLx for async SQL
- Manage database connections
- Handle migrations
- Optimize database queries

---

## 📖 Core Concepts

### Diesel ORM

```rust
use diesel::prelude::*;

#[derive(Queryable)]
struct User {
    id: i32,
    name: String,
    email: String,
}

fn get_users(conn: &mut PgConnection) -> Vec<User> {
    users::table.load::<User>(conn).expect("Error loading users")
}
```

### SQLx (Async)

```rust
use sqlx::postgres::PgPoolOptions;

#[derive(sqlx::FromRow)]
struct User {
    id: i64,
    name: String,
}

let pool = PgPoolOptions::new()
    .connect("postgres://localhost/mydb").await?;

let users = sqlx::query_as::<_, User>("SELECT * FROM users")
    .fetch_all(&pool).await?;
```

### Migrations

```bash
# Diesel
diesel migration generate create_users
diesel migration run

# SQLx
sqlx migrate add create_users
sqlx migrate run
```

### Connection Pooling

```rust
use r2d2::Pool;
use diesel::r2d2::ConnectionManager;

let manager = ConnectionManager::<PgConnection>::new(database_url);
let pool = Pool::builder().build(manager)?;
```

---

## 🔑 Key Takeaways

1. **Diesel** for type-safe ORM
2. **SQLx** for async SQL
3. **Migrations** for schema management
4. **Connection pools** for performance
5. **Type safety** prevents SQL errors

Complete 10 labs, then proceed to Module 08: Network Programming
