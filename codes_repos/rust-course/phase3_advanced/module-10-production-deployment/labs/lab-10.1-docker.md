# Lab 10.1: Docker Containerization

## Objective
Containerize Rust applications with Docker.

## Exercises

### Exercise 1: Basic Dockerfile
```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bullseye-slim
COPY --from=builder /app/target/release/myapp /usr/local/bin/
CMD ["myapp"]
```

### Exercise 2: Multi-stage Build
```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
COPY src ./src
RUN cargo build --release

FROM debian:bullseye-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/myapp /usr/local/bin/
CMD ["myapp"]
```

## Success Criteria
✅ Create Dockerfile  
✅ Build Docker image  
✅ Run containerized app

## Next Steps
Lab 10.2: CI/CD with GitHub Actions
