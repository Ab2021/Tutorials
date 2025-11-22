# Lab 5.1: Profiling and Benchmarking

## Objective
Profile and benchmark Rust code for performance.

## Exercises

### Exercise 1: Criterion Benchmarks
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

### Exercise 2: Flamegraph Profiling
```bash
cargo install flamegraph
cargo flamegraph
```

## Success Criteria
✅ Write benchmarks  
✅ Profile code  
✅ Identify bottlenecks

## Next Steps
Lab 5.2: Memory Optimization
