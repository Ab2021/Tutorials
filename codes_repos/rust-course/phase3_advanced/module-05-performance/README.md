# Module 05: Performance Optimization

## 🎯 Learning Objectives

- Profile Rust applications
- Benchmark code with criterion
- Optimize memory usage
- Use SIMD for vectorization
- Apply compiler optimizations

---

## 📖 Core Concepts

### Profiling

```bash
# CPU profiling
cargo install flamegraph
cargo flamegraph

# Memory profiling
valgrind --tool=massif target/release/myapp
```

### Benchmarking with Criterion

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

### Memory Optimization

```rust
// Use references instead of cloning
fn process(data: &[u8]) { }

// Use Cow for conditional cloning
use std::borrow::Cow;
fn process_text(text: Cow<str>) { }

// Reuse allocations
let mut buffer = Vec::with_capacity(1024);
```

### SIMD

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

unsafe {
    let a = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
    let b = _mm_set_ps(5.0, 6.0, 7.0, 8.0);
    let result = _mm_add_ps(a, b);
}
```

---

## 🔑 Key Takeaways

1. **Profile first** - Measure before optimizing
2. **Criterion** for accurate benchmarks
3. **Avoid allocations** when possible
4. **SIMD** for data parallelism
5. **Release mode** for production

Complete 10 labs, then proceed to Module 06: Web Development
