# Module 04: Macros and Metaprogramming

## 🎯 Learning Objectives

- Write declarative macros
- Create procedural macros
- Build derive macros
- Understand macro hygiene
- Generate code at compile time

---

## 📖 Core Concepts

### Declarative Macros

```rust
macro_rules! vec_of_strings {
    ($($x:expr),*) => {
        vec![$($x.to_string()),*]
    };
}

let v = vec_of_strings!["hello", "world"];
```

### Derive Macros

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn;

#[proc_macro_derive(HelloMacro)]
pub fn hello_macro_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    impl_hello_macro(&ast)
}
```

### Attribute Macros

```rust
#[route(GET, "/")]
fn index() -> &'static str {
    "Hello, world!"
}
```

### Function-like Macros

```rust
#[proc_macro]
pub fn sql(input: TokenStream) -> TokenStream {
    // Parse SQL and generate code
}
```

---

## 🔑 Key Takeaways

1. **Declarative macros** for pattern matching
2. **Procedural macros** for code generation
3. **Derive macros** for automatic trait implementation
4. **Attribute macros** for annotations
5. **Macros** run at compile time

Complete 10 labs, then proceed to Module 05: Performance
