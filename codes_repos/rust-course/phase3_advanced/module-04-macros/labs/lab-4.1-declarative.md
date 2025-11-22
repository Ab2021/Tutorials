# Lab 4.1: Declarative Macros

## Objective
Write declarative macros with macro_rules!

## Exercises

### Exercise 1: Simple Macro
```rust
macro_rules! say_hello {
    () => {
        println!("Hello!");
    };
}

fn main() {
    say_hello!();
}
```

### Exercise 2: Macro with Arguments
```rust
macro_rules! create_function {
    ($func_name:ident) => {
        fn $func_name() {
            println!("Function {:?} called", stringify!($func_name));
        }
    };
}

create_function!(foo);
create_function!(bar);

fn main() {
    foo();
    bar();
}
```

### Exercise 3: vec! Macro Clone
```rust
macro_rules! my_vec {
    ($($x:expr),*) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}
```

## Success Criteria
✅ Write declarative macros  
✅ Use pattern matching  
✅ Handle repetitions

## Next Steps
Lab 4.2: Procedural Macros
