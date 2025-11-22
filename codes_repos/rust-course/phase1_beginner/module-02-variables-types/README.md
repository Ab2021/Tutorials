# Module 2: Basic Syntax and Data Types

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand variables and mutability in Rust
- Work with primitive data types
- Use compound types (tuples and arrays)
- Write and call functions
- Implement control flow with if/else and loops
- Use pattern matching with match expressions

---

## 📖 Theoretical Concepts

### 2.1 Variables and Mutability

#### Immutable by Default

In Rust, variables are **immutable by default**:

```rust
let x = 5;
x = 6;  // ❌ ERROR: cannot assign twice to immutable variable
```

#### Mutable Variables

Use `mut` to make variables mutable:

```rust
let mut x = 5;
println!("The value of x is: {}", x);
x = 6;  // ✅ OK
println!("The value of x is: {}", x);
```

#### Constants

Constants are always immutable and must have type annotations:

```rust
const MAX_POINTS: u32 = 100_000;
const PI: f64 = 3.14159;
```

**Differences between `let` and `const`:**
- Constants use `const` keyword
- Type must be annotated
- Can be declared in any scope, including global
- Can only be set to constant expressions, not computed at runtime
- Naming convention: SCREAMING_SNAKE_CASE

#### Shadowing

You can declare a new variable with the same name:

```rust
let x = 5;
let x = x + 1;  // Shadows previous x
let x = x * 2;  // Shadows again
println!("The value of x is: {}", x);  // 12

// Can change type with shadowing
let spaces = "   ";
let spaces = spaces.len();  // Now it's a number
```

**Shadowing vs Mutability:**
- Shadowing creates a new variable (can change type)
- Mutability modifies the same variable (type stays the same)

---

### 2.2 Data Types

Rust is **statically typed** - all types must be known at compile time.

#### Scalar Types

Represent a single value.

##### Integers

| Length  | Signed | Unsigned |
|---------|--------|----------|
| 8-bit   | i8     | u8       |
| 16-bit  | i16    | u16      |
| 32-bit  | i32    | u32      |
| 64-bit  | i64    | u64      |
| 128-bit | i128   | u128     |
| arch    | isize  | usize    |

```rust
let a: i32 = 42;           // Signed 32-bit (default)
let b: u8 = 255;           // Unsigned 8-bit
let c = 1_000_000;         // Underscores for readability
let d = 0xff;              // Hexadecimal
let e = 0o77;              // Octal
let f = 0b1111_0000;       // Binary
let g = b'A';              // Byte (u8 only)
```

**Integer Overflow:**
- Debug mode: panics
- Release mode: wraps around (use with caution)

##### Floating-Point

```rust
let x: f64 = 2.0;      // f64 (default, double precision)
let y: f32 = 3.0;      // f32 (single precision)
```

##### Boolean

```rust
let t: bool = true;
let f: bool = false;
```

##### Character

```rust
let c: char = 'z';
let z: char = 'ℤ';
let heart: char = '❤';
```

- Uses single quotes
- 4 bytes in size
- Represents a Unicode Scalar Value

---

### 2.3 Compound Types

#### Tuples

Group multiple values of different types:

```rust
let tup: (i32, f64, u8) = (500, 6.4, 1);

// Destructuring
let (x, y, z) = tup;

// Access by index
let five_hundred = tup.0;
let six_point_four = tup.1;
let one = tup.2;
```

**Unit Type:**
```rust
let unit: () = ();  // Empty tuple, represents no value
```

#### Arrays

Fixed-length collections of the same type:

```rust
let a: [i32; 5] = [1, 2, 3, 4, 5];

// Initialize with same value
let b = [3; 5];  // [3, 3, 3, 3, 3]

// Access elements
let first = a[0];
let second = a[1];
```

**Key Points:**
- Fixed length
- Allocated on the stack
- Accessing out of bounds causes runtime panic

---

### 2.4 Functions

```rust
fn main() {
    println!("Hello, world!");
    
    another_function(5);
    
    let result = add(3, 4);
    println!("Result: {}", result);
}

fn another_function(x: i32) {
    println!("The value of x is: {}", x);
}

fn add(a: i32, b: i32) -> i32 {
    a + b  // Expression (no semicolon)
}
```

**Key Concepts:**

**Parameters:**
- Must declare type for each parameter

**Return Values:**
- Declared with `->`
- Return value is the final expression (no semicolon)
- Or use `return` keyword for early returns

**Statements vs Expressions:**
```rust
let y = {
    let x = 3;
    x + 1  // Expression (no semicolon)
};  // y = 4

let z = {
    let x = 3;
    x + 1;  // Statement (with semicolon)
};  // z = () (unit type)
```

---

### 2.5 Control Flow

#### if Expressions

```rust
let number = 6;

if number % 4 == 0 {
    println!("divisible by 4");
} else if number % 3 == 0 {
    println!("divisible by 3");
} else if number % 2 == 0 {
    println!("divisible by 2");
} else {
    println!("not divisible by 4, 3, or 2");
}
```

**if in let statement:**
```rust
let condition = true;
let number = if condition { 5 } else { 6 };
```

#### Loops

**loop - Infinite Loop:**
```rust
loop {
    println!("again!");
    break;  // Exit the loop
}

// Return value from loop
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;
    }
};
```

**while - Conditional Loop:**
```rust
let mut number = 3;

while number != 0 {
    println!("{}!", number);
    number -= 1;
}
```

**for - Iterate Over Collection:**
```rust
let a = [10, 20, 30, 40, 50];

for element in a {
    println!("the value is: {}", element);
}

// Range
for number in 1..4 {  // 1, 2, 3
    println!("{}", number);
}

// Inclusive range
for number in 1..=4 {  // 1, 2, 3, 4
    println!("{}", number);
}

// Reverse
for number in (1..4).rev() {  // 3, 2, 1
    println!("{}", number);
}
```

---

### 2.6 Pattern Matching

#### match Expression

```rust
let number = 3;

match number {
    1 => println!("One!"),
    2 | 3 => println!("Two or Three!"),
    4..=10 => println!("Four through Ten!"),
    _ => println!("Something else!"),
}
```

**match with Return Value:**
```rust
let number = 5;

let description = match number {
    1 => "one",
    2 => "two",
    3 => "three",
    _ => "many",
};
```

**Key Points:**
- Must be exhaustive (cover all cases)
- Use `_` as catch-all
- Can return values
- Can match ranges with `..=`
- Can match multiple patterns with `|`

---

## 🔑 Key Takeaways

1. Variables are immutable by default; use `mut` for mutability
2. Rust has strong static typing with type inference
3. Scalar types: integers, floats, booleans, characters
4. Compound types: tuples and arrays
5. Functions must declare parameter and return types
6. Expressions return values; statements don't
7. Control flow: if, loop, while, for
8. match provides powerful pattern matching

---

## 📚 Additional Resources

- [The Rust Book - Chapter 3](https://doc.rust-lang.org/book/ch03-00-common-programming-concepts.html)
- [Rust by Example - Primitives](https://doc.rust-lang.org/rust-by-example/primitives.html)
- [Rust by Example - Flow Control](https://doc.rust-lang.org/rust-by-example/flow_control.html)

---

## ⏭️ Next Steps

Proceed to the labs directory to practice:
1. Lab 2.1: Variable exercises
2. Lab 2.2: Temperature converter
3. Lab 2.3: Fibonacci sequence generator
4. Lab 2.4: Pattern matching with match expressions

After completing the labs, move on to **Module 3: Ownership and Borrowing**.
