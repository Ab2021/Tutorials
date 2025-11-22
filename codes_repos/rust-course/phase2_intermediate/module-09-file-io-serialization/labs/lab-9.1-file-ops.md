# Lab 9.1: File Operations

## Objective
Perform file I/O operations in Rust.

## Exercises

### Exercise 1: Reading Files
```rust
use std::fs;

fn main() -> std::io::Result<()> {
    let contents = fs::read_to_string("file.txt")?;
    println!("{}", contents);
    Ok(())
}
```

### Exercise 2: Writing Files
```rust
fs::write("output.txt", "Hello, world!")?;
```

### Exercise 3: Buffered I/O
```rust
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

let file = File::open("input.txt")?;
let reader = BufReader::new(file);

for line in reader.lines() {
    println!("{}", line?);
}
```

## Success Criteria
✅ Read files  
✅ Write files  
✅ Use buffered I/O

## Next Steps
Lab 9.2: JSON Serialization with Serde
