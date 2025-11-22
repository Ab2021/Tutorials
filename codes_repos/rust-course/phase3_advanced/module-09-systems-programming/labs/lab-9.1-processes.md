# Lab 9.1: Process Management

## Objective
Manage system processes from Rust.

## Exercises

### Exercise 1: Spawning Processes
```rust
use std::process::Command;

fn main() {
    let output = Command::new("ls")
        .arg("-la")
        .output()
        .expect("Failed to execute");
    
    println!("{}", String::from_utf8_lossy(&output.stdout));
}
```

### Exercise 2: Piping Commands
```rust
let output = Command::new("echo")
    .arg("hello")
    .output()?;

let output2 = Command::new("wc")
    .arg("-c")
    .stdin(std::process::Stdio::piped())
    .output()?;
```

## Success Criteria
✅ Spawn processes  
✅ Capture output  
✅ Pipe commands

## Next Steps
Lab 9.2: Signal Handling
