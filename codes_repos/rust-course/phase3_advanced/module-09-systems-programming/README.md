# Module 09: Systems Programming

## 🎯 Learning Objectives

- Interact with OS interfaces
- Manage processes
- Handle signals
- Use memory-mapped I/O
- Build system-level tools

---

## 📖 Core Concepts

### Process Management

```rust
use std::process::Command;

let output = Command::new("ls")
    .arg("-la")
    .output()
    .expect("Failed to execute");

println!("{}", String::from_utf8_lossy(&output.stdout));
```

### Signal Handling

```rust
use signal_hook::{consts::SIGINT, iterator::Signals};

let mut signals = Signals::new(&[SIGINT])?;

for sig in signals.forever() {
    println!("Received signal {:?}", sig);
}
```

### Memory-Mapped I/O

```rust
use memmap2::MmapOptions;
use std::fs::File;

let file = File::open("large_file.dat")?;
let mmap = unsafe { MmapOptions::new().map(&file)? };

// Access file as byte slice
let data = &mmap[0..100];
```

### File System Operations

```rust
use std::fs;

fs::create_dir_all("path/to/dir")?;
fs::remove_dir_all("path/to/dir")?;
fs::copy("source", "dest")?;
```

---

## 🔑 Key Takeaways

1. **std::process** for process management
2. **Signal handling** for graceful shutdown
3. **Memory-mapped I/O** for large files
4. **File system** operations
5. **Platform-specific** code when needed

Complete 10 labs, then proceed to Module 10: Production Deployment
