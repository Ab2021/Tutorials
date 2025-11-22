# Module 10: Building CLI Applications

## 🎯 Learning Objectives

- Parse command-line arguments with clap
- Manage configuration
- Implement logging
- Handle errors gracefully
- Build complete CLI tools

---

## 📖 Core Concepts

### Argument Parsing with clap

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "mytool")]
#[command(about = "A CLI tool", long_about = None)]
struct Cli {
    /// Input file
    #[arg(short, long)]
    input: String,
    
    /// Output file
    #[arg(short, long)]
    output: Option<String>,
    
    /// Verbose mode
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();
    println!("Input: {}", cli.input);
}
```

### Configuration Management

```rust
use config::{Config, File};

let settings = Config::builder()
    .add_source(File::with_name("config"))
    .build()?;

let port: u16 = settings.get("server.port")?;
```

### Logging

```rust
use log::{info, warn, error};
use env_logger;

fn main() {
    env_logger::init();
    
    info!("Starting application");
    warn!("This is a warning");
    error!("An error occurred");
}
```

### Error Reporting

```rust
use anyhow::{Context, Result};

fn run() -> Result<()> {
    let file = fs::read_to_string("config.toml")
        .context("Failed to read config file")?;
    Ok(())
}
```

### Progress Bars

```rust
use indicatif::ProgressBar;

let pb = ProgressBar::new(100);
for _ in 0..100 {
    pb.inc(1);
    // Do work
}
pb.finish_with_message("Done!");
```

---

## 🔑 Key Takeaways

1. **clap** for argument parsing
2. **config** for configuration management
3. **log** and **env_logger** for logging
4. **anyhow** for error handling
5. **indicatif** for progress bars

**Congratulations on completing Phase 2!**  
Proceed to **Phase 3: Advanced - Production Systems**
