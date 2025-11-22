# Lab 10.1: Command-Line Argument Parsing with clap

## Objective
Parse command-line arguments using clap.

## Exercises

### Exercise 1: Basic CLI
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

### Exercise 2: Subcommands
```rust
#[derive(Parser)]
enum Commands {
    Add { name: String },
    Remove { name: String },
    List,
}
```

## Success Criteria
✅ Parse arguments with clap  
✅ Handle options and flags  
✅ Implement subcommands

## Next Steps
Lab 10.2: Logging and Error Reporting
