# Lab 3.4: Building a Word Counter

## Objective
Build a complete word counter application that analyzes text files, demonstrating ownership, borrowing, and slices in a real-world project.

## Project Overview
Create a command-line tool that:
- Counts total words in text
- Counts unique words
- Finds most frequent words
- Calculates average word length
- Handles file input

## Setup
```bash
cargo new word_counter
cd word_counter
```

## Part 1: Basic Word Counting

### Step 1: Count Total Words
```rust
fn count_words(text: &str) -> usize {
    // TODO: Count total words (split by whitespace)
}

fn main() {
    let text = "The quick brown fox jumps over the lazy dog";
    println!("Total words: {}", count_words(text));
}
```

### Step 2: Count Unique Words
```rust
use std::collections::HashSet;

fn count_unique_words(text: &str) -> usize {
    // TODO: Count unique words (case-insensitive)
    // Hint: Use HashSet
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog";
    println!("Unique words: {}", count_unique_words(text));
}
```

## Part 2: Word Frequency Analysis

### Step 3: Word Frequency Map
```rust
use std::collections::HashMap;

fn word_frequency(text: &str) -> HashMap<String, usize> {
    // TODO: Create a map of word -> count
    // Convert to lowercase for case-insensitive counting
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox";
    let freq = word_frequency(text);
    
    for (word, count) in &freq {
        println!("{}: {}", word, count);
    }
}
```

### Step 4: Find Most Common Words
```rust
fn most_common_words(text: &str, n: usize) -> Vec<(String, usize)> {
    // TODO: Return top N most frequent words
    // Return as vector of (word, count) tuples, sorted by count
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox the";
    let top_words = most_common_words(text, 3);
    
    println!("Top 3 words:");
    for (word, count) in top_words {
        println!("  {}: {}", word, count);
    }
}
```

## Part 3: Text Statistics

### Step 5: Average Word Length
```rust
fn average_word_length(text: &str) -> f64 {
    // TODO: Calculate average length of words
}

fn main() {
    let text = "The quick brown fox";
    println!("Average word length: {:.2}", average_word_length(text));
}
```

### Step 6: Longest and Shortest Words
```rust
fn longest_word(text: &str) -> Option<&str> {
    // TODO: Find longest word
}

fn shortest_word(text: &str) -> Option<&str> {
    // TODO: Find shortest word
}

fn main() {
    let text = "The quick brown fox jumps";
    
    if let Some(longest) = longest_word(text) {
        println!("Longest word: {}", longest);
    }
    
    if let Some(shortest) = shortest_word(text) {
        println!("Shortest word: {}", shortest);
    }
}
```

## Part 4: Complete Application

### Step 7: Text Analyzer Struct
```rust
use std::collections::HashMap;

struct TextAnalyzer {
    text: String,
}

impl TextAnalyzer {
    fn new(text: String) -> Self {
        TextAnalyzer { text }
    }
    
    fn word_count(&self) -> usize {
        // TODO: Implement
    }
    
    fn unique_word_count(&self) -> usize {
        // TODO: Implement
    }
    
    fn word_frequency(&self) -> HashMap<String, usize> {
        // TODO: Implement
    }
    
    fn average_word_length(&self) -> f64 {
        // TODO: Implement
    }
    
    fn print_report(&self) {
        println!("=== Text Analysis Report ===");
        println!("Total words: {}", self.word_count());
        println!("Unique words: {}", self.unique_word_count());
        println!("Average word length: {:.2}", self.average_word_length());
        
        println!("\nTop 5 most common words:");
        // TODO: Print top 5 words
    }
}

fn main() {
    let text = String::from("Your sample text here...");
    let analyzer = TextAnalyzer::new(text);
    analyzer.print_report();
}
```

## Part 5: File Input (Advanced)

### Step 8: Read from File
```rust
use std::fs;
use std::io;

fn read_file(filename: &str) -> io::Result<String> {
    fs::read_to_string(filename)
}

fn main() -> io::Result<()> {
    let contents = read_file("sample.txt")?;
    let analyzer = TextAnalyzer::new(contents);
    analyzer.print_report();
    Ok(())
}
```

## Complete Solution

<details>
<summary>Click to reveal complete solution</summary>

```rust
use std::collections::HashMap;
use std::fs;
use std::io;

struct TextAnalyzer {
    text: String,
}

impl TextAnalyzer {
    fn new(text: String) -> Self {
        TextAnalyzer { text }
    }
    
    fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
    
    fn unique_word_count(&self) -> usize {
        let words: std::collections::HashSet<_> = self.text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        words.len()
    }
    
    fn word_frequency(&self) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        
        for word in self.text.split_whitespace() {
            let word = word.to_lowercase();
            *freq.entry(word).or_insert(0) += 1;
        }
        
        freq
    }
    
    fn average_word_length(&self) -> f64 {
        let words: Vec<&str> = self.text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let total_length: usize = words.iter().map(|w| w.len()).sum();
        total_length as f64 / words.len() as f64
    }
    
    fn most_common_words(&self, n: usize) -> Vec<(String, usize)> {
        let freq = self.word_frequency();
        let mut freq_vec: Vec<_> = freq.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        freq_vec.into_iter().take(n).collect()
    }
    
    fn longest_word(&self) -> Option<String> {
        self.text
            .split_whitespace()
            .max_by_key(|w| w.len())
            .map(|s| s.to_string())
    }
    
    fn shortest_word(&self) -> Option<String> {
        self.text
            .split_whitespace()
            .min_by_key(|w| w.len())
            .map(|s| s.to_string())
    }
    
    fn print_report(&self) {
        println!("=== Text Analysis Report ===\n");
        println!("Total words: {}", self.word_count());
        println!("Unique words: {}", self.unique_word_count());
        println!("Average word length: {:.2} characters", self.average_word_length());
        
        if let Some(longest) = self.longest_word() {
            println!("Longest word: {} ({} chars)", longest, longest.len());
        }
        
        if let Some(shortest) = self.shortest_word() {
            println!("Shortest word: {} ({} chars)", shortest, shortest.len());
        }
        
        println!("\nTop 5 most common words:");
        for (i, (word, count)) in self.most_common_words(5).iter().enumerate() {
            println!("  {}. {} (appears {} times)", i + 1, word, count);
        }
    }
}

fn read_file(filename: &str) -> io::Result<String> {
    fs::read_to_string(filename)
}

fn main() -> io::Result<()> {
    // Sample text for testing
    let sample_text = String::from(
        "The quick brown fox jumps over the lazy dog. \
         The dog was really lazy, and the fox was very quick. \
         Quick brown foxes are amazing animals."
    );
    
    println!("Analyzing sample text...\n");
    let analyzer = TextAnalyzer::new(sample_text);
    analyzer.print_report();
    
    // Uncomment to read from file:
    // println!("\n\nAnalyzing file...\n");
    // let contents = read_file("sample.txt")?;
    // let file_analyzer = TextAnalyzer::new(contents);
    // file_analyzer.print_report();
    
    Ok(())
}
```

</details>

## Expected Output

```
=== Text Analysis Report ===

Total words: 23
Unique words: 15
Average word length: 4.17 characters
Longest word: amazing (7 chars)
Shortest word: was (3 chars)

Top 5 most common words:
  1. the (4 times)
  2. fox (2 times)
  3. quick (2 times)
  4. was (2 times)
  5. dog (2 times)
```

## Challenges

### Challenge 1: Filter Stop Words
Create a list of common words (the, a, an, is, was) and exclude them from frequency analysis.

### Challenge 2: Sentence Counter
Add functionality to count sentences (split by . ! ?).

### Challenge 3: Reading Level
Calculate a simple reading level score based on average word length and sentence length.

### Challenge 4: Command-Line Arguments
Accept filename as command-line argument using `std::env::args()`.

## Success Criteria
✅ All basic functions implemented  
✅ TextAnalyzer struct works correctly  
✅ Can analyze text and produce report  
✅ Proper use of ownership and borrowing  
✅ No unnecessary clones  
✅ At least one challenge completed

## Key Learnings
- Using HashMap for frequency counting
- Borrowing vs ownership in struct methods
- Working with iterators for text processing
- File I/O with error handling
- Building a complete application structure

## Next Steps
Congratulations on completing Module 3! You now understand:
- Ownership and move semantics
- References and borrowing rules
- String slices and manipulation
- Building real applications with these concepts

Move on to **Module 4: Structs and Enums**!
