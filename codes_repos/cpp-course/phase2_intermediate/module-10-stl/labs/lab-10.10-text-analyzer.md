# Lab 10.10: Text Frequency Analyzer (Capstone)

## Objective
Build a tool that reads text, counts word frequencies, filters common words (stop words), and prints the top N most frequent words.

## Instructions

### Step 1: Setup
Create `analyzer.cpp`.
Include `<iostream>`, `<vector>`, `<map>`, `<algorithm>`, `<string>`, `<sstream>`, `<set>`.

### Step 2: Stop Words
Use a `std::set<string>` for fast lookup of words to ignore (e.g., "the", "a", "and").

```cpp
std::set<std::string> stopWords = {"the", "a", "an", "and", "or", "but"};
```

### Step 3: Processing
1. Read text (hardcoded or cin).
2. Convert to lowercase.
3. Remove punctuation.
4. Check if in stop words.
5. Update count in `map<string, int>`.

### Step 4: Sorting
Maps are sorted by Key. We need to sort by Value (Frequency).
Copy map content to a `vector<pair<string, int>>`.
Sort the vector using a lambda.

## Challenges

### Challenge 1: Top N
Print only the top 3 words.

### Challenge 2: C++20 Ranges
Try to implement the "lowercase and remove punctuation" step using `std::views::transform`.

## Solution

<details>
<summary>Click to reveal solution</summary>

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <cctype>

// Helper to clean string
std::string clean(const std::string& s) {
    std::string res;
    for(char c : s) {
        if(std::isalpha(c)) res += std::tolower(c);
    }
    return res;
}

int main() {
    std::string text = "The quick brown fox jumps over the lazy dog. The dog was not amused.";
    std::set<std::string> stopWords = {"the", "a", "an", "was", "not", "over"};
    std::map<std::string, int> counts;
    
    std::stringstream ss(text);
    std::string word;
    while(ss >> word) {
        word = clean(word);
        if(word.empty() || stopWords.count(word)) continue;
        counts[word]++;
    }
    
    // Move to vector for sorting
    using Pair = std::pair<std::string, int>;
    std::vector<Pair> sorted(counts.begin(), counts.end());
    
    std::sort(sorted.begin(), sorted.end(), [](const Pair& a, const Pair& b) {
        return a.second > b.second; // Descending freq
    });
    
    std::cout << "Top Words:\n";
    for(int i=0; i<3 && i<sorted.size(); ++i) {
        std::cout << sorted[i].first << ": " << sorted[i].second << "\n";
    }
    
    return 0;
}
```
</details>

## Success Criteria
✅ Used `map` for counting
✅ Used `set` for filtering
✅ Used `vector` + `sort` for ranking
✅ Cleaned and processed strings

## Key Learnings
- Real-world data processing often requires multiple containers
- Maps are great for counting, Vectors are great for sorting
- STL makes complex tasks (like this analyzer) concise

## Next Steps
Congratulations! You've completed Module 10.

Proceed to **Module 11: Error Handling and Exceptions**.
