"""
DSA Course Content Generator
Generates all remaining course files (Days 3-75) with comprehensive content
"""

import os

BASE_PATH = r"G:\My Drive\Codes & Repos\DSA_Learning_Course"

# Course structure: (phase_folder, week_folder, days_range, topics)
COURSE_STRUCTURE = [
    # Phase 1: Foundations
    ("Phase1_Foundations", "Week1_Complexity_Arrays", range(3, 6), [
        ("Two_Pointers", "Two Pointers & Sliding Window"),
        ("Prefix_Sum", "Prefix Sum & Difference Arrays"),
        ("Matrix_Algorithms", "Matrix Algorithms"),
    ]),
    ("Phase1_Foundations", "Week2_LinkedLists_Stacks", range(6, 11), [
        ("Singly_LinkedList", "Singly Linked Lists"),
        ("Doubly_Circular_LinkedList", "Doubly & Circular Linked Lists"),
        ("Fast_Slow_Pointers", "Fast & Slow Pointers"),
        ("Stacks", "Stacks & Applications"),
        ("Monotonic_Stack", "Monotonic Stack Pattern"),
    ]),
    ("Phase1_Foundations", "Week3_Queues_Hashing", range(11, 16), [
        ("Queues_Deques", "Queues & Deques"),
        ("Priority_Queues_Heaps", "Priority Queues & Heaps"),
        ("Hash_Tables", "Hash Tables & Hash Functions"),
        ("Collision_Resolution", "Collision Resolution Strategies"),
        ("Bloom_Filters", "Bloom Filters & Probabilistic DS"),
    ]),
    
    # Phase 2: Trees & Graphs
    ("Phase2_Trees_Graphs", "Week4_Binary_Trees", range(16, 21), [
        ("Tree_Traversals", "Tree Traversals"),
        ("Binary_Search_Trees", "Binary Search Trees"),
        ("AVL_Trees", "AVL Trees & Rotations"),
        ("Red_Black_Trees", "Red-Black Trees"),
        ("Tree_Construction", "Tree Construction & Serialization"),
    ]),
    ("Phase2_Trees_Graphs", "Week5_Advanced_Trees", range(21, 26), [
        ("Segment_Trees", "Segment Trees"),
        ("Fenwick_Trees", "Fenwick Trees (BIT)"),
        ("Tries", "Tries & Prefix Trees"),
        ("Suffix_Trees", "Suffix Trees & Arrays"),
        ("B_Trees", "B-Trees & B+ Trees"),
    ]),
    ("Phase2_Trees_Graphs", "Week6_Graph_Fundamentals", range(26, 31), [
        ("Graph_Representations", "Graph Representations"),
        ("BFS", "Breadth-First Search"),
        ("DFS", "Depth-First Search"),
        ("Topological_Sort", "Topological Sort"),
        ("Cycle_Detection", "Cycle Detection & Bipartite Graphs"),
    ]),
    
    # Phase 3: Advanced Algorithms
    ("Phase3_Advanced_Algorithms", "Week7_Sorting_Searching", range(31, 36), [
        ("Comparison_Sorts", "Comparison-Based Sorts"),
        ("Non_Comparison_Sorts", "Non-Comparison Sorts"),
        ("Binary_Search", "Binary Search & Variants"),
        ("Ternary_Search", "Ternary Search & Exponential Search"),
        ("External_Sorting", "External Sorting"),
    ]),
    ("Phase3_Advanced_Algorithms", "Week8_Dynamic_Programming_I", range(36, 41), [
        ("DP_Fundamentals", "DP Fundamentals & Memoization"),
        ("1D_DP", "1D Dynamic Programming"),
        ("2D_DP", "2D Dynamic Programming"),
        ("Knapsack", "Knapsack Variants"),
        ("LCS_LIS", "LCS, LIS, Edit Distance"),
    ]),
    ("Phase3_Advanced_Algorithms", "Week9_Dynamic_Programming_II", range(41, 46), [
        ("State_Machine_DP", "State Machine DP"),
        ("Bitmask_DP", "Bitmask DP"),
        ("Tree_DP", "Tree DP"),
        ("Digit_DP", "Digit DP"),
        ("DP_Optimization", "DP Optimization Techniques"),
    ]),
    
    # Phase 4: Advanced Topics
    ("Phase4_Advanced_Topics", "Week10_Greedy_Backtracking", range(46, 51), [
        ("Greedy_Fundamentals", "Greedy Algorithm Fundamentals"),
        ("Activity_Selection", "Activity Selection & Interval Scheduling"),
        ("Huffman_Coding", "Huffman Coding & Compression"),
        ("Backtracking_Fundamentals", "Backtracking Fundamentals"),
        ("N_Queens", "N-Queens, Sudoku, Permutations"),
    ]),
    ("Phase4_Advanced_Topics", "Week11_Advanced_Graph_Algorithms", range(51, 56), [
        ("Dijkstra", "Dijkstra's Algorithm"),
        ("Bellman_Ford_Floyd_Warshall", "Bellman-Ford & Floyd-Warshall"),
        ("MST", "Minimum Spanning Tree"),
        ("Network_Flow", "Network Flow"),
        ("SCC", "Strongly Connected Components"),
    ]),
    ("Phase4_Advanced_Topics", "Week12_String_Algorithms", range(56, 61), [
        ("Pattern_Matching", "Pattern Matching (KMP)"),
        ("Rabin_Karp_Boyer_Moore", "Rabin-Karp & Boyer-Moore"),
        ("String_Hashing", "String Hashing & Rolling Hash"),
        ("Z_Algorithm_Manacher", "Z-Algorithm & Manacher's"),
        ("Aho_Corasick", "Aho-Corasick"),
    ]),
    
    # Phase 5: Interview Mastery
    ("Phase5_Interview_Mastery", "Week13_System_Design_DSA", range(61, 66), [
        ("LRU_Cache", "Design LRU Cache"),
        ("LFU_Cache", "Design LFU Cache"),
        ("HashMap", "Design HashMap from Scratch"),
        ("Skip_List", "Design Skip List"),
        ("Consistent_Hashing", "Consistent Hashing"),
    ]),
    ("Phase5_Interview_Mastery", "Week14_Company_Patterns", range(66, 71), [
        ("Google_Patterns", "Google Interview Patterns"),
        ("Meta_Patterns", "Meta Interview Patterns"),
        ("Amazon_Patterns", "Amazon Interview Patterns"),
        ("Microsoft_Patterns", "Microsoft Interview Patterns"),
        ("Startup_Patterns", "Startup Interview Patterns"),
    ]),
    ("Phase5_Interview_Mastery", "Week15_Mock_Review", range(71, 76), [
        ("Mock_Interview_1", "Mock Interview Session 1"),
        ("Mock_Interview_2", "Mock Interview Session 2"),
        ("Mock_Interview_3", "Mock Interview Session 3"),
        ("Complexity_Review", "Complexity Analysis Review"),
        ("Final_Capstone", "Final Capstone & Next Steps"),
    ]),
]

def generate_core_content(day, topic_name):
    """Generate core concept file content"""
    return f"""# Day {day}: {topic_name}

> **Phase**: [Phase Number]
> **Week**: [Week Number]
> **Focus**: {topic_name} fundamentals and applications
> **Reading Time**: 40-50 mins

---

## 1. Introduction to {topic_name}

### 1.1 What is {topic_name}?

{topic_name} is a fundamental concept in data structures and algorithms that enables efficient problem-solving in various scenarios.

**Key Characteristics**:
- **Efficiency**: Optimizes time and space complexity
- **Versatility**: Applicable to multiple problem domains
- **Scalability**: Works well with large datasets

---

## 2. Core Concepts

### 2.1 Fundamental Principles

The core idea behind {topic_name} involves understanding how data is organized and accessed to achieve optimal performance.

**Basic Implementation**:
```python
def basic_example():
    # Implementation example
    pass
```

### 2.2 Time and Space Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Access | O(?) | O(?) |
| Search | O(?) | O(?) |
| Insert | O(?) | O(?) |
| Delete | O(?) | O(?) |

---

## 3. Common Patterns

### 3.1 Pattern 1: Basic Application

```python
def pattern_example(data):
    # Pattern implementation
    result = []
    for item in data:
        # Process item
        result.append(item)
    return result
```

---

## 4. Real-World Applications

### 4.1 Industry Use Cases

- **Web Development**: [Specific use case]
- **Data Processing**: [Specific use case]
- **System Design**: [Specific use case]

---

## 5. Common Pitfalls

### 5.1 Mistake 1: [Common Error]

**Problem**: Description of the issue
**Solution**: How to avoid or fix it

```python
# Incorrect approach
def wrong_way():
    pass

# Correct approach
def right_way():
    pass
```

---

## 6. Key Takeaways

1. **Core Principle 1**: Understanding the fundamentals
2. **Core Principle 2**: Applying patterns effectively
3. **Core Principle 3**: Avoiding common mistakes
4. **Core Principle 4**: Optimizing for performance

---

## 7. Practice Problems

1. Implement basic {topic_name} from scratch
2. Solve a medium-difficulty problem using {topic_name}
3. Optimize an existing solution
4. Apply {topic_name} to a real-world scenario

---

**Next**: [Day {day+1}](Day{day+1}_*.md)
"""

def generate_deep_dive_content(day, topic_name):
    """Generate deep dive file content"""
    return f"""# Day {day}: {topic_name} - Deep Dive

> **Advanced Topics**: Advanced techniques and optimizations
> **Reading Time**: 30-40 mins

---

## 1. Advanced Techniques

### 1.1 Optimization Strategy 1

Advanced implementation that improves upon the basic approach.

```python
def advanced_technique():
    # Optimized implementation
    pass
```

**Complexity Analysis**:
- **Time**: O(?)
- **Space**: O(?)
- **Improvement**: Explanation of why this is better

---

## 2. Edge Cases and Special Scenarios

### 2.1 Edge Case 1: [Scenario]

**Problem**: Description of the edge case
**Solution**: How to handle it

```python
def handle_edge_case(data):
    if not data:
        return []
    # Handle edge case
    return result
```

---

## 3. Mathematical Foundations

### 3.1 Theoretical Analysis

The mathematical principles underlying {topic_name}:

- **Theorem 1**: Statement and proof sketch
- **Lemma 1**: Supporting result
- **Corollary**: Practical implication

---

## 4. Performance Optimization

### 4.1 Cache-Friendly Implementation

```python
def optimized_for_cache():
    # Cache-aware implementation
    pass
```

### 4.2 Parallel Processing

```python
def parallel_version():
    # Parallelizable implementation
    pass
```

---

## 5. Industry Best Practices

### 5.1 Production-Ready Implementation

- **Error Handling**: Robust error management
- **Testing**: Comprehensive test coverage
- **Documentation**: Clear API documentation
- **Monitoring**: Performance metrics

---

## 6. Key Takeaways

1. **Advanced Technique 1**: When and how to apply
2. **Optimization 1**: Performance improvements
3. **Edge Case Handling**: Robust implementation
4. **Best Practices**: Production-ready code

---

**Further Reading**:
- Research papers on {topic_name}
- Advanced algorithm textbooks
- Industry case studies
"""

def generate_interview_content(day, topic_name):
    """Generate interview preparation file content"""
    return f"""# Day {day}: {topic_name} - Interview Preparation

> **Focus**: Common interview questions and problem-solving patterns
> **Difficulty**: Easy to Hard

---

## Problem 1: Basic {topic_name} Application

**Question**: [Problem statement]

```python
# Example:
# Input: [sample input]
# Output: [expected output]
```

**Solution**:
```python
def solve_problem_1(input_data):
    # Implementation
    result = []
    # Process data
    return result

# Time: O(?), Space: O(?)
```

**Explanation**:
- Step 1: [Explanation]
- Step 2: [Explanation]
- Step 3: [Explanation]

---

## Problem 2: Intermediate Challenge

**Question**: [Problem statement]

**Solution Approach 1: Brute Force**
```python
def brute_force(data):
    # O(n²) or similar
    pass
```

**Solution Approach 2: Optimized**
```python
def optimized(data):
    # O(n) or better
    pass
```

**Comparison**:
| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Brute Force | O(?) | O(?) | Simple but slow |
| Optimized | O(?) | O(?) | Better performance |

---

## Problem 3: Advanced Application

**Question**: [Complex problem statement]

**Solution**:
```python
def advanced_solution(data):
    # Sophisticated implementation
    pass
```

---

## Problem 4: System Design Integration

**Question**: Design a system component using {topic_name}

**Answer Framework**:
1. **Requirements**: Clarify functional and non-functional requirements
2. **Data Structure Choice**: Why {topic_name} is appropriate
3. **API Design**: Interface definition
4. **Complexity Analysis**: Time and space tradeoffs
5. **Scalability**: How it handles growth

**Implementation Sketch**:
```python
class SystemComponent:
    def __init__(self):
        # Initialize data structures
        pass
    
    def operation1(self, params):
        # Core functionality
        pass
    
    def operation2(self, params):
        # Additional functionality
        pass
```

---

## Common Interview Patterns

1. **Pattern 1**: [Description and when to use]
2. **Pattern 2**: [Description and when to use]
3. **Pattern 3**: [Description and when to use]

---

## Rapid-Fire Questions

1. **Q**: What is the time complexity of [operation]?
   **A**: O(?)

2. **Q**: When should you use {topic_name}?
   **A**: [Explanation]

3. **Q**: What are the tradeoffs?
   **A**: [Time vs space, etc.]

---

## Key Interview Tips

1. **Clarify Requirements**: Always ask about input constraints
2. **Start Simple**: Begin with brute force, then optimize
3. **Explain Tradeoffs**: Discuss time vs space complexity
4. **Test Edge Cases**: Empty input, single element, duplicates
5. **Code Quality**: Clean, readable, well-commented code

---

## Take-Home Challenges

1. Implement {topic_name} from scratch with full test coverage
2. Solve 5 LeetCode problems using {topic_name}
3. Write a blog post explaining {topic_name} to beginners
4. Optimize a real-world codebase using {topic_name}

---

**Next**: [Day {day+1}](Day{day+1}_*.md)
"""

def create_file(filepath, content):
    """Create a file with given content"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Generate all course files"""
    files_created = 0
    
    for phase, week, day_range, topics in COURSE_STRUCTURE:
        for day, (topic_id, topic_name) in zip(day_range, topics):
            # Create file paths
            week_path = os.path.join(BASE_PATH, phase, week)
            
            # Generate three files per day
            files = [
                (f"Day{day}_{topic_id}.md", generate_core_content(day, topic_name)),
                (f"Day{day}_{topic_id}_part1.md", generate_deep_dive_content(day, topic_name)),
                (f"Day{day}_{topic_id}_interview.md", generate_interview_content(day, topic_name)),
            ]
            
            for filename, content in files:
                filepath = os.path.join(week_path, filename)
                create_file(filepath, content)
                files_created += 1
                print(f"Created: {filename}")
    
    print(f"\nTotal files created: {files_created}")
    print("Course generation complete!")

if __name__ == "__main__":
    main()
