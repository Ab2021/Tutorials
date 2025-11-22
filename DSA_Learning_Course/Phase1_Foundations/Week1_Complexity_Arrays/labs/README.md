# Week 1 Labs: Complexity & Arrays

## Overview
This directory contains 15 hands-on labs designed to reinforce the concepts covered in Week 1: Complexity Analysis and Array Algorithms. Each lab provides practical coding exercises with complete solutions.

## Lab Structure
- **Difficulty Levels**: Easy (🟢), Medium (🟡), Hard (🔴)
- **Estimated Time**: 30 mins - 2 hours per lab
- **Format**: Problem statement → Hints → Solution (collapsible)

## Labs Index

### Complexity Analysis (Labs 1-3)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 01 | [Big-O Analysis Practice](lab_01_big_o_analysis.md) | 🟢 Easy | 30 mins | Time complexity, asymptotic notation |
| 02 | [Time Complexity Calculation](lab_02_time_complexity.md) | 🟡 Medium | 45 mins | Nested loops, recurrence relations |
| 03 | [Space Complexity Optimization](lab_03_space_optimization.md) | 🟡 Medium | 1 hour | Memory usage, in-place algorithms |

### Array Fundamentals (Labs 4-5)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 04 | [Array Rotation Algorithms](lab_04_array_rotation.md) | 🟢 Easy | 45 mins | Rotation, reversal algorithm |
| 05 | [Dutch National Flag Problem](lab_05_dutch_flag.md) | 🟡 Medium | 1 hour | Three-way partitioning |

### Two Pointers Pattern (Labs 6-7)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 06 | [Container With Most Water](lab_06_container_water.md) | 🟡 Medium | 45 mins | Two pointers, greedy approach |
| 07 | [Array Partitioning Schemes](lab_07_array_partitioning.md) | 🟡 Medium | 1 hour | Quicksort partition, Lomuto/Hoare |

### Sliding Window (Lab 8)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 08 | [Maximum Sum Subarray](lab_08_max_sum_subarray.md) | 🟢 Easy | 30 mins | Sliding window, Kadane's algorithm |

### Prefix Sum & Difference Arrays (Labs 9-10)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 09 | [Range Query Implementation](lab_09_range_query.md) | 🟢 Easy | 45 mins | Prefix sum, cumulative sum |
| 10 | [Range Update Operations](lab_10_range_update.md) | 🟡 Medium | 1 hour | Difference array, lazy propagation |

### Matrix Algorithms (Labs 11-12)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 11 | [Matrix Spiral Traversal](lab_11_spiral_matrix.md) | 🟡 Medium | 1 hour | Matrix traversal, boundary tracking |
| 12 | [Matrix Rotation In-Place](lab_12_matrix_rotation.md) | 🟡 Medium | 1 hour | In-place transformation, transpose |

### Advanced Topics (Labs 13-14)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 13 | [Kadane's Algorithm Variants](lab_13_kadane_variants.md) | 🔴 Hard | 1.5 hours | Maximum subarray, circular array |
| 14 | [Custom Dynamic Array](lab_14_dynamic_array.md) | 🟡 Medium | 1.5 hours | Amortized analysis, capacity doubling |

### Week 1 Capstone (Lab 15)
| Lab | Title | Difficulty | Time | Topics |
|-----|-------|------------|------|--------|
| 15 | [Build a Vector Class](lab_15_vector_class.md) | 🔴 Hard | 2 hours | Complete implementation, all concepts |

## Learning Path

### Recommended Order
1. **Start with Easy labs** (1, 4, 8, 9) to build confidence
2. **Progress to Medium labs** (2, 3, 5-7, 10-12, 14) for deeper understanding
3. **Challenge yourself with Hard labs** (13, 15) to master the concepts

### By Topic
- **Complexity Analysis**: Labs 1-3
- **Array Manipulation**: Labs 4-5, 11-12
- **Algorithmic Patterns**: Labs 6-10
- **Advanced Concepts**: Labs 13-15

## Prerequisites
- Completed Day 1-5 readings (Core Concepts, Deep Dive, Interview Prep)
- Basic Python programming knowledge
- Understanding of Big-O notation

## How to Use These Labs

1. **Read the Problem**: Understand requirements before coding
2. **Try Without Hints**: Attempt the solution independently
3. **Use Hints Sparingly**: Only if stuck for >15 minutes
4. **Review Solution**: Compare your approach with the provided solution
5. **Analyze Complexity**: Always calculate time/space complexity
6. **Try Extensions**: Challenge yourself with bonus problems

## Testing Your Solutions

Each lab includes test cases. Run them to verify your implementation:

```python
# Example test runner
def run_tests(function, test_cases):
    for i, (input_data, expected) in enumerate(test_cases):
        result = function(*input_data)
        assert result == expected, f"Test {i+1} failed: expected {expected}, got {result}"
    print(f"✅ All {len(test_cases)} tests passed!")
```

## Additional Resources

- **Visualizations**: [VisuAlgo](https://visualgo.net/) for algorithm animations
- **Practice**: [LeetCode Array Tag](https://leetcode.com/tag/array/)
- **Reading**: CLRS Chapter 2 (Getting Started)

## Progress Tracking

Mark labs as you complete them:
- [ ] Lab 01 - Big-O Analysis Practice
- [ ] Lab 02 - Time Complexity Calculation
- [ ] Lab 03 - Space Complexity Optimization
- [ ] Lab 04 - Array Rotation Algorithms
- [ ] Lab 05 - Dutch National Flag Problem
- [ ] Lab 06 - Container With Most Water
- [ ] Lab 07 - Array Partitioning Schemes
- [ ] Lab 08 - Maximum Sum Subarray
- [ ] Lab 09 - Range Query Implementation
- [ ] Lab 10 - Range Update Operations
- [ ] Lab 11 - Matrix Spiral Traversal
- [ ] Lab 12 - Matrix Rotation In-Place
- [ ] Lab 13 - Kadane's Algorithm Variants
- [ ] Lab 14 - Custom Dynamic Array
- [ ] Lab 15 - Build a Vector Class

## Getting Help

If you're stuck:
1. Review the relevant day's content
2. Check the hints (click to expand)
3. Study similar problems on LeetCode
4. Discuss with peers (without sharing solutions)

---

**Happy Coding! 🚀**
