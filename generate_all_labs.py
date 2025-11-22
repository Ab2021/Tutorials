"""
Lab Generator Script for All 6 Courses
Generates 10-20 labs per week across all course directories
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

# Base directory
BASE_DIR = Path(r"G:\My Drive\Codes & Repos")

# Course configurations
COURSES = {
    "DSA_Learning_Course": {
        "weeks": [
            ("Phase1_Foundations", "Week1_Complexity_Arrays", "Complexity & Arrays"),
            ("Phase1_Foundations", "Week2_LinkedLists_Stacks", "Linked Lists & Stacks"),
            ("Phase1_Foundations", "Week3_Queues_Hashing", "Queues & Hashing"),
            ("Phase2_Trees_Graphs", "Week4_Binary_Trees", "Binary Trees"),
            ("Phase2_Trees_Graphs", "Week5_Advanced_Trees", "Advanced Trees"),
            ("Phase2_Trees_Graphs", "Week6_Graph_Fundamentals", "Graph Fundamentals"),
            ("Phase3_Advanced_Algorithms", "Week7_Sorting_Searching", "Sorting & Searching"),
            ("Phase3_Advanced_Algorithms", "Week8_Dynamic_Programming_I", "Dynamic Programming I"),
            ("Phase3_Advanced_Algorithms", "Week9_Dynamic_Programming_II", "Dynamic Programming II"),
            ("Phase4_Advanced_Topics", "Week10_Greedy_Backtracking", "Greedy & Backtracking"),
            ("Phase4_Advanced_Topics", "Week11_Advanced_Graph", "Advanced Graph Algorithms"),
            ("Phase4_Advanced_Topics", "Week12_String_Algorithms", "String Algorithms"),
            ("Phase5_Interview_Mastery", "Week13_System_Design_DSA", "System Design with DSA"),
            ("Phase5_Interview_Mastery", "Week14_Company_Patterns", "Company-Specific Patterns"),
            ("Phase5_Interview_Mastery", "Week15_Mock_Interviews", "Mock Interviews & Review"),
        ]
    },
    "Computer_Vision_Course": {
        "weeks": [
            ("Phase1_Foundations", "Week1_ImageBasics", "Image Basics & Classical CV"),
            ("Phase1_Foundations", "Week2_DeepLearning", "Deep Learning Foundations"),
            ("Phase2_Classification", "Week3_Classic_Architectures", "Classic Architectures"),
            ("Phase2_Classification", "Week4_Advanced_Recognition", "Advanced Recognition"),
            ("Phase3_Detection", "Week5_Object_Detection", "Object Detection"),
            ("Phase3_Detection", "Week6_Segmentation", "Segmentation"),
            ("Phase4_Generative", "Week7_Generative_Models", "Generative Models"),
            ("Phase4_Generative", "Week8_Frontiers", "Frontiers & Applications"),
        ]
    },
    "ML_System_Design_Course": {
        "weeks": [
            ("Phase1_Foundations", "Week1_Python_Environment", "Python Environment & Tools"),
            ("Phase1_Foundations", "Week2_Math_Foundations", "Mathematical Foundations"),
            ("Phase2_Core_Algorithms", "Week3_Supervised_Learning", "Supervised Learning"),
            ("Phase2_Core_Algorithms", "Week4_Unsupervised_DeepLearning", "Unsupervised & Deep Learning"),
            ("Phase3_System_Design", "Week5_Design_Principles", "ML System Design Principles"),
            ("Phase3_System_Design", "Week6_Data_Engineering", "Data Engineering"),
            ("Phase4_LLMs_GenAI", "Week7_LLM_Foundations", "LLM Foundations"),
            ("Phase4_LLMs_GenAI", "Week8_LLM_Systems", "LLM Systems"),
            ("Phase5_Interview_Mastery", "Week9_Behavioral_Coding", "Behavioral & Coding"),
            ("Phase5_Interview_Mastery", "Week10_Mock_Interviews", "Mock Interviews"),
        ]
    },
    "PyTorch_Deep_Learning_Course": {
        "weeks": [
            ("Phase1_Foundations", "Week1_PyTorch_Basics", "PyTorch Basics"),
            ("Phase1_Foundations", "Week2_Data_Workflow", "Data & Workflow"),
            ("Phase2_Computer_Vision", "Week3_CNNs", "CNNs & Architectures"),
            ("Phase2_Computer_Vision", "Week4_Generative_Deployment", "Generative & Deployment"),
            ("Phase3_NLP_Transformers", "Week5_Sequences", "Sequences & RNNs"),
            ("Phase3_NLP_Transformers", "Week6_LLMs", "Transformers & LLMs"),
            ("Phase4_Advanced_Topics", "Week7_Modern_AI", "Modern AI Topics"),
            ("Phase4_Advanced_Topics", "Week8_Systems_Capstone", "Systems & Capstone"),
        ]
    },
    "System_Design_Course": {
        "weeks": [
            ("Phase1_Foundations", "labs", "Foundations (Days 1-10)"),
            ("Phase2_BuildingBlocks", "labs", "Building Blocks (Days 11-20)"),
            ("Phase3_AdvancedArchitectures", "labs", "Advanced Architectures (Days 21-30)"),
            ("Phase4_CaseStudies", "labs", "Case Studies (Days 31-40)"),
        ]
    },
    "Reinforcement_Learning_Course": {
        "weeks": [
            ("Phase1_Foundations", "labs", "RL Foundations (Days 1-10)"),
            ("Phase2_ValueBased_DeepRL", "labs", "Value-Based Deep RL (Days 11-20)"),
            ("Phase3_PolicyBased", "labs", "Policy-Based Methods (Days 21-30)"),
            ("Phase4_Advanced", "labs", "Advanced Topics (Days 31-40)"),
        ]
    },
}

# Lab templates for different difficulty levels
LAB_TEMPLATE = """# Lab {num:02d}: {title}

## Difficulty
{difficulty}

## Estimated Time
{time}

## Learning Objectives
{objectives}

## Prerequisites
{prerequisites}

## Problem Statement

{problem_statement}

## Requirements

{requirements}

## Starter Code

```python
{starter_code}
```

## Hints

<details>
<summary>Hint 1</summary>

{hint1}
</details>

<details>
<summary>Hint 2</summary>

{hint2}
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Approach

{approach_description}

```python
{solution_code}
```

### Time Complexity
{time_complexity}

### Space Complexity
{space_complexity}

### Explanation
{explanation}

</details>

## Extensions

{extensions}

## Related Concepts
{related_concepts}

---

**Next**: [Lab {next_num:02d}](lab_{next_num:02d}_{next_file}.md)
"""

def create_labs_for_week(course_name: str, phase: str, week: str, week_title: str, lab_count: int = 15):
    """Generate labs for a specific week."""
    
    # Create labs directory
    if "labs" in week:
        labs_dir = BASE_DIR / course_name / phase / week
    else:
        labs_dir = BASE_DIR / course_name / phase / week / "labs"
    
    labs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {lab_count} labs for {course_name}/{phase}/{week}...")
    
    # Get lab topics based on course and week
    lab_topics = get_lab_topics(course_name, week_title, lab_count)
    
    # Create README
    create_readme(labs_dir, week_title, lab_topics)
    
    # Create individual lab files
    for i, topic in enumerate(lab_topics, 1):
        create_lab_file(labs_dir, i, topic, lab_count)
    
    print(f"✅ Created {lab_count} labs in {labs_dir}")

def get_lab_topics(course_name: str, week_title: str, count: int) -> List[Dict]:
    """Generate lab topics based on course and week."""
    
    # This is a simplified version - in practice, you'd have detailed topic lists
    topics = []
    
    difficulties = ["🟢 Easy", "🟡 Medium", "🔴 Hard"]
    times = ["30 mins", "45 mins", "1 hour", "1.5 hours", "2 hours"]
    
    for i in range(count):
        # Distribute difficulties: 40% easy, 40% medium, 20% hard
        if i < count * 0.4:
            difficulty = difficulties[0]
            time = times[0] if i % 2 == 0 else times[1]
        elif i < count * 0.8:
            difficulty = difficulties[1]
            time = times[1] if i % 2 == 0 else times[2]
        else:
            difficulty = difficulties[2]
            time = times[3] if i % 2 == 0 else times[4]
        
        topics.append({
            "title": f"{week_title} - Exercise {i+1}",
            "difficulty": difficulty,
            "time": time,
            "objectives": f"- Master concept {i+1} from {week_title}\n- Apply techniques in practical scenarios\n- Optimize solutions for efficiency",
            "prerequisites": f"- Completed {week_title} readings\n- Understanding of core concepts",
        })
    
    return topics

def create_readme(labs_dir: Path, week_title: str, topics: List[Dict]):
    """Create README.md for the labs directory."""
    
    readme_content = f"""# {week_title} - Labs

## Overview
This directory contains {len(topics)} hands-on labs for {week_title}.

## Labs Index

| Lab | Title | Difficulty | Time |
|-----|-------|------------|------|
"""
    
    for i, topic in enumerate(topics, 1):
        readme_content += f"| {i:02d} | [Lab {i:02d}](lab_{i:02d}.md) | {topic['difficulty']} | {topic['time']} |\n"
    
    readme_content += """
## How to Use

1. Read the problem statement carefully
2. Attempt the solution independently
3. Use hints if stuck
4. Review the solution and compare approaches
5. Complete the extensions for extra practice

## Progress Tracking

"""
    
    for i in range(1, len(topics) + 1):
        readme_content += f"- [ ] Lab {i:02d}\n"
    
    readme_path = labs_dir / "README.md"
    readme_path.write_text(readme_content, encoding='utf-8')

def create_lab_file(labs_dir: Path, num: int, topic: Dict, total_labs: int):
    """Create individual lab markdown file."""
    
    lab_content = f"""# Lab {num:02d}: {topic['title']}

## Difficulty
{topic['difficulty']}

## Estimated Time
{topic['time']}

## Learning Objectives
{topic['objectives']}

## Prerequisites
{topic['prerequisites']}

## Problem Statement

[Detailed problem description will be added here]

## Requirements

1. Implement the core functionality
2. Handle edge cases
3. Optimize for time and space complexity
4. Write clean, documented code

## Starter Code

```python
def solution():
    \"\"\"
    TODO: Implement your solution here
    \"\"\"
    pass

# Test cases
def test_solution():
    # Add test cases here
    pass
```

## Hints

<details>
<summary>Hint 1</summary>

Consider the time complexity of your approach. Can you optimize it?
</details>

<details>
<summary>Hint 2</summary>

Think about edge cases: empty inputs, single elements, duplicates, etc.
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Approach

[Solution approach will be detailed here]

```python
def solution_optimized():
    \"\"\"
    Optimized solution with explanation
    \"\"\"
    pass
```

### Time Complexity
O(n) - [Explanation]

### Space Complexity
O(1) - [Explanation]

### Explanation
[Detailed walkthrough of the solution]

</details>

## Extensions

1. Extend the problem to handle [variation 1]
2. Optimize for [specific constraint]
3. Implement [alternative approach]

## Related Concepts
- Related topic 1
- Related topic 2

---

"""
    
    if num < total_labs:
        lab_content += f"**Next**: [Lab {num+1:02d}](lab_{num+1:02d}.md)\n"
    else:
        lab_content += "**Congratulations!** You've completed all labs for this week! 🎉\n"
    
    lab_path = labs_dir / f"lab_{num:02d}.md"
    lab_path.write_text(lab_content, encoding='utf-8')

def main():
    """Main function to generate all labs."""
    
    print("=" * 60)
    print("Lab Generator for All 6 Courses")
    print("=" * 60)
    print()
    
    total_labs = 0
    
    for course_name, config in COURSES.items():
        print(f"\n📚 Processing {course_name}...")
        print("-" * 60)
        
        for phase, week, week_title in config["weeks"]:
            # Determine lab count (15 for most, can vary)
            lab_count = 15
            
            create_labs_for_week(course_name, phase, week, week_title, lab_count)
            total_labs += lab_count
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE! Generated {total_labs} labs across all courses")
    print("=" * 60)

if __name__ == "__main__":
    main()
