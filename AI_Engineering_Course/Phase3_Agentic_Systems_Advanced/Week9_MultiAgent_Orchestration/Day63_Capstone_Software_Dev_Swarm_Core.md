# Day 63: Capstone: Building a Software Dev Swarm
## Core Concepts & Theory

### The Final Challenge

We have learned about Tools, RAG, Security, Multi-Agent Patterns, and Testing.
Now we combine them to build the "Holy Grail" of Agentic AI: **A Software Development Swarm**.
**Goal:** "Create a Snake game in Python." -> The swarm writes code, tests it, fixes bugs, and delivers a working file.

### 1. The Architecture

We will use a **Hierarchical Manager-Worker** structure with **Shared State**.

*   **Roles:**
    *   **Product Manager (PM):** Breaks the goal into a Spec.
    *   **Architect:** Defines the file structure (`game.py`, `utils.py`).
    *   **Engineer:** Writes the code.
    *   **QA Engineer:** Runs the code and reports errors.
*   **Shared State:**
    *   **File System:** The actual code files.
    *   **Blackboard:** The current "Build Status" (Passing/Failing) and "Active Task".

### 2. The Workflow (The Loop)

1.  **Spec:** PM writes `spec.md`.
2.  **Design:** Architect writes `files.json` (list of files to create).
3.  **Coding Loop:**
    *   Engineer picks a file from `files.json`.
    *   Engineer writes code.
    *   QA runs `python game.py`.
    *   **If Error:** QA posts error to Blackboard. Engineer fixes.
    *   **If Success:** Mark file as Done.
4.  **Delivery:** All files Done.

### 3. Tooling

*   **File Tools:** `write_file`, `read_file`, `list_dir`.
*   **Execution Tools:** `run_python` (Sandboxed).
*   **Communication:** AutoGen GroupChat or LangGraph State.

### 4. Key Challenges

*   **Context Management:** The Engineer needs to see `utils.py` to write `game.py`. We must handle imports correctly.
*   **Hallucinated APIs:** The Engineer might use a made-up library. QA must catch `ModuleNotFoundError`.
*   **Infinite Loops:** The Engineer might try to fix a bug, fail, try the same fix, fail... We need a "Max Retries" counter.

### 5. Success Criteria

*   **Functional:** The code runs without crashing.
*   **Completeness:** It implements the requested features (Snake moves, eats food).
*   **Cleanliness:** Code is commented and structured.

### Summary

This Capstone represents the cutting edge of AI Engineering (as of late 2024). It moves beyond "Chat" to "Work". It requires orchestrating multiple specialized intelligences to produce a complex artifact.
