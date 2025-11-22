# Day 2: Complex Event Processing (CEP)

## Core Concepts & Theory

### What is CEP?
Detecting **patterns** across a stream of events.
-   "If Event A happens, followed by Event B within 10 minutes, trigger Alert."

### Pattern API
-   `begin("start")`: Define start state.
-   `where(condition)`: Filter.
-   `next("middle")`: Strict contiguity (A immediately followed by B).
-   `followedBy("middle")`: Relaxed contiguity (A ... B).
-   `within(Time)`: Time constraint.

### Architectural Reasoning
**NFA (Nondeterministic Finite Automaton)**
Flink compiles the pattern into an NFA.
-   State is stored for every partial match.
-   If you have a pattern "A followed by B", and you get "A", Flink stores "A" in state waiting for "B".

### Key Components
-   `CEP.pattern(stream, pattern)`: Applies the pattern.
-   `PatternSelectFunction`: Extracts the result when a match is found.
