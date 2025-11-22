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


### Advanced Theory: CEP Optimization & NFA
**1. NFA State Explosion**
The NFA (Nondeterministic Finite Automaton) stores partial matches.
-   **Pattern**: `A -> B`.
-   **Stream**: `A1, A2, A3 ...`.
-   **State**: Flink stores `A1`, `A2`, `A3` waiting for `B`.
-   **Risk**: If `B` never comes, state grows infinitely.
-   **Fix**: Always use `within(Time.minutes(10))` to purge old state.

**2. Skip Strategies**
When multiple matches overlap:
-   `NO_SKIP`: Find all matches. (Expensive).
-   `SKIP_PAST_LAST_EVENT`: Once a match is found, ignore overlapping events. (Faster).

**3. Iterative Conditions**
`where(ctx -> event.price > prev_event.price)`.
-   Allows comparing the current event with previously matched events in the sequence.
