# Day 2: CEP - Deep Dive

## Deep Dive & Internals

### Contiguity Types
1.  **Strict (`next`)**: A, B. Input: A, C, B. Match: No.
2.  **Relaxed (`followedBy`)**: A, B. Input: A, C, B. Match: Yes (skips C).
3.  **Non-Deterministic Relaxed (`followedByAny`)**: A, B. Input: A, C, B1, B2. Matches: (A, B1) AND (A, B2).

### After Match Skip Strategy
What happens after a match?
-   **NO_SKIP**: All possible matches. (Expensive).
-   **SKIP_PAST_LAST_EVENT**: Discard partial matches that overlap.
-   **SKIP_TO_NEXT**: Jump to the next start.

### Advanced Reasoning
**CEP vs SQL MATCH_RECOGNIZE**
Flink SQL supports `MATCH_RECOGNIZE` (standard SQL CEP).
-   **SQL**: Standard, declarative, easier to optimize.
-   **CEP Library**: More flexible (imperative conditions), allows complex Java/Python logic in conditions.

### Performance Implications
-   **State Explosion**: A pattern like "A followed by B" with no time limit will store "A" forever. ALWAYS use `.within(Time)`.
