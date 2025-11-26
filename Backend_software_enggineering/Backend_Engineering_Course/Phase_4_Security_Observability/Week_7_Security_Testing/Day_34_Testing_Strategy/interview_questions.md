# Day 34: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between a Mock and a Stub?
**Answer:**
*   **Stub**: Provides canned answers to calls. (e.g., "When `get_user(1)` is called, return `{'name': 'Alice'}`").
*   **Mock**: Verifies *behavior*. (e.g., "Assert that `send_email` was called exactly once with argument `hello@example.com`").

### Q2: Why are "Flaky Tests" dangerous?
**Answer:**
*   **Definition**: A test that passes sometimes and fails sometimes (without code changes).
*   **Danger**: Developers lose trust in the test suite. They start ignoring failures ("Oh, that's just the flaky test").
*   **Causes**: Race conditions, relying on external APIs (network), random ordering of tests.

### Q3: Should you test private methods?
**Answer:**
*   **Generally No**.
*   **Reasoning**: Private methods are implementation details. You should test the *Public Interface*. If the private method is complex, maybe it belongs in its own class (where it becomes public).
*   **Exception**: If it's critical logic and hard to reach via public methods.

---

## Scenario-Based Questions

### Q4: You have a legacy codebase with 0% test coverage. How do you start refactoring?
**Answer:**
1.  **Characterization Test**: Write a high-level Integration/E2E test that captures the *current behavior* (even bugs).
2.  **Refactor**: Now you can change code. If the test fails, you broke something.
3.  **Unit Tests**: As you touch specific modules, add unit tests for them.

### Q5: Your CI pipeline takes 45 minutes to run. Developers are complaining. What do you do?
**Answer:**
1.  **Parallelize**: Run tests across 10 nodes.
2.  **Split**: Separate "Fast Unit Tests" (run on every commit) from "Slow Integration Tests" (run on merge or nightly).
3.  **Optimize**: Look for `sleep()` calls or slow DB setups. Use in-memory DB (SQLite) for logic tests if possible (carefully).

---

## Behavioral / Role-Specific Questions

### Q6: A developer says "I don't have time to write tests". What do you say?
**Answer:**
*   **"You don't have time NOT to write tests."**
*   **Cost**: Fixing a bug in Prod costs 100x more than fixing it in Dev.
*   **Velocity**: Tests allow us to move fast *later* without fear of breaking things.
