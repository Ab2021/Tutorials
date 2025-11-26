# Day 35: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the main advantage of Contract Testing over E2E Testing?
**Answer:**
*   **Speed & Feedback**: Contract tests run fast (like unit tests) and give immediate feedback to the developer. E2E tests are slow and run late in the pipeline.
*   **Isolation**: Contract tests verify the *interaction* between two services without spinning up the whole world.

### Q2: What is "Chaos Engineering"?
**Answer:**
*   **Concept**: Intentionally injecting failure into a system to test its resilience. (e.g., Netflix Chaos Monkey).
*   **Goal**: Verify that the system degrades gracefully (e.g., if Redis is down, the site still works, just slower).

### Q3: How do you handle "Test Data" in E2E tests?
**Answer:**
*   **Anti-Pattern**: Using a shared "Seed Data" (e.g., User ID 1). If two tests run in parallel, they clash.
*   **Pattern**: Create fresh data for each test.
    *   `user = create_user()`
    *   `login(user)`
    *   `delete_user(user)`

---

## Scenario-Based Questions

### Q4: You deployed a change to Service A, and Service B started failing with 500 errors. How could Contract Testing have prevented this?
**Answer:**
*   If Service A (Provider) broke the contract (e.g., removed a field), the **Pact Verification** step in Service A's CI pipeline would have failed.
*   The build would have stopped. The bad code would never reach Production.

### Q5: Your E2E tests are failing 20% of the time due to "Element not found". How do you fix it?
**Answer:**
*   **Wait**: Don't use `sleep(5)`. Use explicit waits: `wait_until_element_visible()`.
*   **Retries**: Add automatic retries for flaky steps.
*   **Selectors**: Use robust selectors (`data-testid="submit-btn"`) instead of fragile XPath (`//div/div[2]/button`).

---

## Behavioral / Role-Specific Questions

### Q6: A manager wants to achieve 100% E2E test coverage. Is this a good goal?
**Answer:**
*   **No**.
*   **Ice Cream Cone Anti-Pattern**: Too many E2E tests make the suite slow (hours) and brittle (constant maintenance).
*   **Advice**: Stick to the Pyramid. 70% Unit, 20% Integration, 10% E2E (Critical Paths only).
