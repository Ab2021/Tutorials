# Day 34: Testing Strategy & Tools

## 1. The Testing Pyramid

You can't test everything manually.
*   **Unit Tests (70%)**: Test individual functions. Fast (ms). Mock everything.
*   **Integration Tests (20%)**: Test how modules work together. Slower (s). Real DB/API.
*   **E2E Tests (10%)**: Test the whole user flow. Slow (mins). Selenium/Playwright.

---

## 2. Unit Testing (Pytest)

*   **Goal**: Does `add(2, 2)` return `4`?
*   **Mocking**: If `add` saves to DB, we don't want to hit the real DB. We **Mock** the DB call.
    *   `mock_db.save.assert_called_with(4)`

### 2.1 TDD (Test Driven Development)
1.  **Red**: Write a failing test.
2.  **Green**: Write just enough code to pass.
3.  **Refactor**: Clean up the code.

---

## 3. Integration Testing

*   **Goal**: Does the API endpoint `/users` actually save to the Postgres DB?
*   **Setup**: Spin up a **Test DB** (Docker).
*   **Teardown**: Wipe the DB after each test.
*   **Tools**: `pytest-docker`, `Testcontainers`.

---

## 4. Coverage

*   **Metric**: What % of code lines are executed by tests?
*   **Goal**: > 80% is good. 100% is often diminishing returns.
*   **Tool**: `pytest-cov`.

---

## 5. Summary

Today we built a safety net.
*   **Unit**: Fast feedback.
*   **Integration**: Real-world verification.
*   **Mock**: Fake it 'til you make it.

**Tomorrow (Day 35)**: We finish the week with **Contract Testing** (Pact) and **E2E Testing**. Ensuring microservices don't break each other.
