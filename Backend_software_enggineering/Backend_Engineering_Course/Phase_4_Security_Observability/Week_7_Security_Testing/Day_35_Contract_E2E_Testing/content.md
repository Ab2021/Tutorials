# Day 35: Contract & E2E Testing

## 1. The Microservice Nightmare

Service A calls Service B.
*   Service B changes field `user_id` to `userId`.
*   Service A crashes in Production.
*   **Unit Tests** passed (because A mocked B).
*   **Integration Tests** passed (if they ran against an old B).
*   **Solution**: Contract Testing.

---

## 2. Consumer-Driven Contracts (Pact)

*   **Consumer (Service A)**: Defines expectations. "I expect `GET /user/1` to return `{id: 1}`". This is the **Pact**.
*   **Provider (Service B)**: Verifies the Pact. "Do I actually return `{id: 1}`?".
*   **Flow**:
    1.  Consumer generates Pact file (JSON).
    2.  Uploads to Pact Broker.
    3.  Provider downloads Pact and runs verification tests.
    4.  If Provider fails, they cannot deploy.

---

## 3. E2E Testing (End-to-End)

Testing the whole system from the user's perspective.
*   **Tools**: Playwright, Cypress, Selenium.
*   **Scenario**: "User logs in, adds item to cart, pays."
*   **Pros**: Catch integration bugs. High confidence.
*   **Cons**: Slow, Flaky, Hard to debug.

### 3.1 Best Practices
*   **Keep it minimal**: Only test critical paths (Checkout, Login).
*   **Clean State**: Create a fresh user for every test.
*   **Headless**: Run without a UI for speed in CI.

---

## 4. Test Environments

*   **Local**: Docker Compose.
*   **Ephemeral (Preview)**: Spin up a full environment for every Pull Request. (Expensive but safe).
*   **Staging**: Mirror of Prod.
*   **Production**: Canary testing / Feature Flags.

---

## 5. Summary

Today we ensured compatibility.
*   **Pact**: Stop microservices from breaking each other.
*   **E2E**: Verify the user journey.

**Week 7 Wrap-Up**:
We have covered:
1.  OAuth2 & MFA.
2.  App Security (OWASP).
3.  Secrets Management (Vault).
4.  Testing Strategy (Unit/Integration).
5.  Contract & E2E Testing.

**Next Week (Week 8)**: We enter the final phase of "Production Readiness". **Observability**. Logs, Metrics, Tracing. How to debug when things go wrong at 3 AM.
