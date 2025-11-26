# Day 25: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between Swagger and OpenAPI?
**Answer:**
*   **OpenAPI**: The Specification (The Standard). Like "HTML".
*   **Swagger**: The Tooling (The Editor, The UI, The Codegen). Like "Chrome".
*   *History*: SmartBear donated the Swagger Spec to the Linux Foundation, and it was renamed OpenAPI.

### Q2: Why is "Design-First" API development recommended for large teams?
**Answer:**
*   **Parallelism**: Frontend and Backend agree on the YAML contract first. Frontend can build Mocks using the YAML while Backend builds the implementation.
*   **Governance**: Architects can review the YAML for consistency (naming conventions, security) before any code is written.
*   **Contract Testing**: You can validate that the implementation matches the design automatically.

### Q3: How do you handle documentation for multiple API versions?
**Answer:**
*   **Strategy**: Host separate specs.
    *   `api.com/docs/v1` -> Loads `openapi-v1.yaml`.
    *   `api.com/docs/v2` -> Loads `openapi-v2.yaml`.
*   **Swagger UI**: Usually has a dropdown in the top bar to switch definitions.

---

## Scenario-Based Questions

### Q4: You changed a required field to optional in the code, but forgot to update the docs. A client complains. How do you prevent this?
**Answer:**
*   **CI/CD Check**:
    1.  Generate the OpenAPI spec from code during the build.
    2.  Diff it against the committed `openapi.yaml`.
    3.  If they differ, fail the build.
*   **Or**: Use a tool like `dredd` or `schemathesis` that runs tests against your running API to verify it matches the spec.

### Q5: A developer asks "Why do we need SDKs? Can't users just use `requests` or `fetch`?"
**Answer:**
*   **DX**: SDKs provide type safety, auto-completion (IntelliSense), and handle retry logic/auth automatically.
*   **Adoption**: A good SDK lowers the barrier to entry. "pip install my-api" is easier than reading 50 pages of HTTP docs.

---

## Behavioral / Role-Specific Questions

### Q6: You are documenting a legacy API with no specs. Where do you start?
**Answer:**
*   **Don't write YAML manually**.
*   **Traffic Sniffing**: Use a proxy (Charles/Fiddler) or a middleware to record real requests/responses.
*   **Auto-Gen**: Use tools (like Optic) that analyze the traffic and reverse-engineer an OpenAPI spec.
*   **Refine**: Once you have a base spec, manually refine descriptions and types.
