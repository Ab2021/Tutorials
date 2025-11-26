# Day 22: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is URL Versioning (`/v1/resource`) considered "Un-RESTful" by purists?
**Answer:**
*   **REST Principle**: A URL identifies a *Resource* (Noun).
*   **Argument**: `v1/user/123` and `v2/user/123` point to the *same* user (Alice). Having two URLs for the same resource implies two different resources.
*   **Counter-Argument**: Pragmatism beats purity. URL versioning is developer-friendly and cache-friendly.

### Q2: What is Semantic Versioning (SemVer)?
**Answer:**
*   Format: `MAJOR.MINOR.PATCH` (e.g., 2.1.4).
*   **MAJOR**: Breaking changes.
*   **MINOR**: New features (Backwards compatible).
*   **PATCH**: Bug fixes (Backwards compatible).
*   *API Context*: Usually we only expose MAJOR versions in the URL (`/v1`, `/v2`). MINOR/PATCH updates happen silently in place.

### Q3: How do you handle Database Schema changes when you have `v1` and `v2` APIs running simultaneously?
**Answer:**
*   **The Challenge**: `v1` expects `name` column. `v2` expects `first_name` and `last_name`.
*   **Strategy**: **Expand and Contract**.
    1.  **Expand**: Add `first_name` and `last_name` columns. Keep `name`.
    2.  **Write Dual**: Update code to write to ALL columns.
    3.  **Migrate**: Backfill new columns from old data.
    4.  **Contract**: Once `v1` is turned off, remove the `name` column.

---

## Scenario-Based Questions

### Q4: You need to deprecate a field `phone_number` in the API because of privacy laws. How do you do it safely?
**Answer:**
1.  **Mark Deprecated**: Update OpenAPI/Swagger docs to mark it deprecated.
2.  **Monitor**: Check logs to see who is still reading that field.
3.  **Notify**: Email those specific developers.
4.  **Nullify**: Start returning `null` or a masked value `***-***-1234` (if allowed) before removing it completely.
5.  **Remove**: In the next Major version (`v2`), remove the field.

### Q5: A client asks for a "Date Header" versioning strategy (e.g., `Stripe-Version: 2023-10-15`). Is this good?
**Answer:**
*   **Pros**: Extremely granular. You can release breaking changes daily without a massive `v2` rewrite.
*   **Cons**: **Maintenance Nightmare**. The backend needs a massive `if/else` chain or a middleware pipeline to transform the response based on the date.
*   **Verdict**: Only do this if you are Stripe (Huge team, critical need for velocity). For most, `v1/v2` is enough.

---

## Behavioral / Role-Specific Questions

### Q6: A Product Manager wants to release a breaking change to `v1` without bumping the version because "it's just a small fix". What do you say?
**Answer:**
**"No."**
*   **Reasoning**: "Small" for us might be "Catastrophic" for a client. If a mobile app crashes because of this, we lose trust/revenue.
*   **Alternative**:
    1.  Make it non-breaking (e.g., support both old and new behavior).
    2.  If it *must* break, we must release `v2`.
    3.  Or, use a Feature Flag to opt-in users to the fix.
