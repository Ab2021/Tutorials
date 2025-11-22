# Day 5: Triggers - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the default trigger for an Event Time window?**
    -   *A*: `EventTimeTrigger`. It fires once when the watermark passes the window end.

2.  **Q: Why would you use a custom trigger?**
    -   *A*: To get early results (speculative) before the window closes, or to handle late data specially.

3.  **Q: What is the difference between FIRE and PURGE?**
    -   *A*: FIRE emits a result but keeps the data. PURGE deletes the data.

### Production Challenges
-   **Challenge**: **Duplicate Results**.
    -   *Scenario*: Using a trigger that FIREs multiple times without PURGING.
    -   *Fix*: Ensure your downstream consumer can handle updates (idempotency or upserts).

### Troubleshooting Scenarios
**Scenario**: Global Window not producing output.
-   *Cause*: Forgot to set a Trigger.
-   *Fix*: `.trigger(CountTrigger.of(100))`.
