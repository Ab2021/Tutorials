# Day 3: Clickstream - Deep Dive

## Deep Dive & Internals

### Session Windows
**Definition**: A period of activity separated by a gap of inactivity.
**Flink Implementation**: `EventTimeSessionWindows.withGap(Time.minutes(30))`.
-   **Merging**: Session windows merge. If Event A is at 10:00 and Event B is at 10:20, they merge into one session (10:00-10:20). If Event C comes at 10:40, it extends the session.
-   **Trigger**: Fires when the gap passes.

### Bot Detection
Bots skew analytics.
-   **Heuristics**: High request rate, missing UserAgent, known Data Center IPs.
-   **Implementation**: Flink Filter or Side Output.
-   **Bloom Filter**: Check IP against a massive blacklist efficiently.

### User Unification (Identity Resolution)
User starts anonymous (CookieID), then logs in (UserID).
-   **Problem**: Associate previous anonymous events with the UserID.
-   **Solution**:
    1.  **Late Binding**: Do it in the Data Warehouse (Join Cookie table with User table).
    2.  **Real-time**: Harder. Flink needs to update the "Anonymous Profile" with the UserID.

### Performance Implications
-   **Skew**: "Null" UserID (anonymous users) might all go to one partition.
    -   *Fix*: Generate a random UUID for anonymous users instead of Null.
