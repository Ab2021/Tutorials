# Day 10 Deep Dive: The Interview Framework

## Step 1: Requirements (5 min)
*   **Ask:** "Is this a global app?", "Read heavy or Write heavy?", "Consistency vs Availability?".
*   **Define:**
    *   Functional: "User can post tweet", "User can view feed".
    *   Non-Functional: "100ms latency", "Eventually consistent".

## Step 2: Estimation (5 min)
*   **DAU:** 100M.
*   **QPS:** 100M * 10 / 86400 = 10k QPS.
*   **Storage:** 100M * 1KB = 100GB/day.

## Step 3: High Level Design (10 min)
*   **Diagram:**
    *   Client -> CDN -> LB -> Web Server.
    *   Web Server -> Redis (Cache).
    *   Web Server -> DB (Master/Slave).
*   **Justify:** "I chose Redis for fast reads because we are read-heavy."

## Step 4: Deep Dive (20 min)
*   **Interviewer:** "How do you handle the Celebrity problem?"
*   **You:** "We can use a hybrid approach. For celebrities, push tweets to a separate Redis list..."
*   **Interviewer:** "The DB is slow."
*   **You:** "Let's shard by UserID..."

## Step 5: Wrap Up (5 min)
*   **Bottlenecks:** "Single point of failure in LB."
*   **Improvements:** "Add monitoring, multi-region."
