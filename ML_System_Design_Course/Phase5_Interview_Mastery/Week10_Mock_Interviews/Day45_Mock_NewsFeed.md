# Day 45: Mock Interview 1 - News Feed (Facebook/LinkedIn)

> **Phase**: 5 - Interview Mastery
> **Week**: 10 - Mock Interviews
> **Focus**: Ranking & Personalization
> **Reading Time**: 60 mins

---

## 1. Problem Statement

"Design a News Feed for a social network. Users follow friends and pages. Goal: Maximize engagement (Time Spent)."

---

## 2. Step-by-Step Design

### Step 1: Requirements
*   **Users**: 1B DAU.
*   **Latency**: < 200ms for feed load.
*   **Freshness**: New posts appear within 1 min.

### Step 2: Data
*   **Actors**: User Profile (Age, Location).
*   **Objects**: Post (Text, Image, Video).
*   **Interactions**: Like, Comment, Share, View, Skip.
*   **Graph**: Who follows whom.

### Step 3: Retrieval (Candidate Generation)
*   **Pull Strategy**: When user opens app, query DB for all friends' recent posts.
*   **Fan-out**: For celebrities (10M followers), "Push" strategy is too expensive. Use Pull.
*   **Candidates**: Friends' posts + "You might like" (Collaborative Filtering).

### Step 4: Ranking
*   **Model**: Multi-Task Learning (MTL).
*   **Heads**: $P(\text{Click})$, $P(\text{Like})$, $P(\text{Share})$.
*   **Score**: $w_1 \cdot \text{Click} + w_2 \cdot \text{Like} + w_3 \cdot \text{Share}$.
*   **Features**:
    *   *User*: Past CTR.
    *   *Post*: Age, Media Type.
    *   *Context*: Time of day, Connection speed.

### Step 5: Infrastructure
*   **Feed Service**: Aggregates candidates.
*   **Ranking Service**: Runs the MTL model.
*   **Redis**: Caches the computed feed for 5 minutes.

---

## 3. Deep Dive Questions

**Interviewer**: "How do you handle diversity? I don't want to see 10 posts from the same person."
**Candidate**: "I would implement a Re-Ranking rule. Iterate through the ranked list. Maintain a 'Author Count' map. If Author X has appeared 2 times already, demote their subsequent posts by multiplying score by 0.5 or moving them down."

**Interviewer**: "How do you handle Ad injection?"
**Candidate**: "Ads are just another type of candidate. We rank them alongside organic posts using $eCPM$ (Expected Revenue). We guarantee an Ad every K slots (e.g., slot 5, 10, 15) to balance revenue vs. user experience."

---

## 4. Evaluation
*   **Offline**: AUC, LogLoss on historical logs.
*   **Online**: A/B Test. Metric: Time Spent per User. Guardrail Metric: App Latency.

---

## 5. Further Reading
- [Facebook News Feed Architecture](https://engineering.fb.com/2017/02/02/ml-applications/serving-a-billion-personalized-news-feeds/)
