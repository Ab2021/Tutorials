# Day 44: System Design Round - The Template

> **Phase**: 5 - Interview Mastery
> **Week**: 10 - Mock Interviews
> **Focus**: Managing the 45 Minutes
> **Reading Time**: 45 mins

---

## 1. The 45-Minute Clock

You must drive the interview. Don't wait for questions.

*   **0-5 min**: Requirements & Constraints (Clarify!).
*   **5-10 min**: Data Engineering (Sources, Labels, Features).
*   **10-20 min**: Modeling (Baseline, Deep Learning, Loss).
*   **20-30 min**: Evaluation (Offline vs Online).
*   **30-40 min**: Serving & Infrastructure (Architecture diagram).
*   **40-45 min**: Scaling & Bottlenecks (Edge cases).

---

## 2. The Cheat Sheet

### 2.1 Back-of-the-Envelope Calculations
*   **DAU (Daily Active Users)**: 100 Million.
*   **Requests**: 100M * 10 requests/day = 1 Billion/day.
*   **QPS**: $10^9 / 10^5 \text{ seconds} = 10,000$ QPS.
*   **Storage**: 1KB per log * 1B = 1TB/day.

### 2.2 Standard Components
*   **Load Balancer**: Nginx.
*   **Queue**: Kafka.
*   **Cache**: Redis.
*   **Blob Storage**: S3.
*   **Vector DB**: Pinecone/Milvus.

---

## 3. Soft Skills

*   **Draw as you talk**: Use Excalidraw or a whiteboard. Visuals anchor the conversation.
*   **Check in**: "Does this architecture make sense to you? Should I dive deeper into the Model or the Serving?"
*   **Admit unknowns**: "I'm not sure about the exact latency of X, but assuming it's 10ms..."

---

## 4. Interview Preparation

### Checklist
1.  [ ] Memorize powers of 2 and 10 ($2^{10} \approx 10^3$, $2^{20} \approx 10^6$).
2.  [ ] Practice drawing a generic ML pipeline (Data -> Train -> Serve) in 2 minutes.

---

## 5. Further Reading
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [Grokking the System Design Interview](https://www.designgurus.io/course/grokking-the-system-design-interview)
