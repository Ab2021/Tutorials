# Day 60: The Ultimate Quiz

## 1. Database
**Q: When would you choose Cassandra over Postgres?**
*   A: When you need massive write throughput and can tolerate eventual consistency (AP system).

## 2. Architecture
**Q: What is the difference between a Load Balancer and a Reverse Proxy?**
*   A: LB distributes traffic. Reverse Proxy handles security/static files. Often the same tool (Nginx) does both.

## 3. Caching
**Q: What is Cache Stampede?**
*   A: 1000 requests hit DB when cache expires. Fix: Mutex or Early Expiry.

## 4. Security
**Q: How do you store passwords?**
*   A: Salt + Hash (Argon2 or bcrypt).

## 5. AI
**Q: What is RAG?**
*   A: Retrieval Augmented Generation. Injecting private data into LLM prompt.

## 6. DevOps
**Q: Why use Docker?**
*   A: Consistency. "It works on my machine" -> "It works everywhere".
