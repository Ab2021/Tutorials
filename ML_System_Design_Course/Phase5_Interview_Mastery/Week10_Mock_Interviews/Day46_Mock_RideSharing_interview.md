# Day 46: Mock Interview: Ride Sharing - Interview Questions

> **Topic**: Uber/Lyft Marketplace
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. Design a Ride Sharing System (Uber).
**Answer:**
*   **Goal**: Match Riders with Drivers. Minimize ETA. Maximize Revenue.
*   **Components**: Location Service, Matching Service, Pricing Service.

### 2. How do you handle Location updates?
**Answer:**
*   Driver app sends GPS every 5s.
*   **Geospatial Index**: QuadTree or Google S2 or Uber H3.
*   Update driver position in Redis (GeoHash).

### 3. Explain Uber H3.
**Answer:**
*   Hexagonal Hierarchical Spatial Index.
*   Hexagons tile the sphere perfectly (unlike squares).
*   Easy to calculate neighbors (always 6).

### 4. How do you match Rider to Driver?
**Answer:**
*   Find drivers in radius K.
*   Filter (Available, Car Type).
*   Rank by ETA.
*   Send request to top driver.

### 5. How do you calculate ETA?
**Answer:**
*   **Routing Engine** (Graph).
*   **ML Model**: Predict traffic.
*   Features: Time of day, Weather, Historical speed on segment.
*   DeeprETA (Deep Learning).

### 6. What is "Surge Pricing"?
**Answer:**
*   Dynamic pricing to balance Supply and Demand.
*   If Demand > Supply in an area -> Increase Price.
*   Encourages drivers to move there. Reduces demand.

### 7. How do you predict Demand?
**Answer:**
*   Time-series forecasting (LSTM/Prophet).
*   Predict demand per H3 hexagon for next 15 mins.

### 8. How do you handle "Batched Matching"?
**Answer:**
*   Don't match instantly (Greedy).
*   Wait 5 seconds. Collect all requests.
*   Bipartite Matching (Hungarian Algorithm) to optimize global utility.

### 9. What is "Uber Pool" (Shared Rides)?
**Answer:**
*   Complex routing.
*   Constraint: Detour time < X mins.
*   Knapsack-like problem.

### 10. How do you handle "Fraud"?
**Answer:**
*   **Driver Fraud**: Fake GPS, Collusion (Driver and Rider are same person).
*   **Rider Fraud**: Stolen card.

### 11. What metrics do you optimize?
**Answer:**
*   **ETA**: Wait time.
*   **Completion Rate**: % of requests fulfilled.
*   **Driver Utilization**: % time with passenger.

### 12. How do you store Trip History?
**Answer:**
*   **SQL**: Transactional data (Trip ID, Status).
*   **Data Lake**: GPS traces for training.

### 13. How do you handle "Cancellations"?
**Answer:**
*   If driver cancels, re-match immediately with high priority.
*   Penalty features for driver ranking.

### 14. What is "Session-based" pricing?
**Answer:**
*   Price might depend on user history (Willingness to Pay).
*   Controversial.

### 15. How do you scale the Matching Service?
**Answer:**
*   Sharding by City / Geohash.
*   New York requests go to NY shard.

### 16. What happens if GPS is lost?
**Answer:**
*   Dead Reckoning (Extrapolate position based on speed/direction).
*   Map Matching (Snap to road).

### 17. How do you incentivize drivers?
**Answer:**
*   "Quests": Complete 10 rides for $50 bonus.
*   Predictive positioning: "Go to downtown, demand will be high in 10 mins".

### 18. What is the "Marketplace" concept?
**Answer:**
*   Two-sided market.
*   Need to keep both sides happy.
*   Liquidity is key.

### 19. How do you handle "Airports"?
**Answer:**
*   Queue system (FIFO).
*   Geofence.

### 20. What is "Shadow Banning"?
**Answer:**
*   Fraudulent users see app working but never get matched.
