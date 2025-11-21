# Day 34: Design Uber

## 1. Requirements
*   **Functional:** Request Ride, Match Driver, Track Ride, Payment.
*   **Non-Functional:** Real-time location tracking, High consistency (No double booking), Low latency matching.
*   **Scale:** 100M users, 10M drivers.

## 2. Architecture
*   **Web Socket Server:** Maintains persistent connection with Drivers (Location updates) and Riders (Ride status).
*   **Location Service:** Ingests driver locations (Kafka -> Redis/S2).
*   **Matching Service:** Finds nearest driver.
*   **Trip Service:** Manages state machine (Requested -> Matched -> Started -> Ended).

## 3. Location Updates
*   **Driver App:** Sends GPS every 4s.
*   **Server:**
    *   Updates Redis (Current Location).
    *   Logs to Cassandra (Location History) for analytics/disputes.
*   **Optimization:** Don't send every point. Send delta. Use UDP (if packet loss is acceptable) or WebSocket.

## 4. Matching Algorithm
*   **Naive:** Find nearest driver.
*   **Problem:** If everyone picks nearest, some drivers get no rides, and some areas get starved.
*   **Global Optimization:** Match a *batch* of riders and drivers every few seconds to maximize total trips / minimize total wait time. (Bipartite Matching).
