# Day 46 (Part 1): Advanced Mock - Ride Sharing ETA

> **Phase**: 6 - Deep Dive
> **Topic**: Logistics & Time
> **Focus**: H3, Routing, and Pricing
> **Reading Time**: 60 mins

---

## 1. Geospatial Indexing (H3)

Lat/Lon is hard to query.

### 1.1 Uber H3
*   Hexagonal Hierarchical Spatial Index.
*   **Why Hexagons?**: Neighbors are equidistant (unlike squares).
*   **Bucketing**: Map Lat/Lon to HexID. Aggregate traffic/demand per HexID.

---

## 2. ETA Prediction

### 2.1 Graph Features
*   It's not just A to B. It's the *Route*.
*   **Segment Embeddings**: Learn embedding for every road segment.
*   **Global Context**: Day of week, Weather, Event (Concert).

### 2.2 DeeprETA (Uber)
*   Transformer-based.
*   Encodes route sequence. Attention on traffic hotspots.

---

## 3. Tricky Interview Questions

### Q1: Surge Pricing Algorithm?
> **Answer**:
> *   **Goal**: Balance Supply (Drivers) and Demand (Riders).
> *   **PID Controller**: If Demand > Supply, increase Price.
> *   **ML**: Predict future demand. Pre-surge.

### Q2: How to test ETA accuracy?
> **Answer**:
> *   **Metric**: MAE (Mean Absolute Error) or MAPE (Percentage).
> *   **Segment**: Analyze error by City, Time, Distance. (Maybe bad at long trips?).

### Q3: Driver Dispatch?
> **Answer**:
> *   Bipartite Matching (Drivers <-> Riders).
> *   **Hungarian Algorithm** (O(N^3)) - too slow.
> *   **Greedy with Geohash**: Match within H3 cell.

---

## 4. Practical Edge Case: GPS Drift
*   **Problem**: GPS says car is in building.
*   **Fix**: **Map Matching** (Hidden Markov Model). Snap point to nearest valid road segment consistent with trajectory.

