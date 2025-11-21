# Day 26 Deep Dive: Uber Marketplace

## 1. Requirement
*   Match Rider with Driver.
*   Real-time updates (Drivers move every 4s).
*   High throughput.

## 2. Architecture Evolution
### V1: Postgres (PostGIS)
*   **Index:** GiST (Generalized Search Tree).
*   **Issue:** Write lock contention. Updating index every 4s for 100k drivers killed the DB.

### V2: Redis (Geohash)
*   **Storage:** Redis `GEOADD`, `GEORADIUS`.
*   **Pros:** In-memory. Fast.
*   **Cons:** Single threaded. Sharding is tricky (Geo-sharding).

### V3: Google S2 (In-Memory Go Service)
*   **Sharding:** Divide world into S2 cells.
*   **State:** Each shard manages a set of cells.
*   **Gossip:** Drivers send location to any node -> Gossip propagates to correct shard.
*   **Matching:** Query the shard responsible for Rider's cell.

## 3. Why S2?
*   **Hilbert Curve:** A space-filling curve that preserves locality.
*   **1D Index:** Converts 2D map into 1D line.
*   **Range Query:** "Nearby" becomes a range scan on the 1D line.

## 4. Code: Geohash (Python)
```python
import pygeohash as pgh

lat, lon = 37.7749, -122.4194
hash = pgh.encode(lat, lon, precision=5)
print(hash) # '9q8yy'

# Find neighbors
neighbors = pgh.neighbors(hash)
print(neighbors)
```
