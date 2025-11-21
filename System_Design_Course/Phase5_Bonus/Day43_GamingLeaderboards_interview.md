# Day 43: Gaming Leaderboards - Interview Prep

## Common Interview Questions

### Q1: Design a real-time gaming leaderboard for 100M players

**Approach**:
1. **Clarify Requirements**
   - Read/write ratio? (Usually read-heavy: 100:1)
   - Consistency requirements? (Eventual consistency OK)
   - Query patterns? (Top-K, player rank, range queries)
   - Time-based leaderboards? (Daily, weekly, all-time)

2. **High-Level Design**
   ```
   Client → API Gateway → Write Service → Redis + Cassandra
                       → Read Service → Redis (with cache)
   ```

3. **Data Store Selection**
   - **Redis Sorted Sets** for hot data (fast reads/writes)
   - **Cassandra** for persistent storage
   - **Reasoning**: Redis provides O(log N) operations, Cassandra handles scale

4. **Key Operations**
   - Update score: `ZADD leaderboard:game_id {player_id: score}`
   - Get top 100: `ZREVRANGE leaderboard:game_id 0 99 WITHSCORES`
   - Get rank: `ZREVRANK leaderboard:game_id player_id`

5. **Scaling Strategy**
   - Shard by game_id using consistent hashing
   - Cache top-K results (TTL: 60s)
   - Use read replicas for Redis

**Follow-up**: How would you handle 1B players?
- **Answer**: Tiered storage - keep top 10M in Redis, rest in Cassandra. For rank queries outside top 10M, use approximate ranking.

---

### Q2: How do you handle score updates for millions of concurrent players?

**Answer**:

1. **Write Path Optimization**
   ```python
   # Batch updates using Redis pipeline
   pipeline = redis.pipeline()
   for player_id, score in batch:
       pipeline.zadd(f"leaderboard:{game_id}", {player_id: score})
   pipeline.execute()
   ```

2. **Async Processing**
   - Accept score update → Return 202 Accepted
   - Queue update in Kafka
   - Background workers process queue
   - Eventually update Redis + Cassandra

3. **Rate Limiting**
   - Limit updates per player (e.g., 10/minute)
   - Prevents spam and abuse

4. **Sharding**
   - Partition by game_id to distribute load
   - Each shard handles ~10M players

**Trade-off**: Eventual consistency vs. immediate updates
- **Choice**: Eventual consistency (acceptable for gaming)

---

### Q3: How would you implement time-based leaderboards (daily, weekly, monthly)?

**Answer**:

1. **Separate Sorted Sets**
   ```
   leaderboard:game_id:daily
   leaderboard:game_id:weekly
   leaderboard:game_id:monthly
   leaderboard:game_id:all_time
   ```

2. **Automatic Expiry**
   ```python
   # Daily leaderboard expires in 24 hours
   redis.zadd("leaderboard:game_id:daily", {player_id: score})
   redis.expire("leaderboard:game_id:daily", 86400)
   ```

3. **Reset Strategy**
   - **Option A**: Scheduled job deletes/renames key at midnight
   - **Option B**: Include timestamp in key (e.g., `leaderboard:game_id:daily:2024-01-15`)

4. **Implementation**
   ```python
   def get_time_key(game_id, timeframe):
       if timeframe == 'daily':
           return f"leaderboard:{game_id}:daily:{date.today()}"
       elif timeframe == 'weekly':
           week = date.today().isocalendar()[1]
           return f"leaderboard:{game_id}:weekly:{week}"
       # ... similar for monthly
   ```

**Follow-up**: How do you handle timezone differences?
- **Answer**: Use UTC for all timestamps. Client-side converts to local timezone for display.

---

### Q4: How do you prevent cheating in leaderboards?

**Answer**:

1. **Server-Side Validation**
   ```python
   def validate_score(player_id, new_score, game_session):
       # Check if score is possible given game session duration
       max_possible = game_session.duration * MAX_SCORE_PER_SECOND
       if new_score > max_possible:
           flag_as_cheater(player_id)
           return False
       return True
   ```

2. **Anomaly Detection**
   - Track score progression over time
   - Flag sudden jumps (e.g., 100 → 10,000 in one update)
   - Use statistical methods (Z-score > 3)

3. **Rate Limiting**
   - Limit score update frequency
   - Prevent automated bots

4. **Cryptographic Signatures**
   - Client sends score + HMAC signature
   - Server verifies signature using shared secret
   ```python
   signature = hmac.new(secret, f"{player_id}:{score}:{timestamp}".encode(), 'sha256')
   ```

5. **Manual Review Queue**
   - Flag suspicious players for human review
   - Temporary ban until verified

---

### Q5: How would you implement regional leaderboards with global aggregation?

**Answer**:

1. **Regional Sharding**
   ```
   US Region:  leaderboard:game_id:region:us
   EU Region:  leaderboard:game_id:region:eu
   APAC Region: leaderboard:game_id:region:apac
   ```

2. **Write Path**
   - Player updates go to their regional Redis instance
   - Low latency (local writes)

3. **Global Aggregation**
   - **Option A**: Real-time merge (expensive)
     ```python
     redis.zunionstore("leaderboard:game_id:global", 
                       ["leaderboard:game_id:region:us",
                        "leaderboard:game_id:region:eu",
                        "leaderboard:game_id:region:apac"],
                       aggregate='MAX')
     ```
   - **Option B**: Periodic aggregation (every 5 minutes)
     - Background job merges regional leaderboards
     - Acceptable staleness for global view

4. **Read Path**
   - Regional queries: Read from local Redis (fast)
   - Global queries: Read from aggregated leaderboard (slightly stale)

**Trade-off**: Consistency vs. latency
- **Choice**: Eventual consistency with periodic aggregation

---

### Q6: How do you handle ties in rankings?

**Answer**:

1. **Lexicographical Ordering**
   - Use player_id as tiebreaker
   - Consistent but arbitrary

2. **Timestamp-Based**
   - Earlier timestamp = higher rank
   - Encode in score: `score * 1e10 + (MAX_TIMESTAMP - timestamp)`
   ```python
   composite_score = score * 1e10 + (2**32 - int(time.time()))
   redis.zadd(key, {player_id: composite_score})
   ```

3. **Shared Rank**
   - Both players get same rank (e.g., two players at rank 5)
   - Next player gets rank 7 (not 6)
   ```python
   def get_rank_with_ties(player_id):
       score = redis.zscore(key, player_id)
       # Count players with strictly higher scores
       higher_count = redis.zcount(key, score + 1, '+inf')
       return higher_count + 1
   ```

**Recommendation**: Use timestamp-based for competitive games, shared rank for casual games.

---

### Q7: How would you optimize for read-heavy workloads?

**Answer**:

1. **Multi-Level Caching**
   ```
   L1: In-memory cache (local to service instance)
   L2: Redis cache (shared across instances)
   L3: Redis sorted set (source of truth)
   ```

2. **Cache Top-K Results**
   ```python
   def get_top_100_cached(game_id):
       cache_key = f"cache:top_100:{game_id}"
       
       # Check L1 cache
       if cache_key in local_cache:
           return local_cache[cache_key]
       
       # Check L2 cache
       cached = redis.get(cache_key)
       if cached:
           return json.loads(cached)
       
       # Fetch from source
       result = redis.zrevrange(f"leaderboard:{game_id}", 0, 99, withscores=True)
       
       # Cache with TTL
       redis.setex(cache_key, 60, json.dumps(result))
       local_cache[cache_key] = result
       
       return result
   ```

3. **Read Replicas**
   - Redis read replicas for horizontal scaling
   - Route reads to replicas, writes to primary

4. **CDN for Static Content**
   - Cache leaderboard pages at edge locations
   - Invalidate on updates (or use TTL)

**Result**: 10x-100x improvement in read throughput

---

### Q8: What happens if Redis goes down?

**Answer**:

1. **High Availability Setup**
   - Redis Sentinel for automatic failover
   - 3-node cluster (1 primary, 2 replicas)
   - Failover time: ~30 seconds

2. **Persistence**
   - RDB snapshots (every 5 minutes)
   - AOF (append-only file) for durability
   - Trade-off: Performance vs. durability

3. **Fallback to Cassandra**
   ```python
   def get_top_k(game_id, k=100):
       try:
           # Try Redis first
           return redis.zrevrange(f"leaderboard:{game_id}", 0, k-1)
       except RedisConnectionError:
           # Fallback to Cassandra
           return cassandra.query(
               f"SELECT * FROM leaderboard WHERE game_id = {game_id} "
               f"ORDER BY score DESC LIMIT {k}"
           )
   ```

4. **Graceful Degradation**
   - Serve stale cached data if Redis is down
   - Display "Leaderboard temporarily unavailable" message

**SLA**: 99.99% uptime with proper HA setup

---

## System Design Patterns

### Pattern 1: Lambda Architecture for Leaderboards

```
Real-time Layer (Speed):  Redis Sorted Sets
Batch Layer (Accuracy):   Cassandra + Spark
Serving Layer:            Merged view
```

**Use Case**: When you need both real-time updates and historical analysis.

### Pattern 2: CQRS (Command Query Responsibility Segregation)

```
Write Model:  Kafka → Write Service → Redis + Cassandra
Read Model:   Read Service → Redis (optimized for queries)
```

**Benefit**: Optimize read and write paths independently.

### Pattern 3: Event Sourcing

```
Score Update Event → Event Store → Rebuild Leaderboard
```

**Benefit**: Full audit trail, can rebuild leaderboard from events.

---

## Code Snippets

### Complete Leaderboard Service (Python)

```python
import redis
import time
from typing import List, Dict, Optional

class LeaderboardService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def update_score(self, game_id: str, player_id: str, score: float) -> bool:
        """Update player's score in leaderboard"""
        key = f"leaderboard:{game_id}"
        
        # Validate score
        if score < 0:
            raise ValueError("Score cannot be negative")
        
        # Update Redis sorted set
        self.redis.zadd(key, {player_id: score})
        
        # Update timestamp
        self.redis.hset(f"player_activity:{game_id}", player_id, time.time())
        
        return True
    
    def get_top_k(self, game_id: str, k: int = 100) -> List[Dict]:
        """Get top K players"""
        key = f"leaderboard:{game_id}"
        
        results = self.redis.zrevrange(key, 0, k-1, withscores=True)
        
        return [
            {
                'rank': idx + 1,
                'player_id': player_id.decode(),
                'score': int(score)
            }
            for idx, (player_id, score) in enumerate(results)
        ]
    
    def get_player_rank(self, game_id: str, player_id: str) -> Optional[Dict]:
        """Get specific player's rank and score"""
        key = f"leaderboard:{game_id}"
        
        rank = self.redis.zrevrank(key, player_id)
        score = self.redis.zscore(key, player_id)
        
        if rank is None:
            return None
        
        return {
            'player_id': player_id,
            'rank': rank + 1,
            'score': int(score)
        }
    
    def get_neighbors(self, game_id: str, player_id: str, context: int = 5) -> List[Dict]:
        """Get players ranked around target player"""
        key = f"leaderboard:{game_id}"
        
        rank = self.redis.zrevrank(key, player_id)
        if rank is None:
            return []
        
        start = max(0, rank - context)
        end = rank + context
        
        results = self.redis.zrevrange(key, start, end, withscores=True)
        
        return [
            {
                'rank': start + idx + 1,
                'player_id': pid.decode(),
                'score': int(score),
                'is_current_player': pid.decode() == player_id
            }
            for idx, (pid, score) in enumerate(results)
        ]
```

### Redis Commands Cheat Sheet

```bash
# Add/update player score
ZADD leaderboard:game123 1000 player456

# Get top 10 players
ZREVRANGE leaderboard:game123 0 9 WITHSCORES

# Get player's rank (0-indexed)
ZREVRANK leaderboard:game123 player456

# Get player's score
ZSCORE leaderboard:game123 player456

# Get total number of players
ZCARD leaderboard:game123

# Get players with scores between 500-1000
ZRANGEBYSCORE leaderboard:game123 500 1000

# Remove player from leaderboard
ZREM leaderboard:game123 player456

# Increment player's score by 10
ZINCRBY leaderboard:game123 10 player456
```

---

## Key Metrics to Monitor

1. **Latency**
   - P50, P95, P99 for read/write operations
   - Target: < 100ms for reads, < 500ms for writes

2. **Throughput**
   - Requests per second (RPS)
   - Target: 100K+ RPS

3. **Cache Hit Rate**
   - Percentage of requests served from cache
   - Target: > 95%

4. **Error Rate**
   - Failed requests / total requests
   - Target: < 0.1%

5. **Leaderboard Size**
   - Number of players per leaderboard
   - Monitor memory usage

---

## Common Pitfalls

1. **Not handling ties properly** → Inconsistent rankings
2. **No rate limiting** → Vulnerable to spam/cheating
3. **No caching** → Poor read performance
4. **Single Redis instance** → Bottleneck at scale
5. **No persistence** → Data loss on crash
6. **Ignoring time zones** → Confusing time-based leaderboards
7. **No monitoring** → Can't detect issues

---

## Interview Tips

1. **Start with clarifying questions** - Don't jump into design
2. **Discuss trade-offs** - No perfect solution, explain choices
3. **Consider scale** - What works for 1K users won't work for 1B
4. **Think about failures** - What happens when Redis goes down?
5. **Mention monitoring** - Production systems need observability
6. **Use concrete numbers** - "100K RPS" is better than "high throughput"
7. **Draw diagrams** - Visual representation helps communication

---

## Further Reading

- Redis Sorted Sets: https://redis.io/docs/data-types/sorted-sets/
- Leaderboard Patterns: https://redislabs.com/solutions/use-cases/leaderboards/
- Cassandra Data Modeling: https://cassandra.apache.org/doc/latest/
- Consistent Hashing: https://en.wikipedia.org/wiki/Consistent_hashing
