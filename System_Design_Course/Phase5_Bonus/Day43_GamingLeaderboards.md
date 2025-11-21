# Day 43: Gaming Leaderboards - Core Concepts

## Overview
Design a real-time gaming leaderboard system that can handle millions of concurrent players, support multiple game modes, and provide sub-second query performance for rankings.

## Key Requirements

### Functional Requirements
- **Real-time Score Updates**: Players' scores update instantly
- **Global & Regional Rankings**: Support multiple leaderboard scopes
- **Time-based Leaderboards**: Daily, weekly, monthly, all-time
- **Top-K Queries**: Efficiently retrieve top N players
- **Player Rank Lookup**: Find a specific player's rank quickly
- **Range Queries**: Get players ranked between positions X and Y

### Non-Functional Requirements
- **Low Latency**: < 100ms for reads, < 500ms for writes
- **High Throughput**: Handle 100K+ score updates/sec
- **Consistency**: Eventually consistent is acceptable
- **Scalability**: Support billions of players
- **Availability**: 99.99% uptime

## System Architecture

### High-Level Design

```
┌─────────────┐
│   Clients   │
│ (Game Apps) │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   API Gateway   │
│  (Rate Limit)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Write  │ │  Read    │
│ Service│ │ Service  │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌────────────────────┐
│   Redis Sorted Set │
│   (Primary Store)  │
└────────────────────┘
    │
    ▼
┌────────────────────┐
│  Cassandra/DynamoDB│
│  (Persistent Store)│
└────────────────────┘
```

### Data Model

**Redis Sorted Set** (Primary):
```
Key: leaderboard:{game_id}:{mode}:{timeframe}
Score: player_score
Member: player_id
```

**Cassandra Table** (Persistent):
```sql
CREATE TABLE leaderboard_scores (
    game_id text,
    mode text,
    timeframe text,
    player_id text,
    score bigint,
    timestamp timestamp,
    PRIMARY KEY ((game_id, mode, timeframe), score, player_id)
) WITH CLUSTERING ORDER BY (score DESC, player_id ASC);
```

## Core Components

### 1. Score Update Service

**Write Path**:
```python
class ScoreUpdateService:
    def update_score(self, player_id, game_id, mode, score):
        # 1. Validate score
        if not self.validate_score(score):
            raise InvalidScoreError()
        
        # 2. Update Redis sorted set (atomic)
        key = f"leaderboard:{game_id}:{mode}:all_time"
        self.redis.zadd(key, {player_id: score})
        
        # 3. Update time-based leaderboards
        for timeframe in ['daily', 'weekly', 'monthly']:
            time_key = f"leaderboard:{game_id}:{mode}:{timeframe}"
            self.redis.zadd(time_key, {player_id: score})
            self.set_expiry(time_key, timeframe)
        
        # 4. Async write to persistent storage
        self.queue.publish({
            'player_id': player_id,
            'game_id': game_id,
            'mode': mode,
            'score': score,
            'timestamp': time.time()
        })
```

### 2. Leaderboard Query Service

**Top-K Query**:
```python
class LeaderboardQueryService:
    def get_top_k(self, game_id, mode, k=100):
        key = f"leaderboard:{game_id}:{mode}:all_time"
        
        # ZREVRANGE returns top K with scores
        results = self.redis.zrevrange(
            key, 
            start=0, 
            end=k-1, 
            withscores=True
        )
        
        return [
            {
                'rank': idx + 1,
                'player_id': player_id,
                'score': score
            }
            for idx, (player_id, score) in enumerate(results)
        ]
```

**Player Rank Lookup**:
```python
def get_player_rank(self, game_id, mode, player_id):
    key = f"leaderboard:{game_id}:{mode}:all_time"
    
    # ZREVRANK returns 0-indexed rank
    rank = self.redis.zrevrank(key, player_id)
    score = self.redis.zscore(key, player_id)
    
    if rank is None:
        return None
    
    return {
        'player_id': player_id,
        'rank': rank + 1,  # Convert to 1-indexed
        'score': score
    }
```

### 3. Time-based Leaderboard Management

```python
class TimeBasedLeaderboardManager:
    def reset_daily_leaderboard(self, game_id, mode):
        key = f"leaderboard:{game_id}:{mode}:daily"
        
        # Archive current leaderboard
        archive_key = f"archive:{key}:{date.today()}"
        self.redis.rename(key, archive_key)
        
        # Set expiry on archive (30 days)
        self.redis.expire(archive_key, 30 * 24 * 3600)
        
        # Create new empty leaderboard
        self.redis.zadd(key, {})
    
    def schedule_resets(self):
        # Use cron jobs or distributed scheduler
        schedule.every().day.at("00:00").do(
            self.reset_daily_leaderboard
        )
        schedule.every().monday.at("00:00").do(
            self.reset_weekly_leaderboard
        )
```

## Advanced Optimizations

### 1. Sharding Strategy

**Problem**: Single Redis instance can't handle billions of players.

**Solution**: Shard by game_id and mode:
```python
def get_shard_key(game_id, mode):
    # Consistent hashing
    hash_value = hashlib.md5(f"{game_id}:{mode}".encode()).hexdigest()
    shard_id = int(hash_value, 16) % NUM_SHARDS
    return f"redis_shard_{shard_id}"

def get_redis_client(game_id, mode):
    shard_key = self.get_shard_key(game_id, mode)
    return self.redis_pool[shard_key]
```

### 2. Caching Layer

```python
class LeaderboardCache:
    def get_top_k_cached(self, game_id, mode, k=100):
        cache_key = f"cache:top_{k}:{game_id}:{mode}"
        
        # Check L1 cache (in-memory)
        if cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # Check L2 cache (Redis)
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Fetch from source
        result = self.query_service.get_top_k(game_id, mode, k)
        
        # Cache with TTL
        self.redis.setex(cache_key, 60, json.dumps(result))
        self.local_cache[cache_key] = result
        
        return result
```

### 3. Approximate Rankings for Large Datasets

For very large leaderboards (billions of players), exact rankings are expensive.

**Solution**: Use probabilistic data structures:
```python
class ApproximateLeaderboard:
    def __init__(self):
        # Use HyperLogLog for cardinality estimation
        self.hll = HyperLogLog()
        
        # Use Count-Min Sketch for frequency estimation
        self.cms = CountMinSketch(width=1000, depth=7)
    
    def estimate_rank(self, player_id, score):
        # Count players with higher scores
        higher_count = 0
        
        # Sample-based estimation
        sample_size = 10000
        samples = self.redis.zrevrange(
            self.leaderboard_key, 
            0, 
            sample_size
        )
        
        for sample_player, sample_score in samples:
            if sample_score > score:
                higher_count += 1
        
        # Extrapolate to full population
        total_players = self.hll.cardinality()
        estimated_rank = (higher_count / sample_size) * total_players
        
        return int(estimated_rank)
```

## Trade-offs & Design Decisions

### Redis vs Other Solutions

| Solution | Pros | Cons |
|----------|------|------|
| **Redis Sorted Sets** | O(log N) operations, simple API | Memory-bound, single-threaded |
| **Cassandra** | Horizontally scalable, persistent | Complex queries, eventual consistency |
| **PostgreSQL** | ACID, complex queries | Poor write performance at scale |
| **Custom B-Tree** | Optimized for use case | High development cost |

**Decision**: Use Redis for hot data + Cassandra for cold storage.

### Consistency Model

- **Strong Consistency**: Not required for gaming leaderboards
- **Eventual Consistency**: Acceptable (players won't notice 1-2 second delays)
- **Monotonic Reads**: Ensure players see their own updates

### Score Update Strategies

**Option 1: Last Write Wins**
```python
self.redis.zadd(key, {player_id: score})
```

**Option 2: Max Score Wins**
```python
current_score = self.redis.zscore(key, player_id) or 0
if score > current_score:
    self.redis.zadd(key, {player_id: score})
```

**Option 3: Cumulative Scores**
```python
self.redis.zincrby(key, score, player_id)
```

## Monitoring & Metrics

```python
class LeaderboardMetrics:
    def track_metrics(self):
        # Latency metrics
        self.statsd.timing('leaderboard.read.latency', read_time)
        self.statsd.timing('leaderboard.write.latency', write_time)
        
        # Throughput metrics
        self.statsd.increment('leaderboard.reads')
        self.statsd.increment('leaderboard.writes')
        
        # Cache hit rate
        hit_rate = cache_hits / (cache_hits + cache_misses)
        self.statsd.gauge('leaderboard.cache.hit_rate', hit_rate)
        
        # Leaderboard size
        size = self.redis.zcard(leaderboard_key)
        self.statsd.gauge('leaderboard.size', size)
```

## Key Takeaways

1. **Redis Sorted Sets** are ideal for real-time leaderboards
2. **Sharding** is essential for scaling beyond single-instance limits
3. **Caching** dramatically reduces read latency
4. **Eventual consistency** is acceptable for gaming use cases
5. **Time-based leaderboards** require careful expiry management
6. **Approximate rankings** can handle billions of players efficiently

## References
- Redis Sorted Sets: https://redis.io/docs/data-types/sorted-sets/
- Cassandra Data Modeling: https://cassandra.apache.org/doc/latest/
- HyperLogLog: https://en.wikipedia.org/wiki/HyperLogLog
