# Day 43: Gaming Leaderboards - Deep Dive

## Advanced Implementation Patterns

### 1. Multi-Dimensional Leaderboards

Many games require rankings across multiple dimensions (kills, deaths, assists, win rate, etc.).

**Composite Score Approach**:
```python
class MultiDimensionalLeaderboard:
    def calculate_composite_score(self, stats):
        """
        Calculate weighted composite score
        Example: ELO-style rating for competitive games
        """
        weights = {
            'kills': 3.0,
            'assists': 1.5,
            'deaths': -2.0,
            'win_rate': 10.0,
            'objectives': 2.0
        }
        
        score = sum(stats.get(key, 0) * weight 
                   for key, weight in weights.items())
        
        # Normalize to prevent negative scores
        return max(0, score + 1000)
    
    def update_multi_dimensional(self, player_id, game_id, stats):
        # Update individual stat leaderboards
        for stat, value in stats.items():
            key = f"leaderboard:{game_id}:{stat}"
            self.redis.zadd(key, {player_id: value})
        
        # Update composite leaderboard
        composite_score = self.calculate_composite_score(stats)
        composite_key = f"leaderboard:{game_id}:composite"
        self.redis.zadd(composite_key, {player_id: composite_score})
```

**Separate Leaderboards Approach**:
```python
def get_multi_stat_rankings(self, player_id, game_id):
    """Get player's rank across all stat categories"""
    stats = ['kills', 'deaths', 'assists', 'win_rate']
    
    pipeline = self.redis.pipeline()
    for stat in stats:
        key = f"leaderboard:{game_id}:{stat}"
        pipeline.zrevrank(key, player_id)
        pipeline.zscore(key, player_id)
    
    results = pipeline.execute()
    
    rankings = {}
    for i, stat in enumerate(stats):
        rank = results[i*2]
        score = results[i*2 + 1]
        rankings[stat] = {
            'rank': rank + 1 if rank is not None else None,
            'score': score
        }
    
    return rankings
```

### 2. Regional/Geo-Distributed Leaderboards

**Architecture**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  US Region  │     │  EU Region  │     │ APAC Region │
│   Redis     │     │   Redis     │     │   Redis     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Global    │
                    │ Aggregator  │
                    └─────────────┘
```

**Implementation**:
```python
class RegionalLeaderboard:
    def __init__(self):
        self.regions = {
            'us': RedisClient('us-redis.example.com'),
            'eu': RedisClient('eu-redis.example.com'),
            'apac': RedisClient('apac-redis.example.com')
        }
    
    def update_regional(self, player_id, game_id, score, region):
        """Update player's regional leaderboard"""
        key = f"leaderboard:{game_id}:regional:{region}"
        redis_client = self.regions[region]
        redis_client.zadd(key, {player_id: score})
    
    def get_global_top_k(self, game_id, k=100):
        """Merge regional leaderboards for global view"""
        # Fetch top K from each region
        regional_tops = []
        for region, client in self.regions.items():
            key = f"leaderboard:{game_id}:regional:{region}"
            top = client.zrevrange(key, 0, k-1, withscores=True)
            regional_tops.extend(top)
        
        # Merge and sort
        merged = sorted(regional_tops, key=lambda x: x[1], reverse=True)
        
        return merged[:k]
```

**Optimization with Background Aggregation**:
```python
class GlobalLeaderboardAggregator:
    def aggregate_periodically(self, game_id):
        """Run every 5 minutes to update global leaderboard"""
        global_key = f"leaderboard:{game_id}:global"
        
        # Use Redis ZUNIONSTORE for efficient merging
        regional_keys = [
            f"leaderboard:{game_id}:regional:us",
            f"leaderboard:{game_id}:regional:eu",
            f"leaderboard:{game_id}:regional:apac"
        ]
        
        # Merge with MAX aggregate (highest score wins)
        self.redis.zunionstore(
            global_key,
            regional_keys,
            aggregate='MAX'
        )
        
        # Set expiry
        self.redis.expire(global_key, 600)  # 10 minutes
```

### 3. Cheating Detection & Score Validation

**Anomaly Detection**:
```python
class CheatDetector:
    def __init__(self):
        self.score_history = defaultdict(list)
        self.max_score_jump = 1000  # Game-specific threshold
    
    def validate_score_update(self, player_id, new_score):
        """Detect suspicious score jumps"""
        history = self.score_history[player_id]
        
        if not history:
            # First score, accept
            history.append(new_score)
            return True
        
        last_score = history[-1]
        score_jump = new_score - last_score
        
        # Check for impossible score jump
        if score_jump > self.max_score_jump:
            self.flag_for_review(player_id, last_score, new_score)
            return False
        
        # Check for statistical anomaly
        if len(history) >= 10:
            avg_jump = np.mean(np.diff(history))
            std_jump = np.std(np.diff(history))
            
            # Z-score > 3 is suspicious
            z_score = (score_jump - avg_jump) / (std_jump + 1e-6)
            if abs(z_score) > 3:
                self.flag_for_review(player_id, last_score, new_score)
                return False
        
        history.append(new_score)
        return True
    
    def flag_for_review(self, player_id, old_score, new_score):
        """Queue for manual review"""
        self.redis.sadd('flagged_players', player_id)
        self.redis.hset(
            f'flag:{player_id}',
            mapping={
                'old_score': old_score,
                'new_score': new_score,
                'timestamp': time.time()
            }
        )
```

**Rate Limiting Score Updates**:
```python
class ScoreUpdateRateLimiter:
    def check_rate_limit(self, player_id):
        """Prevent score update spam"""
        key = f"rate_limit:score_update:{player_id}"
        
        # Allow 10 updates per minute
        current = self.redis.incr(key)
        if current == 1:
            self.redis.expire(key, 60)
        
        if current > 10:
            raise RateLimitExceeded(
                f"Player {player_id} exceeded score update limit"
            )
```

### 4. Efficient Pagination & Neighbor Queries

**Get Players Around a Specific Rank**:
```python
class LeaderboardPagination:
    def get_neighbors(self, game_id, player_id, context=5):
        """
        Get players ranked around the target player
        Example: If player is rank 1000, get ranks 995-1005
        """
        key = f"leaderboard:{game_id}:all_time"
        
        # Get player's rank
        rank = self.redis.zrevrank(key, player_id)
        if rank is None:
            return None
        
        # Calculate range
        start = max(0, rank - context)
        end = rank + context
        
        # Fetch range with scores
        results = self.redis.zrevrange(
            key,
            start,
            end,
            withscores=True
        )
        
        return [
            {
                'rank': start + idx + 1,
                'player_id': pid,
                'score': score,
                'is_current_player': pid == player_id
            }
            for idx, (pid, score) in enumerate(results)
        ]
```

**Cursor-based Pagination**:
```python
def paginate_leaderboard(self, game_id, cursor=None, page_size=50):
    """
    Efficient pagination using score as cursor
    """
    key = f"leaderboard:{game_id}:all_time"
    
    if cursor is None:
        # First page: get top scores
        results = self.redis.zrevrange(
            key, 0, page_size - 1, withscores=True
        )
    else:
        # Subsequent pages: get scores below cursor
        results = self.redis.zrevrangebyscore(
            key,
            max=cursor,
            min='-inf',
            start=0,
            num=page_size,
            withscores=True
        )
    
    if not results:
        return {'data': [], 'next_cursor': None}
    
    next_cursor = results[-1][1] - 1  # Score of last item - 1
    
    return {
        'data': [
            {'player_id': pid, 'score': score}
            for pid, score in results
        ],
        'next_cursor': next_cursor
    }
```

### 5. Handling Ties

**Lexicographical Ordering**:
```python
def handle_ties_with_timestamp(self, game_id, player_id, score):
    """
    Use timestamp as tiebreaker
    Encode: score * 1e10 + (MAX_TIMESTAMP - timestamp)
    """
    timestamp = time.time()
    MAX_TIMESTAMP = 2**32  # Arbitrary large number
    
    # Earlier timestamp = higher rank
    composite_score = score * 1e10 + (MAX_TIMESTAMP - timestamp)
    
    key = f"leaderboard:{game_id}:all_time"
    self.redis.zadd(key, {player_id: composite_score})

def get_rank_with_ties(self, game_id, player_id):
    """
    Get rank considering ties
    """
    key = f"leaderboard:{game_id}:all_time"
    
    # Get player's score
    score = self.redis.zscore(key, player_id)
    if score is None:
        return None
    
    # Count players with strictly higher scores
    higher_count = self.redis.zcount(key, score + 1, '+inf')
    
    # Rank is higher_count + 1
    return higher_count + 1
```

### 6. Memory Optimization

**Problem**: Storing billions of players in memory is expensive.

**Solution 1: Evict Inactive Players**:
```python
class InactivePlayerEviction:
    def evict_inactive(self, game_id, inactive_days=30):
        """Remove players who haven't played in 30 days"""
        key = f"leaderboard:{game_id}:all_time"
        activity_key = f"player_activity:{game_id}"
        
        cutoff = time.time() - (inactive_days * 86400)
        
        # Get all players
        all_players = self.redis.zrange(key, 0, -1)
        
        for player_id in all_players:
            last_active = self.redis.hget(activity_key, player_id)
            
            if last_active and float(last_active) < cutoff:
                # Archive to persistent storage
                self.archive_player(game_id, player_id)
                
                # Remove from Redis
                self.redis.zrem(key, player_id)
```

**Solution 2: Tiered Storage**:
```python
class TieredLeaderboard:
    """
    Keep only top 1M players in Redis
    Store rest in Cassandra
    """
    MAX_REDIS_PLAYERS = 1_000_000
    
    def update_with_tiering(self, game_id, player_id, score):
        key = f"leaderboard:{game_id}:all_time"
        
        # Always update Cassandra
        self.cassandra.update(game_id, player_id, score)
        
        # Check if player qualifies for Redis tier
        current_size = self.redis.zcard(key)
        
        if current_size < self.MAX_REDIS_PLAYERS:
            # Redis not full, add player
            self.redis.zadd(key, {player_id: score})
        else:
            # Check if score beats lowest in Redis
            lowest = self.redis.zrange(key, 0, 0, withscores=True)
            if lowest and score > lowest[0][1]:
                # Remove lowest, add new player
                self.redis.zrem(key, lowest[0][0])
                self.redis.zadd(key, {player_id: score})
```

### 7. Real-time Updates with WebSockets

```python
class RealtimeLeaderboardUpdates:
    def __init__(self):
        self.pubsub = self.redis.pubsub()
        self.websocket_connections = {}
    
    async def broadcast_update(self, game_id, update_data):
        """Broadcast leaderboard changes to connected clients"""
        channel = f"leaderboard_updates:{game_id}"
        
        # Publish to Redis pub/sub
        self.redis.publish(channel, json.dumps(update_data))
    
    async def handle_websocket(self, websocket, game_id):
        """Handle WebSocket connection for real-time updates"""
        channel = f"leaderboard_updates:{game_id}"
        
        # Subscribe to updates
        self.pubsub.subscribe(channel)
        
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    # Send update to client
                    await websocket.send(message['data'])
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.pubsub.unsubscribe(channel)
```

### 8. Batch Processing for Historical Analysis

```python
class LeaderboardAnalytics:
    def generate_daily_snapshot(self, game_id):
        """Create daily snapshot for trend analysis"""
        key = f"leaderboard:{game_id}:all_time"
        snapshot_key = f"snapshot:{game_id}:{date.today()}"
        
        # Copy entire sorted set
        self.redis.zunionstore(snapshot_key, [key])
        
        # Set expiry (keep for 1 year)
        self.redis.expire(snapshot_key, 365 * 86400)
    
    def analyze_rank_changes(self, game_id, player_id):
        """Track player's rank progression over time"""
        today = date.today()
        snapshots = []
        
        for days_ago in range(7):
            snapshot_date = today - timedelta(days=days_ago)
            snapshot_key = f"snapshot:{game_id}:{snapshot_date}"
            
            rank = self.redis.zrevrank(snapshot_key, player_id)
            score = self.redis.zscore(snapshot_key, player_id)
            
            if rank is not None:
                snapshots.append({
                    'date': str(snapshot_date),
                    'rank': rank + 1,
                    'score': score
                })
        
        return snapshots
```

## Performance Benchmarks

**Redis Sorted Set Operations**:
- `ZADD`: O(log N) - ~1ms for 1M entries
- `ZREVRANK`: O(log N) - ~1ms for 1M entries
- `ZREVRANGE`: O(log N + M) - ~2ms for top 100 from 1M entries
- `ZCOUNT`: O(log N) - ~1ms for 1M entries

**Throughput**:
- Single Redis instance: ~100K ops/sec
- With sharding (10 shards): ~1M ops/sec
- With caching: ~10M reads/sec

## Production Checklist

- [ ] Implement score validation and sanitization
- [ ] Set up monitoring for latency and throughput
- [ ] Configure Redis persistence (RDB + AOF)
- [ ] Implement backup strategy for leaderboard data
- [ ] Set up alerts for anomalous score updates
- [ ] Test failover scenarios
- [ ] Implement rate limiting on API endpoints
- [ ] Set up CDN for static leaderboard pages
- [ ] Configure auto-scaling for read replicas
- [ ] Implement graceful degradation for Redis failures

## Key Takeaways

1. **Redis Sorted Sets** provide O(log N) performance for all operations
2. **Sharding** enables horizontal scaling beyond single-instance limits
3. **Tiered storage** optimizes memory usage for massive player bases
4. **Cheating detection** is critical for competitive integrity
5. **Real-time updates** enhance user engagement
6. **Regional leaderboards** reduce latency for global games
7. **Pagination** must be efficient for large datasets
8. **Monitoring** is essential for production systems
