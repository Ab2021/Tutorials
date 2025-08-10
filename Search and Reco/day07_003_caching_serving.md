# Day 7.3: Caching Strategies & Model Serving

## Learning Objectives
By the end of this session, students will be able to:
- Implement intelligent caching strategies for recommendation systems
- Design high-performance model serving architectures
- Optimize cache hit rates and response times
- Handle cache invalidation and consistency challenges
- Build distributed caching solutions with Redis Cluster
- Implement smart cache warming and precomputation strategies

## 1. Intelligent Caching Framework

### 1.1 Multi-Level Cache Architecture

```python
import asyncio
import hashlib
import json
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import redis
import pickle
import numpy as np
from datetime import datetime, timedelta

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISTRIBUTED = "l3_distributed"

class CacheTier(Enum):
    HOT = "hot"      # Frequently accessed data
    WARM = "warm"    # Moderately accessed data
    COLD = "cold"    # Rarely accessed data

@dataclass
class CacheEntry:
    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    tier: CacheTier = CacheTier.WARM

class CacheEvictionStrategy(ABC):
    @abstractmethod
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        pass

class LRUEviction(CacheEvictionStrategy):
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        return cache_size > max_size

class LFUEviction(CacheEvictionStrategy):
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        return cache_size > max_size

class TimeBasedEviction(CacheEvictionStrategy):
    def __init__(self, max_age_seconds: int = 3600):
        self.max_age_seconds = max_age_seconds
    
    def should_evict(self, entry: CacheEntry, cache_size: int, max_size: int) -> bool:
        current_time = time.time()
        return (current_time - entry.timestamp) > self.max_age_seconds

class MultiLevelCache:
    def __init__(self, 
                 l1_size: int = 1000,
                 l2_redis_host: str = "localhost",
                 l2_redis_port: int = 6379,
                 eviction_strategy: CacheEvictionStrategy = None):
        
        # L1 Cache - In-memory
        self.l1_cache: OrderedDict = OrderedDict()
        self.l1_max_size = l1_size
        self.l1_lock = threading.RLock()
        
        # L2 Cache - Redis
        self.redis_client = redis.Redis(
            host=l2_redis_host, 
            port=l2_redis_port, 
            decode_responses=False
        )
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'total_requests': 0
        }
        
        self.eviction_strategy = eviction_strategy or LRUEviction()
        self.cache_warmup_thread = None
        self.warmup_running = False
        
    def _generate_key(self, user_id: int, context: Dict = None) -> str:
        """Generate cache key from user ID and context"""
        context_str = json.dumps(context or {}, sort_keys=True)
        key_data = f"{user_id}:{context_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, user_id: int, context: Dict = None) -> Optional[List]:
        """Retrieve recommendations from cache"""
        cache_key = self._generate_key(user_id, context)
        self.stats['total_requests'] += 1
        
        # Try L1 cache first
        l1_result = self._get_from_l1(cache_key)
        if l1_result is not None:
            self.stats['l1_hits'] += 1
            return l1_result
        
        self.stats['l1_misses'] += 1
        
        # Try L2 cache (Redis)
        l2_result = await self._get_from_l2(cache_key)
        if l2_result is not None:
            self.stats['l2_hits'] += 1
            # Promote to L1
            self._set_to_l1(cache_key, l2_result)
            return l2_result
        
        self.stats['l2_misses'] += 1
        return None
    
    def _get_from_l1(self, key: str) -> Optional[List]:
        """Get from L1 in-memory cache"""
        with self.l1_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                current_time = time.time()
                
                # Check TTL
                if current_time - entry.timestamp > entry.ttl:
                    del self.l1_cache[key]
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = current_time
                
                # Move to end (LRU)
                self.l1_cache.move_to_end(key)
                return entry.value
        return None
    
    async def _get_from_l2(self, key: str) -> Optional[List]:
        """Get from L2 Redis cache"""
        try:
            cached_data = self.redis_client.get(f"rec:{key}")
            if cached_data:
                entry_data = pickle.loads(cached_data)
                current_time = time.time()
                
                # Check TTL
                if current_time - entry_data['timestamp'] > entry_data['ttl']:
                    self.redis_client.delete(f"rec:{key}")
                    return None
                
                return entry_data['value']
        except Exception as e:
            logging.error(f"Redis get error: {e}")
        return None
    
    async def set(self, user_id: int, recommendations: List, 
                  context: Dict = None, ttl: int = 3600):
        """Store recommendations in cache"""
        cache_key = self._generate_key(user_id, context)
        current_time = time.time()
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=recommendations,
            timestamp=current_time,
            ttl=ttl,
            size_bytes=len(str(recommendations))
        )
        
        # Set in L1
        self._set_to_l1(cache_key, recommendations, entry)
        
        # Set in L2 (Redis)
        await self._set_to_l2(cache_key, entry)
    
    def _set_to_l1(self, key: str, value: List, entry: CacheEntry = None):
        """Set in L1 in-memory cache"""
        with self.l1_lock:
            if entry is None:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl=3600,
                    size_bytes=len(str(value))
                )
            
            self.l1_cache[key] = entry
            
            # Evict if necessary
            while len(self.l1_cache) > self.l1_max_size:
                if isinstance(self.eviction_strategy, LRUEviction):
                    oldest_key = next(iter(self.l1_cache))
                    del self.l1_cache[oldest_key]
                elif isinstance(self.eviction_strategy, LFUEviction):
                    # Find least frequently used
                    lfu_key = min(self.l1_cache.keys(), 
                                 key=lambda k: self.l1_cache[k].access_count)
                    del self.l1_cache[lfu_key]
    
    async def _set_to_l2(self, key: str, entry: CacheEntry):
        """Set in L2 Redis cache"""
        try:
            entry_data = {
                'value': entry.value,
                'timestamp': entry.timestamp,
                'ttl': entry.ttl,
                'access_count': entry.access_count
            }
            
            serialized_data = pickle.dumps(entry_data)
            self.redis_client.setex(
                f"rec:{key}", 
                int(entry.ttl), 
                serialized_data
            )
        except Exception as e:
            logging.error(f"Redis set error: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.stats['total_requests']
        if total_requests == 0:
            return {'hit_rate': 0, 'l1_hit_rate': 0, 'l2_hit_rate': 0}
        
        l1_hit_rate = self.stats['l1_hits'] / total_requests
        l2_hit_rate = self.stats['l2_hits'] / total_requests
        overall_hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits']) / total_requests
        
        return {
            'hit_rate': overall_hit_rate,
            'l1_hit_rate': l1_hit_rate,
            'l2_hit_rate': l2_hit_rate,
            'total_requests': total_requests,
            'l1_size': len(self.l1_cache),
            'l1_max_size': self.l1_max_size
        }
```

### 1.2 Smart Cache Warming System

```python
class CacheWarmupStrategy(ABC):
    @abstractmethod
    async def identify_warmup_candidates(self) -> List[Tuple[int, Dict]]:
        pass

class PopularityBasedWarmup(CacheWarmupStrategy):
    def __init__(self, user_activity_db, min_activity_threshold: int = 10):
        self.user_activity_db = user_activity_db
        self.min_activity_threshold = min_activity_threshold
    
    async def identify_warmup_candidates(self) -> List[Tuple[int, Dict]]:
        """Identify popular users for cache warming"""
        # Simulate getting popular users from database
        popular_users = [
            (user_id, {}) for user_id in range(1, 101)  # Top 100 users
        ]
        return popular_users

class TimeBasedWarmup(CacheWarmupStrategy):
    def __init__(self, peak_hours: List[int] = None):
        self.peak_hours = peak_hours or [9, 12, 15, 18, 21]  # Peak traffic hours
    
    async def identify_warmup_candidates(self) -> List[Tuple[int, Dict]]:
        """Identify users likely to be active during peak hours"""
        current_hour = datetime.now().hour
        if current_hour in self.peak_hours:
            # Return more candidates during peak hours
            return [(user_id, {}) for user_id in range(1, 201)]
        return [(user_id, {}) for user_id in range(1, 51)]

class CacheWarmupManager:
    def __init__(self, 
                 cache: MultiLevelCache,
                 recommendation_engine,
                 warmup_strategies: List[CacheWarmupStrategy]):
        
        self.cache = cache
        self.recommendation_engine = recommendation_engine
        self.warmup_strategies = warmup_strategies
        self.warmup_running = False
        self.warmup_stats = defaultdict(int)
    
    async def start_warmup(self, batch_size: int = 10):
        """Start cache warming process"""
        if self.warmup_running:
            return
        
        self.warmup_running = True
        logging.info("Starting cache warmup process")
        
        try:
            # Collect candidates from all strategies
            all_candidates = []
            for strategy in self.warmup_strategies:
                candidates = await strategy.identify_warmup_candidates()
                all_candidates.extend(candidates)
            
            # Remove duplicates
            unique_candidates = list(set(all_candidates))
            
            # Process in batches
            for i in range(0, len(unique_candidates), batch_size):
                batch = unique_candidates[i:i + batch_size]
                await self._process_warmup_batch(batch)
                
                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            logging.info(f"Cache warmup completed. Warmed {len(unique_candidates)} entries")
            
        except Exception as e:
            logging.error(f"Cache warmup error: {e}")
        finally:
            self.warmup_running = False
    
    async def _process_warmup_batch(self, batch: List[Tuple[int, Dict]]):
        """Process a batch of warmup candidates"""
        tasks = []
        for user_id, context in batch:
            task = self._warmup_single_user(user_id, context)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _warmup_single_user(self, user_id: int, context: Dict):
        """Warm cache for a single user"""
        try:
            # Check if already cached
            cached_result = await self.cache.get(user_id, context)
            if cached_result is not None:
                self.warmup_stats['already_cached'] += 1
                return
            
            # Generate recommendations
            recommendations = await self.recommendation_engine.get_recommendations(
                user_id, context
            )
            
            # Cache the results
            await self.cache.set(user_id, recommendations, context, ttl=3600)
            self.warmup_stats['warmed'] += 1
            
        except Exception as e:
            self.warmup_stats['errors'] += 1
            logging.error(f"Warmup error for user {user_id}: {e}")
```

### 1.3 Distributed Cache with Redis Cluster

```python
import redis.sentinel
from rediscluster import RedisCluster

class DistributedCacheManager:
    def __init__(self, 
                 cluster_nodes: List[Dict],
                 sentinel_hosts: List[Tuple] = None,
                 master_name: str = "mymaster"):
        
        self.cluster_nodes = cluster_nodes
        self.sentinel_hosts = sentinel_hosts
        self.master_name = master_name
        
        # Initialize Redis Cluster
        self.cluster_client = RedisCluster(
            startup_nodes=cluster_nodes,
            decode_responses=False,
            skip_full_coverage_check=True
        )
        
        # Initialize Sentinel for high availability
        if sentinel_hosts:
            self.sentinel = redis.sentinel.Sentinel(sentinel_hosts)
            self.master = self.sentinel.master_for(
                master_name, 
                socket_timeout=0.1
            )
        
        self.consistent_hash = ConsistentHash()
        self._initialize_hash_ring()
    
    def _initialize_hash_ring(self):
        """Initialize consistent hashing ring"""
        for i, node in enumerate(self.cluster_nodes):
            node_id = f"{node['host']}:{node['port']}"
            self.consistent_hash.add_node(node_id)
    
    async def get_distributed(self, key: str) -> Optional[Any]:
        """Get from distributed cache"""
        try:
            # Use consistent hashing to determine node
            node_id = self.consistent_hash.get_node(key)
            
            # Get from cluster
            result = self.cluster_client.get(f"dist:{key}")
            if result:
                return pickle.loads(result)
            
        except Exception as e:
            logging.error(f"Distributed cache get error: {e}")
            
            # Fallback to sentinel master
            if hasattr(self, 'master'):
                try:
                    result = self.master.get(f"dist:{key}")
                    if result:
                        return pickle.loads(result)
                except Exception as e2:
                    logging.error(f"Sentinel fallback error: {e2}")
        
        return None
    
    async def set_distributed(self, key: str, value: Any, ttl: int = 3600):
        """Set in distributed cache"""
        try:
            serialized_value = pickle.dumps(value)
            
            # Set in cluster
            self.cluster_client.setex(f"dist:{key}", ttl, serialized_value)
            
            # Also set in sentinel master for redundancy
            if hasattr(self, 'master'):
                self.master.setex(f"dist:{key}", ttl, serialized_value)
                
        except Exception as e:
            logging.error(f"Distributed cache set error: {e}")

class ConsistentHash:
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
    
    def _hash(self, key: str) -> int:
        return hash(key) & 0x7FFFFFFF
    
    def add_node(self, node: str):
        """Add a node to the hash ring"""
        for i in range(self.replicas):
            replica_key = f"{node}:{i}"
            hash_value = self._hash(replica_key)
            self.ring[hash_value] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str):
        """Remove a node from the hash ring"""
        for i in range(self.replicas):
            replica_key = f"{node}:{i}"
            hash_value = self._hash(replica_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> str:
        """Get the node responsible for a key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node with hash >= key hash
        for ring_key in self.sorted_keys:
            if ring_key >= hash_value:
                return self.ring[ring_key]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
```

## 2. High-Performance Model Serving

### 2.1 Model Serving Engine with Connection Pooling

```python
import asyncio
import aiohttp
import uvloop
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Any
import torch
import tensorflow as tf
from queue import Queue
import threading
import time

class ModelServingEngine:
    def __init__(self, 
                 model_registry,
                 max_workers: int = 10,
                 batch_size: int = 32,
                 max_batch_wait_ms: int = 50):
        
        self.model_registry = model_registry
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_batch_wait_ms = max_batch_wait_ms
        
        # Request batching
        self.request_queue = asyncio.Queue()
        self.batch_processor = None
        
        # Model pools
        self.model_pools = {}
        self.pool_locks = {}
        
        # Performance monitoring
        self.serving_stats = {
            'total_requests': 0,
            'batch_requests': 0,
            'avg_latency': 0,
            'model_load_time': {},
            'active_models': set()
        }
        
        # Initialize model pools
        self._initialize_model_pools()
        
        # Start batch processor
        self._start_batch_processor()
    
    def _initialize_model_pools(self):
        """Initialize model pools for each registered model"""
        for model_name in self.model_registry.list_models():
            self.model_pools[model_name] = Queue(maxsize=self.max_workers)
            self.pool_locks[model_name] = threading.Lock()
            
            # Pre-load models into pool
            for _ in range(min(3, self.max_workers)):  # Start with 3 instances
                model_instance = self._load_model(model_name)
                if model_instance:
                    self.model_pools[model_name].put(model_instance)
    
    def _load_model(self, model_name: str):
        """Load a model instance"""
        try:
            start_time = time.time()
            model_config = self.model_registry.get_model_config(model_name)
            
            if model_config['framework'] == 'pytorch':
                model = torch.jit.load(model_config['path'])
                model.eval()
            elif model_config['framework'] == 'tensorflow':
                model = tf.saved_model.load(model_config['path'])
            else:
                raise ValueError(f"Unsupported framework: {model_config['framework']}")
            
            load_time = time.time() - start_time
            self.serving_stats['model_load_time'][model_name] = load_time
            self.serving_stats['active_models'].add(model_name)
            
            return {
                'model': model,
                'framework': model_config['framework'],
                'config': model_config
            }
            
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def _get_model_instance(self, model_name: str):
        """Get a model instance from the pool"""
        try:
            # Try to get from pool first
            model_instance = self.model_pools[model_name].get_nowait()
            return model_instance
        except:
            # Pool is empty, create new instance
            return self._load_model(model_name)
    
    def _return_model_instance(self, model_name: str, model_instance):
        """Return model instance to pool"""
        try:
            self.model_pools[model_name].put_nowait(model_instance)
        except:
            # Pool is full, discard this instance
            pass
    
    async def predict_single(self, 
                           model_name: str, 
                           input_data: Dict) -> Dict:
        """Single prediction request"""
        start_time = time.time()
        
        try:
            # Get model instance
            model_instance = self._get_model_instance(model_name)
            if not model_instance:
                raise HTTPException(status_code=500, 
                                  detail=f"Model {model_name} not available")
            
            # Make prediction
            prediction = await self._run_inference(model_instance, input_data)
            
            # Return model to pool
            self._return_model_instance(model_name, model_instance)
            
            # Update stats
            latency = time.time() - start_time
            self._update_latency_stats(latency)
            self.serving_stats['total_requests'] += 1
            
            return {
                'prediction': prediction,
                'model_name': model_name,
                'latency_ms': latency * 1000
            }
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _run_inference(self, model_instance: Dict, input_data: Dict):
        """Run inference on model"""
        model = model_instance['model']
        framework = model_instance['framework']
        
        if framework == 'pytorch':
            # Convert input to tensor
            if isinstance(input_data['features'], list):
                input_tensor = torch.FloatTensor(input_data['features'])
            else:
                input_tensor = torch.FloatTensor(input_data['features'].values())
            
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0))
                prediction = output.squeeze().numpy().tolist()
                
        elif framework == 'tensorflow':
            # Convert input to tensor
            if isinstance(input_data['features'], list):
                input_tensor = tf.constant([input_data['features']], dtype=tf.float32)
            else:
                input_tensor = tf.constant([list(input_data['features'].values())], 
                                         dtype=tf.float32)
            
            output = model(input_tensor)
            prediction = output.numpy().squeeze().tolist()
        
        return prediction
    
    def _start_batch_processor(self):
        """Start the batch processing loop"""
        async def batch_processor():
            while True:
                batch_requests = []
                
                # Collect requests for batching
                try:
                    # Wait for first request
                    first_request = await asyncio.wait_for(
                        self.request_queue.get(), 
                        timeout=1.0
                    )
                    batch_requests.append(first_request)
                    
                    # Collect additional requests up to batch size
                    start_time = time.time()
                    while (len(batch_requests) < self.batch_size and 
                           (time.time() - start_time) * 1000 < self.max_batch_wait_ms):
                        try:
                            request = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=0.01
                            )
                            batch_requests.append(request)
                        except asyncio.TimeoutError:
                            break
                    
                    # Process batch
                    if batch_requests:
                        await self._process_batch(batch_requests)
                        self.serving_stats['batch_requests'] += 1
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Batch processor error: {e}")
        
        # Start batch processor task
        self.batch_processor = asyncio.create_task(batch_processor())
    
    async def _process_batch(self, batch_requests: List):
        """Process a batch of requests"""
        # Group requests by model
        model_batches = defaultdict(list)
        for request in batch_requests:
            model_name = request['model_name']
            model_batches[model_name].append(request)
        
        # Process each model's batch
        tasks = []
        for model_name, requests in model_batches.items():
            task = self._process_model_batch(model_name, requests)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _process_model_batch(self, model_name: str, requests: List):
        """Process batch for a specific model"""
        try:
            # Get model instance
            model_instance = self._get_model_instance(model_name)
            if not model_instance:
                # Set error for all requests
                for request in requests:
                    request['future'].set_exception(
                        Exception(f"Model {model_name} not available")
                    )
                return
            
            # Prepare batch input
            batch_input = []
            for request in requests:
                if isinstance(request['input_data']['features'], list):
                    batch_input.append(request['input_data']['features'])
                else:
                    batch_input.append(list(request['input_data']['features'].values()))
            
            # Run batch inference
            framework = model_instance['framework']
            model = model_instance['model']
            
            if framework == 'pytorch':
                input_tensor = torch.FloatTensor(batch_input)
                with torch.no_grad():
                    batch_output = model(input_tensor)
                    batch_predictions = batch_output.numpy().tolist()
            
            elif framework == 'tensorflow':
                input_tensor = tf.constant(batch_input, dtype=tf.float32)
                batch_output = model(input_tensor)
                batch_predictions = batch_output.numpy().tolist()
            
            # Set results for each request
            for i, request in enumerate(requests):
                result = {
                    'prediction': batch_predictions[i],
                    'model_name': model_name,
                    'batch_size': len(requests)
                }
                request['future'].set_result(result)
            
            # Return model to pool
            self._return_model_instance(model_name, model_instance)
            
        except Exception as e:
            # Set error for all requests
            for request in requests:
                request['future'].set_exception(e)
    
    async def predict_batch(self, 
                          model_name: str, 
                          input_data: Dict) -> Dict:
        """Batch prediction request"""
        # Create future for result
        future = asyncio.Future()
        
        # Add to batch queue
        request = {
            'model_name': model_name,
            'input_data': input_data,
            'future': future
        }
        
        await self.request_queue.put(request)
        
        # Wait for result
        result = await future
        return result
    
    def _update_latency_stats(self, latency: float):
        """Update latency statistics"""
        current_avg = self.serving_stats['avg_latency']
        total_requests = self.serving_stats['total_requests']
        
        # Exponential moving average
        alpha = 0.1  # Smoothing factor
        self.serving_stats['avg_latency'] = (
            alpha * latency + (1 - alpha) * current_avg
        )
    
    def get_serving_stats(self) -> Dict:
        """Get serving statistics"""
        return {
            **self.serving_stats,
            'avg_latency_ms': self.serving_stats['avg_latency'] * 1000,
            'active_models': list(self.serving_stats['active_models']),
            'pool_sizes': {
                name: pool.qsize() 
                for name, pool in self.model_pools.items()
            }
        }
```

### 2.2 FastAPI Application with Advanced Features

```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import asyncio
from typing import Optional
import jwt
import time

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    user_id: int
    features: Dict[str, float]
    model_name: str = "default"
    context: Optional[Dict] = None

class PredictionResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    scores: List[float]
    model_name: str
    latency_ms: float
    cached: bool = False

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]
    batch_id: Optional[str] = None

# FastAPI app initialization
def create_serving_app(cache: MultiLevelCache, 
                      serving_engine: ModelServingEngine,
                      warmup_manager: CacheWarmupManager) -> FastAPI:
    
    app = FastAPI(
        title="AI/ML Recommendation Serving API",
        description="High-performance recommendation serving with caching",
        version="1.0.0"
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Security
    security = HTTPBearer()
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        try:
            # Simple JWT verification (implement proper verification in production)
            token = credentials.credentials
            payload = jwt.decode(token, "secret", algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "cache_stats": cache.get_cache_stats(),
            "serving_stats": serving_engine.get_serving_stats()
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: PredictionRequest,
        background_tasks: BackgroundTasks,
        user_data: dict = Depends(verify_token)
    ):
        """Single prediction endpoint with caching"""
        start_time = time.time()
        
        # Check cache first
        cached_result = await cache.get(request.user_id, request.context)
        
        if cached_result is not None:
            return PredictionResponse(
                user_id=request.user_id,
                recommendations=cached_result['recommendations'],
                scores=cached_result['scores'],
                model_name=cached_result['model_name'],
                latency_ms=(time.time() - start_time) * 1000,
                cached=True
            )
        
        # Get prediction from serving engine
        try:
            prediction = await serving_engine.predict_single(
                request.model_name,
                {
                    'user_id': request.user_id,
                    'features': request.features,
                    'context': request.context
                }
            )
            
            # Format response
            recommendations = prediction['prediction'][:10]  # Top 10
            scores = [float(score) for score in recommendations]
            item_ids = list(range(len(recommendations)))  # Placeholder
            
            result = {
                'recommendations': item_ids,
                'scores': scores,
                'model_name': request.model_name
            }
            
            # Cache the result in background
            background_tasks.add_task(
                cache.set,
                request.user_id,
                result,
                request.context,
                3600
            )
            
            return PredictionResponse(
                user_id=request.user_id,
                recommendations=item_ids,
                scores=scores,
                model_name=request.model_name,
                latency_ms=(time.time() - start_time) * 1000,
                cached=False
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict/batch")
    async def predict_batch(
        request: BatchPredictionRequest,
        background_tasks: BackgroundTasks,
        user_data: dict = Depends(verify_token)
    ):
        """Batch prediction endpoint"""
        start_time = time.time()
        
        try:
            # Process batch requests
            tasks = []
            for pred_request in request.requests:
                task = serving_engine.predict_batch(
                    pred_request.model_name,
                    {
                        'user_id': pred_request.user_id,
                        'features': pred_request.features,
                        'context': pred_request.context
                    }
                )
                tasks.append(task)
            
            # Wait for all predictions
            batch_results = await asyncio.gather(*tasks)
            
            # Format response
            responses = []
            for i, result in enumerate(batch_results):
                pred_request = request.requests[i]
                
                recommendations = result['prediction'][:10]
                scores = [float(score) for score in recommendations]
                item_ids = list(range(len(recommendations)))
                
                responses.append(PredictionResponse(
                    user_id=pred_request.user_id,
                    recommendations=item_ids,
                    scores=scores,
                    model_name=pred_request.model_name,
                    latency_ms=result.get('latency_ms', 0),
                    cached=False
                ))
                
                # Cache each result in background
                cache_data = {
                    'recommendations': item_ids,
                    'scores': scores,
                    'model_name': pred_request.model_name
                }
                background_tasks.add_task(
                    cache.set,
                    pred_request.user_id,
                    cache_data,
                    pred_request.context,
                    3600
                )
            
            return {
                'batch_id': request.batch_id,
                'results': responses,
                'total_latency_ms': (time.time() - start_time) * 1000,
                'batch_size': len(request.requests)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/cache/warmup")
    async def trigger_cache_warmup(
        background_tasks: BackgroundTasks,
        user_data: dict = Depends(verify_token)
    ):
        """Trigger cache warmup process"""
        background_tasks.add_task(warmup_manager.start_warmup)
        return {"message": "Cache warmup started"}
    
    @app.get("/cache/stats")
    async def get_cache_stats(user_data: dict = Depends(verify_token)):
        """Get cache statistics"""
        return cache.get_cache_stats()
    
    @app.get("/serving/stats")
    async def get_serving_stats(user_data: dict = Depends(verify_token)):
        """Get serving statistics"""
        return serving_engine.get_serving_stats()
    
    @app.delete("/cache/clear")
    async def clear_cache(
        user_id: Optional[int] = None,
        user_data: dict = Depends(verify_token)
    ):
        """Clear cache (specific user or all)"""
        # Implement cache clearing logic
        return {"message": "Cache cleared"}
    
    return app

# Application startup
async def start_serving_application():
    """Start the complete serving application"""
    
    # Initialize components
    cache = MultiLevelCache(
        l1_size=1000,
        eviction_strategy=LRUEviction()
    )
    
    # Mock model registry and serving engine
    class MockModelRegistry:
        def list_models(self):
            return ['collaborative_filtering', 'content_based', 'hybrid']
        
        def get_model_config(self, model_name):
            return {
                'framework': 'pytorch',
                'path': f'/models/{model_name}.pt',
                'version': '1.0'
            }
    
    model_registry = MockModelRegistry()
    serving_engine = ModelServingEngine(model_registry)
    
    # Initialize warmup strategies
    warmup_strategies = [
        PopularityBasedWarmup(None),
        TimeBasedWarmup()
    ]
    warmup_manager = CacheWarmupManager(cache, serving_engine, warmup_strategies)
    
    # Create FastAPI app
    app = create_serving_app(cache, serving_engine, warmup_manager)
    
    # Start server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop="uvloop",
        workers=1,
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # Set up uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    asyncio.run(start_serving_application())
```

## 3. Study Questions

### Beginner Level
1. What are the different levels in a multi-level cache hierarchy?
2. Why is cache warming important for recommendation systems?
3. What is the difference between LRU and LFU cache eviction strategies?
4. How does consistent hashing help in distributed caching?
5. What are the benefits of request batching in model serving?

### Intermediate Level
1. Implement a cache invalidation strategy for when user preferences change significantly.
2. Design a cache partitioning strategy based on user segments (new users, power users, etc.).
3. Create a cache performance monitoring system with alerting.
4. Implement cache compression to optimize memory usage.
5. Design a graceful degradation strategy when cache services are unavailable.

### Advanced Level
1. Implement a machine learning-based cache replacement policy that predicts future access patterns.
2. Design a multi-region cache replication system with conflict resolution.
3. Create an adaptive batching system that adjusts batch sizes based on system load.
4. Implement cache-aware load balancing for model serving clusters.
5. Design a cache warming system that uses reinforcement learning to optimize warming strategies.

## 4. Practical Exercises

1. **Cache Performance Analysis**: Build a system to analyze cache hit rates across different user segments and time periods.

2. **Distributed Cache Simulation**: Implement a simulation of a distributed cache system with node failures and recovery.

3. **Model Serving Optimization**: Create a benchmark comparing single predictions vs. batch predictions across different batch sizes.

4. **Cache-Aware Recommendation**: Implement a recommendation system that considers cache hit probability when ranking items.

5. **Real-time Cache Monitoring**: Build a dashboard showing real-time cache performance metrics and model serving statistics.