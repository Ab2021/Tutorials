# Day 7.1: Real-time Data Processing and Streaming Architectures

## Learning Objectives
- Master real-time data processing concepts and architectures
- Implement streaming recommendation systems with Apache Kafka and Flink
- Design event-driven recommendation pipelines
- Build real-time feature engineering and model inference systems
- Understand stream processing patterns and fault tolerance mechanisms

## 1. Streaming Data Fundamentals

### Event-Driven Architecture for Recommendations

```python
import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import logging
from enum import Enum
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

class EventType(Enum):
    USER_INTERACTION = "user_interaction"
    ITEM_UPDATE = "item_update"
    USER_PROFILE_UPDATE = "user_profile_update"
    RECOMMENDATION_REQUEST = "recommendation_request"
    RECOMMENDATION_RESPONSE = "recommendation_response"
    SYSTEM_METRIC = "system_metric"

@dataclass
class StreamEvent:
    """Base streaming event class"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        event_type = EventType(data['event_type'])
        timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            event_id=data['event_id'],
            event_type=event_type,
            timestamp=timestamp,
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            data=data.get('data', {})
        )

@dataclass
class UserInteractionEvent(StreamEvent):
    """User interaction event"""
    item_id: str = None
    interaction_type: str = None  # click, view, purchase, rating
    interaction_value: float = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.USER_INTERACTION
        if self.context is None:
            self.context = {}

@dataclass
class RecommendationRequestEvent(StreamEvent):
    """Recommendation request event"""
    num_recommendations: int = 10
    request_context: Dict[str, Any] = None
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = EventType.RECOMMENDATION_REQUEST
        if self.request_context is None:
            self.request_context = {}
        if self.filters is None:
            self.filters = {}

class StreamProcessor(ABC):
    """Abstract base class for stream processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.metrics = defaultdict(int)
        
    @abstractmethod
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process a single event"""
        pass
        
    async def start(self):
        """Start the processor"""
        self.is_running = True
        logging.info(f"Started processor: {self.name}")
        
    async def stop(self):
        """Stop the processor"""
        self.is_running = False
        logging.info(f"Stopped processor: {self.name}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics"""
        return dict(self.metrics)

class EventStream:
    """In-memory event stream implementation"""
    
    def __init__(self, name: str, max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self.events = deque(maxlen=max_size)
        self.subscribers = []
        self.lock = threading.Lock()
        
    def publish(self, event: StreamEvent):
        """Publish event to stream"""
        with self.lock:
            self.events.append(event)
            
        # Notify subscribers asynchronously
        for subscriber in self.subscribers:
            try:
                asyncio.create_task(subscriber(event))
            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")
                
    def subscribe(self, callback: Callable[[StreamEvent], None]):
        """Subscribe to stream events"""
        self.subscribers.append(callback)
        
    def get_recent_events(self, count: int = 100) -> List[StreamEvent]:
        """Get recent events from stream"""
        with self.lock:
            return list(self.events)[-count:]

class StreamingRecommendationEngine:
    """Real-time recommendation engine with streaming data processing"""
    
    def __init__(self):
        self.event_streams = {}
        self.processors = {}
        self.user_profiles = {}
        self.item_features = {}
        self.recent_interactions = defaultdict(lambda: deque(maxlen=100))
        self.recommendation_cache = {}
        
        # Initialize streams
        self.event_streams['interactions'] = EventStream('user_interactions')
        self.event_streams['requests'] = EventStream('recommendation_requests')
        self.event_streams['responses'] = EventStream('recommendation_responses')
        
        # Setup processors
        self._setup_processors()
        
    def _setup_processors(self):
        """Setup stream processors"""
        
        # User profile updater
        profile_updater = UserProfileProcessor()
        self.processors['profile_updater'] = profile_updater
        self.event_streams['interactions'].subscribe(
            lambda event: asyncio.create_task(profile_updater.process_event(event))
        )
        
        # Real-time recommender
        recommender = RealTimeRecommendationProcessor(self)
        self.processors['recommender'] = recommender
        self.event_streams['requests'].subscribe(
            lambda event: asyncio.create_task(recommender.process_event(event))
        )
        
    async def process_user_interaction(self, user_id: str, item_id: str, 
                                     interaction_type: str, interaction_value: float = 1.0,
                                     context: Dict[str, Any] = None):
        """Process user interaction event"""
        
        event = UserInteractionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type,
            interaction_value=interaction_value,
            context=context or {}
        )
        
        # Update recent interactions
        self.recent_interactions[user_id].append({
            'item_id': item_id,
            'interaction_type': interaction_type,
            'interaction_value': interaction_value,
            'timestamp': event.timestamp,
            'context': context or {}
        })
        
        # Publish to stream
        self.event_streams['interactions'].publish(event)
        
        # Invalidate cache for this user
        if user_id in self.recommendation_cache:
            del self.recommendation_cache[user_id]
            
    async def get_recommendations(self, user_id: str, num_recommendations: int = 10,
                                request_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get real-time recommendations"""
        
        # Check cache first
        cache_key = f"{user_id}:{num_recommendations}"
        if cache_key in self.recommendation_cache:
            cached_result = self.recommendation_cache[cache_key]
            if datetime.now() - cached_result['timestamp'] < timedelta(minutes=5):
                return cached_result['recommendations']
        
        # Create recommendation request event
        request_event = RecommendationRequestEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            num_recommendations=num_recommendations,
            request_context=request_context or {}
        )
        
        # Publish request
        self.event_streams['requests'].publish(request_event)
        
        # For synchronous interface, generate recommendations directly
        recommendations = await self._generate_recommendations(user_id, num_recommendations, request_context)
        
        # Cache result
        self.recommendation_cache[cache_key] = {
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
        
        return recommendations
        
    async def _generate_recommendations(self, user_id: str, num_recommendations: int,
                                      context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate recommendations using real-time data"""
        
        # Get user's recent interactions
        recent_interactions = list(self.recent_interactions[user_id])
        
        if not recent_interactions:
            # Cold start - return popular items
            return self._get_popular_recommendations(num_recommendations)
        
        # Simple collaborative filtering based on recent interactions
        user_items = set(interaction['item_id'] for interaction in recent_interactions)
        
        # Find similar users based on recent interactions
        similar_users = self._find_similar_users(user_id, user_items)
        
        # Generate recommendations
        recommendations = []
        candidate_items = set()
        
        # Collect items from similar users
        for similar_user_id, similarity in similar_users[:10]:  # Top 10 similar users
            similar_user_interactions = list(self.recent_interactions[similar_user_id])
            for interaction in similar_user_interactions:
                if interaction['item_id'] not in user_items:
                    candidate_items.add(interaction['item_id'])
        
        # Score and rank candidates
        scored_items = []
        for item_id in candidate_items:
            score = self._calculate_item_score(user_id, item_id, recent_interactions, context)
            scored_items.append({
                'item_id': item_id,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })
        
        # Sort by score and return top N
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        return scored_items[:num_recommendations]
    
    def _find_similar_users(self, user_id: str, user_items: set) -> List[Tuple[str, float]]:
        """Find users similar to the given user"""
        similarities = []
        
        for other_user_id, interactions in self.recent_interactions.items():
            if other_user_id == user_id:
                continue
                
            other_items = set(interaction['item_id'] for interaction in interactions)
            
            # Jaccard similarity
            intersection = len(user_items & other_items)
            union = len(user_items | other_items)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Threshold for similarity
                    similarities.append((other_user_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _calculate_item_score(self, user_id: str, item_id: str, 
                            recent_interactions: List[Dict], context: Dict[str, Any] = None) -> float:
        """Calculate score for an item"""
        
        # Base score
        score = 0.5
        
        # Boost based on interaction types in user history
        interaction_weights = {'purchase': 1.0, 'click': 0.5, 'view': 0.2, 'rating': 0.8}
        
        for interaction in recent_interactions:
            interaction_type = interaction['interaction_type']
            weight = interaction_weights.get(interaction_type, 0.3)
            
            # Time decay
            time_diff = (datetime.now() - interaction['timestamp']).total_seconds() / 3600  # hours
            decay_factor = np.exp(-time_diff / 24)  # Decay over 24 hours
            
            score += weight * decay_factor * 0.1
        
        # Context-based adjustments
        if context:
            if context.get('device') == 'mobile':
                score *= 1.1  # Boost for mobile
            if context.get('time_of_day') in ['evening', 'night']:
                score *= 1.05  # Evening boost
        
        # Add some randomness for exploration
        score += np.random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _get_popular_recommendations(self, num_recommendations: int) -> List[Dict[str, Any]]:
        """Get popular items for cold start"""
        
        # Count item interactions across all users
        item_counts = defaultdict(int)
        
        for user_interactions in self.recent_interactions.values():
            for interaction in user_interactions:
                item_counts[interaction['item_id']] += 1
        
        # Sort by popularity
        popular_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, count in popular_items[:num_recommendations]:
            recommendations.append({
                'item_id': item_id,
                'score': min(1.0, count / 100.0),  # Normalize score
                'timestamp': datetime.now().isoformat()
            })
        
        # Fill with random items if not enough popular items
        if len(recommendations) < num_recommendations:
            for i in range(len(recommendations), num_recommendations):
                recommendations.append({
                    'item_id': f'item_{i}', 
                    'score': 0.1,
                    'timestamp': datetime.now().isoformat()
                })
        
        return recommendations

class UserProfileProcessor(StreamProcessor):
    """Processor for updating user profiles in real-time"""
    
    def __init__(self):
        super().__init__("UserProfileProcessor")
        self.user_profiles = {}
        
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process user interaction event to update profile"""
        
        if event.event_type != EventType.USER_INTERACTION:
            return None
            
        self.metrics['events_processed'] += 1
        
        user_id = event.user_id
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'interaction_count': 0,
                'preferred_categories': defaultdict(float),
                'interaction_types': defaultdict(int),
                'last_activity': event.timestamp,
                'session_count': 0,
                'avg_session_length': 0.0
            }
        
        profile = self.user_profiles[user_id]
        
        # Update interaction count
        profile['interaction_count'] += 1
        profile['last_activity'] = event.timestamp
        
        # Update interaction types
        if hasattr(event, 'interaction_type'):
            profile['interaction_types'][event.interaction_type] += 1
        
        # Update preferred categories (if available in event data)
        if 'category' in event.data:
            category = event.data['category']
            profile['preferred_categories'][category] += 1.0
        
        # Session tracking
        if event.session_id:
            # Simple session tracking logic
            profile['session_count'] += 1
        
        self.metrics['profiles_updated'] += 1
        
        return None  # No output event
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile"""
        return self.user_profiles.get(user_id, {})

class RealTimeRecommendationProcessor(StreamProcessor):
    """Processor for handling recommendation requests"""
    
    def __init__(self, recommendation_engine):
        super().__init__("RealTimeRecommendationProcessor")
        self.recommendation_engine = recommendation_engine
        
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process recommendation request event"""
        
        if event.event_type != EventType.RECOMMENDATION_REQUEST:
            return None
            
        self.metrics['requests_processed'] += 1
        
        try:
            # Generate recommendations
            recommendations = await self.recommendation_engine._generate_recommendations(
                event.user_id,
                event.data.get('num_recommendations', 10),
                event.data.get('request_context', {})
            )
            
            # Create response event
            response_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.RECOMMENDATION_RESPONSE,
                timestamp=datetime.now(),
                user_id=event.user_id,
                session_id=event.session_id,
                data={
                    'request_id': event.event_id,
                    'recommendations': recommendations,
                    'processing_time_ms': (datetime.now() - event.timestamp).total_seconds() * 1000
                }
            )
            
            # Publish response
            self.recommendation_engine.event_streams['responses'].publish(response_event)
            
            self.metrics['recommendations_generated'] += 1
            
            return response_event
            
        except Exception as e:
            logging.error(f"Error processing recommendation request: {e}")
            self.metrics['errors'] += 1
            return None
```

## 2. Apache Kafka Integration

### Kafka-based Streaming Architecture

```python
import asyncio
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
import threading
from typing import Dict, List, Callable

class KafkaStreamProcessor:
    """Kafka-based stream processor for recommendations"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumers = {}
        self.processors = {}
        self.running = False
        
    def setup_producer(self):
        """Setup Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                batch_size=16384,
                linger_ms=10,  # Small delay to batch messages
                compression_type='gzip'
            )
            logging.info("Kafka producer initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def setup_consumer(self, topic: str, group_id: str, processor: Callable):
        """Setup Kafka consumer for a topic"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                consumer_timeout_ms=1000
            )
            
            self.consumers[topic] = consumer
            self.processors[topic] = processor
            
            logging.info(f"Kafka consumer initialized for topic: {topic}")
            
        except Exception as e:
            logging.error(f"Failed to initialize Kafka consumer for topic {topic}: {e}")
            raise
    
    async def produce_event(self, topic: str, event: StreamEvent, key: str = None):
        """Produce event to Kafka topic"""
        if not self.producer:
            self.setup_producer()
            
        try:
            # Convert event to dictionary
            event_data = event.to_dict()
            
            # Send message
            future = self.producer.send(
                topic, 
                value=event_data, 
                key=key or event.user_id
            )
            
            # Wait for acknowledgment
            record_metadata = future.get(timeout=10)
            
            logging.debug(f"Sent event to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
            
        except KafkaError as e:
            logging.error(f"Failed to send event to Kafka: {e}")
            raise
    
    def start_consumers(self):
        """Start all Kafka consumers"""
        self.running = True
        
        for topic, consumer in self.consumers.items():
            processor = self.processors[topic]
            
            # Start consumer in separate thread
            consumer_thread = threading.Thread(
                target=self._consume_messages,
                args=(topic, consumer, processor),
                daemon=True
            )
            consumer_thread.start()
            
            logging.info(f"Started consumer for topic: {topic}")
    
    def _consume_messages(self, topic: str, consumer: KafkaConsumer, processor: Callable):
        """Consume messages from Kafka topic"""
        while self.running:
            try:
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Convert message to StreamEvent
                            event_data = message.value
                            event = StreamEvent.from_dict(event_data)
                            
                            # Process event
                            asyncio.run(processor(event))
                            
                        except Exception as e:
                            logging.error(f"Error processing message: {e}")
                            
            except Exception as e:
                logging.error(f"Error consuming from topic {topic}: {e}")
                
    def stop(self):
        """Stop all consumers and producer"""
        self.running = False
        
        # Close consumers
        for consumer in self.consumers.values():
            consumer.close()
            
        # Close producer
        if self.producer:
            self.producer.close()
            
        logging.info("Kafka stream processor stopped")

class KafkaRecommendationSystem:
    """Kafka-based real-time recommendation system"""
    
    TOPICS = {
        'USER_INTERACTIONS': 'user-interactions',
        'RECOMMENDATION_REQUESTS': 'recommendation-requests', 
        'RECOMMENDATION_RESPONSES': 'recommendation-responses',
        'USER_PROFILES': 'user-profiles',
        'ITEM_UPDATES': 'item-updates'
    }
    
    def __init__(self, kafka_servers: str = 'localhost:9092'):
        self.kafka_processor = KafkaStreamProcessor(kafka_servers)
        self.recommendation_engine = StreamingRecommendationEngine()
        self.user_profile_processor = UserProfileProcessor()
        
        # Setup Kafka consumers
        self._setup_consumers()
        
    def _setup_consumers(self):
        """Setup Kafka consumers for different topics"""
        
        # User interactions consumer
        self.kafka_processor.setup_consumer(
            self.TOPICS['USER_INTERACTIONS'],
            'user-interaction-processor',
            self._process_user_interaction
        )
        
        # Recommendation requests consumer
        self.kafka_processor.setup_consumer(
            self.TOPICS['RECOMMENDATION_REQUESTS'],
            'recommendation-processor',
            self._process_recommendation_request
        )
        
        # Item updates consumer
        self.kafka_processor.setup_consumer(
            self.TOPICS['ITEM_UPDATES'],
            'item-update-processor',
            self._process_item_update
        )
    
    async def _process_user_interaction(self, event: StreamEvent):
        """Process user interaction from Kafka"""
        try:
            # Update local recommendation engine
            await self.recommendation_engine.process_user_interaction(
                event.user_id,
                event.data.get('item_id'),
                event.data.get('interaction_type'),
                event.data.get('interaction_value', 1.0),
                event.data.get('context', {})
            )
            
            # Update user profile
            await self.user_profile_processor.process_event(event)
            
            # Publish updated user profile
            profile = self.user_profile_processor.get_user_profile(event.user_id)
            profile_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.USER_PROFILE_UPDATE,
                timestamp=datetime.now(),
                user_id=event.user_id,
                data=profile
            )
            
            await self.kafka_processor.produce_event(
                self.TOPICS['USER_PROFILES'],
                profile_event,
                key=event.user_id
            )
            
        except Exception as e:
            logging.error(f"Error processing user interaction: {e}")
    
    async def _process_recommendation_request(self, event: StreamEvent):
        """Process recommendation request from Kafka"""
        try:
            # Generate recommendations
            recommendations = await self.recommendation_engine.get_recommendations(
                event.user_id,
                event.data.get('num_recommendations', 10),
                event.data.get('request_context', {})
            )
            
            # Create response event
            response_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.RECOMMENDATION_RESPONSE,
                timestamp=datetime.now(),
                user_id=event.user_id,
                session_id=event.session_id,
                data={
                    'request_id': event.event_id,
                    'recommendations': recommendations,
                    'processing_time_ms': (datetime.now() - event.timestamp).total_seconds() * 1000
                }
            )
            
            # Publish response
            await self.kafka_processor.produce_event(
                self.TOPICS['RECOMMENDATION_RESPONSES'],
                response_event,
                key=event.user_id
            )
            
        except Exception as e:
            logging.error(f"Error processing recommendation request: {e}")
    
    async def _process_item_update(self, event: StreamEvent):
        """Process item update from Kafka"""
        try:
            # Update item features in recommendation engine
            item_id = event.data.get('item_id')
            features = event.data.get('features', {})
            
            if item_id:
                self.recommendation_engine.item_features[item_id] = features
                
                # Invalidate related caches
                # (In practice, this would be more sophisticated)
                self.recommendation_engine.recommendation_cache.clear()
                
        except Exception as e:
            logging.error(f"Error processing item update: {e}")
    
    async def start(self):
        """Start the Kafka-based recommendation system"""
        try:
            # Setup producer
            self.kafka_processor.setup_producer()
            
            # Start consumers
            self.kafka_processor.start_consumers()
            
            logging.info("Kafka recommendation system started")
            
        except Exception as e:
            logging.error(f"Failed to start Kafka recommendation system: {e}")
            raise
    
    async def stop(self):
        """Stop the Kafka-based recommendation system"""
        self.kafka_processor.stop()
        logging.info("Kafka recommendation system stopped")
    
    async def send_user_interaction(self, user_id: str, item_id: str, 
                                  interaction_type: str, interaction_value: float = 1.0,
                                  context: Dict[str, Any] = None):
        """Send user interaction event to Kafka"""
        
        event = UserInteractionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            item_id=item_id,
            interaction_type=interaction_type,
            interaction_value=interaction_value,
            context=context or {}
        )
        
        await self.kafka_processor.produce_event(
            self.TOPICS['USER_INTERACTIONS'],
            event,
            key=user_id
        )
    
    async def request_recommendations(self, user_id: str, num_recommendations: int = 10,
                                    request_context: Dict[str, Any] = None) -> str:
        """Send recommendation request to Kafka and return request ID"""
        
        request_event = RecommendationRequestEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            num_recommendations=num_recommendations,
            request_context=request_context or {}
        )
        
        await self.kafka_processor.produce_event(
            self.TOPICS['RECOMMENDATION_REQUESTS'],
            request_event,
            key=user_id
        )
        
        return request_event.event_id
```

## 3. Apache Flink Integration

### Flink Stream Processing for Real-time Recommendations

```python
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import json

# Note: This is a conceptual implementation of Flink-like stream processing
# In practice, you would use PyFlink or implement this in Java/Scala

class FlinkStreamProcessor:
    """Flink-inspired stream processor for recommendations"""
    
    def __init__(self, parallelism: int = 4, checkpoint_interval: int = 30):
        self.parallelism = parallelism
        self.checkpoint_interval = checkpoint_interval
        self.operators = []
        self.state_backend = {}
        self.watermark_strategy = None
        
    def add_source(self, source_function: Callable) -> 'DataStream':
        """Add data source"""
        stream = DataStream(self, source_function)
        return stream
    
    def execute(self, job_name: str = "RecommendationJob"):
        """Execute the streaming job"""
        logging.info(f"Starting Flink job: {job_name}")
        
        # In practice, this would submit the job to Flink cluster
        asyncio.run(self._run_streaming_job())
    
    async def _run_streaming_job(self):
        """Run the streaming job"""
        # Simulate continuous processing
        while True:
            try:
                # Process each operator
                for operator in self.operators:
                    await operator.process()
                    
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                logging.error(f"Error in streaming job: {e}")

class DataStream:
    """Flink-inspired DataStream for stream processing"""
    
    def __init__(self, processor: FlinkStreamProcessor, source_function: Callable = None):
        self.processor = processor
        self.source_function = source_function
        self.transformations = []
        
    def map(self, map_function: Callable) -> 'DataStream':
        """Apply map transformation"""
        new_stream = DataStream(self.processor)
        new_stream.transformations = self.transformations + [('map', map_function)]
        return new_stream
    
    def filter(self, filter_function: Callable) -> 'DataStream':
        """Apply filter transformation"""
        new_stream = DataStream(self.processor)
        new_stream.transformations = self.transformations + [('filter', filter_function)]
        return new_stream
    
    def key_by(self, key_selector: Callable) -> 'KeyedStream':
        """Create keyed stream"""
        keyed_stream = KeyedStream(self.processor, key_selector)
        keyed_stream.transformations = self.transformations
        return keyed_stream
    
    def window(self, window_assigner) -> 'WindowedStream':
        """Apply windowing"""
        windowed_stream = WindowedStream(self.processor, window_assigner)
        windowed_stream.transformations = self.transformations
        return windowed_stream
    
    def sink(self, sink_function: Callable):
        """Add sink"""
        operator = StreamOperator('sink', sink_function, self.transformations)
        self.processor.operators.append(operator)

class KeyedStream(DataStream):
    """Keyed stream for stateful operations"""
    
    def __init__(self, processor: FlinkStreamProcessor, key_selector: Callable):
        super().__init__(processor)
        self.key_selector = key_selector
        
    def reduce(self, reduce_function: Callable) -> DataStream:
        """Apply reduce operation"""
        new_stream = DataStream(self.processor)
        new_stream.transformations = self.transformations + [('reduce', reduce_function)]
        return new_stream
    
    def aggregate(self, aggregate_function) -> DataStream:
        """Apply aggregation"""
        new_stream = DataStream(self.processor)
        new_stream.transformations = self.transformations + [('aggregate', aggregate_function)]
        return new_stream

class WindowedStream(DataStream):
    """Windowed stream for time-based operations"""
    
    def __init__(self, processor: FlinkStreamProcessor, window_assigner):
        super().__init__(processor)
        self.window_assigner = window_assigner
        
    def reduce(self, reduce_function: Callable) -> DataStream:
        """Reduce over window"""
        new_stream = DataStream(self.processor)
        new_stream.transformations = self.transformations + [('window_reduce', reduce_function)]
        return new_stream
    
    def apply(self, window_function: Callable) -> DataStream:
        """Apply window function"""
        new_stream = DataStream(self.processor)
        new_stream.transformations = self.transformations + [('window_apply', window_function)]
        return new_stream

class StreamOperator:
    """Stream operator for processing elements"""
    
    def __init__(self, operator_type: str, function: Callable, transformations: List):
        self.operator_type = operator_type
        self.function = function
        self.transformations = transformations
        
    async def process(self):
        """Process stream elements"""
        # In practice, this would process actual stream elements
        pass

class FlinkRecommendationJob:
    """Flink job for real-time recommendations"""
    
    def __init__(self):
        self.processor = FlinkStreamProcessor(parallelism=4)
        self.user_profiles = {}
        self.item_features = {}
        self.recommendation_cache = {}
        
    def create_user_interaction_stream(self):
        """Create stream for user interactions"""
        
        def interaction_source():
            # In practice, this would read from Kafka, socket, etc.
            while True:
                yield {
                    'user_id': f'user_{np.random.randint(1, 1000)}',
                    'item_id': f'item_{np.random.randint(1, 5000)}',
                    'interaction_type': np.random.choice(['click', 'view', 'purchase']),
                    'timestamp': datetime.now().isoformat(),
                    'value': np.random.uniform(0.1, 1.0)
                }
        
        return self.processor.add_source(interaction_source)
    
    def create_recommendation_pipeline(self):
        """Create complete recommendation pipeline"""
        
        # User interaction stream
        interactions = self.create_user_interaction_stream()
        
        # Extract features from interactions
        features = interactions.map(self.extract_interaction_features)
        
        # Key by user for stateful processing
        keyed_features = features.key_by(lambda x: x['user_id'])
        
        # Aggregate user profiles over time windows
        user_profiles = keyed_features.window(
            TumblingEventTimeWindows.of(timedelta(minutes=5))
        ).aggregate(UserProfileAggregator())
        
        # Generate recommendations
        recommendations = user_profiles.map(self.generate_recommendations)
        
        # Sink to output (Kafka, database, etc.)
        recommendations.sink(self.recommendation_sink)
        
        return self.processor
    
    def extract_interaction_features(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from interaction"""
        return {
            'user_id': interaction['user_id'],
            'item_id': interaction['item_id'],
            'interaction_type': interaction['interaction_type'],
            'timestamp': interaction['timestamp'],
            'value': interaction['value'],
            'features': {
                'interaction_weight': self.get_interaction_weight(interaction['interaction_type']),
                'time_of_day': datetime.fromisoformat(interaction['timestamp']).hour,
                'day_of_week': datetime.fromisoformat(interaction['timestamp']).weekday()
            }
        }
    
    def get_interaction_weight(self, interaction_type: str) -> float:
        """Get weight for interaction type"""
        weights = {
            'view': 0.1,
            'click': 0.5,
            'purchase': 1.0,
            'rating': 0.8
        }
        return weights.get(interaction_type, 0.3)
    
    def generate_recommendations(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for user"""
        user_id = user_profile['user_id']
        
        # Simple collaborative filtering
        similar_users = self.find_similar_users(user_profile)
        candidate_items = self.get_candidate_items(similar_users)
        
        # Score and rank items
        recommendations = []
        for item_id in candidate_items[:50]:  # Top 50 candidates
            score = self.calculate_item_score(user_profile, item_id)
            recommendations.append({
                'item_id': item_id,
                'score': score
            })
        
        # Sort by score and take top 10
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'user_id': user_id,
            'recommendations': recommendations[:10],
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
    
    def find_similar_users(self, user_profile: Dict[str, Any]) -> List[str]:
        """Find similar users"""
        # Simplified similarity based on interaction patterns
        user_items = set(user_profile.get('items', []))
        similar_users = []
        
        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id == user_profile['user_id']:
                continue
                
            other_items = set(other_profile.get('items', []))
            
            # Jaccard similarity
            intersection = len(user_items & other_items)
            union = len(user_items | other_items)
            
            if union > 0 and intersection / union > 0.1:
                similar_users.append(other_user_id)
        
        return similar_users[:20]  # Top 20 similar users
    
    def get_candidate_items(self, similar_users: List[str]) -> List[str]:
        """Get candidate items from similar users"""
        candidate_items = set()
        
        for user_id in similar_users:
            if user_id in self.user_profiles:
                user_items = self.user_profiles[user_id].get('items', [])
                candidate_items.update(user_items)
        
        return list(candidate_items)
    
    def calculate_item_score(self, user_profile: Dict[str, Any], item_id: str) -> float:
        """Calculate score for item"""
        # Simple scoring based on item popularity and user preferences
        base_score = 0.5
        
        # Item popularity boost
        if item_id in self.item_features:
            popularity = self.item_features[item_id].get('popularity', 0.5)
            base_score += popularity * 0.3
        
        # User preference alignment
        user_categories = user_profile.get('preferred_categories', {})
        if item_id in self.item_features:
            item_category = self.item_features[item_id].get('category', 'unknown')
            if item_category in user_categories:
                base_score += user_categories[item_category] * 0.4
        
        # Add randomness for exploration
        base_score += np.random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def recommendation_sink(self, recommendation: Dict[str, Any]):
        """Sink for recommendations"""
        # In practice, this would write to Kafka, database, cache, etc.
        user_id = recommendation['user_id']
        self.recommendation_cache[user_id] = recommendation
        
        logging.info(f"Generated recommendations for user {user_id}: "
                    f"{len(recommendation['recommendations'])} items")

class UserProfileAggregator:
    """Aggregator for user profiles"""
    
    def __init__(self):
        self.profiles = {}
    
    def add(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add interaction to profile"""
        user_id = interaction['user_id']
        
        if user_id not in self.profiles:
            self.profiles[user_id] = {
                'user_id': user_id,
                'items': set(),
                'interaction_types': {},
                'preferred_categories': {},
                'total_interactions': 0
            }
        
        profile = self.profiles[user_id]
        profile['items'].add(interaction['item_id'])
        profile['total_interactions'] += 1
        
        # Update interaction types
        interaction_type = interaction['interaction_type']
        profile['interaction_types'][interaction_type] = \
            profile['interaction_types'].get(interaction_type, 0) + 1
        
        return profile
    
    def get_result(self, user_id: str) -> Dict[str, Any]:
        """Get aggregated profile"""
        profile = self.profiles.get(user_id, {})
        
        # Convert sets to lists for serialization
        if 'items' in profile:
            profile['items'] = list(profile['items'])
        
        return profile

class TumblingEventTimeWindows:
    """Tumbling event time windows"""
    
    @staticmethod
    def of(duration: timedelta):
        return WindowAssigner(duration, 'tumbling')

class WindowAssigner:
    """Window assigner for time-based windows"""
    
    def __init__(self, duration: timedelta, window_type: str):
        self.duration = duration
        self.window_type = window_type
```

## Key Takeaways

1. **Event-Driven Architecture**: Streaming systems enable real-time processing of user interactions and immediate recommendation updates

2. **Kafka Integration**: Apache Kafka provides robust, scalable message streaming for high-throughput recommendation systems

3. **Stream Processing**: Frameworks like Flink enable complex event processing and stateful computations over streaming data

4. **Real-time Updates**: User profiles and recommendations can be updated continuously as new data arrives

5. **Fault Tolerance**: Streaming systems provide checkpointing and recovery mechanisms for reliable processing

6. **Scalability**: Distributed streaming architectures can handle massive volumes of real-time data

## Study Questions

### Beginner Level
1. What are the key components of an event-driven recommendation system?
2. How does Kafka ensure message delivery and ordering?
3. What is the difference between batch and stream processing?
4. How do you handle late-arriving events in streaming systems?

### Intermediate Level
1. Design a fault-tolerant streaming pipeline for real-time recommendations
2. How would you implement exactly-once processing semantics?
3. What are the trade-offs between latency and throughput in streaming systems?
4. How do you handle schema evolution in streaming data?

### Advanced Level
1. Implement a multi-tenant streaming recommendation system
2. Design a system that handles both streaming and batch processing (Lambda architecture)
3. How would you implement complex event processing for recommendation systems?
4. Design a globally distributed streaming architecture with cross-region replication

## Next Session Preview

Tomorrow we'll explore **Online Learning and Model Updates**, covering:
- Incremental learning algorithms for recommendations
- Model versioning and A/B testing in production
- Concept drift detection and adaptation
- Online optimization techniques
- Real-time model serving and updates
- Continuous learning pipelines

We'll implement systems that continuously learn and adapt to changing user behavior!