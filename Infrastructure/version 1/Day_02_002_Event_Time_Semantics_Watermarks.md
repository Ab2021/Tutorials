# Day 2.2: Event-Time Semantics & Watermark Generation

## ⏰ Streaming Ingestion & Real-Time Feature Pipelines - Part 2

**Focus**: Event-Time Processing, Watermarks, and Temporal Semantics  
**Duration**: 2-3 hours  
**Level**: Intermediate to Advanced  

---

## 🎯 Learning Objectives

- Master event-time vs processing-time semantics
- Understand watermark generation algorithms and strategies
- Learn to handle late-arriving and out-of-order events
- Implement temporal joins and windowing operations

---

## ⏳ Temporal Semantics in Stream Processing

### **Time Dimensions in Streaming Systems**

#### **1. Event Time vs Processing Time**

```
Timeline Visualization:

Event Time:    e1---e2-------e3-e4--------e5------→
               10:00 10:01   10:03 10:03  10:05

Processing:    ----e1---------e2--e4-e3----e5----→
               10:02         10:04 10:04  10:06

Watermark:     --------W(10:00)----W(10:02)-----→
                      ↑              ↑
                   Safe to process  Process events
                   events ≤ 10:00   ≤ 10:02
```

**Mathematical Definition**:
```
Let E = {(e₁, t₁), (e₂, t₂), ..., (eₙ, tₙ)} be a stream of events
where tᵢ = event timestamp, pᵢ = processing timestamp

Event-time ordering: t₁ ≤ t₂ ≤ ... ≤ tₙ
Processing-time ordering: p₁ ≤ p₂ ≤ ... ≤ pₙ

Skew = |tᵢ - pᵢ| for each event eᵢ
```

#### **2. Types of Time in Distributed Systems**

```python
from enum import Enum
from datetime import datetime, timezone

class TimeSemantics(Enum):
    EVENT_TIME = "time_when_event_occurred"
    PROCESSING_TIME = "time_when_event_processed"
    INGESTION_TIME = "time_when_event_entered_system"

class TemporalEvent:
    def __init__(self, data, event_time, processing_time=None, ingestion_time=None):
        self.data = data
        self.event_time = event_time
        self.processing_time = processing_time or datetime.now(timezone.utc)
        self.ingestion_time = ingestion_time or self.processing_time
        self.skew = abs((self.processing_time - self.event_time).total_seconds())
    
    def calculate_skew_metrics(self):
        """Calculate temporal skew metrics"""
        return {
            'event_to_processing_skew_seconds': self.skew,
            'event_to_ingestion_skew_seconds': abs(
                (self.ingestion_time - self.event_time).total_seconds()
            ),
            'ingestion_to_processing_skew_seconds': abs(
                (self.processing_time - self.ingestion_time).total_seconds()
            ),
            'is_late_event': self.skew > 300,  # 5 minutes threshold
            'severity': self.classify_lateness()
        }
    
    def classify_lateness(self):
        """Classify event lateness severity"""
        if self.skew < 60:  # 1 minute
            return 'on_time'
        elif self.skew < 300:  # 5 minutes
            return 'slightly_late'
        elif self.skew < 3600:  # 1 hour
            return 'late'
        else:
            return 'very_late'
```

---

## 🌊 Watermark Generation Theory

### **Watermark Definition and Purpose**

A **watermark** W(t) is a timestamp assertion that "no more events with timestamp ≤ t will arrive". It enables the system to make progress on event-time computations despite out-of-order arrivals.

#### **Watermark Properties**
```
1. Monotonicity: W(t₁) ≤ W(t₂) for t₁ < t₂
2. Completeness: ∀ events e with timestamp ≤ W(t), e has been processed
3. Progression: W(t) advances over time to prevent infinite waiting
```

### **Watermark Generation Algorithms**

#### **1. Perfect Watermarks (Theoretical)**
```python
class PerfectWatermarkGenerator:
    """Theoretical perfect watermarks - knows all future events"""
    
    def __init__(self, complete_event_stream):
        self.all_events = sorted(complete_event_stream, key=lambda e: e.event_time)
        self.current_index = 0
    
    def generate_watermark(self, current_processing_time):
        """Generate perfect watermark with complete knowledge"""
        # In reality, this is impossible - we can't know all future events
        
        # Find all events that should have arrived by now
        processed_events = []
        while (self.current_index < len(self.all_events) and 
               self.all_events[self.current_index].expected_arrival_time <= current_processing_time):
            processed_events.append(self.all_events[self.current_index])
            self.current_index += 1
        
        if processed_events:
            # Watermark = minimum event time of unprocessed events
            return min(e.event_time for e in self.all_events[self.current_index:])
        
        return current_processing_time  # No more events expected
```

#### **2. Heuristic Watermarks (Practical)**
```python
from collections import deque
import statistics

class HeuristicWatermarkGenerator:
    """Practical watermark generation using heuristics"""
    
    def __init__(self, max_out_of_orderness_ms=5000, confidence_level=0.95):
        self.max_out_of_orderness = max_out_of_orderness_ms
        self.confidence_level = confidence_level
        self.skew_history = deque(maxlen=1000)
        self.last_watermark = 0
        
    def bounded_out_of_orderness_watermark(self, current_processing_time, latest_event_time):
        """Generate watermark assuming bounded out-of-orderness"""
        # Watermark = latest_event_time - max_expected_delay
        watermark = latest_event_time - self.max_out_of_orderness
        
        # Ensure monotonicity
        watermark = max(watermark, self.last_watermark)
        self.last_watermark = watermark
        
        return watermark
    
    def adaptive_watermark(self, recent_events):
        """Adaptive watermark based on observed skew patterns"""
        if not recent_events:
            return self.last_watermark
        
        # Calculate skew for recent events
        current_time = datetime.now().timestamp() * 1000
        skews = []
        
        for event in recent_events:
            event_skew = current_time - event.event_time
            skews.append(event_skew)
            self.skew_history.append(event_skew)
        
        if len(self.skew_history) < 10:
            # Not enough data, use conservative approach
            return self.bounded_out_of_orderness_watermark(
                current_time, 
                max(e.event_time for e in recent_events)
            )
        
        # Calculate percentile-based delay
        observed_delays = list(self.skew_history)
        percentile_delay = statistics.quantiles(
            observed_delays, 
            n=100
        )[int(self.confidence_level * 100) - 1]
        
        latest_event_time = max(e.event_time for e in recent_events)
        adaptive_watermark = latest_event_time - percentile_delay
        
        # Ensure monotonicity and reasonable progression
        watermark = max(adaptive_watermark, self.last_watermark)
        self.last_watermark = watermark
        
        return watermark
    
    def punctuated_watermark(self, event):
        """Generate watermark based on special punctuation events"""
        if hasattr(event, 'is_watermark_trigger') and event.is_watermark_trigger:
            # Some systems insert special events that indicate safe watermark advancement
            return event.event_time
        
        return None  # No watermark update
```

#### **3. Multi-Source Watermark Coordination**
```python
class MultiSourceWatermarkCoordinator:
    """Coordinate watermarks across multiple input sources"""
    
    def __init__(self, source_configs):
        self.sources = {
            source_id: {
                'current_watermark': 0,
                'reliability': config.get('reliability', 0.9),
                'typical_delay_ms': config.get('typical_delay_ms', 1000),
                'last_update': 0
            }
            for source_id, config in source_configs.items()
        }
        
    def compute_global_watermark(self):
        """Compute global watermark from multiple sources"""
        current_time = datetime.now().timestamp() * 1000
        source_watermarks = []
        
        for source_id, source_info in self.sources.items():
            # Adjust watermark based on source reliability
            time_since_update = current_time - source_info['last_update']
            
            if time_since_update > source_info['typical_delay_ms'] * 3:
                # Source appears stalled, use conservative estimate
                estimated_watermark = (
                    source_info['current_watermark'] - 
                    source_info['typical_delay_ms']
                )
            else:
                estimated_watermark = source_info['current_watermark']
            
            # Weight by reliability
            weighted_watermark = estimated_watermark * source_info['reliability']
            source_watermarks.append(weighted_watermark)
        
        # Global watermark = minimum of all source watermarks
        # This ensures correctness across all sources
        global_watermark = min(source_watermarks) if source_watermarks else 0
        
        return {
            'global_watermark': global_watermark,
            'source_watermarks': {
                sid: info['current_watermark'] 
                for sid, info in self.sources.items()
            },
            'confidence_score': self.calculate_confidence(),
            'lagging_sources': self.identify_lagging_sources()
        }
    
    def calculate_confidence(self):
        """Calculate confidence in global watermark"""
        current_time = datetime.now().timestamp() * 1000
        confidence_scores = []
        
        for source_info in self.sources.values():
            time_freshness = max(0, 1 - (
                (current_time - source_info['last_update']) / 
                (source_info['typical_delay_ms'] * 5)
            ))
            
            source_confidence = source_info['reliability'] * time_freshness
            confidence_scores.append(source_confidence)
        
        return sum(confidence_scores) / len(confidence_scores)
```

---

## 🪟 Windowing Operations with Event-Time

### **Window Types and Semantics**

#### **1. Tumbling Windows**
```python
from datetime import datetime, timedelta

class TumblingWindow:
    """Non-overlapping windows of fixed size"""
    
    def __init__(self, window_size_ms):
        self.window_size = window_size_ms
        
    def assign_window(self, event_timestamp):
        """Assign event to appropriate tumbling window"""
        # Window start = floor(event_time / window_size) * window_size
        window_start = (event_timestamp // self.window_size) * self.window_size
        window_end = window_start + self.window_size
        
        return {
            'window_start': window_start,
            'window_end': window_end,
            'window_id': f"tumbling_{window_start}_{window_end}"
        }
    
    def calculate_window_metrics(self, events_in_window):
        """Calculate metrics for events in window"""
        if not events_in_window:
            return None
            
        event_times = [e.event_time for e in events_in_window]
        
        return {
            'count': len(events_in_window),
            'min_event_time': min(event_times),
            'max_event_time': max(event_times),
            'span_ms': max(event_times) - min(event_times),
            'completeness_score': self.assess_completeness(events_in_window)
        }
    
    def assess_completeness(self, events_in_window):
        """Assess if window received all expected events"""
        # This would require domain knowledge about expected event rates
        # Simplified example
        expected_events_per_window = 100  # Domain-specific
        actual_events = len(events_in_window)
        
        return min(1.0, actual_events / expected_events_per_window)
```

#### **2. Sliding Windows**
```python
class SlidingWindow:
    """Overlapping windows with fixed slide interval"""
    
    def __init__(self, window_size_ms, slide_interval_ms):
        self.window_size = window_size_ms
        self.slide_interval = slide_interval_ms
        
        if slide_interval_ms > window_size_ms:
            raise ValueError("Slide interval cannot exceed window size")
    
    def assign_windows(self, event_timestamp):
        """Assign event to all applicable sliding windows"""
        windows = []
        
        # Find the earliest window that contains this event
        first_window_start = self.find_first_window_start(event_timestamp)
        
        # Generate all windows that contain this event
        window_start = first_window_start
        while window_start + self.window_size > event_timestamp:
            windows.append({
                'window_start': window_start,
                'window_end': window_start + self.window_size,
                'window_id': f"sliding_{window_start}_{window_start + self.window_size}"
            })
            window_start -= self.slide_interval
            
            # Prevent infinite loop for very small slide intervals
            if len(windows) > 1000:
                break
        
        return windows
    
    def find_first_window_start(self, event_timestamp):
        """Find the start of the first window containing the event"""
        # Latest possible window start that still contains the event
        return ((event_timestamp - self.window_size + self.slide_interval) // 
                self.slide_interval) * self.slide_interval
    
    def calculate_overlap_factor(self):
        """Calculate how much windows overlap"""
        return self.window_size / self.slide_interval
```

#### **3. Session Windows**
```python
class SessionWindow:
    """Variable-size windows based on activity sessions"""
    
    def __init__(self, session_timeout_ms):
        self.session_timeout = session_timeout_ms
        self.active_sessions = {}  # key -> session_info
    
    def process_event(self, event, session_key):
        """Process event and update session windows"""
        current_time = event.event_time
        
        if session_key not in self.active_sessions:
            # Start new session
            self.active_sessions[session_key] = {
                'session_start': current_time,
                'last_activity': current_time,
                'event_count': 1,
                'events': [event]
            }
        else:
            session = self.active_sessions[session_key]
            time_since_last_activity = current_time - session['last_activity']
            
            if time_since_last_activity <= self.session_timeout:
                # Continue existing session
                session['last_activity'] = current_time
                session['event_count'] += 1
                session['events'].append(event)
            else:
                # Close old session and start new one
                closed_session = self.close_session(session_key)
                
                self.active_sessions[session_key] = {
                    'session_start': current_time,
                    'last_activity': current_time,
                    'event_count': 1,
                    'events': [event]
                }
                
                return closed_session
        
        return None  # No session closed
    
    def close_session(self, session_key):
        """Close and return completed session"""
        if session_key not in self.active_sessions:
            return None
            
        session = self.active_sessions.pop(session_key)
        
        return {
            'session_key': session_key,
            'session_start': session['session_start'],
            'session_end': session['last_activity'],
            'duration_ms': session['last_activity'] - session['session_start'],
            'event_count': session['event_count'],
            'events': session['events'],
            'window_id': f"session_{session_key}_{session['session_start']}"
        }
    
    def expire_inactive_sessions(self, current_watermark):
        """Expire sessions that haven't seen activity"""
        expired_sessions = []
        
        for session_key, session in list(self.active_sessions.items()):
            if current_watermark - session['last_activity'] > self.session_timeout:
                expired_sessions.append(self.close_session(session_key))
        
        return expired_sessions
```

---

## 🔄 Late Event Handling Strategies

### **1. Allowed Lateness Configuration**
```python
class LateEventHandler:
    """Handle late-arriving events with configurable strategies"""
    
    def __init__(self, allowed_lateness_ms=300000):  # 5 minutes
        self.allowed_lateness = allowed_lateness_ms
        self.late_event_strategies = {
            'drop': self.drop_late_event,
            'side_output': self.route_to_side_output,
            'recompute': self.trigger_recomputation,
            'approximate': self.approximate_update
        }
        self.late_events_buffer = {}
        
    def process_late_event(self, event, current_watermark, strategy='recompute'):
        """Process late-arriving event based on configured strategy"""
        lateness = current_watermark - event.event_time
        
        if lateness <= self.allowed_lateness:
            # Within allowed lateness, process normally
            return self.late_event_strategies[strategy](event, lateness)
        else:
            # Beyond allowed lateness, always drop
            return self.drop_late_event(event, lateness)
    
    def drop_late_event(self, event, lateness):
        """Drop late event and log metrics"""
        return {
            'action': 'dropped',
            'event_id': event.id,
            'lateness_ms': lateness,
            'reason': 'exceeded_allowed_lateness'
        }
    
    def route_to_side_output(self, event, lateness):
        """Route late event to side output for separate processing"""
        return {
            'action': 'side_output',
            'event_id': event.id,
            'lateness_ms': lateness,
            'side_output_topic': 'late_events_stream'
        }
    
    def trigger_recomputation(self, event, lateness):
        """Trigger recomputation of affected windows"""
        affected_windows = self.find_affected_windows(event)
        
        for window in affected_windows:
            self.schedule_window_recomputation(window, event)
        
        return {
            'action': 'recompute',
            'event_id': event.id,
            'lateness_ms': lateness,
            'affected_windows': len(affected_windows),
            'recomputation_cost': self.estimate_recomputation_cost(affected_windows)
        }
    
    def approximate_update(self, event, lateness):
        """Update window results approximately without full recomputation"""
        # For additive/commutative operations, we can update incrementally
        affected_windows = self.find_affected_windows(event)
        
        for window in affected_windows:
            self.apply_incremental_update(window, event)
        
        return {
            'action': 'approximate',
            'event_id': event.id,
            'lateness_ms': lateness,
            'accuracy_impact': self.estimate_accuracy_loss(event, lateness)
        }
```

### **2. Lateness Metrics and Monitoring**
```python
class LatenessMonitor:
    """Monitor and analyze event lateness patterns"""
    
    def __init__(self):
        self.lateness_histogram = {}
        self.source_lateness_stats = {}
        
    def record_event_lateness(self, event, processing_time):
        """Record lateness metrics for analysis"""
        lateness = processing_time - event.event_time
        source = getattr(event, 'source', 'unknown')
        
        # Update histogram
        lateness_bucket = self.get_lateness_bucket(lateness)
        self.lateness_histogram[lateness_bucket] = (
            self.lateness_histogram.get(lateness_bucket, 0) + 1
        )
        
        # Update per-source stats
        if source not in self.source_lateness_stats:
            self.source_lateness_stats[source] = {
                'total_events': 0,
                'total_lateness': 0,
                'max_lateness': 0,
                'late_events': 0
            }
        
        stats = self.source_lateness_stats[source]
        stats['total_events'] += 1
        stats['total_lateness'] += lateness
        stats['max_lateness'] = max(stats['max_lateness'], lateness)
        
        if lateness > 60000:  # 1 minute threshold
            stats['late_events'] += 1
    
    def get_lateness_bucket(self, lateness_ms):
        """Bucket lateness for histogram analysis"""
        if lateness_ms < 0:
            return 'future_events'
        elif lateness_ms < 1000:
            return '0-1s'
        elif lateness_ms < 5000:
            return '1-5s'
        elif lateness_ms < 30000:
            return '5-30s'
        elif lateness_ms < 300000:
            return '30s-5m'
        else:
            return '5m+'
    
    def analyze_lateness_patterns(self):
        """Analyze lateness patterns and suggest optimizations"""
        analysis = {
            'overall_lateness_distribution': self.lateness_histogram,
            'problematic_sources': [],
            'recommendations': []
        }
        
        for source, stats in self.source_lateness_stats.items():
            if stats['total_events'] == 0:
                continue
                
            avg_lateness = stats['total_lateness'] / stats['total_events']
            late_event_ratio = stats['late_events'] / stats['total_events']
            
            if late_event_ratio > 0.05:  # >5% late events
                analysis['problematic_sources'].append({
                    'source': source,
                    'late_event_ratio': late_event_ratio,
                    'avg_lateness_ms': avg_lateness,
                    'max_lateness_ms': stats['max_lateness']
                })
        
        # Generate recommendations
        if analysis['problematic_sources']:
            analysis['recommendations'].extend([
                'Consider increasing allowed lateness threshold',
                'Investigate source-specific delays',
                'Implement adaptive watermark generation',
                'Review upstream data pipeline performance'
            ])
        
        return analysis
```

This completes Part 2 of Day 2, focusing on the deep theoretical understanding of event-time semantics, watermark generation, and temporal operations in streaming systems.