# Day 45: Distributed Job Scheduler - Interview Prep

## Common Interview Questions

### Q1: Design a distributed job scheduler like Apache Airflow

**Approach**:

1. **Clarify Requirements**
   - Scale: 1M jobs per day
   - Job types: Cron jobs, one-time jobs, dependent jobs
   - Execution: Distributed across multiple workers
   - Reliability: At-least-once execution guarantee
   - Latency: Jobs start within 1 minute of scheduled time

2. **High-Level Architecture**
   ```
   Scheduler → Job Queue (Kafka) → Workers
                                 ↓
                           Metadata DB (PostgreSQL)
   ```

3. **Core Components**
   - **Scheduler**: Determines when jobs should run
   - **Job Queue**: Holds pending jobs (Kafka/RabbitMQ)
   - **Workers**: Execute jobs in parallel
   - **Metadata Store**: Job definitions and execution history
   - **Coordinator**: Leader election and health monitoring

4. **Data Model**
   ```python
   # Job Definition
   {
     "job_id": "daily_report",
     "schedule": "0 0 * * *",  # Cron: daily at midnight
     "command": "python generate_report.py",
     "dependencies": ["data_ingestion"],
     "max_retries": 3,
     "timeout": 3600
   }
   
   # Job Execution
   {
     "execution_id": "uuid",
     "job_id": "daily_report",
     "status": "SUCCESS",
     "start_time": "2024-01-15T00:00:00Z",
     "end_time": "2024-01-15T00:15:00Z",
     "worker_id": "worker-1"
   }
   ```

5. **Key Design Decisions**
   - **Leader Election**: Use ZooKeeper/etcd for coordinator election
   - **Job Queue**: Kafka for durability and ordering
   - **Execution Guarantee**: At-least-once (idempotent jobs)
   - **Retry Strategy**: Exponential backoff

**Follow-ups**:
- **Q**: How do you handle job dependencies?
  - **A**: Build DAG (Directed Acyclic Graph), use topological sort for execution order
  
- **Q**: What if scheduler crashes?
  - **A**: Use leader election; standby coordinator takes over

---

### Q2: How would you implement dependency management between jobs?

**Answer**:

**1. DAG Representation**
```python
class JobDAG:
    def __init__(self):
        self.graph = {}  # job_id -> [dependent_job_ids]
        self.execution_status = {}  # job_id -> status
    
    def add_dependency(self, job_id, depends_on):
        """job_id depends on depends_on"""
        if depends_on not in self.graph:
            self.graph[depends_on] = []
        self.graph[depends_on].append(job_id)
    
    def can_execute(self, job_id):
        """Check if all dependencies are satisfied"""
        # Find all jobs this job depends on
        dependencies = [
            dep_id for dep_id, dependents in self.graph.items()
            if job_id in dependents
        ]
        
        # Check if all dependencies succeeded
        for dep_id in dependencies:
            if self.execution_status.get(dep_id) != 'SUCCESS':
                return False
        
        return True
```

**2. Execution Strategy**
```python
def execute_dag(dag):
    """Execute jobs in dependency order"""
    # Topological sort
    in_degree = {}
    for job_id in all_jobs:
        in_degree[job_id] = count_dependencies(job_id)
    
    # Start with jobs that have no dependencies
    queue = [job_id for job_id, degree in in_degree.items() if degree == 0]
    
    while queue:
        job_id = queue.pop(0)
        
        # Execute job
        execute_job(job_id)
        
        # Update dependent jobs
        for dependent in dag.graph.get(job_id, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
```

**3. Handling Failures**
```python
def handle_job_failure(job_id):
    """Mark all dependent jobs as failed"""
    failed_jobs = set()
    queue = [job_id]
    
    while queue:
        current = queue.pop(0)
        failed_jobs.add(current)
        
        # Fail all dependents
        for dependent in dag.graph.get(current, []):
            if dependent not in failed_jobs:
                mark_as_failed(dependent, reason=f"Dependency {current} failed")
                queue.append(dependent)
```

---

### Q3: How do you ensure exactly-once execution of jobs?

**Answer**:

**Challenge**: Exactly-once is hard in distributed systems.

**Practical Approach**: At-least-once + Idempotency

**1. At-Least-Once Guarantee**
```python
class ReliableJobQueue:
    def enqueue(self, job):
        """Enqueue job with persistence"""
        # Write to durable queue (Kafka)
        self.kafka.produce(
            topic='jobs',
            key=job.job_id,
            value=json.dumps(job)
        )
        
        # Kafka guarantees at-least-once delivery
```

**2. Idempotency**
```python
class IdempotentJobExecutor:
    def execute(self, job_id, command):
        """Execute job idempotently"""
        # Check if already executed today
        execution_key = f"executed:{job_id}:{date.today()}"
        
        if self.redis.exists(execution_key):
            print(f"Job {job_id} already executed")
            return
        
        # Execute job
        result = subprocess.run(command, shell=True)
        
        # Mark as executed (24-hour TTL)
        self.redis.setex(execution_key, 86400, '1')
        
        return result
```

**3. Distributed Locking**
```python
def execute_with_lock(job_id):
    """Ensure only one instance runs"""
    lock_key = f"lock:{job_id}"
    lock_acquired = redis.set(lock_key, '1', nx=True, ex=3600)
    
    if not lock_acquired:
        print(f"Job {job_id} already running")
        return
    
    try:
        execute_job(job_id)
    finally:
        redis.delete(lock_key)
```

**Trade-off**: Exactly-once is theoretically impossible in distributed systems (see Two Generals Problem). At-least-once + idempotency is the practical solution.

---

### Q4: How would you implement retry logic with exponential backoff?

**Answer**:

```python
class RetryManager:
    def execute_with_retry(self, job, max_retries=3):
        """Execute job with exponential backoff"""
        attempt = 0
        
        while attempt < max_retries:
            try:
                # Execute job
                result = self.execute_job(job)
                return result
            
            except Exception as e:
                attempt += 1
                
                if attempt >= max_retries:
                    # Max retries exceeded
                    self.mark_as_failed(job, error=str(e))
                    raise
                
                # Calculate backoff delay
                delay = self.calculate_backoff(attempt)
                
                # Log retry
                print(f"Job {job.job_id} failed, retrying in {delay}s "
                      f"(attempt {attempt}/{max_retries})")
                
                # Wait before retry
                time.sleep(delay)
    
    def calculate_backoff(self, attempt):
        """Exponential backoff with jitter"""
        base_delay = 2 ** attempt  # 2s, 4s, 8s, 16s, ...
        jitter = random.uniform(0, base_delay * 0.1)  # Add 10% jitter
        return base_delay + jitter
```

**Why Jitter?**
- Prevents "thundering herd" problem
- Multiple jobs failing at same time won't all retry simultaneously

**Alternative Strategies**:
- **Linear backoff**: 1s, 2s, 3s, 4s, ...
- **Fibonacci backoff**: 1s, 1s, 2s, 3s, 5s, 8s, ...
- **Capped exponential**: 2s, 4s, 8s, 16s, 32s, 60s (max), 60s, ...

---

### Q5: How do you handle job scheduling at scale (millions of jobs)?

**Answer**:

**1. Sharding by Time**
```python
class ShardedScheduler:
    def __init__(self, num_shards=10):
        self.num_shards = num_shards
        self.schedulers = [Scheduler() for _ in range(num_shards)]
    
    def add_job(self, job):
        """Shard jobs by scheduled time"""
        # Hash scheduled time to shard
        shard_id = hash(job.scheduled_time) % self.num_shards
        self.schedulers[shard_id].add_job(job)
    
    def get_due_jobs(self):
        """Get due jobs from all shards"""
        all_due_jobs = []
        for scheduler in self.schedulers:
            all_due_jobs.extend(scheduler.get_due_jobs())
        return all_due_jobs
```

**2. Efficient Data Structure**
```python
import heapq

class EfficientScheduler:
    def __init__(self):
        # Min-heap ordered by next_run_time
        self.job_heap = []
    
    def add_job(self, job):
        """Add job to heap"""
        heapq.heappush(self.job_heap, (job.next_run_time, job))
    
    def get_due_jobs(self):
        """Get all jobs due now (O(k log n) where k = due jobs)"""
        now = datetime.now()
        due_jobs = []
        
        while self.job_heap and self.job_heap[0][0] <= now:
            next_run_time, job = heapq.heappop(self.job_heap)
            due_jobs.append(job)
            
            # Reschedule recurring jobs
            if job.is_recurring:
                job.next_run_time = self.calculate_next_run(job)
                heapq.heappush(self.job_heap, (job.next_run_time, job))
        
        return due_jobs
```

**3. Distributed Scheduling**
```python
class DistributedScheduler:
    """Multiple scheduler instances for high availability"""
    
    def __init__(self):
        self.is_leader = False
        self.zookeeper = ZooKeeperClient()
    
    def run(self):
        """Run scheduler with leader election"""
        # Elect leader
        self.elect_leader()
        
        if self.is_leader:
            # Only leader schedules jobs
            while True:
                due_jobs = self.get_due_jobs()
                for job in due_jobs:
                    self.enqueue_job(job)
                
                time.sleep(10)  # Check every 10 seconds
        else:
            # Standby mode
            self.watch_leader()
```

**Optimization**: Use time-series database (InfluxDB) for job metadata to handle billions of executions.

---

### Q6: How would you implement priority queues for jobs?

**Answer**:

**1. Multiple Kafka Topics**
```python
class PriorityJobQueue:
    def __init__(self):
        self.topics = {
            'critical': 'jobs_priority_critical',
            'high': 'jobs_priority_high',
            'normal': 'jobs_priority_normal',
            'low': 'jobs_priority_low'
        }
    
    def enqueue(self, job):
        """Enqueue to appropriate priority topic"""
        priority = job.priority
        topic = self.topics.get(priority, 'jobs_priority_normal')
        
        self.kafka.produce(topic, value=json.dumps(job))
    
    def dequeue(self):
        """Consume from topics in priority order"""
        # Try critical first, then high, then normal, then low
        for priority in ['critical', 'high', 'normal', 'low']:
            topic = self.topics[priority]
            message = self.kafka.poll(topic, timeout=0.1)
            
            if message:
                return json.loads(message.value)
        
        return None
```

**2. Redis Sorted Set**
```python
class RedisPriorityQueue:
    def enqueue(self, job):
        """Add job with priority score"""
        # Higher priority = lower score (for ZPOPMIN)
        score = -job.priority
        
        self.redis.zadd('job_queue', {job.job_id: score})
    
    def dequeue(self):
        """Get highest priority job"""
        # ZPOPMIN returns lowest score (highest priority)
        result = self.redis.zpopmin('job_queue', count=1)
        
        if result:
            job_id, score = result[0]
            return self.get_job(job_id)
        
        return None
```

**3. Dynamic Priority Adjustment**
```python
def adjust_priority_for_latency(job):
    """Increase priority for overdue jobs"""
    scheduled_time = job.scheduled_time
    delay = (datetime.now() - scheduled_time).total_seconds()
    
    # Increase priority by 1 for every 5 minutes of delay
    priority_boost = int(delay / 300)
    
    job.priority += priority_boost
    return job
```

---

### Q7: How do you monitor and debug job failures?

**Answer**:

**1. Comprehensive Logging**
```python
class JobLogger:
    def log_execution(self, job_id, status, output=None, error=None):
        """Log job execution details"""
        log_entry = {
            'job_id': job_id,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'output': output,
            'error': error,
            'worker_id': self.worker_id
        }
        
        # Write to Elasticsearch for searchability
        self.es.index(index='job_logs', document=log_entry)
        
        # Also write to S3 for long-term storage
        self.s3.put_object(
            Bucket='job-logs',
            Key=f"{date.today()}/{job_id}/{uuid.uuid4()}.json",
            Body=json.dumps(log_entry)
        )
```

**2. Metrics and Alerting**
```python
class JobMetrics:
    def track_execution(self, job_id, duration, status):
        """Track job metrics"""
        # Execution count
        self.statsd.increment(f'jobs.{status}.count')
        
        # Execution duration
        self.statsd.timing(f'jobs.{job_id}.duration', duration)
        
        # Success rate
        success_rate = self.calculate_success_rate(job_id)
        self.statsd.gauge(f'jobs.{job_id}.success_rate', success_rate)
        
        # Alert on low success rate
        if success_rate < 0.95:
            self.alert(f"Job {job_id} success rate dropped to {success_rate}")
```

**3. Dead Letter Queue**
```python
class DeadLetterQueue:
    def handle_permanent_failure(self, job):
        """Move permanently failed jobs to DLQ"""
        dlq_entry = {
            'job_id': job.job_id,
            'failure_reason': job.error,
            'retry_count': job.retry_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Write to DLQ topic
        self.kafka.produce('dead_letter_queue', value=json.dumps(dlq_entry))
        
        # Alert on-call engineer
        self.pagerduty.trigger_incident(
            title=f"Job {job.job_id} permanently failed",
            details=dlq_entry
        )
```

**4. Execution Timeline**
```python
def get_job_timeline(job_id):
    """Get execution history for debugging"""
    executions = db.query(
        "SELECT * FROM job_executions WHERE job_id = %s "
        "ORDER BY start_time DESC LIMIT 100",
        (job_id,)
    )
    
    return [
        {
            'execution_id': e['execution_id'],
            'status': e['status'],
            'duration': (e['end_time'] - e['start_time']).total_seconds(),
            'worker_id': e['worker_id'],
            'error': e['error']
        }
        for e in executions
    ]
```

---

## System Design Patterns

### Pattern 1: Leader-Follower for High Availability

```
Leader Scheduler → Job Queue
Follower Scheduler (standby) → Takes over on leader failure
```

### Pattern 2: Work Stealing for Load Balancing

```
Worker 1 (idle) → Steals job from Worker 2 (overloaded)
```

### Pattern 3: Circuit Breaker for Failing Jobs

```
Job fails 5 times → Circuit opens → Stop retrying for 10 minutes
```

---

## Key Metrics to Monitor

1. **Job Success Rate**: % of jobs that succeed
2. **Execution Latency**: Time from scheduled to started
3. **Queue Depth**: Number of pending jobs
4. **Worker Utilization**: % of workers busy
5. **Retry Rate**: % of jobs that require retries
6. **DLQ Size**: Number of permanently failed jobs

---

## Common Pitfalls

1. **No idempotency** → Duplicate executions cause issues
2. **No timeouts** → Jobs hang indefinitely
3. **No monitoring** → Can't detect failures
4. **Single point of failure** → Scheduler crash stops all jobs
5. **No backpressure** → Queue overflow
6. **Ignoring dependencies** → Jobs run in wrong order

---

## Interview Tips

1. **Start with requirements** - Clarify scale, latency, reliability
2. **Draw architecture diagram** - Visual representation helps
3. **Discuss trade-offs** - Exactly-once vs. at-least-once
4. **Consider failures** - What if scheduler/worker crashes?
5. **Mention monitoring** - Logging, metrics, alerting
6. **Think about scale** - Sharding, distributed coordination

---

## Further Reading

- Apache Airflow: https://airflow.apache.org/
- Celery: https://docs.celeryproject.org/
- Kubernetes CronJobs: https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/
- AWS Step Functions: https://aws.amazon.com/step-functions/
