# Day 45: Distributed Job Scheduler - Core Concepts

## Overview
Design a distributed job scheduler that can execute millions of tasks reliably, similar to Apache Airflow, Kubernetes CronJobs, or AWS Step Functions.

## Key Requirements

### Functional Requirements
- **Job Scheduling**: Execute jobs at specified times (cron, one-time, recurring)
- **Dependency Management**: Jobs can depend on other jobs
- **Retry Logic**: Automatic retries on failure
- **Priority Queues**: High-priority jobs execute first
- **Job Monitoring**: Track job status and execution history
- **Distributed Execution**: Run jobs across multiple workers

### Non-Functional Requirements
- **Reliability**: Jobs must execute exactly once (or at-least-once)
- **Scalability**: Handle millions of jobs per day
- **Fault Tolerance**: Survive worker/coordinator failures
- **Low Latency**: Jobs start within seconds of scheduled time
- **Observability**: Comprehensive logging and metrics

## System Architecture

### High-Level Design

```
┌──────────────┐
│   Scheduler  │  (Determines when jobs should run)
│  Coordinator │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   Job Queue      │  (Kafka/RabbitMQ)
│  (Pending Jobs)  │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│  (Execute jobs)
└────┬───┘ └───┬────┘
     │         │
     ▼         ▼
┌──────────────────┐
│   Metadata DB    │  (Job definitions, execution history)
│   (PostgreSQL)   │
└──────────────────┘
```

### Components

1. **Scheduler**: Determines which jobs to run and when
2. **Job Queue**: Holds pending jobs (Kafka, RabbitMQ, SQS)
3. **Workers**: Execute jobs in parallel
4. **Metadata Store**: Stores job definitions and execution history
5. **Coordinator**: Manages worker health and job distribution

## Core Components

### 1. Job Definition

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class JobDefinition:
    job_id: str
    name: str
    schedule: str  # Cron expression or ISO timestamp
    command: str   # Shell command or function to execute
    dependencies: List[str]  # Job IDs this job depends on
    max_retries: int = 3
    timeout_seconds: int = 3600
    priority: int = 0  # Higher = more important
    metadata: dict = None
    
class JobScheduler:
    def create_job(self, job_def: JobDefinition):
        """Register a new job"""
        # Validate cron expression
        if not self.validate_cron(job_def.schedule):
            raise InvalidScheduleError()
        
        # Store in database
        self.db.execute(
            "INSERT INTO jobs (job_id, name, schedule, command, dependencies, "
            "max_retries, timeout_seconds, priority, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (job_def.job_id, job_def.name, job_def.schedule, job_def.command,
             json.dumps(job_def.dependencies), job_def.max_retries,
             job_def.timeout_seconds, job_def.priority, json.dumps(job_def.metadata))
        )
        
        # Update scheduler
        self.scheduler.add_job(job_def)
```

### 2. Scheduler (Time-based Triggering)

```python
import croniter
from datetime import datetime, timedelta

class CronScheduler:
    def __init__(self):
        self.jobs = {}  # job_id -> JobDefinition
        self.next_run_times = {}  # job_id -> next_run_time
    
    def add_job(self, job_def: JobDefinition):
        """Add job to scheduler"""
        self.jobs[job_def.job_id] = job_def
        self.next_run_times[job_def.job_id] = self.calculate_next_run(job_def)
    
    def calculate_next_run(self, job_def: JobDefinition):
        """Calculate next execution time"""
        if self.is_cron(job_def.schedule):
            # Cron expression (e.g., "0 0 * * *" for daily at midnight)
            cron = croniter.croniter(job_def.schedule, datetime.now())
            return cron.get_next(datetime)
        else:
            # One-time job (ISO timestamp)
            return datetime.fromisoformat(job_def.schedule)
    
    def get_due_jobs(self):
        """Get jobs that should run now"""
        now = datetime.now()
        due_jobs = []
        
        for job_id, next_run in self.next_run_times.items():
            if next_run <= now:
                job_def = self.jobs[job_id]
                due_jobs.append(job_def)
                
                # Update next run time
                if self.is_cron(job_def.schedule):
                    self.next_run_times[job_id] = self.calculate_next_run(job_def)
                else:
                    # One-time job, remove from scheduler
                    del self.next_run_times[job_id]
        
        return due_jobs
    
    def run_scheduler_loop(self):
        """Main scheduler loop"""
        while True:
            due_jobs = self.get_due_jobs()
            
            for job in due_jobs:
                # Check dependencies
                if self.dependencies_met(job):
                    # Enqueue job
                    self.enqueue_job(job)
                else:
                    # Reschedule for later
                    self.reschedule(job, delay_seconds=60)
            
            # Sleep until next check
            time.sleep(10)  # Check every 10 seconds
```

### 3. Dependency Management (DAG)

```python
class DAGScheduler:
    def __init__(self):
        self.job_graph = {}  # job_id -> list of dependent job_ids
        self.execution_status = {}  # job_id -> status
    
    def build_dag(self, jobs: List[JobDefinition]):
        """Build dependency graph"""
        for job in jobs:
            self.job_graph[job.job_id] = job.dependencies
    
    def dependencies_met(self, job_id: str):
        """Check if all dependencies have completed successfully"""
        dependencies = self.job_graph.get(job_id, [])
        
        for dep_id in dependencies:
            status = self.execution_status.get(dep_id)
            if status != 'SUCCESS':
                return False
        
        return True
    
    def get_ready_jobs(self):
        """Get jobs whose dependencies are satisfied"""
        ready = []
        
        for job_id in self.job_graph:
            if self.execution_status.get(job_id) is None:  # Not yet executed
                if self.dependencies_met(job_id):
                    ready.append(job_id)
        
        return ready
    
    def mark_completed(self, job_id: str, status: str):
        """Mark job as completed"""
        self.execution_status[job_id] = status
        
        # Trigger dependent jobs
        if status == 'SUCCESS':
            self.trigger_dependents(job_id)
    
    def trigger_dependents(self, job_id: str):
        """Trigger jobs that depend on this job"""
        for dependent_id, dependencies in self.job_graph.items():
            if job_id in dependencies and self.dependencies_met(dependent_id):
                self.enqueue_job(dependent_id)
```

### 4. Job Queue

```python
from kafka import KafkaProducer, KafkaConsumer
import json

class JobQueue:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def enqueue(self, job_def: JobDefinition):
        """Add job to queue"""
        message = {
            'job_id': job_def.job_id,
            'command': job_def.command,
            'timeout': job_def.timeout_seconds,
            'max_retries': job_def.max_retries,
            'priority': job_def.priority
        }
        
        # Use priority as partition key for priority queues
        partition = job_def.priority % 10
        
        self.producer.send(
            'job_queue',
            value=message,
            partition=partition
        )
    
    def dequeue(self, consumer_group='workers'):
        """Consume jobs from queue"""
        consumer = KafkaConsumer(
            'job_queue',
            bootstrap_servers=['localhost:9092'],
            group_id=consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for message in consumer:
            job = message.value
            yield job
```

### 5. Worker (Job Execution)

```python
import subprocess
import signal
from contextlib import contextmanager

class JobWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.queue = JobQueue()
        self.current_job = None
    
    def run(self):
        """Main worker loop"""
        for job in self.queue.dequeue():
            self.execute_job(job)
    
    def execute_job(self, job):
        """Execute a single job"""
        job_id = job['job_id']
        self.current_job = job_id
        
        # Update status to RUNNING
        self.update_status(job_id, 'RUNNING')
        
        try:
            # Execute with timeout
            with self.timeout(job['timeout']):
                result = subprocess.run(
                    job['command'],
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Success
            self.update_status(job_id, 'SUCCESS', output=result.stdout)
            self.log_execution(job_id, 'SUCCESS', result.stdout)
            
        except subprocess.TimeoutExpired:
            # Timeout
            self.handle_failure(job, 'TIMEOUT')
            
        except subprocess.CalledProcessError as e:
            # Command failed
            self.handle_failure(job, 'FAILED', error=e.stderr)
        
        finally:
            self.current_job = None
    
    def handle_failure(self, job, status, error=None):
        """Handle job failure with retries"""
        job_id = job['job_id']
        
        # Get retry count
        retry_count = self.get_retry_count(job_id)
        
        if retry_count < job['max_retries']:
            # Retry with exponential backoff
            delay = 2 ** retry_count  # 1s, 2s, 4s, 8s, ...
            self.schedule_retry(job, delay)
            self.update_status(job_id, 'RETRYING', retry_count=retry_count + 1)
        else:
            # Max retries exceeded
            self.update_status(job_id, status, error=error)
            self.log_execution(job_id, status, error)
    
    @contextmanager
    def timeout(self, seconds):
        """Context manager for timeout"""
        def timeout_handler(signum, frame):
            raise subprocess.TimeoutExpired(cmd='', timeout=seconds)
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
```

### 6. Metadata Store

```python
class MetadataStore:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create_execution_record(self, job_id, status='PENDING'):
        """Create execution record"""
        execution_id = str(uuid.uuid4())
        
        self.db.execute(
            "INSERT INTO job_executions (execution_id, job_id, status, "
            "start_time, worker_id) VALUES (%s, %s, %s, %s, %s)",
            (execution_id, job_id, status, datetime.now(), self.worker_id)
        )
        
        return execution_id
    
    def update_execution_status(self, execution_id, status, output=None, error=None):
        """Update execution status"""
        self.db.execute(
            "UPDATE job_executions SET status = %s, end_time = %s, "
            "output = %s, error = %s WHERE execution_id = %s",
            (status, datetime.now(), output, error, execution_id)
        )
    
    def get_execution_history(self, job_id, limit=100):
        """Get execution history for a job"""
        cursor = self.db.execute(
            "SELECT * FROM job_executions WHERE job_id = %s "
            "ORDER BY start_time DESC LIMIT %s",
            (job_id, limit)
        )
        
        return cursor.fetchall()
```

## Advanced Features

### 1. Distributed Coordination (Leader Election)

```python
from kazoo.client import KazooClient

class CoordinatorElection:
    def __init__(self):
        self.zk = KazooClient(hosts='localhost:2181')
        self.zk.start()
        self.is_leader = False
    
    def elect_leader(self):
        """Elect coordinator using ZooKeeper"""
        election_path = "/scheduler/leader"
        
        # Create ephemeral node
        try:
            self.zk.create(election_path, ephemeral=True, makepath=True)
            self.is_leader = True
            print(f"I am the leader!")
        except:
            self.is_leader = False
            print(f"Another instance is the leader")
        
        # Watch for leader changes
        @self.zk.DataWatch(election_path)
        def watch_leader(data, stat):
            if stat is None:  # Leader died
                self.elect_leader()
```

### 2. Job Idempotency

```python
class IdempotentJobExecutor:
    def execute_idempotent(self, job_id, command):
        """Execute job with idempotency guarantee"""
        # Check if already executed
        execution_key = f"executed:{job_id}:{date.today()}"
        
        if self.redis.exists(execution_key):
            print(f"Job {job_id} already executed today")
            return
        
        # Execute job
        result = self.execute(command)
        
        # Mark as executed (with expiry)
        self.redis.setex(execution_key, 86400, '1')  # 24 hours
        
        return result
```

### 3. Dynamic Priority Adjustment

```python
class PriorityManager:
    def adjust_priority(self, job_id):
        """Increase priority for overdue jobs"""
        job = self.get_job(job_id)
        scheduled_time = job.next_run_time
        delay = (datetime.now() - scheduled_time).total_seconds()
        
        if delay > 300:  # 5 minutes late
            # Increase priority
            new_priority = min(job.priority + 10, 100)
            self.update_job_priority(job_id, new_priority)
```

## Monitoring & Observability

```python
class JobMonitoring:
    def track_metrics(self, job_id, status, duration):
        """Track job execution metrics"""
        # Execution count
        self.statsd.increment(f'jobs.{status}.count')
        
        # Execution duration
        self.statsd.timing(f'jobs.{job_id}.duration', duration)
        
        # Success rate
        if status == 'SUCCESS':
            self.statsd.increment('jobs.success')
        else:
            self.statsd.increment('jobs.failure')
        
        # Queue depth
        queue_depth = self.get_queue_depth()
        self.statsd.gauge('jobs.queue_depth', queue_depth)
    
    def alert_on_failure(self, job_id, error):
        """Send alert on job failure"""
        if self.is_critical_job(job_id):
            self.pagerduty.trigger_incident(
                title=f"Critical job {job_id} failed",
                details=error
            )
```

## Key Takeaways

1. **Distributed coordination** requires leader election (ZooKeeper, etcd)
2. **Dependency management** uses DAG (Directed Acyclic Graph)
3. **Retry logic** with exponential backoff improves reliability
4. **Idempotency** prevents duplicate executions
5. **Priority queues** ensure critical jobs run first
6. **Monitoring** is essential for production systems
7. **Fault tolerance** requires persistent metadata storage

## References
- Apache Airflow: https://airflow.apache.org/
- Kubernetes CronJobs: https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/
- AWS Step Functions: https://aws.amazon.com/step-functions/
- Celery: https://docs.celeryproject.org/
