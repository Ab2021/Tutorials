# Day 45: Distributed Job Scheduler - Deep Dive

## Advanced Implementation Patterns

### 1. Workflow Orchestration (DAG Execution)

```python
from collections import defaultdict, deque

class WorkflowEngine:
    """Execute complex workflows with dependencies"""
    
    def __init__(self):
        self.tasks = {}  # task_id -> Task
        self.graph = defaultdict(list)  # task_id -> [dependent_task_ids]
        self.reverse_graph = defaultdict(list)  # task_id -> [dependency_task_ids]
        self.execution_state = {}  # task_id -> ExecutionState
    
    def add_task(self, task_id, task_func, dependencies=None):
        """Add task to workflow"""
        self.tasks[task_id] = task_func
        
        if dependencies:
            for dep in dependencies:
                self.graph[dep].append(task_id)
                self.reverse_graph[task_id].append(dep)
    
    def execute_workflow(self):
        """Execute workflow using topological sort"""
        # Find tasks with no dependencies (starting points)
        in_degree = {task_id: len(deps) for task_id, deps in self.reverse_graph.items()}
        
        # Initialize queue with tasks that have no dependencies
        queue = deque([task_id for task_id in self.tasks if in_degree.get(task_id, 0) == 0])
        
        execution_order = []
        
        while queue:
            # Get next task to execute
            task_id = queue.popleft()
            execution_order.append(task_id)
            
            # Execute task
            self.execute_task(task_id)
            
            # Update dependent tasks
            for dependent in self.graph[task_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for cycles
        if len(execution_order) != len(self.tasks):
            raise CyclicDependencyError("Workflow contains cycles")
        
        return execution_order
    
    def execute_task(self, task_id):
        """Execute single task"""
        task_func = self.tasks[task_id]
        
        try:
            result = task_func()
            self.execution_state[task_id] = {
                'status': 'SUCCESS',
                'result': result,
                'timestamp': time.time()
            }
        except Exception as e:
            self.execution_state[task_id] = {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': time.time()
            }
            
            # Fail all dependent tasks
            self.fail_dependents(task_id)
    
    def fail_dependents(self, failed_task_id):
        """Mark all dependent tasks as failed"""
        visited = set()
        queue = deque([failed_task_id])
        
        while queue:
            task_id = queue.popleft()
            if task_id in visited:
                continue
            
            visited.add(task_id)
            
            for dependent in self.graph[task_id]:
                self.execution_state[dependent] = {
                    'status': 'SKIPPED',
                    'reason': f'Dependency {failed_task_id} failed',
                    'timestamp': time.time()
                }
                queue.append(dependent)
```

### 2. Dynamic Task Generation

```python
class DynamicWorkflow:
    """Generate tasks dynamically based on runtime data"""
    
    def execute_dynamic_workflow(self, initial_task):
        """Execute workflow with dynamic task generation"""
        completed_tasks = set()
        pending_tasks = [initial_task]
        
        while pending_tasks:
            task = pending_tasks.pop(0)
            
            # Execute task
            result = self.execute_task(task)
            completed_tasks.add(task.id)
            
            # Generate new tasks based on result
            new_tasks = task.generate_next_tasks(result)
            
            # Add new tasks to pending queue
            for new_task in new_tasks:
                if new_task.id not in completed_tasks:
                    pending_tasks.append(new_task)
        
        return completed_tasks

# Example: Fan-out pattern
class FanOutTask:
    def generate_next_tasks(self, result):
        """Generate multiple parallel tasks"""
        items = result['items']
        
        return [
            ProcessItemTask(item_id=item['id'])
            for item in items
        ]

# Example: Conditional branching
class ConditionalTask:
    def generate_next_tasks(self, result):
        """Generate tasks based on condition"""
        if result['value'] > 100:
            return [HighValueTask()]
        else:
            return [LowValueTask()]
```

### 3. Distributed Locking for Job Execution

```python
import redis
from contextlib import contextmanager

class DistributedJobLock:
    """Ensure only one instance of a job runs at a time"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    @contextmanager
    def acquire_lock(self, job_id, timeout=3600):
        """Acquire distributed lock for job"""
        lock_key = f"job_lock:{job_id}"
        lock_value = str(uuid.uuid4())
        
        # Try to acquire lock
        acquired = self.redis.set(
            lock_key,
            lock_value,
            nx=True,  # Only set if not exists
            ex=timeout  # Expiry in seconds
        )
        
        if not acquired:
            raise JobAlreadyRunningError(f"Job {job_id} is already running")
        
        try:
            yield
        finally:
            # Release lock (only if we still own it)
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            self.redis.eval(lua_script, 1, lock_key, lock_value)

# Usage
def execute_job_with_lock(job_id):
    with lock_manager.acquire_lock(job_id):
        # Execute job
        run_job(job_id)
```

### 4. Job Batching and Coalescing

```python
class JobBatcher:
    """Batch multiple similar jobs into one execution"""
    
    def __init__(self, batch_size=100, max_wait_seconds=60):
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self.pending_jobs = defaultdict(list)
        self.batch_timers = {}
    
    def add_job(self, job_type, job_data):
        """Add job to batch"""
        self.pending_jobs[job_type].append(job_data)
        
        # Start timer if first job of this type
        if len(self.pending_jobs[job_type]) == 1:
            self.batch_timers[job_type] = time.time()
        
        # Execute batch if size threshold reached
        if len(self.pending_jobs[job_type]) >= self.batch_size:
            self.execute_batch(job_type)
        
        # Execute batch if time threshold reached
        elif time.time() - self.batch_timers[job_type] >= self.max_wait_seconds:
            self.execute_batch(job_type)
    
    def execute_batch(self, job_type):
        """Execute batched jobs"""
        jobs = self.pending_jobs[job_type]
        
        if not jobs:
            return
        
        # Execute all jobs in batch
        batch_executor = self.get_batch_executor(job_type)
        batch_executor.execute_batch(jobs)
        
        # Clear batch
        self.pending_jobs[job_type] = []
        del self.batch_timers[job_type]

# Example: Batch email sending
class EmailBatchExecutor:
    def execute_batch(self, email_jobs):
        """Send multiple emails in one API call"""
        recipients = [job['recipient'] for job in email_jobs]
        
        # Single API call for all emails
        self.email_service.send_bulk(recipients)
```

### 5. Job Checkpointing and Recovery

```python
class CheckpointableJob:
    """Job that can resume from checkpoint after failure"""
    
    def __init__(self, job_id):
        self.job_id = job_id
        self.checkpoint_store = Redis()
    
    def execute_with_checkpoints(self, items):
        """Execute job with periodic checkpointing"""
        # Load last checkpoint
        last_checkpoint = self.load_checkpoint()
        start_index = last_checkpoint.get('index', 0)
        
        for i in range(start_index, len(items)):
            item = items[i]
            
            try:
                # Process item
                self.process_item(item)
                
                # Save checkpoint every 100 items
                if i % 100 == 0:
                    self.save_checkpoint({'index': i})
            
            except Exception as e:
                # Save checkpoint before failing
                self.save_checkpoint({'index': i, 'error': str(e)})
                raise
        
        # Clear checkpoint on success
        self.clear_checkpoint()
    
    def save_checkpoint(self, data):
        """Save checkpoint to Redis"""
        key = f"checkpoint:{self.job_id}"
        self.checkpoint_store.setex(key, 86400, json.dumps(data))
    
    def load_checkpoint(self):
        """Load checkpoint from Redis"""
        key = f"checkpoint:{self.job_id}"
        data = self.checkpoint_store.get(key)
        return json.loads(data) if data else {}
    
    def clear_checkpoint(self):
        """Clear checkpoint after successful completion"""
        key = f"checkpoint:{self.job_id}"
        self.checkpoint_store.delete(key)
```

### 6. Resource-Aware Scheduling

```python
class ResourceAwareScheduler:
    """Schedule jobs based on available resources"""
    
    def __init__(self):
        self.workers = {}  # worker_id -> WorkerResources
        self.pending_jobs = []  # Priority queue of jobs
    
    def schedule_job(self, job):
        """Schedule job to worker with sufficient resources"""
        required_resources = job.resource_requirements
        
        # Find worker with sufficient resources
        for worker_id, worker in self.workers.items():
            if worker.has_capacity(required_resources):
                # Assign job to worker
                worker.allocate_resources(required_resources)
                self.send_job_to_worker(worker_id, job)
                return
        
        # No worker available, add to pending queue
        heapq.heappush(self.pending_jobs, (-job.priority, job))
    
    def on_job_completed(self, worker_id, job):
        """Release resources when job completes"""
        worker = self.workers[worker_id]
        worker.release_resources(job.resource_requirements)
        
        # Try to schedule pending jobs
        while self.pending_jobs:
            _, pending_job = heapq.heappop(self.pending_jobs)
            
            if worker.has_capacity(pending_job.resource_requirements):
                worker.allocate_resources(pending_job.resource_requirements)
                self.send_job_to_worker(worker_id, pending_job)
                break
            else:
                # Put back in queue
                heapq.heappush(self.pending_jobs, (-pending_job.priority, pending_job))
                break

class WorkerResources:
    def __init__(self, cpu_cores, memory_gb, gpu_count=0):
        self.total_cpu = cpu_cores
        self.total_memory = memory_gb
        self.total_gpu = gpu_count
        
        self.available_cpu = cpu_cores
        self.available_memory = memory_gb
        self.available_gpu = gpu_count
    
    def has_capacity(self, requirements):
        """Check if worker has sufficient resources"""
        return (
            self.available_cpu >= requirements.get('cpu', 0) and
            self.available_memory >= requirements.get('memory', 0) and
            self.available_gpu >= requirements.get('gpu', 0)
        )
    
    def allocate_resources(self, requirements):
        """Allocate resources for job"""
        self.available_cpu -= requirements.get('cpu', 0)
        self.available_memory -= requirements.get('memory', 0)
        self.available_gpu -= requirements.get('gpu', 0)
    
    def release_resources(self, requirements):
        """Release resources after job completion"""
        self.available_cpu += requirements.get('cpu', 0)
        self.available_memory += requirements.get('memory', 0)
        self.available_gpu += requirements.get('gpu', 0)
```

### 7. Job Versioning and Rollback

```python
class VersionedJobScheduler:
    """Support multiple versions of job definitions"""
    
    def __init__(self):
        self.job_versions = {}  # (job_id, version) -> JobDefinition
        self.active_versions = {}  # job_id -> version
    
    def deploy_job_version(self, job_id, version, job_def):
        """Deploy new version of job"""
        # Store new version
        self.job_versions[(job_id, version)] = job_def
        
        # Update active version
        self.active_versions[job_id] = version
        
        # Log deployment
        self.log_deployment(job_id, version)
    
    def rollback_job(self, job_id, target_version):
        """Rollback job to previous version"""
        if (job_id, target_version) not in self.job_versions:
            raise VersionNotFoundError(f"Version {target_version} not found")
        
        # Revert to target version
        self.active_versions[job_id] = target_version
        
        # Log rollback
        self.log_rollback(job_id, target_version)
    
    def get_active_job_definition(self, job_id):
        """Get currently active version of job"""
        version = self.active_versions.get(job_id)
        if not version:
            raise JobNotFoundError(f"Job {job_id} not found")
        
        return self.job_versions[(job_id, version)]
```

### 8. Advanced Retry Strategies

```python
class SmartRetryManager:
    """Intelligent retry logic with multiple strategies"""
    
    def __init__(self):
        self.retry_strategies = {
            'exponential_backoff': self.exponential_backoff,
            'linear_backoff': self.linear_backoff,
            'fibonacci_backoff': self.fibonacci_backoff,
            'jittered_backoff': self.jittered_backoff
        }
    
    def exponential_backoff(self, attempt):
        """Exponential backoff: 1s, 2s, 4s, 8s, ..."""
        return 2 ** attempt
    
    def linear_backoff(self, attempt):
        """Linear backoff: 1s, 2s, 3s, 4s, ..."""
        return attempt + 1
    
    def fibonacci_backoff(self, attempt):
        """Fibonacci backoff: 1s, 1s, 2s, 3s, 5s, 8s, ..."""
        if attempt <= 1:
            return 1
        
        a, b = 1, 1
        for _ in range(attempt - 1):
            a, b = b, a + b
        return b
    
    def jittered_backoff(self, attempt):
        """Exponential backoff with jitter to avoid thundering herd"""
        base_delay = 2 ** attempt
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
    
    def should_retry(self, job, error):
        """Determine if job should be retried based on error type"""
        # Don't retry on certain errors
        non_retryable_errors = [
            'InvalidInputError',
            'AuthenticationError',
            'PermissionDeniedError'
        ]
        
        if error.__class__.__name__ in non_retryable_errors:
            return False
        
        # Check retry count
        if job.retry_count >= job.max_retries:
            return False
        
        return True
    
    def schedule_retry(self, job, error):
        """Schedule job retry with appropriate delay"""
        if not self.should_retry(job, error):
            return False
        
        # Get retry strategy
        strategy = job.retry_strategy or 'exponential_backoff'
        backoff_func = self.retry_strategies[strategy]
        
        # Calculate delay
        delay = backoff_func(job.retry_count)
        
        # Schedule retry
        retry_time = datetime.now() + timedelta(seconds=delay)
        self.schedule_job_at(job, retry_time)
        
        # Increment retry count
        job.retry_count += 1
        
        return True
```

### 9. Job Observability Dashboard

```python
class JobMetricsDashboard:
    """Real-time dashboard for job monitoring"""
    
    def get_dashboard_metrics(self):
        """Get metrics for dashboard"""
        return {
            'total_jobs': self.get_total_jobs(),
            'running_jobs': self.get_running_jobs(),
            'pending_jobs': self.get_pending_jobs(),
            'failed_jobs_24h': self.get_failed_jobs_last_24h(),
            'success_rate': self.get_success_rate(),
            'avg_execution_time': self.get_avg_execution_time(),
            'queue_depth': self.get_queue_depth(),
            'worker_utilization': self.get_worker_utilization(),
            'top_failing_jobs': self.get_top_failing_jobs(limit=10),
            'slowest_jobs': self.get_slowest_jobs(limit=10)
        }
    
    def get_job_timeline(self, job_id):
        """Get execution timeline for a job"""
        executions = self.db.query(
            "SELECT * FROM job_executions WHERE job_id = %s "
            "ORDER BY start_time DESC LIMIT 100",
            (job_id,)
        )
        
        timeline = []
        for execution in executions:
            timeline.append({
                'execution_id': execution['execution_id'],
                'status': execution['status'],
                'start_time': execution['start_time'],
                'end_time': execution['end_time'],
                'duration': (execution['end_time'] - execution['start_time']).total_seconds(),
                'worker_id': execution['worker_id']
            })
        
        return timeline
```

### 10. Cost Optimization

```python
class CostOptimizedScheduler:
    """Schedule jobs to minimize cloud costs"""
    
    def __init__(self):
        self.spot_instance_price = 0.03  # per hour
        self.on_demand_price = 0.10  # per hour
    
    def schedule_with_cost_optimization(self, job):
        """Schedule job considering cost"""
        # Use spot instances for non-critical jobs
        if job.priority < 5 and job.max_execution_time < 3600:
            return self.schedule_on_spot_instance(job)
        
        # Use on-demand for critical jobs
        else:
            return self.schedule_on_demand(job)
    
    def schedule_during_off_peak(self, job):
        """Schedule non-urgent jobs during off-peak hours"""
        current_hour = datetime.now().hour
        
        # Off-peak hours: 10 PM - 6 AM
        if 22 <= current_hour or current_hour < 6:
            self.execute_job_now(job)
        else:
            # Schedule for next off-peak window
            next_off_peak = datetime.now().replace(hour=22, minute=0, second=0)
            if current_hour >= 22:
                next_off_peak += timedelta(days=1)
            
            self.schedule_job_at(job, next_off_peak)
```

## Production Best Practices

1. **Idempotency**: Ensure jobs can be safely retried
2. **Timeouts**: Always set execution timeouts
3. **Monitoring**: Track success rate, latency, queue depth
4. **Alerting**: Alert on critical job failures
5. **Graceful Shutdown**: Handle worker termination gracefully
6. **Dead Letter Queue**: Capture permanently failed jobs
7. **Rate Limiting**: Prevent overwhelming downstream services
8. **Backpressure**: Handle queue overflow gracefully

## Key Takeaways

1. **DAG execution** enables complex workflows
2. **Distributed locking** prevents duplicate executions
3. **Checkpointing** enables recovery from failures
4. **Resource-aware scheduling** optimizes utilization
5. **Smart retries** improve reliability
6. **Observability** is critical for debugging
7. **Cost optimization** reduces cloud spend
