# Day 5.4: Resource Scheduling & Autoscaling Systems

## ⚖️ Compute & Accelerator Optimization - Part 4

**Focus**: GPU Cluster Scheduling, Dynamic Resource Allocation, Multi-Tenant Optimization  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master resource scheduling algorithms for GPU clusters and understand their theoretical foundations
- Learn dynamic resource allocation strategies for ML workloads
- Understand multi-tenant optimization and resource sharing policies
- Analyze performance isolation and fairness mechanisms in shared GPU environments

---

## 🗓️ GPU Cluster Scheduling Theory

### **Scheduling Problem Formulation**

GPU cluster scheduling for ML workloads presents unique challenges due to the discrete nature of GPU resources, long-running jobs, and varying resource requirements across different model architectures.

**Mathematical Framework:**
```
Scheduling Optimization Problem:
Maximize: Σᵢ wᵢ × completedᵢ(t) - Σⱼ pⱼ × penaltyⱼ(t)

Subject to:
- Resource constraints: Σᵢ rᵢⱼ ≤ Rⱼ ∀j (resources)
- Job requirements: allocatedᵢ ≥ min_resourcesᵢ ∀i (jobs)
- Fairness constraints: shareᵢ ≥ guaranteed_shareᵢ ∀i (users/teams)
- Locality constraints: co-location preferences for multi-GPU jobs

Where:
- wᵢ = priority weight for job i
- rᵢⱼ = resource j required by job i
- Rⱼ = total available resource j
- completedᵢ(t) = completion indicator for job i at time t
- penaltyⱼ(t) = penalty for constraint violations
```

### **GPU-Specific Scheduling Challenges**

**Resource Granularity Issues:**
```
Traditional CPU Scheduling: Fine-grained resource allocation (cores, memory)
GPU Scheduling Challenges:
1. Discrete GPU units: Cannot easily partition single GPU
2. Memory constraints: GPU memory often the limiting factor
3. Communication topology: GPU-to-GPU bandwidth varies significantly
4. Power and thermal limits: May limit concurrent GPU utilization

Bin Packing Problem:
- Jobs have multi-dimensional resource requirements [GPUs, CPU, memory]
- GPUs cannot be fractionally allocated (in most cases)
- Optimal packing is NP-hard
- Heuristics needed for practical solutions
```

**Gang Scheduling for Distributed Training:**
```
Distributed ML Requirements:
- All-or-nothing allocation: Distributed jobs need all resources simultaneously
- Communication sensitivity: Jobs sensitive to network topology
- Synchronization barriers: Periodic global synchronization points

Gang Scheduling Characteristics:
- Coordinated scheduling across multiple nodes
- Higher resource fragmentation
- Potential for convoy effects (large jobs blocking smaller ones)
- Better performance for parallel workloads
```

### **Multi-Level Scheduling Architecture**

**Hierarchical Scheduling Framework:**

**Level 1: Cluster-Level Scheduling**
```
Responsibilities:
- Quota enforcement across teams/users
- Fair sharing policies
- Resource reservation and allocation
- Long-term capacity planning

Algorithms:
- Dominant Resource Fairness (DRF)
- Weighted Fair Queuing
- Lottery Scheduling with resource awareness
```

**Level 2: Node-Level Scheduling**
```
Responsibilities:
- Bin packing optimization
- Local resource management
- Power and thermal management
- Hardware topology awareness

Optimization Objectives:
- Maximize utilization
- Minimize fragmentation
- Respect job placement preferences
- Balance power consumption
```

**Level 3: Job-Level Scheduling**
```
Responsibilities:
- Task scheduling within jobs
- Dynamic resource adjustment
- Fault tolerance and recovery
- Performance monitoring and optimization

Techniques:
- Elastic scaling for data parallel jobs
- Pipeline parallelism adjustment
- Dynamic batch size adaptation
```

---

## 🔄 Dynamic Resource Allocation Strategies

### **Elastic Training Theory**

**Elasticity Mathematical Model:**
```
Performance Model: P(n) = f(n, model, data)
Cost Model: C(n) = g(n, resource_prices)
Efficiency Model: E(n) = P(n) / C(n)

Optimal Resource Allocation:
n* = argmax E(n) subject to constraints

Dynamic Adjustment:
- Scale up: when marginal benefit > marginal cost
- Scale down: when efficiency decreases significantly
- Maintain minimum resources for convergence guarantees
```

**Scaling Efficiency Analysis:**
```
Strong Scaling Efficiency: η_s(n) = T(1) / (n × T(n))
Weak Scaling Efficiency: η_w(n) = T(1) / T(n) for proportionally scaled problem

Practical Considerations:
- Communication overhead increases with scale
- Synchronization delays grow with cluster size
- Memory per worker may decrease (model parallelism)

Efficiency Thresholds:
- Scale up if η_s(n+k) > threshold (e.g., 0.7)
- Scale down if η_s(n) < threshold
- Hysteresis to prevent oscillation
```

### **Adaptive Resource Management**

**Resource Demand Prediction:**
```
Demand Forecasting Model:
D(t+h) = f(D(t-k:t), job_characteristics, historical_patterns)

Features for Prediction:
1. Historical resource usage patterns
2. Job queue characteristics
3. Time-of-day patterns
4. Project/team submission patterns
5. Model architecture resource requirements

Prediction Horizon:
- Short-term (minutes): Reactive scaling
- Medium-term (hours): Proactive resource provisioning
- Long-term (days/weeks): Capacity planning
```

**Proactive vs Reactive Scaling:**
```
Reactive Scaling:
- Responds to current resource pressure
- Lower resource waste, higher response latency
- Suitable for cost-sensitive environments

Proactive Scaling:
- Anticipates future resource needs
- Higher resource waste, lower response latency
- Suitable for performance-critical environments

Hybrid Approach:
- Base allocation: Proactive based on predictions
- Burst capacity: Reactive based on current demand
- Economic optimization: Balance waste vs performance
```

---

## 🏢 Multi-Tenant GPU Optimization

### **Resource Isolation Mechanisms**

**GPU Virtualization Strategies:**

**1. Temporal Multiplexing**
```
Time-sharing Model:
- Single GPU serves multiple jobs sequentially
- Context switching overhead
- Memory isolation through save/restore

Performance Analysis:
T_effective = T_computation + T_context_switch × switch_frequency
Context switch time: 1-10ms depending on memory size
Suitable for: Inference workloads, development/debugging
```

**2. Spatial Partitioning (MIG)**
```
Multi-Instance GPU (NVIDIA A100/H100):
- Hardware-level isolation
- Fixed partitioning ratios (1/7, 2/7, 3/7, 4/7, 1/2, 1/1)
- Guaranteed memory and compute resources

MIG Partitioning Strategy:
Partition_config = {instances: [size₁, size₂, ..., sizeₖ]}
Subject to: Σᵢ sizeᵢ ≤ 1.0
Objective: Maximize allocation efficiency while meeting SLAs
```

**3. Memory Partitioning**
```
CUDA Memory Allocation Limits:
- Per-process memory limits
- Reserved memory pools
- Priority-based memory allocation

Implementation:
cudaMemPool_t pools[NUM_TENANTS];
cudaMemPoolSetAttribute(pool, cudaMemPoolReuseFollowEventDependencies, &enable);
```

### **Fair Sharing Algorithms**

**Dominant Resource Fairness (DRF) for GPUs:**
```
Multi-resource Fairness Problem:
Users have different resource preferences (CPU vs GPU vs Memory)

DRF Algorithm:
1. Calculate dominant share for each user:
   DS_i = max_r (allocated_i,r / total_r)
2. Allocate resources to user with smallest dominant share
3. Repeat until no more resources available

GPU-specific Adaptations:
- Consider GPU memory as separate resource from GPU compute
- Account for GPU-to-GPU communication bandwidth
- Include specialized hardware (Tensor cores, NVLink)
```

**Weighted Fair Queuing for ML Workloads:**
```
Priority-based Resource Allocation:
Virtual time: V_i(t) = V_i(t-1) + service_time / weight_i

Properties:
- Higher weight → lower virtual time advancement
- Resources allocated to job with lowest virtual time
- Provides both fairness and priority support

ML-specific Considerations:
- Gang scheduling requirements
- Minimum resource requirements per job
- Burst allowances for checkpointing/saving
```

### **Performance Isolation and QoS**

**Interference Mitigation Strategies:**

**Memory Bandwidth Isolation:**
```
Problem: Co-located jobs competing for memory bandwidth
Solution: Bandwidth throttling and reservation

Implementation Strategy:
1. Monitor memory bandwidth usage per job
2. Enforce limits through kernel scheduling
3. Reserve minimum bandwidth for high-priority jobs

Performance Impact:
- Memory-bound jobs: Significant impact (2-5x slowdown possible)
- Compute-bound jobs: Minimal impact
- Mixed workloads: Requires careful profiling and limits
```

**L2 Cache Partitioning:**
```
Cache Pollution Problem:
- Large datasets from one job evict cache lines of others
- Particularly problematic for inference workloads

Mitigation Techniques:
1. Cache-aware scheduling: Co-locate compatible workloads
2. Priority hints: Mark critical data for cache retention
3. Temporal separation: Separate I/O intensive from compute intensive
```

**Network QoS for Multi-GPU Jobs:**
```
InfiniBand/Ethernet QoS:
- Traffic classes with guaranteed bandwidth
- Priority queues for latency-sensitive communication
- Congestion control for bulk data transfers

Implementation:
- RDMA queue pairs with different service levels
- Traffic shaping at switch level
- Application-level flow control
```

---

## 📊 Autoscaling System Design

### **Autoscaling Decision Engine**

**Multi-Metric Scaling Policies:**
```
Scaling Decision Function:
scale_decision = f(CPU_util, GPU_util, Memory_util, Queue_length, Response_time)

Threshold-based Scaling:
- Scale up: if any metric > upper_threshold for duration > T_up
- Scale down: if all metrics < lower_threshold for duration > T_down
- Hysteresis: T_down >> T_up to prevent oscillation

Predictive Scaling:
predicted_load = LSTM(historical_metrics, time_features)
scale_decision = optimize(predicted_load, cost_model, performance_SLA)
```

**Scaling Velocity Control:**
```
Problem: Too aggressive scaling causes instability
Solution: Rate limiting with exponential backoff

Algorithm:
max_scale_out = min(current_instances × scale_factor, max_allowed_increment)
max_scale_in = min(current_instances × scale_factor, max_allowed_decrement)

Cooldown Periods:
- Scale-out cooldown: 5-15 minutes (allow metrics to stabilize)
- Scale-in cooldown: 10-30 minutes (avoid premature scale-down)
```

### **Cost-Performance Optimization**

**Economic Model for Autoscaling:**
```
Objective Function:
Total_Cost = Compute_Cost + Storage_Cost + Network_Cost + SLA_Penalty

Compute_Cost = Σᵢ (instance_hours_i × price_per_hour_i)
SLA_Penalty = Σⱼ max(0, response_time_j - SLA_target) × penalty_rate

Optimization:
minimize Total_Cost
subject to: SLA constraints, resource constraints, scaling velocity limits
```

**Spot Instance Integration:**
```
Spot Instance Strategy for ML Training:
1. Checkpointing frequency: Based on spot price volatility
2. Mixed instance types: Combine spot and on-demand
3. Fault tolerance: Automatic restart on spot termination

Risk Management:
Probability of interruption: P_interrupt(t) = historical_data + price_trends
Checkpoint interval: T_checkpoint = f(P_interrupt, checkpoint_cost, progress_loss)

Expected Cost Calculation:
E[Cost] = (1 - P_interrupt) × spot_price + P_interrupt × (spot_price + restart_cost)
```

---

## 🔧 Advanced Scheduling Algorithms

### **Machine Learning-Aware Scheduling**

**Job Characteristic-Based Scheduling:**
```
Job Profiling Features:
1. Resource requirements: [GPU_count, GPU_memory, CPU, RAM]
2. Communication pattern: [all-reduce_frequency, message_size]
3. I/O pattern: [read_bandwidth, write_bandwidth, storage_type]
4. Runtime characteristics: [estimated_duration, checkpointing_frequency]

Scheduling Algorithm:
1. Cluster jobs by resource requirements
2. Co-locate compatible jobs (similar communication patterns)
3. Anti-affinity for resource-competitive jobs
4. Topology-aware placement for multi-GPU jobs
```

**Priority Aging and Starvation Prevention:**
```
Age-based Priority Adjustment:
priority_effective(t) = priority_base + α × waiting_time(t)

Where α is the aging factor

Starvation Prevention:
- Maximum wait time limits
- Priority escalation for long-waiting jobs
- Resource reservation for starved jobs

Implementation:
if waiting_time > MAX_WAIT:
    priority = MAX_PRIORITY
    reserve_resources(job, min_resources)
```

### **Fault-Tolerant Scheduling**

**Checkpoint-Aware Scheduling:**
```
Checkpointing Strategy Integration:
1. Schedule jobs with similar checkpoint intervals together
2. Coordinate checkpoint timing to reduce I/O contention
3. Reserve storage bandwidth for checkpoint operations

Failure Recovery:
- Automatic restart from latest checkpoint
- Resource reallocation after node failures
- Degraded mode operation with reduced resources
```

**Preemption Policies:**
```
Preemption Decision Model:
preempt_benefit = priority_diff × remaining_work_preempted
preemption_cost = checkpoint_cost + restart_cost + wasted_work

Preemption Conditions:
1. preempt_benefit > preemption_cost
2. Preempted job can checkpoint within time limit
3. Higher priority job has been waiting > threshold

Graceful Preemption:
1. Send preemption warning to job
2. Allow time for checkpoint creation
3. Forcefully terminate if timeout exceeded
```

---

## 📈 Performance Monitoring and Optimization

### **Cluster-Level Metrics**

**Resource Utilization Metrics:**
```
GPU Utilization:
U_gpu = (Σᵢ active_gpu_time_i) / (total_gpus × measurement_period)

Memory Utilization:
U_memory = (Σᵢ allocated_memory_i) / (total_memory × measurement_period)

Queue Metrics:
- Average queue length
- Average wait time
- Job throughput (jobs completed per hour)
- SLA violation rate
```

**Scheduling Efficiency Metrics:**
```
Fragmentation Ratio:
F = (allocated_resources - used_resources) / allocated_resources

Fairness Index (Jain's Fairness):
J = (Σᵢ xᵢ)² / (n × Σᵢ xᵢ²)
where xᵢ is the resource allocation for user i

Load Balance:
LB = 1 - (max_load - min_load) / max_load
```

### **Optimization Feedback Loop**

**Adaptive Parameter Tuning:**
```
Parameters to Tune:
- Scaling thresholds
- Cooldown periods  
- Queue priorities
- Resource reservation ratios

Tuning Algorithm:
1. Monitor performance metrics
2. Identify suboptimal patterns
3. Adjust parameters using optimization algorithms
4. A/B test changes before full deployment

Multi-armed Bandit Approach:
- Each configuration is an "arm"
- Reward based on multiple objectives (utilization, fairness, cost)
- Explore-exploit trade-off in parameter space
```

This comprehensive framework for GPU cluster scheduling and autoscaling provides the theoretical foundation for building efficient, fair, and cost-effective ML infrastructure. The key insight is that effective resource management requires understanding the unique characteristics of ML workloads and their resource requirements across different scales and phases of the ML lifecycle.