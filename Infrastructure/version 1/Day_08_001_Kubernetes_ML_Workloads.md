# Day 8.1: Kubernetes for ML Workloads

## ☸️ Infrastructure as Code & Automation - Part 1

**Focus**: Container Orchestration, Resource Management, ML-Specific Kubernetes Patterns  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master Kubernetes orchestration patterns specifically designed for ML workloads
- Learn advanced resource management and auto-scaling strategies for ML systems
- Understand StatefulSets, custom resources, and ML operators for complex deployments
- Analyze performance optimization and cost management in Kubernetes environments

---

## ☸️ Kubernetes Theoretical Framework for ML

### **Container Orchestration for ML Workloads**

Kubernetes provides a powerful platform for orchestrating ML workloads, but ML systems have unique requirements that differ from traditional web applications.

**ML Workload Characteristics:**
```
ML Workload Classification:
1. Training Workloads:
   - Resource-intensive: High CPU/GPU/memory requirements
   - Batch processing: Long-running jobs with defined completion
   - Variable duration: Minutes to days depending on model complexity
   - Fault tolerance: Checkpointing and restart capabilities required

2. Inference Workloads:
   - Latency-sensitive: Sub-second response time requirements
   - Stateless/stateful: Depending on model complexity and caching needs
   - Scalable: Dynamic scaling based on request volume
   - High availability: Multiple replicas with load balancing

3. Data Processing Workloads:
   - I/O intensive: High disk and network throughput requirements
   - Streaming/batch: Real-time and scheduled processing patterns
   - Memory intensive: Large datasets require substantial RAM
   - Parallel processing: Distributed computation across multiple nodes

4. Pipeline Orchestration:
   - Multi-stage workflows: Complex dependency graphs
   - Resource coordination: Different resource needs per stage
   - Data flow management: Efficient data passing between stages
   - Failure handling: Stage-level retry and recovery mechanisms

Kubernetes Resource Model for ML:
Resource_Requirements = f(Workload_Type, Data_Size, Model_Complexity, SLA_Requirements)

Training Resource Calculation:
Training_Resources = {
    'cpu': base_cpu + (data_size_gb * cpu_per_gb) + (model_params * cpu_per_param),
    'memory': base_memory + (data_size_gb * memory_per_gb) + model_memory,
    'gpu': ceil(model_complexity / gpu_capacity),
    'storage': data_size + model_size + checkpoint_size
}
```

**Kubernetes Scheduling for ML:**
```
ML-Aware Scheduling Considerations:
1. Resource Affinity:
   - GPU node affinity for training workloads
   - High-memory node affinity for large model inference
   - SSD affinity for I/O intensive data processing
   - Network affinity for distributed training

2. Anti-Affinity Patterns:
   - Spread replicas across availability zones
   - Avoid co-locating resource-intensive workloads
   - Separate training and inference workloads
   - Distribute data processing across nodes

3. Topology-Aware Scheduling:
   - InfiniBand network topology for distributed training
   - NUMA topology awareness for memory optimization
   - GPU interconnect topology (NVLink, NVSwitch)
   - Storage topology for data locality

Kubernetes Scheduler Extensions:
apiVersion: v1
kind: Pod
spec:
  schedulerName: ml-aware-scheduler
  nodeSelector:
    ml-workload-type: "training"
    gpu-type: "v100"
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/arch
            operator: In
            values: ["amd64"]
          - key: accelerator
            operator: In
            values: ["nvidia-tesla-v100"]
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: workload-type
              operator: In
              values: ["training"]
          topologyKey: kubernetes.io/hostname

Custom Scheduler for ML Workloads:
type MLScheduler struct {
    gpuManager     GPUManager
    networkManager NetworkManager
    costOptimizer  CostOptimizer
}

func (s *MLScheduler) Schedule(pod *v1.Pod, nodes []*v1.Node) (*v1.Node, error) {
    // Extract ML-specific requirements
    requirements := s.extractMLRequirements(pod)
    
    // Filter nodes based on resource availability
    candidateNodes := s.filterNodesByResources(nodes, requirements)
    
    // Score nodes based on ML-specific criteria
    scoredNodes := s.scoreNodesForML(candidateNodes, requirements)
    
    // Select optimal node
    selectedNode := s.selectOptimalNode(scoredNodes)
    
    return selectedNode, nil
}

func (s *MLScheduler) scoreNodesForML(nodes []*v1.Node, req MLRequirements) []ScoredNode {
    var scored []ScoredNode
    
    for _, node := range nodes {
        score := 0.0
        
        // GPU utilization score
        gpuScore := s.calculateGPUScore(node, req.GPURequirements)
        score += gpuScore * 0.4
        
        // Network topology score
        networkScore := s.calculateNetworkScore(node, req.NetworkRequirements)
        score += networkScore * 0.3
        
        // Cost efficiency score
        costScore := s.calculateCostScore(node, req.ResourceRequirements)
        score += costScore * 0.2
        
        // Data locality score
        localityScore := s.calculateLocalityScore(node, req.DataRequirements)
        score += localityScore * 0.1
        
        scored = append(scored, ScoredNode{Node: node, Score: score})
    }
    
    return scored
}
```

---

## 📦 Advanced Resource Management

### **Resource Quotas and Limits for ML**

**Multi-Tenant Resource Management:**
```
Resource Quota Strategy for ML Teams:
1. Hierarchical Resource Allocation:
   Organization Level: Total cluster resources
   ├── Department Level: Percentage allocation per department
   │   ├── Team Level: Team-specific quotas and priorities
   │   │   ├── User Level: Individual user limits
   │   │   └── Project Level: Project-specific allocations

Resource Quota Configuration:
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team-alpha
spec:
  hard:
    # Compute resources
    requests.cpu: "100"
    requests.memory: "500Gi"
    requests.nvidia.com/gpu: "20"
    limits.cpu: "200"
    limits.memory: "1000Gi"
    limits.nvidia.com/gpu: "20"
    
    # Storage resources
    requests.storage: "10Ti"
    persistentvolumeclaims: "50"
    
    # Object counts
    pods: "100"
    services: "20"
    secrets: "50"
    configmaps: "50"
    
    # Custom resources
    training-jobs: "10"
    inference-services: "20"

Limit Range for ML Workloads:
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-workload-limits
spec:
  limits:
  # Training workloads
  - type: Container
    default:
      cpu: "4"
      memory: "16Gi"
      nvidia.com/gpu: "1"
    defaultRequest:
      cpu: "2"
      memory: "8Gi"
    max:
      cpu: "32"
      memory: "256Gi"
      nvidia.com/gpu: "8"
    min:
      cpu: "0.5"
      memory: "1Gi"
  
  # Inference workloads  
  - type: Container
    default:
      cpu: "1"
      memory: "4Gi"
    defaultRequest:
      cpu: "0.5"
      memory: "2Gi"
    max:
      cpu: "8"
      memory: "32Gi"
      nvidia.com/gpu: "2"

Priority-Based Resource Allocation:
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-training-high-priority
value: 1000
globalDefault: false
description: "High priority for critical ML training jobs"

---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-inference-critical
value: 2000
globalDefault: false
description: "Critical priority for production inference workloads"

---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: ml-experimentation
value: 100
globalDefault: false
description: "Low priority for experimental workloads"
```

**Dynamic Resource Allocation:**
```
Vertical Pod Autoscaler (VPA) for ML:
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-training-vpa
spec:
  targetRef:
    apiVersion: batch/v1
    kind: Job
    name: model-training-job
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: training-container
      minAllowed:
        cpu: "1"
        memory: "2Gi"
      maxAllowed:
        cpu: "32"
        memory: "256Gi"
      controlledResources: ["cpu", "memory"]

Horizontal Pod Autoscaler (HPA) for Inference:
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-inference-service
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60

Cluster Autoscaler for ML Workloads:
# Node pool configuration for training workloads
resource "google_container_node_pool" "ml_training_pool" {
  name       = "ml-training-pool"
  cluster    = google_container_cluster.ml_cluster.name
  node_count = 0  # Start with 0, scale as needed
  
  autoscaling {
    min_node_count = 0
    max_node_count = 20
  }
  
  node_config {
    machine_type = "n1-highmem-8"
    
    guest_accelerator {
      type  = "nvidia-tesla-v100"
      count = 2
    }
    
    disk_size_gb = 500
    disk_type    = "pd-ssd"
    
    labels = {
      workload-type = "training"
      gpu-type     = "v100"
    }
    
    taint {
      key    = "training-workload"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  }
}

# Node pool configuration for inference workloads
resource "google_container_node_pool" "ml_inference_pool" {
  name       = "ml-inference-pool"
  cluster    = google_container_cluster.ml_cluster.name
  node_count = 2  # Minimum for high availability
  
  autoscaling {
    min_node_count = 2
    max_node_count = 50
  }
  
  node_config {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-standard"
    
    labels = {
      workload-type = "inference"
    }
  }
}
```

### **GPU Resource Management**

**GPU Sharing and Virtualization:**
```
GPU Resource Allocation Strategies:
1. Exclusive GPU Allocation:
   - One GPU per pod/container
   - Maximum performance isolation
   - Resource wastage for small models
   - Simple scheduling and management

2. GPU Sharing (Time-Slicing):
   - Multiple pods share same GPU
   - Time-based scheduling of GPU access
   - Improved resource utilization
   - Potential performance interference

3. Multi-Instance GPU (MIG):
   - Hardware-level GPU partitioning
   - Isolated GPU memory and compute
   - Supported on A100 and H100 GPUs
   - Fine-grained resource allocation

4. Virtual GPU (vGPU):
   - Software-based GPU virtualization
   - Fractional GPU allocation
   - Memory and compute isolation
   - Requires compatible hardware/software

GPU Resource Configuration:
# Enable GPU sharing with time-slicing
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-sharing-config
data:
  config.yaml: |
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4  # Allow 4 pods to share each GPU

# MIG-enabled GPU configuration
apiVersion: v1
kind: Node
metadata:
  name: gpu-node-mig
  labels:
    nvidia.com/mig.strategy: mixed
spec:
  allocatable:
    nvidia.com/gpu: "7"  # A100 can be partitioned into 7 MIG instances
    nvidia.com/mig-1g.5gb: "7"
    nvidia.com/mig-2g.10gb: "3"
    nvidia.com/mig-3g.20gb: "2"
    nvidia.com/mig-7g.40gb: "1"

GPU Workload Scheduling:
type GPUScheduler struct {
    gpuInventory map[string]*GPUNode
    allocation   map[string]*GPUAllocation
}

func (s *GPUScheduler) AllocateGPU(request *GPURequest) (*GPUAllocation, error) {
    // Determine allocation strategy based on workload type
    strategy := s.determineAllocationStrategy(request)
    
    switch strategy {
    case ExclusiveAllocation:
        return s.allocateExclusiveGPU(request)
    case SharedAllocation:
        return s.allocateSharedGPU(request)
    case MIGAllocation:
        return s.allocateMIGGPU(request)
    case VirtualAllocation:
        return s.allocateVirtualGPU(request)
    }
    
    return nil, errors.New("no suitable GPU allocation found")
}

func (s *GPUScheduler) determineAllocationStrategy(request *GPURequest) AllocationStrategy {
    // Training workloads typically need exclusive access
    if request.WorkloadType == "training" && request.GPUMemoryGB > 16 {
        return ExclusiveAllocation
    }
    
    // Small inference workloads can share GPUs
    if request.WorkloadType == "inference" && request.GPUMemoryGB < 4 {
        return SharedAllocation
    }
    
    // Medium workloads benefit from MIG
    if request.GPUMemoryGB >= 4 && request.GPUMemoryGB <= 20 {
        return MIGAllocation
    }
    
    return ExclusiveAllocation
}

GPU Utilization Monitoring:
class GPUUtilizationMonitor:
    def __init__(self):
        self.metrics_collector = PrometheusMetrics()
        self.gpu_manager = GPUManager()
    
    def collect_gpu_metrics(self):
        for node in self.gpu_manager.get_gpu_nodes():
            for gpu in node.gpus:
                metrics = self.gpu_manager.get_gpu_metrics(gpu.id)
                
                # Core utilization metrics
                self.metrics_collector.gauge('gpu_utilization_percent').labels(
                    node=node.name, gpu=gpu.id
                ).set(metrics.utilization_percent)
                
                # Memory utilization
                self.metrics_collector.gauge('gpu_memory_utilization_percent').labels(
                    node=node.name, gpu=gpu.id
                ).set(metrics.memory_utilization_percent)
                
                # Temperature and power
                self.metrics_collector.gauge('gpu_temperature_celsius').labels(
                    node=node.name, gpu=gpu.id
                ).set(metrics.temperature)
                
                self.metrics_collector.gauge('gpu_power_watts').labels(
                    node=node.name, gpu=gpu.id
                ).set(metrics.power_usage)
```

---

## 🔄 StatefulSets and Persistent Storage

### **StatefulSets for ML Applications**

**Stateful ML Workloads:**
```
StatefulSet Use Cases in ML:
1. Distributed Training:
   - Parameter servers in distributed training
   - Consistent network identities for communication
   - Ordered deployment and scaling
   - Persistent storage for checkpoints

2. Feature Stores:
   - Online feature serving with persistent state
   - Consistent storage across replicas
   - Ordered deployment for data consistency
   - Persistent volumes for feature data

3. Model Serving with State:
   - Models with internal state (RNNs, memory networks)
   - Session-based inference with user context
   - Caching layers with persistent storage
   - A/B testing with consistent user routing

4. Data Processing Pipelines:
   - Streaming processors with state
   - Watermark management in event processing
   - Checkpoint storage for fault tolerance
   - Ordered processing guarantees

Distributed Training StatefulSet:
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: distributed-training
spec:
  serviceName: "training-headless"
  replicas: 4
  selector:
    matchLabels:
      app: distributed-training
  template:
    metadata:
      labels:
        app: distributed-training
    spec:
      containers:
      - name: training-worker
        image: tensorflow/tensorflow:2.11.0-gpu
        command:
        - python
        - distributed_training.py
        - --ps_hosts=$(PS_HOSTS)
        - --worker_hosts=$(WORKER_HOSTS)
        - --job_name=worker
        - --task_index=$(POD_INDEX)
        env:
        - name: PS_HOSTS
          value: "training-ps-0.training-headless:2222,training-ps-1.training-headless:2222"
        - name: WORKER_HOSTS
          value: "distributed-training-0.training-headless:2223,distributed-training-1.training-headless:2223,distributed-training-2.training-headless:2223,distributed-training-3.training-headless:2223"
        - name: POD_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['statefulset.kubernetes.io/pod-name']
        ports:
        - containerPort: 2223
          name: worker-port
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "4"
            memory: "16Gi"
          limits:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
        volumeMounts:
        - name: checkpoint-storage
          mountPath: /checkpoints
        - name: data-storage
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: checkpoint-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi
  - metadata:
      name: data-storage
    spec:
      accessModes: ["ReadOnlyMany"]
      storageClassName: "shared-nfs"
      resources:
        requests:
          storage: 1Ti

Feature Store StatefulSet:
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: online-feature-store
spec:
  serviceName: "feature-store-headless"
  replicas: 3
  selector:
    matchLabels:
      app: online-feature-store
  template:
    metadata:
      labels:
        app: online-feature-store
    spec:
      containers:
      - name: feature-server
        image: feast/feature-server:latest
        ports:
        - containerPort: 6566
          name: grpc
        - containerPort: 8080
          name: rest
        env:
        - name: FEAST_REDIS_HOST
          value: "redis-cluster"
        - name: FEAST_OFFLINE_STORE_TYPE
          value: "bigquery"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        volumeMounts:
        - name: feature-config
          mountPath: /feature_repo
        - name: feature-cache
          mountPath: /cache
        livenessProbe:
          grpc:
            port: 6566
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          grpc:
            port: 6566
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: feature-cache
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "high-iops-ssd"
      resources:
        requests:
          storage: 50Gi
```

### **Persistent Volume Strategies**

**Storage Classes for ML Workloads:**
```
Storage Performance Characteristics:
1. High-Performance Storage (NVMe SSD):
   - Use case: Model checkpoints, temporary data
   - IOPS: >100,000 IOPS
   - Latency: <1ms
   - Cost: High

2. Balanced Storage (SSD):
   - Use case: Feature data, model artifacts
   - IOPS: 10,000-50,000 IOPS  
   - Latency: 1-5ms
   - Cost: Medium

3. High-Throughput Storage (HDD):
   - Use case: Training datasets, logs
   - Throughput: >1GB/s sequential
   - Latency: 10-50ms
   - Cost: Low

4. Shared Storage (NFS/EFS):
   - Use case: Shared datasets, model registry
   - Concurrent access: Multiple pods
   - Consistency: Strong consistency
   - Cost: Variable

Storage Class Definitions:
# High-performance NVMe storage
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nvme-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  zones: us-central1-a,us-central1-b,us-central1-c
  replication-type: regional-pd
allowVolumeExpansion: true
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

# Shared NFS storage for datasets
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: shared-nfs
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs-server.default.svc.cluster.local
  share: /shared/datasets
  subdir: ${pvc.metadata.namespace}-${pvc.metadata.name}
allowVolumeExpansion: true
reclaimPolicy: Retain
volumeBindingMode: Immediate

# High-throughput storage for large datasets
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: high-throughput
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-standard
  zones: us-central1-a,us-central1-b,us-central1-c
  replication-type: regional-pd
allowVolumeExpansion: true
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer

Data Locality Optimization:
class DataLocalityScheduler:
    def __init__(self):
        self.storage_topology = self.build_storage_topology()
        self.data_location_cache = {}
    
    def schedule_with_data_locality(self, pod, available_nodes):
        # Identify data requirements
        data_requirements = self.extract_data_requirements(pod)
        
        # Score nodes based on data locality
        scored_nodes = []
        for node in available_nodes:
            locality_score = self.calculate_data_locality_score(node, data_requirements)
            network_score = self.calculate_network_proximity_score(node, data_requirements)
            storage_score = self.calculate_storage_performance_score(node, data_requirements)
            
            total_score = (
                locality_score * 0.5 +
                network_score * 0.3 +
                storage_score * 0.2
            )
            
            scored_nodes.append((node, total_score))
        
        # Select node with highest score
        best_node = max(scored_nodes, key=lambda x: x[1])[0]
        return best_node
    
    def calculate_data_locality_score(self, node, data_requirements):
        score = 0.0
        total_data_size = 0
        
        for data_volume in data_requirements:
            data_size = data_volume.size
            total_data_size += data_size
            
            # Check if data is already on this node
            if self.is_data_local(node, data_volume):
                score += data_size
            # Check if data is in same zone
            elif self.is_data_in_same_zone(node, data_volume):
                score += data_size * 0.7
            # Check if data is in same region
            elif self.is_data_in_same_region(node, data_volume):
                score += data_size * 0.3
        
        return score / total_data_size if total_data_size > 0 else 0

Volume Snapshot and Backup:
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: model-checkpoint-snapshot-v1
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: model-checkpoints-pvc

---
apiVersion: v1
kind: CronJob
metadata:
  name: model-backup-job
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup-container
            image: backup-tool:latest
            command:
            - /bin/bash
            - -c
            - |
              # Create snapshot
              kubectl create -f snapshot-template.yaml
              
              # Wait for snapshot to be ready
              kubectl wait --for=condition=ReadyToUse volumesnapshot/model-checkpoint-snapshot-$(date +%Y%m%d)
              
              # Export to external storage
              backup-tool --source=snapshot --destination=gs://ml-backups/$(date +%Y%m%d)/
            volumeMounts:
            - name: kubectl-config
              mountPath: /root/.kube
          volumes:
          - name: kubectl-config
            secret:
              secretName: kubectl-config
          restartPolicy: OnFailure
```

---

## 🤖 ML Operators and Custom Resources

### **Kubernetes Operators for ML**

**Custom Resource Definitions (CRDs):**
```
Training Job CRD:
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trainingjobs.ml.io
spec:
  group: ml.io
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              framework:
                type: string
                enum: ["tensorflow", "pytorch", "xgboost"]
              replicas:
                type: integer
                minimum: 1
                maximum: 100
              resources:
                type: object
                properties:
                  cpu:
                    type: string
                  memory:
                    type: string
                  gpu:
                    type: integer
              modelConfig:
                type: object
                properties:
                  hyperparameters:
                    type: object
                  dataPath:
                    type: string
                  outputPath:
                    type: string
              distributedStrategy:
                type: string
                enum: ["parameter_server", "mirrored", "multi_worker_mirrored"]
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Pending", "Running", "Succeeded", "Failed"]
              completionTime:
                type: string
                format: date-time
              metrics:
                type: object
  scope: Namespaced
  names:
    plural: trainingjobs
    singular: trainingjob
    kind: TrainingJob

Training Job Example:
apiVersion: ml.io/v1
kind: TrainingJob
metadata:
  name: image-classifier-training
spec:
  framework: tensorflow
  replicas: 4
  resources:
    cpu: "8"
    memory: "32Gi"
    gpu: 2
  modelConfig:
    hyperparameters:
      learning_rate: 0.001
      batch_size: 128
      epochs: 100
    dataPath: "gs://ml-datasets/imagenet"
    outputPath: "gs://ml-models/image-classifier-v1"
  distributedStrategy: "multi_worker_mirrored"

Inference Service CRD:
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: inferenceservices.ml.io
spec:
  group: ml.io
  versions:
  - name: v1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              modelUri:
                type: string
              framework:
                type: string
                enum: ["tensorflow", "pytorch", "onnx", "scikit-learn"]
              runtime:
                type: string
                enum: ["triton", "torchserve", "tensorflow-serving"]
              scaling:
                type: object
                properties:
                  minReplicas:
                    type: integer
                    minimum: 0
                  maxReplicas:
                    type: integer
                    minimum: 1
                  targetUtilization:
                    type: integer
                    minimum: 1
                    maximum: 100
              canaryTrafficPercent:
                type: integer
                minimum: 0
                maximum: 100
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Pending", "Ready", "Failed"]
              url:
                type: string
              replicas:
                type: integer
  scope: Namespaced
  names:
    plural: inferenceservices
    singular: inferenceservice
    kind: InferenceService
```

**ML Operator Implementation:**
```
Training Job Operator:
type TrainingJobReconciler struct {
    client.Client
    Scheme *runtime.Scheme
    Log    logr.Logger
}

func (r *TrainingJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := r.Log.WithValues("trainingjob", req.NamespacedName)
    
    // Fetch TrainingJob instance
    var trainingJob mlv1.TrainingJob
    if err := r.Get(ctx, req.NamespacedName, &trainingJob); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }
    
    // Reconcile based on current phase
    switch trainingJob.Status.Phase {
    case "":
        return r.reconcileCreation(ctx, &trainingJob)
    case "Pending":
        return r.reconcilePending(ctx, &trainingJob)
    case "Running":
        return r.reconcileRunning(ctx, &trainingJob)
    case "Succeeded", "Failed":
        return r.reconcileCompleted(ctx, &trainingJob)
    }
    
    return ctrl.Result{}, nil
}

func (r *TrainingJobReconciler) reconcileCreation(ctx context.Context, job *mlv1.TrainingJob) (ctrl.Result, error) {
    // Create ConfigMap for training configuration
    configMap := r.buildConfigMap(job)
    if err := r.Create(ctx, configMap); err != nil {
        return ctrl.Result{}, err
    }
    
    // Create Service for distributed training communication
    service := r.buildService(job)
    if err := r.Create(ctx, service); err != nil {
        return ctrl.Result{}, err
    }
    
    // Create Job for training execution
    kubeJob := r.buildKubernetesJob(job)
    if err := r.Create(ctx, kubeJob); err != nil {
        return ctrl.Result{}, err
    }
    
    // Update status to Pending
    job.Status.Phase = "Pending"
    return ctrl.Result{RequeueAfter: time.Second * 30}, r.Status().Update(ctx, job)
}

func (r *TrainingJobReconciler) buildKubernetesJob(trainingJob *mlv1.TrainingJob) *batchv1.Job {
    return &batchv1.Job{
        ObjectMeta: metav1.ObjectMeta{
            Name:      trainingJob.Name + "-job",
            Namespace: trainingJob.Namespace,
            OwnerReferences: []metav1.OwnerReference{
                *metav1.NewControllerRef(trainingJob, mlv1.GroupVersion.WithKind("TrainingJob")),
            },
        },
        Spec: batchv1.JobSpec{
            Parallelism: &trainingJob.Spec.Replicas,
            Completions: &trainingJob.Spec.Replicas,
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: map[string]string{
                        "app":           "training-worker",
                        "training-job":  trainingJob.Name,
                    },
                },
                Spec: corev1.PodSpec{
                    RestartPolicy: corev1.RestartPolicyNever,
                    Containers: []corev1.Container{
                        {
                            Name:  "training-container",
                            Image: r.getFrameworkImage(trainingJob.Spec.Framework),
                            Resources: corev1.ResourceRequirements{
                                Requests: corev1.ResourceList{
                                    corev1.ResourceCPU:    resource.MustParse(trainingJob.Spec.Resources.CPU),
                                    corev1.ResourceMemory: resource.MustParse(trainingJob.Spec.Resources.Memory),
                                },
                                Limits: corev1.ResourceList{
                                    corev1.ResourceCPU:    resource.MustParse(trainingJob.Spec.Resources.CPU),
                                    corev1.ResourceMemory: resource.MustParse(trainingJob.Spec.Resources.Memory),
                                },
                            },
                            Env: r.buildEnvironmentVariables(trainingJob),
                            VolumeMounts: []corev1.VolumeMount{
                                {
                                    Name:      "config",
                                    MountPath: "/config",
                                },
                                {
                                    Name:      "data",
                                    MountPath: "/data",
                                },
                                {
                                    Name:      "output",
                                    MountPath: "/output",
                                },
                            },
                        },
                    },
                    Volumes: r.buildVolumes(trainingJob),
                },
            },
        },
    }
}

Inference Service Operator:
type InferenceServiceReconciler struct {
    client.Client
    Scheme *runtime.Scheme
    Log    logr.Logger
}

func (r *InferenceServiceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    var inferenceService mlv1.InferenceService
    if err := r.Get(ctx, req.NamespacedName, &inferenceService); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }
    
    // Create or update Deployment
    deployment := r.buildDeployment(&inferenceService)
    if err := r.createOrUpdate(ctx, deployment); err != nil {
        return ctrl.Result{}, err
    }
    
    // Create or update Service
    service := r.buildService(&inferenceService)
    if err := r.createOrUpdate(ctx, service); err != nil {
        return ctrl.Result{}, err
    }
    
    // Create or update HPA
    hpa := r.buildHPA(&inferenceService)
    if err := r.createOrUpdate(ctx, hpa); err != nil {
        return ctrl.Result{}, err
    }
    
    // Update status
    if err := r.updateStatus(ctx, &inferenceService); err != nil {
        return ctrl.Result{}, err
    }
    
    return ctrl.Result{RequeueAfter: time.Minute * 5}, nil
}
```

This comprehensive framework for Kubernetes ML workloads provides the theoretical foundations and practical strategies for orchestrating complex machine learning systems at scale. The key insight is that ML workloads have unique resource, scheduling, and state management requirements that necessitate specialized Kubernetes patterns and custom operators for optimal performance and resource utilization.