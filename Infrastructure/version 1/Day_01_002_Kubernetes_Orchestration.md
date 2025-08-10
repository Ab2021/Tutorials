# Day 1.2: Kubernetes & Container Orchestration

## ⚙️ AI/ML Infrastructure Overview & Cluster Management - Part 2

**Focus**: Container Orchestration Theory, Kubernetes Architecture, ML Workload Scheduling  
**Duration**: 2-3 hours  
**Level**: Beginner to Intermediate  

---

## 🎯 Learning Objectives

- Master Kubernetes architecture principles and container orchestration for AI/ML workloads
- Learn advanced scheduling strategies and resource allocation for ML training and inference
- Understand custom resource definitions and operators for ML-specific orchestration
- Analyze multi-tenancy patterns and isolation strategies for shared ML infrastructure

---

## ⚙️ Kubernetes Orchestration Theory

### **Container Orchestration for ML Workloads**

Kubernetes orchestration for AI/ML requires specialized understanding of resource-intensive workloads, GPU scheduling, distributed training patterns, and long-running computational tasks.

**Kubernetes ML Architecture Framework:**
```
Kubernetes ML Orchestration Components:
1. Control Plane Layer:
   - API Server with ML-specific APIs
   - etcd cluster for ML metadata storage
   - Scheduler with ML-aware algorithms
   - Controller manager with ML operators

2. Worker Node Layer:
   - Kubelet with GPU device plugins
   - Container runtime optimized for ML
   - Network plugins for high-bandwidth ML communication
   - Storage plugins for large dataset access

3. ML-Specific Extensions:
   - Custom Resource Definitions (CRDs) for ML jobs
   - ML operators for training workflows
   - GPU device plugins and resource managers
   - Distributed training coordinators

4. Storage and Networking:
   - Persistent volumes for datasets and models
   - High-performance network fabrics
   - Shared file systems for distributed access
   - Container registries for ML images

Kubernetes Resource Mathematical Models:
Resource Allocation:
Total_Cluster_Resources = Σ(Node_i_Resources × Availability_Factor_i)
ML_Job_Resource_Request = CPU_Request + Memory_Request + GPU_Request + Storage_Request

Scheduling Optimization:
Optimal_Placement = arg max(Resource_Utilization × Performance_Score - Fragmentation_Cost)
Pod_Affinity_Score = Σ(Affinity_Weight_i × Affinity_Match_i)

GPU Utilization:
GPU_Efficiency = (Actual_GPU_Compute_Time / Total_GPU_Time) × GPU_Memory_Utilization
Multi_GPU_Scaling_Factor = Actual_Speedup / Theoretical_Linear_Speedup

Network Performance:
Inter_Pod_Bandwidth = Node_Network_Bandwidth / Pod_Density
Distributed_Training_Efficiency = Communication_Efficiency × Computation_Efficiency
```

**Comprehensive Kubernetes ML Orchestration System:**
```
Kubernetes ML Orchestration Implementation:
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import yaml
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import kubernetes
from kubernetes import client, config

class PodPhase(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"

class MLJobType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    DATA_PROCESSING = "data_processing"
    DISTRIBUTED_TRAINING = "distributed_training"
    BATCH_INFERENCE = "batch_inference"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "nvidia.com/gpu"
    STORAGE = "ephemeral-storage"
    CUSTOM = "custom"

@dataclass
class MLWorkload:
    workload_id: str
    workload_name: str
    job_type: MLJobType
    resource_requirements: Dict[str, str]
    environment_variables: Dict[str, str]
    command: List[str]
    arguments: List[str]
    image: str
    volumes: List[Dict[str, Any]]
    node_selector: Dict[str, str]
    tolerations: List[Dict[str, Any]]
    affinity_rules: Dict[str, Any]
    priority_class: str
    max_runtime_seconds: int
    retry_policy: Dict[str, Any]

@dataclass
class KubernetesNode:
    node_name: str
    node_labels: Dict[str, str]
    node_annotations: Dict[str, str]
    allocatable_resources: Dict[str, str]
    capacity_resources: Dict[str, str]
    conditions: List[Dict[str, Any]]
    taints: List[Dict[str, Any]]
    gpu_info: Optional[Dict[str, Any]]
    network_info: Dict[str, Any]
    storage_info: Dict[str, Any]

class KubernetesMLOrchestrator:
    def __init__(self):
        self.cluster_manager = KubernetesClusterManager()
        self.ml_scheduler = MLAwareScheduler()
        self.resource_manager = KubernetesResourceManager()
        self.workload_controller = MLWorkloadController()
        self.monitoring_system = KubernetesMLMonitoring()
        self.operator_manager = MLOperatorManager()
        self.storage_manager = KubernetesStorageManager()
    
    def setup_kubernetes_ml_orchestration(self, orchestration_config):
        """Set up comprehensive Kubernetes ML orchestration system"""
        
        orchestration_setup = {
            'setup_id': self._generate_setup_id(),
            'timestamp': datetime.utcnow(),
            'cluster_configuration': {},
            'ml_scheduling_setup': {},
            'resource_management': {},
            'workload_controllers': {},
            'ml_operators': {},
            'storage_configuration': {},
            'monitoring_setup': {},
            'multi_tenancy_config': {}
        }
        
        try:
            # Phase 1: Cluster Configuration
            logging.info("Phase 1: Configuring Kubernetes cluster for ML workloads")
            cluster_config = self.cluster_manager.configure_ml_cluster(
                cluster_config=orchestration_config.get('cluster', {}),
                ml_requirements=orchestration_config.get('ml_requirements', {})
            )
            orchestration_setup['cluster_configuration'] = cluster_config
            
            # Phase 2: ML-Aware Scheduling
            logging.info("Phase 2: Setting up ML-aware scheduling")
            scheduling_setup = self.ml_scheduler.setup_ml_scheduling(
                cluster_config=cluster_config,
                scheduling_policies=orchestration_config.get('scheduling', {})
            )
            orchestration_setup['ml_scheduling_setup'] = scheduling_setup
            
            # Phase 3: Resource Management
            logging.info("Phase 3: Configuring Kubernetes resource management")
            resource_setup = self.resource_manager.setup_resource_management(
                cluster_config=cluster_config,
                resource_policies=orchestration_config.get('resources', {})
            )
            orchestration_setup['resource_management'] = resource_setup
            
            # Phase 4: ML Workload Controllers
            logging.info("Phase 4: Setting up ML workload controllers")
            controller_setup = self.workload_controller.setup_workload_controllers(
                orchestration_setup=orchestration_setup,
                controller_config=orchestration_config.get('controllers', {})
            )
            orchestration_setup['workload_controllers'] = controller_setup
            
            # Phase 5: ML Operators
            logging.info("Phase 5: Deploying ML operators")
            operator_setup = self.operator_manager.deploy_ml_operators(
                orchestration_setup=orchestration_setup,
                operator_config=orchestration_config.get('operators', {})
            )
            orchestration_setup['ml_operators'] = operator_setup
            
            # Phase 6: Storage Configuration
            logging.info("Phase 6: Configuring ML storage systems")
            storage_setup = self.storage_manager.configure_ml_storage(
                orchestration_setup=orchestration_setup,
                storage_config=orchestration_config.get('storage', {})
            )
            orchestration_setup['storage_configuration'] = storage_setup
            
            # Phase 7: Monitoring Setup
            logging.info("Phase 7: Setting up Kubernetes ML monitoring")
            monitoring_setup = self.monitoring_system.setup_ml_monitoring(
                orchestration_setup=orchestration_setup,
                monitoring_config=orchestration_config.get('monitoring', {})
            )
            orchestration_setup['monitoring_setup'] = monitoring_setup
            
            # Phase 8: Multi-Tenancy Configuration
            logging.info("Phase 8: Configuring multi-tenancy for ML workloads")
            tenancy_setup = self._configure_ml_multi_tenancy(
                orchestration_setup, orchestration_config.get('multi_tenancy', {})
            )
            orchestration_setup['multi_tenancy_config'] = tenancy_setup
            
            logging.info("Kubernetes ML orchestration setup completed successfully")
            
            return orchestration_setup
            
        except Exception as e:
            logging.error(f"Error in Kubernetes ML orchestration setup: {str(e)}")
            orchestration_setup['error'] = str(e)
            return orchestration_setup
    
    def _generate_setup_id(self):
        """Generate unique setup identifier"""
        return f"k8s_ml_setup_{int(time.time())}_{np.random.randint(1000, 9999)}"
    
    def _configure_ml_multi_tenancy(self, orchestration_setup, tenancy_config):
        """Configure multi-tenancy for ML workloads"""
        
        tenancy_setup = {
            'namespace_isolation': {},
            'resource_quotas': {},
            'network_policies': {},
            'rbac_configuration': {},
            'pod_security_policies': {}
        }
        
        # Configure namespace isolation
        tenancy_setup['namespace_isolation'] = {
            'tenant_namespaces': tenancy_config.get('tenant_namespaces', []),
            'namespace_labels': tenancy_config.get('namespace_labels', {}),
            'namespace_annotations': tenancy_config.get('namespace_annotations', {}),
            'default_resource_limits': tenancy_config.get('default_limits', {})
        }
        
        # Configure resource quotas per tenant
        tenancy_setup['resource_quotas'] = self._configure_tenant_resource_quotas(tenancy_config)
        
        # Configure network isolation
        tenancy_setup['network_policies'] = self._configure_tenant_network_policies(tenancy_config)
        
        # Configure RBAC
        tenancy_setup['rbac_configuration'] = self._configure_tenant_rbac(tenancy_config)
        
        return tenancy_setup
    
    def _configure_tenant_resource_quotas(self, tenancy_config):
        """Configure resource quotas for each tenant"""
        
        tenant_quotas = {}
        default_quota = tenancy_config.get('default_quota', {})
        
        for tenant in tenancy_config.get('tenants', []):
            tenant_name = tenant['name']
            tenant_quota = {**default_quota, **tenant.get('custom_quota', {})}
            
            # ML-specific resource quotas
            quota_spec = {
                'requests.cpu': tenant_quota.get('cpu_requests', '100'),
                'requests.memory': tenant_quota.get('memory_requests', '200Gi'),
                'requests.nvidia.com/gpu': tenant_quota.get('gpu_requests', '10'),
                'limits.cpu': tenant_quota.get('cpu_limits', '200'),
                'limits.memory': tenant_quota.get('memory_limits', '400Gi'),
                'limits.nvidia.com/gpu': tenant_quota.get('gpu_limits', '20'),
                'persistentvolumeclaims': tenant_quota.get('pvc_count', '50'),
                'requests.storage': tenant_quota.get('storage_requests', '1Ti'),
                'pods': tenant_quota.get('pod_limit', '100'),
                'services': tenant_quota.get('service_limit', '20')
            }
            
            tenant_quotas[tenant_name] = quota_spec
        
        return tenant_quotas

class KubernetesClusterManager:
    def __init__(self):
        self.node_manager = NodeManager()
        self.network_manager = NetworkManager()
        self.security_manager = SecurityManager()
        self.addon_manager = AddonManager()
    
    def configure_ml_cluster(self, cluster_config, ml_requirements):
        """Configure Kubernetes cluster optimized for ML workloads"""
        
        cluster_setup = {
            'cluster_specification': {},
            'node_configuration': {},
            'network_configuration': {},
            'security_configuration': {},
            'ml_addons': {}
        }
        
        try:
            # Configure cluster specification
            cluster_spec = self._define_cluster_specification(cluster_config, ml_requirements)
            cluster_setup['cluster_specification'] = cluster_spec
            
            # Configure nodes for ML workloads
            node_config = self.node_manager.configure_ml_nodes(
                cluster_spec, ml_requirements
            )
            cluster_setup['node_configuration'] = node_config
            
            # Configure networking for ML
            network_config = self.network_manager.configure_ml_networking(
                cluster_spec, ml_requirements
            )
            cluster_setup['network_configuration'] = network_config
            
            # Configure security for ML workloads
            security_config = self.security_manager.configure_ml_security(
                cluster_spec, ml_requirements
            )
            cluster_setup['security_configuration'] = security_config
            
            # Install ML-specific addons
            addon_config = self.addon_manager.install_ml_addons(
                cluster_setup, ml_requirements
            )
            cluster_setup['ml_addons'] = addon_config
            
            return cluster_setup
            
        except Exception as e:
            logging.error(f"Error configuring ML cluster: {str(e)}")
            cluster_setup['error'] = str(e)
            return cluster_setup
    
    def _define_cluster_specification(self, cluster_config, ml_requirements):
        """Define comprehensive cluster specification for ML workloads"""
        
        cluster_spec = {
            'cluster_name': cluster_config.get('name', 'ml-cluster'),
            'kubernetes_version': cluster_config.get('version', '1.24.0'),
            'cluster_size': self._calculate_cluster_size(ml_requirements),
            'control_plane_config': self._design_control_plane(cluster_config, ml_requirements),
            'worker_node_config': self._design_worker_nodes(cluster_config, ml_requirements),
            'cluster_networking': self._design_cluster_networking(cluster_config),
            'cluster_dns': cluster_config.get('dns_config', {'type': 'coredns'}),
            'cluster_addons': self._select_ml_addons(ml_requirements)
        }
        
        return cluster_spec
    
    def _calculate_cluster_size(self, ml_requirements):
        """Calculate optimal cluster size for ML requirements"""
        
        # Estimate based on expected workloads
        expected_training_jobs = ml_requirements.get('concurrent_training_jobs', 10)
        expected_inference_workloads = ml_requirements.get('inference_workloads', 20)
        expected_data_processing_jobs = ml_requirements.get('data_processing_jobs', 5)
        
        # Resource requirements per job type (rough estimates)
        training_resources_per_job = {
            'cpu': 8,
            'memory_gb': 32,
            'gpu': 1
        }
        
        inference_resources_per_workload = {
            'cpu': 2,
            'memory_gb': 8,
            'gpu': 0.5
        }
        
        data_processing_resources_per_job = {
            'cpu': 4,
            'memory_gb': 16,
            'gpu': 0
        }
        
        # Calculate total resource needs
        total_cpu_needed = (
            expected_training_jobs * training_resources_per_job['cpu'] +
            expected_inference_workloads * inference_resources_per_workload['cpu'] +
            expected_data_processing_jobs * data_processing_resources_per_job['cpu']
        )
        
        total_memory_needed = (
            expected_training_jobs * training_resources_per_job['memory_gb'] +
            expected_inference_workloads * inference_resources_per_workload['memory_gb'] +
            expected_data_processing_jobs * data_processing_resources_per_job['memory_gb']
        )
        
        total_gpu_needed = (
            expected_training_jobs * training_resources_per_job['gpu'] +
            expected_inference_workloads * inference_resources_per_workload['gpu']
        )
        
        # Add overhead and safety margin (30%)
        safety_margin = 1.3
        
        cluster_size = {
            'total_cpu_cores': int(total_cpu_needed * safety_margin),
            'total_memory_gb': int(total_memory_needed * safety_margin),
            'total_gpus': int(total_gpu_needed * safety_margin),
            'estimated_node_count': self._estimate_node_count(
                total_cpu_needed * safety_margin,
                total_memory_needed * safety_margin,
                total_gpu_needed * safety_margin
            )
        }
        
        return cluster_size
    
    def _estimate_node_count(self, cpu_needed, memory_needed, gpu_needed):
        """Estimate number of nodes needed"""
        
        # Typical node configurations
        cpu_node_config = {'cpu': 32, 'memory_gb': 128, 'gpu': 0}
        gpu_node_config = {'cpu': 16, 'memory_gb': 64, 'gpu': 4}
        
        # Calculate nodes needed for each resource type
        cpu_nodes_for_cpu = int(np.ceil(cpu_needed / cpu_node_config['cpu']))
        cpu_nodes_for_memory = int(np.ceil(memory_needed / cpu_node_config['memory_gb']))
        gpu_nodes_needed = int(np.ceil(gpu_needed / gpu_node_config['gpu']))
        
        # Total nodes
        cpu_nodes = max(cpu_nodes_for_cpu, cpu_nodes_for_memory)
        total_nodes = cpu_nodes + gpu_nodes_needed
        
        return {
            'cpu_nodes': cpu_nodes,
            'gpu_nodes': gpu_nodes_needed,
            'total_nodes': total_nodes
        }
    
    def _design_control_plane(self, cluster_config, ml_requirements):
        """Design control plane configuration for ML cluster"""
        
        # Control plane sizing based on cluster scale
        expected_nodes = ml_requirements.get('expected_nodes', 10)
        expected_pods = ml_requirements.get('expected_pods', 500)
        
        if expected_nodes > 100 or expected_pods > 5000:
            control_plane_size = 'large'
        elif expected_nodes > 50 or expected_pods > 2000:
            control_plane_size = 'medium'
        else:
            control_plane_size = 'small'
        
        control_plane_configs = {
            'small': {
                'master_count': 1,
                'master_instance_type': 'medium',
                'etcd_instance_type': 'medium',
                'api_server_replicas': 1
            },
            'medium': {
                'master_count': 3,
                'master_instance_type': 'large',
                'etcd_instance_type': 'large',
                'api_server_replicas': 2
            },
            'large': {
                'master_count': 5,
                'master_instance_type': 'xlarge',
                'etcd_instance_type': 'xlarge',
                'api_server_replicas': 3
            }
        }
        
        base_config = control_plane_configs[control_plane_size]
        
        # ML-specific control plane configuration
        control_plane_config = {
            **base_config,
            'ml_specific_config': {
                'api_server_flags': [
                    '--feature-gates=GPUManager=true',
                    '--runtime-config=batch/v1=true',
                    '--max-requests-inflight=400',  # Higher for ML workloads
                    '--max-mutating-requests-inflight=200'
                ],
                'scheduler_config': {
                    'profiles': ['ml-aware-scheduler'],
                    'plugins': ['NodeResourcesFit', 'NodeAffinity', 'PodTopologySpread']
                },
                'controller_manager_flags': [
                    '--controllers=*,bootstrapsigner,tokencleaner',
                    '--node-monitor-period=5s',  # Faster node monitoring for ML
                    '--pod-eviction-timeout=30s'  # Faster eviction for failed ML jobs
                ]
            }
        }
        
        return control_plane_config
    
    def _design_worker_nodes(self, cluster_config, ml_requirements):
        """Design worker node configuration for ML workloads"""
        
        node_pools = []
        
        # CPU-optimized node pool for general ML workloads
        cpu_pool = {
            'pool_name': 'cpu-optimized',
            'instance_type': 'cpu-optimized-large',
            'min_nodes': cluster_config.get('min_cpu_nodes', 2),
            'max_nodes': cluster_config.get('max_cpu_nodes', 20),
            'disk_size_gb': 500,
            'node_labels': {
                'node-type': 'cpu-optimized',
                'workload-type': 'general-ml'
            },
            'node_taints': [],
            'kubelet_config': {
                'max-pods': 110,
                'cpu-manager-policy': 'static',
                'memory-manager-policy': 'Static',
                'reserved-cpus': '1',
                'system-reserved': 'cpu=500m,memory=1Gi'
            }
        }
        
        # GPU-optimized node pool for training workloads
        if ml_requirements.get('gpu_required', True):
            gpu_pool = {
                'pool_name': 'gpu-optimized',
                'instance_type': 'gpu-optimized-large',
                'min_nodes': cluster_config.get('min_gpu_nodes', 1),
                'max_nodes': cluster_config.get('max_gpu_nodes', 10),
                'disk_size_gb': 1000,
                'gpu_type': ml_requirements.get('gpu_type', 'nvidia-v100'),
                'gpu_count': ml_requirements.get('gpus_per_node', 4),
                'node_labels': {
                    'node-type': 'gpu-optimized',
                    'workload-type': 'training',
                    'gpu-type': ml_requirements.get('gpu_type', 'nvidia-v100')
                },
                'node_taints': [{
                    'key': 'nvidia.com/gpu',
                    'value': 'true',
                    'effect': 'NoSchedule'
                }],
                'kubelet_config': {
                    'max-pods': 50,  # Lower for GPU nodes
                    'cpu-manager-policy': 'static',
                    'memory-manager-policy': 'Static',
                    'reserved-cpus': '2',
                    'system-reserved': 'cpu=1000m,memory=2Gi'
                }
            }
            node_pools.append(gpu_pool)
        
        # Memory-optimized node pool for large dataset processing
        if ml_requirements.get('memory_intensive_workloads', False):
            memory_pool = {
                'pool_name': 'memory-optimized',
                'instance_type': 'memory-optimized-large',
                'min_nodes': cluster_config.get('min_memory_nodes', 1),
                'max_nodes': cluster_config.get('max_memory_nodes', 5),
                'disk_size_gb': 2000,
                'node_labels': {
                    'node-type': 'memory-optimized',
                    'workload-type': 'data-processing'
                },
                'node_taints': [{
                    'key': 'memory-optimized',
                    'value': 'true',
                    'effect': 'NoSchedule'
                }],
                'kubelet_config': {
                    'max-pods': 30,  # Lower for memory-intensive workloads
                    'memory-manager-policy': 'Static',
                    'reserved-memory': '4Gi',
                    'system-reserved': 'memory=4Gi'
                }
            }
            node_pools.append(memory_pool)
        
        node_pools.append(cpu_pool)
        
        return {
            'node_pools': node_pools,
            'auto_scaling_config': {
                'enabled': True,
                'scale_down_delay_after_add': '10m',
                'scale_down_unneeded_time': '10m',
                'scale_down_utilization_threshold': 0.5,
                'max_node_provision_time': '15m'
            }
        }

class MLAwareScheduler:
    def __init__(self):
        self.scheduling_algorithms = {
            'gpu_aware': GPUAwareScheduling(),
            'resource_affinity': ResourceAffinityScheduling(),
            'workload_aware': WorkloadAwareScheduling(),
            'multi_tenant': MultiTenantScheduling()
        }
        self.scheduler_extender = SchedulerExtender()
        self.priority_classes = PriorityClassManager()
    
    def setup_ml_scheduling(self, cluster_config, scheduling_policies):
        """Set up ML-aware scheduling system"""
        
        scheduling_setup = {
            'scheduler_configuration': {},
            'scheduling_policies': {},
            'priority_classes': {},
            'scheduler_extenders': {},
            'custom_schedulers': {}
        }
        
        try:
            # Configure main scheduler for ML workloads
            scheduler_config = self._configure_ml_scheduler(cluster_config, scheduling_policies)
            scheduling_setup['scheduler_configuration'] = scheduler_config
            
            # Set up ML scheduling policies
            policies = self._setup_ml_scheduling_policies(scheduling_policies)
            scheduling_setup['scheduling_policies'] = policies
            
            # Configure priority classes for ML workloads
            priority_classes = self.priority_classes.setup_ml_priority_classes(scheduling_policies)
            scheduling_setup['priority_classes'] = priority_classes
            
            # Set up scheduler extenders
            extenders = self.scheduler_extender.setup_scheduler_extenders(
                cluster_config, scheduling_policies
            )
            scheduling_setup['scheduler_extenders'] = extenders
            
            # Configure custom schedulers if needed
            custom_schedulers = self._setup_custom_schedulers(scheduling_policies)
            scheduling_setup['custom_schedulers'] = custom_schedulers
            
            return scheduling_setup
            
        except Exception as e:
            logging.error(f"Error setting up ML scheduling: {str(e)}")
            scheduling_setup['error'] = str(e)
            return scheduling_setup
    
    def _configure_ml_scheduler(self, cluster_config, scheduling_policies):
        """Configure main Kubernetes scheduler for ML workloads"""
        
        # ML-optimized scheduler configuration
        scheduler_config = {
            'scheduler_name': 'ml-aware-scheduler',
            'scheduler_config_map': {
                'apiVersion': 'kubescheduler.config.k8s.io/v1beta3',
                'kind': 'KubeSchedulerConfiguration',
                'profiles': [{
                    'schedulerName': 'ml-aware-scheduler',
                    'plugins': {
                        'preFilter': {
                            'enabled': [
                                {'name': 'NodeResourcesFit'},
                                {'name': 'NodeAffinity'},
                                {'name': 'PodTopologySpread'},
                                {'name': 'VolumeBinding'}
                            ]
                        },
                        'filter': {
                            'enabled': [
                                {'name': 'NodeUnschedulable'},
                                {'name': 'NodeName'},
                                {'name': 'TaintToleration'},
                                {'name': 'NodeAffinity'},
                                {'name': 'NodePorts'},
                                {'name': 'NodeResourcesFit'},
                                {'name': 'VolumeRestrictions'},
                                {'name': 'EBSLimits'},
                                {'name': 'GCEPDLimits'},
                                {'name': 'NodeVolumeLimits'},
                                {'name': 'AzureDiskLimits'},
                                {'name': 'VolumeBinding'},
                                {'name': 'VolumeZone'},
                                {'name': 'PodTopologySpread'},
                                {'name': 'InterPodAffinity'}
                            ]
                        },
                        'score': {
                            'enabled': [
                                {'name': 'NodeResourcesFit', 'weight': 10},
                                {'name': 'NodeAffinity', 'weight': 5},
                                {'name': 'PodTopologySpread', 'weight': 3},
                                {'name': 'InterPodAffinity', 'weight': 2},
                                {'name': 'ImageLocality', 'weight': 1},
                                {'name': 'TaintToleration', 'weight': 1},
                                {'name': 'NodePreferAvoidPods', 'weight': 10000},
                                {'name': 'VolumeBinding', 'weight': 1}
                            ]
                        }
                    },
                    'pluginConfig': [{
                        'name': 'NodeResourcesFit',
                        'args': {
                            'apiVersion': 'kubescheduler.config.k8s.io/v1beta3',
                            'kind': 'NodeResourcesFitArgs',
                            'scoringStrategy': {
                                'type': 'LeastAllocated',  # Better for ML workloads
                                'resources': [
                                    {'name': 'cpu', 'weight': 1},
                                    {'name': 'memory', 'weight': 1},
                                    {'name': 'nvidia.com/gpu', 'weight': 5}  # Higher weight for GPU
                                ]
                            }
                        }
                    }]
                }]
            },
            'ml_specific_features': {
                'gpu_scheduling': True,
                'resource_quotas': True,
                'priority_classes': True,
                'pod_disruption_budgets': True,
                'node_affinity_optimization': True
            }
        }
        
        return scheduler_config
    
    def _setup_ml_scheduling_policies(self, scheduling_policies):
        """Set up ML-specific scheduling policies"""
        
        policies = {
            'gpu_scheduling_policy': {
                'gpu_sharing_enabled': scheduling_policies.get('gpu_sharing', True),
                'gpu_memory_tracking': True,
                'gpu_isolation': scheduling_policies.get('gpu_isolation', 'process'),
                'gpu_scheduling_algorithm': 'bin_packing'  # or 'spread'
            },
            'resource_allocation_policy': {
                'overcommit_ratio': {
                    'cpu': scheduling_policies.get('cpu_overcommit_ratio', 1.0),
                    'memory': scheduling_policies.get('memory_overcommit_ratio', 1.0)
                },
                'resource_limits_enforcement': True,
                'resource_requests_required': True
            },
            'workload_isolation_policy': {
                'namespace_isolation': True,
                'network_policies_enabled': True,
                'pod_security_policies_enabled': True,
                'resource_quotas_per_tenant': True
            },
            'performance_optimization_policy': {
                'cpu_topology_awareness': True,
                'numa_topology_awareness': scheduling_policies.get('numa_awareness', True),
                'cache_locality_optimization': True,
                'network_bandwidth_awareness': True
            }
        }
        
        return policies

class KubernetesResourceManager:
    def __init__(self):
        self.quota_manager = ResourceQuotaManager()
        self.limit_range_manager = LimitRangeManager()
        self.device_plugin_manager = DevicePluginManager()
        self.custom_resource_manager = CustomResourceManager()
    
    def setup_resource_management(self, cluster_config, resource_policies):
        """Set up comprehensive Kubernetes resource management"""
        
        resource_setup = {
            'resource_quotas': {},
            'limit_ranges': {},
            'device_plugins': {},
            'custom_resources': {},
            'resource_monitoring': {}
        }
        
        try:
            # Set up resource quotas
            quota_config = self.quota_manager.setup_resource_quotas(
                cluster_config, resource_policies.get('quotas', {})
            )
            resource_setup['resource_quotas'] = quota_config
            
            # Configure limit ranges
            limit_ranges = self.limit_range_manager.setup_limit_ranges(
                cluster_config, resource_policies.get('limits', {})
            )
            resource_setup['limit_ranges'] = limit_ranges
            
            # Set up device plugins (GPU, etc.)
            device_plugins = self.device_plugin_manager.setup_device_plugins(
                cluster_config, resource_policies.get('devices', {})
            )
            resource_setup['device_plugins'] = device_plugins
            
            # Configure custom resources for ML
            custom_resources = self.custom_resource_manager.setup_ml_custom_resources(
                cluster_config, resource_policies.get('custom_resources', {})
            )
            resource_setup['custom_resources'] = custom_resources
            
            # Set up resource monitoring
            monitoring_config = self._setup_resource_monitoring(
                resource_setup, resource_policies
            )
            resource_setup['resource_monitoring'] = monitoring_config
            
            return resource_setup
            
        except Exception as e:
            logging.error(f"Error setting up resource management: {str(e)}")
            resource_setup['error'] = str(e)
            return resource_setup
    
    def _setup_resource_monitoring(self, resource_setup, resource_policies):
        """Set up resource monitoring and alerting"""
        
        monitoring_config = {
            'metrics_collection': {
                'enabled': True,
                'collection_interval': '30s',
                'metrics_retention': '7d',
                'custom_metrics': [
                    'gpu_utilization',
                    'gpu_memory_usage',
                    'training_job_progress',
                    'inference_request_rate'
                ]
            },
            'alerting_rules': {
                'high_resource_utilization': {
                    'threshold': 0.9,
                    'duration': '5m',
                    'severity': 'warning'
                },
                'gpu_memory_exhaustion': {
                    'threshold': 0.95,
                    'duration': '2m',
                    'severity': 'critical'
                },
                'pod_oom_kills': {
                    'threshold': 1,
                    'duration': '1m',
                    'severity': 'critical'
                }
            },
            'resource_optimization': {
                'right_sizing_enabled': True,
                'recommendations_generated': True,
                'auto_scaling_based_on_metrics': True
            }
        }
        
        return monitoring_config
```

This comprehensive Kubernetes orchestration framework provides detailed theoretical foundations and practical implementation strategies for managing AI/ML workloads with specialized scheduling, resource management, and container orchestration patterns optimized for machine learning infrastructure.