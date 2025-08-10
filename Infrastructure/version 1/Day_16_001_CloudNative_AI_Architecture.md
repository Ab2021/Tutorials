# Day 16.1: Cloud-Native AI & Microservices Architecture

## ☁️ Advanced AI Infrastructure Specializations - Part 3

**Focus**: Microservices for ML, Service Mesh, Container Orchestration  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master cloud-native AI architecture design with microservices patterns
- Learn service mesh implementation for ML workloads and inter-service communication
- Understand container orchestration strategies for AI/ML applications
- Analyze API gateway patterns, service discovery, and distributed ML system design

---

## ☁️ Cloud-Native AI Architecture Theory

### **Microservices ML System Design**

Cloud-native AI architecture requires decomposing monolithic ML systems into loosely coupled, independently deployable services that can scale, evolve, and maintain reliability in distributed environments.

**Cloud-Native AI Framework:**
```
Cloud-Native AI Architecture Components:
1. Microservices Layer:
   - ML model services
   - Feature engineering services
   - Data processing services
   - Model management services

2. Service Mesh Layer:
   - Inter-service communication
   - Load balancing and circuit breaking
   - Security and encryption
   - Observability and tracing

3. Container Orchestration Layer:
   - Container lifecycle management
   - Resource scheduling and allocation
   - Auto-scaling and health management
   - Network and storage orchestration

4. API Gateway Layer:
   - Request routing and aggregation
   - Authentication and authorization
   - Rate limiting and throttling
   - API versioning and documentation

Cloud-Native Architecture Mathematical Models:
Service Decomposition:
Service_Boundaries = f(Domain_Cohesion, Coupling_Minimization, Team_Structure)
Optimal_Service_Size = arg min(Communication_Overhead + Coordination_Complexity)

Load Distribution:
Request_Distribution = Σ(Service_i_Capacity × Availability_i × Performance_Weight_i)
Optimal_Routing = arg max(Throughput × Reliability - Latency_Penalty)

Resource Allocation:
Container_Resources = f(Workload_Characteristics, SLA_Requirements, Cost_Constraints)
Cluster_Utilization = (Allocated_Resources / Total_Resources) × Efficiency_Factor

Service Mesh Performance:
Mesh_Overhead = Network_Latency + Proxy_Processing + Security_Overhead
Total_Latency = Service_Latency + Mesh_Overhead + Queue_Time
```

**Comprehensive Cloud-Native AI System:**
```
Cloud-Native AI Infrastructure Implementation:
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import asyncio
import json
import uuid
import time
from datetime import datetime
from collections import defaultdict
import yaml

class ServiceType(Enum):
    MODEL_SERVING = "model_serving"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    MODEL_MANAGEMENT = "model_management"
    INFERENCE_PIPELINE = "inference_pipeline"
    API_GATEWAY = "api_gateway"

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    A_B_TESTING = "a_b_testing"

class ScalingStrategy(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    AUTO_SCALING = "auto_scaling"
    PREDICTIVE_SCALING = "predictive_scaling"

@dataclass
class MLMicroservice:
    service_id: str
    service_name: str
    service_type: ServiceType
    version: str
    image: str
    ports: List[int]
    environment_variables: Dict[str, str]
    resource_requirements: Dict[str, Any]
    health_check_config: Dict[str, Any]
    scaling_config: Dict[str, Any]
    deployment_strategy: DeploymentStrategy
    dependencies: List[str]
    api_specification: Dict[str, Any]

@dataclass
class ServiceMeshConfig:
    mesh_name: str
    proxy_type: str
    tls_mode: str
    traffic_policies: Dict[str, Any]
    security_policies: Dict[str, Any]
    observability_config: Dict[str, Any]
    load_balancing_config: Dict[str, Any]

class CloudNativeAIInfrastructure:
    def __init__(self):
        self.microservice_manager = MLMicroserviceManager()
        self.service_mesh = ServiceMeshManager()
        self.container_orchestrator = ContainerOrchestrator()
        self.api_gateway = MLAPIGateway()
        self.service_discovery = ServiceDiscoverySystem()
        self.config_manager = ConfigurationManager()
        self.monitoring_system = CloudNativeMonitoring()
    
    def deploy_cloudnative_ai_platform(self, deployment_config):
        """Deploy comprehensive cloud-native AI platform"""
        
        deployment_result = {
            'deployment_id': self._generate_deployment_id(),
            'timestamp': datetime.utcnow(),
            'microservices_deployment': {},
            'service_mesh_setup': {},
            'container_orchestration': {},
            'api_gateway_configuration': {},
            'service_discovery_setup': {},
            'configuration_management': {},
            'monitoring_setup': {},
            'performance_benchmarks': {}
        }
        
        try:
            # Phase 1: Microservices Deployment
            logging.info("Phase 1: Deploying ML microservices")
            microservices_result = self.microservice_manager.deploy_microservices(
                services=deployment_config.get('services', []),
                deployment_config=deployment_config.get('deployment', {})
            )
            deployment_result['microservices_deployment'] = microservices_result
            
            # Phase 2: Service Mesh Setup
            logging.info("Phase 2: Setting up service mesh")
            mesh_result = self.service_mesh.setup_service_mesh(
                services=microservices_result['deployed_services'],
                mesh_config=deployment_config.get('service_mesh', {})
            )
            deployment_result['service_mesh_setup'] = mesh_result
            
            # Phase 3: Container Orchestration
            logging.info("Phase 3: Configuring container orchestration")
            orchestration_result = self.container_orchestrator.setup_orchestration(
                services=microservices_result['deployed_services'],
                orchestration_config=deployment_config.get('orchestration', {})
            )
            deployment_result['container_orchestration'] = orchestration_result
            
            # Phase 4: API Gateway Configuration
            logging.info("Phase 4: Configuring API gateway")
            gateway_result = self.api_gateway.setup_api_gateway(
                services=microservices_result['deployed_services'],
                gateway_config=deployment_config.get('api_gateway', {})
            )
            deployment_result['api_gateway_configuration'] = gateway_result
            
            # Phase 5: Service Discovery Setup
            logging.info("Phase 5: Setting up service discovery")
            discovery_result = self.service_discovery.setup_service_discovery(
                services=microservices_result['deployed_services'],
                discovery_config=deployment_config.get('service_discovery', {})
            )
            deployment_result['service_discovery_setup'] = discovery_result
            
            # Phase 6: Configuration Management
            logging.info("Phase 6: Setting up configuration management")
            config_result = self.config_manager.setup_configuration_management(
                deployment_result,
                config_management=deployment_config.get('configuration', {})
            )
            deployment_result['configuration_management'] = config_result
            
            # Phase 7: Monitoring and Observability
            logging.info("Phase 7: Setting up cloud-native monitoring")
            monitoring_result = self.monitoring_system.setup_monitoring(
                deployment_result,
                monitoring_config=deployment_config.get('monitoring', {})
            )
            deployment_result['monitoring_setup'] = monitoring_result
            
            # Phase 8: Performance Benchmarking
            logging.info("Phase 8: Running cloud-native performance benchmarks")
            benchmarks = self._run_cloudnative_benchmarks(deployment_result)
            deployment_result['performance_benchmarks'] = benchmarks
            
            logging.info("Cloud-native AI platform deployment completed successfully")
            
            return deployment_result
            
        except Exception as e:
            logging.error(f"Error in cloud-native AI platform deployment: {str(e)}")
            deployment_result['error'] = str(e)
            return deployment_result
    
    def _run_cloudnative_benchmarks(self, deployment_result):
        """Run comprehensive cloud-native performance benchmarks"""
        
        benchmarks = {
            'service_performance': {},
            'service_mesh_overhead': {},
            'container_efficiency': {},
            'api_gateway_throughput': {},
            'end_to_end_latency': {}
        }
        
        # Service performance benchmarks
        services = deployment_result['microservices_deployment']['deployed_services']
        for service in services:
            benchmarks['service_performance'][service.service_id] = self._benchmark_service_performance(service)
        
        # Service mesh overhead
        mesh_config = deployment_result['service_mesh_setup']
        benchmarks['service_mesh_overhead'] = self._benchmark_mesh_overhead(mesh_config)
        
        # Container efficiency
        orchestration = deployment_result['container_orchestration']
        benchmarks['container_efficiency'] = self._benchmark_container_efficiency(orchestration)
        
        return benchmarks
    
    def _benchmark_service_performance(self, service):
        """Benchmark individual service performance"""
        
        return {
            'average_response_time_ms': np.random.uniform(5, 50),
            'throughput_rps': np.random.uniform(100, 1000),
            'cpu_utilization_percent': np.random.uniform(20, 80),
            'memory_utilization_percent': np.random.uniform(30, 70),
            'error_rate_percent': np.random.uniform(0, 2)
        }
    
    def _benchmark_mesh_overhead(self, mesh_config):
        """Benchmark service mesh overhead"""
        
        return {
            'proxy_latency_overhead_ms': np.random.uniform(0.5, 2.0),
            'encryption_overhead_ms': np.random.uniform(0.1, 0.5),
            'load_balancing_overhead_ms': np.random.uniform(0.1, 0.3),
            'total_mesh_overhead_ms': np.random.uniform(1.0, 3.0)
        }

class MLMicroserviceManager:
    def __init__(self):
        self.service_registry = {}
        self.deployment_strategies = {
            DeploymentStrategy.BLUE_GREEN: BlueGreenDeployment(),
            DeploymentStrategy.CANARY: CanaryDeployment(),
            DeploymentStrategy.ROLLING_UPDATE: RollingUpdateDeployment(),
            DeploymentStrategy.A_B_TESTING: ABTestingDeployment()
        }
        self.scaling_manager = AutoScalingManager()
    
    def deploy_microservices(self, services, deployment_config):
        """Deploy ML microservices with specified strategies"""
        
        deployment_result = {
            'deployed_services': [],
            'deployment_strategies': {},
            'service_dependencies': {},
            'scaling_configurations': {},
            'health_checks': {}
        }
        
        try:
            # Parse service configurations
            microservices = []
            for service_config in services:
                microservice = self._create_microservice(service_config)
                microservices.append(microservice)
                
                # Register service
                self.service_registry[microservice.service_id] = microservice
            
            # Deploy services in dependency order
            deployment_order = self._calculate_deployment_order(microservices)
            
            for service in deployment_order:
                # Deploy individual service
                deployment_strategy = self.deployment_strategies.get(service.deployment_strategy)
                if deployment_strategy:
                    deploy_result = deployment_strategy.deploy(service, deployment_config)
                    deployment_result['deployment_strategies'][service.service_id] = deploy_result
                
                # Configure scaling
                scaling_config = self.scaling_manager.configure_scaling(service, deployment_config)
                deployment_result['scaling_configurations'][service.service_id] = scaling_config
                
                # Set up health checks
                health_check = self._setup_health_check(service)
                deployment_result['health_checks'][service.service_id] = health_check
                
                deployment_result['deployed_services'].append(service)
            
            # Set up service dependencies
            dependencies = self._setup_service_dependencies(microservices)
            deployment_result['service_dependencies'] = dependencies
            
            return deployment_result
            
        except Exception as e:
            logging.error(f"Error deploying microservices: {str(e)}")
            deployment_result['error'] = str(e)
            return deployment_result
    
    def _create_microservice(self, service_config):
        """Create microservice from configuration"""
        
        microservice = MLMicroservice(
            service_id=service_config['id'],
            service_name=service_config['name'],
            service_type=ServiceType(service_config.get('type', 'model_serving')),
            version=service_config.get('version', '1.0.0'),
            image=service_config['image'],
            ports=service_config.get('ports', [8080]),
            environment_variables=service_config.get('env', {}),
            resource_requirements=service_config.get('resources', {}),
            health_check_config=service_config.get('health_check', {}),
            scaling_config=service_config.get('scaling', {}),
            deployment_strategy=DeploymentStrategy(
                service_config.get('deployment_strategy', 'rolling_update')
            ),
            dependencies=service_config.get('dependencies', []),
            api_specification=service_config.get('api_spec', {})
        )
        
        return microservice
    
    def _calculate_deployment_order(self, services):
        """Calculate deployment order based on dependencies"""
        
        # Build dependency graph
        dependency_graph = {}
        for service in services:
            dependency_graph[service.service_id] = service.dependencies
        
        # Topological sort for deployment order
        deployment_order = []
        visited = set()
        temp_visited = set()
        
        def visit(service_id):
            if service_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_id}")
            
            if service_id not in visited:
                temp_visited.add(service_id)
                
                # Visit dependencies first
                for dep in dependency_graph.get(service_id, []):
                    visit(dep)
                
                temp_visited.remove(service_id)
                visited.add(service_id)
                
                # Find service object and add to deployment order
                service = next((s for s in services if s.service_id == service_id), None)
                if service:
                    deployment_order.append(service)
        
        # Visit all services
        for service in services:
            if service.service_id not in visited:
                visit(service.service_id)
        
        return deployment_order
    
    def _setup_health_check(self, service):
        """Set up health check configuration for service"""
        
        health_check_config = service.health_check_config
        
        default_config = {
            'endpoint': '/health',
            'interval_seconds': 30,
            'timeout_seconds': 5,
            'failure_threshold': 3,
            'success_threshold': 1,
            'initial_delay_seconds': 10
        }
        
        # Merge with service-specific config
        final_config = {**default_config, **health_check_config}
        
        # Add service-specific health checks
        if service.service_type == ServiceType.MODEL_SERVING:
            final_config['model_health_endpoint'] = '/model/health'
            final_config['model_warmup_endpoint'] = '/model/warmup'
        
        return final_config

class ServiceMeshManager:
    def __init__(self):
        self.mesh_providers = {
            'istio': IstioServiceMesh(),
            'linkerd': LinkerdServiceMesh(),
            'consul_connect': ConsulConnectServiceMesh()
        }
        self.traffic_manager = TrafficManager()
        self.security_manager = MeshSecurityManager()
    
    def setup_service_mesh(self, services, mesh_config):
        """Set up service mesh for ML microservices"""
        
        mesh_setup = {
            'mesh_configuration': {},
            'traffic_policies': {},
            'security_policies': {},
            'observability_setup': {},
            'sidecar_configurations': {}
        }
        
        try:
            # Select and configure mesh provider
            mesh_provider = mesh_config.get('provider', 'istio')
            mesh = self.mesh_providers.get(mesh_provider)
            
            if not mesh:
                raise ValueError(f"Unsupported service mesh provider: {mesh_provider}")
            
            # Configure mesh
            mesh_configuration = mesh.configure_mesh(services, mesh_config)
            mesh_setup['mesh_configuration'] = mesh_configuration
            
            # Set up traffic management
            traffic_policies = self.traffic_manager.setup_traffic_management(
                services, mesh_config.get('traffic', {})
            )
            mesh_setup['traffic_policies'] = traffic_policies
            
            # Configure security
            security_policies = self.security_manager.setup_mesh_security(
                services, mesh_config.get('security', {})
            )
            mesh_setup['security_policies'] = security_policies
            
            # Set up observability
            observability_setup = self._setup_mesh_observability(services, mesh_config)
            mesh_setup['observability_setup'] = observability_setup
            
            # Configure sidecars for each service
            sidecar_configs = {}
            for service in services:
                sidecar_config = self._configure_sidecar(service, mesh_config)
                sidecar_configs[service.service_id] = sidecar_config
            
            mesh_setup['sidecar_configurations'] = sidecar_configs
            
            return mesh_setup
            
        except Exception as e:
            logging.error(f"Error setting up service mesh: {str(e)}")
            mesh_setup['error'] = str(e)
            return mesh_setup
    
    def _setup_mesh_observability(self, services, mesh_config):
        """Set up service mesh observability"""
        
        observability_config = {
            'metrics_collection': {
                'enabled': True,
                'metrics': [
                    'request_duration',
                    'request_count',
                    'error_rate',
                    'connection_count',
                    'bytes_sent',
                    'bytes_received'
                ],
                'collection_interval_seconds': mesh_config.get('metrics_interval', 30)
            },
            'distributed_tracing': {
                'enabled': mesh_config.get('enable_tracing', True),
                'sampling_rate': mesh_config.get('trace_sampling_rate', 0.01),
                'trace_backend': mesh_config.get('trace_backend', 'jaeger')
            },
            'access_logging': {
                'enabled': mesh_config.get('enable_access_logs', True),
                'format': mesh_config.get('access_log_format', 'json'),
                'include_request_body': False,
                'include_response_body': False
            }
        }
        
        return observability_config
    
    def _configure_sidecar(self, service, mesh_config):
        """Configure sidecar proxy for individual service"""
        
        sidecar_config = {
            'service_id': service.service_id,
            'proxy_image': mesh_config.get('proxy_image', 'envoyproxy/envoy:v1.20.0'),
            'resource_limits': {
                'cpu': mesh_config.get('sidecar_cpu_limit', '100m'),
                'memory': mesh_config.get('sidecar_memory_limit', '128Mi')
            },
            'resource_requests': {
                'cpu': mesh_config.get('sidecar_cpu_request', '10m'),
                'memory': mesh_config.get('sidecar_memory_request', '64Mi')
            },
            'ports': {
                'admin_port': 15000,
                'inbound_port': 15006,
                'outbound_port': 15001
            },
            'configuration': {
                'connect_timeout': mesh_config.get('connect_timeout', '10s'),
                'request_timeout': mesh_config.get('request_timeout', '15s'),
                'retry_policy': {
                    'retry_on': '5xx,reset,connection-failure',
                    'num_retries': mesh_config.get('num_retries', 3),
                    'per_try_timeout': mesh_config.get('per_try_timeout', '5s')
                }
            }
        }
        
        # Service-specific sidecar configuration
        if service.service_type == ServiceType.MODEL_SERVING:
            sidecar_config['configuration']['idle_timeout'] = '300s'  # Longer for model serving
            sidecar_config['configuration']['request_headers_timeout'] = '30s'
        
        return sidecar_config

class ContainerOrchestrator:
    def __init__(self):
        self.orchestrators = {
            'kubernetes': KubernetesOrchestrator(),
            'docker_swarm': DockerSwarmOrchestrator(),
            'nomad': NomadOrchestrator()
        }
        self.resource_manager = ResourceManager()
        self.scheduling_optimizer = SchedulingOptimizer()
    
    def setup_orchestration(self, services, orchestration_config):
        """Set up container orchestration for ML services"""
        
        orchestration_setup = {
            'cluster_configuration': {},
            'resource_allocation': {},
            'scheduling_policies': {},
            'auto_scaling_rules': {},
            'networking_setup': {},
            'storage_configuration': {}
        }
        
        try:
            # Select orchestrator
            orchestrator_type = orchestration_config.get('type', 'kubernetes')
            orchestrator = self.orchestrators.get(orchestrator_type)
            
            if not orchestrator:
                raise ValueError(f"Unsupported orchestrator: {orchestrator_type}")
            
            # Configure cluster
            cluster_config = orchestrator.configure_cluster(services, orchestration_config)
            orchestration_setup['cluster_configuration'] = cluster_config
            
            # Allocate resources
            resource_allocation = self.resource_manager.allocate_resources(
                services, orchestration_config.get('resources', {})
            )
            orchestration_setup['resource_allocation'] = resource_allocation
            
            # Configure scheduling
            scheduling_policies = self.scheduling_optimizer.configure_scheduling(
                services, orchestration_config.get('scheduling', {})
            )
            orchestration_setup['scheduling_policies'] = scheduling_policies
            
            # Set up auto-scaling
            auto_scaling_rules = self._setup_auto_scaling(services, orchestration_config)
            orchestration_setup['auto_scaling_rules'] = auto_scaling_rules
            
            # Configure networking
            networking_setup = self._setup_networking(services, orchestration_config)
            orchestration_setup['networking_setup'] = networking_setup
            
            # Configure storage
            storage_config = self._setup_storage(services, orchestration_config)
            orchestration_setup['storage_configuration'] = storage_config
            
            return orchestration_setup
            
        except Exception as e:
            logging.error(f"Error setting up container orchestration: {str(e)}")
            orchestration_setup['error'] = str(e)
            return orchestration_setup
    
    def _setup_auto_scaling(self, services, config):
        """Set up auto-scaling rules for services"""
        
        auto_scaling_rules = {}
        
        for service in services:
            scaling_config = service.scaling_config
            
            # Default scaling configuration
            default_scaling = {
                'enabled': True,
                'min_replicas': 1,
                'max_replicas': 10,
                'target_cpu_utilization': 70,
                'target_memory_utilization': 80,
                'scale_up_cooldown_seconds': 300,
                'scale_down_cooldown_seconds': 600
            }
            
            # Merge with service-specific configuration
            final_scaling = {**default_scaling, **scaling_config}
            
            # Service-type specific adjustments
            if service.service_type == ServiceType.MODEL_SERVING:
                final_scaling.update({
                    'target_cpu_utilization': 60,  # Lower threshold for model serving
                    'custom_metrics': [
                        {
                            'name': 'inference_queue_length',
                            'target_value': 10
                        },
                        {
                            'name': 'average_response_time_ms',
                            'target_value': 100
                        }
                    ]
                })
            elif service.service_type == ServiceType.DATA_PROCESSING:
                final_scaling.update({
                    'max_replicas': 20,  # Allow more replicas for data processing
                    'custom_metrics': [
                        {
                            'name': 'processing_queue_length',
                            'target_value': 100
                        }
                    ]
                })
            
            auto_scaling_rules[service.service_id] = final_scaling
        
        return auto_scaling_rules
    
    def _setup_networking(self, services, config):
        """Set up container networking configuration"""
        
        networking_config = {
            'network_policies': {},
            'load_balancing': {},
            'ingress_configuration': {},
            'dns_configuration': {}
        }
        
        # Network policies
        for service in services:
            network_policy = {
                'service_id': service.service_id,
                'ingress_rules': self._create_ingress_rules(service, services),
                'egress_rules': self._create_egress_rules(service, services),
                'network_segmentation': config.get('enable_network_segmentation', True)
            }
            networking_config['network_policies'][service.service_id] = network_policy
        
        # Load balancing configuration
        for service in services:
            if service.ports:
                lb_config = {
                    'service_id': service.service_id,
                    'algorithm': self._select_lb_algorithm(service),
                    'health_check': service.health_check_config,
                    'session_affinity': self._determine_session_affinity(service),
                    'timeout_seconds': config.get('lb_timeout', 30)
                }
                networking_config['load_balancing'][service.service_id] = lb_config
        
        # Ingress configuration
        ingress_config = self._setup_ingress_configuration(services, config)
        networking_config['ingress_configuration'] = ingress_config
        
        # DNS configuration
        dns_config = {
            'cluster_domain': config.get('cluster_domain', 'cluster.local'),
            'dns_policy': config.get('dns_policy', 'ClusterFirst'),
            'search_domains': config.get('search_domains', [])
        }
        networking_config['dns_configuration'] = dns_config
        
        return networking_config
    
    def _create_ingress_rules(self, service, all_services):
        """Create network ingress rules for service"""
        
        ingress_rules = []
        
        # Allow traffic from dependent services
        for other_service in all_services:
            if service.service_id in other_service.dependencies:
                ingress_rules.append({
                    'from_service': other_service.service_id,
                    'ports': service.ports,
                    'protocol': 'TCP'
                })
        
        # Allow traffic from API gateway
        ingress_rules.append({
            'from_service': 'api-gateway',
            'ports': service.ports,
            'protocol': 'TCP'
        })
        
        # Allow health check traffic
        ingress_rules.append({
            'from_namespace': 'monitoring',
            'ports': service.ports,
            'protocol': 'TCP',
            'purpose': 'health_check'
        })
        
        return ingress_rules
    
    def _create_egress_rules(self, service, all_services):
        """Create network egress rules for service"""
        
        egress_rules = []
        
        # Allow traffic to dependencies
        for dep_service_id in service.dependencies:
            dep_service = next((s for s in all_services if s.service_id == dep_service_id), None)
            if dep_service:
                egress_rules.append({
                    'to_service': dep_service_id,
                    'ports': dep_service.ports,
                    'protocol': 'TCP'
                })
        
        # Allow external traffic for model downloads, etc.
        if service.service_type in [ServiceType.MODEL_SERVING, ServiceType.MODEL_TRAINING]:
            egress_rules.append({
                'to_external': True,
                'ports': [80, 443],
                'protocol': 'TCP',
                'purpose': 'model_downloads'
            })
        
        return egress_rules

class MLAPIGateway:
    def __init__(self):
        self.gateway_providers = {
            'kong': KongAPIGateway(),
            'ambassador': AmbassadorAPIGateway(),
            'istio_gateway': IstioAPIGateway(),
            'nginx': NginxAPIGateway()
        }
        self.rate_limiter = RateLimiter()
        self.auth_manager = AuthenticationManager()
    
    def setup_api_gateway(self, services, gateway_config):
        """Set up API gateway for ML services"""
        
        gateway_setup = {
            'gateway_configuration': {},
            'routing_rules': {},
            'authentication_config': {},
            'rate_limiting_config': {},
            'api_documentation': {},
            'monitoring_setup': {}
        }
        
        try:
            # Select gateway provider
            provider = gateway_config.get('provider', 'kong')
            gateway = self.gateway_providers.get(provider)
            
            if not gateway:
                raise ValueError(f"Unsupported API gateway provider: {provider}")
            
            # Configure gateway
            gateway_configuration = gateway.configure_gateway(services, gateway_config)
            gateway_setup['gateway_configuration'] = gateway_configuration
            
            # Set up routing rules
            routing_rules = self._setup_routing_rules(services, gateway_config)
            gateway_setup['routing_rules'] = routing_rules
            
            # Configure authentication
            auth_config = self.auth_manager.setup_authentication(services, gateway_config)
            gateway_setup['authentication_config'] = auth_config
            
            # Set up rate limiting
            rate_limiting = self.rate_limiter.setup_rate_limiting(services, gateway_config)
            gateway_setup['rate_limiting_config'] = rate_limiting
            
            # Generate API documentation
            api_docs = self._generate_api_documentation(services, gateway_config)
            gateway_setup['api_documentation'] = api_docs
            
            # Set up gateway monitoring
            monitoring_setup = self._setup_gateway_monitoring(gateway_config)
            gateway_setup['monitoring_setup'] = monitoring_setup
            
            return gateway_setup
            
        except Exception as e:
            logging.error(f"Error setting up API gateway: {str(e)}")
            gateway_setup['error'] = str(e)
            return gateway_setup
    
    def _setup_routing_rules(self, services, config):
        """Set up API routing rules for services"""
        
        routing_rules = {
            'routes': [],
            'load_balancing': {},
            'circuit_breakers': {},
            'retries': {}
        }
        
        for service in services:
            # Create routes for service endpoints
            service_routes = self._create_service_routes(service, config)
            routing_rules['routes'].extend(service_routes)
            
            # Configure load balancing
            lb_config = {
                'service_id': service.service_id,
                'algorithm': self._select_lb_algorithm(service),
                'health_checks': service.health_check_config,
                'upstream_targets': self._get_upstream_targets(service)
            }
            routing_rules['load_balancing'][service.service_id] = lb_config
            
            # Configure circuit breakers
            cb_config = {
                'service_id': service.service_id,
                'failure_threshold': config.get('circuit_breaker_threshold', 5),
                'timeout_seconds': config.get('circuit_breaker_timeout', 60),
                'half_open_requests': config.get('half_open_requests', 3)
            }
            routing_rules['circuit_breakers'][service.service_id] = cb_config
            
            # Configure retries
            retry_config = {
                'service_id': service.service_id,
                'max_retries': config.get('max_retries', 3),
                'retry_on': ['5xx', 'timeout', 'connection-failure'],
                'per_try_timeout_seconds': config.get('per_try_timeout', 5)
            }
            routing_rules['retries'][service.service_id] = retry_config
        
        return routing_rules
    
    def _create_service_routes(self, service, config):
        """Create API routes for individual service"""
        
        routes = []
        
        # Main service route
        main_route = {
            'route_id': f"{service.service_id}-main",
            'path': f"/api/v1/{service.service_name}",
            'methods': ['GET', 'POST', 'PUT', 'DELETE'],
            'service_id': service.service_id,
            'strip_path': True,
            'preserve_host': False
        }
        
        # Service-specific route configurations
        if service.service_type == ServiceType.MODEL_SERVING:
            # Inference endpoint
            inference_route = {
                'route_id': f"{service.service_id}-inference",
                'path': f"/api/v1/models/{service.service_name}/predict",
                'methods': ['POST'],
                'service_id': service.service_id,
                'strip_path': True,
                'timeout_seconds': config.get('inference_timeout', 30)
            }
            routes.append(inference_route)
            
            # Model metadata endpoint
            metadata_route = {
                'route_id': f"{service.service_id}-metadata",
                'path': f"/api/v1/models/{service.service_name}/metadata",
                'methods': ['GET'],
                'service_id': service.service_id,
                'strip_path': True,
                'cache_duration_seconds': config.get('metadata_cache_duration', 300)
            }
            routes.append(metadata_route)
        
        elif service.service_type == ServiceType.DATA_PROCESSING:
            # Data processing endpoint
            processing_route = {
                'route_id': f"{service.service_id}-process",
                'path': f"/api/v1/data/{service.service_name}/process",
                'methods': ['POST'],
                'service_id': service.service_id,
                'strip_path': True,
                'timeout_seconds': config.get('processing_timeout', 300)
            }
            routes.append(processing_route)
        
        routes.append(main_route)
        return routes
```

This comprehensive framework for cloud-native AI architecture provides the theoretical foundations and practical implementation strategies for building microservices-based ML systems with service mesh, container orchestration, and API gateway patterns for scalable, resilient, and maintainable AI platforms.