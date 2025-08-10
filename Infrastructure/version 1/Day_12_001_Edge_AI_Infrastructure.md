# Day 12.1: Edge AI & Distributed Computing Infrastructure

## 🌐 Responsible AI, Privacy & Edge Computing - Part 3

**Focus**: Edge Deployment, Model Optimization, Distributed Inference  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master edge AI infrastructure design and deployment strategies
- Learn model optimization techniques for resource-constrained environments
- Understand distributed inference architectures and edge-cloud coordination
- Analyze IoT integration patterns and real-time edge computing frameworks

---

## 🌐 Edge AI Infrastructure Theory

### **Edge Computing Architecture**

Edge AI infrastructure requires sophisticated approaches to model deployment, resource optimization, and distributed coordination to enable intelligent processing at the network edge.

**Edge AI Framework:**
```
Edge AI Infrastructure Components:
1. Edge Device Management Layer:
   - Device discovery and registration
   - Resource monitoring and allocation
   - Firmware and model update management
   - Security and authentication systems

2. Model Optimization Layer:
   - Model quantization and pruning
   - Neural architecture search for edge
   - Knowledge distillation pipelines
   - Hardware-specific optimization

3. Distributed Inference Layer:
   - Load balancing across edge nodes
   - Hierarchical inference strategies
   - Edge-cloud coordination protocols
   - Latency-aware request routing

4. Orchestration & Coordination Layer:
   - Federated learning coordination
   - Global state synchronization
   - Fault tolerance and recovery
   - Performance monitoring and analytics

Edge Computing Mathematical Models:
Latency Optimization:
Total_Latency = Network_Latency + Computation_Latency + Queuing_Latency
where:
Network_Latency = f(Distance, Bandwidth, Protocol_Overhead)
Computation_Latency = f(Model_Complexity, Hardware_Capacity, Batch_Size)

Resource Allocation:
Optimal_Placement = arg min(Σ(Latency_i × Priority_i + Cost_i))
subject to: Σ(Resource_Usage_i) ≤ Available_Resources

Edge-Cloud Coordination:
Decision_Threshold = f(Local_Confidence, Network_Conditions, SLA_Requirements)
Offload_Decision = Local_Confidence < Decision_Threshold

Energy Efficiency:
Energy_Per_Inference = Computation_Energy + Communication_Energy + Idle_Energy
Battery_Life = Battery_Capacity / Average_Power_Consumption
```

**Comprehensive Edge AI System:**
```
Edge AI Infrastructure Implementation:
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import queue
import time
from datetime import datetime
import json
import asyncio

class EdgeDeviceType(Enum):
    MOBILE_PHONE = "mobile_phone"
    IOT_SENSOR = "iot_sensor"
    EDGE_SERVER = "edge_server"
    EMBEDDED_DEVICE = "embedded_device"
    SMART_CAMERA = "smart_camera"
    AUTONOMOUS_VEHICLE = "autonomous_vehicle"

class ModelOptimizationType(Enum):
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    HARDWARE_OPTIMIZATION = "hardware_optimization"

@dataclass
class EdgeDevice:
    device_id: str
    device_type: EdgeDeviceType
    cpu_cores: int
    memory_mb: int
    storage_gb: int
    battery_capacity_mah: Optional[int]
    network_bandwidth_mbps: float
    location: Dict[str, float]  # latitude, longitude
    capabilities: List[str]
    current_load: float = 0.0
    available: bool = True
    last_heartbeat: datetime = None

@dataclass
class EdgeModel:
    model_id: str
    model_name: str
    model_version: str
    model_size_mb: float
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    inference_time_ms: float
    memory_requirement_mb: float
    accuracy: float
    optimization_applied: List[ModelOptimizationType]
    target_devices: List[EdgeDeviceType]
    deployment_timestamp: datetime

class EdgeAIInfrastructure:
    def __init__(self):
        self.device_manager = EdgeDeviceManager()
        self.model_optimizer = EdgeModelOptimizer()
        self.inference_orchestrator = DistributedInferenceOrchestrator()
        self.edge_cloud_coordinator = EdgeCloudCoordinator()
        self.resource_scheduler = EdgeResourceScheduler()
        self.monitoring_system = EdgeMonitoringSystem()
    
    def deploy_edge_ai_system(self, deployment_config):
        """Deploy comprehensive edge AI system"""
        
        deployment_result = {
            'deployment_id': self._generate_deployment_id(),
            'timestamp': datetime.utcnow(),
            'deployment_config': deployment_config,
            'device_registration': {},
            'model_optimization': {},
            'inference_setup': {},
            'monitoring_setup': {},
            'performance_metrics': {}
        }
        
        try:
            # Phase 1: Device Discovery and Registration
            logging.info("Phase 1: Discovering and registering edge devices")
            device_registration = self.device_manager.discover_and_register_devices(
                discovery_config=deployment_config.get('device_discovery', {})
            )
            deployment_result['device_registration'] = device_registration
            
            # Phase 2: Model Optimization for Edge Deployment
            logging.info("Phase 2: Optimizing models for edge deployment")
            model_optimization = self.model_optimizer.optimize_models_for_edge(
                models=deployment_config.get('models', []),
                target_devices=device_registration['registered_devices'],
                optimization_constraints=deployment_config.get('optimization_constraints', {})
            )
            deployment_result['model_optimization'] = model_optimization
            
            # Phase 3: Set up Distributed Inference
            logging.info("Phase 3: Setting up distributed inference")
            inference_setup = self.inference_orchestrator.setup_distributed_inference(
                optimized_models=model_optimization['optimized_models'],
                edge_devices=device_registration['registered_devices'],
                inference_config=deployment_config.get('inference_config', {})
            )
            deployment_result['inference_setup'] = inference_setup
            
            # Phase 4: Configure Edge-Cloud Coordination
            logging.info("Phase 4: Configuring edge-cloud coordination")
            coordination_setup = self.edge_cloud_coordinator.setup_coordination(
                edge_infrastructure=deployment_result,
                cloud_config=deployment_config.get('cloud_config', {})
            )
            deployment_result['coordination_setup'] = coordination_setup
            
            # Phase 5: Initialize Monitoring and Analytics
            logging.info("Phase 5: Initializing monitoring and analytics")
            monitoring_setup = self.monitoring_system.initialize_monitoring(
                edge_devices=device_registration['registered_devices'],
                inference_endpoints=inference_setup['endpoints'],
                monitoring_config=deployment_config.get('monitoring_config', {})
            )
            deployment_result['monitoring_setup'] = monitoring_setup
            
            # Calculate deployment performance metrics
            performance_metrics = self._calculate_deployment_metrics(deployment_result)
            deployment_result['performance_metrics'] = performance_metrics
            
            logging.info(f"Edge AI system deployment completed successfully. "
                        f"Devices: {len(device_registration['registered_devices'])}, "
                        f"Models: {len(model_optimization['optimized_models'])}")
            
            return deployment_result
            
        except Exception as e:
            logging.error(f"Error in edge AI system deployment: {str(e)}")
            deployment_result['error'] = str(e)
            return deployment_result
    
    def _calculate_deployment_metrics(self, deployment_result):
        """Calculate comprehensive deployment performance metrics"""
        
        metrics = {}
        
        # Device metrics
        devices = deployment_result['device_registration']['registered_devices']
        metrics['device_statistics'] = {
            'total_devices': len(devices),
            'device_types': {device_type.value: len([d for d in devices if d.device_type == device_type]) 
                           for device_type in EdgeDeviceType},
            'total_compute_capacity': sum(d.cpu_cores for d in devices),
            'total_memory_gb': sum(d.memory_mb for d in devices) / 1024,
            'average_network_bandwidth': np.mean([d.network_bandwidth_mbps for d in devices])
        }
        
        # Model metrics
        models = deployment_result['model_optimization']['optimized_models']
        metrics['model_statistics'] = {
            'total_models': len(models),
            'average_model_size_mb': np.mean([m.model_size_mb for m in models]),
            'average_inference_time_ms': np.mean([m.inference_time_ms for m in models]),
            'optimization_distribution': {opt_type.value: len([m for m in models 
                                                             if opt_type in m.optimization_applied]) 
                                        for opt_type in ModelOptimizationType}
        }
        
        # Infrastructure metrics
        inference_setup = deployment_result['inference_setup']
        metrics['infrastructure_statistics'] = {
            'inference_endpoints': len(inference_setup.get('endpoints', [])),
            'load_balancer_instances': len(inference_setup.get('load_balancers', [])),
            'estimated_total_throughput': inference_setup.get('estimated_throughput', 0),
            'estimated_latency_p99': inference_setup.get('estimated_latency_p99', 0)
        }
        
        return metrics

class EdgeDeviceManager:
    def __init__(self):
        self.registered_devices = {}
        self.device_discovery_protocols = {
            'bluetooth': BluetoothDiscovery(),
            'wifi': WiFiDiscovery(),
            'cellular': CellularDiscovery(),
            'ethernet': EthernetDiscovery()
        }
        self.device_monitor = DeviceHealthMonitor()
    
    def discover_and_register_devices(self, discovery_config):
        """Discover and register edge devices"""
        
        discovery_result = {
            'discovered_devices': [],
            'registered_devices': [],
            'discovery_methods_used': [],
            'discovery_duration_seconds': 0,
            'registration_success_rate': 0.0
        }
        
        discovery_start = time.time()
        
        # Use multiple discovery protocols
        enabled_protocols = discovery_config.get('protocols', ['wifi', 'bluetooth'])
        
        for protocol in enabled_protocols:
            if protocol in self.device_discovery_protocols:
                try:
                    protocol_discovery = self.device_discovery_protocols[protocol]
                    discovered = protocol_discovery.discover_devices(
                        timeout_seconds=discovery_config.get('timeout_seconds', 30),
                        discovery_range=discovery_config.get('discovery_range', 100)
                    )
                    
                    discovery_result['discovered_devices'].extend(discovered)
                    discovery_result['discovery_methods_used'].append(protocol)
                    
                except Exception as e:
                    logging.error(f"Error in {protocol} discovery: {str(e)}")
        
        # Remove duplicates based on device_id
        unique_devices = {}
        for device in discovery_result['discovered_devices']:
            unique_devices[device.device_id] = device
        
        discovery_result['discovered_devices'] = list(unique_devices.values())
        
        # Register discovered devices
        registration_successes = 0
        
        for device in discovery_result['discovered_devices']:
            try:
                registration_success = self._register_device(device, discovery_config)
                if registration_success:
                    discovery_result['registered_devices'].append(device)
                    registration_successes += 1
                    
            except Exception as e:
                logging.error(f"Error registering device {device.device_id}: {str(e)}")
        
        discovery_result['discovery_duration_seconds'] = time.time() - discovery_start
        discovery_result['registration_success_rate'] = (
            registration_successes / len(discovery_result['discovered_devices']) 
            if discovery_result['discovered_devices'] else 0.0
        )
        
        # Start device monitoring
        self.device_monitor.start_monitoring(discovery_result['registered_devices'])
        
        return discovery_result
    
    def _register_device(self, device, discovery_config):
        """Register individual edge device"""
        
        # Validate device capabilities
        if not self._validate_device_capabilities(device, discovery_config):
            logging.warning(f"Device {device.device_id} does not meet minimum requirements")
            return False
        
        # Perform security authentication
        if not self._authenticate_device(device, discovery_config):
            logging.warning(f"Device {device.device_id} failed authentication")
            return False
        
        # Initialize device connection
        if not self._initialize_device_connection(device):
            logging.warning(f"Failed to initialize connection to device {device.device_id}")
            return False
        
        # Store device registration
        device.last_heartbeat = datetime.utcnow()
        self.registered_devices[device.device_id] = device
        
        logging.info(f"Successfully registered device {device.device_id} ({device.device_type.value})")
        return True
    
    def _validate_device_capabilities(self, device, discovery_config):
        """Validate device meets minimum capability requirements"""
        
        min_requirements = discovery_config.get('minimum_requirements', {})
        
        # Check CPU cores
        if device.cpu_cores < min_requirements.get('min_cpu_cores', 1):
            return False
        
        # Check memory
        if device.memory_mb < min_requirements.get('min_memory_mb', 512):
            return False
        
        # Check storage
        if device.storage_gb < min_requirements.get('min_storage_gb', 1):
            return False
        
        # Check network bandwidth
        if device.network_bandwidth_mbps < min_requirements.get('min_bandwidth_mbps', 1.0):
            return False
        
        return True
    
    def _authenticate_device(self, device, discovery_config):
        """Authenticate edge device"""
        
        auth_config = discovery_config.get('authentication', {})
        
        if not auth_config.get('enabled', True):
            return True  # Authentication disabled
        
        # Certificate-based authentication
        if auth_config.get('method') == 'certificate':
            return self._verify_device_certificate(device, auth_config)
        
        # Token-based authentication
        elif auth_config.get('method') == 'token':
            return self._verify_device_token(device, auth_config)
        
        # Default to simple validation
        else:
            return self._simple_device_validation(device)
    
    def _verify_device_certificate(self, device, auth_config):
        """Verify device certificate (simplified implementation)"""
        # In practice, this would involve proper PKI certificate validation
        return True  # Simplified for demonstration
    
    def _verify_device_token(self, device, auth_config):
        """Verify device authentication token"""
        # In practice, this would validate JWT tokens or similar
        return True  # Simplified for demonstration
    
    def _simple_device_validation(self, device):
        """Simple device validation"""
        # Basic validation - check if device has valid ID and type
        return (device.device_id is not None and 
                len(device.device_id) > 0 and 
                device.device_type is not None)
    
    def _initialize_device_connection(self, device):
        """Initialize connection to edge device"""
        
        # This would involve setting up communication channels
        # For demonstration, we'll simulate successful connection
        try:
            # Simulate connection setup
            time.sleep(0.1)  # Simulate connection time
            
            # Set device as available
            device.available = True
            device.current_load = 0.0
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize connection to {device.device_id}: {str(e)}")
            return False

class EdgeModelOptimizer:
    def __init__(self):
        self.optimization_engines = {
            ModelOptimizationType.QUANTIZATION: QuantizationOptimizer(),
            ModelOptimizationType.PRUNING: PruningOptimizer(),
            ModelOptimizationType.KNOWLEDGE_DISTILLATION: KnowledgeDistillationOptimizer(),
            ModelOptimizationType.NEURAL_ARCHITECTURE_SEARCH: NASOptimizer(),
            ModelOptimizationType.HARDWARE_OPTIMIZATION: HardwareOptimizer()
        }
        self.model_analyzer = ModelAnalyzer()
    
    def optimize_models_for_edge(self, models, target_devices, optimization_constraints):
        """Optimize models for edge deployment"""
        
        optimization_result = {
            'optimized_models': [],
            'optimization_statistics': {},
            'device_compatibility_matrix': {},
            'performance_improvements': {}
        }
        
        for model_config in models:
            try:
                # Analyze model characteristics
                model_analysis = self.model_analyzer.analyze_model(model_config)
                
                # Determine optimal optimization strategy
                optimization_strategy = self._determine_optimization_strategy(
                    model_analysis, target_devices, optimization_constraints
                )
                
                # Apply optimizations
                optimized_model = self._apply_optimizations(
                    model_config, optimization_strategy
                )
                
                # Validate optimized model
                validation_result = self._validate_optimized_model(
                    optimized_model, target_devices, optimization_constraints
                )
                
                if validation_result['valid']:
                    optimization_result['optimized_models'].append(optimized_model)
                    
                    # Record optimization statistics
                    self._record_optimization_statistics(
                        model_config, optimized_model, optimization_strategy, 
                        optimization_result['optimization_statistics']
                    )
                
            except Exception as e:
                logging.error(f"Error optimizing model {model_config.get('name', 'unknown')}: {str(e)}")
        
        # Generate device compatibility matrix
        optimization_result['device_compatibility_matrix'] = self._generate_compatibility_matrix(
            optimization_result['optimized_models'], target_devices
        )
        
        # Calculate performance improvements
        optimization_result['performance_improvements'] = self._calculate_performance_improvements(
            models, optimization_result['optimized_models']
        )
        
        return optimization_result
    
    def _determine_optimization_strategy(self, model_analysis, target_devices, constraints):
        """Determine optimal optimization strategy for model and devices"""
        
        strategy = {
            'optimizations': [],
            'target_size_reduction': 0.5,  # Default 50% size reduction
            'target_speedup': 2.0,         # Default 2x speedup
            'acceptable_accuracy_loss': 0.05  # Default 5% accuracy loss
        }
        
        # Analyze constraints
        max_model_size = constraints.get('max_model_size_mb', float('inf'))
        max_inference_time = constraints.get('max_inference_time_ms', float('inf'))
        max_accuracy_loss = constraints.get('max_accuracy_loss', 0.05)
        
        # Determine device capabilities
        min_memory = min(device.memory_mb for device in target_devices)
        min_compute = min(device.cpu_cores for device in target_devices)
        
        # Strategy selection based on constraints and capabilities
        if model_analysis['model_size_mb'] > max_model_size:
            strategy['optimizations'].append(ModelOptimizationType.QUANTIZATION)
            strategy['optimizations'].append(ModelOptimizationType.PRUNING)
        
        if model_analysis['estimated_inference_time_ms'] > max_inference_time:
            strategy['optimizations'].append(ModelOptimizationType.HARDWARE_OPTIMIZATION)
            
            if min_compute < 4:  # Low compute devices
                strategy['optimizations'].append(ModelOptimizationType.KNOWLEDGE_DISTILLATION)
        
        if min_memory < 1024:  # Less than 1GB memory
            strategy['optimizations'].append(ModelOptimizationType.QUANTIZATION)
        
        # Advanced optimization for very constrained devices
        if any(device.device_type in [EdgeDeviceType.IOT_SENSOR, EdgeDeviceType.EMBEDDED_DEVICE] 
               for device in target_devices):
            strategy['optimizations'].append(ModelOptimizationType.NEURAL_ARCHITECTURE_SEARCH)
            strategy['target_size_reduction'] = 0.8  # More aggressive reduction
        
        # Update strategy parameters
        strategy['acceptable_accuracy_loss'] = min(max_accuracy_loss, strategy['acceptable_accuracy_loss'])
        
        return strategy
    
    def _apply_optimizations(self, model_config, optimization_strategy):
        """Apply optimization techniques to model"""
        
        optimized_model = EdgeModel(
            model_id=f"{model_config['id']}_optimized",
            model_name=model_config['name'],
            model_version=f"{model_config['version']}_edge",
            model_size_mb=model_config['size_mb'],
            input_shape=tuple(model_config['input_shape']),
            output_shape=tuple(model_config['output_shape']),
            inference_time_ms=model_config.get('inference_time_ms', 100),
            memory_requirement_mb=model_config.get('memory_requirement_mb', model_config['size_mb'] * 2),
            accuracy=model_config.get('accuracy', 0.9),
            optimization_applied=[],
            target_devices=[],
            deployment_timestamp=datetime.utcnow()
        )
        
        # Apply each optimization in sequence
        for optimization_type in optimization_strategy['optimizations']:
            optimizer = self.optimization_engines.get(optimization_type)
            if optimizer:
                optimized_model = optimizer.optimize(optimized_model, optimization_strategy)
                optimized_model.optimization_applied.append(optimization_type)
        
        return optimized_model

class QuantizationOptimizer:
    def optimize(self, model, strategy):
        """Apply quantization optimization"""
        
        # Simulate quantization effects
        quantization_factor = 0.25  # 8-bit quantization reduces size by ~75%
        
        # Reduce model size
        model.model_size_mb *= quantization_factor
        
        # Reduce memory requirement
        model.memory_requirement_mb *= quantization_factor
        
        # Improve inference speed (due to smaller model)
        model.inference_time_ms *= 0.7  # ~30% speedup
        
        # Small accuracy loss
        model.accuracy *= 0.98  # ~2% accuracy loss
        
        logging.info(f"Applied quantization to {model.model_name}: "
                    f"Size: {model.model_size_mb:.2f}MB, "
                    f"Speed: {model.inference_time_ms:.2f}ms")
        
        return model

class PruningOptimizer:
    def optimize(self, model, strategy):
        """Apply pruning optimization"""
        
        # Simulate pruning effects
        pruning_ratio = 0.5  # Remove 50% of parameters
        
        # Reduce model size based on pruning ratio
        model.model_size_mb *= (1 - pruning_ratio)
        
        # Reduce memory requirement
        model.memory_requirement_mb *= (1 - pruning_ratio)
        
        # Moderate speedup
        model.inference_time_ms *= 0.8  # ~20% speedup
        
        # Minimal accuracy loss with careful pruning
        model.accuracy *= 0.995  # ~0.5% accuracy loss
        
        logging.info(f"Applied pruning to {model.model_name}: "
                    f"Size: {model.model_size_mb:.2f}MB, "
                    f"Speed: {model.inference_time_ms:.2f}ms")
        
        return model

class KnowledgeDistillationOptimizer:
    def optimize(self, model, strategy):
        """Apply knowledge distillation optimization"""
        
        # Simulate knowledge distillation creating smaller student model
        distillation_factor = 0.3  # Student model is 30% of teacher size
        
        # Significant size reduction
        model.model_size_mb *= distillation_factor
        
        # Memory reduction
        model.memory_requirement_mb *= distillation_factor
        
        # Significant speedup
        model.inference_time_ms *= 0.4  # ~60% speedup
        
        # Moderate accuracy loss
        model.accuracy *= 0.95  # ~5% accuracy loss
        
        logging.info(f"Applied knowledge distillation to {model.model_name}: "
                    f"Size: {model.model_size_mb:.2f}MB, "
                    f"Speed: {model.inference_time_ms:.2f}ms")
        
        return model

class DistributedInferenceOrchestrator:
    def __init__(self):
        self.load_balancer = EdgeLoadBalancer()
        self.inference_scheduler = InferenceScheduler()
        self.latency_optimizer = LatencyOptimizer()
        self.fault_tolerance_manager = FaultToleranceManager()
    
    def setup_distributed_inference(self, optimized_models, edge_devices, inference_config):
        """Set up distributed inference across edge devices"""
        
        inference_setup = {
            'endpoints': [],
            'load_balancers': [],
            'routing_policies': {},
            'failover_configurations': {},
            'estimated_throughput': 0,
            'estimated_latency_p50': 0,
            'estimated_latency_p99': 0
        }
        
        try:
            # Create inference endpoints on edge devices
            endpoints = self._create_inference_endpoints(optimized_models, edge_devices, inference_config)
            inference_setup['endpoints'] = endpoints
            
            # Set up load balancing
            load_balancers = self._setup_load_balancing(endpoints, inference_config)
            inference_setup['load_balancers'] = load_balancers
            
            # Configure routing policies
            routing_policies = self._configure_routing_policies(endpoints, inference_config)
            inference_setup['routing_policies'] = routing_policies
            
            # Set up failover configurations
            failover_configs = self._setup_failover_configurations(endpoints, inference_config)
            inference_setup['failover_configurations'] = failover_configs
            
            # Calculate performance estimates
            performance_estimates = self._calculate_performance_estimates(endpoints, inference_config)
            inference_setup.update(performance_estimates)
            
            # Start inference services
            self._start_inference_services(inference_setup)
            
            logging.info(f"Distributed inference setup completed: "
                        f"{len(endpoints)} endpoints, "
                        f"{len(load_balancers)} load balancers")
            
            return inference_setup
            
        except Exception as e:
            logging.error(f"Error setting up distributed inference: {str(e)}")
            inference_setup['error'] = str(e)
            return inference_setup
    
    def _create_inference_endpoints(self, models, devices, config):
        """Create inference endpoints on edge devices"""
        
        endpoints = []
        
        # Device-model compatibility matrix
        compatibility_matrix = self._build_compatibility_matrix(models, devices)
        
        for device in devices:
            # Find compatible models for this device
            compatible_models = compatibility_matrix.get(device.device_id, [])
            
            if compatible_models:
                # Create endpoint for each compatible model
                for model in compatible_models:
                    endpoint = {
                        'endpoint_id': f"{device.device_id}_{model.model_id}",
                        'device_id': device.device_id,
                        'model_id': model.model_id,
                        'endpoint_url': f"http://{device.device_id}:8080/predict/{model.model_id}",
                        'max_batch_size': self._calculate_max_batch_size(device, model),
                        'estimated_throughput_qps': self._estimate_throughput(device, model),
                        'estimated_latency_ms': model.inference_time_ms,
                        'health_check_url': f"http://{device.device_id}:8080/health",
                        'status': 'initializing'
                    }
                    endpoints.append(endpoint)
                    
                    logging.info(f"Created endpoint {endpoint['endpoint_id']} on device {device.device_id}")
        
        return endpoints
    
    def _build_compatibility_matrix(self, models, devices):
        """Build compatibility matrix between models and devices"""
        
        compatibility_matrix = {device.device_id: [] for device in devices}
        
        for device in devices:
            for model in models:
                if self._is_model_compatible_with_device(model, device):
                    compatibility_matrix[device.device_id].append(model)
        
        return compatibility_matrix
    
    def _is_model_compatible_with_device(self, model, device):
        """Check if model is compatible with device"""
        
        # Check memory requirements
        if model.memory_requirement_mb > device.memory_mb * 0.8:  # Leave 20% memory headroom
            return False
        
        # Check device type compatibility
        if device.device_type not in model.target_devices and model.target_devices:
            return False
        
        # Check storage requirements
        if model.model_size_mb > device.storage_gb * 1024 * 0.5:  # Use max 50% storage
            return False
        
        return True
    
    def _calculate_max_batch_size(self, device, model):
        """Calculate maximum batch size for device-model combination"""
        
        # Available memory for inference
        available_memory = device.memory_mb * 0.6  # Use 60% of memory for inference
        
        # Memory per sample (rough estimate)
        memory_per_sample = np.prod(model.input_shape) * 4 / (1024 * 1024)  # 4 bytes per float32
        
        # Maximum batch size
        max_batch_size = int(available_memory / (model.memory_requirement_mb + memory_per_sample))
        
        return max(1, min(max_batch_size, 32))  # Cap at 32 for stability
    
    def _estimate_throughput(self, device, model):
        """Estimate throughput for device-model combination"""
        
        # Base throughput calculation
        base_throughput = 1000 / model.inference_time_ms  # QPS for single instance
        
        # Scale by device capabilities
        cpu_factor = device.cpu_cores / 4.0  # Normalize to 4 cores
        memory_factor = device.memory_mb / 2048.0  # Normalize to 2GB
        
        scaling_factor = min(cpu_factor, memory_factor)  # Bottleneck determines scaling
        
        estimated_throughput = base_throughput * scaling_factor
        
        return max(0.1, estimated_throughput)  # Minimum 0.1 QPS

class EdgeCloudCoordinator:
    def __init__(self):
        self.cloud_connector = CloudConnector()
        self.hybrid_scheduler = HybridInferenceScheduler()
        self.data_sync_manager = EdgeCloudDataSyncManager()
        self.latency_monitor = LatencyMonitor()
    
    def setup_coordination(self, edge_infrastructure, cloud_config):
        """Set up coordination between edge and cloud infrastructure"""
        
        coordination_setup = {
            'cloud_connection': {},
            'hybrid_policies': {},
            'data_synchronization': {},
            'fallback_strategies': {},
            'coordination_metrics': {}
        }
        
        try:
            # Establish cloud connection
            cloud_connection = self.cloud_connector.establish_connection(cloud_config)
            coordination_setup['cloud_connection'] = cloud_connection
            
            # Configure hybrid inference policies
            hybrid_policies = self._configure_hybrid_policies(edge_infrastructure, cloud_config)
            coordination_setup['hybrid_policies'] = hybrid_policies
            
            # Set up data synchronization
            data_sync = self.data_sync_manager.setup_synchronization(
                edge_infrastructure, cloud_config
            )
            coordination_setup['data_synchronization'] = data_sync
            
            # Configure fallback strategies
            fallback_strategies = self._configure_fallback_strategies(
                edge_infrastructure, cloud_config
            )
            coordination_setup['fallback_strategies'] = fallback_strategies
            
            # Initialize coordination monitoring
            coordination_metrics = self._initialize_coordination_monitoring(coordination_setup)
            coordination_setup['coordination_metrics'] = coordination_metrics
            
            logging.info("Edge-cloud coordination setup completed successfully")
            
            return coordination_setup
            
        except Exception as e:
            logging.error(f"Error setting up edge-cloud coordination: {str(e)}")
            coordination_setup['error'] = str(e)
            return coordination_setup
    
    def _configure_hybrid_policies(self, edge_infrastructure, cloud_config):
        """Configure hybrid inference policies"""
        
        policies = {
            'routing_rules': [],
            'offloading_thresholds': {},
            'prioritization_criteria': {},
            'resource_allocation': {}
        }
        
        # Define routing rules based on request characteristics
        policies['routing_rules'] = [
            {
                'condition': 'latency_sensitive',
                'threshold': cloud_config.get('latency_threshold_ms', 100),
                'action': 'route_to_edge',
                'fallback': 'route_to_cloud'
            },
            {
                'condition': 'high_accuracy_required',
                'threshold': cloud_config.get('accuracy_threshold', 0.95),
                'action': 'route_to_cloud',
                'fallback': 'route_to_edge'
            },
            {
                'condition': 'edge_capacity_exceeded',
                'threshold': cloud_config.get('capacity_threshold', 0.8),
                'action': 'route_to_cloud',
                'fallback': 'queue_request'
            }
        ]
        
        # Define offloading thresholds
        policies['offloading_thresholds'] = {
            'cpu_utilization': cloud_config.get('cpu_threshold', 0.8),
            'memory_utilization': cloud_config.get('memory_threshold', 0.8),
            'latency_threshold': cloud_config.get('latency_threshold_ms', 200),
            'confidence_threshold': cloud_config.get('confidence_threshold', 0.7)
        }
        
        return policies
```

This comprehensive framework for edge AI infrastructure provides the theoretical foundations and practical implementation strategies for deploying intelligent systems at the network edge with optimized models, distributed inference, and edge-cloud coordination.