# Day 9.4: Cost Optimization & Resource Efficiency

## 💰 Monitoring, Observability & Debugging - Part 4

**Focus**: ML Cost Analysis, Resource Optimization, Efficiency Monitoring  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master cost analysis and optimization strategies for ML infrastructure and workloads
- Learn resource efficiency monitoring and automated optimization techniques
- Understand cost attribution across ML pipelines and multi-tenant environments
- Analyze trade-offs between performance, accuracy, and cost in ML system design

---

## 💰 ML Cost Optimization Theoretical Framework

### **Cost Attribution and Analysis Architecture**

ML systems have unique cost characteristics that require specialized monitoring and optimization approaches, considering compute intensity, data storage, and model complexity trade-offs.

**ML Cost Taxonomy:**
```
ML Infrastructure Cost Categories:
1. Compute Costs:
   - Training infrastructure (GPU/CPU hours)
   - Inference serving resources
   - Data processing and ETL workloads
   - Model development and experimentation

2. Storage Costs:
   - Training data storage (hot/warm/cold tiers)
   - Model artifact storage and versioning
   - Feature store storage and caching
   - Intermediate data and checkpoints

3. Network Costs:
   - Data transfer between services
   - Cross-region replication
   - API calls and model predictions
   - Data ingestion and egress

4. Platform Costs:
   - Managed ML services (SageMaker, Vertex AI)
   - Container orchestration overhead
   - Monitoring and observability tools
   - Security and compliance tools

Cost Optimization Mathematical Model:
Total_Cost = Compute_Cost + Storage_Cost + Network_Cost + Platform_Cost

Cost_Efficiency = Business_Value / Total_Cost

Resource_Utilization = Actual_Usage / Provisioned_Capacity

Cost_Per_Prediction = Total_Inference_Cost / Number_of_Predictions

Training_Cost_Efficiency = Model_Accuracy_Improvement / Training_Cost

Pareto_Efficiency_Score = Performance_Gain / (Cost_Increase + ε)

Multi-Objective Cost Optimization:
minimize: w1 × Cost + w2 × (1 - Performance) + w3 × Latency
subject to:
    Accuracy ≥ minimum_accuracy_threshold
    Latency ≤ maximum_latency_sla
    Budget ≤ allocated_budget
    Resource_Usage ≤ capacity_constraints

Cost Attribution Model:
Service_Cost = Σ(Resource_Type × Usage_Duration × Unit_Price × Allocation_Weight)

Where Allocation_Weight considers:
- Resource sharing across teams/projects
- Peak vs. off-peak usage patterns
- Reserved vs. on-demand pricing
- Multi-tenancy overhead
```

**Cost Monitoring Infrastructure:**
```
ML Cost Tracking System Architecture:
1. Resource Usage Collectors:
   - Kubernetes resource metrics
   - Cloud provider billing APIs
   - Custom application metrics
   - Third-party service costs

2. Cost Attribution Engine:
   - Multi-dimensional cost allocation
   - Tenant and project attribution
   - Resource sharing calculations
   - Time-based cost distribution

3. Optimization Recommendation Engine:
   - Right-sizing recommendations
   - Scheduling optimizations
   - Resource pooling opportunities
   - Architecture optimization suggestions

4. Budget Management System:
   - Budget tracking and alerting
   - Forecasting and trend analysis
   - Cost anomaly detection
   - Automated cost controls

Cost Metrics Collection Framework:
class MLCostCollector:
    def __init__(self, cloud_providers, k8s_client):
        self.cloud_providers = cloud_providers
        self.k8s_client = k8s_client
        self.cost_cache = {}
        self.resource_mappings = self._build_resource_mappings()
    
    def collect_compute_costs(self, time_range='24h'):
        """Collect compute costs across all resources"""
        
        cost_data = {
            'kubernetes_costs': self._collect_k8s_costs(time_range),
            'cloud_vm_costs': self._collect_vm_costs(time_range),
            'managed_service_costs': self._collect_managed_service_costs(time_range),
            'gpu_costs': self._collect_gpu_costs(time_range)
        }
        
        return cost_data
    
    def _collect_k8s_costs(self, time_range):
        """Collect Kubernetes workload costs"""
        
        # Get resource usage from Prometheus
        resource_usage = self._query_resource_usage(time_range)
        
        # Map to cost information
        k8s_costs = {}
        
        for namespace, usage in resource_usage.items():
            namespace_cost = 0
            
            # Calculate CPU cost
            cpu_hours = usage['cpu_seconds'] / 3600
            cpu_cost = cpu_hours * self._get_cpu_unit_cost()
            
            # Calculate memory cost
            memory_gb_hours = (usage['memory_bytes'] / (1024**3)) * (usage['duration_seconds'] / 3600)
            memory_cost = memory_gb_hours * self._get_memory_unit_cost()
            
            # Calculate GPU cost if applicable
            gpu_cost = 0
            if 'gpu_seconds' in usage:
                gpu_hours = usage['gpu_seconds'] / 3600
                gpu_cost = gpu_hours * self._get_gpu_unit_cost(usage.get('gpu_type', 'default'))
            
            # Calculate storage cost
            storage_cost = usage.get('storage_gb', 0) * self._get_storage_unit_cost()
            
            namespace_cost = cpu_cost + memory_cost + gpu_cost + storage_cost
            
            k8s_costs[namespace] = {
                'total_cost': namespace_cost,
                'cpu_cost': cpu_cost,
                'memory_cost': memory_cost,
                'gpu_cost': gpu_cost,
                'storage_cost': storage_cost,
                'workloads': self._get_workload_costs(namespace, usage)
            }
        
        return k8s_costs
    
    def _get_workload_costs(self, namespace, usage):
        """Calculate costs per workload within namespace"""
        
        workload_costs = {}
        
        # Get workload-specific resource usage
        workload_usage = self._query_workload_usage(namespace)
        
        for workload_name, w_usage in workload_usage.items():
            # Calculate proportional cost based on resource usage
            cpu_proportion = w_usage['cpu_seconds'] / max(usage['cpu_seconds'], 1)
            memory_proportion = w_usage['memory_bytes'] / max(usage['memory_bytes'], 1)
            
            workload_cost = (
                cpu_proportion * usage['cpu_cost'] +
                memory_proportion * usage['memory_cost']
            )
            
            # Add workload-specific metrics
            workload_costs[workload_name] = {
                'cost': workload_cost,
                'cpu_usage': w_usage['cpu_seconds'] / 3600,  # CPU hours
                'memory_usage': w_usage['memory_bytes'] / (1024**3),  # GB
                'cost_per_hour': workload_cost / max(w_usage['duration_seconds'] / 3600, 0.001),
                'workload_type': self._classify_workload_type(workload_name)
            }
        
        return workload_costs
    
    def _classify_workload_type(self, workload_name):
        """Classify workload type for cost analysis"""
        
        if 'training' in workload_name.lower():
            return 'ml_training'
        elif 'inference' in workload_name.lower() or 'serving' in workload_name.lower():
            return 'ml_inference' 
        elif 'pipeline' in workload_name.lower() or 'etl' in workload_name.lower():
            return 'data_processing'
        elif 'notebook' in workload_name.lower() or 'jupyter' in workload_name.lower():
            return 'experimentation'
        else:
            return 'other'
```

---

## 📊 Advanced Cost Analysis and Attribution

### **Multi-Dimensional Cost Attribution**

**Cost Attribution Implementation:**
```
Advanced ML Cost Attribution System:
class MLCostAttributor:
    def __init__(self, cost_collector, metadata_store):
        self.cost_collector = cost_collector
        self.metadata_store = metadata_store
        self.attribution_rules = self._load_attribution_rules()
        
    def attribute_costs(self, time_range='24h', attribution_dimensions=None):
        """Perform multi-dimensional cost attribution"""
        
        if not attribution_dimensions:
            attribution_dimensions = ['team', 'project', 'model', 'environment', 'workload_type']
        
        # Collect raw cost data
        raw_costs = self.cost_collector.collect_all_costs(time_range)
        
        # Perform attribution across dimensions
        attributed_costs = {}
        
        for dimension in attribution_dimensions:
            attributed_costs[dimension] = self._attribute_by_dimension(
                raw_costs, dimension
            )
        
        # Create cross-dimensional analysis
        attributed_costs['cross_dimensional'] = self._perform_cross_dimensional_attribution(
            raw_costs, attribution_dimensions
        )
        
        # Calculate cost efficiency metrics
        attributed_costs['efficiency_metrics'] = self._calculate_efficiency_metrics(
            attributed_costs
        )
        
        return attributed_costs
    
    def _attribute_by_dimension(self, raw_costs, dimension):
        """Attribute costs by a specific dimension"""
        
        dimension_costs = {}
        
        for resource_id, cost_data in raw_costs.items():
            # Get dimension value for this resource
            dimension_value = self._get_dimension_value(resource_id, dimension)
            
            if dimension_value not in dimension_costs:
                dimension_costs[dimension_value] = {
                    'total_cost': 0,
                    'compute_cost': 0,
                    'storage_cost': 0,
                    'network_cost': 0,
                    'resources': []
                }
            
            # Allocate costs
            allocation_weight = self._calculate_allocation_weight(
                resource_id, dimension, dimension_value
            )
            
            allocated_cost = cost_data['total_cost'] * allocation_weight
            
            dimension_costs[dimension_value]['total_cost'] += allocated_cost
            dimension_costs[dimension_value]['compute_cost'] += cost_data.get('compute_cost', 0) * allocation_weight
            dimension_costs[dimension_value]['storage_cost'] += cost_data.get('storage_cost', 0) * allocation_weight
            dimension_costs[dimension_value]['network_cost'] += cost_data.get('network_cost', 0) * allocation_weight
            dimension_costs[dimension_value]['resources'].append({
                'resource_id': resource_id,
                'allocated_cost': allocated_cost,
                'allocation_weight': allocation_weight
            })
        
        return dimension_costs
    
    def _calculate_allocation_weight(self, resource_id, dimension, dimension_value):
        """Calculate allocation weight for shared resources"""
        
        # Get sharing information
        sharing_info = self.metadata_store.get_resource_sharing(resource_id)
        
        if not sharing_info or len(sharing_info) == 1:
            return 1.0  # No sharing, full allocation
        
        # Calculate weight based on usage patterns
        if dimension == 'team':
            # Allocate based on team usage
            team_usage = sharing_info.get(dimension_value, {}).get('usage_percentage', 0)
            return team_usage / 100.0
        
        elif dimension == 'project':
            # Allocate based on project resource consumption
            project_usage = sharing_info.get(dimension_value, {}).get('resource_hours', 0)
            total_usage = sum(info.get('resource_hours', 0) for info in sharing_info.values())
            return project_usage / max(total_usage, 1)
        
        elif dimension == 'model':
            # Allocate based on model training/inference time
            model_time = sharing_info.get(dimension_value, {}).get('computation_time', 0)
            total_time = sum(info.get('computation_time', 0) for info in sharing_info.values())
            return model_time / max(total_time, 1)
        
        else:
            # Equal sharing by default
            return 1.0 / len(sharing_info)
    
    def _perform_cross_dimensional_attribution(self, raw_costs, dimensions):
        """Perform cross-dimensional cost analysis"""
        
        cross_attribution = {}
        
        # Generate all dimension combinations
        from itertools import combinations
        
        for r in range(2, min(len(dimensions) + 1, 4)):  # Limit to avoid explosion
            for dim_combo in combinations(dimensions, r):
                combo_key = '_'.join(dim_combo)
                cross_attribution[combo_key] = {}
                
                for resource_id, cost_data in raw_costs.items():
                    # Get dimension values for this resource
                    dim_values = []
                    for dim in dim_combo:
                        dim_value = self._get_dimension_value(resource_id, dim)
                        dim_values.append(f"{dim}:{dim_value}")
                    
                    combo_value = '|'.join(dim_values)
                    
                    if combo_value not in cross_attribution[combo_key]:
                        cross_attribution[combo_key][combo_value] = {
                            'cost': 0,
                            'resource_count': 0
                        }
                    
                    cross_attribution[combo_key][combo_value]['cost'] += cost_data['total_cost']
                    cross_attribution[combo_key][combo_value]['resource_count'] += 1
        
        return cross_attribution
    
    def _calculate_efficiency_metrics(self, attributed_costs):
        """Calculate cost efficiency metrics"""
        
        efficiency_metrics = {}
        
        # Calculate efficiency by team
        if 'team' in attributed_costs:
            team_efficiency = {}
            for team, team_data in attributed_costs['team'].items():
                # Get business metrics for team
                business_metrics = self.metadata_store.get_team_business_metrics(team)
                
                if business_metrics:
                    team_efficiency[team] = {
                        'cost_per_model': team_data['total_cost'] / max(business_metrics.get('models_deployed', 1), 1),
                        'cost_per_prediction': team_data['total_cost'] / max(business_metrics.get('predictions_served', 1), 1),
                        'cost_per_experiment': team_data['total_cost'] / max(business_metrics.get('experiments_run', 1), 1),
                        'roi_score': business_metrics.get('business_value', 0) / max(team_data['total_cost'], 1)
                    }
            
            efficiency_metrics['team_efficiency'] = team_efficiency
        
        # Calculate efficiency by workload type
        if 'workload_type' in attributed_costs:
            workload_efficiency = {}
            for workload_type, workload_data in attributed_costs['workload_type'].items():
                workload_metrics = self.metadata_store.get_workload_metrics(workload_type)
                
                if workload_metrics:
                    if workload_type == 'ml_training':
                        workload_efficiency[workload_type] = {
                            'cost_per_training_job': workload_data['total_cost'] / max(workload_metrics.get('training_jobs', 1), 1),
                            'cost_per_model_accuracy_point': workload_data['total_cost'] / max(workload_metrics.get('accuracy_improvement', 0.01), 0.01)
                        }
                    elif workload_type == 'ml_inference':
                        workload_efficiency[workload_type] = {
                            'cost_per_prediction': workload_data['total_cost'] / max(workload_metrics.get('predictions', 1), 1),
                            'cost_per_latency_ms': workload_data['total_cost'] / max(workload_metrics.get('avg_latency_ms', 1), 1)
                        }
            
            efficiency_metrics['workload_efficiency'] = workload_efficiency
        
        return efficiency_metrics

ML Cost Optimization Engine:
class MLCostOptimizer:
    def __init__(self, cost_attributor, resource_manager):
        self.cost_attributor = cost_attributor
        self.resource_manager = resource_manager
        self.optimization_strategies = self._load_optimization_strategies()
    
    def generate_optimization_recommendations(self, cost_analysis, business_constraints=None):
        """Generate comprehensive cost optimization recommendations"""
        
        recommendations = []
        
        # Analyze current cost patterns
        cost_patterns = self._analyze_cost_patterns(cost_analysis)
        
        # Generate right-sizing recommendations
        recommendations.extend(self._generate_rightsizing_recommendations(cost_patterns))
        
        # Generate scheduling optimizations
        recommendations.extend(self._generate_scheduling_optimizations(cost_patterns))
        
        # Generate resource pooling recommendations
        recommendations.extend(self._generate_pooling_recommendations(cost_patterns))
        
        # Generate architecture optimizations
        recommendations.extend(self._generate_architecture_optimizations(cost_patterns))
        
        # Generate ML-specific optimizations
        recommendations.extend(self._generate_ml_optimizations(cost_patterns))
        
        # Filter and rank recommendations
        recommendations = self._filter_and_rank_recommendations(
            recommendations, business_constraints
        )
        
        return recommendations
    
    def _generate_rightsizing_recommendations(self, cost_patterns):
        """Generate right-sizing recommendations"""
        
        recommendations = []
        
        # Analyze resource utilization
        for resource_type, utilization_data in cost_patterns['utilization'].items():
            for resource_id, util_info in utilization_data.items():
                
                avg_utilization = util_info['avg_utilization']
                peak_utilization = util_info['peak_utilization']
                cost = util_info['cost']
                
                # Under-utilized resources
                if avg_utilization < 0.3 and peak_utilization < 0.6:
                    potential_savings = cost * (1 - avg_utilization * 1.5)  # Conservative estimate
                    
                    recommendations.append({
                        'type': 'rightsizing',
                        'subtype': 'downsize',
                        'resource_id': resource_id,
                        'resource_type': resource_type,
                        'current_utilization': avg_utilization,
                        'recommended_action': 'Reduce resource allocation',
                        'potential_savings': potential_savings,
                        'risk_level': 'low' if peak_utilization < 0.4 else 'medium',
                        'implementation_complexity': 'low'
                    })
                
                # Over-utilized resources (potential performance issues)
                elif avg_utilization > 0.8 or peak_utilization > 0.95:
                    additional_cost = cost * 0.5  # Estimate for upsizing
                    
                    recommendations.append({
                        'type': 'rightsizing',
                        'subtype': 'upsize',
                        'resource_id': resource_id,
                        'resource_type': resource_type,
                        'current_utilization': avg_utilization,
                        'recommended_action': 'Increase resource allocation',
                        'additional_cost': additional_cost,
                        'performance_benefit': 'Reduced latency and improved reliability',
                        'risk_level': 'low',
                        'implementation_complexity': 'low'
                    })
        
        return recommendations
    
    def _generate_ml_optimizations(self, cost_patterns):
        """Generate ML-specific cost optimizations"""
        
        recommendations = []
        
        # Model optimization recommendations
        model_costs = cost_patterns.get('model_costs', {})
        
        for model_name, model_cost_data in model_costs.items():
            # Check for expensive models with low utilization
            if (model_cost_data['cost_per_prediction'] > 0.01 and 
                model_cost_data['daily_predictions'] < 1000):
                
                recommendations.append({
                    'type': 'ml_optimization',
                    'subtype': 'model_optimization',
                    'model_name': model_name,
                    'recommended_action': 'Consider model quantization or distillation',
                    'current_cost_per_prediction': model_cost_data['cost_per_prediction'],
                    'potential_savings': model_cost_data['total_cost'] * 0.3,
                    'accuracy_trade_off': 'Minimal (typically <2% accuracy loss)',
                    'implementation_complexity': 'medium'
                })
            
            # Check for training cost optimization opportunities
            training_cost = model_cost_data.get('training_cost', 0)
            if training_cost > 1000:  # Expensive training
                
                recommendations.append({
                    'type': 'ml_optimization',
                    'subtype': 'training_optimization',
                    'model_name': model_name,
                    'recommended_action': 'Implement early stopping and learning rate scheduling',
                    'current_training_cost': training_cost,
                    'potential_savings': training_cost * 0.2,
                    'accuracy_impact': 'Potentially improved through better convergence',
                    'implementation_complexity': 'low'
                })
        
        # Batch processing optimizations
        batch_costs = cost_patterns.get('batch_processing_costs', {})
        
        for pipeline_name, pipeline_data in batch_costs.items():
            if pipeline_data['avg_utilization'] < 0.5:
                recommendations.append({
                    'type': 'ml_optimization',
                    'subtype': 'batch_optimization',
                    'pipeline_name': pipeline_name,
                    'recommended_action': 'Implement dynamic batching and resource scaling',
                    'current_utilization': pipeline_data['avg_utilization'],
                    'potential_savings': pipeline_data['cost'] * 0.4,
                    'implementation_complexity': 'medium'
                })
        
        return recommendations
    
    def implement_optimization(self, recommendation):
        """Implement a specific optimization recommendation"""
        
        implementation_result = {
            'recommendation_id': recommendation.get('id'),
            'status': 'pending',
            'implementation_steps': [],
            'estimated_timeline': '',
            'rollback_plan': ''
        }
        
        if recommendation['type'] == 'rightsizing':
            implementation_result = self._implement_rightsizing(recommendation)
        
        elif recommendation['type'] == 'scheduling':
            implementation_result = self._implement_scheduling_optimization(recommendation)
        
        elif recommendation['type'] == 'ml_optimization':
            implementation_result = self._implement_ml_optimization(recommendation)
        
        return implementation_result
    
    def _implement_rightsizing(self, recommendation):
        """Implement right-sizing recommendation"""
        
        resource_id = recommendation['resource_id']
        action = recommendation['subtype']
        
        implementation_steps = []
        
        if action == 'downsize':
            # Create implementation plan for downsizing
            implementation_steps = [
                f"1. Analyze current workload patterns for {resource_id}",
                f"2. Calculate optimal resource allocation",
                f"3. Schedule maintenance window",
                f"4. Update resource configuration",
                f"5. Monitor performance post-change",
                f"6. Validate cost savings"
            ]
            
            # Execute downsizing through resource manager
            try:
                self.resource_manager.scale_resource(
                    resource_id=resource_id,
                    scale_factor=0.7,  # Conservative 30% reduction
                    dry_run=False
                )
                
                status = 'in_progress'
                
            except Exception as e:
                status = 'failed'
                implementation_steps.append(f"Error: {str(e)}")
        
        return {
            'status': status,
            'implementation_steps': implementation_steps,
            'estimated_timeline': '2-24 hours',
            'rollback_plan': 'Scale resource back to original configuration if performance degrades'
        }

Automated Cost Anomaly Detection:
class MLCostAnomalyDetector:
    def __init__(self, cost_data_source):
        self.cost_data_source = cost_data_source
        self.anomaly_models = self._initialize_anomaly_models()
        self.baseline_costs = self._calculate_baseline_costs()
    
    def detect_cost_anomalies(self, detection_window='24h'):
        """Detect cost anomalies in ML infrastructure"""
        
        # Get current cost data
        current_costs = self.cost_data_source.get_costs(detection_window)
        
        anomalies = []
        
        # Check for absolute cost anomalies
        absolute_anomalies = self._detect_absolute_anomalies(current_costs)
        anomalies.extend(absolute_anomalies)
        
        # Check for relative cost anomalies
        relative_anomalies = self._detect_relative_anomalies(current_costs)
        anomalies.extend(relative_anomalies)
        
        # Check for pattern anomalies
        pattern_anomalies = self._detect_pattern_anomalies(current_costs)
        anomalies.extend(pattern_anomalies)
        
        # Check for ML-specific anomalies
        ml_anomalies = self._detect_ml_cost_anomalies(current_costs)
        anomalies.extend(ml_anomalies)
        
        # Rank anomalies by severity
        anomalies = self._rank_anomalies_by_severity(anomalies)
        
        return anomalies
    
    def _detect_ml_cost_anomalies(self, current_costs):
        """Detect ML-specific cost anomalies"""
        
        anomalies = []
        
        # Training cost spikes
        training_costs = [c for c in current_costs if c.get('workload_type') == 'ml_training']
        
        for cost_entry in training_costs:
            model_name = cost_entry.get('model_name')
            current_cost = cost_entry['cost']
            
            # Compare with historical training costs for this model
            historical_cost = self.baseline_costs.get('training', {}).get(model_name, 0)
            
            if current_cost > historical_cost * 3:  # 3x increase
                anomalies.append({
                    'type': 'training_cost_spike',
                    'severity': 'high',
                    'model_name': model_name,
                    'current_cost': current_cost,
                    'expected_cost': historical_cost,
                    'cost_increase': current_cost - historical_cost,
                    'possible_causes': [
                        'Hyperparameter tuning with expensive configurations',
                        'Data size increase',
                        'Model architecture change',
                        'Infrastructure misconfiguration'
                    ]
                })
        
        # Inference cost per prediction anomalies
        inference_costs = [c for c in current_costs if c.get('workload_type') == 'ml_inference']
        
        for cost_entry in inference_costs:
            model_name = cost_entry.get('model_name')
            cost_per_prediction = cost_entry.get('cost_per_prediction', 0)
            
            baseline_cpp = self.baseline_costs.get('inference', {}).get(model_name, {}).get('cost_per_prediction', 0)
            
            if cost_per_prediction > baseline_cpp * 2:  # 2x increase in cost per prediction
                anomalies.append({
                    'type': 'inference_efficiency_degradation',
                    'severity': 'medium',
                    'model_name': model_name,
                    'current_cpp': cost_per_prediction,
                    'baseline_cpp': baseline_cpp,
                    'efficiency_degradation': (cost_per_prediction - baseline_cpp) / baseline_cpp,
                    'possible_causes': [
                        'Model complexity increase',
                        'Resource over-provisioning',
                        'Inefficient batching',
                        'Infrastructure performance issues'
                    ]
                })
        
        return anomalies
    
    def _calculate_severity_score(self, anomaly):
        """Calculate severity score for anomaly prioritization"""
        
        base_score = 0
        
        # Cost impact scoring
        cost_impact = anomaly.get('cost_increase', 0)
        if cost_impact > 10000:  # $10K+ impact
            base_score += 8
        elif cost_impact > 1000:  # $1K+ impact
            base_score += 5
        elif cost_impact > 100:  # $100+ impact
            base_score += 2
        
        # Anomaly type scoring
        if anomaly['type'] in ['training_cost_spike', 'runaway_costs']:
            base_score += 3
        elif anomaly['type'] in ['inference_efficiency_degradation']:
            base_score += 2
        
        # Business impact scoring
        if anomaly.get('affects_production', False):
            base_score += 3
        
        return min(base_score, 10)  # Cap at 10
```

This comprehensive framework for ML cost optimization and resource efficiency provides the theoretical foundations and practical strategies for implementing sophisticated cost management across machine learning infrastructure. The key insight is that ML systems require specialized cost optimization approaches that balance performance, accuracy, and cost while considering the unique characteristics of training, inference, and data processing workloads.