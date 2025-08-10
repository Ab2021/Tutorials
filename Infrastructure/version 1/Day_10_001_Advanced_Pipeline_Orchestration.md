# Day 10.1: Advanced Pipeline Orchestration

## 🔄 Advanced MLOps & Unified Pipelines - Part 1

**Focus**: Complex Workflow Management, Multi-Stage Dependencies, Dynamic Pipeline Configuration  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master advanced pipeline orchestration patterns for complex ML workflows
- Learn dynamic pipeline configuration and conditional execution strategies
- Understand multi-tenant pipeline management and resource allocation
- Analyze pipeline optimization techniques for performance and cost efficiency

---

## 🔄 Advanced Pipeline Orchestration Theory

### **Pipeline Architecture Patterns**

Modern ML systems require sophisticated pipeline orchestration that can handle complex dependencies, dynamic resource allocation, and multi-stage workflows with conditional execution paths.

**Pipeline Orchestration Taxonomy:**
```
ML Pipeline Classification:
1. Linear Pipelines:
   - Sequential execution stages
   - Simple dependency chains
   - Predictable resource requirements
   - Limited flexibility and parallelism

2. DAG-Based Pipelines:
   - Directed Acyclic Graph execution
   - Complex dependency management
   - Parallel execution opportunities
   - Dynamic resource allocation

3. Event-Driven Pipelines:
   - Reactive execution patterns
   - Asynchronous processing
   - Real-time adaptation
   - Stream processing integration

4. Hybrid Pipelines:
   - Combination of batch and streaming
   - Multi-modal processing
   - Adaptive execution strategies
   - Cross-system orchestration

Pipeline Complexity Mathematical Model:
Complexity_Score = Node_Count × Dependency_Density × Conditional_Branches × Resource_Variability

Where:
- Node_Count: Number of pipeline stages
- Dependency_Density: (Edge_Count / (Node_Count * (Node_Count - 1)))
- Conditional_Branches: Number of conditional execution paths
- Resource_Variability: Standard deviation of resource requirements

Pipeline Efficiency Optimization:
Execution_Time = max(Critical_Path_Duration, Resource_Constraint_Duration)
Cost_Efficiency = Business_Value / (Compute_Cost + Storage_Cost + Network_Cost)
Resource_Utilization = Actual_Usage / Provisioned_Capacity

Pipeline Reliability Model:
Pipeline_Success_Rate = ∏(Stage_Success_Rate_i) × Retry_Success_Factor
Mean_Time_To_Recovery = Σ(Failure_Rate_i × Recovery_Time_i)
```

**Dynamic Pipeline Configuration:**
```
Dynamic Pipeline Configuration System:
class DynamicPipelineOrchestrator:
    def __init__(self):
        self.pipeline_templates = {}
        self.execution_context = ExecutionContext()
        self.resource_manager = ResourceManager()
        self.dependency_resolver = DependencyResolver()
        self.optimization_engine = PipelineOptimizer()
    
    def create_dynamic_pipeline(self, base_template, runtime_parameters):
        """Create dynamically configured pipeline based on runtime parameters"""
        
        # Start with base template
        pipeline_config = deepcopy(base_template)
        
        # Apply runtime parameter transformations
        pipeline_config = self._apply_parameter_transformations(
            pipeline_config, runtime_parameters
        )
        
        # Perform conditional stage inclusion/exclusion
        pipeline_config = self._apply_conditional_logic(
            pipeline_config, runtime_parameters
        )
        
        # Optimize pipeline structure
        pipeline_config = self.optimization_engine.optimize_pipeline(
            pipeline_config, runtime_parameters
        )
        
        # Resolve resource requirements
        pipeline_config = self._resolve_resource_requirements(
            pipeline_config, runtime_parameters
        )
        
        # Validate pipeline configuration
        validation_result = self._validate_pipeline_config(pipeline_config)
        if not validation_result.valid:
            raise PipelineConfigurationError(validation_result.errors)
        
        return pipeline_config
    
    def _apply_parameter_transformations(self, config, parameters):
        """Apply parameter-based transformations to pipeline configuration"""
        
        # Data size based transformations
        if 'data_size' in parameters:
            data_size = parameters['data_size']
            
            # Adjust batch sizes based on data volume
            for stage in config.get('stages', []):
                if 'batch_size' in stage.get('parameters', {}):
                    optimal_batch_size = self._calculate_optimal_batch_size(
                        data_size, stage['resource_requirements']
                    )
                    stage['parameters']['batch_size'] = optimal_batch_size
                
                # Adjust parallelism based on data size
                if data_size > 1e9:  # > 1GB
                    stage['parallelism'] = max(stage.get('parallelism', 1), 4)
                elif data_size > 1e6:  # > 1MB
                    stage['parallelism'] = max(stage.get('parallelism', 1), 2)
        
        # Model complexity based transformations
        if 'model_complexity' in parameters:
            complexity = parameters['model_complexity']
            
            for stage in config.get('stages', []):
                if stage.get('type') == 'training':
                    # Adjust resource allocation based on model complexity
                    if complexity == 'high':
                        stage['resource_requirements']['gpu'] = max(
                            stage['resource_requirements'].get('gpu', 0), 2
                        )
                        stage['resource_requirements']['memory'] = max(
                            stage['resource_requirements'].get('memory', '8Gi'), '16Gi'
                        )
                    elif complexity == 'medium':
                        stage['resource_requirements']['gpu'] = max(
                            stage['resource_requirements'].get('gpu', 0), 1
                        )
        
        # Performance requirements based transformations
        if 'latency_requirement' in parameters:
            latency_req = parameters['latency_requirement']
            
            for stage in config.get('stages', []):
                if stage.get('type') == 'inference':
                    if latency_req == 'ultra_low':
                        # Enable aggressive optimizations
                        stage['optimizations'] = stage.get('optimizations', [])
                        stage['optimizations'].extend([
                            'model_quantization',
                            'batch_processing',
                            'cache_warming'
                        ])
                        
                        # Allocate more resources
                        stage['resource_requirements']['cpu'] = '4'
                        stage['resource_requirements']['memory'] = '8Gi'
        
        return config
    
    def _apply_conditional_logic(self, config, parameters):
        """Apply conditional logic for stage inclusion/exclusion"""
        
        filtered_stages = []
        
        for stage in config.get('stages', []):
            # Check stage conditions
            include_stage = True
            
            if 'conditions' in stage:
                for condition in stage['conditions']:
                    if not self._evaluate_condition(condition, parameters):
                        include_stage = False
                        break
            
            # Apply global conditions
            if include_stage:
                # Skip expensive preprocessing if data is already processed
                if (stage.get('type') == 'preprocessing' and 
                    parameters.get('data_preprocessed', False)):
                    continue
                
                # Skip validation if in production mode
                if (stage.get('type') == 'validation' and 
                    parameters.get('environment') == 'production'):
                    continue
                
                # Include feature engineering only for training
                if (stage.get('type') == 'feature_engineering' and 
                    parameters.get('pipeline_mode') != 'training'):
                    continue
            
            if include_stage:
                filtered_stages.append(stage)
        
        config['stages'] = filtered_stages
        return config
    
    def _evaluate_condition(self, condition, parameters):
        """Evaluate a single condition against parameters"""
        
        condition_type = condition.get('type')
        
        if condition_type == 'parameter_equals':
            param_name = condition['parameter']
            expected_value = condition['value']
            return parameters.get(param_name) == expected_value
        
        elif condition_type == 'parameter_greater_than':
            param_name = condition['parameter']
            threshold = condition['threshold']
            param_value = parameters.get(param_name, 0)
            return param_value > threshold
        
        elif condition_type == 'parameter_in_list':
            param_name = condition['parameter']
            allowed_values = condition['allowed_values']
            return parameters.get(param_name) in allowed_values
        
        elif condition_type == 'expression':
            # Evaluate custom expression
            expression = condition['expression']
            try:
                # Safe evaluation with limited scope
                safe_dict = {k: v for k, v in parameters.items() if isinstance(v, (int, float, str, bool))}
                return eval(expression, {"__builtins__": {}}, safe_dict)
            except:
                return False
        
        return True
    
    def _resolve_resource_requirements(self, config, parameters):
        """Resolve dynamic resource requirements"""
        
        for stage in config.get('stages', []):
            # Calculate dynamic resource requirements
            base_requirements = stage.get('resource_requirements', {})
            
            # Factor in data size
            data_size = parameters.get('data_size', 0)
            if data_size > 0:
                # Scale memory requirements based on data size
                base_memory = self._parse_memory_string(base_requirements.get('memory', '2Gi'))
                scaled_memory = max(base_memory, data_size * 0.1)  # 10% of data size
                stage['resource_requirements']['memory'] = f"{int(scaled_memory)}Mi"
                
                # Scale CPU requirements for large datasets
                if data_size > 1e9:  # > 1GB
                    base_cpu = float(base_requirements.get('cpu', '1'))
                    scaled_cpu = min(base_cpu * 2, 16)  # Cap at 16 CPUs
                    stage['resource_requirements']['cpu'] = str(scaled_cpu)
            
            # Factor in performance requirements
            if parameters.get('priority') == 'high':
                # Increase resource allocation for high priority jobs
                current_cpu = float(stage['resource_requirements'].get('cpu', '1'))
                current_memory = self._parse_memory_string(stage['resource_requirements'].get('memory', '2Gi'))
                
                stage['resource_requirements']['cpu'] = str(current_cpu * 1.5)
                stage['resource_requirements']['memory'] = f"{int(current_memory * 1.5)}Mi"
        
        return config

Multi-Tenant Pipeline Management:
class MultiTenantPipelineManager:
    def __init__(self):
        self.tenant_configs = {}
        self.resource_quotas = {}
        self.isolation_policies = {}
        self.scheduling_policies = {}
        
    def submit_tenant_pipeline(self, tenant_id, pipeline_config, execution_context):
        """Submit pipeline for multi-tenant execution"""
        
        # Validate tenant permissions
        self._validate_tenant_permissions(tenant_id, pipeline_config)
        
        # Apply tenant-specific configurations
        tenant_pipeline = self._apply_tenant_config(tenant_id, pipeline_config)
        
        # Apply resource quotas and limits
        tenant_pipeline = self._apply_resource_quotas(tenant_id, tenant_pipeline)
        
        # Apply isolation policies
        tenant_pipeline = self._apply_isolation_policies(tenant_id, tenant_pipeline)
        
        # Schedule pipeline execution
        execution_plan = self._create_tenant_execution_plan(tenant_id, tenant_pipeline)
        
        # Submit for execution
        pipeline_id = self._submit_for_execution(tenant_pipeline, execution_plan)
        
        return {
            'pipeline_id': pipeline_id,
            'tenant_id': tenant_id,
            'execution_plan': execution_plan,
            'estimated_completion': self._estimate_completion_time(tenant_pipeline),
            'resource_allocation': self._get_resource_allocation(tenant_pipeline)
        }
    
    def _apply_tenant_config(self, tenant_id, pipeline_config):
        """Apply tenant-specific configuration overrides"""
        
        tenant_config = self.tenant_configs.get(tenant_id, {})
        
        # Apply tenant-specific defaults
        for stage in pipeline_config.get('stages', []):
            # Apply tenant-specific resource defaults
            if 'resource_defaults' in tenant_config:
                defaults = tenant_config['resource_defaults']
                stage_requirements = stage.get('resource_requirements', {})
                
                for resource, default_value in defaults.items():
                    if resource not in stage_requirements:
                        stage_requirements[resource] = default_value
                
                stage['resource_requirements'] = stage_requirements
            
            # Apply tenant-specific environment variables
            if 'environment_variables' in tenant_config:
                stage_env = stage.get('environment', {})
                stage_env.update(tenant_config['environment_variables'])
                stage['environment'] = stage_env
            
            # Apply tenant-specific secrets and config maps
            if 'secrets' in tenant_config:
                stage['secrets'] = stage.get('secrets', []) + tenant_config['secrets']
            
            if 'config_maps' in tenant_config:
                stage['config_maps'] = stage.get('config_maps', []) + tenant_config['config_maps']
        
        return pipeline_config
    
    def _apply_resource_quotas(self, tenant_id, pipeline_config):
        """Apply tenant resource quotas and limits"""
        
        quota = self.resource_quotas.get(tenant_id, {})
        
        if not quota:
            return pipeline_config
        
        # Calculate total pipeline resource requirements
        total_requirements = self._calculate_total_resources(pipeline_config)
        
        # Check against quotas
        for resource_type, requirement in total_requirements.items():
            quota_limit = quota.get(resource_type, float('inf'))
            
            if requirement > quota_limit:
                # Scale down resource requirements proportionally
                scale_factor = quota_limit / requirement
                
                for stage in pipeline_config.get('stages', []):
                    stage_req = stage.get('resource_requirements', {})
                    if resource_type in stage_req:
                        if resource_type in ['cpu']:
                            current_value = float(stage_req[resource_type])
                            stage_req[resource_type] = str(current_value * scale_factor)
                        elif resource_type in ['memory']:
                            current_value = self._parse_memory_string(stage_req[resource_type])
                            stage_req[resource_type] = f"{int(current_value * scale_factor)}Mi"
                        elif resource_type in ['gpu']:
                            current_value = int(stage_req[resource_type])
                            stage_req[resource_type] = max(1, int(current_value * scale_factor))
        
        return pipeline_config
    
    def _apply_isolation_policies(self, tenant_id, pipeline_config):
        """Apply tenant isolation policies"""
        
        isolation_policy = self.isolation_policies.get(tenant_id, {})
        
        # Apply namespace isolation
        if 'namespace_prefix' in isolation_policy:
            namespace_prefix = isolation_policy['namespace_prefix']
            
            for stage in pipeline_config.get('stages', []):
                stage['namespace'] = f"{namespace_prefix}-{stage.get('name', 'stage')}"
        
        # Apply network isolation
        if 'network_policy' in isolation_policy:
            network_policy = isolation_policy['network_policy']
            
            for stage in pipeline_config.get('stages', []):
                stage['network_policy'] = network_policy
        
        # Apply security context
        if 'security_context' in isolation_policy:
            security_context = isolation_policy['security_context']
            
            for stage in pipeline_config.get('stages', []):
                stage['security_context'] = security_context
        
        # Apply resource tagging
        if 'resource_tags' in isolation_policy:
            resource_tags = isolation_policy['resource_tags']
            
            for stage in pipeline_config.get('stages', []):
                stage_labels = stage.get('labels', {})
                stage_labels.update(resource_tags)
                stage['labels'] = stage_labels
        
        return pipeline_config
```

---

## 🚀 Advanced Workflow Engines

### **Kubeflow Pipelines Advanced Patterns**

**Complex Kubeflow Pipeline Implementation:**
```
Advanced Kubeflow Pipeline Patterns:
import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from typing import NamedTuple, Dict, Any
import numpy as np

# Advanced data preprocessing component with dynamic configuration
@create_component_from_func
def advanced_data_preprocessing(
    input_data_path: str,
    preprocessing_config: Dict[str, Any],
    output_data_path: str,
    quality_report_path: str
) -> NamedTuple('Outputs', [('processed_data_path', str), ('quality_metrics', Dict[str, float])]):
    
    import pandas as pd
    import numpy as np
    import json
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    
    # Load data
    data = pd.read_csv(input_data_path)
    
    quality_metrics = {}
    
    # Apply preprocessing steps based on configuration
    if preprocessing_config.get('handle_missing_values', True):
        missing_strategy = preprocessing_config.get('missing_strategy', 'mean')
        
        if missing_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=preprocessing_config.get('knn_neighbors', 5))
        else:
            imputer = SimpleImputer(strategy=missing_strategy)
        
        # Track missing value statistics
        missing_before = data.isnull().sum().sum()
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        missing_after = data.isnull().sum().sum()
        quality_metrics['missing_values_imputed'] = missing_before - missing_after
    
    # Apply scaling based on configuration
    scaling_method = preprocessing_config.get('scaling_method', 'standard')
    if scaling_method:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        quality_metrics['scaling_method'] = scaling_method
    
    # Feature engineering based on configuration
    if preprocessing_config.get('create_interaction_features', False):
        # Create interaction features for top correlated features
        correlation_threshold = preprocessing_config.get('correlation_threshold', 0.8)
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr().abs()
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > correlation_threshold:
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                    high_corr_pairs.append((col1, col2))
        
        quality_metrics['interaction_features_created'] = len(high_corr_pairs)
    
    # Outlier detection and handling
    if preprocessing_config.get('handle_outliers', False):
        outlier_method = preprocessing_config.get('outlier_method', 'iqr')
        outlier_threshold = preprocessing_config.get('outlier_threshold', 1.5)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for column in numeric_columns:
            if outlier_method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
                outliers_removed += outlier_mask.sum()
                
                # Cap outliers instead of removing
                data.loc[data[column] < lower_bound, column] = lower_bound
                data.loc[data[column] > upper_bound, column] = upper_bound
        
        quality_metrics['outliers_capped'] = outliers_removed
    
    # Calculate final quality metrics
    quality_metrics.update({
        'final_row_count': len(data),
        'final_column_count': len(data.columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
        'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(data.select_dtypes(include=['object']).columns)
    })
    
    # Save processed data and quality report
    data.to_csv(output_data_path, index=False)
    
    with open(quality_report_path, 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    return (output_data_path, quality_metrics)

# Dynamic model training component with hyperparameter optimization
@create_component_from_func
def dynamic_model_training(
    training_data_path: str,
    model_config: Dict[str, Any],
    hyperparameter_space: Dict[str, Any],
    model_output_path: str,
    metrics_output_path: str
) -> NamedTuple('Outputs', [('model_path', str), ('best_metrics', Dict[str, float]), ('optimization_history', str)]):
    
    import pandas as pd
    import numpy as np
    import json
    import joblib
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import RandomizedSearchCV
    import optuna
    
    # Load training data
    data = pd.read_csv(training_data_path)
    
    # Prepare features and target
    target_column = model_config['target_column']
    feature_columns = [col for col in data.columns if col != target_column]
    
    X = data[feature_columns]
    y = data[target_column]
    
    # Split data
    test_size = model_config.get('test_size', 0.2)
    random_state = model_config.get('random_state', 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Define model types
    model_types = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'svm': SVC,
        'logistic_regression': LogisticRegression
    }
    
    best_model = None
    best_score = 0
    optimization_history = []
    
    # Perform hyperparameter optimization for each model type
    for model_name in model_config.get('model_types', ['random_forest']):
        print(f"Optimizing {model_name}...")
        
        if model_name not in model_types:
            continue
        
        model_class = model_types[model_name]
        param_space = hyperparameter_space.get(model_name, {})
        
        if model_config.get('optimization_method', 'random_search') == 'optuna':
            # Use Optuna for more advanced optimization
            
            def objective(trial):
                # Define parameters based on model type
                if model_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': random_state
                    }
                elif model_name == 'gradient_boosting':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'random_state': random_state
                    }
                # Add other model parameter definitions...
                
                model = model_class(**params)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                return cv_scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=model_config.get('n_trials', 50))
            
            # Train best model
            best_params = study.best_params
            model = model_class(**best_params, random_state=random_state)
            
        else:
            # Use RandomizedSearchCV
            model = model_class(random_state=random_state)
            
            if param_space:
                search = RandomizedSearchCV(
                    model, 
                    param_space, 
                    n_iter=model_config.get('n_trials', 50),
                    cv=5,
                    scoring='accuracy',
                    random_state=random_state
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                best_params = search.best_params_
            else:
                model.fit(X_train, y_train)
                best_params = model.get_params()
        
        # Evaluate model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_metrics = {
            'model_type': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'best_parameters': best_params
        }
        
        optimization_history.append(model_metrics)
        
        # Track best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_metrics = model_metrics
    
    # Save best model
    joblib.dump(best_model, model_output_path)
    
    # Save metrics and optimization history
    with open(metrics_output_path, 'w') as f:
        json.dump({
            'best_metrics': best_metrics,
            'optimization_history': optimization_history
        }, f, indent=2)
    
    return (model_output_path, best_metrics, json.dumps(optimization_history))

# Conditional pipeline component for A/B testing
@create_component_from_func
def ab_test_decision(
    model_a_metrics: Dict[str, float],
    model_b_metrics: Dict[str, float],
    statistical_threshold: float = 0.05,
    min_improvement_threshold: float = 0.02
) -> NamedTuple('Outputs', [('selected_model', str), ('confidence_score', float), ('test_results', Dict[str, Any])]):
    
    import numpy as np
    from scipy import stats
    import json
    
    # Extract key metrics
    accuracy_a = model_a_metrics.get('accuracy', 0)
    accuracy_b = model_b_metrics.get('accuracy', 0)
    
    # Simulate statistical test (in real scenario, you'd have actual prediction data)
    # This is a simplified example
    n_samples = 1000  # Assume 1000 test samples
    
    # Simulate prediction accuracies as binomial distributions
    successes_a = int(accuracy_a * n_samples)
    successes_b = int(accuracy_b * n_samples)
    
    # Perform two-proportion z-test
    p1 = successes_a / n_samples
    p2 = successes_b / n_samples
    
    p_combined = (successes_a + successes_b) / (2 * n_samples)
    
    se = np.sqrt(p_combined * (1 - p_combined) * (2 / n_samples))
    
    if se > 0:
        z_score = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    else:
        z_score = 0
        p_value = 1.0
    
    # Decision logic
    improvement = accuracy_b - accuracy_a
    is_significant = p_value < statistical_threshold
    is_meaningful = abs(improvement) >= min_improvement_threshold
    
    if is_significant and is_meaningful and improvement > 0:
        selected_model = 'model_b'
        confidence_score = 1 - p_value
    elif is_significant and is_meaningful and improvement < 0:
        selected_model = 'model_a'
        confidence_score = 1 - p_value
    else:
        # No significant difference, choose based on other criteria
        if accuracy_b >= accuracy_a:
            selected_model = 'model_b'
        else:
            selected_model = 'model_a'
        confidence_score = 0.5
    
    test_results = {
        'accuracy_a': accuracy_a,
        'accuracy_b': accuracy_b,
        'improvement': improvement,
        'z_score': z_score,
        'p_value': p_value,
        'is_significant': is_significant,
        'is_meaningful': is_meaningful,
        'selected_model': selected_model,
        'confidence_score': confidence_score
    }
    
    return (selected_model, confidence_score, test_results)

# Complex pipeline with conditional execution and loops
@dsl.pipeline(
    name='Advanced ML Pipeline with Conditional Execution',
    description='A complex ML pipeline demonstrating advanced patterns'
)
def advanced_ml_pipeline(
    input_data_path: str,
    target_column: str,
    preprocessing_config: Dict[str, Any] = {},
    model_config: Dict[str, Any] = {},
    hyperparameter_space: Dict[str, Any] = {},
    enable_ab_testing: bool = True,
    max_iterations: int = 3
):
    
    # Data preprocessing
    preprocessing_task = advanced_data_preprocessing(
        input_data_path=input_data_path,
        preprocessing_config=preprocessing_config,
        output_data_path='/tmp/processed_data.csv',
        quality_report_path='/tmp/quality_report.json'
    )
    
    # Model training with iteration capability
    with dsl.ParallelFor([{'model_variant': 'a'}, {'model_variant': 'b'}]) as item:
        model_training_task = dynamic_model_training(
            training_data_path=preprocessing_task.outputs['processed_data_path'],
            model_config=model_config,
            hyperparameter_space=hyperparameter_space,
            model_output_path=f'/tmp/model_{item.model_variant}.pkl',
            metrics_output_path=f'/tmp/metrics_{item.model_variant}.json'
        )
    
    # Conditional A/B testing
    with dsl.Condition(enable_ab_testing == True):
        ab_test_task = ab_test_decision(
            model_a_metrics=model_training_task.outputs['best_metrics'],
            model_b_metrics=model_training_task.outputs['best_metrics'],  # This would be different in real scenario
            statistical_threshold=0.05,
            min_improvement_threshold=0.02
        )
        
        # Model deployment based on A/B test results
        with dsl.Condition(ab_test_task.outputs['confidence_score'] > 0.8):
            deploy_model_task = deploy_winning_model(
                selected_model=ab_test_task.outputs['selected_model'],
                model_a_path=f'/tmp/model_a.pkl',
                model_b_path=f'/tmp/model_b.pkl',
                deployment_target='production'
            )
    
    # Fallback deployment if A/B testing is disabled
    with dsl.Condition(enable_ab_testing == False):
        fallback_deploy_task = deploy_single_model(
            model_path='/tmp/model_a.pkl',
            deployment_target='production'
        )

# Helper components for deployment
@create_component_from_func
def deploy_winning_model(
    selected_model: str,
    model_a_path: str,
    model_b_path: str,
    deployment_target: str
) -> str:
    
    import shutil
    
    if selected_model == 'model_a':
        source_path = model_a_path
    else:
        source_path = model_b_path
    
    destination_path = f'/tmp/deployed_model_{deployment_target}.pkl'
    shutil.copy2(source_path, destination_path)
    
    print(f"Deployed {selected_model} to {deployment_target}")
    return destination_path

@create_component_from_func
def deploy_single_model(
    model_path: str,
    deployment_target: str
) -> str:
    
    import shutil
    
    destination_path = f'/tmp/deployed_model_{deployment_target}.pkl'
    shutil.copy2(model_path, destination_path)
    
    print(f"Deployed model to {deployment_target}")
    return destination_path
```

---

## 🎛️ Pipeline Optimization and Performance

### **Intelligent Pipeline Optimization**

**Pipeline Performance Optimizer:**
```
Advanced Pipeline Performance Optimization:
class PipelinePerformanceOptimizer:
    def __init__(self):
        self.execution_history = ExecutionHistoryStore()
        self.resource_monitor = ResourceMonitor()
        self.cost_calculator = CostCalculator()
        self.ml_models = self._initialize_optimization_models()
    
    def optimize_pipeline_execution(self, pipeline_config, constraints=None):
        """Optimize pipeline execution based on historical data and ML predictions"""
        
        # Analyze historical execution patterns
        historical_analysis = self._analyze_execution_history(pipeline_config)
        
        # Predict resource requirements
        resource_predictions = self._predict_resource_requirements(
            pipeline_config, historical_analysis
        )
        
        # Optimize stage ordering
        optimized_order = self._optimize_stage_ordering(
            pipeline_config, resource_predictions
        )
        
        # Optimize parallelization
        parallelization_plan = self._optimize_parallelization(
            pipeline_config, resource_predictions, constraints
        )
        
        # Optimize resource allocation
        resource_allocation = self._optimize_resource_allocation(
            pipeline_config, resource_predictions, constraints
        )
        
        # Generate optimized pipeline configuration
        optimized_config = self._generate_optimized_config(
            pipeline_config,
            optimized_order,
            parallelization_plan,
            resource_allocation
        )
        
        return {
            'optimized_config': optimized_config,
            'predicted_execution_time': self._predict_execution_time(optimized_config),
            'predicted_cost': self._predict_execution_cost(optimized_config),
            'optimization_improvements': self._calculate_improvements(pipeline_config, optimized_config)
        }
    
    def _predict_resource_requirements(self, pipeline_config, historical_analysis):
        """Predict resource requirements using ML models"""
        
        predictions = {}
        
        for stage in pipeline_config.get('stages', []):
            stage_name = stage['name']
            stage_type = stage.get('type', 'unknown')
            
            # Extract features for prediction
            features = self._extract_stage_features(stage, historical_analysis)
            
            # Predict CPU requirements
            cpu_model = self.ml_models.get(f'cpu_prediction_{stage_type}')
            if cpu_model:
                predicted_cpu = cpu_model.predict([features])[0]
            else:
                # Fallback to historical average
                predicted_cpu = historical_analysis.get(stage_name, {}).get('avg_cpu', 1.0)
            
            # Predict memory requirements  
            memory_model = self.ml_models.get(f'memory_prediction_{stage_type}')
            if memory_model:
                predicted_memory = memory_model.predict([features])[0]
            else:
                predicted_memory = historical_analysis.get(stage_name, {}).get('avg_memory', 2048)
            
            # Predict execution time
            time_model = self.ml_models.get(f'time_prediction_{stage_type}')
            if time_model:
                predicted_time = time_model.predict([features])[0]
            else:
                predicted_time = historical_analysis.get(stage_name, {}).get('avg_duration', 300)
            
            predictions[stage_name] = {
                'cpu': predicted_cpu,
                'memory': predicted_memory,  # MB
                'execution_time': predicted_time,  # seconds
                'confidence': self._calculate_prediction_confidence(stage, historical_analysis)
            }
        
        return predictions
    
    def _optimize_stage_ordering(self, pipeline_config, resource_predictions):
        """Optimize stage execution order to minimize total execution time"""
        
        stages = pipeline_config.get('stages', [])
        dependencies = self._extract_dependencies(stages)
        
        # Create dependency graph
        graph = self._build_dependency_graph(stages, dependencies)
        
        # Find optimal ordering using topological sort with resource consideration
        ordered_stages = self._resource_aware_topological_sort(graph, resource_predictions)
        
        return ordered_stages
    
    def _optimize_parallelization(self, pipeline_config, resource_predictions, constraints):
        """Optimize parallel execution opportunities"""
        
        stages = pipeline_config.get('stages', [])
        dependencies = self._extract_dependencies(stages)
        
        # Identify parallelizable stage groups
        parallel_groups = self._identify_parallel_groups(stages, dependencies)
        
        # Calculate optimal parallelism for each group
        parallelization_plan = {}
        
        for group in parallel_groups:
            # Calculate resource constraints
            max_parallel_jobs = self._calculate_max_parallel_jobs(
                group, resource_predictions, constraints
            )
            
            # Determine optimal parallelism level
            optimal_parallelism = self._determine_optimal_parallelism(
                group, resource_predictions, max_parallel_jobs
            )
            
            parallelization_plan[tuple(group)] = {
                'max_parallel': max_parallel_jobs,
                'optimal_parallel': optimal_parallelism,
                'execution_strategy': self._select_execution_strategy(group, optimal_parallelism)
            }
        
        return parallelization_plan
    
    def _optimize_resource_allocation(self, pipeline_config, resource_predictions, constraints):
        """Optimize resource allocation across pipeline stages"""
        
        total_cpu_budget = constraints.get('max_cpu', float('inf'))
        total_memory_budget = constraints.get('max_memory', float('inf'))
        
        stages = pipeline_config.get('stages', [])
        
        # Formulate as optimization problem
        # Minimize execution time subject to resource constraints
        
        allocation = {}
        remaining_cpu = total_cpu_budget
        remaining_memory = total_memory_budget
        
        # Sort stages by resource efficiency (performance per resource unit)
        stage_efficiency = []
        for stage in stages:
            stage_name = stage['name']
            predictions = resource_predictions.get(stage_name, {})
            
            predicted_time = predictions.get('execution_time', 300)
            predicted_cpu = predictions.get('cpu', 1.0)
            predicted_memory = predictions.get('memory', 2048)
            
            # Calculate efficiency (lower time per resource unit is better)
            cpu_efficiency = predicted_time / predicted_cpu
            memory_efficiency = predicted_time / (predicted_memory / 1024)  # Per GB
            
            stage_efficiency.append({
                'stage': stage_name,
                'cpu_efficiency': cpu_efficiency,
                'memory_efficiency': memory_efficiency,
                'predicted_cpu': predicted_cpu,
                'predicted_memory': predicted_memory
            })
        
        # Allocate resources based on efficiency
        for stage_info in sorted(stage_efficiency, key=lambda x: x['cpu_efficiency']):
            stage_name = stage_info['stage']
            
            # Allocate slightly more than predicted to account for variance
            safety_factor = 1.2
            allocated_cpu = min(
                stage_info['predicted_cpu'] * safety_factor,
                remaining_cpu
            )
            allocated_memory = min(
                stage_info['predicted_memory'] * safety_factor,
                remaining_memory
            )
            
            allocation[stage_name] = {
                'cpu': allocated_cpu,
                'memory': allocated_memory,
                'safety_factor': safety_factor
            }
            
            remaining_cpu -= allocated_cpu
            remaining_memory -= allocated_memory
        
        return allocation
    
    def _calculate_max_parallel_jobs(self, stage_group, resource_predictions, constraints):
        """Calculate maximum number of parallel jobs for a stage group"""
        
        if not constraints:
            return len(stage_group)  # No constraints, run all in parallel
        
        max_cpu = constraints.get('max_cpu', float('inf'))
        max_memory = constraints.get('max_memory', float('inf'))
        
        # Calculate resource requirements per job
        max_cpu_per_job = 0
        max_memory_per_job = 0
        
        for stage_name in stage_group:
            predictions = resource_predictions.get(stage_name, {})
            max_cpu_per_job = max(max_cpu_per_job, predictions.get('cpu', 1.0))
            max_memory_per_job = max(max_memory_per_job, predictions.get('memory', 2048))
        
        # Calculate maximum parallel jobs based on resource constraints
        max_parallel_cpu = int(max_cpu / max_cpu_per_job) if max_cpu_per_job > 0 else float('inf')
        max_parallel_memory = int(max_memory / max_memory_per_job) if max_memory_per_job > 0 else float('inf')
        
        return min(len(stage_group), max_parallel_cpu, max_parallel_memory)
    
    def _generate_optimized_config(self, original_config, optimized_order, 
                                 parallelization_plan, resource_allocation):
        """Generate optimized pipeline configuration"""
        
        optimized_config = deepcopy(original_config)
        
        # Apply optimized stage ordering
        stage_map = {stage['name']: stage for stage in optimized_config.get('stages', [])}
        optimized_config['stages'] = [stage_map[name] for name in optimized_order if name in stage_map]
        
        # Apply resource allocation
        for stage in optimized_config['stages']:
            stage_name = stage['name']
            if stage_name in resource_allocation:
                allocation = resource_allocation[stage_name]
                stage['resource_requirements'] = {
                    'cpu': str(allocation['cpu']),
                    'memory': f"{int(allocation['memory'])}Mi"
                }
        
        # Apply parallelization plan
        for stage_group, plan in parallelization_plan.items():
            for stage_name in stage_group:
                stage = stage_map.get(stage_name)
                if stage:
                    stage['parallelism'] = plan['optimal_parallel']
                    stage['execution_strategy'] = plan['execution_strategy']
        
        return optimized_config
```

This comprehensive framework for advanced pipeline orchestration provides the theoretical foundations and practical strategies for implementing sophisticated ML workflow management systems. The key insight is that modern ML pipelines require dynamic configuration, intelligent optimization, and multi-tenant resource management to achieve optimal performance and cost efficiency.