# Day 9.3: Logging & Debugging Strategies for ML Systems

## 🐛 Monitoring, Observability & Debugging - Part 3

**Focus**: Structured Logging, Error Analysis, Root Cause Investigation  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master structured logging patterns for ML systems with complex data flows
- Learn advanced debugging techniques for distributed ML pipelines and model failures
- Understand log aggregation and analysis strategies for performance troubleshooting
- Analyze error patterns and implement automated root cause analysis for ML workloads

---

## 📝 ML Logging Theoretical Framework

### **Structured Logging Architecture**

ML systems require comprehensive logging strategies that capture model behavior, data quality issues, and system performance across distributed components.

**ML Logging Taxonomy:**
```
ML System Logging Categories:
1. Model Behavior Logging:
   - Prediction inputs and outputs
   - Model confidence scores and uncertainty
   - Feature importance and attribution
   - Model drift and performance degradation

2. Data Quality Logging:
   - Input validation and schema compliance
   - Feature distribution and statistical properties
   - Data lineage and transformation history
   - Anomaly detection and outlier identification

3. System Performance Logging:
   - Resource utilization and bottlenecks
   - Service latency and throughput metrics
   - Error rates and failure patterns
   - Scaling events and capacity changes

4. Business Impact Logging:
   - Business metric correlations
   - A/B testing results and statistical significance
   - Cost optimization and ROI tracking
   - Compliance and audit trail requirements

Logging Information Theory:
Log_Value = Information_Content × Business_Impact × Debugging_Utility - Storage_Cost

Optimal_Log_Level = argmax(Expected_Debugging_Value - Log_Processing_Cost)

Log Sampling Strategy:
Sample_Rate = f(Service_Criticality, Error_Rate, Debug_Context, Storage_Budget)

Structured Log Schema:
{
  "timestamp": ISO8601_timestamp,
  "service": service_name,
  "component": component_name,
  "level": log_level,
  "message": human_readable_message,
  "ml_context": {
    "model_name": model_identifier,
    "model_version": version_string,
    "experiment_id": experiment_identifier,
    "prediction_id": unique_prediction_id
  },
  "data_context": {
    "batch_id": batch_identifier,
    "feature_version": feature_schema_version,
    "data_source": data_origin,
    "processing_stage": pipeline_stage
  },
  "performance_context": {
    "latency_ms": operation_duration,
    "memory_usage_mb": memory_consumption,
    "cpu_usage_percent": cpu_utilization,
    "gpu_usage_percent": gpu_utilization
  },
  "business_context": {
    "user_segment": customer_segment,
    "product_area": business_unit,
    "geographic_region": location,
    "request_priority": priority_level
  },
  "trace_context": {
    "trace_id": distributed_trace_id,
    "span_id": current_span_id,
    "parent_span_id": parent_span_id
  }
}
```

**Log Level Strategy for ML Systems:**
```
ML-Specific Log Levels:
1. CRITICAL (50):
   - Model serving failures
   - Data corruption detected
   - Security breaches
   - Business-critical pipeline failures

2. ERROR (40):
   - Prediction errors and exceptions
   - Data validation failures
   - Model loading failures
   - Feature store unavailability

3. WARNING (30):
   - Model performance degradation
   - Data drift detection
   - Resource utilization alerts
   - Non-critical service disruptions

4. INFO (20):
   - Model deployment events
   - Batch processing completion
   - Performance milestone achievements
   - Configuration changes

5. DEBUG (10):
   - Detailed prediction flows
   - Feature transformation steps
   - Internal state changes
   - Performance profiling data

6. TRACE (5) - Custom ML Level:
   - Individual feature values
   - Model internal computations
   - Data lineage tracking
   - Detailed performance metrics

Dynamic Log Level Configuration:
class MLLogLevelManager:
    def __init__(self):
        self.base_levels = {
            'production': logging.WARNING,
            'staging': logging.INFO,
            'development': logging.DEBUG
        }
        self.service_overrides = {}
        self.experiment_overrides = {}
        self.error_rate_thresholds = {
            'high': 0.05,    # Switch to DEBUG when error rate > 5%
            'critical': 0.15  # Switch to TRACE when error rate > 15%
        }
    
    def get_log_level(self, service_name, environment, current_error_rate=0.0, experiment_id=None):
        """Dynamically determine appropriate log level"""
        
        # Start with base level for environment
        base_level = self.base_levels.get(environment, logging.INFO)
        
        # Check for service-specific overrides
        if service_name in self.service_overrides:
            base_level = min(base_level, self.service_overrides[service_name])
        
        # Check for experiment-specific overrides
        if experiment_id and experiment_id in self.experiment_overrides:
            base_level = min(base_level, self.experiment_overrides[experiment_id])
        
        # Adjust based on error rate
        if current_error_rate > self.error_rate_thresholds['critical']:
            return logging.DEBUG  # Detailed logging for critical issues
        elif current_error_rate > self.error_rate_thresholds['high']:
            return max(logging.INFO, base_level)
        
        return base_level
    
    def set_experiment_log_level(self, experiment_id, level, duration_minutes=60):
        """Temporarily increase logging for specific experiments"""
        self.experiment_overrides[experiment_id] = level
        
        # Schedule removal of override (in production, use proper scheduling)
        import threading
        import time
        
        def remove_override():
            time.sleep(duration_minutes * 60)
            if experiment_id in self.experiment_overrides:
                del self.experiment_overrides[experiment_id]
        
        threading.Thread(target=remove_override, daemon=True).start()
```

---

## 🔧 Advanced Logging Implementation

### **Structured Logging with Context Propagation**

**Python ML Logging Framework:**
```
ML-Specific Logging Implementation:
import logging
import json
import time
import threading
import uuid
import numpy as np
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass
class MLContext:
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    experiment_id: Optional[str] = None
    prediction_id: Optional[str] = None
    batch_id: Optional[str] = None
    feature_version: Optional[str] = None

@dataclass
class PerformanceContext:
    latency_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    gpu_usage_percent: Optional[float] = None

@dataclass
class BusinessContext:
    user_segment: Optional[str] = None
    product_area: Optional[str] = None
    geographic_region: Optional[str] = None
    request_priority: Optional[str] = None

class MLLogger:
    """Advanced ML-specific logger with context management"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Configure structured JSON formatter
        formatter = MLJSONFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Context storage (thread-local)
        self._context = threading.local()
        
        # Performance tracking
        self._operation_stack = threading.local()
    
    def set_ml_context(self, ml_context: MLContext):
        """Set ML-specific context for current thread"""
        self._context.ml_context = ml_context
    
    def set_business_context(self, business_context: BusinessContext):
        """Set business context for current thread"""
        self._context.business_context = business_context
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get all current context information"""
        context = {}
        
        if hasattr(self._context, 'ml_context'):
            context['ml_context'] = asdict(self._context.ml_context)
        
        if hasattr(self._context, 'business_context'):
            context['business_context'] = asdict(self._context.business_context)
        
        if hasattr(self._context, 'trace_context'):
            context['trace_context'] = self._context.trace_context
        
        return context
    
    @contextmanager
    def operation_context(self, operation_name: str, **kwargs):
        """Context manager for tracking operation performance"""
        
        # Initialize operation stack if needed
        if not hasattr(self._operation_stack, 'stack'):
            self._operation_stack.stack = []
        
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        operation_info = {
            'operation_id': operation_id,
            'operation_name': operation_name,
            'start_time': start_time,
            'parent_operation': self._operation_stack.stack[-1]['operation_id'] if self._operation_stack.stack else None,
            **kwargs
        }
        
        self._operation_stack.stack.append(operation_info)
        
        try:
            self.info(f"Starting operation: {operation_name}", extra={
                'operation_id': operation_id,
                'operation_type': 'start',
                'operation_name': operation_name,
                **kwargs
            })
            
            yield operation_info
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.error(f"Operation failed: {operation_name}", extra={
                'operation_id': operation_id,
                'operation_type': 'error',
                'operation_name': operation_name,
                'duration_ms': duration,
                'error_type': type(e).__name__,
                'error_message': str(e),
                **kwargs
            })
            raise
            
        finally:
            duration = (time.time() - start_time) * 1000
            self._operation_stack.stack.pop()
            
            self.info(f"Completed operation: {operation_name}", extra={
                'operation_id': operation_id,
                'operation_type': 'complete',
                'operation_name': operation_name,
                'duration_ms': duration,
                **kwargs
            })
    
    def log_prediction(self, model_name: str, model_version: str, 
                      input_data: Any, prediction: Any, confidence: Optional[float] = None,
                      latency_ms: Optional[float] = None, **kwargs):
        """Log ML prediction with comprehensive context"""
        
        prediction_id = str(uuid.uuid4())
        
        log_data = {
            'prediction_id': prediction_id,
            'model_name': model_name,
            'model_version': model_version,
            'event_type': 'prediction',
            **kwargs
        }
        
        # Add input characteristics
        if hasattr(input_data, 'shape'):
            log_data['input_shape'] = list(input_data.shape)
            log_data['input_dtype'] = str(input_data.dtype)
        elif isinstance(input_data, (list, dict)):
            log_data['input_size'] = len(input_data)
            log_data['input_type'] = type(input_data).__name__
        
        # Add prediction characteristics
        if hasattr(prediction, 'shape'):
            log_data['prediction_shape'] = list(prediction.shape)
            log_data['prediction_dtype'] = str(prediction.dtype)
        elif isinstance(prediction, (list, dict)):
            log_data['prediction_size'] = len(prediction)
            log_data['prediction_type'] = type(prediction).__name__
        
        # Add confidence if available
        if confidence is not None:
            log_data['confidence'] = float(confidence)
        
        # Add performance metrics
        if latency_ms is not None:
            log_data['latency_ms'] = latency_ms
        
        self.info("Model prediction completed", extra=log_data)
        return prediction_id
    
    def log_data_quality_check(self, dataset_name: str, check_type: str, 
                              result: Dict[str, Any], **kwargs):
        """Log data quality check results"""
        
        log_data = {
            'dataset_name': dataset_name,
            'check_type': check_type,
            'event_type': 'data_quality_check',
            'quality_result': result,
            **kwargs
        }
        
        # Determine log level based on quality check results
        if result.get('status') == 'failed':
            level = logging.ERROR
        elif result.get('status') == 'warning':
            level = logging.WARNING
        else:
            level = logging.INFO
        
        self.log(level, f"Data quality check: {check_type} for {dataset_name}", extra=log_data)
    
    def log_model_performance(self, model_name: str, model_version: str,
                             metrics: Dict[str, float], dataset: str = "validation", **kwargs):
        """Log model performance metrics"""
        
        log_data = {
            'model_name': model_name,
            'model_version': model_version,
            'dataset': dataset,
            'event_type': 'model_performance',
            'performance_metrics': metrics,
            **kwargs
        }
        
        # Check for performance degradation
        if any(metric < 0.8 for metric in metrics.values() if isinstance(metric, (int, float))):
            level = logging.WARNING
            message = f"Model performance degradation detected for {model_name}"
        else:
            level = logging.INFO
            message = f"Model performance metrics for {model_name}"
        
        self.log(level, message, extra=log_data)
    
    def log_feature_importance(self, model_name: str, features: Dict[str, float], 
                              method: str = "default", **kwargs):
        """Log feature importance analysis"""
        
        log_data = {
            'model_name': model_name,
            'event_type': 'feature_importance',
            'importance_method': method,
            'feature_importance': features,
            'top_features': sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:10],
            **kwargs
        }
        
        self.info(f"Feature importance analysis for {model_name}", extra=log_data)
    
    # Standard logging methods with context
    def debug(self, msg, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def _log_with_context(self, level, msg, *args, **kwargs):
        """Internal method to add context to all log messages"""
        extra = kwargs.get('extra', {})
        
        # Add current context
        current_context = self.get_current_context()
        extra.update(current_context)
        
        # Add operation context if available
        if hasattr(self._operation_stack, 'stack') and self._operation_stack.stack:
            current_operation = self._operation_stack.stack[-1]
            extra['current_operation'] = {
                'operation_id': current_operation['operation_id'],
                'operation_name': current_operation['operation_name']
            }
        
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)

class MLJSONFormatter(logging.Formatter):
    """Custom JSON formatter for ML logs"""
    
    def format(self, record):
        # Base log structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                extra_fields[key] = self._serialize_value(value)
        
        if extra_fields:
            log_entry.update(extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    def _serialize_value(self, value):
        """Safely serialize values for JSON logging"""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        elif isinstance(value, np.ndarray):
            return {
                'type': 'numpy_array',
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'size': value.size
            }
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)

# Example usage in ML service
class MLModelService:
    def __init__(self):
        self.logger = MLLogger(__name__)
        self.model = None
        self.feature_extractor = None
    
    def initialize(self):
        """Initialize model service with logging"""
        with self.logger.operation_context("service_initialization"):
            try:
                self.logger.info("Initializing ML model service")
                
                # Set ML context
                ml_context = MLContext(
                    model_name="customer_classifier",
                    model_version="v2.1.0"
                )
                self.logger.set_ml_context(ml_context)
                
                # Load model
                self.model = self._load_model()
                self.feature_extractor = self._load_feature_extractor()
                
                self.logger.info("ML model service initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize ML model service: {str(e)}")
                raise
    
    def predict(self, input_data, user_segment="default"):
        """Make prediction with comprehensive logging"""
        
        # Set business context
        business_context = BusinessContext(user_segment=user_segment)
        self.logger.set_business_context(business_context)
        
        with self.logger.operation_context("prediction_request", 
                                         input_size=len(input_data) if hasattr(input_data, '__len__') else None):
            
            prediction_start = time.time()
            
            try:
                # Extract features
                features = self._extract_features(input_data)
                
                # Make prediction
                prediction = self._make_prediction(features)
                
                # Calculate performance metrics
                latency_ms = (time.time() - prediction_start) * 1000
                
                # Log prediction
                prediction_id = self.logger.log_prediction(
                    model_name="customer_classifier",
                    model_version="v2.1.0",
                    input_data=input_data,
                    prediction=prediction,
                    confidence=getattr(prediction, 'confidence', None),
                    latency_ms=latency_ms
                )
                
                return {
                    'prediction': prediction,
                    'prediction_id': prediction_id,
                    'latency_ms': latency_ms
                }
                
            except Exception as e:
                self.logger.error(f"Prediction failed: {str(e)}", extra={
                    'error_type': type(e).__name__,
                    'input_data_type': type(input_data).__name__
                })
                raise
    
    def _extract_features(self, input_data):
        """Extract features with logging"""
        with self.logger.operation_context("feature_extraction"):
            # Feature extraction logic here
            features = self.feature_extractor.transform(input_data)
            
            self.logger.debug("Features extracted", extra={
                'feature_count': len(features) if hasattr(features, '__len__') else None,
                'feature_type': type(features).__name__
            })
            
            return features
    
    def _make_prediction(self, features):
        """Make model prediction with logging"""
        with self.logger.operation_context("model_inference"):
            prediction = self.model.predict(features)
            
            self.logger.debug("Model inference completed", extra={
                'prediction_type': type(prediction).__name__
            })
            
            return prediction
```

---

## 🔍 Advanced Debugging Techniques

### **ML-Specific Debugging Strategies**

**Root Cause Analysis Framework:**
```
ML Root Cause Analysis System:
class MLRootCauseAnalyzer:
    def __init__(self, log_aggregator, metrics_collector):
        self.log_aggregator = log_aggregator
        self.metrics_collector = metrics_collector
        self.analysis_rules = self._load_analysis_rules()
        self.symptom_patterns = self._load_symptom_patterns()
    
    def analyze_performance_degradation(self, service_name, time_window='1h'):
        """Analyze performance degradation with ML-specific insights"""
        
        analysis_result = {
            'symptoms': [],
            'root_causes': [],
            'recommendations': [],
            'confidence_score': 0.0
        }
        
        # Collect relevant data
        logs = self.log_aggregator.get_logs(
            service=service_name,
            time_range=time_window,
            level='WARNING'
        )
        
        metrics = self.metrics_collector.get_metrics(
            service=service_name,
            time_range=time_window
        )
        
        # Analyze symptoms
        analysis_result['symptoms'] = self._identify_symptoms(logs, metrics)
        
        # Determine root causes
        analysis_result['root_causes'] = self._identify_root_causes(
            analysis_result['symptoms'], logs, metrics
        )
        
        # Generate recommendations
        analysis_result['recommendations'] = self._generate_recommendations(
            analysis_result['root_causes']
        )
        
        # Calculate confidence score
        analysis_result['confidence_score'] = self._calculate_confidence_score(
            analysis_result['symptoms'], analysis_result['root_causes']
        )
        
        return analysis_result
    
    def _identify_symptoms(self, logs, metrics):
        """Identify performance symptoms from logs and metrics"""
        symptoms = []
        
        # Analyze latency symptoms
        latency_metrics = [m for m in metrics if 'latency' in m['name'].lower()]
        for metric in latency_metrics:
            if self._detect_latency_spike(metric['values']):
                symptoms.append({
                    'type': 'latency_spike',
                    'severity': self._calculate_latency_severity(metric['values']),
                    'metric_name': metric['name'],
                    'evidence': metric['values'][-10:]  # Last 10 data points
                })
        
        # Analyze error rate symptoms
        error_logs = [log for log in logs if log['level'] in ['ERROR', 'CRITICAL']]
        if len(error_logs) > 0:
            error_patterns = self._analyze_error_patterns(error_logs)
            for pattern in error_patterns:
                symptoms.append({
                    'type': 'error_rate_increase',
                    'severity': pattern['severity'],
                    'error_type': pattern['error_type'],
                    'frequency': pattern['frequency'],
                    'evidence': pattern['sample_logs']
                })
        
        # Analyze ML-specific symptoms
        ml_symptoms = self._analyze_ml_symptoms(logs, metrics)
        symptoms.extend(ml_symptoms)
        
        return symptoms
    
    def _analyze_ml_symptoms(self, logs, metrics):
        """Analyze ML-specific performance symptoms"""
        symptoms = []
        
        # Model accuracy degradation
        accuracy_logs = [log for log in logs if 'model_performance' in log.get('event_type', '')]
        if accuracy_logs:
            accuracy_trend = self._analyze_accuracy_trend(accuracy_logs)
            if accuracy_trend['degradation_detected']:
                symptoms.append({
                    'type': 'model_accuracy_degradation',
                    'severity': 'high' if accuracy_trend['degradation_rate'] > 0.1 else 'medium',
                    'degradation_rate': accuracy_trend['degradation_rate'],
                    'affected_models': accuracy_trend['affected_models'],
                    'evidence': accuracy_trend['evidence']
                })
        
        # Feature drift detection
        drift_logs = [log for log in logs if 'feature_drift' in log.get('message', '').lower()]
        if drift_logs:
            drift_analysis = self._analyze_feature_drift(drift_logs)
            symptoms.append({
                'type': 'feature_drift',
                'severity': drift_analysis['severity'],
                'affected_features': drift_analysis['affected_features'],
                'drift_magnitude': drift_analysis['drift_magnitude'],
                'evidence': drift_analysis['evidence']
            })
        
        # Data quality issues
        quality_logs = [log for log in logs if 'data_quality_check' in log.get('event_type', '')]
        if quality_logs:
            quality_issues = self._analyze_data_quality_issues(quality_logs)
            for issue in quality_issues:
                symptoms.append({
                    'type': 'data_quality_issue',
                    'severity': issue['severity'],
                    'quality_metric': issue['metric'],
                    'failure_rate': issue['failure_rate'],
                    'evidence': issue['evidence']
                })
        
        # Resource exhaustion (ML-specific)
        gpu_metrics = [m for m in metrics if 'gpu' in m['name'].lower()]
        for metric in gpu_metrics:
            if self._detect_resource_exhaustion(metric['values'], threshold=0.95):
                symptoms.append({
                    'type': 'gpu_resource_exhaustion',
                    'severity': 'high',
                    'resource_type': 'gpu',
                    'utilization_pattern': metric['values'][-20:],
                    'evidence': f"GPU utilization consistently above 95%"
                })
        
        return symptoms
    
    def _identify_root_causes(self, symptoms, logs, metrics):
        """Identify root causes based on symptoms"""
        root_causes = []
        
        # Apply ML-specific root cause analysis rules
        for rule in self.analysis_rules:
            if rule['applies_to_symptoms'](symptoms):
                potential_cause = rule['analyze'](symptoms, logs, metrics)
                if potential_cause:
                    root_causes.append(potential_cause)
        
        # Correlation analysis
        correlation_causes = self._perform_correlation_analysis(symptoms, logs, metrics)
        root_causes.extend(correlation_causes)
        
        # Temporal analysis
        temporal_causes = self._perform_temporal_analysis(symptoms, logs, metrics)
        root_causes.extend(temporal_causes)
        
        # Remove duplicates and rank by confidence
        root_causes = self._deduplicate_and_rank_causes(root_causes)
        
        return root_causes
    
    def _perform_correlation_analysis(self, symptoms, logs, metrics):
        """Perform correlation analysis to identify related issues"""
        causes = []
        
        # Look for correlated events across different services
        service_events = {}
        for log in logs:
            service = log.get('service', 'unknown')
            timestamp = log['timestamp']
            
            if service not in service_events:
                service_events[service] = []
            service_events[service].append(timestamp)
        
        # Analyze temporal correlations
        correlations = self._calculate_temporal_correlations(service_events)
        
        for correlation in correlations:
            if correlation['correlation_score'] > 0.8:
                causes.append({
                    'type': 'cascading_failure',
                    'confidence': correlation['correlation_score'],
                    'primary_service': correlation['primary_service'],
                    'affected_services': correlation['affected_services'],
                    'time_lag': correlation['time_lag'],
                    'description': f"Failure in {correlation['primary_service']} caused issues in {', '.join(correlation['affected_services'])}"
                })
        
        return causes

Model Debugging Utilities:
class MLModelDebugger:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger
        self.debug_hooks = []
    
    def debug_prediction(self, input_data, expected_output=None):
        """Comprehensive debugging of model prediction"""
        
        debug_info = {
            'input_analysis': {},
            'intermediate_outputs': {},
            'prediction_analysis': {},
            'issues_detected': []
        }
        
        with self.logger.operation_context("model_debugging"):
            
            # Analyze input data
            debug_info['input_analysis'] = self._analyze_input_data(input_data)
            
            # Trace through model layers (if supported)
            if hasattr(self.model, 'layers') or hasattr(self.model, 'named_modules'):
                debug_info['intermediate_outputs'] = self._trace_model_execution(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_data)
            
            # Analyze prediction
            debug_info['prediction_analysis'] = self._analyze_prediction(prediction, expected_output)
            
            # Detect potential issues
            debug_info['issues_detected'] = self._detect_prediction_issues(
                input_data, prediction, debug_info
            )
            
            # Log comprehensive debug information
            self.logger.debug("Model prediction debug trace", extra={
                'debug_info': debug_info,
                'input_shape': getattr(input_data, 'shape', None),
                'prediction_shape': getattr(prediction, 'shape', None)
            })
        
        return debug_info
    
    def _analyze_input_data(self, input_data):
        """Analyze input data for potential issues"""
        analysis = {}
        
        if hasattr(input_data, 'shape'):
            analysis['shape'] = list(input_data.shape)
            analysis['dtype'] = str(input_data.dtype)
            
            # Check for NaN or infinite values
            if hasattr(input_data, 'isnan'):
                nan_count = int(np.sum(np.isnan(input_data)))
                inf_count = int(np.sum(np.isinf(input_data)))
                
                analysis['nan_values'] = nan_count
                analysis['infinite_values'] = inf_count
                
                if nan_count > 0 or inf_count > 0:
                    analysis['data_quality_issues'] = True
            
            # Statistical analysis
            if input_data.size > 0:
                analysis['statistics'] = {
                    'mean': float(np.mean(input_data)),
                    'std': float(np.std(input_data)),
                    'min': float(np.min(input_data)),
                    'max': float(np.max(input_data))
                }
        
        return analysis
    
    def _trace_model_execution(self, input_data):
        """Trace execution through model layers"""
        intermediate_outputs = {}
        
        # This would be framework-specific implementation
        # Example for PyTorch-like models
        if hasattr(self.model, 'named_modules'):
            current_input = input_data
            
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf module
                    try:
                        current_input = module(current_input)
                        intermediate_outputs[name] = {
                            'output_shape': list(current_input.shape) if hasattr(current_input, 'shape') else None,
                            'output_dtype': str(current_input.dtype) if hasattr(current_input, 'dtype') else None,
                            'activation_stats': self._calculate_activation_stats(current_input)
                        }
                    except Exception as e:
                        intermediate_outputs[name] = {
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                        break
        
        return intermediate_outputs
    
    def _calculate_activation_stats(self, activation):
        """Calculate statistics for layer activations"""
        if not hasattr(activation, 'detach'):
            return None
        
        # Convert to numpy for analysis
        act_np = activation.detach().cpu().numpy() if hasattr(activation, 'cpu') else activation
        
        return {
            'mean': float(np.mean(act_np)),
            'std': float(np.std(act_np)),
            'min': float(np.min(act_np)),
            'max': float(np.max(act_np)),
            'zero_fraction': float(np.mean(act_np == 0)),
            'nan_count': int(np.sum(np.isnan(act_np)))
        }
    
    def _detect_prediction_issues(self, input_data, prediction, debug_info):
        """Detect potential issues with model prediction"""
        issues = []
        
        # Check for NaN in prediction
        if hasattr(prediction, 'isnan') and np.any(np.isnan(prediction)):
            issues.append({
                'type': 'nan_prediction',
                'severity': 'high',
                'description': 'Model prediction contains NaN values'
            })
        
        # Check for extreme values
        if hasattr(prediction, 'max') and hasattr(prediction, 'min'):
            pred_max = float(np.max(prediction))
            pred_min = float(np.min(prediction))
            
            if abs(pred_max) > 1e6 or abs(pred_min) > 1e6:
                issues.append({
                    'type': 'extreme_values',
                    'severity': 'medium',
                    'description': f'Prediction contains extreme values: min={pred_min}, max={pred_max}'
                })
        
        # Check for dead neurons in intermediate outputs
        for layer_name, layer_info in debug_info.get('intermediate_outputs', {}).items():
            if 'activation_stats' in layer_info:
                stats = layer_info['activation_stats']
                if stats and stats['zero_fraction'] > 0.9:
                    issues.append({
                        'type': 'dead_neurons',
                        'severity': 'medium',
                        'layer': layer_name,
                        'description': f'Layer {layer_name} has {stats["zero_fraction"]:.1%} dead neurons'
                    })
        
        # Check for gradient explosion indicators
        for layer_name, layer_info in debug_info.get('intermediate_outputs', {}).items():
            if 'activation_stats' in layer_info:
                stats = layer_info['activation_stats']
                if stats and (abs(stats['max']) > 100 or abs(stats['min']) > 100):
                    issues.append({
                        'type': 'gradient_explosion',
                        'severity': 'high',
                        'layer': layer_name,
                        'description': f'Layer {layer_name} shows signs of gradient explosion'
                    })
        
        return issues

Automated Log Analysis:
class MLLogAnalyzer:
    def __init__(self, elasticsearch_client):
        self.es_client = elasticsearch_client
        self.anomaly_detectors = self._initialize_anomaly_detectors()
    
    def analyze_error_patterns(self, time_range='24h', service_filter=None):
        """Analyze error patterns in ML system logs"""
        
        # Query error logs
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"timestamp": {"gte": f"now-{time_range}"}}},
                        {"terms": {"level": ["ERROR", "CRITICAL"]}}
                    ]
                }
            },
            "aggs": {
                "error_types": {
                    "terms": {"field": "error_type.keyword", "size": 50}
                },
                "services": {
                    "terms": {"field": "service.keyword", "size": 20}
                },
                "error_timeline": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "1h"
                    }
                }
            }
        }
        
        if service_filter:
            query["query"]["bool"]["must"].append(
                {"term": {"service.keyword": service_filter}}
            )
        
        response = self.es_client.search(index="ml-logs-*", body=query)
        
        # Analyze results
        analysis = {
            'error_summary': self._analyze_error_summary(response),
            'error_patterns': self._identify_error_patterns(response),
            'anomalies': self._detect_error_anomalies(response),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_error_recommendations(analysis)
        
        return analysis
    
    def _identify_error_patterns(self, search_response):
        """Identify patterns in error occurrences"""
        patterns = []
        
        # Analyze error type clustering
        error_buckets = search_response['aggregations']['error_types']['buckets']
        
        for bucket in error_buckets:
            error_type = bucket['key']
            count = bucket['doc_count']
            
            if count > 10:  # Significant error count
                # Get sample logs for this error type
                sample_query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"error_type.keyword": error_type}},
                                {"range": {"timestamp": {"gte": "now-24h"}}}
                            ]
                        }
                    },
                    "size": 5,
                    "sort": [{"timestamp": {"order": "desc"}}]
                }
                
                sample_response = self.es_client.search(index="ml-logs-*", body=sample_query)
                sample_logs = [hit['_source'] for hit in sample_response['hits']['hits']]
                
                patterns.append({
                    'error_type': error_type,
                    'frequency': count,
                    'sample_logs': sample_logs,
                    'pattern_analysis': self._analyze_error_pattern(sample_logs)
                })
        
        return patterns
    
    def _analyze_error_pattern(self, error_logs):
        """Analyze a specific error pattern"""
        pattern_info = {
            'common_services': [],
            'common_operations': [],
            'time_distribution': {},
            'potential_causes': []
        }
        
        # Extract common characteristics
        services = [log.get('service') for log in error_logs if log.get('service')]
        operations = [log.get('operation_name') for log in error_logs if log.get('operation_name')]
        
        from collections import Counter
        
        if services:
            service_counts = Counter(services)
            pattern_info['common_services'] = service_counts.most_common(3)
        
        if operations:
            operation_counts = Counter(operations)
            pattern_info['common_operations'] = operation_counts.most_common(3)
        
        # Analyze ML-specific patterns
        ml_contexts = [log.get('ml_context', {}) for log in error_logs]
        model_names = [ctx.get('model_name') for ctx in ml_contexts if ctx.get('model_name')]
        
        if model_names:
            model_counts = Counter(model_names)
            pattern_info['affected_models'] = model_counts.most_common(5)
        
        return pattern_info
```

This comprehensive framework for logging and debugging ML systems provides the theoretical foundations and practical strategies for implementing robust observability and troubleshooting capabilities. The key insight is that ML systems require specialized logging approaches that capture model behavior, data quality, and system performance patterns essential for effective debugging and root cause analysis.