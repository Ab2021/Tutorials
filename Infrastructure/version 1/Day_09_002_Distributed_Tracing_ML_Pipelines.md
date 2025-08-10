# Day 9.2: Distributed Tracing for ML Pipelines

## 🔍 Monitoring, Observability & Debugging - Part 2

**Focus**: End-to-End Request Tracing, Performance Analysis, Dependency Mapping  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master distributed tracing implementation for complex ML pipeline workflows
- Learn advanced trace analysis techniques for performance optimization and bottleneck identification
- Understand context propagation and correlation across ML system components
- Analyze trace data for capacity planning and system optimization in ML environments

---

## 🔍 Distributed Tracing Theoretical Framework

### **ML Pipeline Tracing Architecture**

Distributed tracing in ML systems requires tracking requests across multiple services, data processing stages, and model inference paths, providing end-to-end visibility into complex workflows.

**ML Tracing Taxonomy:**
```
ML System Tracing Layers:
1. Request-Level Tracing:
   - API gateway to final response
   - User request journey tracking
   - Cross-service call correlation
   - Error propagation analysis

2. Pipeline-Level Tracing:
   - Data ingestion to model output
   - Feature engineering workflows
   - Model training pipelines
   - Batch processing jobs

3. Model-Level Tracing:
   - Preprocessing operations
   - Feature extraction steps
   - Model inference execution
   - Post-processing operations

4. Infrastructure-Level Tracing:
   - Container orchestration calls
   - Storage I/O operations
   - Network communication
   - Resource allocation events

Tracing Mathematical Model:
Trace_Completeness = (Traced_Operations / Total_Operations) × Context_Propagation_Rate

Performance_Bottleneck_Score = Σ(Operation_Duration × Operation_Frequency × Business_Impact)

Critical_Path_Analysis:
Critical_Path = max(Σ(Operation_Duration)) for all paths in trace_dag

Latency_Attribution:
Service_Contribution = Service_Duration / Total_Request_Duration

Distributed Tracing Overhead:
Overhead_Impact = (Tracing_CPU_Cost + Tracing_Memory_Cost + Tracing_Network_Cost) / Base_Operation_Cost
```

**Trace Context Propagation:**
```
ML Pipeline Context Propagation Model:
1. Request Context:
   - Trace ID: Unique identifier for entire request
   - Span ID: Current operation identifier
   - Parent Span ID: Hierarchical relationship
   - Baggage: Cross-cutting concerns (user_id, experiment_id, model_version)

2. ML-Specific Context:
   - Model Version: Track which model version processed request
   - Feature Store Version: Track feature data lineage
   - Experiment ID: Connect to A/B testing framework
   - Data Batch ID: Connect to training data lineage

3. Business Context:
   - Customer Segment: Track performance by customer type
   - Product Context: Track performance by product area
   - Geographic Region: Track performance by location
   - Request Priority: Track SLA adherence

Context Propagation Implementation:
class MLTraceContext:
    def __init__(self, trace_id=None, span_id=None, parent_span_id=None):
        self.trace_id = trace_id or self.generate_trace_id()
        self.span_id = span_id or self.generate_span_id()
        self.parent_span_id = parent_span_id
        self.baggage = {}
        self.ml_context = {}
        self.business_context = {}
    
    def add_ml_context(self, model_version=None, experiment_id=None, 
                      feature_version=None, data_batch_id=None):
        self.ml_context.update({
            'model_version': model_version,
            'experiment_id': experiment_id,
            'feature_version': feature_version,
            'data_batch_id': data_batch_id
        })
    
    def add_business_context(self, customer_segment=None, product_area=None,
                           region=None, priority=None):
        self.business_context.update({
            'customer_segment': customer_segment,
            'product_area': product_area,
            'region': region,
            'priority': priority
        })
    
    def propagate_to_headers(self):
        """Convert context to HTTP headers for propagation"""
        headers = {
            'X-Trace-Id': self.trace_id,
            'X-Span-Id': self.span_id,
            'X-Parent-Span-Id': self.parent_span_id or '',
        }
        
        # Add baggage
        for key, value in self.baggage.items():
            headers[f'X-Baggage-{key.title()}'] = str(value)
        
        # Add ML context
        for key, value in self.ml_context.items():
            if value:
                headers[f'X-ML-{key.replace("_", "-").title()}'] = str(value)
        
        # Add business context
        for key, value in self.business_context.items():
            if value:
                headers[f'X-Business-{key.replace("_", "-").title()}'] = str(value)
        
        return headers
    
    @classmethod
    def from_headers(cls, headers):
        """Extract context from HTTP headers"""
        trace_id = headers.get('X-Trace-Id')
        span_id = headers.get('X-Span-Id')
        parent_span_id = headers.get('X-Parent-Span-Id') or None
        
        context = cls(trace_id, span_id, parent_span_id)
        
        # Extract baggage
        for key, value in headers.items():
            if key.startswith('X-Baggage-'):
                baggage_key = key[10:].lower()
                context.baggage[baggage_key] = value
        
        # Extract ML context
        for key, value in headers.items():
            if key.startswith('X-ML-'):
                ml_key = key[5:].lower().replace('-', '_')
                context.ml_context[ml_key] = value
        
        # Extract business context
        for key, value in headers.items():
            if key.startswith('X-Business-'):
                business_key = key[11:].lower().replace('-', '_')
                context.business_context[business_key] = value
        
        return context
```

---

## 🚀 Jaeger Implementation for ML Systems

### **Jaeger Configuration and Deployment**

**Kubernetes Deployment with ML-Specific Configuration:**
```
Jaeger Operator Deployment:
apiVersion: v1
kind: Namespace
metadata:
  name: jaeger-system

---
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: ml-platform-jaeger
  namespace: jaeger-system
spec:
  strategy: production
  
  collector:
    replicas: 3
    resources:
      requests:
        cpu: 500m
        memory: 1Gi
      limits:
        cpu: 1000m
        memory: 2Gi
    
    # ML-specific configuration
    config: |
      processors:
        batch:
          timeout: 5s
          send_batch_size: 1024
        
        # Custom processor for ML context enrichment
        ml_context_processor:
          model_registry_endpoint: "http://model-registry.ml-platform.svc.cluster.local"
          feature_store_endpoint: "http://feature-store.ml-platform.svc.cluster.local"
        
        # Sampling configuration for ML workloads
        sampling:
          default_strategy:
            type: probabilistic
            param: 0.1  # 10% sampling for general traffic
          
          per_service_strategies:
            - service: "ml-inference-*"
              type: adaptive
              max_traces_per_second: 100
            
            - service: "ml-training-*"
              type: probabilistic
              param: 1.0  # 100% sampling for training jobs
            
            - service: "data-pipeline-*"
              type: rate_limiting
              max_traces_per_second: 50
  
  query:
    replicas: 2
    resources:
      requests:
        cpu: 300m
        memory: 512Mi
      limits:
        cpu: 500m
        memory: 1Gi
    
    # Custom UI configuration for ML workflows
    options:
      query.ui-config: |
        {
          "dependencies": {
            "dagMaxNumServices": 100
          },
          "tracking": {
            "gaID": "UA-000000-2"
          },
          "menu": [
            {
              "label": "ML Dashboards",
              "items": [
                {
                  "label": "Model Performance",
                  "url": "/model-performance"
                },
                {
                  "label": "Training Jobs",
                  "url": "/training-jobs"
                },
                {
                  "label": "Data Pipelines",
                  "url": "/data-pipelines"
                }
              ]
            }
          ]
        }
  
  storage:
    type: elasticsearch
    elasticsearch:
      nodeCount: 3
      resources:
        requests:
          cpu: 1000m
          memory: 2Gi
        limits:
          cpu: 2000m
          memory: 4Gi
      redundancyPolicy: SingleRedundancy
      
      # ML-optimized index configuration
      esIndexCleaner:
        enabled: true
        numberOfDays: 30  # Longer retention for ML analysis
        schedule: "55 23 * * *"
      
      # Custom index templates for ML traces
      indexSettings: |
        {
          "index.number_of_shards": 3,
          "index.number_of_replicas": 1,
          "index.codec": "best_compression",
          "index.mapping.total_fields.limit": 2000
        }

---
# Service Monitor for Prometheus integration
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: jaeger-metrics
  namespace: jaeger-system
spec:
  selector:
    matchLabels:
      app: jaeger
  endpoints:
  - port: admin-http
    path: /metrics
    interval: 30s

ML-Specific Jaeger Agent Configuration:
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-agent-config
  namespace: jaeger-system
data:
  agent.yaml: |
    reporter:
      type: grpc
      grpc:
        host-port: ml-platform-jaeger-collector.jaeger-system.svc.cluster.local:14250
        tls:
          enabled: false
    
    processors:
      - jaeger-compact
      - jaeger-binary
    
    # ML-specific sampling strategies
    sampling:
      strategies:
        default_strategy:
          type: adaptive
          param: 0.1
          
        per_service_strategies:
          # High sampling for critical ML services
          - service: "ml-inference-production"
            type: probabilistic
            param: 1.0
            
          - service: "model-serving-*"
            type: adaptive
            param: 0.5
            
          # Lower sampling for batch processing
          - service: "data-pipeline-batch"
            type: rate_limiting
            param: 10
            
        per_operation_strategies:
          # Always trace model predictions
          - service: "*"
            operation: "predict"
            type: probabilistic
            param: 1.0
            
          - service: "*"
            operation: "train_model"
            type: probabilistic
            param: 1.0

---  
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: jaeger-agent
  namespace: jaeger-system
spec:
  selector:
    matchLabels:
      app: jaeger-agent
  template:
    metadata:
      labels:
        app: jaeger-agent
    spec:
      containers:
      - name: jaeger-agent
        image: jaegertracing/jaeger-agent:1.50.0
        args:
        - --reporter.grpc.host-port=ml-platform-jaeger-collector.jaeger-system.svc.cluster.local:14250
        - --processor.jaeger-compact.server-queue-size=1000
        - --processor.jaeger-compact.server-max-packet-size=65000
        - --processor.jaeger-compact.server-host-port=:6831
        - --processor.jaeger-binary.server-queue-size=1000
        - --processor.jaeger-binary.server-max-packet-size=65000
        - --processor.jaeger-binary.server-host-port=:6832
        - --processor.zipkin-compact.server-queue-size=1000
        - --processor.zipkin-compact.server-max-packet-size=65000
        - --processor.zipkin-compact.server-host-port=:5775
        
        ports:
        - containerPort: 5775
          protocol: UDP
        - containerPort: 6831
          protocol: UDP
        - containerPort: 6832
          protocol: UDP
        - containerPort: 5778
          protocol: TCP
        
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        
        volumeMounts:
        - name: config
          mountPath: /etc/jaeger
          readOnly: true
      
      volumes:
      - name: config
        configMap:
          name: jaeger-agent-config
      
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
```

**ML Service Instrumentation:**
```
Python ML Service Instrumentation:
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
import numpy as np
import time

class MLServiceTracer:
    def __init__(self, service_name, jaeger_agent_host="localhost", jaeger_agent_port=6831):
        # Configure tracer
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer_provider().get_tracer(service_name)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_agent_host,
            agent_port=jaeger_agent_port,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Auto-instrument common libraries
        FlaskInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument()
        
        self.tracer = tracer
        self.service_name = service_name
    
    def trace_model_prediction(self, model_name, model_version):
        """Decorator for tracing model predictions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("model_prediction") as span:
                    # Add ML-specific attributes
                    span.set_attribute("ml.model.name", model_name)
                    span.set_attribute("ml.model.version", model_version)
                    span.set_attribute("ml.operation.type", "prediction")
                    
                    # Add input characteristics
                    if args and hasattr(args[0], 'shape'):
                        input_data = args[0]
                        span.set_attribute("ml.input.shape", str(input_data.shape))
                        span.set_attribute("ml.input.dtype", str(input_data.dtype))
                        span.set_attribute("ml.input.size", int(np.prod(input_data.shape)))
                    
                    # Record start time
                    start_time = time.time()
                    
                    try:
                        # Execute prediction
                        result = func(*args, **kwargs)
                        
                        # Add result characteristics
                        if hasattr(result, 'shape'):
                            span.set_attribute("ml.output.shape", str(result.shape))
                            span.set_attribute("ml.output.dtype", str(result.dtype))
                        
                        # Add confidence score if available
                        if hasattr(result, 'confidence') or (isinstance(result, tuple) and len(result) > 1):
                            confidence = getattr(result, 'confidence', result[1] if isinstance(result, tuple) else None)
                            if confidence is not None:
                                if np.isscalar(confidence):
                                    span.set_attribute("ml.prediction.confidence", float(confidence))
                                else:
                                    span.set_attribute("ml.prediction.confidence.mean", float(np.mean(confidence)))
                                    span.set_attribute("ml.prediction.confidence.std", float(np.std(confidence)))
                        
                        span.set_attribute("ml.prediction.success", True)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        
                        return result
                        
                    except Exception as e:
                        # Record error
                        span.set_attribute("ml.prediction.success", False)
                        span.set_attribute("ml.error.type", type(e).__name__)
                        span.set_attribute("ml.error.message", str(e))
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise
                    
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        span.set_attribute("ml.prediction.duration_ms", duration * 1000)
            
            return wrapper
        return decorator
    
    def trace_feature_extraction(self, feature_store_version=None):
        """Decorator for tracing feature extraction"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("feature_extraction") as span:
                    span.set_attribute("ml.operation.type", "feature_extraction")
                    
                    if feature_store_version:
                        span.set_attribute("ml.feature_store.version", feature_store_version)
                    
                    # Add context about feature extraction
                    if 'feature_names' in kwargs:
                        span.set_attribute("ml.features.count", len(kwargs['feature_names']))
                        span.set_attribute("ml.features.names", ','.join(kwargs['feature_names'][:10]))  # Limit to first 10
                    
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Add result characteristics
                        if hasattr(result, 'shape'):
                            span.set_attribute("ml.features.shape", str(result.shape))
                            span.set_attribute("ml.features.dtype", str(result.dtype))
                        
                        span.set_attribute("ml.feature_extraction.success", True)
                        return result
                        
                    except Exception as e:
                        span.set_attribute("ml.feature_extraction.success", False)
                        span.set_attribute("ml.error.type", type(e).__name__)
                        span.set_attribute("ml.error.message", str(e))
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise
                    
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("ml.feature_extraction.duration_ms", duration * 1000)
            
            return wrapper
        return decorator
    
    def trace_data_processing(self, pipeline_stage=None):
        """Decorator for tracing data processing operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span("data_processing") as span:
                    span.set_attribute("ml.operation.type", "data_processing")
                    
                    if pipeline_stage:
                        span.set_attribute("ml.pipeline.stage", pipeline_stage)
                    
                    # Add input data characteristics
                    if args and hasattr(args[0], '__len__'):
                        input_data = args[0]
                        span.set_attribute("ml.data.input_size", len(input_data))
                    
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Add output data characteristics
                        if hasattr(result, '__len__'):
                            span.set_attribute("ml.data.output_size", len(result))
                        
                        span.set_attribute("ml.data_processing.success", True)
                        return result
                        
                    except Exception as e:
                        span.set_attribute("ml.data_processing.success", False)
                        span.set_attribute("ml.error.type", type(e).__name__)
                        span.set_attribute("ml.error.message", str(e))
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise
                    
                    finally:
                        duration = time.time() - start_time
                        span.set_attribute("ml.data_processing.duration_ms", duration * 1000)
            
            return wrapper
        return decorator

Example ML Service Implementation:
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Initialize tracer
tracer = MLServiceTracer("ml-inference-service")

# Load model (in real implementation, this would be more sophisticated)
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with comprehensive tracing"""
    
    # Extract trace context from headers
    trace_context = MLTraceContext.from_headers(request.headers)
    
    with tracer.tracer.start_as_current_span("prediction_request") as span:
        # Add request context
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", request.url)
        span.set_attribute("http.user_agent", request.headers.get('User-Agent', ''))
        
        # Add ML context from headers
        for key, value in trace_context.ml_context.items():
            span.set_attribute(f"ml.context.{key}", value)
        
        # Add business context
        for key, value in trace_context.business_context.items():
            span.set_attribute(f"business.context.{key}", value)
        
        try:
            # Parse input data
            input_data = request.json
            
            # Extract features with tracing
            features = extract_features(input_data, trace_context)
            
            # Make prediction with tracing
            prediction = make_prediction(features, trace_context)
            
            # Post-process results with tracing
            result = post_process_prediction(prediction, trace_context)
            
            span.set_attribute("prediction.success", True)
            return jsonify(result)
            
        except Exception as e:
            span.set_attribute("prediction.success", False)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            return jsonify({"error": str(e)}), 500

@tracer.trace_feature_extraction(feature_store_version="v1.2.0")
def extract_features(input_data, context):
    """Extract features with detailed tracing"""
    
    # Simulate feature extraction from feature store
    with tracer.tracer.start_as_current_span("feature_store_lookup") as span:
        span.set_attribute("feature_store.query_type", "batch")
        span.set_attribute("feature_store.entity_count", len(input_data.get('entities', [])))
        
        # Simulate feature store call
        time.sleep(0.01)  # Simulate network latency
        
        features = pd.DataFrame(input_data)
        return features

@tracer.trace_model_prediction(model_name="customer_classifier", model_version="v2.1.0")
def make_prediction(features, context):
    """Make model prediction with tracing"""
    
    with tracer.tracer.start_as_current_span("model_inference") as span:
        span.set_attribute("model.framework", "scikit-learn")
        span.set_attribute("model.algorithm", "random_forest")
        span.set_attribute("model.n_features", features.shape[1])
        
        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        
        # Add prediction metadata
        span.set_attribute("prediction.class_count", len(prediction))
        span.set_attribute("prediction.confidence.mean", float(np.mean(np.max(prediction_proba, axis=1))))
        
        return {
            'prediction': prediction.tolist(),
            'probability': prediction_proba.tolist(),
            'confidence': np.max(prediction_proba, axis=1).tolist()
        }

@tracer.trace_data_processing(pipeline_stage="post_processing")
def post_process_prediction(prediction, context):
    """Post-process prediction results with tracing"""
    
    with tracer.tracer.start_as_current_span("result_formatting") as span:
        span.set_attribute("post_processing.type", "classification_formatting")
        
        # Format results
        formatted_result = {
            'predictions': prediction['prediction'],
            'probabilities': prediction['probability'],
            'confidence_scores': prediction['confidence'],
            'model_version': context.ml_context.get('model_version', 'unknown'),
            'request_id': context.trace_id
        }
        
        return formatted_result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📊 Advanced Trace Analysis

### **Performance Analysis and Optimization**

**Trace Analytics Implementation:**
```
Advanced Trace Analysis System:
class MLTraceAnalyzer:
    def __init__(self, jaeger_query_endpoint):
        self.jaeger_client = JaegerClient(jaeger_query_endpoint)
        self.analysis_cache = {}
        
    def analyze_ml_pipeline_performance(self, service_name, time_range='1h'):
        """Comprehensive analysis of ML pipeline performance"""
        
        # Fetch traces
        traces = self.jaeger_client.get_traces(
            service=service_name,
            lookback=time_range,
            limit=10000
        )
        
        analysis_result = {
            'performance_metrics': self.calculate_performance_metrics(traces),
            'bottleneck_analysis': self.identify_bottlenecks(traces),
            'error_analysis': self.analyze_errors(traces),
            'dependency_analysis': self.analyze_dependencies(traces),
            'capacity_analysis': self.analyze_capacity_patterns(traces)
        }
        
        return analysis_result
    
    def calculate_performance_metrics(self, traces):
        """Calculate detailed performance metrics from traces"""
        
        metrics = {
            'latency_distribution': {},
            'throughput_analysis': {},
            'success_rate': {},
            'resource_utilization': {}
        }
        
        # Process each trace
        latencies = []
        throughput_buckets = {}
        success_count = 0
        total_count = 0
        
        for trace in traces:
            total_count += 1
            
            # Calculate end-to-end latency
            start_time = min(span.start_time for span in trace.spans)
            end_time = max(span.start_time + span.duration for span in trace.spans)
            latency = (end_time - start_time) / 1000  # Convert to milliseconds
            latencies.append(latency)
            
            # Track throughput by time bucket (5-minute buckets)
            time_bucket = int(start_time / (5 * 60 * 1000000)) * (5 * 60 * 1000000)
            throughput_buckets[time_bucket] = throughput_buckets.get(time_bucket, 0) + 1
            
            # Check for errors
            has_error = any(span.status.code == 'ERROR' for span in trace.spans)
            if not has_error:
                success_count += 1
        
        # Calculate latency statistics
        if latencies:
            metrics['latency_distribution'] = {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'std': np.std(latencies)
            }
        
        # Calculate throughput statistics
        throughput_values = list(throughput_buckets.values())
        if throughput_values:
            metrics['throughput_analysis'] = {
                'mean_requests_per_5min': np.mean(throughput_values),
                'max_requests_per_5min': np.max(throughput_values),
                'throughput_variability': np.std(throughput_values) / np.mean(throughput_values)
            }
        
        # Calculate success rate
        metrics['success_rate'] = {
            'overall': success_count / total_count if total_count > 0 else 0,
            'total_requests': total_count,
            'successful_requests': success_count,
            'failed_requests': total_count - success_count
        }
        
        return metrics
    
    def identify_bottlenecks(self, traces):
        """Identify performance bottlenecks in ML pipelines"""
        
        operation_stats = {}
        service_stats = {}
        
        for trace in traces:
            for span in trace.spans:
                operation_name = span.operation_name
                service_name = span.process.service_name
                duration = span.duration / 1000  # Convert to milliseconds
                
                # Track operation statistics
                if operation_name not in operation_stats:
                    operation_stats[operation_name] = {
                        'durations': [],
                        'count': 0,
                        'service': service_name
                    }
                
                operation_stats[operation_name]['durations'].append(duration)
                operation_stats[operation_name]['count'] += 1
                
                # Track service statistics
                if service_name not in service_stats:
                    service_stats[service_name] = {
                        'durations': [],
                        'count': 0
                    }
                
                service_stats[service_name]['durations'].append(duration)
                service_stats[service_name]['count'] += 1
        
        # Calculate bottleneck scores
        bottlenecks = []
        
        for operation, stats in operation_stats.items():
            if stats['count'] > 10:  # Minimum sample size
                mean_duration = np.mean(stats['durations'])
                p95_duration = np.percentile(stats['durations'], 95)
                frequency = stats['count']
                
                # Bottleneck score combines duration and frequency
                bottleneck_score = mean_duration * np.log(frequency + 1)
                
                bottlenecks.append({
                    'operation': operation,
                    'service': stats['service'],
                    'mean_duration_ms': mean_duration,
                    'p95_duration_ms': p95_duration,
                    'frequency': frequency,
                    'bottleneck_score': bottleneck_score,
                    'optimization_potential': self.calculate_optimization_potential(stats['durations'])
                })
        
        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)
        
        return {
            'top_bottlenecks': bottlenecks[:10],
            'service_breakdown': self._calculate_service_breakdown(service_stats),
            'recommendations': self._generate_optimization_recommendations(bottlenecks)
        }
    
    def analyze_errors(self, traces):
        """Analyze error patterns in ML traces"""
        
        error_patterns = {
            'error_types': {},
            'error_services': {},
            'error_operations': {},
            'error_correlation': {},
            'error_timeline': {}
        }
        
        for trace in traces:
            trace_has_error = False
            trace_errors = []
            
            for span in trace.spans:
                if span.status.code == 'ERROR':
                    trace_has_error = True
                    error_info = {
                        'service': span.process.service_name,
                        'operation': span.operation_name,
                        'error_type': span.tags.get('error.type', 'unknown'),
                        'error_message': span.tags.get('error.message', ''),
                        'timestamp': span.start_time
                    }
                    trace_errors.append(error_info)
                    
                    # Track error types
                    error_type = error_info['error_type']
                    if error_type not in error_patterns['error_types']:
                        error_patterns['error_types'][error_type] = {
                            'count': 0,
                            'services': set(),
                            'operations': set()
                        }
                    
                    error_patterns['error_types'][error_type]['count'] += 1
                    error_patterns['error_types'][error_type]['services'].add(error_info['service'])
                    error_patterns['error_types'][error_type]['operations'].add(error_info['operation'])
            
            # Analyze error correlation within traces
            if len(trace_errors) > 1:
                for i, error1 in enumerate(trace_errors):
                    for error2 in trace_errors[i+1:]:
                        correlation_key = f"{error1['service']}->{error2['service']}"
                        if correlation_key not in error_patterns['error_correlation']:
                            error_patterns['error_correlation'][correlation_key] = 0
                        error_patterns['error_correlation'][correlation_key] += 1
        
        # Convert sets to lists for JSON serialization
        for error_type, stats in error_patterns['error_types'].items():
            stats['services'] = list(stats['services'])
            stats['operations'] = list(stats['operations'])
        
        return error_patterns
    
    def analyze_dependencies(self, traces):
        """Analyze service dependencies and communication patterns"""
        
        dependency_graph = {}
        communication_stats = {}
        
        for trace in traces:
            # Build parent-child relationships
            span_map = {span.span_id: span for span in trace.spans}
            
            for span in trace.spans:
                if span.parent_span_id and span.parent_span_id in span_map:
                    parent_span = span_map[span.parent_span_id]
                    parent_service = parent_span.process.service_name
                    child_service = span.process.service_name
                    
                    if parent_service != child_service:  # Cross-service call
                        dependency_key = f"{parent_service}->{child_service}"
                        
                        if dependency_key not in dependency_graph:
                            dependency_graph[dependency_key] = {
                                'call_count': 0,
                                'total_duration': 0,
                                'error_count': 0,
                                'operations': set()
                            }
                        
                        dependency_graph[dependency_key]['call_count'] += 1
                        dependency_graph[dependency_key]['total_duration'] += span.duration
                        dependency_graph[dependency_key]['operations'].add(span.operation_name)
                        
                        if span.status.code == 'ERROR':
                            dependency_graph[dependency_key]['error_count'] += 1
        
        # Calculate dependency metrics
        dependency_metrics = {}
        for dep_key, stats in dependency_graph.items():
            dependency_metrics[dep_key] = {
                'call_frequency': stats['call_count'],
                'average_duration_ms': stats['total_duration'] / stats['call_count'] / 1000,
                'error_rate': stats['error_count'] / stats['call_count'],
                'operations': list(stats['operations']),
                'criticality_score': self._calculate_dependency_criticality(stats)
            }
        
        return {
            'dependency_graph': dependency_metrics,
            'critical_paths': self._identify_critical_paths(dependency_metrics),
            'communication_patterns': self._analyze_communication_patterns(traces)
        }
    
    def _calculate_dependency_criticality(self, stats):
        """Calculate how critical a dependency is based on frequency and impact"""
        frequency_score = np.log(stats['call_count'] + 1)
        duration_score = stats['total_duration'] / stats['call_count'] / 1000000  # Normalize
        error_score = stats['error_count'] * 10  # Errors are heavily weighted
        
        return frequency_score + duration_score + error_score

Automated Performance Recommendations:
class MLPerformanceOptimizer:
    def __init__(self, trace_analyzer):
        self.trace_analyzer = trace_analyzer
        self.optimization_rules = self._load_optimization_rules()
    
    def generate_optimization_recommendations(self, service_name):
        """Generate automated optimization recommendations"""
        
        # Analyze current performance
        analysis = self.trace_analyzer.analyze_ml_pipeline_performance(service_name)
        
        recommendations = []
        
        # Check for common ML performance issues
        recommendations.extend(self._check_model_loading_optimization(analysis))
        recommendations.extend(self._check_batch_processing_optimization(analysis))
        recommendations.extend(self._check_feature_extraction_optimization(analysis))
        recommendations.extend(self._check_caching_opportunities(analysis))
        recommendations.extend(self._check_scaling_recommendations(analysis))
        
        # Prioritize recommendations
        recommendations.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return {
            'high_impact': [r for r in recommendations if r['impact_score'] > 8],
            'medium_impact': [r for r in recommendations if 5 <= r['impact_score'] <= 8],
            'low_impact': [r for r in recommendations if r['impact_score'] < 5],
            'total_recommendations': len(recommendations)
        }
    
    def _check_model_loading_optimization(self, analysis):
        """Check for model loading performance issues"""
        recommendations = []
        
        # Look for repeated model loading operations
        bottlenecks = analysis['bottleneck_analysis']['top_bottlenecks']
        
        for bottleneck in bottlenecks:
            if 'model_load' in bottleneck['operation'].lower():
                if bottleneck['frequency'] > 100:  # Frequent model loading
                    recommendations.append({
                        'type': 'model_caching',
                        'title': 'Implement Model Caching',
                        'description': f"Model loading operation '{bottleneck['operation']}' is called {bottleneck['frequency']} times with average duration {bottleneck['mean_duration_ms']:.2f}ms",
                        'impact_score': 9,
                        'implementation': 'Use model registry with in-memory caching or shared model cache',
                        'estimated_improvement': f"Reduce latency by {bottleneck['mean_duration_ms'] * 0.8:.2f}ms per request"
                    })
        
        return recommendations
    
    def _check_batch_processing_optimization(self, analysis):
        """Check for batch processing optimization opportunities"""
        recommendations = []
        
        # Analyze request patterns for batching opportunities
        throughput = analysis['performance_metrics']['throughput_analysis']
        
        if throughput and throughput['throughput_variability'] > 0.5:
            recommendations.append({
                'type': 'request_batching',
                'title': 'Implement Request Batching',
                'description': f"High throughput variability ({throughput['throughput_variability']:.2f}) suggests batching opportunities",
                'impact_score': 7,
                'implementation': 'Implement request queuing and batch processing for predictions',
                'estimated_improvement': 'Reduce resource usage by 30-50% and improve throughput consistency'
            })
        
        return recommendations
```

This comprehensive framework for distributed tracing in ML pipelines provides the theoretical foundations and practical strategies for implementing end-to-end observability across complex machine learning systems. The key insight is that ML systems require specialized tracing approaches that capture model-specific context, data lineage, and performance characteristics alongside traditional request tracing.