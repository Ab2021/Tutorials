# Day 9.1: Prometheus & Grafana for ML Metrics

## 📊 Monitoring, Observability & Debugging - Part 1

**Focus**: ML-Specific Metrics Collection, Time-Series Monitoring, Advanced Alerting  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master Prometheus metric collection patterns for ML workloads and infrastructure
- Learn Grafana dashboard design for ML system observability and performance monitoring
- Understand advanced alerting strategies for ML pipelines and model performance
- Analyze metric correlation and anomaly detection in production ML systems

---

## 📈 ML Metrics Theoretical Framework

### **ML System Observability Architecture**

Machine learning systems require comprehensive observability across multiple dimensions: infrastructure performance, model accuracy, data quality, and business impact.

**ML Observability Taxonomy:**
```
ML System Observability Layers:
1. Infrastructure Layer:
   - Compute resource utilization (CPU, GPU, memory)
   - Storage I/O performance and capacity
   - Network throughput and latency
   - Container and orchestration metrics

2. Data Layer:
   - Data pipeline throughput and latency
   - Data quality metrics and drift detection
   - Feature store performance and availability
   - Batch and streaming processing metrics

3. Model Layer:
   - Model performance metrics (accuracy, precision, recall)
   - Inference latency and throughput
   - Model drift and degradation indicators
   - A/B testing and experimentation metrics

4. Business Layer:
   - Conversion rates and business KPIs
   - Cost optimization and ROI metrics
   - User experience and satisfaction scores
   - Compliance and governance metrics

Observability Mathematical Model:
System_Health = f(Infrastructure_Metrics, Data_Quality, Model_Performance, Business_Impact)

Metric Correlation Analysis:
Correlation_Matrix = {
    Performance_Degradation: [CPU_Usage, Memory_Usage, Model_Accuracy, Latency],
    Data_Drift: [Feature_Distribution, Model_Accuracy, Prediction_Confidence],
    Cost_Efficiency: [Resource_Utilization, Throughput, Model_Complexity]
}

Alert Severity Calculation:
Alert_Severity = w1 * Impact_Score + w2 * Urgency_Score + w3 * Confidence_Score
Where:
- Impact_Score: Business impact of the issue (0-1)
- Urgency_Score: Time sensitivity of the issue (0-1)  
- Confidence_Score: Statistical confidence in the alert (0-1)
```

**Prometheus Metric Types for ML:**
```
ML-Specific Metric Categories:

1. Counter Metrics:
   - Total predictions served
   - Total training iterations
   - Total data processing records
   - Total model deployments
   - Total prediction errors

2. Gauge Metrics:
   - Current model accuracy
   - Active training jobs
   - Available GPU memory
   - Queue lengths
   - Resource utilization percentages

3. Histogram Metrics:
   - Prediction latency distribution
   - Training batch processing time
   - Data processing latency
   - Model size distribution
   - Request payload sizes

4. Summary Metrics:
   - Quantile-based latency metrics
   - Performance percentiles
   - Cost distribution summaries
   - Accuracy score distributions

Prometheus Metric Naming Convention:
ml_{component}_{metric_type}_{unit}

Examples:
- ml_inference_requests_total (counter)
- ml_model_accuracy_score (gauge)
- ml_prediction_latency_seconds (histogram)
- ml_training_duration_seconds (histogram)
- ml_data_pipeline_throughput_records_per_second (gauge)
```

---

## 🔍 Advanced Prometheus Configuration

### **ML-Specific Service Discovery**

**Kubernetes Service Discovery for ML Workloads:**
```
Prometheus Configuration for ML Systems:
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ml-production'
    environment: 'prod'

rule_files:
  - "ml_rules/*.yml"
  - "infrastructure_rules/*.yml"
  - "business_rules/*.yml"

scrape_configs:
# Kubernetes pods with ML annotations
- job_name: 'kubernetes-pods-ml'
  kubernetes_sd_configs:
  - role: pod
    namespaces:
      names:
      - ml-training
      - ml-inference
      - ml-platform
  
  relabel_configs:
  # Only scrape pods with prometheus.io/scrape annotation
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
    action: keep
    regex: true
  
  # Use custom metrics path if specified
  - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
    action: replace
    target_label: __metrics_path__
    regex: (.+)
  
  # Use custom port if specified
  - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
    action: replace
    regex: ([^:]+)(?::\d+)?;(\d+)
    replacement: $1:$2
    target_label: __address__
  
  # Add ML-specific labels
  - source_labels: [__meta_kubernetes_pod_label_ml_component]
    target_label: ml_component
  - source_labels: [__meta_kubernetes_pod_label_model_name]
    target_label: model_name
  - source_labels: [__meta_kubernetes_pod_label_model_version]
    target_label: model_version

# Training job metrics
- job_name: 'ml-training-jobs'
  kubernetes_sd_configs:
  - role: pod
    namespaces:
      names:
      - ml-training
  
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_job_type]
    action: keep
    regex: training
  - source_labels: [__meta_kubernetes_pod_name]
    target_label: training_job
  
  scrape_interval: 30s
  metrics_path: /metrics
  
# Inference service metrics
- job_name: 'ml-inference-services'
  kubernetes_sd_configs:
  - role: service
    namespaces:
      names:
      - ml-inference
  
  relabel_configs:
  - source_labels: [__meta_kubernetes_service_label_service_type]
    action: keep
    regex: inference
  - source_labels: [__meta_kubernetes_service_name]
    target_label: inference_service
  
  scrape_interval: 5s
  metrics_path: /metrics

# GPU monitoring
- job_name: 'gpu-metrics'
  static_configs:
  - targets: ['gpu-exporter:9400']
  scrape_interval: 10s
  
# Custom ML pipeline metrics
- job_name: 'ml-pipelines'
  consul_sd_configs:
  - server: 'consul.ml-platform.svc.cluster.local:8500'
    services: ['ml-pipeline', 'data-pipeline', 'feature-pipeline']
  
  relabel_configs:
  - source_labels: [__meta_consul_service]
    target_label: pipeline_type

# Cost monitoring
- job_name: 'cost-metrics'
  static_configs:
  - targets: ['cost-exporter:9090']
  scrape_interval: 300s  # 5 minutes

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

# Remote write for long-term storage
remote_write:
  - url: "https://prometheus-remote-write.monitoring.svc.cluster.local/api/v1/write"
    queue_config:
      max_samples_per_send: 10000
      batch_send_deadline: 5s
      min_shards: 4
      max_shards: 200
```

**Custom ML Metrics Exporters:**
```
Model Performance Exporter:
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

class MLModelExporter:
    def __init__(self, model_name, model_version, port=8000):
        self.model_name = model_name
        self.model_version = model_version
        
        # Define Prometheus metrics
        self.accuracy_gauge = Gauge(
            'ml_model_accuracy_score',
            'Current model accuracy score',
            ['model_name', 'model_version', 'dataset']
        )
        
        self.precision_gauge = Gauge(
            'ml_model_precision_score',
            'Current model precision score',
            ['model_name', 'model_version', 'dataset']
        )
        
        self.recall_gauge = Gauge(
            'ml_model_recall_score',
            'Current model recall score',
            ['model_name', 'model_version', 'dataset']
        )
        
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions made',
            ['model_name', 'model_version', 'status']
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Time spent on predictions',
            ['model_name', 'model_version'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.model_confidence = Histogram(
            'ml_prediction_confidence_score',
            'Distribution of prediction confidence scores',
            ['model_name', 'model_version'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        self.feature_drift_gauge = Gauge(
            'ml_feature_drift_score',
            'Feature drift detection score',
            ['model_name', 'model_version', 'feature_name']
        )
        
        self.data_quality_gauge = Gauge(
            'ml_data_quality_score',
            'Data quality score for input features',
            ['model_name', 'model_version', 'quality_metric']
        )
        
        # Start HTTP server
        start_http_server(port)
    
    def update_model_performance(self, y_true, y_pred, dataset_name):
        """Update model performance metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        self.accuracy_gauge.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            dataset=dataset_name
        ).set(accuracy)
        
        self.precision_gauge.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            dataset=dataset_name
        ).set(precision)
        
        self.recall_gauge.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            dataset=dataset_name
        ).set(recall)
    
    def record_prediction(self, prediction_time, confidence_score, status='success'):
        """Record individual prediction metrics"""
        self.prediction_counter.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            status=status
        ).inc()
        
        self.prediction_latency.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(prediction_time)
        
        if confidence_score is not None:
            self.model_confidence.labels(
                model_name=self.model_name,
                model_version=self.model_version
            ).observe(confidence_score)
    
    def update_feature_drift(self, feature_name, drift_score):
        """Update feature drift metrics"""
        self.feature_drift_gauge.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            feature_name=feature_name
        ).set(drift_score)
    
    def update_data_quality(self, quality_metrics):
        """Update data quality metrics"""
        for metric_name, score in quality_metrics.items():
            self.data_quality_gauge.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                quality_metric=metric_name
            ).set(score)

Training Job Metrics Exporter:
class TrainingJobExporter:
    def __init__(self, job_name, port=8001):
        self.job_name = job_name
        
        # Training progress metrics
        self.epoch_gauge = Gauge(
            'ml_training_current_epoch',
            'Current training epoch',
            ['job_name']
        )
        
        self.loss_gauge = Gauge(
            'ml_training_loss',
            'Current training loss',
            ['job_name', 'loss_type']
        )
        
        self.learning_rate_gauge = Gauge(
            'ml_training_learning_rate',
            'Current learning rate',
            ['job_name']
        )
        
        self.batch_processing_time = Histogram(
            'ml_training_batch_duration_seconds',
            'Time spent processing training batches',
            ['job_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.gpu_utilization_gauge = Gauge(
            'ml_training_gpu_utilization_percent',
            'GPU utilization during training',
            ['job_name', 'gpu_id']
        )
        
        self.memory_usage_gauge = Gauge(
            'ml_training_memory_usage_bytes',
            'Memory usage during training',
            ['job_name', 'memory_type']
        )
        
        self.samples_processed_counter = Counter(
            'ml_training_samples_processed_total',
            'Total number of training samples processed',
            ['job_name']
        )
        
        # Checkpoint metrics
        self.checkpoint_size_gauge = Gauge(
            'ml_training_checkpoint_size_bytes',
            'Size of training checkpoints',
            ['job_name']
        )
        
        self.checkpoint_save_time = Histogram(
            'ml_training_checkpoint_save_duration_seconds',
            'Time spent saving checkpoints',
            ['job_name']
        )
        
        start_http_server(port)
    
    def update_training_progress(self, epoch, losses, learning_rate):
        """Update training progress metrics"""
        self.epoch_gauge.labels(job_name=self.job_name).set(epoch)
        self.learning_rate_gauge.labels(job_name=self.job_name).set(learning_rate)
        
        for loss_type, loss_value in losses.items():
            self.loss_gauge.labels(
                job_name=self.job_name,
                loss_type=loss_type
            ).set(loss_value)
    
    def record_batch_processing(self, batch_time, batch_size):
        """Record batch processing metrics"""
        self.batch_processing_time.labels(job_name=self.job_name).observe(batch_time)
        self.samples_processed_counter.labels(job_name=self.job_name).inc(batch_size)
    
    def update_resource_usage(self, gpu_utilization, memory_usage):
        """Update resource usage metrics"""
        for gpu_id, utilization in gpu_utilization.items():
            self.gpu_utilization_gauge.labels(
                job_name=self.job_name,
                gpu_id=str(gpu_id)
            ).set(utilization)
        
        for memory_type, usage in memory_usage.items():
            self.memory_usage_gauge.labels(
                job_name=self.job_name,
                memory_type=memory_type
            ).set(usage)

Data Pipeline Metrics Exporter:
class DataPipelineExporter:
    def __init__(self, pipeline_name, port=8002):
        self.pipeline_name = pipeline_name
        
        # Pipeline throughput metrics
        self.records_processed_counter = Counter(
            'ml_pipeline_records_processed_total',
            'Total number of records processed',
            ['pipeline_name', 'stage', 'status']
        )
        
        self.processing_latency = Histogram(
            'ml_pipeline_processing_latency_seconds',
            'Time spent processing pipeline stages',
            ['pipeline_name', 'stage'],
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        )
        
        self.queue_size_gauge = Gauge(
            'ml_pipeline_queue_size',
            'Number of items in pipeline queues',
            ['pipeline_name', 'queue_name']
        )
        
        self.error_rate_gauge = Gauge(
            'ml_pipeline_error_rate',
            'Error rate in pipeline processing',
            ['pipeline_name', 'stage']
        )
        
        # Data quality metrics
        self.data_freshness_gauge = Gauge(
            'ml_pipeline_data_freshness_seconds',
            'Age of the most recent data processed',
            ['pipeline_name', 'data_source']
        )
        
        self.schema_violations_counter = Counter(
            'ml_pipeline_schema_violations_total',
            'Number of schema violations detected',
            ['pipeline_name', 'violation_type']
        )
        
        start_http_server(port)
    
    def record_processing(self, stage, processing_time, record_count, status='success'):
        """Record pipeline processing metrics"""
        self.records_processed_counter.labels(
            pipeline_name=self.pipeline_name,
            stage=stage,
            status=status
        ).inc(record_count)
        
        self.processing_latency.labels(
            pipeline_name=self.pipeline_name,
            stage=stage
        ).observe(processing_time)
    
    def update_queue_sizes(self, queue_sizes):
        """Update queue size metrics"""
        for queue_name, size in queue_sizes.items():
            self.queue_size_gauge.labels(
                pipeline_name=self.pipeline_name,
                queue_name=queue_name
            ).set(size)
    
    def update_data_freshness(self, data_sources):
        """Update data freshness metrics"""
        for source_name, age_seconds in data_sources.items():
            self.data_freshness_gauge.labels(
                pipeline_name=self.pipeline_name,
                data_source=source_name
            ).set(age_seconds)
```

---

## 📊 Advanced Grafana Dashboard Design

### **ML-Specific Dashboard Patterns**

**Model Performance Dashboard:**
```
Grafana Dashboard JSON Structure:
{
  "dashboard": {
    "id": null,
    "title": "ML Model Performance Dashboard",
    "tags": ["ml", "model", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Model Accuracy Over Time",
        "type": "stat",
        "targets": [
          {
            "expr": "ml_model_accuracy_score{model_name=\"$model_name\"}",
            "legendFormat": "{{dataset}} Accuracy",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "min": 0,
            "max": 1,
            "thresholds": {
              "mode": "absolute",
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.8},
                {"color": "green", "value": 0.9}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Prediction Latency Distribution",  
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(ml_prediction_latency_seconds_bucket{model_name=\"$model_name\"}[5m])",
            "legendFormat": "{{le}}",
            "refId": "A"
          }
        ],
        "options": {
          "yAxis": {"unit": "s", "logBase": 2},
          "colorMode": "spectrum"
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Predictions per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_predictions_total{model_name=\"$model_name\"}[1m])",
            "legendFormat": "{{status}} predictions/sec",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"unit": "reqps", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Feature Drift Detection",
        "type": "table",
        "targets": [
          {
            "expr": "ml_feature_drift_score{model_name=\"$model_name\"}",
            "format": "table",
            "refId": "A"
          }
        ],
        "transformations": [
          {
            "id": "organize",
            "options": {
              "excludeByName": {"Time": true, "__instance__": true},
              "renameByName": {
                "feature_name": "Feature",
                "Value": "Drift Score"
              }
            }
          }
        ],
        "fieldConfig": {
          "overrides": [
            {
              "matcher": {"id": "byName", "options": "Drift Score"},
              "properties": [
                {
                  "id": "thresholds",
                  "value": {
                    "mode": "absolute",
                    "steps": [
                      {"color": "green", "value": 0},
                      {"color": "yellow", "value": 0.3},
                      {"color": "red", "value": 0.7}
                    ]
                  }
                }
              ]
            }
          ]
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "templating": {
      "list": [
        {
          "name": "model_name",
          "type": "query",
          "query": "label_values(ml_model_accuracy_score, model_name)",
          "refresh": 1,
          "includeAll": false,
          "multi": false
        },
        {
          "name": "model_version",
          "type": "query", 
          "query": "label_values(ml_model_accuracy_score{model_name=\"$model_name\"}, model_version)",
          "refresh": 1,
          "includeAll": false,
          "multi": false
        }
      ]
    },
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "30s"
  }
}

Training Job Dashboard Configuration:
{
  "dashboard": {
    "title": "ML Training Job Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Training Loss",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_training_loss{job_name=\"$job_name\"}",
            "legendFormat": "{{loss_type}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"logBase": 10, "min": 0.001}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_training_gpu_utilization_percent{job_name=\"$job_name\"}",
            "legendFormat": "GPU {{gpu_id}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"unit": "percent", "max": 100, "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Batch Processing Time",
        "type": "histogram",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_training_batch_duration_seconds_bucket{job_name=\"$job_name\"}[5m]))",
            "legendFormat": "95th percentile",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.50, rate(ml_training_batch_duration_seconds_bucket{job_name=\"$job_name\"}[5m]))",
            "legendFormat": "50th percentile", 
            "refId": "B"
          }
        ],
        "yAxes": [
          {"unit": "s", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_training_memory_usage_bytes{job_name=\"$job_name\"}",
            "legendFormat": "{{memory_type}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"unit": "bytes", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ]
  }
}

Data Pipeline Dashboard:
{
  "dashboard": {
    "title": "ML Data Pipeline Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Pipeline Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_pipeline_records_processed_total{pipeline_name=\"$pipeline_name\", status=\"success\"}[5m])",
            "legendFormat": "{{stage}} records/sec",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"unit": "rps", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Processing Latency by Stage",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_pipeline_processing_latency_seconds_bucket{pipeline_name=\"$pipeline_name\"}[5m]))",
            "legendFormat": "{{stage}} - 95th percentile",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"unit": "s", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(ml_pipeline_records_processed_total{pipeline_name=\"$pipeline_name\", status=\"error\"}[5m]) / rate(ml_pipeline_records_processed_total{pipeline_name=\"$pipeline_name\"}[5m])",
            "refId": "A"
          }
        ],
        "valueName": "current",
        "format": "percentunit",
        "thresholds": "0.01,0.05",
        "colorBackground": true,
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Data Freshness",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_pipeline_data_freshness_seconds{pipeline_name=\"$pipeline_name\"}",
            "legendFormat": "{{data_source}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {"unit": "s", "min": 0}
        ],
        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 8}
      }
    ]
  }
}
```

**Advanced Dashboard Features:**
```
Custom Grafana Plugins for ML:
1. ML Model Comparison Plugin:
   - Side-by-side model performance comparison
   - A/B testing visualization
   - Statistical significance testing
   - ROC curve and confusion matrix plots

2. Feature Importance Visualization:
   - Dynamic feature importance charts
   - Feature correlation heatmaps
   - SHAP value integration
   - Feature drift detection plots

3. Cost Optimization Dashboard:
   - Resource utilization vs. performance trade-offs
   - Cost per prediction metrics
   - Spot instance utilization tracking
   - Budget alert integration

Dashboard Automation Script:
#!/bin/bash
# Automated dashboard provisioning
create_ml_dashboards() {
    local grafana_url=$1
    local api_key=$2
    
    # Create ML monitoring folder
    curl -X POST \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d '{"title":"ML Monitoring","uid":"ml-monitoring"}' \
        "$grafana_url/api/folders"
    
    # Deploy model performance dashboard
    curl -X POST \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d @model-performance-dashboard.json \
        "$grafana_url/api/dashboards/db"
    
    # Deploy training job dashboard
    curl -X POST \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d @training-job-dashboard.json \
        "$grafana_url/api/dashboards/db"
    
    # Deploy data pipeline dashboard
    curl -X POST \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d @data-pipeline-dashboard.json \
        "$grafana_url/api/dashboards/db"
    
    # Create alerts
    create_ml_alerts "$grafana_url" "$api_key"
}

create_ml_alerts() {
    local grafana_url=$1
    local api_key=$2
    
    # Model accuracy alert
    curl -X POST \
        -H "Authorization: Bearer $api_key" \
        -H "Content-Type: application/json" \
        -d '{
            "alert": {
                "name": "Model Accuracy Drop",
                "message": "Model accuracy has dropped below threshold",
                "frequency": "60s",
                "conditions": [
                    {
                        "query": {
                            "queryType": "",
                            "refId": "A",
                            "model": {
                                "expr": "ml_model_accuracy_score < 0.85",
                                "intervalMs": 1000,
                                "maxDataPoints": 43200
                            }
                        },
                        "reducer": {
                            "type": "last",
                            "params": []
                        },
                        "evaluator": {
                            "type": "lt",
                            "params": [0.85]
                        }
                    }
                ],
                "executionErrorState": "alerting",
                "noDataState": "no_data",
                "for": "5m"
            }
        }' \
        "$grafana_url/api/alerts"
}
```

---

## 🚨 Intelligent Alerting Strategies

### **ML-Specific Alert Rules**

**Prometheus Alert Rules for ML Systems:**
```
ML Alert Rules Configuration:
# /etc/prometheus/ml_rules.yml
groups:
- name: ml_model_performance
  rules:
  - alert: ModelAccuracyDrop
    expr: ml_model_accuracy_score < 0.85
    for: 5m
    labels:
      severity: critical
      service: ml-model
    annotations:
      summary: "Model {{ $labels.model_name }} accuracy dropped"
      description: "Model {{ $labels.model_name }} version {{ $labels.model_version }} accuracy is {{ $value }}, below threshold of 0.85"
      runbook_url: "https://wiki.company.com/ml-runbooks/model-accuracy-drop"
  
  - alert: HighPredictionLatency
    expr: histogram_quantile(0.95, rate(ml_prediction_latency_seconds_bucket[5m])) > 0.5
    for: 2m
    labels:
      severity: warning
      service: ml-inference
    annotations:
      summary: "High prediction latency for {{ $labels.model_name }}"
      description: "95th percentile latency is {{ $value }}s for model {{ $labels.model_name }}"
  
  - alert: PredictionErrorRate
    expr: rate(ml_predictions_total{status="error"}[5m]) / rate(ml_predictions_total[5m]) > 0.05
    for: 3m
    labels:
      severity: critical
      service: ml-inference
    annotations:
      summary: "High prediction error rate"
      description: "Error rate is {{ $value | humanizePercentage }} for model {{ $labels.model_name }}"
  
  - alert: FeatureDriftDetected
    expr: ml_feature_drift_score > 0.7
    for: 10m
    labels:
      severity: warning
      service: ml-model
    annotations:
      summary: "Feature drift detected"
      description: "Feature {{ $labels.feature_name }} drift score is {{ $value }} for model {{ $labels.model_name }}"

- name: ml_training_jobs
  rules:
  - alert: TrainingJobStalled
    expr: increase(ml_training_current_epoch[30m]) == 0
    for: 30m
    labels:
      severity: warning
      service: ml-training
    annotations:
      summary: "Training job {{ $labels.job_name }} appears stalled"
      description: "No progress in training epochs for 30 minutes"
  
  - alert: TrainingLossExplosion
    expr: ml_training_loss > 1000
    for: 1m
    labels:
      severity: critical
      service: ml-training
    annotations:
      summary: "Training loss explosion detected"
      description: "Training loss is {{ $value }} for job {{ $labels.job_name }}"
  
  - alert: GPUUtilizationLow
    expr: avg_over_time(ml_training_gpu_utilization_percent[15m]) < 30
    for: 15m
    labels:
      severity: warning
      service: ml-training
    annotations:
      summary: "Low GPU utilization during training"
      description: "Average GPU utilization is {{ $value }}% for job {{ $labels.job_name }}"
  
  - alert: TrainingOOM
    expr: increase(ml_training_memory_usage_bytes[5m]) > 0.9 * kube_node_status_allocatable{resource="memory"}
    for: 1m
    labels:
      severity: critical
      service: ml-training
    annotations:
      summary: "Training job approaching memory limit"
      description: "Memory usage is {{ $value | humanizeBytes }} for job {{ $labels.job_name }}"

- name: ml_data_pipelines
  rules:
  - alert: DataPipelineStalled
    expr: rate(ml_pipeline_records_processed_total[10m]) == 0
    for: 10m
    labels:
      severity: critical
      service: ml-pipeline
    annotations:
      summary: "Data pipeline {{ $labels.pipeline_name }} stalled"
      description: "No records processed in the last 10 minutes"
  
  - alert: DataQualityDegraded
    expr: ml_data_quality_score < 0.8
    for: 5m
    labels:
      severity: warning
      service: ml-pipeline
    annotations:
      summary: "Data quality degraded"
      description: "Data quality score is {{ $value }} for {{ $labels.quality_metric }}"
  
  - alert: DataFreshnessIssue
    expr: ml_pipeline_data_freshness_seconds > 3600
    for: 5m
    labels:
      severity: warning
      service: ml-pipeline
    annotations:
      summary: "Stale data detected"
      description: "Data is {{ $value | humanizeDuration }} old from {{ $labels.data_source }}"
  
  - alert: HighPipelineErrorRate
    expr: rate(ml_pipeline_records_processed_total{status="error"}[5m]) / rate(ml_pipeline_records_processed_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
      service: ml-pipeline
    annotations:
      summary: "High error rate in pipeline {{ $labels.pipeline_name }}"
      description: "Error rate is {{ $value | humanizePercentage }} in stage {{ $labels.stage }}"

- name: ml_infrastructure
  rules:
  - alert: MLNodeResourceExhaustion
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.9
    for: 5m
    labels:
      severity: critical
      service: ml-infrastructure
    annotations:
      summary: "ML node {{ $labels.instance }} running out of memory"
      description: "Memory usage is above 90% on ML node {{ $labels.instance }}"
  
  - alert: GPUTemperatureHigh
    expr: nvidia_gpu_temperature_celsius > 85
    for: 5m
    labels:
      severity: warning
      service: ml-infrastructure
    annotations:
      summary: "High GPU temperature"
      description: "GPU {{ $labels.gpu }} temperature is {{ $value }}°C on node {{ $labels.instance }}"
  
  - alert: StorageSpaceRunningOut
    expr: (node_filesystem_avail_bytes{mountpoint=~"/data.*"} / node_filesystem_size_bytes{mountpoint=~"/data.*"}) < 0.1
    for: 5m
    labels:
      severity: critical
      service: ml-infrastructure
    annotations:
      summary: "ML storage space critically low"
      description: "Only {{ $value | humanizePercentage }} space remaining on {{ $labels.mountpoint }}"
```

**Advanced Alerting Logic:**
```
Multi-Condition Alert Logic:
class MLAlertManager:
    def __init__(self):
        self.alert_conditions = {}
        self.alert_history = {}
        self.escalation_rules = {}
    
    def evaluate_ml_alert(self, metric_name, current_value, thresholds, context):
        """Evaluate ML-specific alert conditions with context awareness"""
        
        # Get historical context
        historical_values = self.get_historical_values(metric_name, context, window='1h')
        
        # Calculate dynamic thresholds based on historical patterns
        dynamic_threshold = self.calculate_dynamic_threshold(
            historical_values, 
            base_threshold=thresholds['base'],
            seasonality=context.get('seasonality', False)
        )
        
        # Check multiple conditions
        conditions = {
            'absolute_threshold': current_value < thresholds['absolute'],
            'dynamic_threshold': current_value < dynamic_threshold,
            'trend_degradation': self.check_trend_degradation(historical_values),
            'anomaly_detection': self.detect_anomaly(current_value, historical_values),
            'confidence_interval': self.check_confidence_interval(current_value, historical_values)
        }
        
        # Calculate composite alert score
        alert_score = self.calculate_composite_score(conditions, context)
        
        # Determine alert severity
        severity = self.determine_severity(alert_score, context)
        
        if alert_score > 0.7:  # Alert threshold
            return self.create_alert(
                metric_name=metric_name,
                current_value=current_value,
                conditions=conditions,
                severity=severity,
                context=context
            )
        
        return None
    
    def calculate_dynamic_threshold(self, historical_values, base_threshold, seasonality=False):
        """Calculate dynamic thresholds based on historical patterns"""
        if len(historical_values) < 10:
            return base_threshold
        
        if seasonality:
            # Account for seasonal patterns
            seasonal_component = self.extract_seasonal_component(historical_values)
            trend_component = self.extract_trend_component(historical_values)
            adjusted_threshold = base_threshold * (1 + seasonal_component + trend_component)
        else:
            # Use statistical approach
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            adjusted_threshold = max(base_threshold, mean - 2 * std)
        
        return adjusted_threshold
    
    def check_trend_degradation(self, values, window=10):
        """Check if there's a degrading trend in recent values"""
        if len(values) < window:
            return False
        
        recent_values = values[-window:]
        trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        # Negative slope indicates degradation
        return trend_slope < -0.01
    
    def detect_anomaly(self, current_value, historical_values):
        """Detect if current value is anomalous using statistical methods"""
        if len(historical_values) < 30:
            return False
        
        # Use isolation forest for anomaly detection
        from sklearn.ensemble import IsolationForest
        
        values_array = np.array(historical_values + [current_value]).reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(values_array)
        
        # Check if current value is anomalous
        return predictions[-1] == -1
    
    def calculate_composite_score(self, conditions, context):
        """Calculate composite alert score based on multiple conditions"""
        weights = {
            'absolute_threshold': 0.3,
            'dynamic_threshold': 0.25,
            'trend_degradation': 0.2,
            'anomaly_detection': 0.15,
            'confidence_interval': 0.1
        }
        
        # Adjust weights based on context
        if context.get('critical_model', False):
            weights['absolute_threshold'] *= 1.5
            weights['dynamic_threshold'] *= 1.3
        
        if context.get('real_time_serving', False):
            weights['trend_degradation'] *= 1.2
        
        # Calculate weighted score
        score = sum(
            weights[condition] * (1.0 if triggered else 0.0)
            for condition, triggered in conditions.items()
        )
        
        return min(score, 1.0)  # Cap at 1.0

Business Impact Alert Correlation:
class BusinessImpactCorrelator:
    def __init__(self):
        self.business_metrics = {}
        self.ml_metrics = {}
        self.correlation_models = {}
    
    def correlate_ml_business_impact(self, ml_alert, business_context):
        """Correlate ML alerts with business impact"""
        
        # Get relevant business metrics
        business_metrics = self.get_business_metrics(
            time_range=business_context['time_range'],
            segments=business_context.get('segments', [])
        )
        
        # Calculate correlation between ML metrics and business outcomes
        correlations = {}
        for metric_name, values in business_metrics.items():
            ml_metric_values = self.get_ml_metric_values(
                ml_alert['metric_name'],
                len(values)
            )
            
            correlation = np.corrcoef(ml_metric_values, values)[0, 1]
            correlations[metric_name] = correlation
        
        # Estimate business impact
        impact_score = self.estimate_business_impact(
            ml_alert, correlations, business_context
        )
        
        # Update alert with business context
        ml_alert['business_impact'] = {
            'score': impact_score,
            'correlations': correlations,
            'estimated_revenue_impact': self.estimate_revenue_impact(impact_score),
            'affected_users': self.estimate_affected_users(impact_score)
        }
        
        return ml_alert
    
    def estimate_revenue_impact(self, impact_score):
        """Estimate potential revenue impact based on ML performance degradation"""
        # This would be customized based on business model
        base_revenue_per_hour = 10000  # Example: $10K/hour
        impact_multiplier = impact_score * 0.5  # 50% max revenue impact
        
        return base_revenue_per_hour * impact_multiplier
    
    def estimate_affected_users(self, impact_score):
        """Estimate number of users affected by ML performance issues"""
        total_active_users = 100000  # Example: 100K active users
        affected_percentage = impact_score * 0.3  # Max 30% of users affected
        
        return int(total_active_users * affected_percentage)
```

This comprehensive framework for Prometheus and Grafana ML metrics monitoring provides the theoretical foundations and practical strategies for building robust observability systems for machine learning platforms. The key insight is that ML systems require specialized monitoring approaches that account for model performance, data quality, and business impact correlation alongside traditional infrastructure metrics.