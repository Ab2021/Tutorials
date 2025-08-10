# Day 6.4: Performance Monitoring & SLA Management

## 📊 Model Serving & Production Inference - Part 4

**Focus**: Real-Time Monitoring, SLA Definition, Alerting Systems, Performance Optimization  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master performance monitoring strategies for production ML systems
- Learn SLA definition, measurement, and enforcement techniques
- Understand alerting systems and incident response frameworks
- Analyze performance optimization based on monitoring insights

---

## 📈 Performance Monitoring Framework

### **Multi-Dimensional Monitoring Strategy**

Production ML systems require monitoring across multiple dimensions to ensure reliable operation and early detection of issues.

**Monitoring Taxonomy:**
```
Infrastructure Metrics:
- Compute: CPU utilization, GPU utilization, memory usage
- Network: Latency, bandwidth, packet loss, connection counts
- Storage: I/O throughput, disk usage, cache hit rates
- Application: Thread pools, connection pools, garbage collection

Model Performance Metrics:
- Accuracy: Online accuracy estimation, drift detection
- Prediction Quality: Confidence distributions, prediction variance
- Feature Health: Feature availability, feature drift, data quality
- Model Behavior: Prediction patterns, edge case handling

Business Metrics:
- User Experience: Response times, error rates, availability
- Business Impact: Revenue attribution, conversion rates, engagement
- Operational Efficiency: Cost per prediction, resource utilization
- Compliance: Audit trails, data governance adherence

Temporal Characteristics:
- Real-time: <1 second latency, immediate alerting
- Near real-time: 1-60 seconds, operational dashboards
- Batch: Minutes to hours, trend analysis and reporting
- Historical: Long-term patterns, capacity planning
```

**Metrics Collection Architecture:**
```
Collection Strategy:
Push Model: Services actively send metrics to collectors
- Advantages: Real-time data, fine-grained control
- Disadvantages: Network overhead, complex failure handling

Pull Model: Collectors scrape metrics from services
- Advantages: Simple service implementation, reliable collection
- Disadvantages: Polling overhead, potential data loss

Hybrid Model: Critical metrics pushed, others pulled
- Balances real-time needs with operational simplicity

Metric Aggregation Pipeline:
Raw Metrics → Collection → Aggregation → Storage → Visualization → Alerting

Aggregation Functions:
- Statistical: mean, median, percentiles (P50, P95, P99)
- Temporal: rate, increase, derivative
- Logical: max, min, count, sum
- Custom: business-specific calculations
```

### **SLA Definition and Measurement**

**Service Level Objectives (SLOs) Framework:**
```
SLO Categories for ML Systems:

Availability SLOs:
- Definition: Percentage of time system is operational
- Measurement: (Total time - Downtime) / Total time
- Target: 99.9% (8.76 hours downtime/year)
- Error Budget: 0.1% (43.8 minutes/month)

Latency SLOs:
- Definition: Response time percentiles
- Measurement: P99 latency over rolling time window
- Target: P99 < 100ms, P95 < 50ms, P50 < 20ms
- Error Budget: Percentage of requests exceeding target

Throughput SLOs:
- Definition: Requests processed per unit time
- Measurement: Successful requests per second
- Target: 10,000 RPS sustained, 15,000 RPS peak
- Error Budget: Periods below minimum threshold

Accuracy SLOs:
- Definition: Model prediction quality
- Measurement: Online accuracy estimation
- Target: Accuracy > 95%, F1-score > 0.9
- Error Budget: Time periods below accuracy threshold
```

**SLO Mathematical Framework:**
```
Error Budget Calculation:
Error_Budget = (1 - SLO_Target) × Time_Period

Example for 99.9% availability SLO over 30 days:
Error_Budget = (1 - 0.999) × 30 days = 0.001 × 30 = 0.03 days = 43.2 minutes

Burn Rate Analysis:
Burn_Rate = Actual_Error_Rate / Error_Budget_Rate

Where:
Actual_Error_Rate = Failed_Requests / Total_Requests (current period)
Error_Budget_Rate = (1 - SLO_Target) = allowed error rate

Burn Rate Interpretation:
- Burn_Rate = 1: Consuming error budget at exactly SLO rate
- Burn_Rate > 1: Consuming error budget faster than sustainable
- Burn_Rate < 1: Under SLO target, building error budget reserve

Time to Exhaustion:
Time_Remaining = Current_Error_Budget / (Burn_Rate × Error_Budget_Rate × Time_Unit)
```

**Multi-Window SLO Monitoring:**
```
SLO Window Strategy:
Short Window (5 minutes): Fast alerting for critical issues
Medium Window (1 hour): Operational decision making
Long Window (28 days): SLO compliance reporting

Multi-Burn-Rate Alerting:
def calculate_alert_conditions(burn_rates, windows):
    alerts = []
    
    # Fast burn: 2% budget in 1 hour (14.4x burn rate)
    if burn_rates['1h'] > 14.4 and burn_rates['5m'] > 14.4:
        alerts.append(('page', 'Critical: Fast burn detected'))
    
    # Moderate burn: 5% budget in 6 hours (6x burn rate)  
    elif burn_rates['6h'] > 6 and burn_rates['30m'] > 6:
        alerts.append(('ticket', 'Warning: Moderate burn detected'))
    
    # Slow burn: 10% budget in 3 days (2.4x burn rate)
    elif burn_rates['3d'] > 2.4 and burn_rates['6h'] > 2.4:
        alerts.append(('email', 'Notice: Slow burn detected'))
    
    return alerts

Implementation:
class SLOMonitor:
    def __init__(self, slo_target, windows):
        self.slo_target = slo_target
        self.windows = windows
        self.error_budget_rate = 1 - slo_target
    
    def check_burn_rates(self, metrics):
        burn_rates = {}
        
        for window in self.windows:
            error_rate = self.calculate_error_rate(metrics, window)
            burn_rate = error_rate / self.error_budget_rate
            burn_rates[window] = burn_rate
        
        return self.calculate_alert_conditions(burn_rates, self.windows)
```

---

## 🚨 Alerting and Incident Response

### **Intelligent Alerting Systems**

**Alert Fatigue Reduction:**
```
Alert Classification:
Priority Levels:
- P0 (Critical): Customer-impacting, immediate response required
- P1 (High): Service degradation, response within 1 hour
- P2 (Medium): Performance issues, response within 24 hours
- P3 (Low): Monitoring, maintenance, or improvement items

Alert Routing Strategy:
def route_alert(alert_severity, service, time_of_day):
    if alert_severity == 'P0':
        return ['on_call_engineer', 'backup_engineer', 'incident_commander']
    elif alert_severity == 'P1':
        if is_business_hours(time_of_day):
            return ['primary_team', 'team_lead']
        else:
            return ['on_call_engineer']
    elif alert_severity == 'P2':
        return ['slack_channel', 'email_group']
    else:
        return ['ticket_system']

Alert Suppression and Grouping:
- Temporal suppression: Avoid duplicate alerts within time window
- Causal suppression: Suppress downstream effects of root cause
- Maintenance windows: Suppress expected alerts during maintenance
- Dependency-aware: Group alerts from related services

Noise Reduction Techniques:
1. Hysteresis: Different thresholds for triggering vs resolving alerts
2. Rate limiting: Maximum alerts per time period per service
3. Anomaly detection: Machine learning-based alerting
4. Contextual alerts: Consider business context (traffic patterns, deployments)
```

**Anomaly Detection for Alerting:**
```
Statistical Anomaly Detection:
# Moving average with confidence intervals
def detect_anomaly_statistical(metric_values, window_size=20, threshold=3):
    if len(metric_values) < window_size:
        return False, 0
    
    recent_values = metric_values[-window_size:]
    mean = np.mean(recent_values)
    std = np.std(recent_values)
    
    current_value = metric_values[-1]
    z_score = abs(current_value - mean) / std if std > 0 else 0
    
    return z_score > threshold, z_score

# Seasonal decomposition
def detect_seasonal_anomaly(metric_values, seasonality_period=24*7):  # Weekly
    if len(metric_values) < seasonality_period * 2:
        return False, 0
    
    # Decompose into trend, seasonal, residual components
    decomposition = seasonal_decompose(metric_values, period=seasonality_period)
    residuals = decomposition.resid
    
    # Detect anomalies in residuals
    recent_residual = residuals[-1]
    residual_threshold = 3 * np.std(residuals[:-1])
    
    is_anomaly = abs(recent_residual) > residual_threshold
    return is_anomaly, recent_residual / residual_threshold

Machine Learning Anomaly Detection:
class IsolationForestAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination)
        self.is_trained = False
    
    def train(self, training_data):
        # training_data: historical normal behavior
        self.model.fit(training_data)
        self.is_trained = True
    
    def detect_anomaly(self, current_metrics):
        if not self.is_trained:
            return False, 0
        
        anomaly_score = self.model.decision_function([current_metrics])[0]
        is_anomaly = self.model.predict([current_metrics])[0] == -1
        
        return is_anomaly, anomaly_score
```

### **Incident Response Framework**

**Incident Management Process:**
```
Incident Lifecycle:
1. Detection: Monitoring systems detect issue
2. Response: On-call engineer investigates
3. Mitigation: Immediate actions to reduce impact
4. Resolution: Permanent fix implemented
5. Recovery: Service restored to normal operation
6. Post-mortem: Learn from incident and improve

Incident Severity Classification:
Severity 1 (SEV-1): Complete service outage
- Customer impact: All users affected
- Response time: Immediate (< 15 minutes)
- Resolution target: < 1 hour
- Escalation: Automatic to incident commander

Severity 2 (SEV-2): Partial service degradation  
- Customer impact: Some users affected
- Response time: < 30 minutes
- Resolution target: < 4 hours
- Escalation: Team lead involvement

Severity 3 (SEV-3): Performance issues
- Customer impact: Degraded experience
- Response time: < 2 hours (business hours)
- Resolution target: < 24 hours
- Escalation: Standard team process

Incident Response Playbooks:
def execute_incident_response(incident_type, severity):
    playbook = get_playbook(incident_type)
    
    if severity == 'SEV-1':
        # Immediate actions
        trigger_incident_commander()
        establish_communication_bridge()
        begin_customer_communication()
    
    # Execute diagnosis steps
    for step in playbook.diagnosis_steps:
        result = execute_step(step)
        log_step_result(step, result)
        
        if result.indicates_root_cause():
            break
    
    # Execute mitigation steps
    for mitigation in playbook.mitigation_actions:
        if mitigation.condition_met(current_state):
            execute_mitigation(mitigation)
```

**Automated Incident Response:**
```
Self-Healing Systems:
class AutomatedIncidentResponse:
    def __init__(self):
        self.response_rules = []
        self.safety_limits = SafetyLimits()
    
    def register_response_rule(self, condition, action, safety_check):
        self.response_rules.append({
            'condition': condition,
            'action': action,
            'safety_check': safety_check
        })
    
    def handle_alert(self, alert):
        for rule in self.response_rules:
            if rule['condition'](alert):
                if rule['safety_check'](alert):
                    self.execute_automated_response(rule['action'], alert)
                else:
                    self.escalate_to_human(alert, "Safety check failed")
                break

Common Automated Responses:
1. Circuit Breaker Activation:
   - Condition: Error rate > 10%
   - Action: Enable circuit breaker
   - Safety: Only if fallback available

2. Auto-scaling:
   - Condition: CPU > 80% for 5 minutes
   - Action: Scale out by 50%
   - Safety: Within cost limits

3. Traffic Shifting:
   - Condition: Latency P99 > 500ms
   - Action: Shift traffic to healthy instances
   - Safety: Minimum capacity maintained

4. Model Rollback:
   - Condition: Accuracy drop > 5%
   - Action: Revert to previous model version
   - Safety: Previous version available and tested

Chaos Engineering Integration:
def scheduled_chaos_experiment(experiment_config):
    # Verify system health before experiment
    if not system_health_check():
        skip_experiment("System not healthy")
        return
    
    # Execute controlled failure
    failure_injector = FailureInjector(experiment_config)
    failure_injector.inject_failure()
    
    # Monitor system response
    monitor_duration = experiment_config.get('duration', 300)  # 5 minutes
    for elapsed in range(0, monitor_duration, 30):
        health_metrics = collect_health_metrics()
        
        if health_metrics.indicates_cascade_failure():
            failure_injector.stop_experiment()
            trigger_recovery_procedures()
            break
    
    # Analyze results
    experiment_results = analyze_chaos_experiment_results(experiment_config)
    update_incident_response_playbooks(experiment_results)
```

---

## 🔍 Performance Analysis and Optimization

### **Root Cause Analysis Framework**

**Systematic Performance Investigation:**
```
Performance Debugging Methodology:
1. Symptom Identification: What is the observed problem?
2. Metric Correlation: Which metrics changed together?
3. Timeline Analysis: When did the problem start?
4. Component Isolation: Which system component is affected?
5. Hypothesis Formation: What could cause this pattern?
6. Hypothesis Testing: Can we reproduce or verify the cause?
7. Resolution Implementation: Fix the root cause

Correlation Analysis:
def find_correlated_metrics(target_metric, all_metrics, time_window):
    correlations = {}
    
    target_values = get_metric_values(target_metric, time_window)
    
    for metric_name, metric_values in all_metrics.items():
        if metric_name == target_metric:
            continue
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(target_values, metric_values)[0, 1]
        
        if abs(correlation) > 0.7:  # Strong correlation threshold
            correlations[metric_name] = correlation
    
    # Sort by absolute correlation strength
    return sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

Change Point Detection:
def detect_change_points(metric_values, min_segment_length=10):
    # Use PELT (Pruned Exact Linear Time) algorithm
    change_points = []
    
    # Simplified implementation - would use more sophisticated algorithm
    for i in range(min_segment_length, len(metric_values) - min_segment_length):
        before_segment = metric_values[max(0, i-min_segment_length):i]
        after_segment = metric_values[i:min(len(metric_values), i+min_segment_length)]
        
        # Statistical test for change point
        statistic, p_value = ttest_ind(before_segment, after_segment)
        
        if p_value < 0.01:  # Significant change
            change_points.append({
                'timestamp': i,
                'statistic': statistic,
                'p_value': p_value
            })
    
    return change_points
```

**Performance Profiling Integration:**
```
Distributed Tracing for ML Systems:
class MLDistributedTracer:
    def __init__(self, service_name):
        self.service_name = service_name
        self.tracer = init_tracer(service_name)
    
    def trace_inference_request(self, request_id, model_version):
        with self.tracer.start_span('ml_inference') as span:
            span.set_attribute('request_id', request_id)
            span.set_attribute('model_version', model_version)
            
            # Preprocessing span
            with tracer.start_span('preprocessing', parent=span) as prep_span:
                preprocessing_result = preprocess_data(request.data)
                prep_span.set_attribute('features_extracted', len(preprocessing_result))
            
            # Model inference span
            with tracer.start_span('model_inference', parent=span) as inf_span:
                prediction = model.predict(preprocessing_result)
                inf_span.set_attribute('prediction_confidence', prediction.confidence)
                inf_span.set_attribute('model_latency_ms', inf_span.duration_ms)
            
            # Postprocessing span
            with tracer.start_span('postprocessing', parent=span) as post_span:
                final_result = postprocess_prediction(prediction)
                post_span.set_attribute('result_size_bytes', len(final_result))
            
            return final_result

Custom Metrics Integration:
@measure_time('model_inference_duration')
@count_calls('model_inference_requests')
def predict_with_monitoring(model, features):
    with MetricsCollector() as metrics:
        # Add custom metrics
        metrics.gauge('feature_count', len(features))
        metrics.gauge('model_memory_usage', model.get_memory_usage())
        
        # Execute prediction
        start_time = time.time()
        prediction = model.predict(features)
        inference_time = time.time() - start_time
        
        # Record custom metrics
        metrics.histogram('inference_latency_ms', inference_time * 1000)
        metrics.gauge('prediction_confidence', prediction.confidence)
        
        return prediction
```

### **Capacity Planning and Scaling**

**Predictive Capacity Planning:**
```
Traffic Forecasting:
def forecast_traffic_demand(historical_data, forecast_horizon_days=30):
    # Decompose time series
    decomposition = seasonal_decompose(historical_data, period=24*7)  # Weekly seasonality
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    
    # Fit trend model (linear regression)
    X = np.arange(len(trend)).reshape(-1, 1)
    trend_model = LinearRegression().fit(X[~np.isnan(trend)], trend[~np.isnan(trend)])
    
    # Forecast future trend
    future_X = np.arange(len(historical_data), len(historical_data) + forecast_horizon_days*24).reshape(-1, 1)
    future_trend = trend_model.predict(future_X)
    
    # Add seasonal component
    seasonal_cycle = seasonal[-24*7:]  # Last week's seasonal pattern
    future_seasonal = np.tile(seasonal_cycle, forecast_horizon_days // 7 + 1)[:len(future_trend)]
    
    forecast = future_trend + future_seasonal
    
    # Add confidence intervals
    residuals = historical_data - (trend + seasonal)
    residual_std = np.std(residuals[~np.isnan(residuals)])
    
    confidence_upper = forecast + 1.96 * residual_std  # 95% confidence
    confidence_lower = forecast - 1.96 * residual_std
    
    return {
        'forecast': forecast,
        'upper_bound': confidence_upper,
        'lower_bound': confidence_lower
    }

Resource Capacity Planning:
def calculate_required_capacity(traffic_forecast, performance_requirements):
    # Convert traffic to resource requirements
    max_daily_traffic = np.max(traffic_forecast['upper_bound'])
    
    # Calculate required compute capacity
    requests_per_second = max_daily_traffic / (24 * 3600)
    latency_requirement = performance_requirements['max_latency_ms']
    
    # Little's Law: N = λ × W (concurrency = arrival_rate × service_time)
    service_time_seconds = latency_requirement / 1000
    required_concurrency = requests_per_second * service_time_seconds
    
    # Add buffer for safety margin
    safety_factor = 1.3  # 30% buffer
    total_required_capacity = required_concurrency * safety_factor
    
    # Convert to infrastructure units
    capacity_per_instance = performance_requirements.get('capacity_per_instance', 100)
    required_instances = math.ceil(total_required_capacity / capacity_per_instance)
    
    return {
        'peak_rps': requests_per_second,
        'required_concurrency': required_concurrency,
        'safety_margin': safety_factor,
        'required_instances': required_instances,
        'cost_estimate': required_instances * performance_requirements.get('cost_per_instance_hour', 1.0)
    }

Auto-scaling Configuration:
def generate_autoscaling_config(capacity_analysis, current_metrics):
    base_config = {
        'min_replicas': max(2, capacity_analysis['required_instances'] // 4),  # Minimum viable capacity
        'max_replicas': capacity_analysis['required_instances'] * 2,  # Handle unexpected spikes
        'target_cpu_utilization': 70,  # Leave headroom for processing spikes
        'scale_up_period': '2m',  # Quick scale-up for user experience
        'scale_down_period': '10m',  # Conservative scale-down to avoid thrashing
    }
    
    # Custom metrics-based scaling
    custom_metrics = [
        {
            'metric': 'requests_per_second',
            'target': capacity_analysis['peak_rps'] * 0.8,  # Scale before saturation
            'type': 'external'
        },
        {
            'metric': 'queue_length',
            'target': 50,  # Maximum acceptable queue length
            'type': 'pods'
        }
    ]
    
    return {**base_config, 'custom_metrics': custom_metrics}
```

This comprehensive framework for performance monitoring and SLA management provides the foundation for reliable, observable, and well-governed ML systems in production. The key insight is that effective monitoring requires a multi-layered approach combining infrastructure metrics, model performance indicators, and business outcomes with intelligent alerting and automated response capabilities.