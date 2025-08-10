# Day 8.5: Security Automation & Certificate Management

## 🔒 Infrastructure as Code & Automation - Part 5

**Focus**: Security Automation, Certificate Lifecycle Management, Zero-Trust Architecture  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master automated security controls and threat detection for ML infrastructure
- Learn comprehensive certificate lifecycle management and PKI automation
- Understand zero-trust architecture implementation and service mesh security
- Analyze vulnerability management and automated remediation strategies

---

## 🛡️ Security Automation Framework

### **Automated Security Controls**

Security automation in ML infrastructure requires comprehensive, multi-layered approaches that address the unique attack vectors and compliance requirements of AI/ML systems.

**Security Control Taxonomy:**
```
ML Security Control Matrix:
1. Infrastructure Security:
   - Network segmentation and micro-segmentation
   - Container image vulnerability scanning
   - Kubernetes security policy enforcement
   - Cloud resource configuration validation

2. Data Security:
   - Data-at-rest encryption and key management
   - Data-in-transit encryption and certificate management
   - Data access logging and anomaly detection
   - Privacy-preserving computation controls

3. Model Security:
   - Model poisoning and adversarial attack detection
   - Model artifact integrity verification
   - Model access control and authorization
   - Model privacy and differential privacy enforcement

4. Application Security:
   - API security and rate limiting
   - Authentication and authorization controls
   - Secure software development lifecycle (SSDLC)
   - Runtime application self-protection (RASP)

Security Control Effectiveness Model:
Control_Effectiveness = P(Detection) × P(Response) × (1 - P(False_Positive))
Where:
- P(Detection): Probability of detecting genuine threats
- P(Response): Probability of successful automated response
- P(False_Positive): Probability of false positive alerts

Risk Reduction = Threat_Impact × Threat_Probability × Control_Effectiveness
```

**Kubernetes Security Automation:**
```
Pod Security Standards Enforcement:
apiVersion: v1
kind: Namespace
metadata:
  name: ml-production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

# Network Policies for ML Workloads
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-training-network-policy
  namespace: ml-training
spec:
  podSelector:
    matchLabels:
      app: ml-training
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ml-platform
    - podSelector:
        matchLabels:
          app: model-registry
    ports:
    - protocol: HTTP
      port: 8080
  egress:
  # Allow DNS resolution
  - to: []
    ports:
    - protocol: UDP
      port: 53
  # Allow access to model registry
  - to:
    - namespaceSelector:
        matchLabels:
          name: ml-platform
    - podSelector:
        matchLabels:
          app: model-registry
    ports:
    - protocol: TCP
      port: 8080
  # Allow access to data sources
  - to:
    - namespaceSelector:
        matchLabels:
          name: data-platform
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 443   # HTTPS

# Security Context Constraints
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: ml-restricted-scc
allowHostDirVolumePlugin: false
allowHostIPC: false
allowHostNetwork: false
allowHostPID: false
allowHostPorts: false
allowPrivilegedContainer: false
allowedCapabilities: []
defaultAddCapabilities: []
fsGroup:
  type: MustRunAs
  ranges:
  - min: 1000
    max: 65535
runAsUser:
  type: MustRunAsNonRoot
seLinuxContext:
  type: MustRunAs
supplementalGroups:
  type: MustRunAs
  ranges:
  - min: 1000
    max: 65535
volumes:
- configMap
- downwardAPI
- emptyDir
- persistentVolumeClaim
- projected
- secret

Container Security Scanning:
class ContainerSecurityScanner:
    def __init__(self, scanner_config):
        self.vulnerability_db = VulnerabilityDatabase()
        self.policy_engine = SecurityPolicyEngine(scanner_config)
        self.risk_calculator = RiskCalculator()
    
    def scan_image(self, image_reference):
        scan_result = ScanResult(image_reference)
        
        # Layer-by-layer vulnerability scanning
        image_layers = self.extract_image_layers(image_reference)
        for layer in image_layers:
            layer_vulnerabilities = self.scan_layer(layer)
            scan_result.add_layer_vulnerabilities(layer, layer_vulnerabilities)
        
        # Configuration scanning
        config_issues = self.scan_image_configuration(image_reference)
        scan_result.add_configuration_issues(config_issues)
        
        # Secret detection
        exposed_secrets = self.detect_exposed_secrets(image_layers)
        scan_result.add_exposed_secrets(exposed_secrets)
        
        # Policy evaluation
        policy_violations = self.policy_engine.evaluate(scan_result)
        scan_result.add_policy_violations(policy_violations)
        
        # Risk assessment
        risk_score = self.risk_calculator.calculate_risk(scan_result)
        scan_result.set_risk_score(risk_score)
        
        return scan_result
    
    def scan_layer(self, layer):
        vulnerabilities = []
        
        # Extract package information from layer
        packages = self.extract_packages(layer)
        
        for package in packages:
            # Query vulnerability database
            package_vulns = self.vulnerability_db.query_vulnerabilities(
                package.name, 
                package.version
            )
            
            for vuln in package_vulns:
                # Apply exploitability analysis
                exploitability = self.assess_exploitability(vuln, layer.runtime_context)
                
                vulnerabilities.append(LayerVulnerability(
                    package=package,
                    vulnerability=vuln,
                    exploitability=exploitability,
                    layer_id=layer.id
                ))
        
        return vulnerabilities
    
    def assess_exploitability(self, vulnerability, runtime_context):
        base_score = vulnerability.cvss_score
        
        # Context-specific adjustments
        if runtime_context.network_exposure:
            base_score *= 1.5
        
        if runtime_context.privileged_execution:
            base_score *= 1.3
        
        if runtime_context.sensitive_data_access:
            base_score *= 1.2
        
        # ML-specific considerations
        if vulnerability.affects_ml_libraries:
            base_score *= 1.4  # Model poisoning risk
        
        return min(base_score, 10.0)  # Cap at CVSS max

Admission Controller for Security:
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionWebhook
metadata:
  name: ml-security-validator
webhooks:
- name: security-policy.ml-platform.io
  clientConfig:
    service:
      name: ml-security-webhook
      namespace: ml-platform
      path: "/validate"
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: ["apps", "batch"]
    apiVersions: ["v1"]
    resources: ["deployments", "jobs", "statefulsets"]
  failurePolicy: Fail
  admissionReviewVersions: ["v1", "v1beta1"]

# Security Webhook Implementation
class MLSecurityAdmissionController:
    def __init__(self, security_policies):
        self.policies = security_policies
        self.image_scanner = ContainerSecurityScanner()
        self.risk_assessor = RiskAssessor()
    
    def validate_request(self, admission_request):
        resource = admission_request.object
        violations = []
        
        # Image security validation
        for container in self.extract_containers(resource):
            image_scan = self.image_scanner.scan_image(container.image)
            
            if image_scan.risk_score > self.policies.max_risk_score:
                violations.append(SecurityViolation(
                    type="high_risk_image",
                    message=f"Image {container.image} has risk score {image_scan.risk_score}",
                    severity="high"
                ))
            
            if image_scan.critical_vulnerabilities:
                violations.append(SecurityViolation(
                    type="critical_vulnerabilities",
                    message=f"Image contains {len(image_scan.critical_vulnerabilities)} critical vulnerabilities",
                    severity="critical"
                ))
        
        # Security context validation
        security_context_violations = self.validate_security_context(resource)
        violations.extend(security_context_violations)
        
        # Resource limits validation
        resource_limit_violations = self.validate_resource_limits(resource)
        violations.extend(resource_limit_violations)
        
        # Network policy validation
        network_violations = self.validate_network_configuration(resource)
        violations.extend(network_violations)
        
        return AdmissionResponse(
            allowed=len([v for v in violations if v.severity == "critical"]) == 0,
            violations=violations
        )
```

### **Threat Detection and Response**

**ML-Specific Threat Detection:**
```
AI/ML Threat Detection Framework:
class MLThreatDetector:
    def __init__(self):
        self.anomaly_detectors = {
            'model_performance': ModelPerformanceAnomalyDetector(),
            'data_drift': DataDriftDetector(),
            'access_patterns': AccessPatternAnalyzer(),
            'resource_usage': ResourceUsageAnalyzer()
        }
        self.threat_intelligence = ThreatIntelligenceEngine()
    
    def detect_threats(self, telemetry_data):
        threats = []
        
        # Model poisoning detection
        model_anomalies = self.detect_model_poisoning(telemetry_data)
        threats.extend(model_anomalies)
        
        # Data exfiltration detection
        exfiltration_indicators = self.detect_data_exfiltration(telemetry_data)
        threats.extend(exfiltration_indicators)
        
        # Adversarial attack detection
        adversarial_attacks = self.detect_adversarial_attacks(telemetry_data)
        threats.extend(adversarial_attacks)
        
        # Infrastructure compromise detection
        infrastructure_threats = self.detect_infrastructure_compromise(telemetry_data)
        threats.extend(infrastructure_threats)
        
        return threats
    
    def detect_model_poisoning(self, telemetry_data):
        threats = []
        
        # Sudden accuracy drops
        performance_data = telemetry_data['model_performance']
        accuracy_anomalies = self.anomaly_detectors['model_performance'].detect(
            performance_data['accuracy_timeseries']
        )
        
        for anomaly in accuracy_anomalies:
            if anomaly.severity > 0.8:  # High confidence threshold
                threats.append(ModelPoisoningThreat(
                    model_id=performance_data['model_id'],
                    anomaly_type='accuracy_drop',
                    confidence=anomaly.severity,
                    timestamp=anomaly.timestamp,
                    evidence=anomaly.evidence
                ))
        
        # Gradient pattern anomalies
        training_data = telemetry_data.get('training_metrics', {})
        if 'gradient_norms' in training_data:
            gradient_anomalies = self.detect_gradient_anomalies(
                training_data['gradient_norms']
            )
            
            for anomaly in gradient_anomalies:
                threats.append(ModelPoisoningThreat(
                    model_id=training_data['model_id'],
                    anomaly_type='gradient_anomaly',
                    confidence=anomaly.severity,
                    timestamp=anomaly.timestamp,
                    evidence=anomaly.evidence
                ))
        
        return threats
    
    def detect_data_exfiltration(self, telemetry_data):
        threats = []
        
        # Unusual data access patterns
        access_data = telemetry_data['data_access']
        access_anomalies = self.anomaly_detectors['access_patterns'].detect(
            access_data['access_logs']
        )
        
        for anomaly in access_anomalies:
            # Check against threat intelligence
            threat_indicators = self.threat_intelligence.check_indicators(
                ip_address=anomaly.source_ip,
                user_agent=anomaly.user_agent,
                access_pattern=anomaly.pattern
            )
            
            if threat_indicators.risk_score > 0.7:
                threats.append(DataExfiltrationThreat(
                    source_ip=anomaly.source_ip,
                    data_volume=anomaly.data_volume,
                    time_window=anomaly.time_window,
                    confidence=threat_indicators.risk_score,
                    indicators=threat_indicators.matched_indicators
                ))
        
        return threats

Automated Incident Response:
class AutomatedIncidentResponse:
    def __init__(self, response_policies):
        self.policies = response_policies
        self.isolation_engine = NetworkIsolationEngine()
        self.forensics_collector = ForensicsDataCollector()
        self.notification_service = NotificationService()
    
    def respond_to_threat(self, threat):
        response_plan = self.determine_response_plan(threat)
        
        # Execute immediate containment
        containment_actions = self.execute_containment(threat, response_plan)
        
        # Collect forensics data
        forensics_data = self.forensics_collector.collect(threat)
        
        # Execute eradication steps
        eradication_actions = self.execute_eradication(threat, response_plan)
        
        # Notify stakeholders
        self.notification_service.send_incident_notification(
            threat=threat,
            containment_actions=containment_actions,
            forensics_data=forensics_data
        )
        
        return IncidentResponse(
            threat_id=threat.id,
            response_plan=response_plan,
            containment_actions=containment_actions,
            eradication_actions=eradication_actions,
            forensics_data=forensics_data,
            status="contained"
        )
    
    def execute_containment(self, threat, response_plan):
        actions = []
        
        if threat.type == "model_poisoning":
            # Isolate affected model
            actions.append(self.isolate_model(threat.model_id))
            
            # Revert to previous known-good version
            actions.append(self.revert_model_version(threat.model_id))
            
            # Quarantine training data
            actions.append(self.quarantine_training_data(threat.data_sources))
        
        elif threat.type == "data_exfiltration":
            # Block suspicious IP addresses
            actions.append(self.isolation_engine.block_ip(threat.source_ip))
            
            # Revoke access tokens
            actions.append(self.revoke_access_tokens(threat.user_id))
            
            # Enable enhanced logging
            actions.append(self.enable_enhanced_logging(threat.affected_resources))
        
        elif threat.type == "adversarial_attack":
            # Enable adversarial detection filters
            actions.append(self.enable_adversarial_filters(threat.model_id))
            
            # Implement rate limiting
            actions.append(self.implement_rate_limiting(threat.source_ip))
        
        return actions

Security Orchestration Automation (SOAR):
apiVersion: v1
kind: ConfigMap
metadata:
  name: soar-playbooks
data:
  model-poisoning-playbook.yaml: |
    name: "Model Poisoning Response"
    trigger:
      threat_type: "model_poisoning"
      confidence_threshold: 0.8
    
    steps:
      - name: "immediate_containment"
        type: "parallel"
        actions:
          - isolate_model:
              model_id: "{{ threat.model_id }}"
              duration: "1h"
          - notify_security_team:
              severity: "critical"
              channels: ["slack", "pagerduty"]
          - preserve_evidence:
              artifacts: ["model_weights", "training_logs", "validation_metrics"]
      
      - name: "investigation"
        type: "sequential"
        actions:
          - collect_forensics:
              scope: ["training_environment", "data_sources", "model_registry"]
          - analyze_training_data:
              methods: ["statistical_analysis", "anomaly_detection"]
          - verify_model_integrity:
              baseline_model: "{{ threat.baseline_model_id }}"
      
      - name: "eradication"
        type: "conditional"
        condition: "investigation.confirmed_poisoning"
        actions:
          - retrain_model:
              clean_dataset: true
              validation_required: true
          - update_security_controls:
              controls: ["data_validation", "model_verification"]
      
      - name: "recovery"
        type: "sequential"
        actions:
          - deploy_clean_model:
              validation_required: true
              gradual_rollout: true
          - restore_normal_operations:
              monitoring_enhanced: true
  
  data-exfiltration-playbook.yaml: |
    name: "Data Exfiltration Response"
    trigger:
      threat_type: "data_exfiltration"
      confidence_threshold: 0.7
    
    steps:
      - name: "immediate_containment"
        type: "parallel"
        actions:
          - block_source_ip:
              ip_address: "{{ threat.source_ip }}"
              duration: "24h"
          - revoke_access_tokens:
              user_id: "{{ threat.user_id }}"
              scope: "all"
          - enable_enhanced_logging:
              resources: "{{ threat.affected_resources }}"
              duration: "7d"
      
      - name: "investigation"
        type: "sequential"
        actions:
          - analyze_access_logs:
              time_window: "7d"
              focus: "data_access_patterns"
          - check_data_integrity:
              datasets: "{{ threat.accessed_datasets }}"
          - assess_data_sensitivity:
              classification_required: true
      
      - name: "notification"
        type: "conditional"
        condition: "investigation.sensitive_data_accessed"
        actions:
          - notify_legal_team:
              urgency: "high"
          - prepare_breach_notification:
              regulatory_requirements: true
```

---

## 🔐 Certificate Lifecycle Management

### **Automated PKI Infrastructure**

**Certificate Authority Automation:**
```
PKI Architecture Design:
Root CA (Offline) → Intermediate CA (Online) → End-Entity Certificates

Certificate Hierarchy:
- Root Certificate Authority (10-year validity)
  ├── ML Platform Intermediate CA (5-year validity)
  │   ├── Service Certificates (1-year validity)
  │   ├── Client Certificates (90-day validity)
  │   └── Code Signing Certificates (2-year validity)
  └── Data Platform Intermediate CA (5-year validity)
      ├── Database TLS Certificates (1-year validity)
      └── API Gateway Certificates (6-month validity)

Certificate Lifecycle States:
Requested → Validated → Issued → Active → Renewal_Warning → Expired → Revoked

Certificate Automation Engine:
class CertificateAutomationEngine:
    def __init__(self, ca_config, cert_manager_client):
        self.ca_config = ca_config
        self.cert_manager = cert_manager_client
        self.renewal_scheduler = RenewalScheduler()
        self.validation_engine = CertificateValidationEngine()
    
    def request_certificate(self, cert_request):
        # Validate certificate request
        validation_result = self.validation_engine.validate_request(cert_request)
        if not validation_result.valid:
            raise CertificateValidationError(validation_result.errors)
        
        # Generate certificate signing request (CSR)
        csr = self.generate_csr(cert_request)
        
        # Submit to certificate authority
        cert_response = self.submit_to_ca(csr, cert_request)
        
        # Store certificate and private key
        self.store_certificate(cert_response, cert_request)
        
        # Schedule renewal
        self.renewal_scheduler.schedule_renewal(
            cert_response.certificate,
            renewal_time=cert_response.expiry_date - timedelta(days=30)
        )
        
        return cert_response
    
    def automated_renewal(self, certificate):
        try:
            # Check if renewal is needed
            if not self.needs_renewal(certificate):
                return RenewalResult(status="not_needed", certificate=certificate)
            
            # Create renewal request
            renewal_request = self.create_renewal_request(certificate)
            
            # Request new certificate
            new_certificate = self.request_certificate(renewal_request)
            
            # Update applications using the certificate
            self.update_certificate_references(certificate, new_certificate)
            
            # Revoke old certificate after grace period
            self.schedule_revocation(certificate, grace_period=timedelta(hours=24))
            
            return RenewalResult(
                status="success", 
                old_certificate=certificate,
                new_certificate=new_certificate
            )
            
        except Exception as e:
            # Alert administrators on renewal failure
            self.alert_renewal_failure(certificate, e)
            return RenewalResult(status="failed", error=str(e))
    
    def validate_certificate_chain(self, certificate_chain):
        """Validate certificate chain integrity and trust"""
        validation_results = []
        
        for i, cert in enumerate(certificate_chain):
            # Basic certificate validation
            basic_validation = self.validate_basic_certificate(cert)
            validation_results.append(basic_validation)
            
            # Chain validation
            if i > 0:  # Not root certificate
                parent_cert = certificate_chain[i-1]
                chain_validation = self.validate_certificate_signature(cert, parent_cert)
                validation_results.append(chain_validation)
            
            # Revocation checking
            revocation_status = self.check_revocation_status(cert)
            validation_results.append(revocation_status)
        
        return CertificateChainValidation(
            valid=all(result.valid for result in validation_results),
            validation_results=validation_results
        )

Cert-Manager Integration:
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: ml-platform-ca-issuer
spec:
  ca:
    secretName: ml-platform-ca-key-pair

---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: ml-training-service-tls
  namespace: ml-training
spec:
  secretName: ml-training-service-tls
  duration: 8760h  # 1 year
  renewBefore: 720h  # 30 days
  isCA: false
  privateKey:
    algorithm: RSA
    encoding: PKCS1
    size: 2048
  usages:
    - server auth
    - client auth
  dnsNames:
    - ml-training-service.ml-training.svc.cluster.local
    - ml-training-service
  ipAddresses:
    - 10.96.0.100
  issuerRef:
    name: ml-platform-ca-issuer
    kind: ClusterIssuer
    group: cert-manager.io

---
apiVersion: cert-manager.io/v1alpha1
kind: VenafiIssuer
metadata:
  name: venafi-tpp-issuer
spec:
  tpp:
    url: https://tpp.company.com/vedsdk
    credentialsRef:
      name: venafi-secret
    zone: "ML Platform\\Certificates"

Certificate Monitoring and Alerting:
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: certificate-monitoring
spec:
  groups:
  - name: certificate.rules
    rules:
    - alert: CertificateExpiringSoon
      expr: (certmanager_certificate_expiration_timestamp_seconds - time()) / 86400 < 30
      for: 1h
      labels:
        severity: warning
      annotations:
        summary: "Certificate {{ $labels.name }} expires in less than 30 days"
        description: "Certificate {{ $labels.name }} in namespace {{ $labels.namespace }} expires in {{ $value }} days"
    
    - alert: CertificateExpired
      expr: certmanager_certificate_expiration_timestamp_seconds < time()
      for: 0m
      labels:
        severity: critical
      annotations:
        summary: "Certificate {{ $labels.name }} has expired"
        description: "Certificate {{ $labels.name }} in namespace {{ $labels.namespace }} has expired"
    
    - alert: CertificateRenewalFailed
      expr: increase(certmanager_certificate_renewal_failure_total[1h]) > 0
      for: 0m  
      labels:
        severity: critical
      annotations:
        summary: "Certificate renewal failed for {{ $labels.name }}"
        description: "Certificate {{ $labels.name }} in namespace {{ $labels.namespace }} failed to renew"

External Certificate Integration:
class ExternalCertificateManager:
    def __init__(self, external_ca_config):
        self.ca_clients = {
            'venafi': VenafiClient(external_ca_config['venafi']),
            'digicert': DigiCertClient(external_ca_config['digicert']),
            'lets_encrypt': LetsEncryptClient(external_ca_config['lets_encrypt'])
        }
    
    def request_external_certificate(self, cert_request):
        # Determine appropriate CA based on requirements
        ca_name = self.select_certificate_authority(cert_request)
        ca_client = self.ca_clients[ca_name]
        
        # Validate domain ownership
        if cert_request.domain_validation_required:
            validation_result = self.validate_domain_ownership(
                cert_request.domains, 
                ca_client
            )
            if not validation_result.valid:
                raise DomainValidationError(validation_result.errors)
        
        # Submit certificate request
        cert_response = ca_client.request_certificate(cert_request)
        
        # Create Kubernetes secret
        self.create_certificate_secret(cert_response, cert_request.namespace)
        
        # Register with monitoring
        self.register_certificate_monitoring(cert_response)
        
        return cert_response
    
    def select_certificate_authority(self, cert_request):
        """Select CA based on certificate requirements"""
        if cert_request.extended_validation:
            return 'digicert'  # For EV certificates
        elif cert_request.wildcard_certificate:
            return 'venafi'    # For wildcard certificates
        elif cert_request.public_facing:
            return 'lets_encrypt'  # For public-facing services
        else:
            return 'venafi'    # Default for internal services
```

### **Service Mesh Security Integration**

**Istio Security Configuration:**
```
Zero-Trust Service Mesh Security:
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: ml-platform-mtls
  namespace: ml-platform
spec:
  mtls:
    mode: STRICT

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: ml-training-authz
  namespace: ml-training
spec:
  selector:
    matchLabels:
      app: ml-training-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/ml-platform/sa/ml-orchestrator"]
    - source:
        principals: ["cluster.local/ns/ml-monitoring/sa/prometheus"]
  - to:
    - operation:
        methods: ["POST"]
        paths: ["/api/v1/train"]
  - when:
    - condition:
        key: source.ip
        values: ["10.0.0.0/8"]  # Internal network only

Certificate Management for Service Mesh:
apiVersion: v1
kind: Secret
metadata:
  name: cacerts
  namespace: istio-system
type: Opaque
data:
  root-cert.pem: {{ .Values.certs.rootCert | b64enc }}
  cert-chain.pem: {{ .Values.certs.certChain | b64enc }}

---
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: ml-platform-istio
spec:
  meshConfig:
    defaultConfig:
      proxyStatsMatcher:
        inclusionRegexps:
        - ".*circuit_breakers.*"
        - ".*upstream_rq_retry.*"
        - ".*_cx_.*"
      gatewayTopology:
        numTrustedProxies: 2
  values:
    pilot:
      env:
        EXTERNAL_ISTIOD: false
        PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION: true
    global:
      meshID: ml-platform-mesh
      network: ml-platform-network
      meshConfig:
        certificates:
        - secretName: dns.ml-platform-cert
          dnsNames:
          - "*.ml-platform.company.com"
        - secretName: spiffe.ml-platform-cert
          dnsNames:
          - "spiffe://ml-platform.company.com"

Workload Identity and Certificate Binding:
apiVersion: security.istio.io/v1beta1
kind: WorkloadEntry
metadata:
  name: ml-external-service
  namespace: ml-platform
spec:
  serviceAccount: ml-external-sa
  address: 10.0.1.100
  ports:
    https: 8443
  labels:
    app: ml-external-service
    version: v1

---
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: ml-external-service
  namespace: ml-platform
spec:
  hosts:
  - ml-external-service.company.com
  ports:
  - number: 443
    name: https
    protocol: HTTPS
  location: MESH_EXTERNAL
  resolution: DNS
  workloadSelector:
    labels:
      app: ml-external-service

Certificate Rotation in Service Mesh:
class ServiceMeshCertificateManager:
    def __init__(self, istio_client, cert_manager):
        self.istio = istio_client
        self.cert_manager = cert_manager
        self.rotation_scheduler = CertificateRotationScheduler()
    
    def rotate_root_certificate(self, new_root_cert):
        """Perform zero-downtime root certificate rotation"""
        
        # Phase 1: Deploy new root cert alongside old one
        self.deploy_dual_root_certificates(new_root_cert)
        
        # Phase 2: Update intermediate certificates
        intermediate_certs = self.get_intermediate_certificates()
        for cert in intermediate_certs:
            new_intermediate = self.issue_new_intermediate(cert, new_root_cert)
            self.deploy_dual_intermediate_certificates(cert, new_intermediate)
        
        # Phase 3: Gradual workload certificate updates
        workloads = self.get_mesh_workloads()
        for workload in workloads:
            self.schedule_workload_cert_rotation(workload, new_root_cert)
        
        # Phase 4: Remove old certificates after grace period
        self.schedule_old_cert_cleanup(grace_period=timedelta(days=7))
        
        return CertificateRotationResult(
            phase="completed",
            new_root_cert=new_root_cert,
            affected_workloads=len(workloads)
        )
    
    def validate_mesh_security(self):
        """Validate service mesh security configuration"""
        validation_results = []
        
        # Check mTLS enforcement
        mtls_policies = self.istio.get_peer_authentication_policies()
        for policy in mtls_policies:
            mtls_validation = self.validate_mtls_policy(policy)
            validation_results.append(mtls_validation)
        
        # Check authorization policies
        authz_policies = self.istio.get_authorization_policies()
        for policy in authz_policies:
            authz_validation = self.validate_authorization_policy(policy)
            validation_results.append(authz_validation)
        
        # Check certificate validity
        cert_validation = self.validate_mesh_certificates()
        validation_results.append(cert_validation)
        
        return ServiceMeshSecurityValidation(
            overall_status="secure" if all(r.valid for r in validation_results) else "issues_found",
            validation_results=validation_results
        )

Security Policy as Code:
# policies/security/ml-platform-security-policies.rego
package mlplatform.security

# Require mTLS for all ML workloads
deny[msg] {
    input.kind == "Deployment"
    input.metadata.labels["ml-workload"] == "true"
    not has_mtls_annotation
    msg := "ML workloads must have mTLS enabled"
}

has_mtls_annotation {
    input.metadata.annotations["security.istio.io/tlsMode"] == "istio"
}

# Require specific service account for ML training
deny[msg] {
    input.kind == "Deployment"
    input.metadata.labels["workload-type"] == "training"
    not input.spec.template.spec.serviceAccountName == "ml-training-sa"
    msg := "Training workloads must use ml-training-sa service account"
}

# Require resource limits for GPU workloads
deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    container.resources.requests["nvidia.com/gpu"]
    not container.resources.limits["nvidia.com/gpu"]
    msg := "GPU workloads must specify GPU limits"
}

# Require network policies for sensitive namespaces
deny[msg] {
    input.kind == "Namespace"
    input.metadata.labels["data-classification"] == "sensitive"
    not has_network_policy
    msg := "Sensitive namespaces must have network policies"
}

has_network_policy {
    # This would be checked against existing network policies
    # Implementation depends on policy engine capabilities
}
```

This comprehensive framework for security automation and certificate management provides the theoretical foundations and practical strategies for building secure, compliant, and resilient ML infrastructure. The key insight is that effective security automation requires layered defense strategies, automated threat detection and response, and comprehensive certificate lifecycle management to maintain trust and security across complex ML systems.