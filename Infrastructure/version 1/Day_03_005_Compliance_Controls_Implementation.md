# Day 3.5: Compliance Controls Implementation

## ⚖️ Data Governance, Metadata & Cataloging - Part 5

**Focus**: GDPR/CCPA Compliance Automation, Data Classification, Privacy-Preserving Transformations  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master GDPR/CCPA compliance automation frameworks and implementation strategies
- Understand data classification algorithms and sensitivity labeling systems
- Learn privacy-preserving data transformation techniques and their mathematical foundations
- Implement automated audit trail generation and retention policies

---

## 📜 Privacy Regulation Compliance Theory

### **GDPR/CCPA Mathematical Framework**

#### **Privacy Rights Formalization**
```
Personal Data Universe: P = {p₁, p₂, ..., pₙ}
Processing Operations: O = {collect, store, process, transfer, delete}
Legal Basis: L = {consent, contract, legal_obligation, vital_interest, public_task, legitimate_interest}

Privacy Compliance Function:
C(p, o, l) = {
    COMPLIANT if ∀p ∈ P: has_legal_basis(p, o, l) ∧ purpose_limitation(p, o) ∧ data_minimization(p)
    NON_COMPLIANT otherwise
}
```

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import json
import hashlib
import re
from datetime import datetime, timedelta
import uuid

class PrivacyRegulation(Enum):
    """Supported privacy regulations"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"

class DataSubjectRight(Enum):
    """Data subject rights under privacy regulations"""
    ACCESS = "right_to_access"
    RECTIFICATION = "right_to_rectification"
    ERASURE = "right_to_erasure"
    RESTRICT_PROCESSING = "right_to_restrict_processing"
    DATA_PORTABILITY = "right_to_data_portability"
    OBJECT = "right_to_object"
    AUTOMATED_DECISION_MAKING = "right_regarding_automated_decision_making"

class LegalBasis(Enum):
    """Legal basis for data processing under GDPR"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataSensitivityLevel(Enum):
    """Data sensitivity classification levels"""
    PUBLIC = 1
    INTERNAL = 2
    CONFIDENTIAL = 3
    RESTRICTED = 4
    TOP_SECRET = 5

@dataclass
class PersonalDataElement:
    """Represents a personal data element"""
    element_id: str
    field_name: str
    data_type: str
    sensitivity_level: DataSensitivityLevel
    is_personal_data: bool
    is_sensitive_personal_data: bool
    legal_basis: Set[LegalBasis]
    processing_purposes: Set[str]
    retention_period_days: Optional[int] = None
    source_system: Optional[str] = None
    
    def __hash__(self):
        return hash(self.element_id)

@dataclass
class DataProcessingActivity:
    """Represents a data processing activity"""
    activity_id: str
    activity_name: str
    controller: str
    processor: Optional[str] = None
    data_elements: Set[PersonalDataElement] = field(default_factory=set)
    processing_purposes: Set[str] = field(default_factory=set)
    legal_basis: Set[LegalBasis] = field(default_factory=set)
    data_subjects: Set[str] = field(default_factory=set)
    third_country_transfers: bool = False
    retention_schedule: Dict[str, int] = field(default_factory=dict)
    security_measures: Set[str] = field(default_factory=set)

class PrivacyComplianceEngine:
    """Advanced privacy compliance automation engine"""
    
    def __init__(self, regulation: PrivacyRegulation):
        self.regulation = regulation
        self.compliance_rules = self._load_compliance_rules()
        self.data_classification_engine = DataClassificationEngine()
        self.audit_logger = ComplianceAuditLogger()
        self.policy_engine = PrivacyPolicyEngine()
        
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load regulation-specific compliance rules"""
        
        if self.regulation == PrivacyRegulation.GDPR:
            return {
                'data_subject_rights': list(DataSubjectRight),
                'legal_basis_required': True,
                'consent_requirements': {
                    'explicit_consent_required_for_sensitive': True,
                    'consent_withdrawal_mechanism': True,
                    'clear_and_plain_language': True
                },
                'retention_limits': {
                    'default_max_years': 7,
                    'sensitive_data_max_years': 3
                },
                'breach_notification_hours': 72,
                'privacy_by_design_required': True,
                'dpia_threshold_criteria': [
                    'systematic_monitoring',
                    'sensitive_data_large_scale',
                    'vulnerable_data_subjects',
                    'automated_decision_making'
                ]
            }
        elif self.regulation == PrivacyRegulation.CCPA:
            return {
                'data_subject_rights': [
                    DataSubjectRight.ACCESS,
                    DataSubjectRight.ERASURE,
                    DataSubjectRight.DATA_PORTABILITY,
                    DataSubjectRight.OBJECT
                ],
                'sale_opt_out_required': True,
                'notice_at_collection_required': True,
                'retention_limits': {
                    'default_max_years': 5
                },
                'breach_notification_days': 30
            }
    
    def assess_compliance(self, processing_activity: DataProcessingActivity) -> Dict[str, Any]:
        """Assess compliance of a data processing activity"""
        
        compliance_assessment = {
            'activity_id': processing_activity.activity_id,
            'regulation': self.regulation.value,
            'assessment_timestamp': datetime.utcnow().isoformat(),
            'overall_compliance_score': 0.0,
            'compliance_issues': [],
            'recommendations': [],
            'risk_level': 'low',
            'required_actions': []
        }
        
        # Check legal basis
        legal_basis_check = self._check_legal_basis(processing_activity)
        compliance_assessment['compliance_issues'].extend(legal_basis_check['issues'])
        
        # Check data minimization
        minimization_check = self._check_data_minimization(processing_activity)
        compliance_assessment['compliance_issues'].extend(minimization_check['issues'])
        
        # Check purpose limitation
        purpose_check = self._check_purpose_limitation(processing_activity)
        compliance_assessment['compliance_issues'].extend(purpose_check['issues'])
        
        # Check retention compliance
        retention_check = self._check_retention_compliance(processing_activity)
        compliance_assessment['compliance_issues'].extend(retention_check['issues'])
        
        # Check security measures
        security_check = self._check_security_measures(processing_activity)
        compliance_assessment['compliance_issues'].extend(security_check['issues'])
        
        # Calculate overall compliance score
        total_checks = 5
        failed_checks = len([check for check in [
            legal_basis_check, minimization_check, purpose_check, 
            retention_check, security_check
        ] if check['issues']])
        
        compliance_assessment['overall_compliance_score'] = (total_checks - failed_checks) / total_checks
        
        # Determine risk level
        if compliance_assessment['overall_compliance_score'] < 0.6:
            compliance_assessment['risk_level'] = 'high'
        elif compliance_assessment['overall_compliance_score'] < 0.8:
            compliance_assessment['risk_level'] = 'medium'
        else:
            compliance_assessment['risk_level'] = 'low'
        
        # Generate recommendations
        compliance_assessment['recommendations'] = self._generate_compliance_recommendations(
            compliance_assessment['compliance_issues']
        )
        
        return compliance_assessment
    
    def _check_legal_basis(self, activity: DataProcessingActivity) -> Dict[str, Any]:
        """Check if processing has valid legal basis"""
        issues = []
        
        if not activity.legal_basis:
            issues.append({
                'type': 'missing_legal_basis',
                'severity': 'high',
                'message': 'No legal basis specified for data processing'
            })
        
        # Check sensitive data requires explicit consent or other strong legal basis
        sensitive_data_elements = [
            elem for elem in activity.data_elements 
            if elem.is_sensitive_personal_data
        ]
        
        if sensitive_data_elements:
            strong_legal_bases = {
                LegalBasis.CONSENT, 
                LegalBasis.LEGAL_OBLIGATION, 
                LegalBasis.VITAL_INTERESTS
            }
            
            if not activity.legal_basis.intersection(strong_legal_bases):
                issues.append({
                    'type': 'insufficient_legal_basis_for_sensitive_data',
                    'severity': 'high',
                    'message': 'Sensitive personal data requires explicit consent or strong legal basis'
                })
        
        return {'issues': issues}
    
    def _check_data_minimization(self, activity: DataProcessingActivity) -> Dict[str, Any]:
        """Check data minimization principle compliance"""
        issues = []
        
        # Check if all data elements have associated processing purposes
        elements_without_purpose = []
        
        for element in activity.data_elements:
            if not element.processing_purposes.intersection(activity.processing_purposes):
                elements_without_purpose.append(element.field_name)
        
        if elements_without_purpose:
            issues.append({
                'type': 'data_minimization_violation',
                'severity': 'medium',
                'message': f'Data elements without clear purpose: {elements_without_purpose}'
            })
        
        # Check for excessive data collection patterns
        total_elements = len(activity.data_elements)
        sensitive_elements = len([e for e in activity.data_elements if e.is_sensitive_personal_data])
        
        if sensitive_elements / total_elements > 0.3:  # More than 30% sensitive data
            issues.append({
                'type': 'excessive_sensitive_data_collection',
                'severity': 'medium',
                'message': 'High proportion of sensitive personal data may violate minimization principle'
            })
        
        return {'issues': issues}
    
    def _check_purpose_limitation(self, activity: DataProcessingActivity) -> Dict[str, Any]:
        """Check purpose limitation principle compliance"""
        issues = []
        
        if not activity.processing_purposes:
            issues.append({
                'type': 'missing_processing_purposes',
                'severity': 'high',
                'message': 'No processing purposes specified'
            })
        
        # Check for overly broad purposes
        broad_purposes = [
            'business operations', 'analytics', 'marketing', 'other'
        ]
        
        overly_broad = activity.processing_purposes.intersection(broad_purposes)
        if overly_broad:
            issues.append({
                'type': 'overly_broad_purposes',
                'severity': 'medium',
                'message': f'Purposes may be too broad: {overly_broad}'
            })
        
        return {'issues': issues}
    
    def _check_retention_compliance(self, activity: DataProcessingActivity) -> Dict[str, Any]:
        """Check data retention compliance"""
        issues = []
        
        rules = self.compliance_rules.get('retention_limits', {})
        default_max_days = rules.get('default_max_years', 7) * 365
        sensitive_max_days = rules.get('sensitive_data_max_years', 3) * 365
        
        for element in activity.data_elements:
            retention_period = element.retention_period_days
            
            if retention_period is None:
                issues.append({
                    'type': 'missing_retention_period',
                    'severity': 'medium',
                    'message': f'No retention period specified for {element.field_name}'
                })
                continue
            
            max_allowed = sensitive_max_days if element.is_sensitive_personal_data else default_max_days
            
            if retention_period > max_allowed:
                issues.append({
                    'type': 'excessive_retention_period',
                    'severity': 'high',
                    'message': f'Retention period for {element.field_name} exceeds regulatory limits'
                })
        
        return {'issues': issues}
    
    def _check_security_measures(self, activity: DataProcessingActivity) -> Dict[str, Any]:
        """Check security measures compliance"""
        issues = []
        
        required_measures = set()
        
        # Determine required security measures based on data sensitivity
        has_sensitive_data = any(e.is_sensitive_personal_data for e in activity.data_elements)
        
        if has_sensitive_data:
            required_measures.update([
                'encryption_at_rest',
                'encryption_in_transit',
                'access_controls',
                'audit_logging',
                'regular_security_assessments'
            ])
        else:
            required_measures.update([
                'access_controls',
                'audit_logging'
            ])
        
        missing_measures = required_measures - activity.security_measures
        
        if missing_measures:
            issues.append({
                'type': 'insufficient_security_measures',
                'severity': 'high',
                'message': f'Missing required security measures: {missing_measures}'
            })
        
        return {'issues': issues}
    
    def _generate_compliance_recommendations(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable compliance recommendations"""
        recommendations = []
        
        issue_types = [issue['type'] for issue in issues]
        
        if 'missing_legal_basis' in issue_types:
            recommendations.append({
                'priority': 'high',
                'action': 'Establish legal basis for data processing',
                'description': 'Review GDPR Article 6 legal bases and document appropriate basis for each processing activity'
            })
        
        if 'data_minimization_violation' in issue_types:
            recommendations.append({
                'priority': 'medium',
                'action': 'Implement data minimization controls',
                'description': 'Review data collection practices and eliminate unnecessary data elements'
            })
        
        if 'excessive_retention_period' in issue_types:
            recommendations.append({
                'priority': 'high',
                'action': 'Implement automated data retention policies',
                'description': 'Set up automated deletion of data beyond regulatory retention limits'
            })
        
        if 'insufficient_security_measures' in issue_types:
            recommendations.append({
                'priority': 'high',
                'action': 'Enhance data security controls',
                'description': 'Implement missing security measures including encryption and access controls'
            })
        
        return recommendations

class DataClassificationEngine:
    """Automated data classification and sensitivity labeling"""
    
    def __init__(self):
        self.classification_patterns = self._load_classification_patterns()
        self.ml_classifier = None  # Would be trained ML model in production
        
    def _load_classification_patterns(self) -> Dict[str, Any]:
        """Load pattern-based classification rules"""
        return {
            'personal_identifiers': {
                'patterns': [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
                    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'  # Phone number
                ],
                'field_names': [
                    'email', 'ssn', 'social_security', 'credit_card', 'phone', 'mobile'
                ],
                'sensitivity_level': DataSensitivityLevel.RESTRICTED
            },
            'sensitive_personal_data': {
                'field_names': [
                    'race', 'ethnicity', 'religion', 'political_opinion', 'health',
                    'medical', 'biometric', 'genetic', 'sexual_orientation'
                ],
                'sensitivity_level': DataSensitivityLevel.TOP_SECRET
            },
            'financial_data': {
                'field_names': [
                    'salary', 'income', 'account_number', 'routing_number',
                    'balance', 'transaction', 'payment'
                ],
                'patterns': [
                    r'\$[\d,]+\.?\d*',  # Currency amounts
                    r'\b\d{9,18}\b'     # Account numbers
                ],
                'sensitivity_level': DataSensitivityLevel.CONFIDENTIAL
            }
        }
    
    def classify_data_element(self, field_name: str, 
                            sample_values: List[str] = None) -> Dict[str, Any]:
        """Classify a data element for sensitivity and personal data status"""
        
        classification_result = {
            'field_name': field_name,
            'is_personal_data': False,
            'is_sensitive_personal_data': False,
            'sensitivity_level': DataSensitivityLevel.PUBLIC,
            'classification_confidence': 0.0,
            'detected_patterns': [],
            'recommended_protections': []
        }
        
        field_name_lower = field_name.lower()
        
        # Pattern-based classification
        for category, config in self.classification_patterns.items():
            confidence_score = 0.0
            
            # Check field name patterns
            field_names = config.get('field_names', [])
            for pattern_name in field_names:
                if pattern_name in field_name_lower:
                    confidence_score += 0.5
                    classification_result['detected_patterns'].append(f'field_name:{pattern_name}')
            
            # Check value patterns if sample values provided
            if sample_values:
                patterns = config.get('patterns', [])
                for pattern in patterns:
                    matches = sum(1 for value in sample_values[:100] if re.search(pattern, str(value)))
                    if matches > 0:
                        pattern_confidence = matches / len(sample_values[:100])
                        confidence_score += pattern_confidence
                        classification_result['detected_patterns'].append(f'value_pattern:{pattern}')
            
            # Update classification if confidence is high enough
            if confidence_score > 0.3:  # 30% confidence threshold
                classification_result['sensitivity_level'] = max(
                    classification_result['sensitivity_level'],
                    config['sensitivity_level'],
                    key=lambda x: x.value
                )
                classification_result['classification_confidence'] = max(
                    classification_result['classification_confidence'],
                    confidence_score
                )
                
                if category in ['personal_identifiers', 'financial_data']:
                    classification_result['is_personal_data'] = True
                
                if category == 'sensitive_personal_data':
                    classification_result['is_sensitive_personal_data'] = True
        
        # Generate protection recommendations
        classification_result['recommended_protections'] = self._recommend_protections(
            classification_result
        )
        
        return classification_result
    
    def _recommend_protections(self, classification: Dict[str, Any]) -> List[str]:
        """Recommend data protection measures based on classification"""
        protections = []
        
        if classification['is_personal_data']:
            protections.extend([
                'access_logging',
                'purpose_limitation',
                'retention_policy'
            ])
        
        if classification['is_sensitive_personal_data']:
            protections.extend([
                'encryption_at_rest',
                'encryption_in_transit',
                'explicit_consent_required',
                'enhanced_access_controls'
            ])
        
        sensitivity_level = classification['sensitivity_level']
        
        if sensitivity_level.value >= DataSensitivityLevel.CONFIDENTIAL.value:
            protections.extend([
                'data_masking',
                'role_based_access',
                'audit_trail'
            ])
        
        if sensitivity_level.value >= DataSensitivityLevel.RESTRICTED.value:
            protections.extend([
                'tokenization',
                'data_loss_prevention',
                'regular_access_review'
            ])
        
        if sensitivity_level.value == DataSensitivityLevel.TOP_SECRET.value:
            protections.extend([
                'multi_factor_authentication',
                'data_minimization_enforcement',
                'specialized_handling_procedures'
            ])
        
        return list(set(protections))  # Remove duplicates

class PrivacyPreservingTransformations:
    """Privacy-preserving data transformation techniques"""
    
    def __init__(self):
        self.transformation_methods = {
            'anonymization': self.anonymize_data,
            'pseudonymization': self.pseudonymize_data,
            'k_anonymity': self.apply_k_anonymity,
            'differential_privacy': self.apply_differential_privacy,
            'data_masking': self.mask_data
        }
    
    def anonymize_data(self, data: List[Dict[str, Any]], 
                      identifier_fields: List[str]) -> List[Dict[str, Any]]:
        """Apply anonymization by removing direct identifiers"""
        anonymized_data = []
        
        for record in data:
            anonymized_record = {
                k: v for k, v in record.items() 
                if k not in identifier_fields
            }
            anonymized_data.append(anonymized_record)
        
        return anonymized_data
    
    def pseudonymize_data(self, data: List[Dict[str, Any]], 
                         identifier_fields: List[str],
                         salt: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Apply pseudonymization with reversible mapping"""
        if salt is None:
            salt = str(uuid.uuid4())
        
        pseudonym_mapping = {}
        pseudonymized_data = []
        
        for record in data:
            pseudonymized_record = record.copy()
            
            for field in identifier_fields:
                if field in record:
                    original_value = str(record[field])
                    pseudonym = hashlib.sha256(
                        (original_value + salt).encode()
                    ).hexdigest()[:16]  # Use first 16 characters
                    
                    pseudonymized_record[field] = pseudonym
                    pseudonym_mapping[pseudonym] = original_value
            
            pseudonymized_data.append(pseudonymized_record)
        
        return pseudonymized_data, pseudonym_mapping
    
    def apply_k_anonymity(self, data: List[Dict[str, Any]], 
                         quasi_identifiers: List[str],
                         k: int = 3) -> List[Dict[str, Any]]:
        """Apply k-anonymity by generalizing quasi-identifiers"""
        
        # Group records by quasi-identifier combinations
        groups = {}
        
        for record in data:
            quasi_id_tuple = tuple(
                record.get(field, '') for field in quasi_identifiers
            )
            
            if quasi_id_tuple not in groups:
                groups[quasi_id_tuple] = []
            groups[quasi_id_tuple].append(record)
        
        k_anonymous_data = []
        
        for quasi_id_combo, group_records in groups.items():
            if len(group_records) < k:
                # Need to generalize or suppress
                generalized_records = self._generalize_group(
                    group_records, quasi_identifiers, k
                )
                k_anonymous_data.extend(generalized_records)
            else:
                # Group already satisfies k-anonymity
                k_anonymous_data.extend(group_records)
        
        return k_anonymous_data
    
    def apply_differential_privacy(self, data: List[Dict[str, Any]], 
                                 numeric_fields: List[str],
                                 epsilon: float = 1.0) -> List[Dict[str, Any]]:
        """Apply differential privacy by adding calibrated noise"""
        import numpy as np
        
        noisy_data = []
        
        for record in data:
            noisy_record = record.copy()
            
            for field in numeric_fields:
                if field in record and isinstance(record[field], (int, float)):
                    # Add Laplace noise for differential privacy
                    sensitivity = 1.0  # Assume sensitivity of 1 for simplicity
                    noise_scale = sensitivity / epsilon
                    noise = np.random.laplace(0, noise_scale)
                    
                    noisy_record[field] = record[field] + noise
            
            noisy_data.append(noisy_record)
        
        return noisy_data
    
    def mask_data(self, data: List[Dict[str, Any]], 
                 fields_to_mask: List[str],
                 masking_pattern: str = "***") -> List[Dict[str, Any]]:
        """Apply data masking to specified fields"""
        masked_data = []
        
        for record in data:
            masked_record = record.copy()
            
            for field in fields_to_mask:
                if field in record:
                    original_value = str(record[field])
                    
                    if len(original_value) <= 4:
                        masked_record[field] = masking_pattern
                    else:
                        # Keep first and last characters, mask middle
                        masked_record[field] = (
                            original_value[0] + 
                            masking_pattern + 
                            original_value[-1]
                        )
            
            masked_data.append(masked_record)
        
        return masked_data
    
    def _generalize_group(self, records: List[Dict[str, Any]], 
                         quasi_identifiers: List[str],
                         k: int) -> List[Dict[str, Any]]:
        """Generalize a group of records to achieve k-anonymity"""
        
        generalized_records = []
        
        for record in records:
            generalized_record = record.copy()
            
            # Apply generalization rules (simplified)
            for field in quasi_identifiers:
                if field in record:
                    value = record[field]
                    
                    # Age generalization example
                    if field.lower() == 'age' and isinstance(value, int):
                        age_range = f"{(value // 10) * 10}-{(value // 10) * 10 + 9}"
                        generalized_record[field] = age_range
                    
                    # ZIP code generalization example
                    elif field.lower() in ['zip', 'zipcode', 'postal_code']:
                        generalized_record[field] = str(value)[:3] + "**"
            
            generalized_records.append(generalized_record)
        
        return generalized_records

class ComplianceAuditLogger:
    """Automated compliance audit trail generation"""
    
    def __init__(self):
        self.audit_events = []
        self.retention_policy_days = 2555  # 7 years in days
        
    def log_data_access(self, user_id: str, data_elements: List[str], 
                       purpose: str, legal_basis: str) -> str:
        """Log data access event for compliance audit"""
        
        audit_event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'data_access',
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data_elements': data_elements,
            'purpose': purpose,
            'legal_basis': legal_basis,
            'ip_address': self._get_client_ip(),
            'session_id': self._get_session_id()
        }
        
        self.audit_events.append(audit_event)
        return audit_event['event_id']
    
    def log_data_modification(self, user_id: str, data_elements: List[str],
                            modification_type: str, before_values: Dict[str, Any],
                            after_values: Dict[str, Any]) -> str:
        """Log data modification event"""
        
        audit_event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'data_modification',
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data_elements': data_elements,
            'modification_type': modification_type,
            'before_values': before_values,
            'after_values': after_values,
            'ip_address': self._get_client_ip(),
            'session_id': self._get_session_id()
        }
        
        self.audit_events.append(audit_event)
        return audit_event['event_id']
    
    def log_consent_event(self, data_subject_id: str, consent_type: str,
                         consent_status: bool, purposes: List[str]) -> str:
        """Log consent-related events"""
        
        audit_event = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'consent_event',
            'timestamp': datetime.utcnow().isoformat(),
            'data_subject_id': data_subject_id,
            'consent_type': consent_type,
            'consent_status': consent_status,
            'purposes': purposes,
            'consent_mechanism': 'web_form',  # Could be parameterized
            'ip_address': self._get_client_ip()
        }
        
        self.audit_events.append(audit_event)
        return audit_event['event_id']
    
    def generate_compliance_report(self, start_date: datetime,
                                 end_date: datetime) -> Dict[str, Any]:
        """Generate compliance audit report for date range"""
        
        relevant_events = [
            event for event in self.audit_events
            if start_date <= datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')) <= end_date
        ]
        
        report = {
            'report_id': str(uuid.uuid4()),
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'total_events': len(relevant_events),
            'event_breakdown': {},
            'data_subject_requests': 0,
            'consent_events': 0,
            'data_breaches': 0,
            'policy_violations': 0
        }
        
        # Analyze events by type
        for event in relevant_events:
            event_type = event['event_type']
            report['event_breakdown'][event_type] = report['event_breakdown'].get(event_type, 0) + 1
            
            if event_type == 'consent_event':
                report['consent_events'] += 1
            elif event_type == 'data_subject_request':
                report['data_subject_requests'] += 1
        
        return report
    
    def _get_client_ip(self) -> str:
        """Get client IP address (simplified)"""
        return "192.168.1.1"  # Would get actual IP in production
    
    def _get_session_id(self) -> str:
        """Get session ID (simplified)"""
        return str(uuid.uuid4())[:8]  # Would get actual session in production
```

This completes Part 5 of Day 3, covering comprehensive compliance controls implementation including GDPR/CCPA automation, data classification systems, and privacy-preserving transformation techniques.