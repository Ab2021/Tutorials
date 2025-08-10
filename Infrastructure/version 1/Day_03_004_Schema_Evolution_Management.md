# Day 3.4: Schema Evolution Management & Compatibility

## 🔄 Data Governance, Metadata & Cataloging - Part 4

**Focus**: Schema Compatibility Rules, Version Migration, Registry Consensus Algorithms  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master schema compatibility rules and validation algorithms
- Understand forward/backward/full compatibility strategies and trade-offs
- Learn schema registry distributed consensus and conflict resolution
- Implement automated version migration and rollback procedures

---

## 📋 Schema Compatibility Theory

### **Compatibility Mathematics**

#### **Schema Compatibility Definitions**
```
Given schemas S₁ (producer) and S₂ (consumer):

Backward Compatibility: 
- Consumer(S₂) can read data produced by Producer(S₁)
- BC(S₁, S₂) = ∀d ∈ Data(S₁): readable(d, S₂) = true

Forward Compatibility:
- Consumer(S₁) can read data produced by Producer(S₂)  
- FC(S₁, S₂) = ∀d ∈ Data(S₂): readable(d, S₁) = true

Full Compatibility:
- FC(S₁, S₂) ∧ BC(S₁, S₂) = true
```

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import json
import hashlib
from datetime import datetime
import copy

class CompatibilityLevel(Enum):
    """Schema compatibility levels"""
    BACKWARD = "backward"
    FORWARD = "forward" 
    FULL = "full"
    NONE = "none"

class SchemaType(Enum):
    """Supported schema types"""
    AVRO = "avro"
    JSON_SCHEMA = "json_schema"
    PROTOBUF = "protobuf"
    PARQUET = "parquet"

class FieldChangeType(Enum):
    """Types of field changes in schema evolution"""
    FIELD_ADDED = "field_added"
    FIELD_REMOVED = "field_removed"
    FIELD_RENAMED = "field_renamed"
    TYPE_CHANGED = "type_changed"
    DEFAULT_CHANGED = "default_changed"
    REQUIRED_CHANGED = "required_changed"
    NESTED_CHANGED = "nested_changed"

@dataclass
class SchemaField:
    """Represents a field in a schema"""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: Optional[str] = None
    nested_fields: Optional[List['SchemaField']] = None
    
    def __hash__(self):
        return hash((self.name, self.type, self.required))

@dataclass
class Schema:
    """Represents a data schema"""
    name: str
    version: int
    schema_type: SchemaType
    fields: List[SchemaField]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    hash: Optional[str] = None
    
    def __post_init__(self):
        if self.hash is None:
            self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate hash of schema for quick comparison"""
        schema_content = {
            'name': self.name,
            'fields': [(f.name, f.type, f.required, f.default) for f in self.fields],
            'schema_type': self.schema_type.value
        }
        return hashlib.sha256(
            json.dumps(schema_content, sort_keys=True).encode()
        ).hexdigest()

class SchemaCompatibilityValidator:
    """Advanced schema compatibility validation engine"""
    
    def __init__(self):
        self.compatibility_rules = {
            SchemaType.AVRO: AvroCompatibilityRules(),
            SchemaType.JSON_SCHEMA: JsonSchemaCompatibilityRules(),
            SchemaType.PROTOBUF: ProtobufCompatibilityRules()
        }
        
    def validate_compatibility(self, old_schema: Schema, new_schema: Schema,
                             compatibility_level: CompatibilityLevel) -> Dict[str, Any]:
        """Validate schema compatibility between two versions"""
        
        if old_schema.schema_type != new_schema.schema_type:
            return {
                'compatible': False,
                'errors': ['Schema types must match'],
                'compatibility_level': compatibility_level.value
            }
        
        # Get appropriate rule engine
        rule_engine = self.compatibility_rules[old_schema.schema_type]
        
        # Perform compatibility check
        if compatibility_level == CompatibilityLevel.BACKWARD:
            return rule_engine.check_backward_compatibility(old_schema, new_schema)
        elif compatibility_level == CompatibilityLevel.FORWARD:
            return rule_engine.check_forward_compatibility(old_schema, new_schema)
        elif compatibility_level == CompatibilityLevel.FULL:
            return rule_engine.check_full_compatibility(old_schema, new_schema)
        else:
            return {'compatible': True, 'errors': [], 'warnings': []}
    
    def analyze_schema_changes(self, old_schema: Schema, 
                             new_schema: Schema) -> Dict[str, Any]:
        """Analyze detailed changes between schema versions"""
        
        changes_analysis = {
            'schema_name': new_schema.name,
            'version_change': f"{old_schema.version} -> {new_schema.version}",
            'field_changes': [],
            'breaking_changes': [],
            'non_breaking_changes': [],
            'change_summary': {
                'fields_added': 0,
                'fields_removed': 0,
                'fields_modified': 0,
                'breaking_change_count': 0
            }
        }
        
        # Create field maps for comparison
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # Find added fields
        added_fields = set(new_fields.keys()) - set(old_fields.keys())
        for field_name in added_fields:
            field = new_fields[field_name]
            change = {
                'field_name': field_name,
                'change_type': FieldChangeType.FIELD_ADDED.value,
                'old_definition': None,
                'new_definition': self._field_to_dict(field),
                'is_breaking': field.required and field.default is None
            }
            
            changes_analysis['field_changes'].append(change)
            changes_analysis['change_summary']['fields_added'] += 1
            
            if change['is_breaking']:
                changes_analysis['breaking_changes'].append(change)
                changes_analysis['change_summary']['breaking_change_count'] += 1
            else:
                changes_analysis['non_breaking_changes'].append(change)
        
        # Find removed fields
        removed_fields = set(old_fields.keys()) - set(new_fields.keys())
        for field_name in removed_fields:
            field = old_fields[field_name]
            change = {
                'field_name': field_name,
                'change_type': FieldChangeType.FIELD_REMOVED.value,
                'old_definition': self._field_to_dict(field),
                'new_definition': None,
                'is_breaking': True  # Removing fields is always breaking
            }
            
            changes_analysis['field_changes'].append(change)
            changes_analysis['breaking_changes'].append(change)
            changes_analysis['change_summary']['fields_removed'] += 1
            changes_analysis['change_summary']['breaking_change_count'] += 1
        
        # Find modified fields
        common_fields = set(old_fields.keys()) & set(new_fields.keys())
        for field_name in common_fields:
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]
            
            field_changes = self._compare_fields(old_field, new_field)
            if field_changes:
                for field_change in field_changes:
                    field_change['field_name'] = field_name
                    changes_analysis['field_changes'].append(field_change)
                    changes_analysis['change_summary']['fields_modified'] += 1
                    
                    if field_change['is_breaking']:
                        changes_analysis['breaking_changes'].append(field_change)
                        changes_analysis['change_summary']['breaking_change_count'] += 1
                    else:
                        changes_analysis['non_breaking_changes'].append(field_change)
        
        return changes_analysis
    
    def _field_to_dict(self, field: SchemaField) -> Dict[str, Any]:
        """Convert field to dictionary representation"""
        return {
            'name': field.name,
            'type': field.type,
            'required': field.required,
            'default': field.default,
            'description': field.description
        }
    
    def _compare_fields(self, old_field: SchemaField, 
                       new_field: SchemaField) -> List[Dict[str, Any]]:
        """Compare two fields and identify changes"""
        changes = []
        
        # Type change
        if old_field.type != new_field.type:
            is_breaking = not self._is_type_compatible(old_field.type, new_field.type)
            changes.append({
                'change_type': FieldChangeType.TYPE_CHANGED.value,
                'old_definition': {'type': old_field.type},
                'new_definition': {'type': new_field.type},
                'is_breaking': is_breaking
            })
        
        # Required change
        if old_field.required != new_field.required:
            # Making field required is breaking, making optional is not
            is_breaking = new_field.required and not old_field.required
            changes.append({
                'change_type': FieldChangeType.REQUIRED_CHANGED.value,
                'old_definition': {'required': old_field.required},
                'new_definition': {'required': new_field.required},
                'is_breaking': is_breaking
            })
        
        # Default value change
        if old_field.default != new_field.default:
            changes.append({
                'change_type': FieldChangeType.DEFAULT_CHANGED.value,
                'old_definition': {'default': old_field.default},
                'new_definition': {'default': new_field.default},
                'is_breaking': False  # Default changes are generally non-breaking
            })
        
        return changes
    
    def _is_type_compatible(self, old_type: str, new_type: str) -> bool:
        """Check if type change is compatible"""
        
        # Define compatible type promotions
        compatible_promotions = {
            'int': ['long', 'float', 'double'],
            'long': ['float', 'double'],
            'float': ['double'],
            'string': [],  # String generally can't be promoted
            'boolean': []  # Boolean can't be promoted
        }
        
        if old_type == new_type:
            return True
        
        return new_type in compatible_promotions.get(old_type, [])

class AvroCompatibilityRules:
    """Avro-specific compatibility rules"""
    
    def check_backward_compatibility(self, old_schema: Schema, 
                                   new_schema: Schema) -> Dict[str, Any]:
        """Check backward compatibility for Avro schemas"""
        result = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'compatibility_level': 'backward'
        }
        
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # Check for removed fields without defaults
        for field_name, field in old_fields.items():
            if field_name not in new_fields:
                if field.required and field.default is None:
                    result['compatible'] = False
                    result['errors'].append(
                        f"Required field '{field_name}' removed without default value"
                    )
        
        # Check for added required fields without defaults
        for field_name, field in new_fields.items():
            if field_name not in old_fields:
                if field.required and field.default is None:
                    result['compatible'] = False
                    result['errors'].append(
                        f"New required field '{field_name}' added without default value"
                    )
        
        # Check type changes
        for field_name in set(old_fields.keys()) & set(new_fields.keys()):
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]
            
            if not self._is_avro_type_compatible(old_field.type, new_field.type):
                result['compatible'] = False
                result['errors'].append(
                    f"Incompatible type change for field '{field_name}': "
                    f"{old_field.type} -> {new_field.type}"
                )
        
        return result
    
    def check_forward_compatibility(self, old_schema: Schema, 
                                  new_schema: Schema) -> Dict[str, Any]:
        """Check forward compatibility for Avro schemas"""
        # Forward compatibility is the reverse check
        return self.check_backward_compatibility(new_schema, old_schema)
    
    def check_full_compatibility(self, old_schema: Schema, 
                               new_schema: Schema) -> Dict[str, Any]:
        """Check full compatibility (both forward and backward)"""
        backward_result = self.check_backward_compatibility(old_schema, new_schema)
        forward_result = self.check_forward_compatibility(old_schema, new_schema)
        
        return {
            'compatible': backward_result['compatible'] and forward_result['compatible'],
            'errors': backward_result['errors'] + forward_result['errors'],
            'warnings': backward_result['warnings'] + forward_result['warnings'],
            'compatibility_level': 'full'
        }
    
    def _is_avro_type_compatible(self, old_type: str, new_type: str) -> bool:
        """Check Avro-specific type compatibility"""
        
        # Avro type promotion rules
        avro_promotions = {
            'int': ['long', 'float', 'double'],
            'long': ['float', 'double'],
            'float': ['double'],
            'string': ['bytes'],  # In some cases
            'bytes': ['string']   # In some cases
        }
        
        if old_type == new_type:
            return True
        
        return new_type in avro_promotions.get(old_type, [])

class JsonSchemaCompatibilityRules:
    """JSON Schema-specific compatibility rules"""
    
    def check_backward_compatibility(self, old_schema: Schema, 
                                   new_schema: Schema) -> Dict[str, Any]:
        """Check backward compatibility for JSON schemas"""
        result = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'compatibility_level': 'backward'
        }
        
        # JSON Schema is more flexible than Avro
        # Main concerns: required fields and type restrictions
        
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # Check for new required fields (breaking for backward compatibility)
        for field_name, field in new_fields.items():
            if (field_name not in old_fields and 
                field.required and field.default is None):
                result['compatible'] = False
                result['errors'].append(
                    f"New required field '{field_name}' breaks backward compatibility"
                )
        
        # Check for type restrictions
        for field_name in set(old_fields.keys()) & set(new_fields.keys()):
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]
            
            if self._is_type_restriction(old_field.type, new_field.type):
                result['warnings'].append(
                    f"Type restriction for field '{field_name}': "
                    f"{old_field.type} -> {new_field.type}"
                )
        
        return result
    
    def check_forward_compatibility(self, old_schema: Schema, 
                                  new_schema: Schema) -> Dict[str, Any]:
        """Check forward compatibility for JSON schemas"""
        result = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'compatibility_level': 'forward'
        }
        
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # Check for removed fields (can break forward compatibility)
        removed_fields = set(old_fields.keys()) - set(new_fields.keys())
        for field_name in removed_fields:
            old_field = old_fields[field_name]
            if old_field.required:
                result['compatible'] = False
                result['errors'].append(
                    f"Removed required field '{field_name}' breaks forward compatibility"
                )
        
        return result
    
    def check_full_compatibility(self, old_schema: Schema, 
                               new_schema: Schema) -> Dict[str, Any]:
        """Check full compatibility for JSON schemas"""
        backward_result = self.check_backward_compatibility(old_schema, new_schema)
        forward_result = self.check_forward_compatibility(old_schema, new_schema)
        
        return {
            'compatible': backward_result['compatible'] and forward_result['compatible'],
            'errors': backward_result['errors'] + forward_result['errors'],
            'warnings': backward_result['warnings'] + forward_result['warnings'],
            'compatibility_level': 'full'
        }
    
    def _is_type_restriction(self, old_type: str, new_type: str) -> bool:
        """Check if new type is more restrictive than old type"""
        
        # Type hierarchy from most permissive to most restrictive
        type_hierarchy = {
            'any': 5,
            'object': 4,
            'array': 3,
            'string': 2,
            'number': 2,
            'integer': 1,
            'boolean': 1
        }
        
        old_level = type_hierarchy.get(old_type, 0)
        new_level = type_hierarchy.get(new_type, 0)
        
        return new_level < old_level

class ProtobufCompatibilityRules:
    """Protocol Buffers-specific compatibility rules"""
    
    def check_backward_compatibility(self, old_schema: Schema, 
                                   new_schema: Schema) -> Dict[str, Any]:
        """Check backward compatibility for Protobuf schemas"""
        result = {
            'compatible': True,
            'errors': [],
            'warnings': [],
            'compatibility_level': 'backward'
        }
        
        # Protobuf has specific rules about field numbers and required fields
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        # Check for removed required fields
        for field_name, field in old_fields.items():
            if field_name not in new_fields and field.required:
                result['compatible'] = False
                result['errors'].append(
                    f"Required field '{field_name}' cannot be removed in Protobuf"
                )
        
        # Check for changed field types (more restrictive in Protobuf)
        for field_name in set(old_fields.keys()) & set(new_fields.keys()):
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]
            
            if not self._is_protobuf_type_compatible(old_field.type, new_field.type):
                result['compatible'] = False
                result['errors'].append(
                    f"Incompatible type change for field '{field_name}' in Protobuf: "
                    f"{old_field.type} -> {new_field.type}"
                )
        
        return result
    
    def check_forward_compatibility(self, old_schema: Schema, 
                                  new_schema: Schema) -> Dict[str, Any]:
        """Check forward compatibility for Protobuf schemas"""
        return self.check_backward_compatibility(new_schema, old_schema)
    
    def check_full_compatibility(self, old_schema: Schema, 
                               new_schema: Schema) -> Dict[str, Any]:
        """Check full compatibility for Protobuf schemas"""
        backward_result = self.check_backward_compatibility(old_schema, new_schema)
        forward_result = self.check_forward_compatibility(old_schema, new_schema)
        
        return {
            'compatible': backward_result['compatible'] and forward_result['compatible'],
            'errors': backward_result['errors'] + forward_result['errors'],
            'warnings': backward_result['warnings'] + forward_result['warnings'],
            'compatibility_level': 'full'
        }
    
    def _is_protobuf_type_compatible(self, old_type: str, new_type: str) -> bool:
        """Check Protobuf-specific type compatibility"""
        
        # Protobuf type compatibility is more restrictive
        protobuf_compatible = {
            'int32': ['int64', 'uint32', 'uint64'],
            'int64': ['uint64'],
            'uint32': ['uint64'],
            'float': ['double'],
            'string': ['bytes'],
            'bytes': ['string']
        }
        
        if old_type == new_type:
            return True
        
        return new_type in protobuf_compatible.get(old_type, [])

class SchemaEvolutionManager:
    """Manages schema evolution and migration strategies"""
    
    def __init__(self):
        self.compatibility_validator = SchemaCompatibilityValidator()
        self.migration_strategies = {}
        
    def plan_schema_migration(self, current_schema: Schema, 
                            target_schema: Schema,
                            compatibility_level: CompatibilityLevel) -> Dict[str, Any]:
        """Plan migration strategy for schema evolution"""
        
        # Validate compatibility
        compatibility_result = self.compatibility_validator.validate_compatibility(
            current_schema, target_schema, compatibility_level
        )
        
        migration_plan = {
            'migration_id': f"{current_schema.name}_{current_schema.version}_to_{target_schema.version}",
            'compatibility_check': compatibility_result,
            'migration_strategy': None,
            'steps': [],
            'estimated_duration': 0,
            'risk_level': 'low',
            'rollback_plan': None
        }
        
        if not compatibility_result['compatible']:
            # Breaking changes require special handling
            migration_plan['migration_strategy'] = 'blue_green_deployment'
            migration_plan['risk_level'] = 'high'
            migration_plan['steps'] = self._create_breaking_change_migration_steps(
                current_schema, target_schema
            )
        else:
            # Compatible changes can use rolling deployment
            migration_plan['migration_strategy'] = 'rolling_deployment'
            migration_plan['risk_level'] = 'low'
            migration_plan['steps'] = self._create_compatible_migration_steps(
                current_schema, target_schema
            )
        
        # Estimate duration based on complexity
        migration_plan['estimated_duration'] = self._estimate_migration_duration(
            migration_plan['steps']
        )
        
        # Create rollback plan
        migration_plan['rollback_plan'] = self._create_rollback_plan(
            current_schema, target_schema, migration_plan['migration_strategy']
        )
        
        return migration_plan
    
    def _create_breaking_change_migration_steps(self, current_schema: Schema, 
                                              target_schema: Schema) -> List[Dict[str, Any]]:
        """Create migration steps for breaking changes"""
        steps = [
            {
                'step_number': 1,
                'description': 'Deploy new schema version alongside current version',
                'action': 'deploy_parallel_schema',
                'estimated_time_minutes': 15,
                'validation_criteria': ['new_schema_deployed', 'health_checks_pass']
            },
            {
                'step_number': 2,
                'description': 'Update producers to write to both schema versions',
                'action': 'dual_write_producers',
                'estimated_time_minutes': 30,
                'validation_criteria': ['dual_write_active', 'no_data_loss']
            },
            {
                'step_number': 3,
                'description': 'Migrate consumers to new schema version',
                'action': 'migrate_consumers',
                'estimated_time_minutes': 60,
                'validation_criteria': ['consumers_migrated', 'processing_stable']
            },
            {
                'step_number': 4,
                'description': 'Stop writing to old schema version',
                'action': 'stop_old_writes',
                'estimated_time_minutes': 10,
                'validation_criteria': ['old_writes_stopped', 'new_writes_only']
            },
            {
                'step_number': 5,
                'description': 'Remove old schema version',
                'action': 'cleanup_old_schema',
                'estimated_time_minutes': 15,
                'validation_criteria': ['old_schema_removed', 'cleanup_complete']
            }
        ]
        
        return steps
    
    def _create_compatible_migration_steps(self, current_schema: Schema, 
                                         target_schema: Schema) -> List[Dict[str, Any]]:
        """Create migration steps for compatible changes"""
        steps = [
            {
                'step_number': 1,
                'description': 'Deploy new schema version',
                'action': 'deploy_new_schema',
                'estimated_time_minutes': 10,
                'validation_criteria': ['schema_deployed', 'validation_passed']
            },
            {
                'step_number': 2,
                'description': 'Rolling update of producers',
                'action': 'update_producers',
                'estimated_time_minutes': 20,
                'validation_criteria': ['producers_updated', 'data_flow_stable']
            },
            {
                'step_number': 3,
                'description': 'Rolling update of consumers',
                'action': 'update_consumers',
                'estimated_time_minutes': 25,
                'validation_criteria': ['consumers_updated', 'processing_stable']
            }
        ]
        
        return steps
    
    def _estimate_migration_duration(self, steps: List[Dict[str, Any]]) -> int:
        """Estimate total migration duration in minutes"""
        base_duration = sum(step['estimated_time_minutes'] for step in steps)
        
        # Add buffer for validation and potential issues
        buffer_percentage = 0.3  # 30% buffer
        total_duration = int(base_duration * (1 + buffer_percentage))
        
        return total_duration
    
    def _create_rollback_plan(self, current_schema: Schema, target_schema: Schema,
                            migration_strategy: str) -> Dict[str, Any]:
        """Create rollback plan for migration"""
        
        rollback_plan = {
            'rollback_strategy': migration_strategy,
            'rollback_steps': [],
            'rollback_triggers': [
                'data_corruption_detected',
                'processing_errors_exceed_threshold',
                'consumer_failures',
                'manual_rollback_requested'
            ],
            'estimated_rollback_time_minutes': 0
        }
        
        if migration_strategy == 'blue_green_deployment':
            rollback_plan['rollback_steps'] = [
                {
                    'step': 'Switch traffic back to old schema version',
                    'estimated_time_minutes': 5
                },
                {
                    'step': 'Verify data integrity',
                    'estimated_time_minutes': 10
                },
                {
                    'step': 'Remove new schema deployment',
                    'estimated_time_minutes': 10
                }
            ]
        else:  # rolling_deployment
            rollback_plan['rollback_steps'] = [
                {
                    'step': 'Deploy previous schema version',
                    'estimated_time_minutes': 10
                },
                {
                    'step': 'Rolling rollback of consumers',
                    'estimated_time_minutes': 20
                },
                {
                    'step': 'Rolling rollback of producers',
                    'estimated_time_minutes': 15
                }
            ]
        
        rollback_plan['estimated_rollback_time_minutes'] = sum(
            step['estimated_time_minutes'] for step in rollback_plan['rollback_steps']
        )
        
        return rollback_plan
```

This completes Part 4 of Day 3, covering advanced schema evolution management, compatibility validation algorithms, and automated migration strategies with detailed rollback procedures.