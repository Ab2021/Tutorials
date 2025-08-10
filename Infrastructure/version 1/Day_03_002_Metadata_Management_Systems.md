# Day 3.2: Metadata Management Systems Architecture

## 🗂️ Data Governance, Metadata & Cataloging - Part 2

**Focus**: Apache Atlas vs DataHub, Graph-Based Metadata Models, Distributed Synchronization  
**Duration**: 2-3 hours  
**Level**: Intermediate to Advanced  

---

## 🎯 Learning Objectives

- Master Apache Atlas and DataHub architectural differences and trade-offs
- Understand graph-based metadata models and query optimization techniques
- Learn distributed metadata synchronization patterns and consistency guarantees
- Implement schema evolution and versioning strategies

---

## 🏗️ Metadata Management Architecture Comparison

### **Apache Atlas vs DataHub: Architectural Deep Dive**

#### **1. Apache Atlas Architecture**
```
Atlas Architecture Stack:

┌─────────────────────────────────────────────────────────┐
│                    Atlas Web UI                         │
├─────────────────────────────────────────────────────────┤
│                    REST API Layer                       │
├─────────────────────────────────────────────────────────┤
│  Type System  │  Entity Store  │  Lineage Engine       │
├─────────────────────────────────────────────────────────┤
│           Apache Kafka (Notification Bus)              │
├─────────────────────────────────────────────────────────┤
│  HBase/JanusGraph  │  Elasticsearch  │  Solr (Search) │
└─────────────────────────────────────────────────────────┘
```

```python
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum

class MetadataArchitectureComparison:
    """Compare Atlas and DataHub architectures"""
    
    def __init__(self):
        self.architecture_profiles = {
            'atlas': self.create_atlas_profile(),
            'datahub': self.create_datahub_profile()
        }
    
    def create_atlas_profile(self) -> Dict[str, Any]:
        """Create Atlas architecture profile"""
        return {
            'metadata_storage': {
                'primary': 'JanusGraph',
                'search_index': 'Elasticsearch/Solr',
                'consistency_model': 'eventual_consistency',
                'transaction_support': True,
                'graph_traversal': 'Gremlin'
            },
            'type_system': {
                'schema_definition': 'JSON-based type definitions',
                'inheritance_support': True,
                'custom_attributes': True,
                'relationship_modeling': 'First-class graph relationships'
            },
            'notification_system': {
                'mechanism': 'Apache Kafka',
                'event_types': ['entity_create', 'entity_update', 'entity_delete'],
                'delivery_guarantee': 'at_least_once',
                'ordering_guarantee': 'partition_ordered'
            },
            'lineage_engine': {
                'storage': 'Graph-based (JanusGraph)',
                'query_language': 'Gremlin',
                'real_time_updates': True,
                'impact_analysis': 'Bidirectional graph traversal'
            },
            'scalability': {
                'horizontal_scaling': 'Limited (JanusGraph constraints)',
                'max_entities': '10M+',
                'query_performance': 'Good for graph operations',
                'write_throughput': 'Moderate'
            }
        }
    
    def create_datahub_profile(self) -> Dict[str, Any]:
        """Create DataHub architecture profile"""
        return {
            'metadata_storage': {
                'primary': 'MySQL/PostgreSQL/Cassandra',
                'search_index': 'Elasticsearch',
                'consistency_model': 'strong_consistency',
                'transaction_support': True,
                'graph_traversal': 'Custom GraphQL'
            },
            'type_system': {
                'schema_definition': 'Avro/PDL schemas',
                'inheritance_support': True,
                'custom_attributes': True,
                'relationship_modeling': 'Entity-relationship with aspects'
            },
            'notification_system': {
                'mechanism': 'Apache Kafka',
                'event_types': ['metadata_change_event', 'metadata_audit_event'],
                'delivery_guarantee': 'exactly_once',
                'ordering_guarantee': 'key_ordered'
            },
            'lineage_engine': {
                'storage': 'Relational + Graph views',
                'query_language': 'GraphQL',
                'real_time_updates': True,
                'impact_analysis': 'Materialized graph views'
            },
            'scalability': {
                'horizontal_scaling': 'Excellent (microservices)',
                'max_entities': '100M+',
                'query_performance': 'Excellent for entity operations',
                'write_throughput': 'High'
            }
        }
    
    def compare_architectures(self, use_case_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Compare architectures for specific use case requirements"""
        
        comparison_results = {
            'atlas_score': 0,
            'datahub_score': 0,
            'detailed_comparison': {},
            'recommendation': None
        }
        
        weights = use_case_requirements.get('weights', {
            'scalability': 0.25,
            'query_performance': 0.20,
            'ease_of_deployment': 0.15,
            'ecosystem_integration': 0.15,
            'customization': 0.15,
            'community_support': 0.10
        })
        
        # Scalability comparison
        if use_case_requirements.get('expected_entities', 0) > 50000000:  # 50M+
            comparison_results['datahub_score'] += weights['scalability'] * 0.8
            comparison_results['atlas_score'] += weights['scalability'] * 0.4
        else:
            comparison_results['datahub_score'] += weights['scalability'] * 0.6
            comparison_results['atlas_score'] += weights['scalability'] * 0.7
        
        # Query performance comparison
        if use_case_requirements.get('query_type') == 'complex_graph_traversal':
            comparison_results['atlas_score'] += weights['query_performance'] * 0.8
            comparison_results['datahub_score'] += weights['query_performance'] * 0.6
        elif use_case_requirements.get('query_type') == 'entity_lookup':
            comparison_results['datahub_score'] += weights['query_performance'] * 0.8
            comparison_results['atlas_score'] += weights['query_performance'] * 0.6
        
        # Ecosystem integration
        hadoop_ecosystem = use_case_requirements.get('requires_hadoop_integration', False)
        if hadoop_ecosystem:
            comparison_results['atlas_score'] += weights['ecosystem_integration'] * 0.9
            comparison_results['datahub_score'] += weights['ecosystem_integration'] * 0.5
        else:
            comparison_results['datahub_score'] += weights['ecosystem_integration'] * 0.8
            comparison_results['atlas_score'] += weights['ecosystem_integration'] * 0.6
        
        # Determine recommendation
        if comparison_results['atlas_score'] > comparison_results['datahub_score']:
            comparison_results['recommendation'] = 'atlas'
        else:
            comparison_results['recommendation'] = 'datahub'
        
        return comparison_results

@dataclass
class MetadataEntity:
    """Base metadata entity representation"""
    guid: str
    entity_type: str
    attributes: Dict[str, Any]
    relationships: List[Dict[str, Any]]
    creation_time: int
    update_time: int
    version: int

class MetadataTypeSystem:
    """Advanced metadata type system implementation"""
    
    def __init__(self):
        self.type_definitions = {}
        self.type_hierarchy = {}
        self.attribute_validators = {}
        
    def define_entity_type(self, type_name: str, type_definition: Dict[str, Any]) -> bool:
        """Define a new metadata entity type"""
        
        # Validate type definition structure
        required_fields = ['name', 'description', 'attributes', 'supertypes']
        if not all(field in type_definition for field in required_fields):
            raise ValueError(f"Type definition missing required fields: {required_fields}")
        
        # Process inheritance hierarchy
        supertypes = type_definition.get('supertypes', [])
        self.type_hierarchy[type_name] = {
            'supertypes': supertypes,
            'subtypes': set(),
            'depth': self.calculate_inheritance_depth(supertypes)
        }
        
        # Update parent types
        for supertype in supertypes:
            if supertype in self.type_hierarchy:
                self.type_hierarchy[supertype]['subtypes'].add(type_name)
        
        # Compile attribute definitions
        compiled_attributes = self.compile_attributes(
            type_definition['attributes'], supertypes
        )
        
        # Store compiled type definition
        self.type_definitions[type_name] = {
            'definition': type_definition,
            'compiled_attributes': compiled_attributes,
            'validation_schema': self.create_validation_schema(compiled_attributes)
        }
        
        return True
    
    def compile_attributes(self, attributes: Dict[str, Any], 
                         supertypes: List[str]) -> Dict[str, Any]:
        """Compile attributes including inherited ones"""
        compiled_attributes = {}
        
        # Inherit attributes from supertypes
        for supertype in supertypes:
            if supertype in self.type_definitions:
                parent_attributes = self.type_definitions[supertype]['compiled_attributes']
                compiled_attributes.update(parent_attributes)
        
        # Add own attributes (override inherited ones)
        compiled_attributes.update(attributes)
        
        return compiled_attributes
    
    def validate_entity(self, entity: MetadataEntity) -> Dict[str, Any]:
        """Validate entity against type system"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if entity type exists
        if entity.entity_type not in self.type_definitions:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Unknown entity type: {entity.entity_type}")
            return validation_result
        
        type_def = self.type_definitions[entity.entity_type]
        
        # Validate required attributes
        required_attrs = {
            name: attr_def for name, attr_def in type_def['compiled_attributes'].items()
            if attr_def.get('required', False)
        }
        
        for attr_name, attr_def in required_attrs.items():
            if attr_name not in entity.attributes:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required attribute: {attr_name}")
        
        # Validate attribute types and constraints
        for attr_name, attr_value in entity.attributes.items():
            if attr_name in type_def['compiled_attributes']:
                attr_def = type_def['compiled_attributes'][attr_name]
                attr_validation = self.validate_attribute(attr_value, attr_def)
                
                if not attr_validation['is_valid']:
                    validation_result['is_valid'] = False
                    validation_result['errors'].extend(attr_validation['errors'])
        
        return validation_result
    
    def validate_attribute(self, value: Any, attribute_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual attribute value"""
        validation_result = {'is_valid': True, 'errors': []}
        
        expected_type = attribute_definition.get('type')
        
        # Type validation
        if expected_type == 'string' and not isinstance(value, str):
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Expected string, got {type(value).__name__}")
        
        elif expected_type == 'integer' and not isinstance(value, int):
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Expected integer, got {type(value).__name__}")
        
        elif expected_type == 'array' and not isinstance(value, list):
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Expected array, got {type(value).__name__}")
        
        # Constraint validation
        constraints = attribute_definition.get('constraints', {})
        
        if 'min_length' in constraints and len(str(value)) < constraints['min_length']:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Value too short (min: {constraints['min_length']})")
        
        if 'max_length' in constraints and len(str(value)) > constraints['max_length']:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Value too long (max: {constraints['max_length']})")
        
        if 'enum' in constraints and value not in constraints['enum']:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Value not in allowed enum: {constraints['enum']}")
        
        return validation_result
```

---

## 🕸️ Graph-Based Metadata Models

### **Graph Theory Applied to Metadata Management**

#### **Graph Representation and Traversal Algorithms**
```python
from collections import defaultdict, deque
import networkx as nx
from typing import Set, Tuple

class MetadataGraph:
    """Graph-based metadata relationship modeling"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self.entity_cache = {}
        self.relationship_types = {}
        self.traversal_cache = {}
        
    def add_entity(self, entity: MetadataEntity) -> bool:
        """Add entity to metadata graph"""
        try:
            # Add node with attributes
            self.graph.add_node(
                entity.guid,
                entity_type=entity.entity_type,
                attributes=entity.attributes,
                version=entity.version,
                creation_time=entity.creation_time
            )
            
            # Cache entity for quick lookup
            self.entity_cache[entity.guid] = entity
            
            return True
        except Exception as e:
            print(f"Error adding entity {entity.guid}: {e}")
            return False
    
    def add_relationship(self, source_guid: str, target_guid: str, 
                        relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """Add relationship between entities"""
        try:
            # Ensure both entities exist
            if source_guid not in self.graph.nodes or target_guid not in self.graph.nodes:
                return False
            
            # Add edge with relationship properties
            self.graph.add_edge(
                source_guid,
                target_guid,
                relationship_type=relationship_type,
                properties=properties or {},
                creation_time=int(time.time())
            )
            
            # Track relationship types
            if relationship_type not in self.relationship_types:
                self.relationship_types[relationship_type] = {
                    'count': 0,
                    'source_types': set(),
                    'target_types': set()
                }
            
            self.relationship_types[relationship_type]['count'] += 1
            self.relationship_types[relationship_type]['source_types'].add(
                self.graph.nodes[source_guid]['entity_type']
            )
            self.relationship_types[relationship_type]['target_types'].add(
                self.graph.nodes[target_guid]['entity_type']
            )
            
            return True
        except Exception as e:
            print(f"Error adding relationship {source_guid} -> {target_guid}: {e}")
            return False
    
    def find_lineage_downstream(self, entity_guid: str, 
                              max_depth: int = 5) -> Dict[str, Any]:
        """Find downstream lineage using graph traversal"""
        if entity_guid not in self.graph.nodes:
            return {'error': 'Entity not found'}
        
        # Use BFS to find downstream dependencies
        downstream_entities = {}
        visited = set()
        queue = deque([(entity_guid, 0)])  # (node, depth)
        
        while queue:
            current_guid, depth = queue.popleft()
            
            if current_guid in visited or depth > max_depth:
                continue
            
            visited.add(current_guid)
            
            # Get entity information
            entity_info = {
                'guid': current_guid,
                'entity_type': self.graph.nodes[current_guid]['entity_type'],
                'depth': depth,
                'relationships': []
            }
            
            # Find all outgoing relationships
            for successor in self.graph.successors(current_guid):
                edge_data = self.graph.get_edge_data(current_guid, successor)
                
                for edge_key, edge_attrs in edge_data.items():
                    relationship_info = {
                        'target_guid': successor,
                        'relationship_type': edge_attrs['relationship_type'],
                        'properties': edge_attrs.get('properties', {})
                    }
                    entity_info['relationships'].append(relationship_info)
                
                # Add to queue for further traversal
                if depth < max_depth:
                    queue.append((successor, depth + 1))
            
            downstream_entities[current_guid] = entity_info
        
        return {
            'source_entity': entity_guid,
            'downstream_entities': downstream_entities,
            'total_entities': len(downstream_entities),
            'max_depth_reached': max(e['depth'] for e in downstream_entities.values())
        }
    
    def find_lineage_upstream(self, entity_guid: str, 
                            max_depth: int = 5) -> Dict[str, Any]:
        """Find upstream lineage using reverse graph traversal"""
        if entity_guid not in self.graph.nodes:
            return {'error': 'Entity not found'}
        
        upstream_entities = {}
        visited = set()
        queue = deque([(entity_guid, 0)])
        
        while queue:
            current_guid, depth = queue.popleft()
            
            if current_guid in visited or depth > max_depth:
                continue
            
            visited.add(current_guid)
            
            entity_info = {
                'guid': current_guid,
                'entity_type': self.graph.nodes[current_guid]['entity_type'],
                'depth': depth,
                'relationships': []
            }
            
            # Find all incoming relationships
            for predecessor in self.graph.predecessors(current_guid):
                edge_data = self.graph.get_edge_data(predecessor, current_guid)
                
                for edge_key, edge_attrs in edge_data.items():
                    relationship_info = {
                        'source_guid': predecessor,
                        'relationship_type': edge_attrs['relationship_type'],
                        'properties': edge_attrs.get('properties', {})
                    }
                    entity_info['relationships'].append(relationship_info)
                
                if depth < max_depth:
                    queue.append((predecessor, depth + 1))
            
            upstream_entities[current_guid] = entity_info
        
        return {
            'target_entity': entity_guid,
            'upstream_entities': upstream_entities,
            'total_entities': len(upstream_entities),
            'max_depth_reached': max(e['depth'] for e in upstream_entities.values())
        }
    
    def analyze_impact(self, entity_guid: str, 
                      change_type: str = 'schema_change') -> Dict[str, Any]:
        """Analyze impact of changes to an entity"""
        
        # Get both upstream and downstream lineage
        downstream = self.find_lineage_downstream(entity_guid)
        upstream = self.find_lineage_upstream(entity_guid)
        
        # Calculate impact scores based on relationship types and entity types
        impact_analysis = {
            'change_source': entity_guid,
            'change_type': change_type,
            'directly_affected': [],
            'indirectly_affected': [],
            'high_impact_entities': [],
            'impact_score': 0
        }
        
        # Analyze downstream impact
        for guid, entity_info in downstream.get('downstream_entities', {}).items():
            if guid == entity_guid:
                continue
                
            impact_level = self.calculate_entity_impact_level(
                entity_info, change_type
            )
            
            if impact_level['severity'] == 'high':
                impact_analysis['high_impact_entities'].append(guid)
            
            if entity_info['depth'] == 1:
                impact_analysis['directly_affected'].append(guid)
            else:
                impact_analysis['indirectly_affected'].append(guid)
            
            impact_analysis['impact_score'] += impact_level['score']
        
        # Add recommendations
        impact_analysis['recommendations'] = self.generate_impact_recommendations(
            impact_analysis
        )
        
        return impact_analysis
    
    def calculate_entity_impact_level(self, entity_info: Dict[str, Any], 
                                    change_type: str) -> Dict[str, Any]:
        """Calculate impact level for a specific entity"""
        
        base_score = 1.0
        severity_multipliers = {
            'schema_change': 2.0,
            'location_change': 1.5,
            'permission_change': 1.2,
            'metadata_update': 0.5
        }
        
        entity_type_multipliers = {
            'DataSet': 2.0,
            'Process': 1.8,
            'Table': 2.2,
            'Column': 1.0
        }
        
        # Base impact score
        impact_score = base_score * severity_multipliers.get(change_type, 1.0)
        impact_score *= entity_type_multipliers.get(entity_info['entity_type'], 1.0)
        
        # Relationship-based multipliers
        critical_relationships = ['contains', 'produces', 'consumes']
        for rel in entity_info['relationships']:
            if rel['relationship_type'] in critical_relationships:
                impact_score *= 1.3
        
        # Determine severity
        if impact_score >= 3.0:
            severity = 'high'
        elif impact_score >= 1.5:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'score': impact_score,
            'severity': severity,
            'factors': {
                'change_type': change_type,
                'entity_type': entity_info['entity_type'],
                'relationship_count': len(entity_info['relationships'])
            }
        }
    
    def optimize_graph_queries(self) -> Dict[str, Any]:
        """Optimize graph for common query patterns"""
        optimization_results = {
            'materialized_views_created': 0,
            'indexes_created': 0,
            'cache_strategies': [],
            'performance_improvement_estimate': 0
        }
        
        # Analyze query patterns (simplified)
        common_traversal_patterns = self.analyze_traversal_patterns()
        
        # Create materialized views for common paths
        for pattern in common_traversal_patterns:
            if pattern['frequency'] > 10:  # Frequently accessed paths
                self.create_materialized_lineage_view(pattern)
                optimization_results['materialized_views_created'] += 1
        
        # Implement caching strategies
        self.implement_query_caching()
        optimization_results['cache_strategies'].append('LRU query cache')
        optimization_results['cache_strategies'].append('Lineage path cache')
        
        return optimization_results
```

This completes Part 2 of Day 3, covering the advanced architecture of metadata management systems with deep comparison of Atlas vs DataHub and sophisticated graph-based metadata modeling techniques.