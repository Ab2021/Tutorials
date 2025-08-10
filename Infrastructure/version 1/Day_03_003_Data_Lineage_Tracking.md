# Day 3.3: Data Lineage Tracking & Impact Analysis

## 🔗 Data Governance, Metadata & Cataloging - Part 3

**Focus**: Lineage Graph Construction, Impact Analysis, Cross-System Tracking  
**Duration**: 2-3 hours  
**Level**: Advanced  

---

## 🎯 Learning Objectives

- Master lineage graph construction algorithms and optimization techniques
- Understand impact analysis and dependency resolution at scale
- Learn column-level vs table-level lineage trade-offs and implementation strategies
- Implement cross-system lineage tracking with distributed consistency

---

## 🕸️ Lineage Graph Construction Theory

### **Mathematical Foundations of Data Lineage**

#### **Lineage Graph Definition**
```
Lineage Graph G = (V, E, T, M)

Where:
- V = set of data entities (tables, columns, processes)
- E = set of directed edges (transformations, dependencies)
- T = temporal dimension (time-based lineage evolution)
- M = metadata attributes (transformation logic, confidence scores)

Edge Properties:
- Transformation Function: f: D₁ → D₂
- Confidence Score: c ∈ [0, 1]
- Temporal Validity: [t_start, t_end]
```

```python
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque

class LineageGranularity(Enum):
    """Granularity levels for lineage tracking"""
    SYSTEM_LEVEL = "system"
    DATABASE_LEVEL = "database"
    TABLE_LEVEL = "table"
    COLUMN_LEVEL = "column"
    FIELD_LEVEL = "field"

class LineageConfidence(Enum):
    """Confidence levels for lineage relationships"""
    CERTAIN = 1.0      # Direct observation or explicit definition
    HIGH = 0.8         # Strong inference with multiple evidence points
    MEDIUM = 0.6       # Moderate inference with some evidence
    LOW = 0.3          # Weak inference or heuristic-based
    UNCERTAIN = 0.1    # Speculative or incomplete information

@dataclass
class LineageNode:
    """Represents a node in the lineage graph"""
    node_id: str
    node_type: str  # table, column, process, system
    qualified_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __hash__(self):
        return hash(self.node_id)

@dataclass
class LineageEdge:
    """Represents an edge in the lineage graph"""
    source_id: str
    target_id: str
    relationship_type: str  # produces, consumes, transforms, contains
    transformation_logic: Optional[str] = None
    confidence: float = 1.0
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_to: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.relationship_type))

class AdvancedLineageTracker:
    """Advanced data lineage tracking and analysis system"""
    
    def __init__(self):
        self.lineage_graph = nx.MultiDiGraph()
        self.temporal_snapshots = {}  # timestamp -> graph snapshot
        self.confidence_threshold = 0.5
        self.node_registry = {}
        self.edge_registry = {}
        self.lineage_cache = {}
        
    def add_lineage_node(self, node: LineageNode) -> bool:
        """Add a node to the lineage graph"""
        try:
            # Add node to graph with all attributes
            self.lineage_graph.add_node(
                node.node_id,
                node_type=node.node_type,
                qualified_name=node.qualified_name,
                attributes=node.attributes,
                created_at=node.created_at,
                updated_at=node.updated_at
            )
            
            # Register node for quick lookup
            self.node_registry[node.node_id] = node
            
            # Invalidate related caches
            self._invalidate_cache_for_node(node.node_id)
            
            return True
        except Exception as e:
            print(f"Error adding lineage node {node.node_id}: {e}")
            return False
    
    def add_lineage_edge(self, edge: LineageEdge) -> bool:
        """Add an edge to the lineage graph"""
        try:
            # Verify both nodes exist
            if (edge.source_id not in self.lineage_graph.nodes or 
                edge.target_id not in self.lineage_graph.nodes):
                return False
            
            # Add edge with all metadata
            self.lineage_graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.relationship_type,
                relationship_type=edge.relationship_type,
                transformation_logic=edge.transformation_logic,
                confidence=edge.confidence,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
                metadata=edge.metadata
            )
            
            # Register edge
            edge_key = (edge.source_id, edge.target_id, edge.relationship_type)
            self.edge_registry[edge_key] = edge
            
            # Invalidate related caches
            self._invalidate_cache_for_edge(edge.source_id, edge.target_id)
            
            return True
        except Exception as e:
            print(f"Error adding lineage edge {edge.source_id} -> {edge.target_id}: {e}")
            return False
    
    def trace_upstream_lineage(self, entity_id: str, 
                             max_depth: int = 10,
                             include_confidence: bool = True) -> Dict[str, Any]:
        """Trace upstream lineage with confidence scoring"""
        
        if entity_id not in self.lineage_graph.nodes:
            return {'error': f'Entity {entity_id} not found'}
        
        # Check cache first
        cache_key = f"upstream_{entity_id}_{max_depth}_{include_confidence}"
        if cache_key in self.lineage_cache:
            return self.lineage_cache[cache_key]
        
        upstream_lineage = {
            'target_entity': entity_id,
            'lineage_paths': [],
            'confidence_distribution': {},
            'total_upstream_entities': 0,
            'max_depth_reached': 0
        }
        
        # Use modified DFS to find all paths
        visited_paths = set()
        all_paths = []
        
        def dfs_upstream(current_id: str, path: List[Dict], depth: int, 
                        accumulated_confidence: float):
            if depth > max_depth:
                return
                
            path_signature = tuple(node['entity_id'] for node in path)
            if path_signature in visited_paths:
                return
            
            visited_paths.add(path_signature)
            upstream_lineage['max_depth_reached'] = max(
                upstream_lineage['max_depth_reached'], depth
            )
            
            # Get predecessors
            predecessors = list(self.lineage_graph.predecessors(current_id))
            
            if not predecessors:  # Root node
                if len(path) > 1:  # Don't include single-node paths
                    all_paths.append({
                        'path': path.copy(),
                        'total_confidence': accumulated_confidence,
                        'length': len(path)
                    })
                return
            
            for pred_id in predecessors:
                # Get edge data
                edge_data = self.lineage_graph.get_edge_data(pred_id, current_id)
                
                for edge_key, edge_attrs in edge_data.items():
                    edge_confidence = edge_attrs.get('confidence', 1.0)
                    
                    # Skip low-confidence edges if threshold is set
                    if edge_confidence < self.confidence_threshold:
                        continue
                    
                    # Calculate accumulated confidence (multiplicative)
                    new_confidence = accumulated_confidence * edge_confidence
                    
                    # Create path node
                    path_node = {
                        'entity_id': pred_id,
                        'entity_type': self.lineage_graph.nodes[pred_id]['node_type'],
                        'qualified_name': self.lineage_graph.nodes[pred_id]['qualified_name'],
                        'relationship_type': edge_attrs['relationship_type'],
                        'transformation_logic': edge_attrs.get('transformation_logic'),
                        'confidence': edge_confidence,
                        'depth': depth
                    }
                    
                    new_path = path + [path_node]
                    dfs_upstream(pred_id, new_path, depth + 1, new_confidence)
        
        # Start DFS from target entity
        start_node = {
            'entity_id': entity_id,
            'entity_type': self.lineage_graph.nodes[entity_id]['node_type'],
            'qualified_name': self.lineage_graph.nodes[entity_id]['qualified_name'],
            'depth': 0
        }
        
        dfs_upstream(entity_id, [start_node], 0, 1.0)
        
        # Process and rank paths
        all_paths.sort(key=lambda x: x['total_confidence'], reverse=True)
        upstream_lineage['lineage_paths'] = all_paths
        upstream_lineage['total_upstream_entities'] = len(
            set(node['entity_id'] for path in all_paths for node in path['path'])
        )
        
        # Calculate confidence distribution
        confidence_buckets = defaultdict(int)
        for path in all_paths:
            confidence_bucket = self._get_confidence_bucket(path['total_confidence'])
            confidence_buckets[confidence_bucket] += 1
        
        upstream_lineage['confidence_distribution'] = dict(confidence_buckets)
        
        # Cache result
        self.lineage_cache[cache_key] = upstream_lineage
        
        return upstream_lineage
    
    def trace_downstream_lineage(self, entity_id: str, 
                               max_depth: int = 10) -> Dict[str, Any]:
        """Trace downstream lineage with impact analysis"""
        
        if entity_id not in self.lineage_graph.nodes:
            return {'error': f'Entity {entity_id} not found'}
        
        downstream_lineage = {
            'source_entity': entity_id,
            'affected_entities': {},
            'impact_levels': {},
            'total_downstream_entities': 0,
            'critical_paths': []
        }
        
        # BFS for downstream traversal
        queue = deque([(entity_id, 0, 1.0, [])])  # (entity, depth, confidence, path)
        visited = set()
        
        while queue:
            current_id, depth, path_confidence, path = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get entity information
            entity_info = {
                'entity_id': current_id,
                'entity_type': self.lineage_graph.nodes[current_id]['node_type'],
                'qualified_name': self.lineage_graph.nodes[current_id]['qualified_name'],
                'depth': depth,
                'path_confidence': path_confidence,
                'lineage_path': path.copy()
            }
            
            downstream_lineage['affected_entities'][current_id] = entity_info
            
            # Calculate impact level
            impact_level = self._calculate_impact_level(entity_info)
            downstream_lineage['impact_levels'][current_id] = impact_level
            
            # Add to critical paths if high impact
            if impact_level['severity'] == 'high':
                downstream_lineage['critical_paths'].append({
                    'target_entity': current_id,
                    'path': path + [current_id],
                    'impact_score': impact_level['score']
                })
            
            # Continue traversal to successors
            for succ_id in self.lineage_graph.successors(current_id):
                edge_data = self.lineage_graph.get_edge_data(current_id, succ_id)
                
                for edge_key, edge_attrs in edge_data.items():
                    edge_confidence = edge_attrs.get('confidence', 1.0)
                    
                    if edge_confidence >= self.confidence_threshold:
                        new_confidence = path_confidence * edge_confidence
                        new_path = path + [current_id]
                        
                        queue.append((succ_id, depth + 1, new_confidence, new_path))
        
        downstream_lineage['total_downstream_entities'] = len(visited) - 1  # Exclude source
        
        return downstream_lineage
    
    def perform_impact_analysis(self, entity_id: str, 
                              change_type: str,
                              change_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive impact analysis for entity changes"""
        
        # Get downstream lineage
        downstream = self.trace_downstream_lineage(entity_id)
        
        if 'error' in downstream:
            return downstream
        
        impact_analysis = {
            'change_source': entity_id,
            'change_type': change_type,
            'change_details': change_details or {},
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'impact_summary': {
                'total_affected_entities': downstream['total_downstream_entities'],
                'high_impact_entities': 0,
                'medium_impact_entities': 0,
                'low_impact_entities': 0
            },
            'detailed_impacts': {},
            'recommended_actions': [],
            'risk_assessment': {}
        }
        
        # Analyze each affected entity
        for entity_id, entity_info in downstream['affected_entities'].items():
            if entity_id == impact_analysis['change_source']:
                continue
            
            # Calculate specific impact for this change type
            specific_impact = self._calculate_specific_impact(
                entity_info, change_type, change_details
            )
            
            impact_analysis['detailed_impacts'][entity_id] = specific_impact
            
            # Update summary counts
            severity = specific_impact['severity']
            impact_analysis['impact_summary'][f'{severity}_impact_entities'] += 1
        
        # Generate recommendations
        impact_analysis['recommended_actions'] = self._generate_impact_recommendations(
            impact_analysis
        )
        
        # Assess overall risk
        impact_analysis['risk_assessment'] = self._assess_change_risk(impact_analysis)
        
        return impact_analysis
    
    def _calculate_specific_impact(self, entity_info: Dict[str, Any], 
                                 change_type: str, 
                                 change_details: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate specific impact based on change type"""
        
        base_impact_scores = {
            'schema_change': 3.0,
            'data_type_change': 4.0,
            'column_removal': 5.0,
            'table_removal': 5.0,
            'location_change': 2.0,
            'permission_change': 2.5,
            'format_change': 3.5
        }
        
        entity_type_multipliers = {
            'table': 1.0,
            'column': 0.8,
            'view': 1.2,
            'process': 1.5,
            'dashboard': 2.0,
            'report': 2.0
        }
        
        # Base impact score
        base_score = base_impact_scores.get(change_type, 2.0)
        
        # Entity type multiplier
        entity_type = entity_info.get('entity_type', 'table')
        type_multiplier = entity_type_multipliers.get(entity_type, 1.0)
        
        # Path confidence affects impact certainty
        confidence_multiplier = entity_info.get('path_confidence', 1.0)
        
        # Depth affects impact severity (closer = higher impact)
        depth_factor = max(0.1, 1.0 - (entity_info.get('depth', 1) * 0.1))
        
        # Calculate final impact score
        impact_score = (base_score * type_multiplier * 
                       confidence_multiplier * depth_factor)
        
        # Determine severity
        if impact_score >= 4.0:
            severity = 'high'
        elif impact_score >= 2.0:
            severity = 'medium'
        else:
            severity = 'low'
        
        return {
            'impact_score': impact_score,
            'severity': severity,
            'confidence': confidence_multiplier,
            'factors': {
                'base_score': base_score,
                'entity_type_multiplier': type_multiplier,
                'confidence_multiplier': confidence_multiplier,
                'depth_factor': depth_factor
            },
            'estimated_effort_hours': self._estimate_remediation_effort(
                impact_score, entity_type, change_type
            )
        }
    
    def _estimate_remediation_effort(self, impact_score: float, 
                                   entity_type: str, change_type: str) -> float:
        """Estimate effort required for remediation"""
        
        base_effort_hours = {
            'schema_change': 4.0,
            'data_type_change': 6.0,
            'column_removal': 8.0,
            'table_removal': 12.0,
            'location_change': 2.0,
            'permission_change': 1.0,
            'format_change': 5.0
        }
        
        entity_effort_multipliers = {
            'table': 1.0,
            'column': 0.5,
            'view': 1.2,
            'process': 2.0,
            'dashboard': 1.5,
            'report': 1.3
        }
        
        base_effort = base_effort_hours.get(change_type, 4.0)
        entity_multiplier = entity_effort_multipliers.get(entity_type, 1.0)
        impact_multiplier = min(3.0, impact_score / 2.0)  # Cap at 3x
        
        return base_effort * entity_multiplier * impact_multiplier
    
    def create_lineage_snapshot(self, snapshot_name: str) -> bool:
        """Create a temporal snapshot of the current lineage graph"""
        try:
            snapshot_data = {
                'timestamp': datetime.utcnow(),
                'snapshot_name': snapshot_name,
                'graph_copy': self.lineage_graph.copy(),
                'node_count': self.lineage_graph.number_of_nodes(),
                'edge_count': self.lineage_graph.number_of_edges(),
                'metadata': {
                    'confidence_threshold': self.confidence_threshold,
                    'node_types': self._get_node_type_distribution(),
                    'relationship_types': self._get_relationship_type_distribution()
                }
            }
            
            self.temporal_snapshots[snapshot_name] = snapshot_data
            return True
        except Exception as e:
            print(f"Error creating lineage snapshot {snapshot_name}: {e}")
            return False
    
    def compare_lineage_evolution(self, snapshot1: str, 
                                snapshot2: str) -> Dict[str, Any]:
        """Compare lineage evolution between two snapshots"""
        
        if (snapshot1 not in self.temporal_snapshots or 
            snapshot2 not in self.temporal_snapshots):
            return {'error': 'One or both snapshots not found'}
        
        snap1 = self.temporal_snapshots[snapshot1]
        snap2 = self.temporal_snapshots[snapshot2]
        
        graph1 = snap1['graph_copy']
        graph2 = snap2['graph_copy']
        
        evolution_analysis = {
            'snapshot1': snapshot1,
            'snapshot2': snapshot2,
            'time_difference': (snap2['timestamp'] - snap1['timestamp']).total_seconds(),
            'node_changes': {
                'added': [],
                'removed': [],
                'modified': []
            },
            'edge_changes': {
                'added': [],
                'removed': [],
                'modified': []
            },
            'statistics': {
                'nodes_added': 0,
                'nodes_removed': 0,
                'edges_added': 0,
                'edges_removed': 0
            }
        }
        
        # Compare nodes
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())
        
        added_nodes = nodes2 - nodes1
        removed_nodes = nodes1 - nodes2
        
        evolution_analysis['node_changes']['added'] = list(added_nodes)
        evolution_analysis['node_changes']['removed'] = list(removed_nodes)
        evolution_analysis['statistics']['nodes_added'] = len(added_nodes)
        evolution_analysis['statistics']['nodes_removed'] = len(removed_nodes)
        
        # Compare edges
        edges1 = set(graph1.edges(keys=True))
        edges2 = set(graph2.edges(keys=True))
        
        added_edges = edges2 - edges1
        removed_edges = edges1 - edges2
        
        evolution_analysis['edge_changes']['added'] = list(added_edges)
        evolution_analysis['edge_changes']['removed'] = list(removed_edges)
        evolution_analysis['statistics']['edges_added'] = len(added_edges)
        evolution_analysis['statistics']['edges_removed'] = len(removed_edges)
        
        return evolution_analysis
```

This completes Part 3 of Day 3, covering advanced data lineage tracking algorithms, impact analysis techniques, and temporal lineage evolution analysis.