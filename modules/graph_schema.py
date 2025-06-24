import uuid
from typing import Any, List, Optional, Dict, Type, Union

# --- Architectural Graph Representation ---
class ArchitecturalNode:
    def __init__(self,
                 node_id: Optional[str] = None,
                 properties: Optional[Dict[str, Any]] = None):
        self.id: str = node_id if node_id is not None else str(uuid.uuid4())
        _props = properties if properties is not None else {}
        if 'layer_type' not in _props:
            _props['layer_type'] = 'generic'
        self.properties: Dict[str, Any] = _props

    @property
    def node_type(self) -> str:
        return self.properties.get('layer_type', 'generic')

    @node_type.setter
    def node_type(self, value: str) -> None:
        self.properties['layer_type'] = value

    def __str__(self) -> str:
        return f"Node(id={self.id}, type={self.node_type}, props={self.properties})"

class ArchitecturalEdge:
    def __init__(self,
                 source_node_id: str,
                 target_node_id: str,
                 edge_id: Optional[str] = None,
                 weight: float = 0.0,
                 properties: Optional[Dict[str, Any]] = None):
        self.id: str = edge_id if edge_id is not None else str(uuid.uuid4())
        if not source_node_id or not target_node_id:
            raise ValueError("Source and target node IDs are required for an edge.")
        self.source_node_id: str = source_node_id
        self.target_node_id: str = target_node_id
        self.weight: float = weight
        self.properties: Dict[str, Any] = properties if properties is not None else {}

    def __str__(self) -> str:
        return f"Edge(id={self.id}, from={self.source_node_id}, to={self.target_node_id}, w={self.weight}, props={self.properties})"

class PartitionSchema:
    def __init__(self, name:str, target_element_type:str, partitions:List[set[str]], schema_id:Optional[str]=None, metadata:Optional[Dict[str,Any]]=None):
        self.id=schema_id if schema_id is not None else str(uuid.uuid4()); self.name=name
        self.target_element_type=target_element_type; self.partitions=partitions
        self.metadata=metadata if metadata is not None else {}
    def __str__(self): return f"PSchema(id={self.id},'{self.name}',type='{self.target_element_type}',parts={len(self.partitions)})"

class NetworkGraph:
    def __init__(self, graph_id: Optional[str] = None, properties: Optional[Dict[str, Any]] = None):
        self.id: str = graph_id if graph_id is not None else str(uuid.uuid4())
        self.nodes: Dict[str, ArchitecturalNode] = {}
        self.edges: Dict[str, ArchitecturalEdge] = {}
        self.adj: Dict[str, Dict[str, List[str]]] = {}
        self.properties: Dict[str, Any] = properties if properties is not None else {}
        self.partition_schemas: List[PartitionSchema] = []

    def add_node(self, node: ArchitecturalNode) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists.")
        self.nodes[node.id] = node
        self.adj[node.id] = {'in': [], 'out': []}

    def add_layer_node(self, name: str, layer_type: str, node_attributes: Optional[Dict[str, Any]] = None) -> None:
        if name in self.nodes:
            raise ValueError(f"Node with name (ID) '{name}' already exists.")
        props = node_attributes.copy() if node_attributes is not None else {}
        props['layer_type'] = layer_type
        node = ArchitecturalNode(node_id=name, properties=props)
        self.add_node(node)

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes: return
        for edge_id in list(self.edges.keys()):
            edge = self.edges[edge_id]
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                self.remove_edge(edge_id)
        if node_id in self.nodes: del self.nodes[node_id]
        if node_id in self.adj: del self.adj[node_id]

    def add_edge(self, edge: ArchitecturalEdge) -> None:
        if edge.id in self.edges: raise ValueError(f"Edge with ID {edge.id} already exists.")
        if edge.source_node_id not in self.nodes: raise ValueError(f"Source node with ID {edge.source_node_id} does not exist.")
        if edge.target_node_id not in self.nodes: raise ValueError(f"Target node with ID {edge.target_node_id} does not exist.")
        self.edges[edge.id] = edge
        self.adj[edge.source_node_id]['out'].append(edge.id)
        self.adj[edge.target_node_id]['in'].append(edge.id)

    def connect_layers(self, source_name: str, dest_name: str, edge_attributes: Optional[Dict[str, Any]] = None) -> None:
        if source_name not in self.nodes: raise ValueError(f"Source node '{source_name}' not found.")
        if dest_name not in self.nodes: raise ValueError(f"Destination node '{dest_name}' not found.")
        edge_id_prop = (edge_attributes or {}).pop('id', None)
        edge_id = edge_id_prop if edge_id_prop else f"edge_{source_name}_to_{dest_name}_{str(uuid.uuid4())[:4]}"
        while edge_id in self.edges:
             edge_id = f"edge_{source_name}_to_{dest_name}_{str(uuid.uuid4())[:4]}"
        edge = ArchitecturalEdge(source_node_id=source_name, target_node_id=dest_name, edge_id=edge_id, properties=edge_attributes)
        self.add_edge(edge)

    def remove_edge(self, edge_id: str) -> None:
        if edge_id not in self.edges: return
        edge = self.edges[edge_id]
        if edge.source_node_id in self.adj and edge_id in self.adj[edge.source_node_id]['out']:
            self.adj[edge.source_node_id]['out'].remove(edge_id)
        if edge.target_node_id in self.adj and edge_id in self.adj[edge.target_node_id]['in']:
            self.adj[edge.target_node_id]['in'].remove(edge_id)
        if edge_id in self.edges: del self.edges[edge_id]

    def get_node(self, node_id: str) -> Optional[ArchitecturalNode]: return self.nodes.get(node_id)
    def get_edge(self, edge_id: str) -> Optional[ArchitecturalEdge]: return self.edges.get(edge_id)
    def get_incoming_edges(self, node_id: str) -> List[ArchitecturalEdge]:
        if node_id not in self.adj: return []
        return [self.edges[edge_id] for edge_id in self.adj[node_id].get('in', []) if edge_id in self.edges]
    def get_outgoing_edges(self, node_id: str) -> List[ArchitecturalEdge]:
        if node_id not in self.adj: return []
        return [self.edges[edge_id] for edge_id in self.adj[node_id].get('out', []) if edge_id in self.edges]

    def add_partition_schema(self, schema: PartitionSchema) -> None: self.partition_schemas.append(schema)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'properties': self.properties,
            'nodes': {node_id: {'id': node.id, 'properties': node.properties}
                      for node_id, node in self.nodes.items()},
            'edges': {edge_id: {'id': edge.id,
                                'source_node_id': edge.source_node_id,
                                'target_node_id': edge.target_node_id,
                                'weight': edge.weight,
                                'properties': edge.properties}
                      for edge_id, edge in self.edges.items()},
            'partition_schemas': [vars(ps) for ps in self.partition_schemas]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkGraph':
        graph = cls(graph_id=data.get('id'), properties=data.get('properties'))
        nodes_data = data.get('nodes', {})
        for node_id, node_data in nodes_data.items():
            node = ArchitecturalNode(node_id=node_id, properties=node_data.get('properties'))
            graph.add_node(node)
        edges_data = data.get('edges', {})
        for edge_id, edge_data in edges_data.items():
            edge = ArchitecturalEdge(source_node_id=edge_data['source_node_id'],
                                     target_node_id=edge_data['target_node_id'],
                                     edge_id=edge_id,
                                     weight=edge_data.get('weight', 0.0),
                                     properties=edge_data.get('properties'))
            graph.add_edge(edge)
        ps_data_list = data.get('partition_schemas', [])
        for ps_data in ps_data_list:
            ps = PartitionSchema(name=ps_data['name'],
                                 target_element_type=ps_data['target_element_type'],
                                 partitions=ps_data['partitions'],
                                 schema_id=ps_data.get('id'),
                                 metadata=ps_data.get('metadata'))
            graph.add_partition_schema(ps)
        return graph

    def __str__(self) -> str:
        return f"NetGraph(id={self.id}, nodes={len(self.nodes)}, edges={len(self.edges)}, partitions={len(self.partition_schemas)})"
