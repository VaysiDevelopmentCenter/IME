# modules/feature_extractors.py
import ast
import statistics
from typing import List, Any, Dict, Callable, Optional # For type hints

# Forward declaration for NetworkGraph if it's used by an extractor and defined in engine
from typing import TYPE_CHECKING, Union # Added Union for PyGData type hint
if TYPE_CHECKING:
    from modules.graph_schema import NetworkGraph # Changed import source
    from torch_geometric.data import Data as PyGData # For type hinting

try:
    import torch
    from torch_geometric.data import Data as PyGData_import
    PYG_AVAILABLE = True
except ImportError:
    torch = None # type: ignore
    PyGData_import = None # type: ignore
    PYG_AVAILABLE = False
    if TYPE_CHECKING: # Make type checker happy with a stub
        class PyGData: # type: ignore
            def __init__(self, x=None, edge_index=None, **kwargs): self.x=x; self.edge_index=edge_index; [setattr(self,k,v) for k,v in kwargs.items()]
    else: # Runtime stub
        PyGData = type(None)


# --- Feature Extractors ---
# List of known layer types for one-hot encoding in GNNFeatureExtractor
# The last type 'Other' will serve as a fallback.
KNOWN_LAYER_TYPES: List[str] = [
    'Input', 'Output', # Special conceptual layers
    'Linear', 'Conv1d', 'Conv2d', 'Conv3d',
    'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'LeakyReLU', 'ELU', 'SELU',
    'Attention', 'MultiheadAttention',
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d',
    'Dropout',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'AdaptiveAvgPool2d',
    'Embedding', 'LSTM', 'GRU',
    'Flatten', 'Unflatten',
    'Other'
]

# Global variable to store the calculated GNN node feature dimension.
# This will be set by GNNFeatureExtractor after its first feature calculation.
# A more dynamic approach might be needed if features change, but this is for initialization.
GNN_NODE_FEATURE_DIM: Optional[int] = None


class GNNFeatureExtractor:
    def __init__(self, known_layer_types: Optional[List[str]] = None,
                 max_out_features_scale: float = 1024.0, # For normalization
                 default_numerical_features: int = 3 # out_features, in_features, num_heads (example)
                 ):
        self.known_layer_types = known_layer_types if known_layer_types is not None else KNOWN_LAYER_TYPES
        self.layer_type_to_idx = {name: i for i, name in enumerate(self.known_layer_types)}
        self.num_layer_types = len(self.known_layer_types)
        self.max_out_features_scale = max_out_features_scale

        # Calculate feature dimension: one-hot for layer_type + numerical features
        # Example numerical features: normalized out_features, normalized in_features, normalized num_heads
        self.node_feature_dim = self.num_layer_types + default_numerical_features

        global GNN_NODE_FEATURE_DIM
        if GNN_NODE_FEATURE_DIM is None:
            GNN_NODE_FEATURE_DIM = self.node_feature_dim
        elif GNN_NODE_FEATURE_DIM != self.node_feature_dim:
            print(f"Warning: GNN_NODE_FEATURE_DIM changing from {GNN_NODE_FEATURE_DIM} to {self.node_feature_dim}")
            GNN_NODE_FEATURE_DIM = self.node_feature_dim


    def extract_pyg_data(self, network_graph: 'NetworkGraph') -> 'Optional[PyGData]':
        if not PYG_AVAILABLE or torch is None or PyGData_import is None:
            print("Warning: PyTorch Geometric or PyTorch not available. Cannot extract PyGData.")
            return None

        node_to_idx = {node_id: i for i, node_id in enumerate(network_graph.nodes.keys())}
        num_nodes = len(network_graph.nodes)

        if num_nodes == 0: # Handle empty graph
             return PyGData_import(x=torch.empty(0, self.node_feature_dim, dtype=torch.float32),
                                   edge_index=torch.empty(2,0, dtype=torch.long))


        node_features_list = []

        for node_id, node in network_graph.nodes.items():
            # 1. One-hot encode layer_type
            one_hot_layer_type = [0.0] * self.num_layer_types
            layer_type_idx = self.layer_type_to_idx.get(node.properties.get('layer_type', 'Other'),
                                                        self.layer_type_to_idx['Other'])
            one_hot_layer_type[layer_type_idx] = 1.0

            # 2. Numerical features (example: out_features, in_features, num_heads)
            # Normalize or scale these. Handle if None.
            out_features = node.properties.get('out_features')
            norm_out_features = float(out_features / self.max_out_features_scale) if isinstance(out_features, int) else 0.0

            in_features = node.properties.get('in_features')
            norm_in_features = float(in_features / self.max_out_features_scale) if isinstance(in_features, int) else 0.0

            num_heads = node.properties.get('num_heads') # Common in Attention layers
            norm_num_heads = float(num_heads / 64.0) if isinstance(num_heads, int) else 0.0 # Assuming max 64 heads for scaling

            # Concatenate features
            # Current order: [one_hot_layer_types, norm_out_features, norm_in_features, norm_num_heads]
            # Ensure this order matches the self.node_feature_dim calculation.
            current_node_features = one_hot_layer_type + [norm_out_features, norm_in_features, norm_num_heads]
            node_features_list.append(current_node_features)

        node_features_tensor = torch.tensor(node_features_list, dtype=torch.float32)

        # Create edge_index
        source_nodes: List[int] = []
        target_nodes: List[int] = []
        for edge in network_graph.edges.values():
            if edge.source_node_id in node_to_idx and edge.target_node_id in node_to_idx:
                source_nodes.append(node_to_idx[edge.source_node_id])
                target_nodes.append(node_to_idx[edge.target_node_id])

        edge_index_tensor = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        return PyGData_import(x=node_features_tensor, edge_index=edge_index_tensor)


# Helper function to be mapped in DEFAULT_FEATURE_CONFIG
# This allows SmartMutationEngine to call it without needing to instantiate GNNFeatureExtractor itself
# or if we want a singleton GNNFeatureExtractor.
_gnn_feature_extractor_instance: Optional[GNNFeatureExtractor] = None

def extract_network_graph_pyg_data(network_graph: 'NetworkGraph') -> 'Optional[PyGData]':
    global _gnn_feature_extractor_instance
    if _gnn_feature_extractor_instance is None:
        _gnn_feature_extractor_instance = GNNFeatureExtractor() # Use default KNOWN_LAYER_TYPES

    # In SmartMutationEngine, the feature is expected as List[float] for FFN,
    # and PyGData for GNN. The _get_rl_state_representation handles this.
    # For now, this function will return PyGData directly.
    # The part of SmartMutationEngine that uses this for GNN will expect PyGData.
    # If this were to be used for an FFN representation of a graph, we'd need to flatten PyGData.
    return _gnn_feature_extractor_instance.extract_pyg_data(network_graph)

def extract_list_features(data: List[Any], max_len_for_padding: int = 10) -> List[float]:
    """
    Extracts features from a list. Assumes list contains primarily numbers for stats.
    Returns a list of 7 features as defined in DEFAULT_FEATURE_CONFIG.
    """
    # Match the feature_names in DEFAULT_FEATURE_CONFIG["entity_type_dispatch"]["list"]["feature_names"]
    # ["length_norm", "min_val", "max_val", "mean_val", "std_dev", "is_sorted_asc", "is_sorted_desc"]

    features = [0.0] * 7 # Initialize with default values

    if not isinstance(data, list):
        return features # Return default if not a list

    # Feature 0: Normalized Length
    features[0] = float(len(data) / max_len_for_padding if max_len_for_padding > 0 else len(data))

    numeric_data = [x for x in data if isinstance(x, (int, float))]

    if not numeric_data:
        # For features 1-6, if no numeric data, they remain 0.0 (except sorted flags which are true for empty/single)
        if len(data) <= 1: # Empty or single non-numeric item list is "sorted"
            features[5] = 1.0 # is_sorted_asc
            features[6] = 1.0 # is_sorted_desc
        return features

    # Feature 1: Min value
    features[1] = float(min(numeric_data))
    # Feature 2: Max value
    features[2] = float(max(numeric_data))
    # Feature 3: Mean value
    features[3] = float(statistics.mean(numeric_data)) if len(numeric_data) > 0 else 0.0
    # Feature 4: Standard deviation
    features[4] = float(statistics.stdev(numeric_data)) if len(numeric_data) >= 2 else 0.0

    # Feature 5: Is sorted ascending
    is_sorted_asc = all(numeric_data[i] <= numeric_data[i+1] for i in range(len(numeric_data)-1))
    features[5] = 1.0 if is_sorted_asc else 0.0
    # Feature 6: Is sorted descending
    is_sorted_desc = all(numeric_data[i] >= numeric_data[i+1] for i in range(len(numeric_data)-1))
    features[6] = 1.0 if is_sorted_desc else 0.0

    return features


class ASTFeatureExtractorVisitor(ast.NodeVisitor):
    def __init__(self):
        self.counts: Dict[str, int] = {
            "FunctionDef": 0, "Assign": 0, "Constant": 0,
            "Name_Load": 0, "Name_Store": 0,
            "If": 0, "For": 0,
            # Add other node types that might be counted if the feature vector evolves
            "While": 0, "Call": 0, "Return":0
        }
        self.depths: List[int] = [0]
        self._current_depth: int = 0

    def visit(self, node: ast.AST) -> None:
        self._current_depth += 1
        self.depths.append(self._current_depth)

        node_type_name = type(node).__name__
        if node_type_name in self.counts:
            self.counts[node_type_name] += 1

        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load): self.counts["Name_Load"] += 1
            elif isinstance(node.ctx, ast.Store): self.counts["Name_Store"] += 1

        super().generic_visit(node)
        self._current_depth -= 1

    def get_feature_vector(self) -> List[float]:
        max_depth = float(max(self.depths))
        # Order must match feature_names in DEFAULT_FEATURE_CONFIG for "Module"
        return [
            float(self.counts.get("FunctionDef", 0)),
            float(self.counts.get("Assign", 0)),
            float(self.counts.get("Constant", 0)),
            float(self.counts.get("Name_Load", 0)),
            float(self.counts.get("Name_Store", 0)),
            max_depth,
            float(self.counts.get("If", 0)),
            float(self.counts.get("For", 0))
        ]

def extract_ast_module_features(data: ast.AST) -> List[float]:
    """Extracts features from a Python AST (ast.Module)."""
    # Expected feature length for ast.Module as per DEFAULT_FEATURE_CONFIG
    expected_len = 8 # Default, but will be overridden by config if available below

    if "DEFAULT_FEATURE_CONFIG" in globals(): # Check if config is loaded
        cfg_entry = DEFAULT_FEATURE_CONFIG["entity_type_dispatch"].get("Module")
        if cfg_entry and "expected_feature_length" in cfg_entry:
            expected_len = cfg_entry["expected_feature_length"]

    if not isinstance(data, ast.AST):
        return [0.0] * expected_len

    extractor = ASTFeatureExtractorVisitor()
    extractor.visit(data)
    features = extractor.get_feature_vector()

    # Pad or truncate to expected_len
    if len(features) < expected_len:
        features.extend([0.0] * (expected_len - len(features)))
    elif len(features) > expected_len:
        features = features[:expected_len]

    return features

# --- Default Feature Configuration ---
# This defines how features are extracted for different entity types.
# It will be used by SmartMutationEngine.

DEFAULT_FEATURE_CONFIG: Dict[str, Any] = {
    "max_vector_size": 10, # NN input size after padding/truncation
    "entity_type_dispatch": {
        "list": {
            "extractor_function_name": "extract_list_features",
            "feature_names": ["length_norm", "min_val", "max_val", "mean_val", "std_dev", "is_sorted_asc", "is_sorted_desc"],
            "expected_feature_length": 7 # Actual number of features returned by the extractor
        },
        "Module": { # For ast.Module objects
            "extractor_function_name": "extract_ast_module_features",
            "feature_names": ["num_FunctionDef", "num_Assign", "num_Constant", "num_Name_Load", "num_Name_Store", "max_depth", "num_If", "num_For"],
            "expected_feature_length": 8
        },
        "NetworkGraph": {
            # This function returns PyGData, not a flat list.
            # SmartMutationEngine's _get_rl_state_representation will handle this.
            "extractor_function_name": "extract_network_graph_pyg_data",
            # feature_names and expected_feature_length are less relevant here
            # as the output is graph structured. For FFN use, one might flatten this.
            "feature_names": ["pyg_data_object"],
            "expected_feature_length": 1 # Placeholder, actual features are in PyGData.x
        }
    },
    "extractor_functions_map": {
        "extract_list_features": extract_list_features,
        "extract_ast_module_features": extract_ast_module_features,
        "extract_network_graph_pyg_data": extract_network_graph_pyg_data,
    }
}

if __name__ == '__main__':
    import sys
    import os
    # Add repository root to sys.path to allow direct execution of this script
    # and importing other modules from the 'modules' package.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    print("--- Feature Extractors Direct Test ---")

    # Test list features
    list_data1 = [1, 2, 3, 4, 5] # Sorted
    list_data2 = [5, 1, 4, 2, 3] # Unsorted
    list_data3 = []              # Empty
    list_data4 = [10]            # Single
    list_data5 = [10, "a", 20]   # Mixed
    print(f"List {list_data1} -> Features: {extract_list_features(list_data1)}")
    print(f"List {list_data2} -> Features: {extract_list_features(list_data2)}")
    print(f"List {list_data3} -> Features: {extract_list_features(list_data3)}")
    print(f"List {list_data4} -> Features: {extract_list_features(list_data4)}")
    print(f"List {list_data5} -> Features: {extract_list_features(list_data5)}")

    # Test AST features
    test_code = """
def foo(a, b):
    c = a + b
    if c > 10:
        print(c)
    for i in range(c):
        pass
    return c
x = 1
foo(x, 20)
    """
    ast_data = ast.parse(test_code)
    print(f"\nAST for test_code -> Features: {extract_ast_module_features(ast_data)}")

    empty_ast = ast.parse("")
    print(f"Empty AST -> Features: {extract_ast_module_features(empty_ast)}")

    # Test GNNFeatureExtractor
    print("\n--- Testing GNNFeatureExtractor ---")
    if PYG_AVAILABLE:
        try:
            from modules.engine import NetworkGraph # Import for test

            # Create a sample NetworkGraph
            graph = NetworkGraph(graph_id="test_graph_for_extractor")
            graph.add_layer_node("input1", layer_type="Input", node_attributes={'out_features': 64})
            graph.add_layer_node("linear1", layer_type="Linear", node_attributes={'in_features': 64, 'out_features': 32, 'activation_function': 'ReLU'})
            graph.add_layer_node("output1", layer_type="Output", node_attributes={'in_features': 32, 'out_features': 10})
            graph.connect_layers("input1", "linear1")
            graph.connect_layers("linear1", "output1")

            print(f"Created NetworkGraph: {graph}")
            for node_id, node_obj in graph.nodes.items():
                print(f"  {node_obj}")
            for edge_id, edge_obj in graph.edges.items():
                print(f"  {edge_obj}")

            # 1. Test direct instantiation and use
            print("\n1. Testing GNNFeatureExtractor instance:")
            extractor_instance = GNNFeatureExtractor()
            pyg_data_instance = extractor_instance.extract_pyg_data(graph)
            if pyg_data_instance:
                print(f"  Extracted PyGData (instance): {pyg_data_instance}")
                print(f"    Node features (x) shape: {pyg_data_instance.x.shape}")
                print(f"    Edge index (edge_index) shape: {pyg_data_instance.edge_index.shape}")
                print(f"    GNN_NODE_FEATURE_DIM set to: {GNN_NODE_FEATURE_DIM}")
                assert GNN_NODE_FEATURE_DIM == pyg_data_instance.x.shape[1]
            else:
                print("  Failed to extract PyGData using instance.")

            # 2. Test helper function (uses singleton)
            print("\n2. Testing extract_network_graph_pyg_data helper:")
            pyg_data_helper = extract_network_graph_pyg_data(graph)
            if pyg_data_helper:
                print(f"  Extracted PyGData (helper): {pyg_data_helper}")
                print(f"    Node features (x) shape: {pyg_data_helper.x.shape}")
                print(f"    Edge index (edge_index) shape: {pyg_data_helper.edge_index.shape}")
                assert GNN_NODE_FEATURE_DIM == pyg_data_helper.x.shape[1] # Should be same dim
            else:
                print("  Failed to extract PyGData using helper.")

            # Test with an empty graph
            print("\n3. Testing with empty graph:")
            empty_graph = NetworkGraph(graph_id="empty_graph")
            pyg_empty_data = extract_network_graph_pyg_data(empty_graph)
            if pyg_empty_data:
                print(f"  Extracted PyGData (empty): {pyg_empty_data}")
                print(f"    Node features (x) shape: {pyg_empty_data.x.shape}")
                print(f"    Edge index (edge_index) shape: {pyg_empty_data.edge_index.shape}")
                assert pyg_empty_data.x.shape[0] == 0
                assert pyg_empty_data.x.shape[1] == GNN_NODE_FEATURE_DIM if GNN_NODE_FEATURE_DIM is not None else True # Dimension should match
                assert pyg_empty_data.edge_index.shape[1] == 0
            else:
                print("  Failed to extract PyGData for empty graph (or PyG not available).")

        except ImportError as e:
            print(f"Could not import NetworkGraph for GNN test: {e}")
        except Exception as e:
            print(f"An error occurred during GNNFeatureExtractor test: {e}")
    else:
        print("PyTorch Geometric not available, skipping GNNFeatureExtractor tests.")


    print("\n--- Feature Extractors Test Complete ---")
