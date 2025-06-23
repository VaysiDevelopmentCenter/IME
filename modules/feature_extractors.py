# modules/feature_extractors.py
import ast
import statistics # For mean, stdev
# import numpy as np # Could be used for more complex stats if added as dependency

# Assuming NetworkGraph is defined elsewhere, e.g., from .engine
try:
    from .engine import NetworkGraph
except ImportError:
    # Fallback for scenarios where this script might be run directly for testing,
    # or if the package structure isn't perfectly set up during development.
    # This assumes 'engine.py' is in a directory that Python can find.
    from engine import NetworkGraph

def extract_list_features(data: list, max_len_for_padding=10) -> list[float]:
    """
    Extracts features from a list. Assumes list contains primarily numbers for stats.
    Returns a list of 7 features.
    """
    features = []
    if not isinstance(data, list):
        return [0.0] * 7 # Default for unexpected type

    # 0: Length (normalized by an arbitrary max_len_for_padding to keep it somewhat scaled)
    features.append(float(len(data) / max_len_for_padding if max_len_for_padding > 0 else len(data)))

    numeric_data = [x for x in data if isinstance(x, (int, float))]

    if not numeric_data: # Handle empty list or list with no numbers
        features.extend([0.0] * 6) # min, max, mean, std_dev, is_sorted_asc, is_sorted_desc
        return features

    # 1: Min value
    min_val = float(min(numeric_data))
    features.append(min_val)
    # 2: Max value
    max_val = float(max(numeric_data))
    features.append(max_val)
    # 3: Mean value
    mean_val = float(statistics.mean(numeric_data)) if len(numeric_data) > 0 else 0.0
    features.append(mean_val)
    # 4: Standard deviation
    std_dev = float(statistics.stdev(numeric_data)) if len(numeric_data) >= 2 else 0.0
    features.append(std_dev)

    # 5: Is sorted ascending
    is_sorted_asc = 1.0 if all(numeric_data[i] <= numeric_data[i+1] for i in range(len(numeric_data)-1)) else 0.0
    features.append(is_sorted_asc)
    # 6: Is sorted descending
    is_sorted_desc = 1.0 if all(numeric_data[i] >= numeric_data[i+1] for i in range(len(numeric_data)-1)) else 0.0
    features.append(is_sorted_desc)

    return features


class ASTFeatureExtractorVisitor(ast.NodeVisitor):
    def __init__(self):
        # Initialize all keys that will be accessed in get_feature_vector
        # and match the order/names in DEFAULT_FEATURE_CONFIG's feature_names for ast.Module
        self.counts = {
            "FunctionDef": 0, "Assign": 0, "Constant": 0,
            "Name_Load": 0, "Name_Store": 0,
            "If": 0, "For": 0,
            # Ensure all keys used by get_feature_vector are here.
            # The DEFAULT_FEATURE_CONFIG for "Module" lists 8 features:
            # "num_FunctionDef", "num_Assign", "num_Constant", "num_Name_Load",
            # "num_Name_Store", "max_depth", "num_If", "num_For"
            # So, the keys above are sufficient for these specific features.
            # Other general counts like "While", "Call", "Return" can be kept if desired for other uses,
            # but won't be part of the fixed 8-feature vector below.
            "While": 0, "Call": 0, "Return":0
        }
        self.depths = [0] # Initialize with 0 to handle empty/unvisited ASTs for max()
        self._current_depth = 0

    def visit(self, node): # This is the entry point for NodeVisitor
        self._current_depth += 1
        self.depths.append(self._current_depth)

        node_type_name = type(node).__name__
        if node_type_name in self.counts: # Only increment if key exists
            self.counts[node_type_name] += 1

        if isinstance(node, ast.Name): # Specific handling for Name nodes
            if isinstance(node.ctx, ast.Load):
                self.counts["Name_Load"] += 1 # Assumes "Name_Load" is in self.counts
            elif isinstance(node.ctx, ast.Store):
                self.counts["Name_Store"] += 1 # Assumes "Name_Store" is in self.counts

        super().generic_visit(node) # Use generic_visit to continue to children
        self._current_depth -= 1

    def get_feature_vector(self) -> list[float]:
        max_depth = float(max(self.depths)) # depths is guaranteed to have at least one element (0)
        # Access counts safely using .get(key, 0) and in the order defined by
        # DEFAULT_FEATURE_CONFIG["entity_type_dispatch"]["Module"]["feature_names"]
        return [
            float(self.counts.get("FunctionDef", 0)),
            float(self.counts.get("Assign", 0)),
            float(self.counts.get("Constant", 0)),
            float(self.counts.get("Name_Load", 0)),
            float(self.counts.get("Name_Store", 0)),
            max_depth, # This is the 6th feature
            float(self.counts.get("If", 0)),
            float(self.counts.get("For", 0))
            # This vector has 8 features.
        ]

def extract_ast_module_features(data: ast.AST) -> list[float]:
    """Extracts features from a Python AST (ast.Module). Returns 8 features as defined."""
    # Expected feature length for ast.Module is 8 as per DEFAULT_FEATURE_CONFIG
    expected_len = DEFAULT_FEATURE_CONFIG["entity_type_dispatch"]["Module"]["expected_feature_length"]

    if not isinstance(data, ast.AST):
        return [0.0] * expected_len

    extractor = ASTFeatureExtractorVisitor()
    extractor.visit(data)
    features = extractor.get_feature_vector()

    # Ensure final feature vector has the expected length
    if len(features) < expected_len:
        features.extend([0.0] * (expected_len - len(features)))
    elif len(features) > expected_len:
        features = features[:expected_len]

    return features


def extract_network_graph_features(data: NetworkGraph) -> list[float]:
    """Extracts features from an assembly NetworkGraph. Returns 8 features."""
    features = []
    if not isinstance(data, NetworkGraph):
        return [0.0] * 8

    num_nodes = len(data.nodes)
    num_edges = len(data.edges)
    features.append(float(num_nodes))
    features.append(float(num_edges))

    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0.0
    features.append(avg_degree)

    num_instr_nodes = sum(1 for n in data.nodes.values() if n.node_type == "instruction")
    features.append(float(num_instr_nodes))
    num_label_nodes = sum(1 for n in data.nodes.values() if n.node_type == "label")
    features.append(float(num_label_nodes))

    num_ctrl_flow_instr = sum(1 for n in data.nodes.values()
                              if n.node_type == "instruction" and n.properties.get("is_control_flow"))
    features.append(float(num_ctrl_flow_instr))

    num_jump_edges = sum(1 for e in data.edges.values()
                         if "jump" in e.properties.get("flow_type", ""))
    features.append(float(num_jump_edges))

    num_partition_schemas = len(data.partition_schemas)
    features.append(float(num_partition_schemas))

    return features


DEFAULT_FEATURE_CONFIG = {
    "max_vector_size": 10, # Ensure this is >= max expected_feature_length from any extractor
    "entity_type_dispatch": {
        "list": { # type(entity.data) is list
            "extractor_function_name": "extract_list_features",
            "feature_names": ["length_norm", "min_val", "max_val", "mean_val", "std_dev", "is_sorted_asc", "is_sorted_desc"],
            "expected_feature_length": 7
        },
        "Module": { # type(entity.data) is ast.Module
            "extractor_function_name": "extract_ast_module_features",
            "feature_names": ["num_FunctionDef", "num_Assign", "num_Constant", "num_Name_Load", "num_Name_Store", "max_depth", "num_If", "num_For"],
            "expected_feature_length": 8
        },
        "NetworkGraph": { # type(entity.data) is NetworkGraph
            "extractor_function_name": "extract_network_graph_features",
            "feature_names": ["num_nodes", "num_edges", "avg_node_degree", "num_instr_nodes", "num_label_nodes", "num_ctrl_flow_instr", "num_jump_edges", "num_partition_schemas"],
            "expected_feature_length": 8
        }
    },
    "extractor_functions_map": {
        "extract_list_features": extract_list_features,
        "extract_ast_module_features": extract_ast_module_features,
        "extract_network_graph_features": extract_network_graph_features,
    }
}
