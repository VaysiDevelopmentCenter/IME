# modules/feature_extractors.py
import ast
import statistics
from typing import List, Any, Dict, Callable # For type hints

# Forward declaration for NetworkGraph if it's used by an extractor and defined in engine
# For this phase, we are deferring NetworkGraph features, so this is less critical now.
# class NetworkGraph: pass

# --- Feature Extractors ---

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
        }
        # Deferred for this restart:
        # "NetworkGraph": {
        #     "extractor_function_name": "extract_network_graph_features",
        #     "feature_names": ["num_nodes", "num_edges", "avg_degree", "num_instr_nodes", ...],
        #     "expected_feature_length": 8 # Or whatever it becomes
        # }
    },
    "extractor_functions_map": {
        "extract_list_features": extract_list_features,
        "extract_ast_module_features": extract_ast_module_features,
        # "extract_network_graph_features": extract_network_graph_features, # Deferred
    }
}

if __name__ == '__main__':
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

    print("\n--- Feature Extractors Test Complete ---")
