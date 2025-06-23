# modules/reprogrammable_selector_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random

try:
    from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool # type: ignore
    from torch_geometric.data import Data as PyGData # type: ignore
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    # Define syntactically valid dummy classes if PyG is not available
    class GCNConv(nn.Module): # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()
            # print("Warning: GCNConv stub used, PyTorch Geometric not available.")
            raise NotImplementedError("GCNConv stub: PyTorch Geometric not available.")

    def global_mean_pool(x, batch): # type: ignore
        # print("Warning: global_mean_pool stub used, PyTorch Geometric not available.")
        raise NotImplementedError("global_mean_pool stub: PyTorch Geometric not available.")

    def global_add_pool(x, batch): # type: ignore
        # print("Warning: global_add_pool stub used, PyTorch Geometric not available.")
        raise NotImplementedError("global_add_pool stub: PyTorch Geometric not available.")

    def global_max_pool(x, batch): # type: ignore
        # print("Warning: global_max_pool stub used, PyTorch Geometric not available.")
        raise NotImplementedError("global_max_pool stub: PyTorch Geometric not available.")

    class PyGData: # type: ignore
        def __init__(self, x=None, edge_index=None, batch=None, **kwargs):
            self.x = x
            self.edge_index = edge_index
            self.batch = batch
            # print("Warning: PyGData stub used, PyTorch Geometric not available.")
            for key, item in kwargs.items():
                setattr(self, key, item)


# Assuming NetworkGraph is defined in .engine
try:
    from .engine import NetworkGraph
    from .gnn_utils import network_graph_to_pyg_data, NODE_FEATURE_DIM as GNN_NODE_FEATURE_DIM
except ImportError:
    NetworkGraph = type(None) # type: ignore
    def network_graph_to_pyg_data_dummy(graph_unused): # type: ignore
        # print("Warning: Using dummy network_graph_to_pyg_data converter.")
        return None
    network_graph_to_pyg_data = network_graph_to_pyg_data_dummy # type: ignore
    GNN_NODE_FEATURE_DIM = 1 # Dummy value, will cause issues if GNN is attempted without PyG
    # print("Warning: ReprogrammableSelectorNN using fallback imports for engine/gnn_utils components.")


class ReprogrammableSelectorNN(nn.Module):
    def __init__(self, nn_config: dict):
        super().__init__()
        if not nn_config or not isinstance(nn_config, dict):
            raise ValueError("nn_config dictionary must be provided.")

        self.nn_config = copy.deepcopy(nn_config)
        self.model_type = self.nn_config.get("model_type", "ffn").lower()

        if self.model_type == "gnn" and not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN model_type but not found.")

        if self.model_type == "gnn":
            if "node_feature_size" not in self.nn_config:
                if GNN_NODE_FEATURE_DIM is None or (GNN_NODE_FEATURE_DIM == 1 and NetworkGraph is type(None)):
                     raise ValueError("GNN_NODE_FEATURE_DIM from gnn_utils seems unavailable/dummy, and 'node_feature_size' not in nn_config for GNN.")
                self.nn_config["node_feature_size"] = GNN_NODE_FEATURE_DIM

        self.model_internal = self._build_model()
        self.optimizer = None
        self.criterion = None

    def _get_activation_function(self, activation_name: str | None) -> nn.Module | None:
        if activation_name is None or activation_name.lower() == "none": return None
        name_lower = activation_name.lower()
        if name_lower == "relu": return nn.ReLU()
        elif name_lower == "sigmoid": return nn.Sigmoid()
        elif name_lower == "tanh": return nn.Tanh()
        elif name_lower == "softmax": return nn.Softmax(dim=-1)
        else: raise ValueError(f"Unsupported activation function: {activation_name}")

    def _build_ffn_model(self) -> nn.Sequential:
        layers_config = self.nn_config.get("layers", [])
        input_size = self.nn_config.get("input_size")
        output_size = self.nn_config.get("output_size")
        output_activation_name = self.nn_config.get("output_activation", "none")

        if input_size is None or output_size is None:
            raise ValueError("FFN nn_config must specify 'input_size' and 'output_size'.")

        model_layers = []
        current_size = input_size
        for i, layer_conf in enumerate(layers_config):
            layer_type = layer_conf.get("type", "").lower()
            if layer_type == "linear":
                out_features = layer_conf.get("size")
                if out_features is None or not isinstance(out_features, int) or out_features <=0:
                    raise ValueError(f"Linear layer config at index {i} must specify a positive integer 'size'.")
                model_layers.append(nn.Linear(current_size, out_features))
                current_size = out_features
                activation_fn = self._get_activation_function(layer_conf.get("activation"))
                if activation_fn: model_layers.append(activation_fn)
            elif layer_type == "dropout":
                rate = layer_conf.get("rate", 0.5)
                if not (isinstance(rate, float) and 0.0 <= rate < 1.0):
                    raise ValueError(f"Dropout layer config at index {i} must have a 'rate' float [0,1).")
                model_layers.append(nn.Dropout(rate))
            else: raise ValueError(f"Unsupported FFN layer type: {layer_type}")

        model_layers.append(nn.Linear(current_size, output_size))
        output_activation_fn = self._get_activation_function(output_activation_name)
        if output_activation_fn: model_layers.append(output_activation_fn)
        return nn.Sequential(*model_layers)

    def _build_gnn_model(self) -> nn.Module:
        if not PYG_AVAILABLE: raise ImportError("PyTorch Geometric required for GNN model type.")

        node_feature_size = self.nn_config.get("node_feature_size")
        gnn_layers_config = self.nn_config.get("gnn_layers", [])
        global_pooling_method_name = self.nn_config.get("global_pooling", "mean").lower()
        post_gnn_mlp_config = self.nn_config.get("post_gnn_mlp_layers", [])
        output_size = self.nn_config.get("output_size")
        output_activation_name = self.nn_config.get("output_activation", "none")

        if node_feature_size is None or output_size is None:
            raise ValueError("GNN nn_config must specify 'node_feature_size' and 'output_size'.")

        class GNNSubModel(nn.Module):
            def __init__(self_sub, parent_nn_instance):
                super().__init__()
                self_sub.parent_nn = parent_nn_instance
                self_sub.gnn_conv_layers = nn.ModuleList()
                current_channels = node_feature_size
                for i, layer_conf in enumerate(gnn_layers_config):
                    layer_type = layer_conf.get("type", "").lower()
                    out_channels = layer_conf.get("out_channels")
                    if out_channels is None: raise ValueError(f"GNN layer {i} must specify 'out_channels'.")

                    if layer_type == "gcnconv":
                        self_sub.gnn_conv_layers.append(GCNConv(current_channels, out_channels))
                    else: raise ValueError(f"Unsupported GNN layer type: {layer_type}")
                    current_channels = out_channels

                    activation_fn = self_sub.parent_nn._get_activation_function(layer_conf.get("activation"))
                    if activation_fn: self_sub.gnn_conv_layers.append(activation_fn)

                if global_pooling_method_name == "mean": self_sub.pooling_layer = global_mean_pool
                elif global_pooling_method_name == "add": self_sub.pooling_layer = global_add_pool
                elif global_pooling_method_name == "max": self_sub.pooling_layer = global_max_pool
                else: raise ValueError(f"Unsupported pooling: {global_pooling_method_name}")

                self_sub.post_mlp = nn.ModuleList()
                mlp_current_size = current_channels
                for i, layer_conf in enumerate(post_gnn_mlp_config):
                    layer_type = layer_conf.get("type", "").lower()
                    if layer_type == "linear":
                        out_features = layer_conf.get("size")
                        if out_features is None: raise ValueError(f"Post-GNN MLP Linear layer {i} needs 'size'.")
                        self_sub.post_mlp.append(nn.Linear(mlp_current_size, out_features))
                        mlp_current_size = out_features
                        activation_fn = self_sub.parent_nn._get_activation_function(layer_conf.get("activation"))
                        if activation_fn: self_sub.post_mlp.append(activation_fn)
                    elif layer_type == "dropout":
                         self_sub.post_mlp.append(nn.Dropout(layer_conf.get("rate", 0.5)))
                    else: raise ValueError(f"Unsupported Post-GNN MLP layer type: {layer_type}")

                self_sub.output_layer = nn.Linear(mlp_current_size, output_size)
                self_sub.output_activation = self_sub.parent_nn._get_activation_function(output_activation_name)

            def forward(self_sub, data: PyGData) -> torch.Tensor:
                x, edge_index = data.x, data.edge_index
                batch = getattr(data, 'batch', None)

                if x.dtype != torch.float32: x = x.to(torch.float32)

                for layer in self_sub.gnn_conv_layers:
                    if isinstance(layer, GCNConv):
                        x = layer(x, edge_index)
                    else: x = layer(x)

                if x.numel() == 0:
                    num_output_channels_gnn = node_feature_size # Default if no conv layers
                    if self_sub.gnn_conv_layers:
                        last_conv_layer = None
                        for l_idx in range(len(self_sub.gnn_conv_layers) -1, -1, -1):
                            if hasattr(self_sub.gnn_conv_layers[l_idx], 'out_channels'):
                                last_conv_layer = self_sub.gnn_conv_layers[l_idx]
                                break
                        if last_conv_layer: num_output_channels_gnn = last_conv_layer.out_channels

                    num_graphs = int(batch.max().item() + 1) if batch is not None and batch.numel() > 0 else 1
                    x = torch.zeros((num_graphs, num_output_channels_gnn), device=x.device)
                else:
                    if batch is None :
                        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                    x = self_sub.pooling_layer(x, batch)

                for layer in self_sub.post_mlp: x = layer(x)
                x = self_sub.output_layer(x)
                if self_sub.output_activation: x = self_sub.output_activation(x)
                return x
        return GNNSubModel(self)

    def _build_model(self) -> nn.Module:
        if self.model_type == "gnn": return self._build_gnn_model()
        elif self.model_type == "ffn": return self._build_ffn_model()
        else: raise ValueError(f"Unknown model_type: {self.model_type}")

    def forward(self, input_data: torch.Tensor | PyGData) -> torch.Tensor:
        if self.model_type == "gnn":
            if not isinstance(input_data, PyGData):
                raise TypeError(f"GNN model expects PyGData input, got {type(input_data)}")
            return self.model_internal(input_data)
        else:
            if not isinstance(input_data, torch.Tensor):
                raise TypeError(f"FFN model expects Tensor input, got {type(input_data)}")
            features = input_data.to(torch.float32)
            expected = self.nn_config.get("input_size")
            if features.shape[-1] != expected:
                 raise ValueError(f"Input feature size ({features.shape[-1]}) != FFN expected ({expected}).")
            return self.model_internal(features)

    def predict(self, features_or_graph_data: list | torch.Tensor | NetworkGraph | PyGData) -> torch.Tensor: # type: ignore
        self.model_internal.eval()
        with torch.no_grad():
            input_for_nn: torch.Tensor | PyGData
            if self.model_type == "gnn":
                if isinstance(features_or_graph_data, PyGData): input_for_nn = features_or_graph_data
                elif NetworkGraph is not type(None) and isinstance(features_or_graph_data, NetworkGraph):
                    if network_graph_to_pyg_data is not None:
                        converted_data = network_graph_to_pyg_data(features_or_graph_data)
                        if converted_data is None: raise ValueError("NetworkGraph to PyGData conversion failed.")
                        input_for_nn = converted_data
                    else: raise RuntimeError("network_graph_to_pyg_data not available.")
                else: raise TypeError(f"GNN predict expects PyGData or NetworkGraph, got {type(features_or_graph_data)}")
            else: # FFN
                if isinstance(features_or_graph_data, torch.Tensor): input_for_nn = features_or_graph_data
                elif isinstance(features_or_graph_data, list): input_for_nn = torch.tensor(features_or_graph_data, dtype=torch.float32)
                else: raise TypeError(f"FFN predict expects list or Tensor, got {type(features_or_graph_data)}")
                if input_for_nn.ndim == 1: input_for_nn = input_for_nn.unsqueeze(0)

            return self.forward(input_for_nn)

    def train_step(self, state_representation: torch.Tensor | PyGData,
                 action_index: int,
                 reward: float,
                 next_state_representation: torch.Tensor | PyGData | None,
                 done_flag: bool,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99):
        """
        (Conceptual RL Training Step)
        Performs a single conceptual update step based on an experience tuple.
        This is a placeholder for a real RL algorithm (e.g., DQN, Policy Gradient).
        Actual backpropagation and optimizer steps are commented out.
        """
        # print(f"NN train_step called for {self.model_type.upper()} model:")
        # print(f"  Action: {action_index}, Reward: {reward}, Done: {done_flag}")
        # print(f"  State type: {type(state_representation)}, Next state type: {type(next_state_representation)}")

        if not hasattr(self, 'optimizer') or self.optimizer is None:
            # Initialize optimizer and criterion on the first training call if not done before
            # This is a simplistic approach; typically done once after model creation/loading.
            self.optimizer = optim.Adam(self.model_internal.parameters(), lr=learning_rate)
            # Assuming classification-like task for operator selection.
            # NN output should be logits if using CrossEntropyLoss.
            # If output_activation is softmax, use NLLLoss after LogSoftmax, or adjust.
            # For Q-learning, typically use MSELoss between Q_predicted and Q_target.
            self.criterion = nn.MSELoss() # Example for Q-learning style
            # print("  Optimizer and Criterion (MSELoss) initialized for conceptual training.")


        self.model_internal.train() # Set model to training mode
        self.optimizer.zero_grad()

        # --- Conceptual Q-Learning Update ---
        # 1. Get Q_predicted for the (state, action)
        # Ensure state_representation is correctly batched if needed by forward
        q_values_current_state = self.forward(state_representation) # Model output for current state

        # If model output is [batch_size, num_actions], and we have a single instance:
        if q_values_current_state.ndim > 1 and q_values_current_state.shape[0] == 1:
            q_values_current_state = q_values_current_state.squeeze(0) # Remove batch dim: [num_actions]

        q_predicted = q_values_current_state[action_index]

        # 2. Calculate Q_target
        q_target = torch.tensor(float(reward), dtype=torch.float32, device=q_predicted.device) # Ensure same device
        if not done_flag and next_state_representation is not None:
            with torch.no_grad(): # Important for target calculation
                q_values_next_state = self.forward(next_state_representation)
                if q_values_next_state.ndim > 1 and q_values_next_state.shape[0] == 1:
                    q_values_next_state = q_values_next_state.squeeze(0)
                max_q_next = q_values_next_state.max()
                q_target += gamma * max_q_next

        # 3. Calculate loss
        # Ensure q_predicted is a scalar or a tensor that can be used with q_target in loss
        loss = self.criterion(q_predicted, q_target)

        # --- Actual backpropagation (commented out for this conceptual step) ---
        # loss.backward()
        # self.optimizer.step()

        # print(f"  Conceptual Loss (e.g., MSE for Q-learning): {loss.item()}")
        # print(f"    Q_predicted: {q_predicted.item()}, Q_target: {q_target.item()}")

        if not hasattr(self, "_train_step_call_count"): self._train_step_call_count = 0
        self._train_step_call_count += 1

        return loss.item() if hasattr(loss, 'item') else None # Return conceptual loss

    def get_config(self) -> dict:
        return copy.deepcopy(self.nn_config)

    def reconfigure(self, new_nn_config: dict):
        print(f"Reconfiguring ReprogrammableSelectorNN.")
        if not new_nn_config or not isinstance(new_nn_config, dict):
            raise ValueError("Invalid new_nn_config: Must be a dictionary.")

        current_model_type = new_nn_config.get("model_type", self.model_type).lower()
        new_nn_config["model_type"] = current_model_type

        if current_model_type == "ffn":
            if 'input_size' not in new_nn_config or 'output_size' not in new_nn_config:
                raise ValueError("For FFN, new_nn_config must specify 'input_size' and 'output_size'.")
        elif current_model_type == "gnn":
            if 'node_feature_size' not in new_nn_config:
                if GNN_NODE_FEATURE_DIM is None or (GNN_NODE_FEATURE_DIM == 1 and NetworkGraph is type(None)):
                    raise ValueError("GNN_NODE_FEATURE_DIM from gnn_utils seems unavailable/dummy, and 'node_feature_size' not in nn_config for GNN.")
                new_nn_config['node_feature_size'] = GNN_NODE_FEATURE_DIM
            if 'output_size' not in new_nn_config: raise ValueError("For GNN, new_nn_config needs 'output_size'.")
        else: raise ValueError(f"Unsupported model_type in new_nn_config: {current_model_type}")

        self.nn_config = copy.deepcopy(new_nn_config)
        self.model_type = current_model_type
        print(f"  NN config updated. New model_type: {self.model_type}")
        try:
            self.model_internal = self._build_model()
            print("  Model has been rebuilt.")
            self.optimizer = None; self.criterion = None
        except Exception as e: print(f"  Error rebuilding model: {e}.")

# Example usage
if __name__ == '__main__':
    if not PYG_AVAILABLE: print("Warning: PyTorch Geometric not available, GNN tests will be limited or fail.")

    print("--- ReprogrammableSelectorNN (with GNN path) Direct Test ---")
    ffn_conf = {
        "model_type": "ffn", "input_size": 10,
        "layers": [{"type": "linear", "size": 8, "activation": "relu"}],
        "output_size": 3, "output_activation": "softmax"
    }
    print("\nTesting FFN configuration:")
    nn_ffn = ReprogrammableSelectorNN(ffn_conf)
    print(nn_ffn.model_internal)
    ffn_features = torch.randn(1, 10)
    print(f"FFN Prediction: {nn_ffn.predict(ffn_features)}")

    if PYG_AVAILABLE and network_graph_to_pyg_data is not None and NetworkGraph is not type(None) and GNN_NODE_FEATURE_DIM > 1:
        gnn_conf = {
            "model_type": "gnn",
            "node_feature_size": GNN_NODE_FEATURE_DIM,
            "gnn_layers": [{"type": "gcnconv", "out_channels": 16, "activation": "relu"}],
            "global_pooling": "mean",
            "post_gnn_mlp_layers": [{"type": "linear", "size": 8, "activation": "relu"}],
            "output_size": 3,
            "output_activation": "softmax"
        }
        print("\nTesting GNN configuration:")
        nn_gnn = ReprogrammableSelectorNN(gnn_conf)
        print(nn_gnn.model_internal)

        # Create a dummy NetworkGraph for GNN input
        # Ensure ArchitecturalNode and ArchitecturalEdge are available if NetworkGraph is not None
        # This part requires ArchitecturalNode and ArchitecturalEdge to be properly imported or defined
        # For the direct test, if engine components are dummied out, this will fail.
        # Assuming engine components are available for this test block.
        try:
            from modules.engine import ArchitecturalNode, ArchitecturalEdge # Try direct import if needed
        except ImportError:
            pass # Already handled by top-level fallbacks if this is run directly

        if 'ArchitecturalNode' in globals() and 'ArchitecturalEdge' in globals() and \
           ArchitecturalNode is not None and ArchitecturalEdge is not None:
            test_g = NetworkGraph("test_g_for_gnn")
            if hasattr(test_g, 'add_node'):
                # Use types that are in ALL_NODE_TYPES from gnn_utils for meaningful features
                # Need to import ALL_NODE_TYPES for this test block too, or use string literals
                # For simplicity, using string literals known to be in gnn_utils.ALL_NODE_TYPES
                node_type_example1 = "FunctionDef"
                node_type_example2 = "IfStatement"

                test_g.add_node(ArchitecturalNode("n1", node_type_example1))
                test_g.add_node(ArchitecturalNode("n2", node_type_example2))
                if hasattr(test_g, 'add_edge'):
                     test_g.add_edge(ArchitecturalEdge("n1", "n2", "e1"))

                pyg_data_obj = network_graph_to_pyg_data(test_g)
                if pyg_data_obj and hasattr(pyg_data_obj, 'num_nodes') and pyg_data_obj.num_nodes > 0 :
                    print(f"PyGData object: x shape {pyg_data_obj.x.shape}, edge_index shape {pyg_data_obj.edge_index.shape}")
                    print(f"GNN Prediction: {nn_gnn.predict(pyg_data_obj)}")
                elif pyg_data_obj :
                     print(f"PyGData object is empty (num_nodes={getattr(pyg_data_obj, 'num_nodes', -1)}). GNN cannot process. Check converter or graph.")
                else:
                    print("Failed to create PyGData object for GNN test.")
            else:
                print("Skipping GNN data creation as NetworkGraph seems to be a dummy type or missing methods.")
        else:
             print("Skipping GNN data creation due to missing ArchitecturalNode/Edge (likely fallback imports).")
    else:
        print("\nSkipping GNN configuration test as PyTorch Geometric or gnn_utils/engine not fully available or GNN_NODE_FEATURE_DIM is dummy.")
    print("--- Test Complete ---")
