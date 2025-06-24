# modules/reprogrammable_selector_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from typing import List, Union, Optional, Dict, Any # For type hints
from collections import namedtuple # Added import

# For this phase, we are focusing on FFN, so PyG imports are not strictly needed here yet.
# PYG_AVAILABLE = False # Assume not available for this focused FFN implementation.
# class PyGData: pass # Dummy for type hints if needed by shared signatures.

class ReprogrammableSelectorNN(nn.Module):
    """
    A reprogrammable neural network for selecting mutation operators.
    Its architecture is defined by a configuration dictionary.
    Currently implements the Feedforward Network (FFN) path.
    """
    def __init__(self, nn_config: Dict[str, Any], learning_rate: float = 0.001):
        super().__init__()

        if not nn_config or not isinstance(nn_config, dict):
            raise ValueError("nn_config dictionary must be provided.")

        self.nn_config = copy.deepcopy(nn_config)
        # For this simplified Step 4, model_type is assumed/defaults to 'ffn'
        self.model_type = self.nn_config.get("model_type", "ffn").lower()
        if self.model_type != "ffn":
            raise ValueError(f"For this implementation phase, model_type must be 'ffn'. Got '{self.model_type}'.")

        self.learning_rate = learning_rate
        self.model_internal = self._build_model()

        # Initialize optimizer and criterion here
        self.optimizer = optim.Adam(self.model_internal.parameters(), lr=self.learning_rate)
        # For DQN Q-value regression. Output layer should be raw logits (output_activation="none").
        self.criterion = nn.MSELoss()

    def _get_activation_function(self, activation_name: Optional[str]) -> Optional[nn.Module]:
        if activation_name is None or activation_name.lower() == "none": return None
        name_lower = activation_name.lower()
        if name_lower == "relu": return nn.ReLU()
        elif name_lower == "sigmoid": return nn.Sigmoid()
        elif name_lower == "tanh": return nn.Tanh()
        elif name_lower == "softmax": return nn.Softmax(dim=-1) # Usually for classification probabilities
        else: raise ValueError(f"Unsupported activation function: {activation_name}")

    def _build_model(self) -> nn.Sequential: # FFN specific implementation
        layers_config = self.nn_config.get("layers", [])
        input_size = self.nn_config.get("input_size")
        output_size = self.nn_config.get("output_size")
        output_activation_name = self.nn_config.get("output_activation", "none") # Default to raw logits

        if input_size is None or not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("FFN nn_config must specify a positive integer 'input_size'.")
        if output_size is None or not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("FFN nn_config must specify a positive integer 'output_size'.")

        model_layers: List[nn.Module] = []
        current_size = input_size
        for i, layer_conf in enumerate(layers_config):
            layer_type = layer_conf.get("type", "").lower()
            if layer_type == "linear":
                out_features = layer_conf.get("size")
                if out_features is None or not isinstance(out_features, int) or out_features <=0:
                    raise ValueError(f"Linear layer config at index {i} must specify 'size' as a positive integer.")
                model_layers.append(nn.Linear(current_size, out_features))
                current_size = out_features
                activation_fn = self._get_activation_function(layer_conf.get("activation"))
                if activation_fn: model_layers.append(activation_fn)
            elif layer_type == "dropout":
                rate = layer_conf.get("rate", 0.5)
                if not (isinstance(rate, float) and 0.0 <= rate < 1.0): # Dropout rate is [0, 1)
                    raise ValueError(f"Dropout layer config at index {i} must have 'rate' as a float in [0,1).")
                model_layers.append(nn.Dropout(rate))
            else: raise ValueError(f"Unsupported FFN layer type in nn_config at index {i}: {layer_type}")

        model_layers.append(nn.Linear(current_size, output_size))
        output_activation_fn = self._get_activation_function(output_activation_name)
        if output_activation_fn: model_layers.append(output_activation_fn)

        return nn.Sequential(*model_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("Input features must be a PyTorch Tensor.")
        if features.dtype != torch.float32:
            features = features.to(torch.float32)

        expected_input_features = self.nn_config.get("input_size")
        if features.shape[-1] != expected_input_features:
            raise ValueError(f"Input feature size ({features.shape[-1]}) does not match model's expected input_size ({expected_input_features}).")
        return self.model_internal(features)

    def predict(self, features: Union[List[float], torch.Tensor]) -> torch.Tensor:
        self.model_internal.eval()
        with torch.no_grad():
            if isinstance(features, list):
                features_tensor = torch.tensor(features, dtype=torch.float32)
            elif isinstance(features, torch.Tensor):
                features_tensor = features.to(torch.float32)
            else:
                raise TypeError(f"FFN model predict expects list or Tensor, got {type(features)}")

            if features_tensor.ndim == 1: # Add batch dimension if flat (single instance)
                features_tensor = features_tensor.unsqueeze(0)

            output = self.forward(features_tensor)
        return output

    def train_on_batch(self, experiences: List[Any], # List of Experience namedtuples from rl_utils
                       gamma: float = 0.99) -> float:
        if not experiences: return 0.0 # No loss if no experiences

        self.model_internal.train() # Set model to training mode

        # Assuming Experience is a namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
        # And state/next_state are feature lists/tensors for FFN

        states = torch.tensor([e.state for e in experiences], dtype=torch.float32)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).unsqueeze(1) # For gather
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)

        # Handle None in next_states for terminal states
        non_final_mask = torch.tensor([e.next_state is not None for e in experiences], dtype=torch.bool)
        non_final_next_states_list = [e.next_state for e in experiences if e.next_state is not None]

        # Q(s,a)
        q_predicted_all_actions = self.forward(states)
        q_predicted_for_taken_actions = q_predicted_all_actions.gather(1, actions).squeeze(1)

        # Max Q(s',a') for non-terminal next states
        next_state_max_q_values = torch.zeros(len(experiences), device=states.device) # Zero for terminal states
        if len(non_final_next_states_list) > 0:
            non_final_next_states_tensor = torch.tensor(non_final_next_states_list, dtype=torch.float32)
            with torch.no_grad():
                next_q_all_actions = self.forward(non_final_next_states_tensor)
            next_state_max_q_values[non_final_mask] = next_q_all_actions.max(1)[0]

        # Q_target = R + gamma * max_Q(s')
        # For done states, Q_target is just R (because next_state_max_q_values[done_indices] is 0)
        q_target_values = rewards + (gamma * next_state_max_q_values)

        loss = self.criterion(q_predicted_for_taken_actions, q_target_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not hasattr(self, "_train_batch_call_count"): self._train_batch_call_count = 0
        self._train_batch_call_count += 1
        return loss.item()

    def get_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self.nn_config)

    def reconfigure(self, new_nn_config: Dict[str, Any]):
        print(f"Reconfiguring ReprogrammableSelectorNN.")
        if not new_nn_config or not isinstance(new_nn_config, dict):
            raise ValueError("Invalid new_nn_config: Must be a dictionary.")

        # For this FFN-focused phase, ensure new config is also FFN
        new_model_type = new_nn_config.get("model_type", "ffn").lower()
        if new_model_type != "ffn":
            raise ValueError(f"Reconfiguration to model_type '{new_model_type}' not supported in this FFN-only phase.")
        new_nn_config["model_type"] = "ffn" # Ensure it's set

        if 'input_size' not in new_nn_config or 'output_size' not in new_nn_config:
            raise ValueError("For FFN, new_nn_config must specify 'input_size' and 'output_size'.")

        self.nn_config = copy.deepcopy(new_nn_config)
        self.model_type = self.nn_config.get("model_type") # Should be 'ffn'
        print(f"  NN config updated. New model_type: {self.model_type}")
        try:
            self.model_internal = self._build_model()
            print("  Model has been rebuilt.")
            # Re-initialize optimizer with new model parameters and stored learning rate
            self.optimizer = optim.Adam(self.model_internal.parameters(), lr=self.learning_rate)
            self.criterion = nn.MSELoss() # Or re-init if criterion could change
        except Exception as e:
            print(f"  Error rebuilding model: {e}. Model may be in an inconsistent state.")


# Example usage (for testing this file directly)
if __name__ == '__main__':
    print("--- ReprogrammableSelectorNN (FFN Path) Direct Test ---")

    Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done')) # For testing train_on_batch

    ffn_conf = {
        "model_type": "ffn",
        "input_size": 10,
        "layers": [{"type": "linear", "size": 64, "activation": "relu"}, {"type": "linear", "size": 32, "activation": "relu"}],
        "output_size": 3, # Number of actions
        "output_activation": "none" # For Q-values
    }
    print("\n1. Testing FFN configuration:")
    nn_ffn = ReprogrammableSelectorNN(ffn_conf, learning_rate=0.01)
    print(f"  Model Structure:\n{nn_ffn.model_internal}")

    ffn_features_single_list = [random.random() for _ in range(10)]
    ffn_features_single_tensor = torch.tensor(ffn_features_single_list, dtype=torch.float32)

    print(f"\n  FFN Prediction (from list): {nn_ffn.predict(ffn_features_single_list)}")
    print(f"  FFN Prediction (from tensor): {nn_ffn.predict(ffn_features_single_tensor)}")

    ffn_features_batch = torch.randn(4, 10) # Batch of 4
    print(f"  FFN Prediction (batch): shape {nn_ffn.predict(ffn_features_batch).shape}")

    print("\n2. Testing FFN train_on_batch:")
    dummy_experiences_ffn = [
        Experience(state=[random.random() for _ in range(10)], action=0, reward=1.0, next_state=[random.random() for _ in range(10)], done=False),
        Experience(state=[random.random() for _ in range(10)], action=1, reward=-1.0, next_state=None, done=True),
        Experience(state=[random.random() for _ in range(10)], action=2, reward=0.5, next_state=[random.random() for _ in range(10)], done=False)
    ]
    try:
        loss_ffn = nn_ffn.train_on_batch(dummy_experiences_ffn)
        print(f"  FFN batch training loss: {loss_ffn}")
        loss_ffn_2 = nn_ffn.train_on_batch(dummy_experiences_ffn) # Another step
        print(f"  FFN batch training loss (2nd step): {loss_ffn_2}")
    except Exception as e:
        print(f"  Error during FFN train_on_batch: {e}")


    print("\n3. Testing reconfigure (FFN to FFN):")
    new_ffn_conf = {
        "model_type": "ffn",
        "input_size": 10,
        "layers": [{"type": "linear", "size": 128, "activation": "tanh"}], # Different layer
        "output_size": 4, # Different output size
        "output_activation": "none"
    }
    nn_ffn.reconfigure(new_ffn_conf)
    print(f"  Model Structure after reconfigure:\n{nn_ffn.model_internal}")
    print(f"  FFN Prediction after reconfigure (output shape {nn_ffn.predict(ffn_features_single_tensor).shape})")
    assert nn_ffn.predict(ffn_features_single_tensor).shape[1] == 4

    print("\n--- Test Complete ---")
