# modules/reprogrammable_selector_nn.py
import torch
import torch.nn as nn
import torch.optim as optim # For future actual training
import copy # For deepcopying configs
import random # For dummy data generation

# Placeholder for a more sophisticated feature config system if needed by the NN directly
# For now, the NN primarily relies on nn_config for its structure and assumes
# input_size in nn_config matches the externally prepared feature vector.

class ReprogrammableSelectorNN(nn.Module):
    """
    A reprogrammable neural network for selecting mutation operators.
    Its architecture is defined by a configuration dictionary.
    """
    def __init__(self, nn_config: dict):
        super().__init__() # Call nn.Module's __init__

        if not nn_config or not isinstance(nn_config, dict):
            raise ValueError("nn_config dictionary must be provided.")

        self.nn_config = copy.deepcopy(nn_config) # Store a copy
        self.feature_config_ref = None # Placeholder if we want to store feature_config too

        self.model = self._build_model()

        # Placeholder for optimizer and loss function for actual training later
        self.optimizer = None
        self.criterion = None

    def _get_activation_function(self, activation_name: str | None) -> nn.Module | None:
        if activation_name is None or activation_name.lower() == "none":
            return None # No activation, effectively linear output from previous layer
        name_lower = activation_name.lower()
        if name_lower == "relu":
            return nn.ReLU()
        elif name_lower == "sigmoid":
            return nn.Sigmoid()
        elif name_lower == "tanh":
            return nn.Tanh()
        elif name_lower == "softmax":
            # Softmax is typically applied to the output layer, dim=-1 usually means over the last dimension (features/scores)
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def _build_model(self) -> nn.Sequential:
        layers_config = self.nn_config.get("layers", [])
        input_size = self.nn_config.get("input_size")
        output_size = self.nn_config.get("output_size")
        output_activation_name = self.nn_config.get("output_activation", "none") # Default to no activation if not specified

        if input_size is None or output_size is None:
            raise ValueError("nn_config must specify 'input_size' and 'output_size'.")
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("'input_size' must be a positive integer.")
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("'output_size' must be a positive integer.")


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

                activation_name = layer_conf.get("activation") # Can be None
                activation_fn = self._get_activation_function(activation_name)
                if activation_fn:
                    model_layers.append(activation_fn)

            elif layer_type == "dropout":
                rate = layer_conf.get("rate", 0.5)
                if not (isinstance(rate, float) and 0.0 <= rate < 1.0): # Dropout rate is [0, 1)
                    raise ValueError(f"Dropout layer config at index {i} must have a 'rate' float between 0.0 and 1.0 (exclusive of 1).")
                model_layers.append(nn.Dropout(rate))

            else:
                raise ValueError(f"Unsupported layer type in nn_config at index {i}: {layer_type}")

        # Add the final output layer
        model_layers.append(nn.Linear(current_size, output_size))
        output_activation_fn = self._get_activation_function(output_activation_name)
        if output_activation_fn:
            model_layers.append(output_activation_fn)

        return nn.Sequential(*model_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the network.
        Assumes features is already a PyTorch tensor.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("Input features must be a PyTorch Tensor.")

        if features.dtype != torch.float32:
            features = features.to(torch.float32)

        # Ensure input features match expected input_size if model is not empty
        if len(self.model) > 0: # Check if model has any layers
            expected_input_features = self.nn_config.get("input_size")
            if features.shape[-1] != expected_input_features:
                raise ValueError(f"Input feature size ({features.shape[-1]}) does not match model's expected input_size ({expected_input_features}).")

        return self.model(features)

    def predict(self, features: list | torch.Tensor) -> torch.Tensor:
        """
        Takes features (list of numbers or PyTorch Tensor), converts to tensor if needed,
        and returns the network's output (operator scores/probabilities).
        Sets model to evaluation mode.
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(features, torch.Tensor):
                features_tensor = torch.tensor(features, dtype=torch.float32)
            else:
                features_tensor = features.to(torch.float32) # Ensure dtype

            if features_tensor.ndim == 1:
                features_tensor = features_tensor.unsqueeze(0) # Add batch dimension if flat

            output = self.forward(features_tensor)
        return output


    def train_step(self, features: torch.Tensor, target_operator_idx: int, learning_rate=0.001):
        """
        (Placeholder/Very Simple Initial Logic for a single training update idea)
        """
        print(f"Placeholder train_step called for ReprogrammableSelectorNN.")
        # print(f"  Features shape: {features.shape if isinstance(features, torch.Tensor) else type(features)}")
        # print(f"  Target operator index: {target_operator_idx}")

        # Conceptual actual training logic (commented out for now)
        # if self.optimizer is None:
        #     self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # if self.criterion is None:
        #     # Assuming classification task, CrossEntropyLoss is common
        #     # It expects raw logits from the model (before softmax) and target as class index
        #     # If model's last layer is Softmax, NLLLoss might be used, or remove Softmax for CrossEntropyLoss
        #     self.criterion = nn.CrossEntropyLoss()

        # self.model.train()
        # self.optimizer.zero_grad()

        # # Ensure features tensor is correctly shaped (e.g., [batch_size, num_features])
        # if features.ndim == 1: features_for_train = features.unsqueeze(0)
        # else: features_for_train = features

        # predictions = self.forward(features_for_train)

        # # Ensure target is a tensor of correct shape for the loss function
        # target_tensor = torch.tensor([target_operator_idx], dtype=torch.long)
        # if predictions.shape[0] != target_tensor.shape[0] and predictions.shape[0] == 1: # if predictions is [1, N]
             # target_tensor = target_tensor.squeeze(0) # if criterion expects [N] vs [1,N]

        # loss = self.criterion(predictions, target_tensor)
        # loss.backward()
        # self.optimizer.step()
        # print(f"  Conceptual loss: {loss.item()}")
        # return loss.item()
        return None


    def get_config(self) -> dict:
        """Returns a deep copy of the current neural network configuration."""
        return copy.deepcopy(self.nn_config)

    def reconfigure(self, new_nn_config: dict):
        """
        Reconfigures the neural network with a new architecture by updating
        the nn_config and rebuilding the model.
        Weights are re-initialized.
        """
        print(f"Reconfiguring ReprogrammableSelectorNN.")

        if not new_nn_config or not isinstance(new_nn_config, dict) or \
           'input_size' not in new_nn_config or 'output_size' not in new_nn_config:
            print("  Warning: Invalid new_nn_config provided to reconfigure. 'input_size' and 'output_size' are mandatory. No changes made.")
            return

        self.nn_config = copy.deepcopy(new_nn_config)
        print(f"  NN config updated. New input_size: {self.nn_config.get('input_size')}, output_size: {self.nn_config.get('output_size')}")

        try:
            self.model = self._build_model() # Rebuild the model structure
            print("  Model has been rebuilt with the new configuration.")
            # Any existing optimizer would likely be invalid now and should be reset if training continues.
            self.optimizer = None
            self.criterion = None
        except Exception as e:
            print(f"  Error rebuilding model with new config: {e}. Model may be in an inconsistent state or using old structure.")


# Example usage (for testing this file directly)
if __name__ == '__main__':
    print("--- ReprogrammableSelectorNN Direct Test ---")

    sample_feature_config_max_size = 10

    sample_nn_config = {
        "input_size": sample_feature_config_max_size,
        "layers": [
            {"type": "linear", "size": 32, "activation": "relu"},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 16, "activation": "relu"}
        ],
        "output_size": 3,
        "output_activation": "softmax"
    }

    print("\n1. Initializing NN with sample_nn_config:")
    selector_nn = None
    try:
        selector_nn = ReprogrammableSelectorNN(nn_config=sample_nn_config)
        print(f"  NN Model Structure:\n{selector_nn.model}")
    except Exception as e:
        print(f"  Error initializing NN: {e}")
        exit()

    print("\n2. Making a prediction with dummy features:")
    dummy_features_1d = torch.randn(sample_feature_config_max_size)
    print(f"  Dummy features (1D tensor for single instance): {dummy_features_1d.shape}")
    try:
        predictions = selector_nn.predict(dummy_features_1d)
        print(f"  Predictions (batch_size=1, num_operators={sample_nn_config['output_size']}): {predictions}")
        if sample_nn_config["output_activation"] == "softmax":
             print(f"  Predicted probabilities sum: {predictions.sum().item()}")

        dummy_features_list = [random.random() for _ in range(sample_feature_config_max_size)]
        print(f"  Dummy features (list): {len(dummy_features_list)} elements")
        predictions_from_list = selector_nn.predict(dummy_features_list)
        print(f"  Predictions from list: {predictions_from_list}")

        # Test with batched input
        dummy_features_batch = torch.randn(4, sample_feature_config_max_size) # Batch of 4
        print(f"  Dummy features (batched tensor): {dummy_features_batch.shape}")
        predictions_batch = selector_nn.predict(dummy_features_batch)
        print(f"  Predictions for batch: {predictions_batch.shape}\n{predictions_batch}")


    except Exception as e:
        print(f"  Error during prediction: {e}")

    print("\n3. Calling placeholder train_step:")
    try:
        selector_nn.train_step(dummy_features_1d, target_operator_idx=1)
    except Exception as e:
        print(f"  Error during train_step: {e}")


    print("\n4. Testing get_config:")
    current_config = selector_nn.get_config()
    print(f"  Retrieved config (input_size): {current_config.get('input_size')}")
    assert current_config["input_size"] == sample_nn_config["input_size"]

    print("\n5. Testing reconfigure:")
    new_config = {
        "input_size": sample_feature_config_max_size, # Input size should match features
        "layers": [
            {"type": "linear", "size": 128, "activation": "relu"},
            {"type": "linear", "size": 64, "activation": "tanh"}
        ],
        "output_size": 4, # Changed output size
        "output_activation": "softmax" # Can also be "none" for logits
    }
    try:
        selector_nn.reconfigure(new_nn_config=new_config)
        print(f"  NN Model Structure after reconfigure:\n{selector_nn.model}")

        predictions_after_reconfig = selector_nn.predict(dummy_features_1d)
        print(f"  Predictions after reconfigure (output_size={new_config['output_size']}): {predictions_after_reconfig}")
        assert predictions_after_reconfig.shape[1] == new_config['output_size']

    except Exception as e:
        print(f"  Error during reconfigure or subsequent prediction: {e}")

    print("\n--- ReprogrammableSelectorNN Direct Test Complete ---")
