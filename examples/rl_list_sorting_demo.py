# examples/rl_list_sorting_demo.py
import sys
import os
import ast
import random
import torch

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.engine import (
    MutableEntity,
    SmartMutationEngine,
    IntegerPerturbationOperator,
    ListElementSwapOperator
)
from modules.reprogrammable_selector_nn import ReprogrammableSelectorNN
from modules.feature_extractors import DEFAULT_FEATURE_CONFIG

def generate_random_list(size=7, min_val=0, max_val=20) -> list[int]:
    return [random.randint(min_val, max_val) for _ in range(size)]

def main():
    print("--- RL List Sorting Demo ---")

    op_list_swap = ListElementSwapOperator()
    op_int_perturb = IntegerPerturbationOperator(perturbation_range=(-3, 3))

    selectable_operators = [op_list_swap, op_int_perturb]
    num_operators = len(selectable_operators)

    print(f"\nOperators for NN to select from (total {num_operators}):")
    for i, op in enumerate(selectable_operators):
        print(f"  Index {i}: {op.__class__.__name__}")

    nn_input_size = DEFAULT_FEATURE_CONFIG["max_vector_size"]

    initial_nn_config = {
        "model_type": "ffn",
        "input_size": nn_input_size,
        "layers": [
            {"type": "linear", "size": 64, "activation": "relu"},
            {"type": "linear", "size": 32, "activation": "relu"}
        ],
        "output_size": num_operators,
        "output_activation": "none" # Raw Q-values
    }
    print(f"\nInitializing ReprogrammableSelectorNN (FFN) with input_size={nn_input_size}, output_size={num_operators}")
    nn_selector = ReprogrammableSelectorNN(nn_config=initial_nn_config, learning_rate=0.001)

    smart_engine = SmartMutationEngine(
        operators=selectable_operators,
        nn_selector=nn_selector,
        replay_buffer_capacity=5000,
        rl_batch_size=64,
        train_frequency=1,
        mutation_probability=1.0,
        max_steps_per_episode=30
    )
    print("SmartMutationEngine initialized for RL.")

    num_episodes = 200
    print_every_n_episodes = 20

    all_episode_final_scores = []
    total_training_steps = 0

    for episode in range(num_episodes):
        initial_list_data = generate_random_list()
        list_entity = MutableEntity(list(initial_list_data))

        current_episode_score = smart_engine.evaluate_entity(list_entity)

        for step in range(smart_engine.max_steps_per_episode):
            mutated_successfully = smart_engine.mutate_once(list_entity)

            if mutated_successfully:
                new_score = smart_engine.evaluate_entity(list_entity)
                current_episode_score = new_score
                if smart_engine.mutation_steps_count % smart_engine.train_frequency == 0 and \
                   len(smart_engine.replay_buffer) >= smart_engine.rl_batch_size:
                    total_training_steps +=1
            if current_episode_score == 1.0:
                break

        all_episode_final_scores.append(current_episode_score)

        if (episode + 1) % print_every_n_episodes == 0:
            avg_final_score = sum(all_episode_final_scores[-print_every_n_episodes:]) / min(print_every_n_episodes, len(all_episode_final_scores))
            print(f"Ep {episode + 1}: Final Score: {current_episode_score:.4f}. Avg final score (last {print_every_n_episodes} eps): {avg_final_score:.4f}. Total NN training steps: {total_training_steps}")

            sample_list_features = smart_engine._extract_features(MutableEntity(generate_random_list()))
            if sample_list_features and 'torch' in globals() and torch is not None:
                sample_list_tensor = torch.tensor([sample_list_features], dtype=torch.float32)
                scores_tensor = nn_selector.predict(sample_list_tensor)
                print(f"  NN Q-Values for sample list: {[f'{s:.2f}' for s in scores_tensor[0].tolist()]}")

    print(f"\n--- RL List Sorting Demo Complete ({smart_engine.mutation_steps_count} total mutations over {num_episodes} episodes) ---")
    print(f"Total times NN's train_on_batch was called: {total_training_steps}")

if __name__ == "__main__":
    main()
