# examples/reprogrammable_nn_selection_demo.py
import sys
import os
import ast
import torch # ReprogrammableSelectorNN uses torch for predict output
import random # For list data

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.engine import (
    MutableEntity,
    NetworkGraph,
    IntegerPerturbationOperator,
    ListElementSwapOperator,
    # StringReplaceOperator, # Not directly used in this demo's entities
    SmartMutationEngine
)
from modules.python_ast_operators import (
    ASTConstantChangeOperator,
    ASTSimpleLocalVariableRenameOperator
)
from modules.graph_utils import create_instruction_node, create_label_node, add_sequential_flow_edge, add_control_flow_edge

from modules.reprogrammable_selector_nn import ReprogrammableSelectorNN
# DEFAULT_FEATURE_CONFIG is imported by SmartMutationEngine by default if no config is passed to its init.
# from modules.feature_extractors import DEFAULT_FEATURE_CONFIG

# --- Helper to build a simple assembly graph ---
def build_simple_assembly_graph() -> NetworkGraph:
    g = NetworkGraph("asm_sample_for_nn_demo") # Unique graph ID
    start_label = create_label_node(g, "START_nn")
    mov_instr = create_instruction_node(g, "MOV", ["EAX", "10"])
    add_instr = create_instruction_node(g, "ADD", ["EAX", "5"])
    end_label = create_label_node(g, "END_nn")
    jmp_instr = create_instruction_node(g, "JMP", ["END_nn"])
    nop_instr = create_instruction_node(g, "NOP")

    add_sequential_flow_edge(g, start_label, mov_instr)
    add_sequential_flow_edge(g, mov_instr, add_instr)
    add_sequential_flow_edge(g, add_instr, jmp_instr)
    add_control_flow_edge(g, jmp_instr, end_label, "jump_unconditional")
    add_sequential_flow_edge(g, end_label, nop_instr)
    return g

# --- Sample Python Code for AST ---
PYTHON_CODE_FOR_AST = """
def simple_func_for_nn(a_param):
    x_var = a_param + 100 # Numeric constant
    a_string = "example"   # String constant
    if x_var > 150:      # Numeric constant
        print("Big value!") # String constant
    return x_var
result_val = simple_func_for_nn(5) # Numeric constant
"""

def main():
    print("--- Reprogrammable NN Operator Selection Demo ---")

    # 1. Define Mutable Entities
    entity_list = MutableEntity([random.randint(0,100) for _ in range(random.randint(3,8))]) # Random list

    ast_tree = ast.parse(PYTHON_CODE_FOR_AST)
    entity_ast = MutableEntity(ast_tree) # ast.Module is the data

    asm_graph = build_simple_assembly_graph()
    entity_asm_graph = MutableEntity(asm_graph)

    entities_map = { # Use a map for ordered iteration and clear naming
        "List Entity": entity_list,
        "Python AST Entity": entity_ast,
        "Assembly Graph Entity": entity_asm_graph
    }

    # 2. Define available Mutation Operators
    op_list_swap = ListElementSwapOperator()
    op_int_perturb = IntegerPerturbationOperator(perturbation_range=(-50,50)) # CORRECTED: used perturbation_range
    op_ast_const = ASTConstantChangeOperator(numeric_perturbation_range=(-5,5), max_string_append=3)
    op_ast_rename_param = ASTSimpleLocalVariableRenameOperator("simple_func_for_nn", "a_param", "input_value")
    op_ast_rename_local = ASTSimpleLocalVariableRenameOperator("simple_func_for_nn", "x_var", "temp_val")

    selectable_operators = [
        op_list_swap,
        op_int_perturb,
        op_ast_const,
        op_ast_rename_param,
        op_ast_rename_local
    ]
    num_operators = len(selectable_operators)
    print(f"\nSelectable Operators for NN (total {num_operators}):")
    for i, op in enumerate(selectable_operators):
        print(f"  Index {i}: {op.__class__.__name__}")

    try:
        from modules.feature_extractors import DEFAULT_FEATURE_CONFIG
        nn_input_size = DEFAULT_FEATURE_CONFIG["max_vector_size"]
    except ImportError:
        print("Error: Could not import DEFAULT_FEATURE_CONFIG. Please ensure feature_extractors.py is correct.")
        return

    initial_nn_config = {
        "input_size": nn_input_size,
        "layers": [
            {"type": "linear", "size": 32, "activation": "relu"},
            {"type": "linear", "size": 16, "activation": "relu"}
        ],
        "output_size": num_operators,
        "output_activation": "softmax"
    }
    print(f"\nInitializing ReprogrammableSelectorNN with input_size={nn_input_size}, output_size={num_operators}")
    nn_selector = ReprogrammableSelectorNN(nn_config=initial_nn_config)

    smart_engine = SmartMutationEngine(
        operators=selectable_operators,
        nn_selector=nn_selector
    )
    print("SmartMutationEngine initialized.")

    # Store last features for reconfig demo
    last_entity_features = None

    for name, entity in entities_map.items():
        print(f"\n--- Processing Entity: {name} ---")
        data_repr = str(entity.data)
        if len(data_repr) > 150: data_repr = data_repr[:150] + "..."
        print(f"Original data (type: {type(entity.data).__name__}): {data_repr}")

        features = smart_engine._extract_features(entity)
        last_entity_features = features # Save for later
        print(f"  Extracted Features (target size {nn_input_size}): {features}")

        if features:
            op_scores_tensor = nn_selector.predict(features)
            op_scores = op_scores_tensor[0].tolist()
            print(f"  NN Operator Scores/Probabilities (raw output for {num_operators} ops):")
            for i, score in enumerate(op_scores):
                print(f"    Op Index {i} ({selectable_operators[i].__class__.__name__}): {score:.4f}")

            predicted_op_index = torch.argmax(op_scores_tensor[0]).item()
            print(f"  NN Highest Score Operator (raw prediction): {selectable_operators[predicted_op_index].__class__.__name__} (Index {predicted_op_index})")

        print("  Running smart_engine.mutate_once():")

        original_data_str_short = ""
        if isinstance(entity.data, list):
            original_data_str_short = str(entity.data)[:100]

        chosen_op_by_engine = smart_engine.select_operator_intelligently(entity)

        if chosen_op_by_engine:
            print(f"    Operator PREDICTED by SmartEngine (after can_apply): {chosen_op_by_engine.__class__.__name__}")
            mutated = smart_engine.mutate_once(entity)
            if mutated:
                new_data_repr = str(entity.data)
                if len(new_data_repr) > 150: new_data_repr = new_data_repr[:150] + "..."
                print(f"    Entity MUTATED. New data: {new_data_repr}")
                if isinstance(entity.data, list) and original_data_str_short != str(entity.data)[:100] :
                     print(f"      (Original list data was: {original_data_str_short}...) ")
            else:
                print("    Entity NOT mutated (mutate_once returned False).")
        else:
            print("    SmartEngine did not select any applicable operator prior to mutate_once call.")

    print("\n--- Illustrating NN Reconfigurability ---")
    current_nn_config = nn_selector.get_config()
    print(f"Current NN config input_size: {current_nn_config['input_size']}, output_size: {current_nn_config['output_size']}")

    new_nn_config_example = {
        "input_size": nn_input_size,
        "layers": [ {"type": "linear", "size": 10, "activation": "tanh"} ],
        "output_size": num_operators,
        "output_activation": "none"
    }
    print(f"\nAttempting to reconfigure NN with new config: {new_nn_config_example}")
    nn_selector.reconfigure(new_nn_config_example)
    reconfigured_nn_config = nn_selector.get_config()
    print(f"  NN reconfigured. New output_activation: {reconfigured_nn_config['output_activation']}")

    if last_entity_features:
        op_scores_tensor_reconfig = nn_selector.predict(last_entity_features)
        print(f"  NN Operator Scores (logits) after reconfig for last entity's features: {op_scores_tensor_reconfig[0].tolist()}")

    print("\n--- Reprogrammable NN Operator Selection Demo Complete ---")

if __name__ == "__main__":
    main()
