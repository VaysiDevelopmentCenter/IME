# examples/simple_mutation_demo.py
import sys
import os
# Add the project root to sys.path to allow importing 'modules'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.engine import (
    MutableEntity,
    IntegerPerturbationOperator,
    StringReplaceOperator,
    ListElementSwapOperator,
    MutationEngine
)
# random is used by the engine, but good to note if direct use was needed here.

def main():
    print("--- IME Simple Mutation Demo ---")

    # 1. Create Mutable Entities with different data types
    entity_int = MutableEntity(100)
    entity_str = MutableEntity("Hello, Mutation World!")
    entity_list = MutableEntity([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Example of an entity type that current basic operators won't target
    entity_dict = MutableEntity({"name": "TestObj", "value": 50, "tags": ["initial", "test"]})

    entities_to_test = {
        "Integer Entity": entity_int,
        "String Entity": entity_str,
        "List Entity": entity_list,
        "Dictionary Entity (no ops)": entity_dict
    }

    # 2. Create instances of our mutation operators
    int_op = IntegerPerturbationOperator(perturbation_range=(-10, 10))
    str_op = StringReplaceOperator()
    list_op = ListElementSwapOperator()

    all_operators = [int_op, str_op, list_op]

    # 3. Instantiate the MutationEngine
    # Using default mutation_probability = 1.0 for this demo
    engine = MutationEngine(operators=all_operators)
    print(f"\nInitialized MutationEngine with operators: {[op.__class__.__name__ for op in engine.operators]}")

    # 4. Run mutation process and print states
    num_mutation_steps = 10 # Number of times engine.mutate_once() is called by engine.run()

    for name, entity in entities_to_test.items():
        print(f"\n--- Mutating: {name} ---")
        original_data_snapshot = str(entity.data) # Take a snapshot for comparison
        print(f"Original State: {original_data_snapshot}")

        successful_mutations = engine.run(entity, num_mutation_steps)

        print(f"State after {num_mutation_steps} mutation attempts ({successful_mutations} successful): {entity.data}")
        if str(entity.data) == original_data_snapshot and successful_mutations > 0:
            print("Note: Data appears unchanged but mutations were reported as successful. This might be due to mutations reversing each other or subtle changes not obvious in str representation.")
        elif str(entity.data) == original_data_snapshot and successful_mutations == 0:
             print("Note: Data is unchanged as no successful mutations were applied (as expected for some types or if no operators are applicable).")


        current_score = engine.evaluate_entity(entity) # Calling placeholder evaluation
        print(f"Placeholder evaluation score for mutated entity: {current_score}")


    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()
