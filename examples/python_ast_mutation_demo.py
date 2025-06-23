# examples/python_ast_mutation_demo.py
import sys
import os
import ast

# Add the project root to sys.path to allow importing 'modules'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.engine import MutableEntity, MutationEngine
from modules.python_ast_operators import (
    ASTConstantChangeOperator,
    ASTSimpleLocalVariableRenameOperator
)

# Helper to unparse, trying ast.unparse first (Python 3.9+)
# then falling back to astor if available (not included by default)
def unparse_ast(tree):
    if hasattr(ast, "unparse"):
        return ast.unparse(tree)
    else:
        try:
            import astor
            return astor.to_source(tree)
        except ImportError:
            return ("# Cannot unparse: Python < 3.9 and astor library not installed.\n"
                    "# AST structure:\n" + ast.dump(tree, indent=2))


PYTHON_CODE_SNIPPET = """
def greet(name, count):
    message = "Hello, " + name + "!" # A string constant here
    secret_number = 42 # A numeric constant
    for i in range(count): # 'count' is a parameter, 'i' is local
        print(message)
    if count > 10: # 10 is a numeric constant
        bonus_message = "That's a lot of greetings!" # Another string constant
        print(bonus_message)
    # Let's ensure 'name' is used in a way that ASTSimpleLocalVariableRenameOperator can find
    if name == "Special": # Using 'name' in a comparison
        print("Special greeting for Special!")
    return len(message) * count

# Global scope code
person_name = "Alice" # 'name' is a common variable name, ensure operator is targeted
result = greet(person_name, 5) # 5 is a numeric constant
final_val = result + 100 # 100 is a numeric constant
print(f"Final value for {person_name} is {final_val}")
"""

def main():
    print("--- Python AST Mutation Demo ---")
    print("\nOriginal Python Code Snippet:")
    print("-------------------------------")
    print(PYTHON_CODE_SNIPPET)
    print("-------------------------------")

    # 1. Parse the Python code string into an AST
    try:
        ast_tree = ast.parse(PYTHON_CODE_SNIPPET)
    except SyntaxError as e:
        print(f"Error parsing code snippet: {e}")
        return

    # 2. Wrap the AST in a MutableEntity
    entity = MutableEntity(ast_tree)
    print("\nAST created and wrapped in MutableEntity.")

    # 3. Create mutation operators
    # Operator to change constants
    constant_changer = ASTConstantChangeOperator(
        numeric_perturbation_range=(-3, 3),
        max_string_append=2
    )

    # Operator to rename 'name' to 'recipient_name' inside the 'greet' function
    # Also, let's try to rename 'i' to 'loop_counter' in 'greet'
    # We'll need two instances if we want to rename two different variables.
    var_renamer_name = ASTSimpleLocalVariableRenameOperator(
        target_function_name="greet",
        target_variable_name="name", # This is a parameter
        new_variable_name="recipient_name"
    )
    var_renamer_i = ASTSimpleLocalVariableRenameOperator(
        target_function_name="greet",
        target_variable_name="i", # This is a local variable in the for loop
        new_variable_name="loop_counter"
    )

    operators = [constant_changer, var_renamer_name, var_renamer_i]

    # Using MutationEngine to apply them (though we could apply manually too)
    # The engine will apply them one by one if it can.
    # For AST, applying multiple NodeTransformers sequentially is fine.
    engine = MutationEngine(operators=operators, mutation_probability=1.0) # Ensure they always try

    print("\nApplying Mutations...")
    successful_mutations = 0

    # Apply constant changer
    print("\n1. Applying ASTConstantChangeOperator:")
    if constant_changer.can_apply(entity):
        constant_changer.apply(entity) # Apply directly to see its effect
        if constant_changer.mutation_applied_flag:
            print("  ASTConstantChangeOperator applied some changes.")
            successful_mutations+=1
        else:
            print("  ASTConstantChangeOperator made no changes (no suitable constants found or random chance).")
        print("\nCode after Constant Changes:")
        print("-------------------------------")
        print(unparse_ast(entity.data))
        print("-------------------------------")
    else:
        print("  ASTConstantChangeOperator cannot be applied.")

    # Apply variable renamer for 'name'
    print("\n2. Applying ASTSimpleLocalVariableRenameOperator for 'name' -> 'recipient_name':")
    if var_renamer_name.can_apply(entity):
        var_renamer_name.apply(entity)
        if var_renamer_name.mutation_applied_flag:
            print("  ASTSimpleLocalVariableRenameOperator (name) applied some changes.")
            successful_mutations+=1
        else:
            print("  ASTSimpleLocalVariableRenameOperator (name) made no changes (target func/var not found).")
        print("\nCode after 'name' Renaming:")
        print("-------------------------------")
        print(unparse_ast(entity.data))
        print("-------------------------------")
    else:
        print("  ASTSimpleLocalVariableRenameOperator (name) cannot be applied.")

    # Apply variable renamer for 'i'
    print("\n3. Applying ASTSimpleLocalVariableRenameOperator for 'i' -> 'loop_counter':")
    if var_renamer_i.can_apply(entity):
        var_renamer_i.apply(entity)
        if var_renamer_i.mutation_applied_flag:
            print("  ASTSimpleLocalVariableRenameOperator (i) applied some changes.")
            successful_mutations+=1
        else:
            print("  ASTSimpleLocalVariableRenameOperator (i) made no changes (target func/var not found).")
        print("\nCode after 'i' Renaming:")
        print("-------------------------------")
        print(unparse_ast(entity.data))
        print("-------------------------------")
    else:
        print("  ASTSimpleLocalVariableRenameOperator (i) cannot be applied.")


    print(f"\nTotal mutation operators attempted: {len(operators)}, Successful applications (operator instances that made a change): {successful_mutations}")

    print("\nFinal Mutated Python Code:")
    print("-------------------------------")
    final_code = unparse_ast(entity.data)
    print(final_code)
    print("-------------------------------")

    # Optional: Try to execute the mutated code to see if it's still valid (can be risky)
    print("\nAttempting to execute mutated code (within a try-except block):")
    try:
        # Create a dictionary to serve as the global scope for exec
        exec_globals = {}
        exec(final_code, exec_globals)
        # If greet was defined, result should be in exec_globals
        if 'final_val' in exec_globals:
            print(f"Execution successful. Mutated final_val: {exec_globals['final_val']}")
        else:
            print("Execution seemed successful, but 'final_val' not found in globals.")
    except Exception as e:
        print(f"Error executing mutated code: {e}")

    print("\n--- Python AST Mutation Demo Complete ---")

if __name__ == "__main__":
    main()
