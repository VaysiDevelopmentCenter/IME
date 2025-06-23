# modules/python_ast_operators.py
import ast
import random
import string # For generating random characters

# Attempt relative import for engine components
try:
    from .engine import MutationOperator, MutableEntity
except ImportError:
    # Fallback for scenarios where this script might be run directly for testing,
    # or if the package structure isn't perfectly set up during development.
    # This assumes 'engine.py' is in a directory that Python can find.
    # For robust package structure, the relative import should work.
    from engine import MutationOperator, MutableEntity


class ASTConstantChangeOperator(MutationOperator, ast.NodeTransformer):
    """
    Mutates constants within a Python AST.
    - Numeric constants (int, float) are perturbed by a small random integer.
    - String constants have a random printable ASCII character (excluding complex whitespace) appended.
    """
    def __init__(self, numeric_perturbation_range=(-5, 5), max_string_append=1):
        super(ASTConstantChangeOperator, self).__init__() # Call MutationOperator's __init__
        # ast.NodeTransformer's __init__ is object.__init__ and takes no arguments

        if not (isinstance(numeric_perturbation_range, tuple) and
                len(numeric_perturbation_range) == 2 and
                all(isinstance(x, int) for x in numeric_perturbation_range) and
                numeric_perturbation_range[0] <= numeric_perturbation_range[1]):
            raise ValueError("numeric_perturbation_range must be a tuple of two integers (min, max) with min <= max.")

        if not (isinstance(max_string_append, int) and max_string_append >= 0):
            raise ValueError("max_string_append must be a non-negative integer.")

        self.numeric_perturbation_range = numeric_perturbation_range
        self.max_string_append = max_string_append
        self.mutation_applied_flag = False # To track if any mutation occurred in the current `apply` call

    def can_apply(self, entity: MutableEntity) -> bool:
        return isinstance(entity.data, ast.AST)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """
        Visits ast.Constant nodes.
        For Python 3.8+, ast.Num, ast.Str, ast.Bytes, ast.NameConstant (True, False, None)
        are represented as ast.Constant.
        """
        original_value = node.value
        new_value = original_value
        mutated_this_node = False

        if isinstance(original_value, (int, float)):
            min_p, max_p = self.numeric_perturbation_range
            perturbation = 0
            if min_p <= max_p : # Ensure valid range before trying to pick
                perturbation = random.randint(min_p, max_p)

            if isinstance(original_value, int) and perturbation == 0 and (min_p != 0 or max_p != 0):
                # Try to ensure a change if the range isn't just (0,0)
                # This loop could be problematic if min_p == max_p == 0. Guarded by (min_p != 0 or max_p != 0)
                while perturbation == 0: # Re-roll if perturbation is zero
                    if min_p <= max_p:
                        perturbation = random.randint(min_p, max_p)
                    else: # Should not happen due to init check, but defensive
                        break

            new_value = original_value + perturbation
            mutated_this_node = True

        elif isinstance(original_value, str):
            if self.max_string_append > 0:
                num_chars_to_append = random.randint(1, self.max_string_append)
                safe_printable = string.ascii_letters + string.digits + string.punctuation + ' '
                append_chars = ''.join(random.choice(safe_printable) for _ in range(num_chars_to_append))
                new_value = original_value + append_chars
                mutated_this_node = True

        if mutated_this_node:
            self.mutation_applied_flag = True
            new_node = ast.Constant(value=new_value)
            ast.copy_location(new_node, node)
            return new_node

        # If not handled or not mutated, call generic_visit for ast.Constant node.
        # For ast.Constant, it has no children that generic_visit would process further,
        # so just returning 'node' is equivalent if no mutation occurred.
        return node # or self.generic_visit(node)

    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            return

        ast_tree_root = entity.data
        self.mutation_applied_flag = False

        modified_tree_root = self.visit(ast_tree_root)
        entity.data = modified_tree_root

        if self.mutation_applied_flag:
            ast.fix_missing_locations(entity.data)


class ASTSimpleLocalVariableRenameOperator(MutationOperator, ast.NodeTransformer):
    """
    (Simplified) Renames a specific local variable within a specific function in a Python AST.
    This is a basic example and does not handle complex scopes (e.g. globals, nonlocals,
    nested function closures) or ensure the new name doesn't clash with other existing names.
    It targets ast.Name nodes (variables, parameters).
    """
    def __init__(self, target_function_name: str,
                 target_variable_name: str,
                 new_variable_name: str):
        super().__init__() # Call MutationOperator's __init__

        # Check for valid identifiers and that names are different
        if not all(isinstance(name, str) and name and name.isidentifier() for name in [target_function_name, target_variable_name, new_variable_name]):
            raise ValueError("Function name, target variable name, and new variable name must be non-empty, valid Python identifiers.")
        if new_variable_name == target_variable_name:
            raise ValueError("New variable name must be different from the target variable name.")

        # Basic keyword check for new variable name
        # This list is not exhaustive for all Python versions but covers common ones.
        # A more robust check would use `keyword.iskeyword()`.
        _keywords = ["def", "class", "return", "yield", "for", "while", "if", "else", "elif",
                     "try", "except", "finally", "with", "as", "import", "from", "pass",
                     "break", "continue", "global", "nonlocal", "lambda", "assert", "del",
                     "in", "is", "not", "or", "and", "None", "True", "False"] # Added common built-in constants
        if new_variable_name in _keywords:
             raise ValueError(f"New variable name '{new_variable_name}' cannot be a Python keyword or common built-in constant.")

        self.target_function_name = target_function_name
        self.target_variable_name = target_variable_name
        self.new_variable_name = new_variable_name

        self._in_target_function_scope = False
        self._current_function_depth = 0 # To ensure we only rename in the top-level of the target function, not nested ones with same name.
        self.mutation_applied_flag = False

    def can_apply(self, entity: MutableEntity) -> bool:
        return isinstance(entity.data, ast.AST)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Visit FunctionDef to identify the scope of the target function."""
        # Check if this is the target function AND we are at the outermost scope for it (or first encounter)
        is_our_target_function_at_correct_depth = (node.name == self.target_function_name and self._current_function_depth == 0)

        original_in_target_scope_flag = self._in_target_function_scope

        if is_our_target_function_at_correct_depth:
            self._in_target_function_scope = True

        self._current_function_depth += 1
        self.generic_visit(node) # This will visit child nodes like arguments and body statements
        self._current_function_depth -= 1

        if is_our_target_function_at_correct_depth:
            self._in_target_function_scope = original_in_target_scope_flag # Restore flag after exiting function scope

        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Visit function arguments (ast.arg holds the argument name in node.arg)."""
        # Rename if it's a parameter of the target function
        if self._in_target_function_scope and self._current_function_depth == 1 and \
           node.arg == self.target_variable_name:
            node.arg = self.new_variable_name
            self.mutation_applied_flag = True
            ast.fix_missing_locations(node) # Good practice for any modified node
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Visit Name nodes (variables, function names used, etc.)."""
        # Rename if it's a variable name (load/store/del context) within the target function
        if self._in_target_function_scope and self._current_function_depth == 1 and \
           node.id == self.target_variable_name:
            new_node = ast.Name(id=self.new_variable_name, ctx=node.ctx)
            ast.copy_location(new_node, node)
            self.mutation_applied_flag = True
            return new_node

        return self.generic_visit(node) # Default behavior for other Name nodes

    def apply(self, entity: MutableEntity) -> None:
        if not self.can_apply(entity):
            return

        ast_tree_root = entity.data
        self._in_target_function_scope = False
        self._current_function_depth = 0 # Reset depth for each apply call
        self.mutation_applied_flag = False

        modified_tree_root = self.visit(ast_tree_root)
        entity.data = modified_tree_root

        if self.mutation_applied_flag:
            ast.fix_missing_locations(entity.data)
