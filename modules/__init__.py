# This file makes the 'modules' directory a Python package.

# Optionally, you can make some classes/functions available directly when importing 'modules'

# From engine.py (Core IME and basic operators)
from .engine import (
    MutableEntity,
    MutationOperator,
    IntegerPerturbationOperator,
    StringReplaceOperator,
    ListElementSwapOperator,
    MutationEngine
)

# From python_ast_operators.py (Python AST specific operators)
# Check if the file and classes exist before attempting to import, to avoid issues if files are added incrementally
try:
    from .python_ast_operators import ASTConstantChangeOperator, ASTSimpleLocalVariableRenameOperator
except ImportError:
    # print("Warning: python_ast_operators.py or its classes not found. AST operators will not be available.")
    pass

# Future modules can be added here:
# from .graph_utils import ...
# from .reprogrammable_selector_nn import ...
# from .feature_extractors import ...
# from .rl_utils import ...
# from .ast_to_graph_converter import ...

# For now, keeping it somewhat minimal to ensure stability during phased re-implementation.
# Users can also do from modules.engine import ... or from modules.python_ast_operators import ...
# if direct imports from 'modules' are not set up for all components.
