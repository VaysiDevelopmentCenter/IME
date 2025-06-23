# This file makes the 'modules' directory a Python package.

# Optionally, you can make some classes/functions available directly when importing 'modules'
# For example:
from .engine import (
    MutableEntity,
    MutationOperator,
    MutationEngine,
    NetworkGraph,
    ArchitecturalNode,
    ArchitecturalEdge,
    PartitionSchema
    # SROs and FMOs are more specialized, typically imported directly
)
from .graph_utils import (
    create_instruction_node,
    create_label_node,
    add_sequential_flow_edge,
    add_control_flow_edge
)
from .python_ast_operators import (
    ASTConstantChangeOperator
    # Add other AST operators here as they are created
)

# For now, keeping it simple and requiring explicit sub-module imports for most things,
# but making very common ones potentially easier to access.
# Users can still do from modules.engine import ... or from modules.python_ast_operators import ...
