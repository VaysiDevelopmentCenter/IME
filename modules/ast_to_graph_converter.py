# modules/ast_to_graph_converter.py
import ast
import uuid

try:
    from .engine import NetworkGraph, ArchitecturalNode, ArchitecturalEdge
except ImportError:
    from engine import NetworkGraph, ArchitecturalNode, ArchitecturalEdge
    print("Warning: ast_to_graph_converter.py using fallback imports for engine components.")


def safe_unparse(node: ast.AST | None) -> str:
    """
    Safely unparses an AST node to a string.
    Returns a placeholder string if unparsing is not possible or fails.
    """
    if node is None:
        return "None"
    if hasattr(ast, "unparse"): # Python 3.9+
        try:
            return ast.unparse(node)
        except Exception:
            return f"<{type(node).__name__}_unparse_failed>"
    else: # Fallback for Python < 3.9
        try:
            import astor
            return astor.to_source(node).strip()
        except ImportError:
            return f"<{type(node).__name__}_Py<3.9_astor_missing>"
        except Exception:
            return f"<{type(node).__name__}_astor_unparse_failed>"


class ASTToGraphConverter(ast.NodeVisitor):
    """
    Converts a Python AST into a NetworkGraph representation.
    Nodes represent key structural elements (Module, FunctionDef, ClassDef, If, For, While, Try,
    Assign, Expr(Call), Return).
    Edges represent containment ('contains_body_first_stmt', 'defines_function', 'defines_class', 'contains_handler')
    and sequential flow ('sequential_statement').
    """

    def __init__(self):
        self.graph = NetworkGraph(graph_id=f"ast_graph_{str(uuid.uuid4())[:8]}")
        self.node_id_map: dict[ast.AST, str] = {}
        self.last_stmt_in_block_stack: list[str | None] = [None]
        self.current_structural_parent_stack: list[str | None] = [None]

    def _generate_unique_node_id(self, base_name: str) -> str:
        name_hint = "".join(c if c.isalnum() or c == '_' else '' for c in base_name.lower()[:20])
        graph_node_id = f"{name_hint}_{str(uuid.uuid4())[:4]}"
        suffix = 1
        temp_id = graph_node_id
        while self.graph.get_node(temp_id):
            temp_id = f"{graph_node_id}_{suffix}"
            suffix += 1
        return temp_id

    def _add_node(self, ast_node: ast.AST, graph_node_type: str, properties: dict = None, name_hint: str = None) -> ArchitecturalNode:
        props = properties.copy() if properties is not None else {}
        props['ast_node_type'] = type(ast_node).__name__
        if hasattr(ast_node, 'lineno'):
            props['lineno'] = ast_node.lineno

        node_id_base = name_hint if name_hint else graph_node_type
        graph_node_id = self._generate_unique_node_id(node_id_base)

        graph_node = ArchitecturalNode(node_id=graph_node_id, node_type=graph_node_type, properties=props)
        self.graph.add_node(graph_node)
        self.node_id_map[ast_node] = graph_node.id
        return graph_node

    def _add_edge(self, source_id: str | None, target_id: str | None, flow_type: str, properties: dict = None):
        if not source_id or not target_id: return

        edge_props = properties.copy() if properties is not None else {}
        edge_props['flow_type'] = flow_type

        edge_id_base = f"{flow_type.replace('_','-')}_{source_id}_to_{target_id}"
        edge_id = self._generate_unique_node_id(edge_id_base)

        edge = ArchitecturalEdge(source_node_id=source_id, target_node_id=target_id, edge_id=edge_id, properties=edge_props)
        try:
            self.graph.add_edge(edge)
        except ValueError as e:
            print(f"Warning: Failed to add edge {edge_id} from {source_id} to {target_id}: {e}")

    def _process_body_statements(self, body_nodes: list[ast.AST], container_graph_node_id: str):
        self.current_structural_parent_stack.append(container_graph_node_id)
        self.last_stmt_in_block_stack.append(None)

        first_stmt_node_id_in_this_body = None

        for i, stmt_ast_node in enumerate(body_nodes):
            self.visit(stmt_ast_node)
            current_stmt_graph_node_id = self.node_id_map.get(stmt_ast_node)

            if current_stmt_graph_node_id:
                if i == 0:
                    first_stmt_node_id_in_this_body = current_stmt_graph_node_id
                    self._add_edge(container_graph_node_id, first_stmt_node_id_in_this_body, "contains_body_first_stmt")

                if self.last_stmt_in_block_stack[-1] is not None:
                    self._add_edge(self.last_stmt_in_block_stack[-1], current_stmt_graph_node_id, "sequential_statement")

                self.last_stmt_in_block_stack[-1] = current_stmt_graph_node_id

        self.last_stmt_in_block_stack.pop()
        self.current_structural_parent_stack.pop()

    def visit_Module(self, node: ast.Module):
        graph_node = self._add_node(node, "Module",
                                    properties={'docstring': ast.get_docstring(node)},
                                    name_hint="module")
        self.current_structural_parent_stack[-1] = graph_node.id
        self._process_body_statements(node.body, graph_node.id)

    def _visit_definitional_statement(self, node, graph_node_type, props, name_hint):
        graph_node = self._add_node(node, graph_node_type, properties=props, name_hint=name_hint)

        if self.last_stmt_in_block_stack[-1] is not None:
             self._add_edge(self.last_stmt_in_block_stack[-1], graph_node.id, "sequential_statement")
        elif self.current_structural_parent_stack[-1] is not None and self.current_structural_parent_stack[-1] != graph_node.id :
             # If it's the first item in a block, its parent link is handled by _process_body_statements's "contains_body_first_stmt"
             # This 'defines_...' edge is more for Module -> FunctionDef at top level, or ClassDef -> MethodDef
             # Let's refine: only add this if it's not already linked by contains_body_first_stmt from the immediate structural parent.
             # This logic might be tricky to get right without seeing the direct parent from _process_body_statements.
             # For now, _process_body_statements handles the primary "containment" edge.
             # This 'defines_' edge can be for logical containment not necessarily first in body.
             # Let's simplify: _process_body_statements handles the first link. Subsequent items are sequential.
             pass


        self.last_stmt_in_block_stack[-1] = graph_node.id
        return graph_node


    def visit_FunctionDef(self, node: ast.FunctionDef):
        node_type = "AsyncFunctionDef" if isinstance(node, ast.AsyncFunctionDef) else "FunctionDef"
        props = {
            'name': node.name, 'args': safe_unparse(node.args),
            'decorator_list': [safe_unparse(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'returns_annotation': safe_unparse(node.returns) if node.returns else "None"
        }
        func_graph_node = self._visit_definitional_statement(node, node_type, props, node.name)
        self._process_body_statements(node.body, func_graph_node.id)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        props = {
            'name': node.name, 'bases': [safe_unparse(b) for b in node.bases],
            'keywords': [f"{k.arg}={safe_unparse(k.value)}" for k in node.keywords if k.arg],
            'decorator_list': [safe_unparse(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node)
        }
        class_graph_node = self._visit_definitional_statement(node, "ClassDef", props, node.name)
        self._process_body_statements(node.body, class_graph_node.id)

    def _create_and_link_simple_statement_node(self, ast_node: ast.AST, graph_node_type: str, name_hint: str = None, additional_props: dict = None):
        props = additional_props if additional_props is not None else {}
        if not 'source_str' in props: props['source_str'] = safe_unparse(ast_node)

        graph_node = self._add_node(ast_node, graph_node_type, properties=props, name_hint=name_hint if name_hint else graph_node_type)

        # Sequential linking is now primarily handled by _process_body_statements
        # self.last_stmt_in_block_stack[-1] will be updated by _process_body_statements *after* this node is visited and mapped
        return graph_node

    def visit_Assign(self, node: ast.Assign):
        props = {'targets': [safe_unparse(t) for t in node.targets], 'value': safe_unparse(node.value)}
        self._create_and_link_simple_statement_node(node, "AssignmentStatement", additional_props=props)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            props = {'call_expression': safe_unparse(node.value)}
            self._create_and_link_simple_statement_node(node, "ExpressionCallStatement", additional_props=props)
        elif isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            props = {'string_value': node.value.value}
            self._create_and_link_simple_statement_node(node, "StringExpressionStatement", additional_props=props)
        else:
            props = {'expression': safe_unparse(node.value)}
            self._create_and_link_simple_statement_node(node, "GenericExpressionStatement", additional_props=props)

    def visit_Return(self, node: ast.Return):
        props = {'value': safe_unparse(node.value) if node.value else "None"}
        self._create_and_link_simple_statement_node(node, "ReturnStatement", additional_props=props)

    def _visit_control_flow_statement(self, node, graph_node_type, props, name_hint):
        control_flow_graph_node = self._create_and_link_simple_statement_node(node, graph_node_type, name_hint=name_hint, additional_props=props)
        return control_flow_graph_node

    def visit_If(self, node: ast.If):
        props = {'test_expr_str': safe_unparse(node.test)}
        if_graph_node = self._visit_control_flow_statement(node, "IfStatement", props, "if")

        self._process_body_statements(node.body, if_graph_node.id)
        if node.orelse:
            self._process_body_statements(node.orelse, if_graph_node.id)

    def visit_For(self, node: ast.For):
        node_type = "AsyncForLoop" if isinstance(node, ast.AsyncFor) else "ForLoop"
        props = {'target_str': safe_unparse(node.target), 'iter_str': safe_unparse(node.iter)}
        for_graph_node = self._visit_control_flow_statement(node, node_type, props, "for")
        self._process_body_statements(node.body, for_graph_node.id)
        if node.orelse: self._process_body_statements(node.orelse, for_graph_node.id)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While):
        props = {'test_expr_str': safe_unparse(node.test)}
        while_graph_node = self._visit_control_flow_statement(node, "WhileLoop", props, "while")
        self._process_body_statements(node.body, while_graph_node.id)
        if node.orelse: self._process_body_statements(node.orelse, while_graph_node.id)

    def visit_Try(self, node: ast.Try):
        try_graph_node = self._visit_control_flow_statement(node, "TryBlock", {}, "try")

        self._process_body_statements(node.body, try_graph_node.id)

        for handler_ast_node in node.handlers:
            handler_props = {
                'exception_type': safe_unparse(handler_ast_node.type) if handler_ast_node.type else "AnyException",
                'as_name': handler_ast_node.name if handler_ast_node.name else "None"
            }
            handler_graph_node = self._add_node(handler_ast_node, "ExceptionHandler", properties=handler_props, name_hint="except")
            self._add_edge(try_graph_node.id, handler_graph_node.id, "contains_handler")
            self._process_body_statements(handler_ast_node.body, handler_graph_node.id)

        if node.orelse: self._process_body_statements(node.orelse, try_graph_node.id)
        if node.finalbody: self._process_body_statements(node.finalbody, try_graph_node.id)

    def generic_visit(self, node: ast.AST):
        # This ensures that if we haven't defined a specific visitor for an AST node type
        # that *could* contain other nodes we *do* care about (like a 'With' statement containing a body),
        # we still traverse into its children.
        # For nodes that become graph nodes themselves, their specific visit_ methods should handle
        # calling _process_body_statements or generic_visit on their relevant children.
        # print(f"DEBUG: Generic visit for {type(node).__name__}, children will be visited.")
        super().generic_visit(node)

    def convert(self, ast_root: ast.Module) -> NetworkGraph:
        self.visit(ast_root)
        return self.graph

# Example Usage:
if __name__ == '__main__':
    sample_code = """
def foo(x: int) -> str:
    \"\"\"This is a foo function.\"\"\"
    y = x + 10
    if x > 5:
        print(f"x is greater than {x}")
        z = y * 2
    else:
        print("x is not greater than 5")
        z = y / 2
    for i in range(z):
        if i % 3 == 0:
            print(f"Divisible by 3: {i}")
            continue
    return f"Result is {z}"

class Bar:
    \"\"\"A simple Bar class.\"\"\"
    class_var: int = 100

    def __init__(self, val):
        self.instance_var = val + Bar.class_var

    def get_val(self):
        return self.instance_var

a = foo(7)
b = Bar(a)
print(b.get_val())
"""
    parsed_ast = ast.parse(sample_code)
    print("\nConverting AST to NetworkGraph...")
    converter = ASTToGraphConverter()
    graph = converter.convert(parsed_ast)

    print("\n--- Generated NetworkGraph Summary ---")
    print(f"Graph ID: {graph.id}")
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Total Edges: {len(graph.edges)}")

    print("\n--- Adjacency List (Outgoing) - Sample ---")
    nodes_to_print = list(graph.nodes.keys())[:25]
    for node_id in nodes_to_print:
        node_obj = graph.get_node(node_id)
        if not node_obj: continue
        node_name_prop = node_obj.properties.get('name', '')
        node_src_prop = node_obj.properties.get('source_str', node_obj.properties.get('ast_node_type', 'N/A'))
        node_display_list = [prop for prop in [node_name_prop, node_src_prop] if prop and prop != 'N/A']
        node_display = " | ".join(node_display_list) if node_display_list else node_obj.properties.get('ast_node_type', 'N/A')

        print(f"  Node {node_id} ({node_obj.node_type} - {node_display}):")
        out_edges = graph.get_outgoing_edges(node_id)
        if out_edges:
            for edge in out_edges:
                target_node_obj = graph.get_node(edge.target_node_id)
                if not target_node_obj: continue
                target_name_prop = target_node_obj.properties.get('name', '')
                target_src_prop = target_node_obj.properties.get('source_str', target_node_obj.properties.get('ast_node_type', 'N/A'))
                target_display_list = [prop for prop in [target_name_prop, target_src_prop] if prop and prop != 'N/A']
                target_display = " | ".join(target_display_list) if target_display_list else target_node_obj.properties.get('ast_node_type', 'N/A')
                print(f"    -> Edge {edge.id} to {edge.target_node_id} ({target_node_obj.node_type} - {target_display}) [type: {edge.properties.get('flow_type')}]")
        else:
            print(f"    (No outgoing edges)")
    if len(graph.nodes) > len(nodes_to_print):
        print(f"  ... and {len(graph.nodes) - len(nodes_to_print)} more nodes.")
