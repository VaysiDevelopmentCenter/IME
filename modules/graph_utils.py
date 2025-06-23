# modules/graph_utils.py
from .engine import ArchitecturalNode, ArchitecturalEdge, NetworkGraph # Use relative import
import uuid
import random # Though not used yet, likely useful for more complex helpers

def create_instruction_node(graph: NetworkGraph,
                            mnemonic: str,
                            operands: list[str] = None,
                            node_id_prefix: str = "instr_",
                            properties: dict = None) -> ArchitecturalNode:
    """
    Creates an ArchitecturalNode representing an assembly instruction and adds it to the graph.
    Generates a unique ID for the node.
    """
    instr_props = {
        "mnemonic": mnemonic.upper(),
        "operands": operands if operands is not None else []
    }
    # Automatically mark common control flow instructions
    if mnemonic.upper() in ["JMP", "JE", "JNE", "JZ", "JNZ", "JC", "JNC", "JO", "JNO", "JS", "JNS",
                             "JP", "JPE", "JNP", "JPO", "JCXZ", "JECXZ", "JRCXZ",
                             "LOOP", "LOOPE", "LOOPZ", "LOOPNE", "LOOPNZ",
                             "CALL", "RET"]:
        instr_props["is_control_flow"] = True

    if properties: # Allow overriding or adding more specific properties
        instr_props.update(properties)

    # Generate a unique ID
    node_id_val = f"{node_id_prefix}{str(uuid.uuid4())[:8]}"
    while graph.get_node(node_id_val):
        node_id_val = f"{node_id_prefix}{str(uuid.uuid4())[:8]}"

    node = ArchitecturalNode(node_id=node_id_val, node_type="instruction", properties=instr_props)
    graph.add_node(node)
    return node

def create_label_node(graph: NetworkGraph,
                      label_name: str,
                      node_id_prefix: str = "label_") -> ArchitecturalNode:
    """
    Creates an ArchitecturalNode representing an assembly label and adds it to the graph.
    Attempts to create a readable ID based on the label name, ensuring uniqueness.
    """
    label_props = {"label_name": label_name}

    clean_label_name = "".join(c if c.isalnum() or c == '_' else '' for c in label_name) # Basic sanitization
    node_id_val = f"{node_id_prefix}{clean_label_name}"

    suffix_counter = 1
    base_node_id = node_id_val
    while graph.get_node(node_id_val): # Ensure unique ID
        node_id_val = f"{base_node_id}_{suffix_counter}"
        suffix_counter += 1

    node = ArchitecturalNode(node_id=node_id_val, node_type="label", properties=label_props)
    graph.add_node(node)
    return node

def add_sequential_flow_edge(graph: NetworkGraph,
                             source_node: ArchitecturalNode,
                             target_node: ArchitecturalNode,
                             edge_id_prefix: str = "seq_") -> ArchitecturalEdge | None:
    """
    Adds an ArchitecturalEdge representing sequential flow between two nodes.
    Returns the created edge, or the existing one if a similar sequential edge already exists.
    Returns None on failure.
    """
    if not source_node or not target_node:
        print(f"Error: Cannot create sequential edge from {source_node.id if source_node else 'None'} to {target_node.id if target_node else 'None'}. Source or target is None.")
        return None
    if source_node.id not in graph.nodes or target_node.id not in graph.nodes:
        print(f"Error: Source node {source_node.id} or target node {target_node.id} not in graph for sequential edge.")
        return None

    edge_props = {"flow_type": "sequential"}

    # Check if a sequential edge already exists from source to target
    for out_edge_id in graph.adj.get(source_node.id, {}).get('out', []):
        existing_edge = graph.get_edge(out_edge_id)
        if existing_edge and existing_edge.target_node_id == target_node.id and \
           existing_edge.properties.get("flow_type") == "sequential":
            # print(f"Info: Sequential edge from {source_node.id} to {target_node.id} already exists: {existing_edge.id}")
            return existing_edge

    # Create a unique ID for the new edge
    base_edge_id = f"{edge_id_prefix}{source_node.id}_to_{target_node.id}"
    edge_id_val = base_edge_id
    suffix_counter = 1
    while graph.get_edge(edge_id_val):
        edge_id_val = f"{base_edge_id}_{suffix_counter}"
        suffix_counter += 1

    edge = ArchitecturalEdge(source_node_id=source_node.id,
                             target_node_id=target_node.id,
                             edge_id=edge_id_val,
                             properties=edge_props)
    try:
        graph.add_edge(edge)
        return edge
    except ValueError as e:
        print(f"Warning: Could not add sequential edge {source_node.id} to {target_node.id} (ID: {edge_id_val}): {e}")
        return None


def add_control_flow_edge(graph: NetworkGraph,
                          source_node: ArchitecturalNode,
                          target_node: ArchitecturalNode,
                          flow_type: str,
                          edge_id_prefix: str = "ctrl_") -> ArchitecturalEdge | None:
    """
    Adds an ArchitecturalEdge representing control flow (jump, call) between two nodes.
    Ensures the source node is marked as a control flow instruction.
    Generates a unique ID for the edge. Returns None on failure.
    """
    if not source_node or not target_node:
        print(f"Error: Cannot create control flow edge. Source or target is None.")
        return None
    if source_node.id not in graph.nodes or target_node.id not in graph.nodes:
        print(f"Error: Source node {source_node.id} or target node {target_node.id} not in graph for control flow edge.")
        return None

    if source_node.node_type != "instruction" or not source_node.properties.get("is_control_flow"):
        print(f"Warning: Source node {source_node.id} ('{source_node.properties.get('mnemonic')}') is not marked as a control flow instruction, but creating '{flow_type}' edge to {target_node.id}.")
        # Allow creation but warn. Could be made stricter.

    if flow_type not in ["jump_unconditional", "jump_conditional_true", "jump_conditional_false", "call", "return_flow"]: # Added return_flow
        print(f"Warning: Control flow_type '{flow_type}' is not standard. Ensure it's handled by consuming logic.")

    edge_props = {"flow_type": flow_type}

    base_edge_id = f"{edge_id_prefix}{source_node.id}_to_{target_node.id}_{flow_type.replace('_','-')}"
    edge_id_val = base_edge_id
    suffix_counter = 1
    while graph.get_edge(edge_id_val):
        edge_id_val = f"{base_edge_id}_{suffix_counter}"
        suffix_counter += 1

    edge = ArchitecturalEdge(source_node_id=source_node.id,
                             target_node_id=target_node.id,
                             edge_id=edge_id_val,
                             properties=edge_props)
    try:
        graph.add_edge(edge)
        return edge
    except ValueError as e:
        print(f"Warning: Could not add control flow edge {source_node.id} to {target_node.id} (ID: {edge_id_val}): {e}")
        return None
