# examples/assembly_graph_demo.py
import sys
import os

# Add the project root to sys.path to allow importing 'modules'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.engine import NetworkGraph # ArchitecturalNode, ArchitecturalEdge are used by helpers
from modules.graph_utils import (
    create_instruction_node,
    create_label_node,
    add_sequential_flow_edge,
    add_control_flow_edge
)

def main():
    print("--- Assembly Language to NetworkGraph Demo ---")

    # Create an empty NetworkGraph
    asm_graph = NetworkGraph(graph_id="simple_asm_snippet_graph")
    print(f"Created NetworkGraph: {asm_graph.id}\n")

    # --- Define and add nodes (labels and instructions) ---
    print("1. Adding Nodes (Labels and Instructions):")

    # Labels
    l_start = create_label_node(asm_graph, "START")
    l_target = create_label_node(asm_graph, "TARGET_LABEL")
    l_end = create_label_node(asm_graph, "END_LABEL")
    print(f"  Added Label: {l_start}")
    print(f"  Added Label: {l_target}")
    print(f"  Added Label: {l_end}")

    # Instructions
    # START:
    #   MOV EAX, 10
    instr1_mov = create_instruction_node(asm_graph, "MOV", ["EAX", "10"])
    print(f"  Added Instruction: {instr1_mov}")

    #   ADD EAX, 5
    instr2_add = create_instruction_node(asm_graph, "ADD", ["EAX", "5"])
    print(f"  Added Instruction: {instr2_add}")

    #   CMP EAX, 15
    instr3_cmp = create_instruction_node(asm_graph, "CMP", ["EAX", "15"])
    print(f"  Added Instruction: {instr3_cmp}")

    #   JE  TARGET_LABEL
    instr4_je = create_instruction_node(asm_graph, "JE", ["TARGET_LABEL"])
    print(f"  Added Instruction: {instr4_je}")

    #   MOV EBX, EAX  (fall-through path)
    instr5_mov_fallthrough = create_instruction_node(asm_graph, "MOV", ["EBX", "EAX"])
    print(f"  Added Instruction: {instr5_mov_fallthrough}")

    #   JMP END_LABEL
    instr6_jmp_end = create_instruction_node(asm_graph, "JMP", ["END_LABEL"])
    print(f"  Added Instruction: {instr6_jmp_end}")

    # TARGET_LABEL:
    #   MOV EBX, 20
    instr7_mov_target = create_instruction_node(asm_graph, "MOV", ["EBX", "20"])
    print(f"  Added Instruction: {instr7_mov_target}")

    # END_LABEL:
    #   NOP
    instr8_nop = create_instruction_node(asm_graph, "NOP")
    print(f"  Added Instruction: {instr8_nop}")

    # --- Define and add edges (control flow) ---
    print("\n2. Adding Edges (Sequential and Control Flow):")

    # Sequential flow from labels to first instruction if applicable
    # (Here, START label points to instr1_mov)
    add_sequential_flow_edge(asm_graph, l_start, instr1_mov)

    # Sequential flow between instructions
    add_sequential_flow_edge(asm_graph, instr1_mov, instr2_add)
    add_sequential_flow_edge(asm_graph, instr2_add, instr3_cmp)
    add_sequential_flow_edge(asm_graph, instr3_cmp, instr4_je)

    # Conditional jump (JE)
    # Fall-through path for JE (instr4_je to instr5_mov_fallthrough)
    add_sequential_flow_edge(asm_graph, instr4_je, instr5_mov_fallthrough)
    # Taken path for JE (instr4_je to l_target)
    add_control_flow_edge(asm_graph, instr4_je, l_target, "jump_conditional_true")

    # Sequential flow after fall-through
    add_sequential_flow_edge(asm_graph, instr5_mov_fallthrough, instr6_jmp_end)

    # Unconditional jump (JMP)
    add_control_flow_edge(asm_graph, instr6_jmp_end, l_end, "jump_unconditional")

    # Flow from TARGET_LABEL to its instruction
    add_sequential_flow_edge(asm_graph, l_target, instr7_mov_target)
    # Sequential flow after target instruction (instr7_mov_target to l_end, assuming it flows to common end)
    # Or, more directly, instr7_mov_target could jump to l_end or just end. Let's make it flow to l_end.
    add_sequential_flow_edge(asm_graph, instr7_mov_target, l_end)

    # Flow from END_LABEL to its instruction
    add_sequential_flow_edge(asm_graph, l_end, instr8_nop)
    # Potentially an edge from instr8_nop if it's not the absolute end (e.g., to a RET or another JMP)
    # For this example, instr8_nop is the last instruction.

    # --- Print Graph Summary ---
    print("\n3. Graph Summary:")
    print(f"  Total Nodes: {len(asm_graph.nodes)}")
    print(f"  Total Edges: {len(asm_graph.edges)}")

    print("\n  Nodes List:")
    for node_id, node in asm_graph.nodes.items():
        print(f"    {node}")

    print("\n  Edges List:")
    for edge_id, edge in asm_graph.edges.items():
        print(f"    {edge}")

    print("\n  Adjacency List (Outgoing):")
    for node_id in asm_graph.nodes.keys():
        outgoing_edge_descs = []
        for edge_id in asm_graph.adj.get(node_id, {}).get('out', []):
            edge = asm_graph.get_edge(edge_id)
            if edge:
                 outgoing_edge_descs.append(f"-> {edge.target_node_id} (type: {edge.properties.get('flow_type')})")
        if outgoing_edge_descs:
            print(f"    {node_id}: {', '.join(outgoing_edge_descs)}")
        else:
            print(f"    {node_id}: (No outgoing edges)")


    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    main()
