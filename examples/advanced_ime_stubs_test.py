# examples/advanced_ime_stubs_test.py
import sys
import os
import random # For SRO/FMO stubs that use it

# Add the project root to sys.path to allow importing 'modules'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.engine import (
    MutableEntity,
    MutationEngine, # Though not directly testing its run loop here
    ArchitecturalNode,
    ArchitecturalEdge,
    NetworkGraph,
    PartitionSchema,
    RepartitionGraphOperator,
    PartitionBasedRewireOperator,
    HierarchicalNoiseInjectionFMO,
    SelfSimilarGrowthFMO
)

def build_simple_graph() -> NetworkGraph:
    """Builds a very simple NetworkGraph for testing."""
    graph = NetworkGraph(graph_id="test_graph_1")

    # Add nodes
    n1 = ArchitecturalNode(node_id="n1", node_type="input")
    n2 = ArchitecturalNode(node_id="n2", node_type="hidden", properties={'bias': 0.1})
    n3 = ArchitecturalNode(node_id="n3", node_type="hidden", properties={'bias': -0.2})
    n4 = ArchitecturalNode(node_id="n4", node_type="output")

    nodes = [n1, n2, n3, n4]
    for n in nodes:
        graph.add_node(n)

    # Add edges
    graph.add_edge(ArchitecturalEdge(source_node_id="n1", target_node_id="n2", weight=0.5))
    graph.add_edge(ArchitecturalEdge(source_node_id="n1", target_node_id="n3", weight=0.6))
    graph.add_edge(ArchitecturalEdge(source_node_id="n2", target_node_id="n4", weight=0.7))
    graph.add_edge(ArchitecturalEdge(source_node_id="n3", target_node_id="n4", weight=0.8))
    graph.add_edge(ArchitecturalEdge(source_node_id="n2", target_node_id="n3", weight=0.2)) # A recurrent/internal connection

    return graph

def main():
    print("--- Advanced IME Stubs Smoke Test ---")

    # 1. Test NetworkGraph and component instantiation
    print("\n1. Testing Graph Component Instantiation...")
    graph = build_simple_graph()
    print(f"  Created graph: {graph}")
    print(f"  Node n2: {graph.get_node('n2')}")
    print(f"  Edges from n1: {[str(e) for e in graph.get_outgoing_edges('n1')]}")

    entity = MutableEntity(graph)
    print(f"  Wrapped graph in MutableEntity: {entity.data.id}")

    # 2. Test RepartitionGraphOperator
    print("\n2. Testing RepartitionGraphOperator...")
    repart_op = RepartitionGraphOperator(min_partitions=2, max_partitions=3)
    if repart_op.can_apply(entity):
        print(f"  RepartitionGraphOperator can apply.")
        repart_op.apply(entity)
        print(f"  Graph after Repartition: {entity.data}")
        for i, schema in enumerate(entity.data.partition_schemas):
            print(f"    Schema {i}: {schema.name}, Partitions: {schema.partitions}")
    else:
        print(f"  RepartitionGraphOperator CANNOT apply (unexpected for this test).")


    # 3. Test PartitionBasedRewireOperator
    print("\n3. Testing PartitionBasedRewireOperator...")
    # Ensure there's a schema from previous step, or add one manually for isolated test
    if not entity.data.partition_schemas:
         # Create a simple manual partition if repart_op didn't run or create one
        manual_partitions = [{entity.data.nodes['n1'].id, entity.data.nodes['n2'].id},
                             {entity.data.nodes['n3'].id, entity.data.nodes['n4'].id}]
        manual_schema = PartitionSchema(name="ManualTestSchema", target_element_type="nodes", partitions=manual_partitions)
        entity.data.add_partition_schema(manual_schema)
        print(f"  Added manual schema for rewire test: {manual_schema.name}")

    rewire_op = PartitionBasedRewireOperator(connection_density=0.5) # Higher density for more chance of edge creation
    if rewire_op.can_apply(entity):
        print(f"  PartitionBasedRewireOperator can apply.")
        rewire_op.apply(entity)
        print(f"  Graph after Rewire (edges might have changed): Edges count = {len(entity.data.edges)}")
    else:
        print(f"  PartitionBasedRewireOperator CANNOT apply (unexpected if schemas exist).")

    # 4. Test HierarchicalNoiseInjectionFMO
    print("\n4. Testing HierarchicalNoiseInjectionFMO...")
    noise_op = HierarchicalNoiseInjectionFMO(target_scales=["micro", "meso", "macro", "meta"], base_noise_level=0.1)
    if noise_op.can_apply(entity):
        print(f"  HierarchicalNoiseInjectionFMO can apply.")
        original_weights = {e.id: e.weight for e in entity.data.edges.values()}
        noise_op.apply(entity)
        print(f"  Graph after Noise Injection: Edges count = {len(entity.data.edges)}")
        # Check if some weights changed
        weights_changed = 0
        for e in entity.data.edges.values():
            if e.id in original_weights and original_weights[e.id] != e.weight:
                weights_changed +=1
        print(f"    Number of edge weights changed by micro noise: {weights_changed} (expected > 0 if edges exist)")

    else:
        print(f"  HierarchicalNoiseInjectionFMO CANNOT apply (unexpected).")

    # 5. Test SelfSimilarGrowthFMO
    print("\n5. Testing SelfSimilarGrowthFMO...")
    growth_op = SelfSimilarGrowthFMO(growth_complexity=1, min_motif_size=2, max_motif_size=3)
    if growth_op.can_apply(entity):
        print(f"  SelfSimilarGrowthFMO can apply.")
        original_node_count = len(entity.data.nodes)
        original_edge_count = len(entity.data.edges)
        growth_op.apply(entity)
        print(f"  Graph after Growth: Nodes={len(entity.data.nodes)}, Edges={len(entity.data.edges)}")
        if len(entity.data.nodes) > original_node_count:
            print(f"    Nodes were added by growth operator (expected > 0).")
        if len(entity.data.edges) > original_edge_count:
            print(f"    Edges were added by growth operator (expected > 0).")

    else:
        print(f"  SelfSimilarGrowthFMO CANNOT apply (check min_motif_size vs graph size).")
        print(f"  (Graph node count: {len(entity.data.nodes)}, min_motif_size: {growth_op.min_motif_size})")


    print("\n--- Advanced IME Stubs Smoke Test Complete ---")

if __name__ == "__main__":
    main()
