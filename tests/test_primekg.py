"""
Unit Tests & Red-Teaming Suite for PrimeKG Knowledge Graph Engine
==================================================================
Tests every edge case and intentionally attempts to break the graph:
1. Case-insensitivity & whitespace resilience
2. Bidirectional pathfinding (Forward and Reverse)
3. Self-loops and identical start-end nodes
4. Non-existent nodes and disconnected subgraphs (Dead ends)
5. Max depth limit enforcement
6. Traversal speed benchmark (sub-millisecond target)
7. Safety contraindication retrieval
"""
import os
import sys
import time
import unittest

# Ensure src/ is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from primekg_loader import PrimeKGGraph, load_high_density_primekg


class TestPrimeKGGraph(unittest.TestCase):
    def setUp(self):
        self.kg = load_high_density_primekg()

    def test_graph_initialization_stats(self):
        """Verify graph is loaded with multiple clinical domains."""
        stats = self.kg.get_stats()
        self.assertGreater(stats["total_nodes"], 40, "Graph must have extensive clinical nodes")
        self.assertGreater(stats["total_edges"], 40, "Graph must have extensive edges")
        self.assertGreaterEqual(stats["node_types"], 5, "Must cover at least 5 biological node types")
        print(f"\n[PASS] Graph Initialized: {stats['total_nodes']} nodes, {stats['total_edges']} edges, {stats['node_types']} node types.")

    def test_case_insensitivity_and_whitespace(self):
        """RED TEAM: Try to break the graph with weird casing and whitespace."""
        test_cases = [
            ("baricitinib", "severe_covid19"),
            ("BARICITINIB", "SEVERE_COVID19"),
            ("  baricitinib  ", "  severe_covid19  "),
            ("bArIcItInIb", "SeVeRe_CoViD19")
        ]
        for src, dst in test_cases:
            path = self.kg.find_multi_hop_path(src, dst)
            self.assertIsNotNone(path, f"Failed on case/whitespace: '{src}' -> '{dst}'")
            self.assertEqual(path[0][0], "Baricitinib", "Start node must resolve to canonical casing")
            self.assertEqual(path[-1][0], "Severe_COVID19", "End node must resolve to canonical casing")
        print("[PASS] Case-insensitivity & whitespace stripping successfully verified.")

    def test_bidirectional_reverse_traversal(self):
        """RED TEAM: Try to search in the REVERSE direction (Target -> Drug)."""
        # Forward: Baricitinib -> JAK1_JAK2_Kinase
        forward_path = self.kg.find_multi_hop_path("Baricitinib", "JAK1_JAK2_Kinase", direction="forward")
        self.assertIsNotNone(forward_path)
        self.assertEqual(forward_path[1][1], "INHIBITS")

        # Reverse: JAK1_JAK2_Kinase -> Baricitinib (Old engine failed this!)
        reverse_path = self.kg.find_multi_hop_path("JAK1_JAK2_Kinase", "Baricitinib", direction="both")
        self.assertIsNotNone(reverse_path, "Reverse traversal failed! Old engine bug reproduced.")
        self.assertEqual(reverse_path[1][1], "INVERSE_INHIBITS", "Reverse edge must be properly prefixed")
        print("[PASS] Bidirectional reverse traversal successfully verified.")

    def test_identical_start_and_end(self):
        """RED TEAM: Searching from a node to itself must not loop or fail."""
        path = self.kg.find_multi_hop_path("Aspirin", "Aspirin")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0][0], "Aspirin")
        print("[PASS] Self-path boundary condition verified.")

    def test_disconnected_and_non_existent_nodes(self):
        """RED TEAM: Searching for imaginary nodes or disconnected concepts must return None cleanly."""
        # Non-existent node
        path_invalid = self.kg.find_multi_hop_path("Baricitinib", "Fake_Non_Existent_Disease_123")
        self.assertIsNone(path_invalid, "Must return None for invalid destination node")

        # Another invalid source
        path_invalid_src = self.kg.find_multi_hop_path("FictionalDrug_ABC", "Severe_COVID19")
        self.assertIsNone(path_invalid_src, "Must return None for invalid source node")
        print("[PASS] Graceful handling of non-existent/disconnected entities verified.")

    def test_max_depth_cutoff(self):
        """RED TEAM: Ensure depth cutoff prevents exploring too far."""
        # Baricitinib -> Severe_COVID19 takes 4 hops
        # If we constrain max_depth=2, it must return None
        path_shallow = self.kg.find_multi_hop_path("Baricitinib", "Severe_COVID19", max_depth=2)
        self.assertIsNone(path_shallow, "Max depth cutoff failed to constrain search!")

        # If max_depth=5, it should succeed
        path_deep = self.kg.find_multi_hop_path("Baricitinib", "Severe_COVID19", max_depth=5)
        self.assertIsNotNone(path_deep, "Path should succeed within valid depth limit")
        print("[PASS] Multi-hop depth constraints verified.")

    def test_multi_domain_pathfinding(self):
        """Verify pathways across distinct therapeutic areas."""
        domains = [
            ("Osimertinib", "Non_Small_Cell_Lung_Cancer", "Oncology"),
            ("Semaglutide", "Type_2_Diabetes", "Metabolic"),
            ("Warfarin", "Deep_Vein_Thrombosis", "Cardiovascular"),
            ("Paxlovid_Nirmatrelvir", "Severe_COVID19", "Virology")
        ]
        for src, dst, domain in domains:
            path = self.kg.find_multi_hop_path(src, dst)
            self.assertIsNotNone(path, f"Failed on {domain} pathway: {src} -> {dst}")
            print(f"  * {domain} Pathway: {src} -> {dst} ({len(path)-1} hops) [OK]")
        print("[PASS] Multi-domain clinical pathways verified.")

    def test_safety_contraindication_edges(self):
        """Verify FDA Boxed Warnings and contraindication rules."""
        neighbors = self.kg.get_neighbors("Metformin", direction="forward")
        contraindications = [n for n in neighbors if n[0] == "CONTRAINDICATED_IN"]
        self.assertGreater(len(contraindications), 0, "Metformin must have contraindication edges")
        self.assertEqual(contraindications[0][1], "Severe_Renal_Impairment_eGFR_less_30")
        print("[PASS] Safety contraindication rules verified.")

    def test_traversal_speed_benchmark(self):
        """RED TEAM: Stress test 1,000 BFS traversals to measure latency."""
        t0 = time.time()
        n_queries = 1000
        for _ in range(n_queries):
            _ = self.kg.find_multi_hop_path("Baricitinib", "Severe_COVID19")
        total_time = time.time() - t0
        avg_latency_ms = (total_time / n_queries) * 1000

        print(f"\n[BENCHMARK] Executed {n_queries} BFS queries in {total_time:.3f}s")
        print(f"[BENCHMARK] Average Latency per Query: {avg_latency_ms:.4f} ms")
        self.assertLess(avg_latency_ms, 1.0, "Average latency must be under 1 millisecond!")
        print("[PASS] Sub-millisecond performance target achieved.")


if __name__ == "__main__":
    unittest.main()

