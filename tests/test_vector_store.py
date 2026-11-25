"""
Unit Tests & Red-Teaming Suite for BiomedicalVectorEngine
==========================================================
Tests every edge case and intentionally attempts to break the retrieval engine:
1. Empty and whitespace-only queries (Zero-division check)
2. Out-of-vocabulary gibberish queries
3. Authority score re-ranking verification (FDA Guideline > Case Report)
4. Out-of-bounds top_k requests (top_k > corpus size)
5. Metric bounds check (all scores must be within [0.0, 1.0])
6. Multi-domain clinical retrieval (Oncology, Cardiology, Diabetes, Virology)
7. Tensor dot-product latency stress test (1,000 queries)
"""
import os
import sys
import time
import unittest

# Ensure src/ is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from vector_store import BiomedicalVectorEngine, EvidenceLevel, REAL_CLINICAL_CORPUS


class TestBiomedicalVectorEngine(unittest.TestCase):
    def setUp(self):
        self.engine = BiomedicalVectorEngine()

    def test_engine_initialization(self):
        """Verify engine loads corpus, builds tensor embeddings, and identifies backend."""
        stats = self.engine.get_stats()
        self.assertGreaterEqual(stats["total_passages"], 10, "Corpus must have multiple clinical passages")
        self.assertEqual(stats["embedding_dimension"], 64, "Embedding dimension must match configuration")
        self.assertIn("device", stats)
        print(f"\n[PASS] Engine Initialized: {stats['total_passages']} passages, Backend: {stats['backend']}")

    def test_empty_and_whitespace_queries(self):
        """RED TEAM: Try to break the engine with empty string and spaces."""
        self.assertEqual(self.engine.search(""), [])
        self.assertEqual(self.engine.search("     "), [])
        self.assertEqual(self.engine.search("\t\n"), [])
        print("[PASS] Empty and whitespace queries handled gracefully with zero errors.")

    def test_gibberish_oov_query(self):
        """RED TEAM: Query with complete gibberish must return empty list without crash."""
        hits = self.engine.search("asdkfjhasdf9823471092384kjlsdf")
        self.assertEqual(hits, [], "Gibberish query must return empty list")
        print("[PASS] Out-of-vocabulary gibberish query safely returned empty.")

    def test_top_k_exceeding_corpus(self):
        """RED TEAM: Requesting top_k=50 when corpus has only 13 passages."""
        hits = self.engine.search("covid baricitinib oxygen", top_k=50)
        self.assertLessEqual(len(hits), len(self.engine.passages))
        self.assertGreater(len(hits), 0)
        print(f"[PASS] Handled top_k > corpus size safely (returned {len(hits)} passages).")

    def test_score_bounds_and_hierarchy(self):
        """RED TEAM: Verify all blended scores lie in [0.0, 1.0]."""
        hits = self.engine.search("metformin egfr kidney lactic acidosis", top_k=5)
        self.assertGreater(len(hits), 0)
        for h in hits:
            self.assertGreaterEqual(h["semantic_score"], 0.0)
            self.assertLessEqual(h["semantic_score"], 1.0)
            self.assertGreaterEqual(h["blended_score"], 0.0)
            self.assertLessEqual(h["blended_score"], 1.0)
        print("[PASS] Score mathematical boundaries strictly preserved within [0.0, 1.0].")

    def test_authority_reranking_behavior(self):
        """
        CRITICAL TEST: Ensure FDA Guideline ranks higher than low-authority evidence.
        Create an artificial corpus with a Case Report and an FDA Guideline.
        """
        mock_corpus = [
            {
                "id": "CASE_REPORT_01",
                "title": "Case Report on Drug X in Asthma",
                "source": "Case Reports in Pulmonology",
                "evidence_type": "CASE_REPORT",
                "evidence_weight": EvidenceLevel.CASE_REPORT,  # 0.20
                "text": "A 45-year-old male was treated with Drug X for acute asthma.",
                "keywords": ["drug x", "asthma"]
            },
            {
                "id": "FDA_GUIDELINE_01",
                "title": "FDA Prescribing Information for Drug X",
                "source": "FDA Guidance",
                "evidence_type": "FDA_GUIDELINE",
                "evidence_weight": EvidenceLevel.FDA_GUIDELINE,  # 1.0
                "text": "Drug X is officially approved for severe asthma maintenance.",
                "keywords": ["drug x", "asthma"]
            }
        ]
        test_engine = BiomedicalVectorEngine(passages=mock_corpus)
        hits = test_engine.search("Drug X asthma", top_k=2)
        
        self.assertEqual(len(hits), 2)
        # FDA guideline MUST be ranked #1 because of authority weighting
        self.assertEqual(hits[0]["id"], "FDA_GUIDELINE_01", "FDA Guideline must outrank Case Report!")
        self.assertEqual(hits[1]["id"], "CASE_REPORT_01")
        print("[PASS] Evidence Authority Hierarchy successfully prioritizes FDA Guidelines over Case Reports.")

    def test_multi_domain_clinical_retrieval(self):
        """Verify targeted retrieval across distinct therapeutic areas."""
        queries = [
            ("semaglutide ozempic glp-1 weekly dose", "PUBMED_33567185"),
            ("osimertinib tagrisso egfr non-small cell lung cancer", "PUBMED_29151359"),
            ("apixaban eliquis afib stroke aristotle", "PUBMED_21875378"),
            ("paxlovid nirmatrelvir covid-19 trial", "PUBMED_35172054")
        ]
        for q, expected_id in queries:
            hits = self.engine.search(q, top_k=1)
            self.assertGreater(len(hits), 0, f"Query returned no hits for: {q}")
            self.assertEqual(hits[0]["id"], expected_id, f"Wrong passage retrieved for: {q}")
            print(f"  * Query: '{q[:35]}...' -> Retrieved: [{hits[0]['id']}] {hits[0]['source'][:25]} [OK]")
        print("[PASS] Multi-domain clinical retrieval verified.")

    def test_retrieval_speed_benchmark(self):
        """RED TEAM: Stress test 1,000 dense searches to measure tensor dot-product latency."""
        t0 = time.time()
        n_queries = 1000
        for _ in range(n_queries):
            _ = self.engine.search("baricitinib covid trial recovery", top_k=2)
        total_time = time.time() - t0
        avg_latency_ms = (total_time / n_queries) * 1000

        print(f"\n[BENCHMARK] Executed {n_queries} Dense Vector searches in {total_time:.3f}s")
        print(f"[BENCHMARK] Average Latency per Search: {avg_latency_ms:.4f} ms")
        self.assertLess(avg_latency_ms, 1.0, "Average latency must be under 1 millisecond!")
        print("[PASS] Sub-millisecond dense retrieval performance achieved.")


if __name__ == "__main__":
    unittest.main()

