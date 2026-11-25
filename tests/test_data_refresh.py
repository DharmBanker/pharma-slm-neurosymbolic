"""
Unit Tests & Red-Teaming for Continuous Knowledge Freshness & Temporal Validity
================================================================================
Tests dynamic recall ingestion, incremental PubMed vector updating,
temporal edge expiration filtering, and adversarial edge cases.
"""
import unittest
import sys
import os

# Ensure src/ is on path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from primekg_loader import PrimeKGGraph, load_high_density_primekg
from vector_store import BiomedicalVectorEngine, REAL_CLINICAL_CORPUS, EvidenceLevel
from guardrail import NeurosymbolicGuardrail
from data_refresh import KnowledgeFreshnessEngine


class TestKnowledgeFreshnessEngine(unittest.TestCase):
    def setUp(self):
        # Create isolated instances for reproducible test state
        self.kg = load_high_density_primekg()
        # Shallow copy corpus so dynamic additions don't bleed across tests
        self.vector_engine = BiomedicalVectorEngine(passages=list(REAL_CLINICAL_CORPUS))
        self.guardrail = NeurosymbolicGuardrail(kg=self.kg)
        self.engine = KnowledgeFreshnessEngine(kg=self.kg, vector_store=self.vector_engine)

    def test_preloaded_recalls_intercepted(self):
        """Pre-registered FDA recalls (Ranitidine) must be deterministically blocked."""
        # Test Ranitidine recall
        recall_info = self.kg.is_recalled("Ranitidine")
        self.assertIsNotNone(recall_info)
        self.assertEqual(recall_info["status"], "RECALLED")
        self.assertIn("NDMA", recall_info["reason"])

        # Test Guardrail interception
        audit = self.guardrail.audit_clinical_response(
            drug_name="Ranitidine",
            user_query="Can I take Ranitidine 150 mg for acid reflux?",
            proposed_response="Take Ranitidine 150 mg twice daily.",
            graph_path_found=False
        )
        self.assertEqual(audit["decision"], "BLOCKED")
        self.assertEqual(audit["trigger"], "DRUG_RECALL_INTERCEPTED")
        self.assertIn("WITHDRAWN_FROM_MARKET", audit["safe_response"])
        print("\n[PASS] Preloaded Ranitidine FDA recall deterministically intercepted.")

    def test_dynamic_fda_recall_ingestion(self):
        """Dynamically ingesting a new FDA recall must immediately block prescribing."""
        drug = "Terfenadine"
        # Before recall: drug is not in recall registry
        self.assertIsNone(self.kg.is_recalled(drug))

        # Dynamically ingest FDA recall
        result = self.engine.ingest_fda_recall(
            drug_name=drug,
            reason="Fatal cardiac arrhythmias (QT prolongation and torsades de pointes)",
            recall_date="1998-02-27",
            regulatory_authority="FDA"
        )
        self.assertEqual(result["status"], "ENFORCED")
        self.assertEqual(result["drug"], drug)

        # Immediate verification in KG
        recall_info = self.kg.is_recalled(drug)
        self.assertIsNotNone(recall_info)
        self.assertEqual(recall_info["status"], "RECALLED")

        # Immediate verification in Guardrail
        audit = self.guardrail.audit_clinical_response(
            drug_name=drug,
            user_query="Prescribe Terfenadine 60 mg for seasonal allergic rhinitis",
            proposed_response="Terfenadine 60 mg orally twice daily.",
            graph_path_found=False
        )
        self.assertEqual(audit["decision"], "BLOCKED")
        self.assertEqual(audit["trigger"], "DRUG_RECALL_INTERCEPTED")
        self.assertIn("torsades de pointes", audit["safe_response"])
        print("[PASS] Dynamic FDA recall ingestion immediately enforced by guardrail.")

    def test_incremental_pubmed_literature_ingestion(self):
        """Newly published clinical trials must become searchable immediately without index restart."""
        # Step 1: Query for Tirzepatide before ingestion -> should return nothing (unindexed entity)
        initial_search = self.vector_engine.search("tirzepatide mounjaro surpass-2", top_k=2)
        self.assertEqual(len(initial_search), 0)

        # Step 2: Dynamically ingest SURPASS-2 Trial (NEJM 2021)
        res = self.engine.ingest_pubmed_trial(
            pmid="34170649",
            title="Tirzepatide versus Semaglutide Once Weekly in Patients with Type 2 Diabetes (SURPASS-2)",
            abstract="In the SURPASS-2 phase 3 open-label trial, Tirzepatide (dual GIP and GLP-1 receptor agonist) at 5mg, 10mg, and 15mg was noninferior and superior to Semaglutide 1mg with respect to HbA1c reduction and weight loss.",
            source="New England Journal of Medicine (NEJM) / PMID:34170649",
            evidence_type="PHASE3_RCT",
            evidence_weight=EvidenceLevel.PHASE3_RCT,
            keywords=["tirzepatide", "mounjaro", "gip", "glp-1", "surpass-2", "diabetes"],
            kg_relations=[
                ("Tirzepatide", "DUAL_AGONIST", "GLP1_and_GIP_Receptors", "drug", "gene/protein", "PMID:34170649"),
                ("GLP1_and_GIP_Receptors", "TREATS", "Type_2_Diabetes", "gene/protein", "disease", "PMID:34170649")
            ]
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["kg_edges_added"], 2)

        # Step 3: Search again -> must now retrieve Tirzepatide as top result!
        updated_search = self.vector_engine.search("tirzepatide mounjaro gip surpass-2", top_k=1)
        self.assertEqual(len(updated_search), 1)
        top_match = updated_search[0]
        self.assertEqual(top_match["id"], "PUBMED_34170649")
        self.assertIn("SURPASS-2", top_match["title"])
        self.assertGreater(top_match["blended_score"], 0.60)

        # Step 4: Verify symbolic multi-hop path exists in PrimeKG
        path = self.kg.find_multi_hop_path("Tirzepatide", "Type_2_Diabetes")
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)  # Tirzepatide -> Receptors -> Type_2_Diabetes
        print("[PASS] Incremental PubMed ingestion updated both vector search & multi-hop KG paths.")

    def test_temporal_validity_auditing(self):
        """Auditing must identify expired and stale guidelines while retaining fresh evidence."""
        # Add an edge with explicit expired validity
        self.kg.add_edge(
            source="OldTherapyX",
            relation="TREATS",
            target="ConditionY",
            source_type="drug",
            target_type="disease",
            citation="Trial_1995",
            valid_until=2018,
            citation_year=1995,
            status="EXPIRED"
        )

        # Add an edge with old citation year (> 5 years ago from 2026)
        self.kg.add_edge(
            source="LegacyGuidelineA",
            relation="RECOMMENDS",
            target="ProtocolB",
            source_type="guideline",
            target_type="protocol",
            citation="Guideline_2017",
            citation_year=2017,
            status="ACTIVE"
        )

        # Add a fresh 2025 guideline
        self.kg.add_edge(
            source="ModernDrugC",
            relation="TREATS",
            target="ModernConditionD",
            source_type="drug",
            target_type="disease",
            citation="Trial_2025",
            citation_year=2025,
            status="ACTIVE"
        )

        audit = self.engine.audit_temporal_validity(current_year=2026, max_staleness_years=5)
        self.assertGreaterEqual(audit["expired_edges"], 1)
        self.assertGreaterEqual(audit["stale_edges_flagged"], 1)
        self.assertGreaterEqual(audit["active_valid_edges"], 1)
        print(f"[PASS] Temporal validity audit: {audit['expired_edges']} expired, {audit['stale_edges_flagged']} stale caught.")

    def test_adversarial_and_edge_cases(self):
        """Red-teaming: malformed dates, empty passages, and duplicate recalls."""
        # 1. Malformed recall date
        res = self.engine.ingest_fda_recall("ObscureDrugZ", reason="Unknown toxicity", recall_date="MALFORMED_DATE")
        self.assertEqual(res["status"], "ENFORCED")
        self.assertIsNotNone(self.kg.is_recalled("ObscureDrugZ"))

        # 2. Case-insensitive recall lookup
        self.assertIsNotNone(self.kg.is_recalled("obscuredrugz"))
        self.assertIsNotNone(self.kg.is_recalled("  OBSCUREDRUGZ  "))

        # 3. Empty text passage ingestion
        res_empty = self.engine.ingest_pubmed_trial(
            pmid="000000",
            title="",
            abstract="",
            source="EmptySource",
            keywords=[]
        )
        self.assertEqual(res_empty["status"], "SUCCESS")

        # 4. Telemetry health check
        health = self.engine.run_system_health_audit()
        self.assertEqual(health["status"], "OPERATIONAL")
        self.assertGreaterEqual(health["knowledge_graph"]["recalled_drugs_registered"], 3)
        print("[PASS] Adversarial edge cases and system health audit passed cleanly.")


if __name__ == "__main__":
    unittest.main()
