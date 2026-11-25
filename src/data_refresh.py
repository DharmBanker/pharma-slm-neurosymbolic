"""
Continuous Knowledge Freshness & Temporal Validity Engine for pharma-slm
========================================================================
Implements Fix #7 of the Pharma-SLM Alignment Roadmap:
Automates dynamic knowledge ingestion, temporal edge freshness auditing,
and regulatory drug recall synchronization without model retraining.

Key Capabilities:
1. FDA Enforcement & Safety Withdrawal Ingestion:
   - Synchronizes FDA drug recall notices and black-box safety warnings.
   - Deterministically prevents revoked therapeutics from being recommended.
2. Incremental PubMed Literature Ingestion:
   - Ingests newly published Phase III clinical trials and clinical practice guidelines.
   - Dynamically updates dense vector embeddings without cold-start downtime.
3. Multi-Hop Temporal Validity Auditing:
   - Tracks edge validity lifetimes (valid_until / citation_year).
   - Flags superseded or outdated medical guidelines for clinical re-verification.
"""
from __future__ import annotations
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primekg_loader import PrimeKGGraph, get_primekg_instance
from vector_store import BiomedicalVectorEngine, get_default_vector_engine, EvidenceLevel


class KnowledgeFreshnessEngine:
    """
    Orchestrates continuous synchronization and temporal validity auditing
    across symbolic (PrimeKG) and dense (BiomedicalVectorEngine) indices.
    """
    def __init__(
        self,
        kg: Optional[PrimeKGGraph] = None,
        vector_store: Optional[BiomedicalVectorEngine] = None
    ):
        self.kg = kg or get_primekg_instance()
        self.vector_store = vector_store or get_default_vector_engine()
        self.refresh_log: List[Dict[str, Any]] = []

    def ingest_fda_recall(
        self,
        drug_name: str,
        reason: str,
        recall_date: str = "2026-01-15",
        regulatory_authority: str = "FDA_CDER"
    ) -> Dict[str, Any]:
        """
        Dynamically registers an FDA recall/withdrawal notice.
        Updates the PrimeKG node status and injects deterministic warning edges.
        """
        self.kg.mark_recalled(drug_name, reason=reason, recall_date=recall_date)
        
        canonical = self.kg.get_canonical_name(drug_name) or drug_name
        # Add symbolic warning edge in graph
        self.kg.add_edge(
            source=canonical,
            relation="WITHDRAWN_DUE_TO",
            target=f"Toxicity_Risk_{reason.replace(' ', '_')[:35]}",
            source_type="drug",
            target_type="safety_signal",
            citation=f"{regulatory_authority}:Enforcement_{recall_date}",
            citation_year=int(recall_date.split("-")[0]) if "-" in recall_date else 2026,
            status="RECALLED"
        )

        record = {
            "timestamp": time.time(),
            "type": "REGULATORY_RECALL",
            "drug": canonical,
            "reason": reason,
            "recall_date": recall_date,
            "authority": regulatory_authority,
            "status": "ENFORCED"
        }
        self.refresh_log.append(record)
        return record

    def ingest_pubmed_trial(
        self,
        pmid: str,
        title: str,
        abstract: str,
        source: str,
        evidence_type: str = "PHASE3_RCT",
        evidence_weight: float = EvidenceLevel.PHASE3_RCT,
        keywords: Optional[List[str]] = None,
        kg_relations: Optional[List[Tuple[str, str, str, str, str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Incrementally ingests a new clinical study into both the dense vector store
        and (optionally) the symbolic knowledge graph.
        """
        # 1. Format and ingest into dense vector index
        kws = keywords or []
        passage_record = {
            "id": f"PUBMED_{pmid}",
            "title": title,
            "source": source,
            "evidence_type": evidence_type,
            "evidence_weight": evidence_weight,
            "text": abstract,
            "keywords": kws
        }
        self.vector_store.add_passage(passage_record)

        # 2. Add symbolic relations to PrimeKG if provided
        added_edges = 0
        if kg_relations:
            for src, rel, tgt, src_type, tgt_type, cit in kg_relations:
                self.kg.add_edge(
                    source=src,
                    relation=rel,
                    target=tgt,
                    source_type=src_type,
                    target_type=tgt_type,
                    citation=cit,
                    citation_year=2026,
                    status="ACTIVE"
                )
                added_edges += 1

        record = {
            "timestamp": time.time(),
            "type": "PUBMED_LITERATURE_INGESTION",
            "pmid": pmid,
            "title": title,
            "vector_passages_total": len(self.vector_store.passages),
            "kg_edges_added": added_edges,
            "status": "SUCCESS"
        }
        self.refresh_log.append(record)
        return record

    def audit_temporal_validity(
        self,
        current_year: int = 2026,
        max_staleness_years: int = 5
    ) -> Dict[str, Any]:
        """
        Audits all edges in the Knowledge Graph for temporal expiration or guideline staleness.
        Flags relations older than max_staleness_years for clinical panel review.
        """
        total_audited = 0
        expired_count = 0
        stale_count = 0
        active_count = 0
        flagged_edges = []

        for (src, rel, tgt), meta in self.kg.edge_metadata.items():
            total_audited += 1
            valid_until = meta.get("valid_until")
            citation_year = meta.get("citation_year")
            status = meta.get("status", "ACTIVE")

            if valid_until is not None and valid_until < current_year:
                expired_count += 1
                flagged_edges.append({
                    "edge": f"{src} -> {rel} -> {tgt}",
                    "reason": f"Explicit validity expired in {valid_until}",
                    "citation": meta.get("citation")
                })
            elif citation_year is not None and (current_year - citation_year) > max_staleness_years:
                stale_count += 1
                flagged_edges.append({
                    "edge": f"{src} -> {rel} -> {tgt}",
                    "reason": f"Published in {citation_year} (> {max_staleness_years} years old)",
                    "citation": meta.get("citation")
                })
            else:
                active_count += 1

        return {
            "current_year": current_year,
            "total_edges_audited": total_audited,
            "active_valid_edges": active_count,
            "expired_edges": expired_count,
            "stale_edges_flagged": stale_count,
            "flagged_edges_sample": flagged_edges[:5]
        }

    def run_system_health_audit(self) -> Dict[str, Any]:
        """
        Returns full operational status and telemetry for the knowledge backbone.
        """
        kg_stats = self.kg.get_stats()
        return {
            "status": "OPERATIONAL",
            "knowledge_graph": {
                "total_nodes": kg_stats["total_nodes"],
                "total_edges": kg_stats["total_edges"],
                "node_types": kg_stats["node_types"],
                "recalled_drugs_registered": len(self.kg.recalls)
            },
            "vector_store": {
                "total_passages": len(self.vector_store.passages),
                "backend": self.vector_store.backend,
                "embedding_dim": self.vector_store.dim
            },
            "recent_refreshes": len(self.refresh_log)
        }


def get_freshness_engine() -> KnowledgeFreshnessEngine:
    """Helper singleton factory."""
    return KnowledgeFreshnessEngine()

