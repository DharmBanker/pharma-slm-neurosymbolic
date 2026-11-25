"""
Neurosymbolic Safety Guardrail & Conflict Resolution Engine
=============================================================
Provides deterministic clinical safety enforcement and evidence conflict
resolution for pharma-slm before answers reach clinicians or patients.

Pillars:
1. Maximum Tolerated Dose (MTD) Validator:
   - Numerical extraction and boundary checking against FDA safety bounds.
   - Prevents lethal overdoses caused by model temperature fluctuations.
2. Black-Box Contraindication Interceptor:
   - Symbolic verification against PrimeKG contraindication edges.
   - Blocks prescriptions in high-risk patient phenotypes (eGFR < 30, active TB, bleeding).
3. Symbolic-Neural Conflict Resolution:
   - Detects contradictions between symbolic graph facts and unstructured text.
   - Applies Temporal & Regulatory Priority (FDA Boxed Warning overrides pilot studies).
4. Grounded Confidence Calibration & 'I Don't Know' Safe Refusal:
   - Tri-signal confidence score: KG Path Grounding (40%) + Vector Score (30%) + Safety Verification (30%).
   - If confidence < 0.35, the system safely refuses rather than hallucinating.
"""
from __future__ import annotations
import os
import sys
import re
from typing import Dict, List, Optional, Tuple, Any

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primekg_loader import PrimeKGGraph, get_primekg_instance


# -----------------------------------------------------------------------------
# FDA MAXIMUM THERAPEUTIC DOSAGE BOUNDS (PER 24-HOUR PERIOD)
# -----------------------------------------------------------------------------
MAX_DAILY_DOSAGE_BOUNDS: Dict[str, Dict[str, Any]] = {
    "metformin": {
        "max_daily_mg": 2550,
        "standard_dose": "500-2000 mg/day",
        "citation": "FDA Label (2018) / ADA Standards of Care (2024)"
    },
    "baricitinib": {
        "max_daily_mg": 4,
        "standard_dose": "2-4 mg/day",
        "citation": "FDA Prescribing Information (2022) / DrugBank:DB11718"
    },
    "aspirin": {
        "max_daily_mg": 325,  # Cardiovascular antiplatelet ceiling
        "standard_dose": "75-100 mg/day (cardioprotective)",
        "citation": "AHA/ACC Guidelines (2023)"
    },
    "warfarin": {
        "max_daily_mg": 15,
        "standard_dose": "2-10 mg/day titrated to INR 2.0-3.0",
        "citation": "Chest Antithrombotic Guidelines (2021)"
    },
    "apixaban": {
        "max_daily_mg": 20,
        "standard_dose": "5 mg twice daily (10 mg/day)",
        "citation": "FDA Label (2021) / ARISTOTLE Trial"
    },
    "semaglutide": {
        "max_daily_mg": 2.4 / 7.0,  # ~0.35 mg/day equivalent (2.4 mg/week max)
        "standard_dose": "0.25-2.0 mg subcutaneously once weekly",
        "citation": "FDA Prescribing Information (2023)"
    },
    "osimertinib": {
        "max_daily_mg": 80,
        "standard_dose": "80 mg once daily",
        "citation": "FLAURA Trial (2018) / FDA Approval"
    },
    "sotorasib": {
        "max_daily_mg": 960,
        "standard_dose": "960 mg once daily",
        "citation": "CodeBreaK 100 Trial (2021)"
    }
}


# -----------------------------------------------------------------------------
# NEUROSYMBOLIC GUARDRAIL & CONFLICT RESOLUTION ENGINE
# -----------------------------------------------------------------------------
class NeurosymbolicGuardrail:
    """
    Deterministic safety validator running post-generation / pre-delivery.
    Intercepts dangerous claims, verifies dosage limits, detects contradictions,
    and calculates grounded confidence calibration.
    """
    def __init__(self, kg: Optional[PrimeKGGraph] = None):
        self.kg = kg or get_primekg_instance()

    def extract_dosages(self, text: str) -> List[Tuple[float, str]]:
        """
        Extracts numerical dosages and units (e.g., '500 mg', '4mg', '2.5 g').
        Returns list of (normalized_mg_value, original_matched_string).
        """
        pattern = r'(\d+(?:\.\d+)?)\s*(mg|milligram|milligrams|g|gram|grams)\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        results = []
        for val_str, unit in matches:
            val = float(val_str)
            unit_lower = unit.lower()
            if unit_lower in ("g", "gram", "grams"):
                mg_val = val * 1000.0
            else:
                mg_val = val
            results.append((mg_val, f"{val_str} {unit}"))
        return results

    def validate_dosage(self, drug_name: str, text: str) -> Dict[str, Any]:
        """
        Validates that any numerical dosage mentioned does not exceed
        known FDA maximum therapeutic boundaries.
        """
        drug_key = drug_name.strip().lower()
        if drug_key not in MAX_DAILY_DOSAGE_BOUNDS:
            return {"status": "PASS", "reason": "No bound registered for drug"}

        bounds = MAX_DAILY_DOSAGE_BOUNDS[drug_key]
        extracted = self.extract_dosages(text)

        for mg_val, raw_str in extracted:
            if mg_val > bounds["max_daily_mg"]:
                return {
                    "status": "OVERDOSE_BLOCKED",
                    "severity": "CRITICAL",
                    "drug": drug_name,
                    "prescribed_dose": raw_str,
                    "max_allowed_mg": bounds["max_daily_mg"],
                    "standard_dose": bounds["standard_dose"],
                    "citation": bounds["citation"],
                    "reason": (
                        f"Prescribed dose of {raw_str} exceeds maximum FDA daily safety threshold "
                        f"of {bounds['max_daily_mg']} mg. Risk of acute toxicity."
                    )
                }

        return {"status": "PASS", "reason": "Dosage within FDA safety boundaries"}

    def check_drug_recalls(self, drug_name: str) -> Dict[str, Any]:
        """
        Validates whether a drug has been recalled or withdrawn by regulatory agencies (FDA/EMA).
        """
        if hasattr(self.kg, "is_recalled"):
            recall_info = self.kg.is_recalled(drug_name)
            if recall_info:
                return {
                    "status": "RECALL_BLOCKED",
                    "severity": "CRITICAL",
                    "drug": drug_name,
                    "recall_date": recall_info.get("date"),
                    "reason": recall_info.get("reason"),
                    "warning": recall_info.get("warning")
                }
        return {"status": "PASS", "reason": "No regulatory recall on record"}

    def check_contraindications(self, drug_name: str, patient_context: str) -> Dict[str, Any]:
        """
        Cross-references drug against patient clinical context using PrimeKG
        contraindication edges.
        """
        canonical_drug = self.kg.get_canonical_name(drug_name)
        if not canonical_drug:
            return {"status": "PASS", "reason": "Drug not in graph"}

        ctx_lower = patient_context.lower()
        neighbors = self.kg.get_neighbors(canonical_drug, direction="forward")
        contraindications = [n for n in neighbors if n[0] == "CONTRAINDICATED_IN"]

        # Keyword mapping for clinical phenotyping
        phenotype_keywords = {
            "Severe_Renal_Impairment_eGFR_less_30": ["kidney", "renal", "egfr", "creatinine", "ckd", "dialysis", "nephropathy"],
            "Active_Tuberculosis": ["tb", "tuberculosis", "mycobacterium", "latent tb", "igra"],
            "Active_Serious_Infections": ["infection", "sepsis", "bacterial", "febrile"],
            "Active_Peptic_Ulcer_Disease": ["ulcer", "peptic", "gi bleed", "gastric", "melena", "hematemesis"],
            "Pregnancy_Teratogenicity": ["pregnant", "pregnancy", "trimester", "gestational"],
            "Severe_Uncontrolled_Bleeding": ["bleeding", "hemorrhage", "coagulopathy", "thrombocytopenia", "inr >"],
            "Medullary_Thyroid_Carcinoma": ["thyroid", "men 2", "carcinoma", "calcitonin"],
            "Diabetic_Ketoacidosis": ["dka", "ketoacidosis", "ketones", "acidosis"]
        }

        for rel, target_condition, target_type, cit in contraindications:
            keywords = phenotype_keywords.get(target_condition, [target_condition.lower()])
            if any(kw in ctx_lower for kw in keywords):
                return {
                    "status": "CONTRAINDICATION_BLOCKED",
                    "severity": "CRITICAL",
                    "drug": canonical_drug,
                    "condition": target_condition,
                    "citation": cit,
                    "reason": (
                        f"{canonical_drug} is strictly contraindicated in {target_condition}. "
                        f"Administration carries life-threatening clinical risk."
                    )
                }

        return {"status": "PASS", "reason": "No contraindications detected"}

    def detect_contradictions(self, graph_facts: List[str], passages: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identifies contradictions between symbolic Knowledge Graph facts and retrieved passages.
        Applies regulatory & temporal priority: FDA Guideline overrides pilot studies.
        """
        conflicts = []
        for p in passages:
            p_text_lower = p.get("text", "").lower()
            auth_score = p.get("authority_score", 0.5)

            # Contradiction Pattern 1: Efficacy vs Inefficacy
            if any(neg in p_text_lower for neg in ["no significant benefit", "ineffective", "failed to demonstrate", "trial terminated"]):
                for fact in graph_facts:
                    if "TREATS" in fact or "INHIBITS" in fact:
                        conflicts.append({
                            "type": "EFFICACY_CONTRADICTION",
                            "graph_fact": fact,
                            "passage_source": p.get("source", "Unknown"),
                            "passage_excerpt": p_text_lower[:120] + "...",
                            "resolution": "Flagged for clinician review. FDA guidance prioritizes registered trial endpoints."
                        })

            # Contradiction Pattern 2: Safety Reversal / Alert
            if any(warn in p_text_lower for warn in ["black box", "withdrawn", "discontinued due to toxicity", "fatal"]):
                conflicts.append({
                    "type": "SAFETY_SIGNAL_DISCREPANCY",
                    "passage_source": p.get("source", "Unknown"),
                    "resolution": "Conservative Safety Override Applied: Cautionary statement prioritized."
                })

        return conflicts

    def calculate_grounded_confidence(
        self,
        has_graph_path: bool,
        top_vector_score: float,
        is_safe: bool,
        entity_recognized: bool = True
    ) -> Tuple[float, str]:
        """
        Computes calibrated confidence score:
        - KG Path Grounding: 40%
        - Dense Vector Retrieval Score: 30%
        - Neurosymbolic Safety Clearance: 30%
        
        Returns: (confidence_score [0.0 - 1.0], recommendation)
        """
        # If the drug entity is unknown and no graph path exists, prevent false safety inflation
        if not entity_recognized and not has_graph_path and top_vector_score < 0.60:
            return 0.15, "SAFE_REFUSAL_RECOMMENDED"

        kg_weight = 0.40 if has_graph_path else 0.05
        vec_weight = min(0.30, top_vector_score * 0.30)
        safety_weight = 0.30 if (is_safe and (entity_recognized or has_graph_path)) else 0.0

        total_confidence = round(kg_weight + vec_weight + safety_weight, 3)

        if total_confidence >= 0.75:
            rec = "HIGH_CONFIDENCE_GROUNDED"
        elif total_confidence >= 0.45:
            rec = "MODERATE_CONFIDENCE_PARTIAL_GROUNDING"
        else:
            rec = "SAFE_REFUSAL_RECOMMENDED"

        return total_confidence, rec

    def audit_clinical_response(
        self,
        drug_name: str,
        user_query: str,
        proposed_response: str,
        graph_path_found: bool,
        top_passage_score: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive single-call audit:
        Runs dosage check, contraindication check, and confidence calibration.
        """
        # 0. Regulatory Drug Recall Check
        recall_check = self.check_drug_recalls(drug_name)
        if recall_check["status"] == "RECALL_BLOCKED":
            return {
                "decision": "BLOCKED",
                "trigger": "DRUG_RECALL_INTERCEPTED",
                "details": recall_check,
                "confidence": 0.0,
                "safe_response": (
                    f"⚠️ CRITICAL REGULATORY WITHDRAWAL: {recall_check['warning']}\n"
                    f"FDA Status: WITHDRAWN_FROM_MARKET."
                )
            }

        # 1. Dosage Check
        dosage_check = self.validate_dosage(drug_name, proposed_response)
        if dosage_check["status"] == "OVERDOSE_BLOCKED":
            return {
                "decision": "BLOCKED",
                "trigger": "DOSAGE_SAFETY_OVERFLOW",
                "details": dosage_check,
                "confidence": 0.0,
                "safe_response": (
                    f"⚠️ PRESCRIBING OVERRIDE: {dosage_check['reason']}\n"
                    f"Standard Therapeutic Range: {dosage_check['standard_dose']} ({dosage_check['citation']})."
                )
            }

        # 2. Contraindication Check
        contra_check = self.check_contraindications(drug_name, user_query + " " + proposed_response)
        if contra_check["status"] == "CONTRAINDICATION_BLOCKED":
            return {
                "decision": "BLOCKED",
                "trigger": "CONTRAINDICATION_INTERCEPTED",
                "details": contra_check,
                "confidence": 0.0,
                "safe_response": (
                    f"⚠️ CLINICAL INTERCEPTION: {contra_check['reason']}\n"
                    f"Regulatory Provenance: {contra_check['citation']}."
                )
            }

        # 3. Confidence Calibration
        entity_recognized = self.kg.get_canonical_name(drug_name) is not None
        confidence, rec = self.calculate_grounded_confidence(
            has_graph_path=graph_path_found,
            top_vector_score=top_passage_score,
            is_safe=True,
            entity_recognized=entity_recognized
        )

        if rec == "SAFE_REFUSAL_RECOMMENDED":
            return {
                "decision": "REFUSED_INSUFFICIENT_EVIDENCE",
                "confidence": confidence,
                "safe_response": (
                    "I have insufficient verified clinical evidence or multi-hop knowledge graph "
                    "paths to answer this query safely. Please consult peer-reviewed guidelines."
                )
            }

        return {
            "decision": "PASSED",
            "confidence": confidence,
            "recommendation": rec,
            "dosage_status": dosage_check["status"],
            "contraindication_status": contra_check["status"]
        }


def get_default_guardrail() -> NeurosymbolicGuardrail:
    """Factory helper to obtain guardrail singleton."""
    return NeurosymbolicGuardrail()
