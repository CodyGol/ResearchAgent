"""Tests for deterministic Decision Brief presentation payload."""

import pytest

from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    ClaimEvidenceRelationship,
    ClaimType,
    Evidence,
    EvidenceMatchType,
    EvidenceType,
    ExtractionMethod,
    Source,
    SourceQuality,
    SourceType,
    VerificationResult,
    VerificationStatus,
    EvidenceConfidence,
    KnowledgeCategory,
)
from services.decision_brief import build_decision_brief_payload
from services.decision_synthesis_schemas import RecommendationStatus


def _synthesis(**kwargs) -> dict:
    base = {
        "decision": "Which vendor?",
        "recommendation_status": RecommendationStatus.RECOMMEND.value,
        "recommended_option": "Vendor A",
        "rationale": "Vendor A is favored on cost.",
        "supporting_criteria": [],
        "limiting_criteria": [],
        "constraint_assessments": [],
        "key_uncertainties": [],
        "decision_limitations": [],
        "critical_missing_context": [],
        "assumptions_relied_on": [],
        "change_conditions": [],
    }
    base.update(kwargs)
    return base


def _frame() -> dict:
    return {
        "decision": "Which vendor?",
        "decision_type": "vendor_selection",
        "options": [
            {"label": "Vendor A", "origin": "explicit"},
            {"label": "Vendor B", "origin": "explicit"},
        ],
        "criteria": [
            {"label": "Cost", "origin": "explicit", "priority": "primary"},
            {"label": "Reliability", "origin": "explicit", "priority": "standard"},
        ],
        "constraints": ["Budget under $20k"],
        "missing_decision_context": [],
        "explicit_assumptions": [],
    }


def _option_evaluation() -> dict:
    return {
        "decision": "Which vendor?",
        "option_evaluations": [
            {
                "option_label": "Vendor A",
                "option_origin": "explicit",
                "criteria_evaluations": [
                    {
                        "criterion_label": "Cost",
                        "criterion_origin": "explicit",
                        "criterion_priority": "primary",
                        "assessment": "favorable",
                        "knowledge_coverage": "grounded",
                        "claim_ids": [1],
                        "verification_ids": [10],
                        "knowledge_categories": ["known"],
                        "reason": "Vendor A is cheaper.",
                    }
                ],
            },
            {
                "option_label": "Vendor B",
                "option_origin": "explicit",
                "criteria_evaluations": [
                    {
                        "criterion_label": "Cost",
                        "criterion_origin": "explicit",
                        "criterion_priority": "primary",
                        "assessment": "unfavorable",
                        "knowledge_coverage": "grounded",
                        "claim_ids": [1],
                        "verification_ids": [10],
                        "knowledge_categories": ["known"],
                        "reason": "Vendor B costs more.",
                    }
                ],
            },
        ],
        "decision_limitations": [],
        "constraints": ["Budget under $20k"],
    }


def _lineage_state(**kwargs) -> dict:
    claim = Claim(id=1, research_run_id=1, text="Vendor A costs $12k/year.", claim_type=ClaimType.FACTUAL)
    evidence = Evidence(
        id=100,
        source_id=200,
        research_run_id=1,
        exact_text="Vendor A pricing starts at $12,000 annually.",
        is_validated=True,
        evidence_type=EvidenceType.DIRECT_QUOTE,
        extraction_method=ExtractionMethod.LLM,
        match_type=EvidenceMatchType.EXACT,
        metadata={"display_id": "E1"},
    )
    source = Source(
        id=200,
        research_run_id=1,
        url="https://example.com/vendor-a",
        title="Vendor A Pricing",
        content_hash="abc",
        source_type=SourceType.WEB,
        source_quality=SourceQuality.OFFICIAL,
    )
    relation = ClaimEvidenceRelation(
        claim_id=1,
        evidence_id=100,
        relationship=ClaimEvidenceRelationship.SUPPORTS,
    )
    verification = VerificationResult(
        id=10,
        claim_id=1,
        research_run_id=1,
        status=VerificationStatus.SUPPORTED,
        confidence=EvidenceConfidence.HIGH,
        knowledge_category=KnowledgeCategory.KNOWN,
    )
    base = {
        "decision_synthesis": _synthesis(),
        "option_evaluation": _option_evaluation(),
        "decision_frame": _frame(),
        "material_claims": [claim],
        "validated_evidence": [evidence],
        "normalized_sources": [source],
        "claim_evidence_relations": [relation],
        "verification_results": [verification],
        "knowledge_state": {
            "known": [
                {
                    "claim_id": 1,
                    "verification_id": 10,
                    "knowledge_category": "known",
                    "verification_status": "supported",
                    "confidence": "high",
                    "relation_ids": [],
                    "evidence_ids": [100],
                }
            ],
            "likely": [],
            "disputed": [],
            "unknown": [],
            "contradicted": [],
            "unverifiable": [],
            "information_gaps": [],
            "metrics": {},
        },
    }
    base.update(kwargs)
    return base


class TestBuildDecisionBriefPayload:
    def test_recommend_payload(self):
        payload = build_decision_brief_payload(_lineage_state())
        assert payload is not None
        assert payload["decision_synthesis"]["recommendation_status"] == "recommend"
        assert payload["decision_synthesis"]["recommended_option"] == "Vendor A"
        assert payload["option_evaluation"] is not None
        assert payload["decision_frame"] is not None

    def test_tentative_recommendation_payload(self):
        payload = build_decision_brief_payload(
            _lineage_state(
                decision_synthesis=_synthesis(
                    recommendation_status=RecommendationStatus.TENTATIVE_RECOMMENDATION.value,
                    recommended_option="Vendor B",
                )
            )
        )
        assert payload is not None
        assert payload["decision_synthesis"]["recommendation_status"] == "tentative_recommendation"

    def test_insufficient_basis_payload(self):
        payload = build_decision_brief_payload(
            _lineage_state(
                decision_synthesis=_synthesis(
                    recommendation_status=RecommendationStatus.INSUFFICIENT_BASIS.value,
                    recommended_option=None,
                    rationale="Not enough evidence.",
                    critical_missing_context=["Salesforce integration evidence"],
                )
            )
        )
        assert payload is not None
        assert payload["decision_synthesis"]["recommended_option"] is None

    def test_no_synthesis_returns_none(self):
        state = _lineage_state()
        state["decision_synthesis"] = None
        assert build_decision_brief_payload(state) is None

    def test_malformed_status_returns_none(self):
        state = _lineage_state(
            decision_synthesis=_synthesis(recommendation_status="strong_maybe")
        )
        assert build_decision_brief_payload(state) is None

    def test_only_referenced_claims_included(self):
        state = _lineage_state()
        state["material_claims"] = [
            Claim(id=1, research_run_id=1, text="Referenced.", claim_type=ClaimType.FACTUAL),
            Claim(id=99, research_run_id=1, text="Unreferenced.", claim_type=ClaimType.FACTUAL),
        ]
        payload = build_decision_brief_payload(state)
        assert payload is not None
        assert "1" in payload["claim_lineage"]
        assert "99" not in payload["claim_lineage"]

    def test_claim_evidence_source_lineage(self):
        payload = build_decision_brief_payload(_lineage_state())
        entry = payload["claim_lineage"]["1"]
        assert entry["text"] == "Vendor A costs $12k/year."
        assert entry["knowledge_category"] == "known"
        assert entry["verification_status"] == "supported"
        assert len(entry["evidence"]) == 1
        assert entry["evidence"][0]["display_id"] == "E1"
        assert entry["evidence"][0]["source_url"] == "https://example.com/vendor-a"
        assert entry["evidence"][0]["source_title"] == "Vendor A Pricing"

    def test_unrelated_claims_excluded(self):
        state = _lineage_state()
        state["option_evaluation"] = {
            "decision": "Which vendor?",
            "option_evaluations": [],
            "decision_limitations": [],
            "constraints": [],
        }
        state["decision_synthesis"] = _synthesis(
            supporting_criteria=[],
            limiting_criteria=[],
            constraint_assessments=[],
            change_conditions=[],
        )
        payload = build_decision_brief_payload(state)
        assert payload is not None
        assert payload["claim_lineage"] == {}

    def test_missing_lineage_objects_handled(self):
        state = _lineage_state(
            material_claims=[],
            claim_evidence_relations=[],
            validated_evidence=[],
            normalized_sources=[],
        )
        payload = build_decision_brief_payload(state)
        assert payload is not None
        assert payload["claim_lineage"] == {}

    def test_duplicate_evidence_deduped(self):
        state = _lineage_state()
        state["claim_evidence_relations"] = [
            ClaimEvidenceRelation(claim_id=1, evidence_id=100, relationship=ClaimEvidenceRelationship.SUPPORTS),
            ClaimEvidenceRelation(claim_id=1, evidence_id=100, relationship=ClaimEvidenceRelationship.QUALIFIES),
        ]
        payload = build_decision_brief_payload(state)
        assert len(payload["claim_lineage"]["1"]["evidence"]) == 1

    def test_empty_optional_fields_do_not_crash(self):
        state = _lineage_state(
            decision_synthesis=_synthesis(
                supporting_criteria=[],
                limiting_criteria=[],
                key_uncertainties=[],
                assumptions_relied_on=[],
                change_conditions=[],
            ),
            option_evaluation=None,
            decision_frame=None,
            knowledge_state=None,
            verification_results=[],
        )
        payload = build_decision_brief_payload(state)
        assert payload is not None

    def test_collects_claim_ids_from_synthesis_refs(self):
        state = _lineage_state()
        state["material_claims"].append(
            Claim(id=2, research_run_id=1, text="Constraint claim.", claim_type=ClaimType.FACTUAL)
        )
        state["decision_synthesis"] = _synthesis(
            constraint_assessments=[
                {
                    "option_label": "Vendor A",
                    "constraint": "Budget under $20k",
                    "compliance": "satisfied",
                    "claim_ids": [2],
                    "reason": "Within budget.",
                }
            ],
            change_conditions=[
                {
                    "description": "New pricing evidence.",
                    "change_type": "evidence_change",
                    "related_claim_ids": [2],
                }
            ],
        )
        payload = build_decision_brief_payload(state)
        assert "2" in payload["claim_lineage"]

    def test_fail_open_invalid_synthesis_type(self):
        state = _lineage_state(decision_synthesis="not-a-dict")
        assert build_decision_brief_payload(state) is None
