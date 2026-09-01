"""Tests for Phase 2C cross-source claim verification."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    ClaimEvidenceRelationship,
    ClaimType,
    Evidence,
    EvidenceConfidence,
    EvidenceType,
    ExtractionMethod,
    Source,
    SourceQuality,
    SourceType,
    VerificationStatus,
)
from services.claim_verification import (
    aggregate_verification_status,
    select_cross_source_candidates,
    source_publisher_domain,
    verify_material_claims,
)


def _source(sid: int, domain: str, url: str | None = None) -> Source:
    return Source(
        id=sid,
        research_run_id=1,
        url=url or f"https://{domain}/page",
        title=domain,
        content="",
        content_hash=f"hash-{sid}",
        source_type=SourceType.WEB,
        source_quality=SourceQuality.REPUTABLE_SECONDARY,
        metadata={"domain": domain},
    )


def _evidence(eid: int, source_id: int, text: str) -> Evidence:
    return Evidence(
        id=eid,
        source_id=source_id,
        research_run_id=1,
        exact_text=text,
        evidence_type=EvidenceType.DIRECT_QUOTE,
        extraction_method=ExtractionMethod.LLM,
        is_validated=True,
    )


def _claim(cid: int, text: str, **kwargs) -> Claim:
    return Claim(
        id=cid,
        research_run_id=1,
        text=text,
        metadata={"is_material": True, **kwargs.get("metadata", {})},
        **{k: v for k, v in kwargs.items() if k != "metadata"},
    )


class TestPublisherIndependence:
    def test_same_domain_not_independent_candidates(self):
        claim = _claim(1, "Max Verstappen won the 2023 Formula 1 World Championship.")
        origin = _evidence(1, 1, "Max Verstappen won the 2023 championship.")
        cross_same_domain = _evidence(2, 2, "Verstappen secured the 2023 title.")
        sources = {
            1: _source(1, "formula1.com"),
            2: _source(2, "formula1.com", "https://formula1.com/other"),
        }
        candidates = select_cross_source_candidates(
            claim,
            [origin, cross_same_domain],
            sources,
            origin_evidence_ids={1},
            origin_domains={"formula1.com"},
        )
        assert candidates == []

    def test_different_domain_selected(self):
        claim = _claim(1, "Max Verstappen won the 2023 Formula 1 World Championship.")
        origin = _evidence(1, 1, "Max Verstappen won the 2023 championship.")
        cross = _evidence(2, 2, "Max Verstappen won the 2023 Formula 1 title.")
        sources = {
            1: _source(1, "formula1.com"),
            2: _source(2, "bbc.com"),
        }
        candidates = select_cross_source_candidates(
            claim,
            [origin, cross],
            sources,
            origin_evidence_ids={1},
            origin_domains={"formula1.com"},
        )
        assert len(candidates) == 1
        assert candidates[0].id == 2

    def test_source_publisher_domain_from_metadata(self):
        s = _source(1, "reuters.com")
        assert source_publisher_domain(s) == "reuters.com"


class TestAggregation:
    def _relations(self, claim_id: int, pairs: list[tuple[int, ClaimEvidenceRelationship]]):
        return [
            ClaimEvidenceRelation(
                claim_id=claim_id,
                evidence_id=eid,
                relationship=rel,
            )
            for eid, rel in pairs
        ]

    def test_two_independent_supports_supported(self):
        claim = _claim(1, "Tokyo is the capital of Japan.")
        evidence = {
            1: _evidence(1, 1, "Tokyo is the capital of Japan."),
            2: _evidence(2, 2, "Tokyo is the capital of Japan."),
        }
        sources = {1: _source(1, "britannica.com"), 2: _source(2, "cia.gov")}
        rels = self._relations(1, [
            (1, ClaimEvidenceRelationship.SUPPORTS),
            (2, ClaimEvidenceRelationship.SUPPORTS),
        ])
        status, conf, _ = aggregate_verification_status(
            claim, rels, evidence, sources
        )
        assert status == VerificationStatus.SUPPORTED
        assert conf == EvidenceConfidence.HIGH

    def test_single_support_partially_supported(self):
        claim = _claim(1, "Tokyo is the capital of Japan.")
        evidence = {1: _evidence(1, 1, "Tokyo is the capital of Japan.")}
        sources = {1: _source(1, "britannica.com")}
        rels = self._relations(1, [(1, ClaimEvidenceRelationship.SUPPORTS)])
        status, conf, _ = aggregate_verification_status(
            claim, rels, evidence, sources
        )
        assert status == VerificationStatus.PARTIALLY_SUPPORTED
        assert conf == EvidenceConfidence.MEDIUM

    def test_support_and_contradict_uncertain(self):
        claim = _claim(1, "Revenue was $391 billion in fiscal 2025.")
        evidence = {
            1: _evidence(1, 1, "Revenue was $391 billion in fiscal 2025."),
            2: _evidence(2, 2, "Revenue was $394 billion in fiscal 2025."),
        }
        sources = {1: _source(1, "apple.com"), 2: _source(2, "reuters.com")}
        rels = self._relations(1, [
            (1, ClaimEvidenceRelationship.SUPPORTS),
            (2, ClaimEvidenceRelationship.CONTRADICTS),
        ])
        status, _, _ = aggregate_verification_status(claim, rels, evidence, sources)
        assert status == VerificationStatus.UNCERTAIN

    def test_contradict_only_contradicted(self):
        claim = _claim(1, "Revenue was $391 billion in fiscal 2025.")
        evidence = {2: _evidence(2, 2, "Revenue was $394 billion in fiscal 2025.")}
        sources = {2: _source(2, "reuters.com")}
        rels = self._relations(1, [(2, ClaimEvidenceRelationship.CONTRADICTS)])
        status, _, _ = aggregate_verification_status(claim, rels, evidence, sources)
        assert status == VerificationStatus.CONTRADICTED

    def test_support_and_qualify_partially_supported(self):
        claim = _claim(1, "Revenue was $391 billion in fiscal 2025.")
        evidence = {
            1: _evidence(1, 1, "Revenue was $391 billion in fiscal 2025."),
            2: _evidence(2, 2, "Non-GAAP revenue was $391 billion in fiscal 2025."),
        }
        sources = {1: _source(1, "apple.com"), 2: _source(2, "sec.gov")}
        rels = self._relations(1, [
            (1, ClaimEvidenceRelationship.SUPPORTS),
            (2, ClaimEvidenceRelationship.QUALIFIES),
        ])
        status, _, _ = aggregate_verification_status(claim, rels, evidence, sources)
        assert status == VerificationStatus.PARTIALLY_SUPPORTED

    def test_opinion_unverifiable(self):
        claim = _claim(1, "This is the best product ever.", claim_type=ClaimType.OPINION)
        status, _, _ = aggregate_verification_status(claim, [], {}, {})
        assert status == VerificationStatus.UNVERIFIABLE


class TestVerifyMaterialClaims:
    @pytest.mark.asyncio
    async def test_cross_source_support_added(self):
        claim = _claim(
            1,
            "Max Verstappen won the 2023 Formula 1 World Championship.",
        )
        origin_ev = _evidence(
            1, 1,
            "Max Verstappen won the 2023 Formula 1 World Championship.",
        )
        cross_ev = _evidence(
            2, 2,
            "Max Verstappen won the 2023 Formula 1 World Championship.",
        )
        sources = [_source(1, "formula1.com"), _source(2, "bbc.com")]
        origin_rel = [
            ClaimEvidenceRelation(
                claim_id=1,
                evidence_id=1,
                relationship=ClaimEvidenceRelationship.SUPPORTS,
                reasoning="origin",
            )
        ]

        verifications, new_rels, metrics = await verify_material_claims(
            [claim],
            [origin_ev, cross_ev],
            sources,
            origin_rel,
            research_run_id=1,
            llm=None,
            use_llm=False,
        )

        assert len(verifications) == 1
        assert verifications[0].status == VerificationStatus.SUPPORTED
        assert verifications[0].knowledge_category is None
        assert len(new_rels) >= 1
        assert any(r.relationship == ClaimEvidenceRelationship.SUPPORTS for r in new_rels)
        assert metrics.cross_source_relations_added >= 1

    @pytest.mark.asyncio
    async def test_origin_supports_preserved_not_rerun(self):
        claim = _claim(1, "Tokyo is the capital of Japan.")
        origin_ev = _evidence(1, 1, "Tokyo is the capital of Japan.")
        origin_rel = [
            ClaimEvidenceRelation(
                claim_id=1,
                evidence_id=1,
                relationship=ClaimEvidenceRelationship.SUPPORTS,
                reasoning="origin validation",
            )
        ]
        sources = [_source(1, "britannica.com")]

        verifications, new_rels, _ = await verify_material_claims(
            [claim],
            [origin_ev],
            sources,
            origin_rel,
            research_run_id=1,
            use_llm=False,
        )

        assert verifications[0].status == VerificationStatus.PARTIALLY_SUPPORTED
        assert new_rels == []

    @pytest.mark.asyncio
    async def test_llm_batch_for_ambiguous(self):
        claim = _claim(1, "Apple reported strong growth in services revenue.")
        origin_ev = _evidence(1, 1, "Apple reported strong growth in services revenue.")
        cross_ev = _evidence(
            2, 2,
            "Services revenue increased year over year at Apple.",
        )
        sources = [_source(1, "apple.com"), _source(2, "reuters.com")]
        origin_rel = [
            ClaimEvidenceRelation(
                claim_id=1, evidence_id=1,
                relationship=ClaimEvidenceRelationship.SUPPORTS,
            )
        ]

        mock_llm = MagicMock()
        from services.claim_verification_schemas import (
            ClaimRelationshipBatchItem,
            ClaimRelationshipBatchOutput,
        )

        output = ClaimRelationshipBatchOutput(
            assessments=[
                ClaimRelationshipBatchItem(
                    evidence_id=2,
                    relationship="supports",
                    reasoning="Corroborates services growth",
                )
            ]
        )
        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=output)
        mock_llm.with_structured_output = MagicMock(return_value=structured)

        verifications, new_rels, metrics = await verify_material_claims(
            [claim],
            [origin_ev, cross_ev],
            sources,
            origin_rel,
            research_run_id=1,
            llm=mock_llm,
            use_llm=True,
        )

        assert metrics.llm_batches == 1
        assert any(r.evidence_id == 2 for r in new_rels)


class TestPhase2CValidationFixtures:
    """Deterministic integration fixtures for Phase 2C primitive validation."""

    @pytest.mark.asyncio
    async def test_b_qualification_not_contradiction(self):
        """TEST B: methodology difference → QUALIFIES, not CONTRADICTS."""
        claim = _claim(
            1,
            "Company X's revenue grew approximately 20% in 2025.",
            temporal_scope="2025",
        )
        ev_a = _evidence(1, 1, "Company X reported 20% revenue growth in 2025.")
        ev_b = _evidence(
            2, 2,
            "On a constant-currency basis, Company X's revenue increased 14% in 2025.",
        )
        sources = [_source(1, "companyx.com"), _source(2, "reuters.com")]
        origin_rel = [
            ClaimEvidenceRelation(
                claim_id=1, evidence_id=1,
                relationship=ClaimEvidenceRelationship.SUPPORTS,
                reasoning="origin",
            )
        ]

        verifications, new_rels, _ = await verify_material_claims(
            [claim], [ev_a, ev_b], sources, origin_rel, 1, use_llm=False
        )

        rel_map = {r.evidence_id: r.relationship for r in new_rels}
        assert rel_map.get(2) == ClaimEvidenceRelationship.QUALIFIES
        assert ClaimEvidenceRelationship.CONTRADICTS not in rel_map.values()
        assert verifications[0].status == VerificationStatus.PARTIALLY_SUPPORTED

    @pytest.mark.asyncio
    async def test_c_support_plus_contradict_uncertain(self):
        """TEST C: credible support + credible contradict → UNCERTAIN."""
        claim = _claim(
            1,
            "The market was valued at $12 billion in 2025.",
            temporal_scope="2025",
        )
        ev_a = _evidence(1, 1, "The market reached $12 billion in 2025.")
        ev_b = _evidence(2, 2, "The market was valued at $7 billion in 2025.")
        sources = [_source(1, "sourcea.com"), _source(2, "sourceb.com")]
        origin_rel = [
            ClaimEvidenceRelation(
                claim_id=1, evidence_id=1,
                relationship=ClaimEvidenceRelationship.SUPPORTS,
                reasoning="origin",
            )
        ]

        verifications, new_rels, _ = await verify_material_claims(
            [claim], [ev_a, ev_b], sources, origin_rel, 1, use_llm=False
        )

        rel_map = {r.evidence_id: r.relationship for r in new_rels}
        assert rel_map.get(2) == ClaimEvidenceRelationship.CONTRADICTS
        assert verifications[0].status == VerificationStatus.UNCERTAIN

    @pytest.mark.asyncio
    async def test_d_contradict_only_contradicted(self):
        """TEST D: no support, only contradict → CONTRADICTED."""
        claim = _claim(
            1,
            "Company X generated $12 billion of revenue in 2025.",
            temporal_scope="2025",
        )
        ev_b = _evidence(
            2, 2,
            "Company X generated $7 billion of revenue in 2025.",
        )
        sources = [_source(2, "reuters.com")]

        verifications, new_rels, _ = await verify_material_claims(
            [claim], [ev_b], sources, [], 1, use_llm=False
        )

        assert len(new_rels) == 1
        assert new_rels[0].relationship == ClaimEvidenceRelationship.CONTRADICTS
        assert verifications[0].status == VerificationStatus.CONTRADICTED

    def test_e_publisher_independence_counting(self):
        """TEST E: same domain pages ≠ independent; different domain = independent."""
        claim = _claim(1, "Max Verstappen won the 2023 championship.")
        evidence = {
            1: _evidence(1, 1, "Max Verstappen won the 2023 championship."),
            2: _evidence(2, 2, "Max Verstappen won the 2023 championship."),
            3: _evidence(3, 3, "Max Verstappen won the 2023 championship."),
        }
        sources = {
            1: _source(1, "formula1.com", "https://formula1.com/page-1"),
            2: _source(2, "formula1.com", "https://formula1.com/page-2"),
            3: _source(3, "bbc.com", "https://bbc.com/page"),
        }
        rels = [
            ClaimEvidenceRelation(claim_id=1, evidence_id=1, relationship=ClaimEvidenceRelationship.SUPPORTS),
            ClaimEvidenceRelation(claim_id=1, evidence_id=2, relationship=ClaimEvidenceRelationship.SUPPORTS),
            ClaimEvidenceRelation(claim_id=1, evidence_id=3, relationship=ClaimEvidenceRelationship.SUPPORTS),
        ]
        from services.claim_verification import _count_independent_supports

        independent = _count_independent_supports(1, rels, evidence, sources)
        assert independent == 2  # formula1.com + bbc.com, not 3

