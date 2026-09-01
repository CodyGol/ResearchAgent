"""Tests for Phase 2D deterministic knowledge state derivation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.models import (
    Claim,
    ClaimEvidenceRelation,
    ClaimEvidenceRelationship,
    ClaimType,
    EvidenceConfidence,
    KnowledgeCategory,
    VerificationResult,
    VerificationStatus,
)
from graph import create_graph
from nodes.knowledge_state import knowledge_state_node
from services.knowledge_state import derive_knowledge_state, map_verification_to_category
from state import Critique


def _claim(cid: int, text: str = "Sample claim.") -> Claim:
    return Claim(id=cid, research_run_id=1, text=text)


def _verification(
    cid: int,
    status: VerificationStatus,
    confidence: EvidenceConfidence,
    vid: int | None = None,
) -> VerificationResult:
    return VerificationResult(
        id=vid or cid,
        claim_id=cid,
        research_run_id=1,
        status=status,
        confidence=confidence,
    )


def _relation(
    cid: int,
    eid: int,
    rel: ClaimEvidenceRelationship,
    rid: int | None = None,
) -> ClaimEvidenceRelation:
    return ClaimEvidenceRelation(
        id=rid,
        claim_id=cid,
        evidence_id=eid,
        relationship=rel,
    )


class TestCategoryMapping:
  @pytest.mark.parametrize(
      ("status", "confidence", "expected"),
      [
          (VerificationStatus.SUPPORTED, EvidenceConfidence.HIGH, KnowledgeCategory.KNOWN),
          (VerificationStatus.PARTIALLY_SUPPORTED, EvidenceConfidence.MEDIUM, KnowledgeCategory.LIKELY),
          (VerificationStatus.UNCERTAIN, EvidenceConfidence.LOW, KnowledgeCategory.DISPUTED),
          (VerificationStatus.INSUFFICIENT_EVIDENCE, EvidenceConfidence.LOW, KnowledgeCategory.UNKNOWN),
          (VerificationStatus.CONTRADICTED, EvidenceConfidence.LOW, None),
          (VerificationStatus.UNVERIFIABLE, EvidenceConfidence.LOW, None),
      ],
  )
  def test_map_verification_to_category(self, status, confidence, expected):
      assert map_verification_to_category(status, confidence) == expected


class TestDeriveKnowledgeState:
  def test_supported_high_goes_to_known(self):
      claims = [_claim(1)]
      verifications = [_verification(1, VerificationStatus.SUPPORTED, EvidenceConfidence.HIGH)]
      relations = [_relation(1, 10, ClaimEvidenceRelationship.SUPPORTS, rid=100)]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=relations,
      )

      assert len(ks.known) == 1
      assert ks.known[0].claim_id == 1
      assert ks.known[0].verification_id == 1
      assert ks.known[0].knowledge_category == KnowledgeCategory.KNOWN
      assert ks.known[0].relation_ids == [100]
      assert ks.known[0].evidence_ids == [10]
      assert updated[0].knowledge_category == KnowledgeCategory.KNOWN

  def test_partially_supported_goes_to_likely(self):
      claims = [_claim(2)]
      verifications = [
          _verification(2, VerificationStatus.PARTIALLY_SUPPORTED, EvidenceConfidence.MEDIUM)
      ]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      assert len(ks.likely) == 1
      assert updated[0].knowledge_category == KnowledgeCategory.LIKELY

  def test_uncertain_goes_to_disputed(self):
      claims = [_claim(3)]
      verifications = [_verification(3, VerificationStatus.UNCERTAIN, EvidenceConfidence.LOW)]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      assert len(ks.disputed) == 1
      assert updated[0].knowledge_category == KnowledgeCategory.DISPUTED

  def test_insufficient_evidence_goes_to_unknown(self):
      claims = [_claim(4)]
      verifications = [
          _verification(4, VerificationStatus.INSUFFICIENT_EVIDENCE, EvidenceConfidence.LOW)
      ]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      assert len(ks.unknown) == 1
      assert updated[0].knowledge_category == KnowledgeCategory.UNKNOWN

  def test_contradicted_bucket_with_null_category(self):
      claims = [_claim(5)]
      verifications = [_verification(5, VerificationStatus.CONTRADICTED, EvidenceConfidence.LOW)]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      assert len(ks.contradicted) == 1
      assert ks.contradicted[0].knowledge_category is None
      assert updated[0].knowledge_category is None

  def test_unverifiable_bucket_with_null_category(self):
      claims = [_claim(6, text="AI will dominate all industries.")]
      verifications = [_verification(6, VerificationStatus.UNVERIFIABLE, EvidenceConfidence.LOW)]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      assert len(ks.unverifiable) == 1
      assert updated[0].knowledge_category is None

  def test_critic_unsupported_areas_become_information_gaps(self):
      critique = Critique(
          quality_score=0.7,
          is_sufficient=True,
          unsupported_areas=["Historical context missing", "  "],
      )

      ks, _ = derive_knowledge_state(
          material_claims=[],
          verification_results=[],
          claim_evidence_relations=[],
          critique=critique,
      )

      assert len(ks.information_gaps) == 1
      assert ks.information_gaps[0].description == "Historical context missing"
      assert ks.information_gaps[0].source == "critic_unsupported_area"
      assert ks.metrics["information_gap_count"] == 1

  def test_entries_do_not_duplicate_claim_text(self):
      claims = [_claim(7, text="Secret factual proposition.")]
      verifications = [_verification(7, VerificationStatus.SUPPORTED, EvidenceConfidence.HIGH)]

      ks, _ = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      dumped = ks.model_dump()
      assert "Secret factual proposition." not in str(dumped)
      assert "claim_text" not in dumped

  def test_relation_and_evidence_ids_derived(self):
      claims = [_claim(8)]
      verifications = [_verification(8, VerificationStatus.SUPPORTED, EvidenceConfidence.HIGH)]
      relations = [
          _relation(8, 21, ClaimEvidenceRelationship.SUPPORTS, rid=201),
          _relation(8, 22, ClaimEvidenceRelationship.QUALIFIES, rid=202),
      ]

      ks, _ = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=relations,
      )

      entry = ks.known[0]
      assert entry.relation_ids == [201, 202]
      assert entry.evidence_ids == [21, 22]

  def test_orphan_material_claim_not_marked_unknown(self):
      claims = [_claim(9), _claim(10)]
      verifications = [_verification(10, VerificationStatus.INSUFFICIENT_EVIDENCE, EvidenceConfidence.LOW)]

      ks, updated = derive_knowledge_state(
          material_claims=claims,
          verification_results=verifications,
          claim_evidence_relations=[],
      )

      assert ks.metrics["orphan_material_claims"] == 1
      assert ks.metrics["orphan_claim_ids"] == [9]
      assert len(ks.unknown) == 1
      assert ks.unknown[0].claim_id == 10
      assert len(updated) == 1

  def test_empty_inputs_produce_empty_state(self):
      ks, updated = derive_knowledge_state(
          material_claims=[],
          verification_results=[],
          claim_evidence_relations=[],
      )

      assert ks.known == []
      assert ks.likely == []
      assert ks.disputed == []
      assert ks.unknown == []
      assert ks.contradicted == []
      assert ks.unverifiable == []
      assert ks.information_gaps == []
      assert updated == []
      assert ks.metrics["known_count"] == 0


class TestGraphRouting:
  def _state(self, *, sufficient: bool, iteration: int = 0, max_iterations: int = 1):
      return {
          "critique": Critique(quality_score=0.8, is_sufficient=sufficient),
          "iteration_count": iteration,
          "query_classification": {
              "research_budget": {"max_iterations": max_iterations},
          },
      }

  def _route_after_critic(self, state: dict) -> str:
      from config import settings

      critique = state.get("critique")
      if not critique:
          return "knowledge_state"
      if critique.is_sufficient:
          return "knowledge_state"
      iteration = state.get("iteration_count", 0)
      classification = state.get("query_classification") or {}
      budget = classification.get("research_budget", {})
      max_iter = budget.get("max_iterations", settings.max_research_iterations)
      if iteration >= max_iter:
          return "knowledge_state"
      return "researcher"

  def test_critic_refinement_routes_to_researcher(self):
      assert self._route_after_critic(self._state(sufficient=False, iteration=0)) == "researcher"
      assert self._route_after_critic(self._state(sufficient=True)) == "knowledge_state"
      assert (
          self._route_after_critic(self._state(sufficient=False, iteration=1, max_iterations=1))
          == "knowledge_state"
      )

  def test_graph_contains_knowledge_state_node(self):
      graph = create_graph()
      assert "knowledge_state" in graph.nodes


class TestKnowledgeStateNode:
  @pytest.mark.asyncio
  async def test_persists_knowledge_category_on_verifications(self):
      state = {
          "material_claims": [_claim(1)],
          "verification_results": [
              _verification(1, VerificationStatus.SUPPORTED, EvidenceConfidence.HIGH)
          ],
          "claim_evidence_relations": [],
          "critique": Critique(quality_score=0.9, is_sufficient=True),
          "is_run_persisted": True,
          "cost_metrics": {},
      }

      mock_repo = MagicMock()
      mock_repo.save_verifications = AsyncMock(
          side_effect=lambda results: [
              r.model_copy(update={"id": r.id or 1}) for r in results
          ]
      )

      with patch("db.evidence_repositories.is_persistence_enabled", return_value=True), patch(
          "db.evidence_repositories.VerificationRepository",
          return_value=mock_repo,
      ):
          result = await knowledge_state_node(state)

      mock_repo.save_verifications.assert_awaited_once()
      saved = mock_repo.save_verifications.await_args.args[0]
      assert saved[0].knowledge_category == KnowledgeCategory.KNOWN
      assert result["knowledge_state"] is not None
      assert result["knowledge_state"]["metrics"]["known_count"] == 1
      assert result["current_node"] == "writer"

  @pytest.mark.asyncio
  async def test_skips_when_no_verification_results(self):
      state = {
          "material_claims": [_claim(1)],
          "verification_results": [],
          "claim_evidence_relations": [],
          "critique": None,
      }

      result = await knowledge_state_node(state)
      assert result["knowledge_state"] is None
      assert result["current_node"] == "writer"
