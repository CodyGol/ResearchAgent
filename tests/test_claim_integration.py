"""Integration tests for claim extraction with F1 fixture."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.models import Evidence, EvidenceType, ExtractionMethod
from services.claim_pipeline import process_evidence_for_claims
from services.claim_schemas import CandidateClaimItem, ClaimExtractionOutput


F1_QUESTION = "Who won the 2023 Formula 1 World Championship?"

F1_EVIDENCE = (
    "Max Verstappen secured his third Formula 1 world championship "
    "during the Qatar Sprint in October 2023."
)


def _mock_llm_for_f1() -> MagicMock:
    candidates = [
        CandidateClaimItem(
            claim_text="Max Verstappen won the 2023 Formula 1 World Championship.",
            claim_type="factual",
            importance="high",
            temporal_scope="2023",
            support_basis="direct",
        ),
        CandidateClaimItem(
            claim_text="The 2023 title was Max Verstappen's third Formula 1 World Championship.",
            claim_type="factual",
            importance="high",
            temporal_scope="2023",
            support_basis="direct",
        ),
        CandidateClaimItem(
            claim_text="Verstappen secured the title during the Qatar Sprint.",
            claim_type="factual",
            importance="medium",
            support_basis="direct",
        ),
    ]
    output = ClaimExtractionOutput(claims=candidates)
    mock = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=output)
    mock.with_structured_output = MagicMock(return_value=structured)
    return mock


class TestF1Integration:
    @pytest.mark.asyncio
    async def test_f1_claims_extracted_and_linked(self):
        evidence = Evidence(
            id=1,
            source_id=1,
            research_run_id=1,
            exact_text=F1_EVIDENCE,
            evidence_type=EvidenceType.DIRECT_QUOTE,
            extraction_method=ExtractionMethod.LLM,
            is_validated=True,
        )

        with patch("db.evidence_repositories.is_persistence_enabled", return_value=False):
            claims, material, relations, metrics = await process_evidence_for_claims(
                [evidence],
                F1_QUESTION,
                research_run_id=1,
                llm=_mock_llm_for_f1(),
                use_llm_validation=False,
            )

        assert metrics.evidence_items_processed == 1
        assert metrics.claims_accepted == 3
        assert metrics.unique_claims_persisted == 3
        assert len(relations) == 3

        texts = {c.text for c in claims}
        assert any("2023" in t and "Verstappen" in t for t in texts)
        assert all(c.metadata.get("support_basis") == "direct" for c in claims)

        for rel in relations:
            assert rel.evidence_id == 1
            assert rel.relationship.value == "supports"
