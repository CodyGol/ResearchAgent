"""Test suite for core logic (no live LLM calls)."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from state import AgentState, ResearchPlan, SearchResult
from tools.search import BLACKLIST, search_tavily
from utils.serialization import DateTimeJSONEncoder, serialize_for_db


class TestBlacklistFiltering:
    """Test 1: Verify blacklist filtering removes SEO spam."""

    @pytest.mark.asyncio
    async def test_blacklist_filters_medium_com(self):
        """Verify that medium.com results are filtered out."""
        # Mock Tavily API response with mixed results
        mock_response = {
            "results": [
                {
                    "title": "Valid Technical Article",
                    "url": "https://arxiv.org/abs/1234.5678",
                    "content": "Technical content from arxiv",
                    "score": 0.9,
                },
                {
                    "title": "SEO Spam Article",
                    "url": "https://medium.com/@user/article",
                    "content": "SEO spam content",
                    "score": 0.8,
                },
                {
                    "title": "Another Valid Source",
                    "url": "https://github.com/user/repo",
                    "content": "GitHub repository",
                    "score": 0.85,
                },
                {
                    "title": "LinkedIn Spam",
                    "url": "https://www.linkedin.com/posts/article",
                    "content": "LinkedIn spam",
                    "score": 0.7,
                },
            ]
        }

        # Mock settings to avoid needing real API keys
        mock_settings = MagicMock()
        mock_settings.tavily_api_key = "test-key"

        # Mock the Tavily client and async executor
        with patch("tools.search.settings", mock_settings), \
             patch("tools.search.TavilyClient") as mock_client_class, \
             patch("tools.search.asyncio.get_event_loop") as mock_get_loop:
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock the event loop and executor
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock(return_value=mock_response)
            
            # Execute search
            results = await search_tavily("test query", max_results=10)

        # Verify blacklisted domains are filtered out
        urls = [r.url for r in results]
        
        # Should contain valid sources
        assert "https://arxiv.org/abs/1234.5678" in urls
        assert "https://github.com/user/repo" in urls
        
        # Should NOT contain blacklisted domains
        assert "https://medium.com/@user/article" not in urls
        assert "https://www.linkedin.com/posts/article" not in urls
        
        # Should have exactly 2 results (2 valid, 2 filtered)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_blacklist_filters_all_spam_domains(self):
        """Verify all blacklisted domains are filtered."""
        # Create mock response with all blacklisted domains
        mock_response = {
            "results": [
                {
                    "title": f"Spam from {domain}",
                    "url": f"https://{domain}/article",
                    "content": "Spam content",
                    "score": 0.5,
                }
                for domain in BLACKLIST[:3]  # Test first 3 blacklisted domains
            ]
        }

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.tavily_api_key = "test-key"

        with patch("tools.search.settings", mock_settings), \
             patch("tools.search.TavilyClient") as mock_client_class, \
             patch("tools.search.asyncio.get_event_loop") as mock_get_loop:
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock(return_value=mock_response)
            
            results = await search_tavily("test query", max_results=10)

        # All results should be filtered out
        assert len(results) == 0

    def test_blacklist_constant_exists(self):
        """Verify BLACKLIST constant is defined."""
        assert BLACKLIST is not None
        assert isinstance(BLACKLIST, list)
        assert len(BLACKLIST) > 0
        assert "medium.com" in BLACKLIST
        assert "linkedin.com" in BLACKLIST


class TestSerialization:
    """Test 2: Verify datetime serialization works correctly."""

    def test_datetime_json_encoder(self):
        """Verify DateTimeJSONEncoder serializes datetime objects."""
        test_data = {
            "name": "test",
            "timestamp": datetime(2024, 1, 15, 10, 30, 45),
            "nested": {
                "created_at": datetime(2024, 1, 1, 0, 0, 0),
            },
        }

        # Serialize using custom encoder
        json_str = json.dumps(test_data, cls=DateTimeJSONEncoder)
        parsed = json.loads(json_str)

        # Verify datetime objects are converted to ISO strings
        assert isinstance(parsed["timestamp"], str)
        assert parsed["timestamp"] == "2024-01-15T10:30:45"
        assert isinstance(parsed["nested"]["created_at"], str)
        assert parsed["nested"]["created_at"] == "2024-01-01T00:00:00"
        
        # Verify other data is preserved
        assert parsed["name"] == "test"

    def test_serialize_for_db(self):
        """Verify serialize_for_db function handles datetime objects."""
        test_data = {
            "id": 1,
            "name": "test_record",
            "created_at": datetime(2024, 1, 15, 10, 30, 45),
            "metadata": {
                "updated_at": datetime(2024, 1, 16, 12, 0, 0),
                "value": 42,
            },
        }

        # Serialize
        serialized = serialize_for_db(test_data)

        # Verify datetime objects are converted to strings
        assert isinstance(serialized["created_at"], str)
        assert serialized["created_at"] == "2024-01-15T10:30:45"
        assert isinstance(serialized["metadata"]["updated_at"], str)
        assert serialized["metadata"]["updated_at"] == "2024-01-16T12:00:00"
        
        # Verify other data is preserved
        assert serialized["id"] == 1
        assert serialized["name"] == "test_record"
        assert serialized["metadata"]["value"] == 42

    def test_serialize_for_db_no_datetime(self):
        """Verify serialize_for_db works with data that has no datetime objects."""
        test_data = {
            "id": 1,
            "name": "test",
            "value": 42,
            "tags": ["a", "b", "c"],
        }

        serialized = serialize_for_db(test_data)

        # Should be identical (no datetime objects to convert)
        assert serialized == test_data


class TestStateStructure:
    """Test 3: Verify AgentState structure matches Pydantic expectations."""

    def test_agent_state_initialization(self):
        """Verify AgentState can be initialized with required fields."""
        state: AgentState = {
            "user_query": "Test query",
            "research_plan": None,
            "research_results": None,
            "critique": None,
            "final_report": None,
            "current_node": "planner",
            "iteration_count": 0,
            "error": None,
        }

        # Verify structure
        assert state["user_query"] == "Test query"
        assert state["research_plan"] is None
        assert state["current_node"] == "planner"
        assert state["iteration_count"] == 0

    def test_research_plan_creation(self):
        """Verify ResearchPlan can be created with Pydantic validation."""
        plan = ResearchPlan(
            query="Test query",
            sub_queries=["Sub query 1", "Sub query 2"],
            search_terms=["term1", "term2"],
            domains=["arxiv.org", "github.com"],
            required_domains=["arxiv.org"],
        )

        assert plan.query == "Test query"
        assert len(plan.sub_queries) == 2
        assert len(plan.search_terms) == 2
        assert plan.domains == ["arxiv.org", "github.com"]
        assert plan.required_domains == ["arxiv.org"]

    def test_research_plan_defaults(self):
        """Verify ResearchPlan defaults work correctly."""
        plan = ResearchPlan(query="Test query")

        assert plan.query == "Test query"
        assert plan.sub_queries == []
        assert plan.search_terms == []
        assert plan.domains is None
        assert plan.required_domains == []

    def test_search_result_validation(self):
        """Verify SearchResult validates score bounds."""
        # Valid score
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            score=0.5,
        )
        assert result.score == 0.5

        # Score at bounds
        result_min = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            score=0.0,
        )
        assert result_min.score == 0.0

        result_max = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            score=1.0,
        )
        assert result_max.score == 1.0

        # Invalid score should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            SearchResult(
                title="Test",
                url="https://example.com",
                content="Content",
                score=1.5,  # Out of bounds
            )
