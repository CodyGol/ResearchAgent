"""Search tool with retry logic and structured error handling."""

import asyncio
import logging
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from tavily import TavilyClient

from config import settings
from state import SearchResult

# Domain blacklist for SEO spam filtering (Sniper Protocol)
BLACKLIST = [
    "medium.com",
    "towardsdatascience.com",
    "linkedin.com",
    "pinterest.com",
    "facebook.com",
    "instagram.com",
]

logger = logging.getLogger(__name__)


class SearchErrorType(str, Enum):
    """Categorization of search errors for structured handling."""

    RETRYABLE = "retryable"  # Network, rate limit, temporary failures
    FATAL = "fatal"  # Invalid API key, malformed query, permanent failures


class SearchError(Exception):
    """Structured search error with type classification."""

    def __init__(
        self,
        message: str,
        error_type: SearchErrorType,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


async def search_tavily(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    domains: list[str] | None = None,
) -> list[SearchResult]:
    """
    Execute Tavily search with exponential backoff retry logic.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
        domains: Optional list of domains to filter results (e.g., ['arxiv.org', 'github.com'])

    Returns:
        List of SearchResult objects

    Raises:
        SearchError: If search fails after all retries
    """
    # Initialize Tavily client
    client = TavilyClient(api_key=settings.tavily_api_key)

    # Build search parameters
    search_params: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",  # Use advanced search for better results
    }
    
    # Add domain filtering if provided (reduces SEO spam for technical queries)
    if domains:
        search_params["include_domains"] = domains

    # Execute search (Tavily Python SDK is synchronous, so we run in executor)
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.search(**search_params),
        )

        # Transform Tavily response to SearchResult objects
        results = []
        filtered_count = 0
        
        for item in response.get("results", []):
            url = item.get("url", "")
            
            # Extract domain from URL
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                # Remove 'www.' prefix for comparison
                domain = domain.replace("www.", "")
            except Exception:
                # If URL parsing fails, check if URL string contains blacklisted domain
                domain = url.lower()
            
            # Check if domain is blacklisted (exact match or subdomain)
            # e.g., "medium.com" matches "medium.com" and "subdomain.medium.com"
            is_blacklisted = any(
                domain == blacklisted or domain.endswith(f".{blacklisted}")
                for blacklisted in BLACKLIST
            )
            
            if is_blacklisted:
                filtered_count += 1
                logger.warning(
                    f"🚫 Filtered blacklisted domain: {domain} (URL: {url[:80]}...)",
                    extra={"domain": domain, "url": url, "title": item.get("title", "Untitled")}
                )
                continue  # Skip this result
            
            results.append(
                SearchResult(
                    title=item.get("title", "Untitled"),
                    url=url,
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),  # Tavily provides relevance score
                )
            )
        
        # Log filtering summary
        if filtered_count > 0:
            logger.info(f"🔍 Sniper Protocol: Filtered {filtered_count} blacklisted result(s)")

        return results

    except Exception as e:
        # Classify error type for structured handling
        error_str = str(e).lower()
        if "api" in error_str and "key" in error_str:
            raise SearchError(
                f"Invalid Tavily API key: {str(e)}",
                error_type=SearchErrorType.FATAL,
            ) from e
        elif "rate limit" in error_str or "429" in error_str:
            raise SearchError(
                f"Rate limit exceeded: {str(e)}",
                error_type=SearchErrorType.RETRYABLE,
                retry_after=60.0,  # Wait 60 seconds for rate limit
            ) from e
        else:
            # Network errors, timeouts, etc. are retryable
            raise SearchError(
                f"Tavily search failed: {str(e)}",
                error_type=SearchErrorType.RETRYABLE,
            ) from e


async def search_tavily_with_retry(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    domains: list[str] | None = None,
) -> list[SearchResult]:
    """
    Wrapper that implements retry logic with exponential backoff.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries
        domains: Optional list of domains to filter results

    Returns:
        List of SearchResult objects

    Raises:
        SearchError: If all retries are exhausted
    """
    last_error: SearchError | None = None

    for attempt in range(max_retries):
        try:
            return await search_tavily(query, max_results, max_retries, retry_delay, domains)
        except SearchError as e:
            last_error = e
            # Fatal errors should not be retried
            if e.error_type == SearchErrorType.FATAL:
                raise

            # Retryable errors: wait and retry
            if attempt < max_retries - 1:
                # Use retry_after if provided, otherwise exponential backoff
                delay = e.retry_after if e.retry_after else retry_delay * (2**attempt)
                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                raise SearchError(
                    f"Search failed after {max_retries} attempts: {str(e)}",
                    error_type=SearchErrorType.RETRYABLE,
                ) from last_error
        except Exception as e:
            # Unexpected error type - wrap it
            raise SearchError(
                f"Unexpected search error: {str(e)}",
                error_type=SearchErrorType.RETRYABLE,
            ) from e

    # Should never reach here, but type checker needs it
    raise SearchError(
        "Unexpected error in search retry logic",
        error_type=SearchErrorType.FATAL,
    )
