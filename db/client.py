"""Supabase client initialization and utilities."""

from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:
    from supabase import Client, create_client


def get_supabase_client() -> "Client":
    """
    Initialize and return Supabase client.

    Tries full Supabase client first, falls back to postgrest if needed.

    Returns:
        Supabase client instance

    Raises:
        ImportError: If neither supabase nor postgrest is installed
        ValueError: If Supabase credentials are invalid
    """
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError(
            "Supabase credentials not configured. "
            "Set SUPABASE_URL and SUPABASE_KEY in .env. "
            "Or set ENABLE_CACHING=false to disable Supabase features."
        )

    # Try full Supabase client first
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_key)
        return client
    except ImportError:
        # Fallback to postgrest directly (works without storage dependencies)
        try:
            from postgrest import SyncPostgrestClient
            
            class PostgrestClientWrapper:
                """Wrapper to make postgrest client compatible with Supabase client interface."""

                def __init__(self, url: str, key: str):
                    self.url = url
                    self.key = key
                    # Supabase REST API endpoint
                    base_url = f"{url}/rest/v1"
                    self._client = SyncPostgrestClient(
                        base_url=base_url,
                        schema="public",
                        headers={
                            "apikey": key,
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                            "Prefer": "return=representation",
                        },
                    )

                def table(self, table_name: str):
                    """Return a table query builder compatible with Supabase interface."""
                    return self._client.from_(table_name)

            client = PostgrestClientWrapper(settings.supabase_url, settings.supabase_key)
            return client  # type: ignore
        except ImportError:
            raise ImportError(
                "Neither supabase nor postgrest package installed. "
                "Run: pip install postgrest httpx"
            ) from None


# Global client instance (lazy initialization)
_client = None


def get_client() -> "Client":
    """Get or create global Supabase client instance."""
    global _client
    if _client is None:
        _client = get_supabase_client()
    return _client
