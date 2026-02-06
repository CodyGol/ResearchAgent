"""Configuration management using pydantic-settings for validation."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic API Configuration
    anthropic_api_key: str = Field(
        ...,
        description="Anthropic API key for Claude 4.5 Sonnet",
        validation_alias="ANTHROPIC_API_KEY",
    )

    # Tavily API Configuration
    tavily_api_key: str = Field(
        ...,
        description="Tavily API key for web search",
        validation_alias="TAVILY_API_KEY",
    )

    # Model Configuration
    model_name: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Anthropic model identifier",
        validation_alias="ANTHROPIC_MODEL",
    )

    # Research Configuration
    max_research_iterations: int = Field(
        default=3,
        description="Maximum number of research-critic cycles",
        ge=1,
        le=10,
    )

    quality_threshold: float = Field(
        default=0.7,
        description="Minimum quality score (0-1) to proceed to writer",
        ge=0.0,
        le=1.0,
    )

    # Supabase Configuration (Optional - system works without it)
    supabase_url: str | None = Field(
        default=None,
        description="Supabase project URL",
        validation_alias="SUPABASE_URL",
    )

    supabase_key: str | None = Field(
        default=None,
        description="Supabase anon/service role API key",
        validation_alias="SUPABASE_KEY",
    )

    # Database Configuration
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of research plans in Supabase",
        validation_alias="ENABLE_CACHING",
    )

    cache_ttl_hours: int = Field(
        default=24,
        description="Cache TTL in hours for research plans",
        ge=1,
        le=168,  # Max 1 week
        validation_alias="CACHE_TTL_HOURS",
    )

    # LangSmith Observability Configuration
    langchain_project: str = Field(
        default="the-oracle",
        description="LangSmith project name for tracing",
        validation_alias="LANGCHAIN_PROJECT",
    )

    environment: str = Field(
        default="local-dev",
        description="Environment identifier (local-dev, staging, production)",
        validation_alias="ENVIRONMENT",
    )


# Global settings instance
try:
    settings = Settings()
    
    # Warn if Supabase is not configured but caching is enabled
    if settings.enable_caching and (not settings.supabase_url or not settings.supabase_key):
        import warnings
        warnings.warn(
            "Supabase not configured but ENABLE_CACHING=true. "
            "Add SUPABASE_URL and SUPABASE_KEY to .env to enable caching.",
            UserWarning
        )
        settings.enable_caching = False  # Disable caching if Supabase not available
        
except Exception as e:
    import sys

    print(
        f"❌ Error loading configuration: {e}\n"
        "Please check your .env file format.\n"
        "Each line should be: KEY=value (no spaces around =)\n"
        "Example:\n"
        "ANTHROPIC_API_KEY=sk-ant-...\n"
        "TAVILY_API_KEY=tvly-...\n"
        "SUPABASE_URL=https://xxx.supabase.co (optional)\n"
        "SUPABASE_KEY=xxx (optional)\n",
        file=sys.stderr,
    )
    sys.exit(1)
