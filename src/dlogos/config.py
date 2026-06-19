"""Central configuration for dLogos.

A single :class:`Settings` (pydantic-settings) reads every environment variable
the system touches and provides sane local-dev defaults so that importing any
module — and running the unit tests — never requires real credentials or
network access. A module-level ``settings`` singleton is the shared accessor.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, sourced from the environment / ``.env``.

    Every field has a default so the object constructs cleanly in tests and in
    a fresh checkout. Secrets default to empty strings; service URLs default to
    local endpoints.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Podcast Index (ingestion) ---
    podcast_index_key: str = Field(default="", alias="PODCAST_INDEX_KEY")
    podcast_index_secret: str = Field(default="", alias="PODCAST_INDEX_SECRET")

    # --- Open-weight extraction endpoint (OpenAI-compatible) ---
    # The smoke run points these at a hosted OpenAI-compatible endpoint
    # (DeepInfra): EXTRACTION_BASE_URL=https://api.deepinfra.com/v1/openai,
    # EXTRACTION_MODEL=deepseek-ai/DeepSeek-V3. The localhost default keeps a
    # fresh checkout / the offline tests self-contained (tests inject fakes and
    # never read these).
    extraction_base_url: str = Field(
        default="http://localhost:8000/v1", alias="EXTRACTION_BASE_URL"
    )
    extraction_api_key: str = Field(
        default="sk-no-key-required", alias="EXTRACTION_API_KEY"
    )
    extraction_model: str = Field(
        default="deepseek-ai/DeepSeek-V3", alias="EXTRACTION_MODEL"
    )

    # --- Open embedding endpoint (OpenAI-compatible; e.g. BGE-M3) ---
    # Smoke run: EMBED_BASE_URL=https://api.deepinfra.com/v1/openai,
    # EMBED_MODEL=BAAI/bge-m3, served behind the same OpenAI ``/embeddings``
    # wire protocol the hosted embedder (resolution/hosted_embedder.py) speaks.
    embed_base_url: str = Field(
        default="http://localhost:8001/v1", alias="EMBED_BASE_URL"
    )
    embed_api_key: str = Field(default="sk-no-key-required", alias="EMBED_API_KEY")
    embed_model: str = Field(default="BAAI/bge-m3", alias="EMBED_MODEL")

    # --- Neo4j / Graphiti graph backend ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(
        default="dlogos-dev-password", alias="NEO4J_PASSWORD"
    )

    # --- Frontier model for the head-to-head eval (strong baseline; both sides) ---
    frontier_base_url: str = Field(
        default="https://api.anthropic.com/v1", alias="FRONTIER_BASE_URL"
    )
    frontier_api_key: str = Field(default="", alias="FRONTIER_API_KEY")
    frontier_model: str = Field(default="claude-opus-4-8", alias="FRONTIER_MODEL")
    # Optional separate endpoint for the web-search-enabled eval arm (arm 2).
    frontier_web_search_base_url: str = Field(
        default="", alias="FRONTIER_WEB_SEARCH_BASE_URL"
    )
    frontier_web_search_api_key: str = Field(
        default="", alias="FRONTIER_WEB_SEARCH_API_KEY"
    )
    frontier_web_search_model: str = Field(
        default="", alias="FRONTIER_WEB_SEARCH_MODEL"
    )

    # --- Resolution: Wikidata SPARQL endpoint (recurring-guest canonicalization) ---
    wikidata_endpoint: str = Field(
        default="https://query.wikidata.org/sparql", alias="WIKIDATA_ENDPOINT"
    )

    # --- ASR: HuggingFace token for the gated pyannote diarization pipeline ---
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # --- ASR: hosted AssemblyAI (diarization + word timestamps, no GPU) ---
    assemblyai_api_key: str = Field(default="", alias="ASSEMBLYAI_API_KEY")


# Module-level singleton — import this, don't construct Settings ad hoc.
settings = Settings()
