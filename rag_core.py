"""Shared configuration, clients, and helpers for the HR RAG example.

Everything that both ingestion and retrieval need lives here so the two
scripts stay small and can't drift apart (e.g. embedding with different
models or dimensions on the write vs. read side).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:  # avoids a version-fragile runtime import
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, sourced from environment variables / .env."""

    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    # 1536 is the native size for text-embedding-3-small. 256/512 trade a little
    # quality for storage and speed. Anything tiny (e.g. 4) makes retrieval useless.
    embedding_dimensions: int = 1536
    answer_model: str = "gpt-5.6-sol"
    chroma_path: str = "./chroma_db"
    collection_name: str = "hr_collection"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )
        return cls(
            openai_api_key=api_key,
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimensions=int(
                os.getenv("EMBEDDING_DIMENSIONS", str(cls.embedding_dimensions))
            ),
            answer_model=os.getenv("ANSWER_MODEL", cls.answer_model),
            chroma_path=os.getenv("CHROMA_PATH", cls.chroma_path),
            collection_name=os.getenv("COLLECTION_NAME", cls.collection_name),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and reuse them for the process lifetime."""
    return Settings.from_env()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


@lru_cache(maxsize=1)
def get_collection() -> "Collection":
    """Open the persistent Chroma collection, creating it if needed."""
    settings = get_settings()
    chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    return chroma_client.get_or_create_collection(settings.collection_name)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed one or more texts in a single API call.

    The single-text path (`embed_text`) delegates here so indexing and querying
    always use identical model + dimension settings.
    """
    settings = get_settings()
    response = get_openai_client().embeddings.create(
        model=settings.embedding_model,
        input=list(texts),
        dimensions=settings.embedding_dimensions,
    )
    return [item.embedding for item in response.data]


def embed_text(text: str) -> list[float]:
    """Embed a single text."""
    return embed_texts([text])[0]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )