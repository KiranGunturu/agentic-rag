"""Ingest the HR PDF into the Chroma vector store."""

from __future__ import annotations

import logging

from chunking import create_chunks
from rag_core import configure_logging, embed_texts, get_collection

logger = logging.getLogger(__name__)

DOC_PATH = "docs/HR.pdf"
CHUNK_SIZE = 500
OVERLAP_SIZE = 100


def ingest(
    doc_path: str = DOC_PATH,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap_size: int = OVERLAP_SIZE,
) -> int:
    """Chunk the PDF, embed the chunks, and store them in Chroma.

    Returns the number of chunks stored.
    """
    chunks, ids, metadatas = create_chunks(
        doc_path, chunk_size=chunk_size, overlap_size=overlap_size
    )
    if not chunks:
        logger.warning("No chunks extracted from %s; nothing to ingest.", doc_path)
        return 0

    logger.info("Embedding %d chunks from %s ...", len(chunks), doc_path)
    embeddings = embed_texts(chunks)

    collection = get_collection()
    # upsert (not add) so re-running with the same IDs updates in place
    # instead of raising a duplicate-ID error.
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(
        "Stored %d chunks in '%s' (collection now holds %d documents).",
        len(chunks),
        collection.name,
        collection.count(),
    )
    return len(chunks)


def main() -> None:
    configure_logging()
    ingest()


if __name__ == "__main__":
    main()