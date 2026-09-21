"""Answer questions about the HR document using retrieved context."""

from __future__ import annotations

import argparse
import logging

from rag_core import (
    configure_logging,
    embed_text,
    get_collection,
    get_openai_client,
    get_settings,
)

logger = logging.getLogger(__name__)

DEFAULT_QUERY = "what is the leave policy"
N_RESULTS = 3

# System-level rules go in `instructions` (the Responses API's system prompt),
# separate from the retrieved context and the user's question.
SYSTEM_INSTRUCTIONS = """\
You answer questions using only the provided context.
- Give direct, natural answers.
- If the context does not contain the answer, reply exactly "I don't know" and do not guess.
"""

USER_TEMPLATE = """\
Context:
{context}

Question: {query}
"""


def retrieve_context(query: str, n_results: int = N_RESULTS) -> list[str]:
    """Return the most relevant document chunks for a query."""
    query_embedding = embed_text(query)
    results = get_collection().query(
        query_embeddings=[query_embedding],  # Chroma expects a list of embeddings
        n_results=n_results,
    )
    documents = results.get("documents") or [[]]
    return documents[0]


def answer_question(query: str, n_results: int = N_RESULTS) -> str:
    """Retrieve context and generate an answer grounded in it."""
    context = retrieve_context(query, n_results)
    if not context:
        logger.warning("No context retrieved for query: %s", query)
        return "I don't know."

    settings = get_settings()
    response = get_openai_client().responses.create(
        model=settings.answer_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=USER_TEMPLATE.format(context="\n\n".join(context), query=query),
    )
    return response.output_text


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Ask a question about the HR document."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Question to ask (default: {DEFAULT_QUERY!r}).",
    )
    parser.add_argument(
        "-n",
        "--n-results",
        type=int,
        default=N_RESULTS,
        help="Number of chunks to retrieve for context.",
    )
    args = parser.parse_args()
    print(answer_question(args.query, n_results=args.n_results))


if __name__ == "__main__":
    main()