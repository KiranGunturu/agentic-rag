"""AWS-native agentic RAG using Strands Agents + Amazon Bedrock.

This is the AWS counterpart to the local OpenAI/ChromaDB scripts. Instead of
embedding and storing chunks yourself, a Bedrock Knowledge Base (Amazon Titan
Text Embeddings V2 over an OpenSearch Serverless vector store) handles chunking,
embedding, and retrieval. A Strands agent running on a Bedrock Claude model
decides when to call the `retrieve` tool and combines the retrieved context with
the model's own knowledge to answer -- the autonomous tool loop that makes this
"agentic" rather than a straight-line pipeline.

Prerequisites
-------------
- AWS credentials on your shell (``aws configure`` or an instance profile).
- Amazon Bedrock model access enabled for the chosen Claude model.
- A Bedrock Knowledge Base created and synced with your documents; note its ID.

Environment (.env or shell):
    AWS_REGION=us-east-2
    KNOWLEDGE_BASE_ID=XXXXXXXXXX
    BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0   # optional
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv
from strands import Agent
from strands_tools import current_time, retrieve, use_aws

logger = logging.getLogger(__name__)

DEFAULT_QUERY = (
    "What do you know about Tesla products? List the current and research ones."
)
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

SYSTEM_PROMPT = """\
You are a product research assistant.
- Use the retrieve tool to look up information in the Bedrock knowledge base
  before answering questions about products.
- Combine what you retrieve with your own knowledge to give a complete answer.
- If the knowledge base has no relevant information, say so instead of guessing.
"""


def build_agent(model_id: str | None = None) -> Agent:
    """Create a Strands agent wired to Bedrock and the knowledge-base tools.

    The `retrieve` tool reads KNOWLEDGE_BASE_ID and AWS_REGION from the
    environment, so make sure those are set before invoking the agent.
    `callback_handler=None` turns off Strands' default token streaming so we
    print the final answer once, matching the local retriever's behavior.
    """
    if not os.getenv("KNOWLEDGE_BASE_ID"):
        raise RuntimeError(
            "KNOWLEDGE_BASE_ID is not set. Create and sync a Bedrock Knowledge "
            "Base, then put its ID in your .env or environment."
        )
    return Agent(
        model=model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        tools=[current_time, use_aws, retrieve],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )


def ask(query: str, model_id: str | None = None) -> str:
    """Run one question through the agent and return the final answer text."""
    agent = build_agent(model_id)
    result = agent(query)
    return str(result)


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Agentic RAG over a Bedrock Knowledge Base using Strands."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Question to ask (default: {DEFAULT_QUERY!r}).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Bedrock model ID (else BEDROCK_MODEL_ID env / default).",
    )
    args = parser.parse_args()
    print(ask(args.query, model_id=args.model_id))


if __name__ == "__main__":
    main()