"""
Agentic RAG demo - Northwind Community Bank (fictional)

Three Bedrock Knowledge Bases, one agent. The agent decides which KBs to search,
searches more than once when a question needs it, and combines the results.

Usage:
    python agentic_rag_demo.py --list                 # show demo questions
    python agentic_rag_demo.py --q 1                  # run demo question 1 with the KBs
    python agentic_rag_demo.py --q 1 --baseline       # same question, NO knowledge bases
    python agentic_rag_demo.py "your own question"    # ad-hoc question

Env vars:
    AWS_REGION        default us-west-2
    KB_PREFIX         only KBs whose name starts with this are used (default "northwind-")
    BEDROCK_MODEL_ID  optional; SDK default model is used if unset
"""
import argparse
import os
import sys

import boto3
from strands import Agent
from strands.memory import MemoryManager
from strands.models import BedrockModel
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
KB_PREFIX = os.getenv("KB_PREFIX", "northwind-")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# Used only when the KB itself has no description set in Bedrock.
# The agent reads these to decide which store to search - keep them specific.
FALLBACK_DESCRIPTIONS = {
    "northwind-policy": (
        "Northwind Community Bank compliance policies: AML alert triage tiers and scenario thresholds "
        "(structuring SCN-01/SCN-01H, wires, dormant accounts), SAR filing SOP with internal deadlines "
        "and approvers, and the customer risk rating (CRR) methodology and risk bands."
    ),
    "northwind-engineering": (
        "Harbor Data Lake engineering runbooks: failed-load recovery steps into Sentinel TM, feed SLAs "
        "and hard cutoffs for WIRE/ACH/CASH/CARD/CUST_MASTER feeds, late-file handling, severity "
        "definitions and the on-call escalation matrix with names and contacts."
    ),
    "northwind-incidents": (
        "Postmortems of past data pipeline incidents affecting AML alerting: timelines, impact, root "
        "cause, resolution, runbook adherence (which steps were skipped) and action items."
    ),
}

SYSTEM_PROMPT = """You are an assistant for the Financial Crimes Compliance (FCC) and Data Platform teams
at Northwind Community Bank.

The knowledge bases cover ONLY: AML/FCC policies, Harbor Data Lake runbooks, and pipeline incident
postmortems. Read each knowledge base's description before deciding whether to search.

Decide first, then act:
1. If the question is about Northwind topics the knowledge bases cover, search them. Break the question into
   parts, search each relevant knowledge base, and search again with a more specific query if results are
   incomplete. Cite the document ID (e.g. NWB-RB-402, PM-2026-0419) for every fact. Show any arithmetic.
2. If the question is about a Northwind topic the knowledge bases clearly do NOT cover (for example retail
   products, fees, HR, lending), do NOT search. Start your answer with:
   "Northwind's knowledge bases don't contain information on this."
   Then give a general industry answer, clearly labeled "General information (not Northwind policy):".
3. If you searched and found nothing relevant, say so the same way, then give the general answer.
4. Never present general knowledge as Northwind policy, and never invent Northwind-specific numbers,
   names or deadlines."""

BASELINE_PROMPT = "You are an assistant for the FCC and Data Platform teams at Northwind Community Bank."

DEMO_QUESTIONS = [
    # 1 - spans all three KBs
    "The wire feed was late yesterday and alerts were missed. What's the SLA, who do I escalate to, "
    "and has this happened before?",
    # 2 - policy lookup + risk band + calculation + SAR timeline
    "A customer rated high-risk had four cash deposits of $2,400 each this week. Does this trigger "
    "escalation, and what's the SAR deadline if it does?",
    # 3 - cross-document comparison against the runbook
    "Compare how incidents PM-2026-0314 and PM-2026-0522 were resolved, and tell me which runbook "
    "steps were missed in each.",
    # 4 - bonus: requires connecting a policy rule to incident history
    "Has the wire feed breached its hard cutoff often enough to require a problem ticket, and was one opened?",

    # 5 - out of scope: agent should NOT search, should say so, then answer generally
    "What is Northwind's overdraft fee for checking accounts, and how do banks typically structure overdraft fees?",
]


def knowledge_base_stores() -> list[BedrockKnowledgeBaseStore]:
    client = boto3.client("bedrock-agent", region_name=AWS_REGION)
    stores = []
    for page in client.get_paginator("list_knowledge_bases").paginate():
        for kb in page.get("knowledgeBaseSummaries", []):
            name = kb["name"]
            if not name.startswith(KB_PREFIX) or kb.get("status") != "ACTIVE":
                continue
            description = kb.get("description") or FALLBACK_DESCRIPTIONS.get(name, name)
            stores.append(
                BedrockKnowledgeBaseStore(
                    name=name,
                    description=description,
                    writable=False,
                    config={"knowledge_base_id": kb["knowledgeBaseId"], "region_name": AWS_REGION},
                )
            )
    return stores


def build_agent(baseline: bool) -> Agent:
    model = BedrockModel(model_id=MODEL_ID, region_name=AWS_REGION) if MODEL_ID \
        else BedrockModel(region_name=AWS_REGION)

    if baseline:
        # No tools, no memory: shows what the model says without the knowledge bases.
        return Agent(model=model, system_prompt=BASELINE_PROMPT)

    stores = knowledge_base_stores()
    if not stores:
        sys.exit(f"No ACTIVE knowledge bases starting with '{KB_PREFIX}' found in {AWS_REGION}.")
    print(f"Using {len(stores)} knowledge base(s): {', '.join(s.name for s in stores)}\n")

    # No use_aws tool: retrieval goes only through the memory stores.
    return Agent(
        model=model,
        memory_manager=MemoryManager(stores=stores),
        system_prompt=SYSTEM_PROMPT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Northwind agentic RAG demo")
    parser.add_argument("question", nargs="?", help="ad-hoc question")
    parser.add_argument("--q", type=int, help="run demo question N (1-based)")
    parser.add_argument("--baseline", action="store_true", help="run without knowledge bases")
    parser.add_argument("--list", action="store_true", help="list demo questions")
    args = parser.parse_args()

    if args.list:
        for i, q in enumerate(DEMO_QUESTIONS, 1):
            print(f"{i}. {q}\n")
        return

    if args.q:
        question = DEMO_QUESTIONS[args.q - 1]
    elif args.question:
        question = args.question
    else:
        parser.error("pass a question, --q N, or --list")

    mode = "BASELINE (no knowledge bases)" if args.baseline else "AGENTIC RAG"
    print(f"=== {mode} ===\nQ: {question}\n")

    # A fresh agent per run keeps demo questions independent of each other.
    # The default callback handler streams the answer and prints each tool call,
    # so the audience can watch the agent choose and repeat searches.
    agent = build_agent(args.baseline)
    agent(question)
    print()


if __name__ == "__main__":
    main()