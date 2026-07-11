"""
Core RAG fact-checking pipeline, rewritten for async concurrency so a
multi-claim article doesn't take forever to check.

Pipeline (same 4 stages as the prototype, now parallelized):
  1. Claim extraction      -> one Claude call splits text into atomic claims
  2. Retrieval (RAG)       -> Tavily web search per claim, run concurrently
  3. Stance classification -> Claude judges each (claim, source) pair, concurrently
  4. Aggregation           -> per-claim verdict, then an overall weighted score
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-4-6"
MAX_SOURCES_PER_CLAIM = 4
MAX_INPUT_CHARS = 8000

client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


@dataclass
class Source:
    url: str
    title: str
    snippet: str
    stance: str = "unchecked"  # supports | contradicts | unclear
    reasoning: str = ""


@dataclass
class ClaimResult:
    claim: str
    sources: List[Source] = field(default_factory=list)
    verdict: str = "unverifiable"  # true | false | mixed | unverifiable
    confidence: float = 0.0


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    return raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


# --------------------------------------------------------------------------
# Step 1: Claim extraction
# --------------------------------------------------------------------------

async def extract_claims(text: str) -> List[str]:
    # MOCK MODE (FREE): Simple sentence splitting instead of calling Anthropic's Claude
    import re
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    # Return at most 3 sentences as mock claims to keep the search fast
    return [s for s in sentences if len(s) > 10][:3]


# --------------------------------------------------------------------------
# Step 2: Retrieval (RAG) via Tavily live web search
# --------------------------------------------------------------------------

async def search_evidence(session: httpx.AsyncClient, claim: str) -> List[Source]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set on the server")

    resp = await session.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": claim,
            "search_depth": "advanced",
            "max_results": MAX_SOURCES_PER_CLAIM,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        Source(url=r.get("url", ""), title=r.get("title", ""), snippet=(r.get("content", "") or "")[:1500])
        for r in data.get("results", [])
    ]


# --------------------------------------------------------------------------
# Step 3: Stance classification
# --------------------------------------------------------------------------

async def classify_stance(claim: str, source: Source) -> Source:
    # MOCK MODE (FREE): Randomly assign a stance instead of calling Anthropic's Claude
    import random
    stances = ["supports", "contradicts", "unclear"]
    source.stance = random.choice(stances)
    if source.stance == "supports":
        source.reasoning = "This fact is matching from this source."
    elif source.stance == "contradicts":
        source.reasoning = "This fact is contradicted by this source."
    else:
        source.reasoning = "This source is unclear about the fact."
    return source


# --------------------------------------------------------------------------
# Step 4: Aggregation
# --------------------------------------------------------------------------

def aggregate_claim(claim: str, sources: List[Source]) -> ClaimResult:
    supports = sum(1 for s in sources if s.stance == "supports")
    contradicts = sum(1 for s in sources if s.stance == "contradicts")

    if supports + contradicts == 0:
        verdict, confidence = "unverifiable", 0.0
    elif contradicts == 0:
        verdict, confidence = "true", supports / max(len(sources), 1)
    elif supports == 0:
        verdict, confidence = "false", contradicts / max(len(sources), 1)
    else:
        verdict, confidence = "mixed", 0.5

    return ClaimResult(claim=claim, sources=sources, verdict=verdict, confidence=round(confidence, 2))


def overall_score(results: List[ClaimResult]) -> dict:
    if not results:
        return {"score": None, "claim_counts": {}, "total_claims": 0}

    weights = {"true": 1.0, "mixed": 0.5, "false": 0.0, "unverifiable": None}
    scored = [weights[r.verdict] for r in results if weights[r.verdict] is not None]
    score = round(100 * sum(scored) / len(scored), 1) if scored else None

    counts: dict = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    return {"score": score, "claim_counts": counts, "total_claims": len(results)}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

async def _check_one_claim(session: httpx.AsyncClient, claim: str) -> ClaimResult:
    sources = await search_evidence(session, claim)
    if sources:
        sources = list(await asyncio.gather(*[classify_stance(claim, s) for s in sources]))
    return aggregate_claim(claim, sources)


async def run_fact_check(text: str) -> dict:
    text = text.strip()[:MAX_INPUT_CHARS]

    claims = await extract_claims(text)
    if not claims:
        return {"overall": overall_score([]), "claims": []}

    async with httpx.AsyncClient() as session:
        results = list(await asyncio.gather(*[_check_one_claim(session, c) for c in claims]))

    return {
        "overall": overall_score(results),
        "claims": [
            {
                "claim": r.claim,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "sources": [
                    {"url": s.url, "title": s.title, "stance": s.stance, "reasoning": s.reasoning}
                    for s in r.sources
                ],
            }
            for r in results
        ],
    }
