import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
import httpx
from anthropic import AsyncAnthropic

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Pipeline Code ---

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
    stance: str = "unchecked"
    reasoning: str = ""

@dataclass
class ClaimResult:
    claim: str
    sources: List[Source] = field(default_factory=list)
    verdict: str = "unverifiable"
    confidence: float = 0.0

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    return raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

async def extract_claims(text: str) -> List[str]:
    # MOCK MODE (FREE): Simple sentence splitting instead of calling Anthropic's Claude
    import re
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    # Return at most 3 sentences as mock claims to keep the search fast
    return [s for s in sentences if len(s) > 10][:3]

async def search_evidence(session: httpx.AsyncClient, claim: str, domains: Optional[List[str]] = None) -> List[Source]:
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set on the server")
    
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": claim,
        "search_depth": "advanced",
        "max_results": MAX_SOURCES_PER_CLAIM,
    }
    if domains:
        payload["include_domains"] = domains

    resp = await session.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        Source(url=r.get("url", ""), title=r.get("title", ""), snippet=(r.get("content", "") or "")[:1500])
        for r in data.get("results", [])
    ]

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

async def _check_one_claim(session: httpx.AsyncClient, claim: str, domains: Optional[List[str]] = None) -> ClaimResult:
    sources = await search_evidence(session, claim, domains)
    if sources:
        sources = list(await asyncio.gather(*[classify_stance(claim, s) for s in sources]))
    return aggregate_claim(claim, sources)

async def run_fact_check(text: str, domains: Optional[List[str]] = None) -> dict:
    text = text.strip()[:MAX_INPUT_CHARS]
    claims = await extract_claims(text)
    if not claims:
        return {"overall": overall_score([]), "claims": []}
    async with httpx.AsyncClient() as session:
        results = list(await asyncio.gather(*[_check_one_claim(session, c, domains) for c in claims]))
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


# --- FastAPI App ---

app = FastAPI(title="RAG Fact Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    domains: Optional[List[str]] = None

@app.post("/api/check")
async def check(req: CheckRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        return await run_fact_check(req.text, req.domains)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fact-check failed: {e}")

@app.get("/api/health")
async def health():
    return {"status": "ok"}
