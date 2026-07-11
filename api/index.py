import os
import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Pipeline Code ---

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_SOURCES_PER_CLAIM = 4
MAX_INPUT_CHARS = 8000

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
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

async def _groq_request(session: httpx.AsyncClient, messages: list, max_tokens: int = 1024, retries: int = 3) -> dict:
    """Make a Groq API request with automatic retry on rate limits."""
    for attempt in range(retries):
        resp = await session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        data = resp.json()
        if "error" in data and "rate_limit" in data["error"].get("type", "").lower():
            wait = 10 * (attempt + 1)
            await asyncio.sleep(wait)
            continue
        if "error" in data and "rate limit" in data["error"].get("message", "").lower():
            wait = 10 * (attempt + 1)
            await asyncio.sleep(wait)
            continue
        return data
    return data  # return last response even if still rate limited

async def extract_claims(text: str) -> List[str]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set on the server")
    prompt = f"""
You are an expert fact-checker. Extract up to 5 atomic, verifiable factual claims from the following text.
Ignore opinions, questions, and vague statements. Return ONLY a JSON list of short strings.

Text:
{text}
"""
    async with httpx.AsyncClient() as session:
        try:
            data = await _groq_request(session, [{"role": "user", "content": prompt}], max_tokens=512)
            if "error" in data:
                raise RuntimeError(f"Groq API Error: {data['error'].get('message', 'Unknown error')}")
            
            if "choices" in data and len(data["choices"]) > 0:
                text_response = data["choices"][0]["message"]["content"]
                claims = json.loads(_strip_fences(text_response))
                return claims[:5] if isinstance(claims, list) else []
            else:
                raise RuntimeError("Groq returned an empty response.")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise e
            raise RuntimeError(f"Failed to extract claims: {str(e)}")


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

async def classify_stance(session: httpx.AsyncClient, claim: str, source: Source) -> Source:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set on the server")
    prompt = f"""
You are an expert fact-checker. Decide if this source SUPPORTS, CONTRADICTS, or is UNCLEAR about the claim.
Return ONLY JSON: {{"stance": "supports|contradicts|unclear", "reasoning": "one sentence"}}

Claim: {claim}
Source: {source.title} - {source.snippet[:500]}
"""
    try:
        data = await _groq_request(session, [{"role": "user", "content": prompt}], max_tokens=150)
        if "error" in data:
            source.stance = "unclear"
            source.reasoning = f"AI rate limited, skipping this source."
            return source
            
        if "choices" in data and len(data["choices"]) > 0:
            text_response = data["choices"][0]["message"]["content"]
            parsed = json.loads(_strip_fences(text_response))
            source.stance = parsed.get("stance", "unclear")
            source.reasoning = parsed.get("reasoning", "")
        else:
            source.stance = "unclear"
            source.reasoning = "AI returned an empty response."
    except Exception as e:
        source.stance = "unclear"
        source.reasoning = f"Failed to parse AI response: {e}"
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
        # Process sources sequentially to avoid rate limits
        classified = []
        for s in sources:
            result = await classify_stance(session, claim, s)
            classified.append(result)
            await asyncio.sleep(2)  # Small delay between Groq calls
        sources = classified
    return aggregate_claim(claim, sources)

async def run_fact_check(text: str, domains: Optional[List[str]] = None) -> dict:
    text = text.strip()[:MAX_INPUT_CHARS]
    claims = await extract_claims(text)
    if not claims:
        return {"overall": overall_score([]), "claims": []}
    async with httpx.AsyncClient() as session:
        # Process claims sequentially to avoid Groq rate limits
        results = []
        for c in claims:
            result = await _check_one_claim(session, c, domains)
            results.append(result)
            await asyncio.sleep(2)  # Small delay between claims
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
