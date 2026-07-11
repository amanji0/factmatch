import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import run_fact_check, MAX_INPUT_CHARS

app = FastAPI(title="RAG Fact Checker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


@app.post("/api/check")
async def check(req: CheckRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        return await run_fact_check(req.text)
    except RuntimeError as e:
        # e.g. missing TAVILY_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fact-check failed: {e}")


@app.get("/api/health")
async def health():
    return {"status": "ok"}

