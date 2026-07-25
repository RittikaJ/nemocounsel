"""
NemoCounsel — Clause Classification API
POST /classify/stream  → streaming (UI shows live logs per clause)
POST /classify         → all results at once
GET  /health           → check server is alive
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import openai, json, os, re

app = FastAPI(title="NemoCounsel Clause API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

VLLM_URL = os.getenv(
    "VLLM_BASE_URL",
    os.getenv("VLLM_URL", "http://localhost:8002/v1"),
)
MODEL_NAME = os.getenv("MODEL_NAME", "clause_lora")
LABELS = json.loads((Path(__file__).with_name("ledgar_labels.json")).read_text())
SYSTEM_PROMPT = (
    "You are a legal-clause classifier. Given a contract clause, respond with "
    "EXACTLY one label from the provided list -- nothing else, no punctuation, "
    "no explanation."
)

client = openai.OpenAI(base_url=VLLM_URL, api_key="not-needed")


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r'\n+', text) if len(c.strip()) > 30]


def classify_one(clause: str) -> str:
    label_list = ", ".join(LABELS)
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Labels: {label_list}\n\n"
                f"Clause: {clause}\n\n"
                "Which label applies? Answer with exactly one label from the list above."
            )},
        ],
        max_tokens=20,
        temperature=0,
    )
    return r.choices[0].message.content.strip()


class ContractRequest(BaseModel):
    contract: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "endpoint": VLLM_URL}


@app.post("/classify")
def classify_all(req: ContractRequest):
    """Returns all clause classifications at once."""
    clauses = split_clauses(req.contract)
    results = [
        {"index": i + 1, "clause": c, "label": classify_one(c)}
        for i, c in enumerate(clauses)
    ]
    return {"total": len(results), "results": results}


@app.post("/classify/stream")
def classify_stream(req: ContractRequest):
    """Streams one SSE event per clause so UI can show live progress."""
    clauses = split_clauses(req.contract)

    def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'total': len(clauses)})}\n\n"
        results = []
        for i, clause in enumerate(clauses):
            yield f"data: {json.dumps({'type': 'progress', 'index': i+1, 'total': len(clauses), 'clause': clause[:80]})}\n\n"
            label = classify_one(clause)
            result = {"index": i + 1, "clause": clause, "label": label}
            results.append(result)
            yield f"data: {json.dumps({'type': 'result', **result})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'results': results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
