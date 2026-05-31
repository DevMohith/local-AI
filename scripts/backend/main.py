"""
DocAI — backend/main.py

What this file does:
  1. Accepts document uploads (PDF, DOCX, TXT)
  2. Extracts text from them
  3. Decides which AI model to use (Llama 3 or Mistral)
  4. Sends text to that model via llama.cpp's HTTP API
  5. Returns extracted field values as JSON
  6. Serves the React frontend (in Docker / production)

How the AI call works:
  llama.cpp exposes the same HTTP API format as OpenAI.
  So calling it looks identical to calling ChatGPT — just a different URL.
  URL in Docker:  http://llama-a:8081/v1/chat/completions
  URL for OpenAI: https://api.openai.com/v1/chat/completions
  Same JSON in, same JSON out.
"""

import os
import re
import json
import httpx
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn


# =============================================================================
# Configuration
# These come from docker-compose.yml → environment section
# =============================================================================

LLAMA_A = os.getenv("LLAMA_A_URL", "http://localhost:8081")  # Llama 3  (general)
LLAMA_B = os.getenv("LLAMA_B_URL", "http://localhost:8082")  # Mistral  (technical)

# Documents containing these words → route to Mistral (more precise)
TECHNICAL_WORDS = [
    "contract", "agreement", "clause", "liability", "indemnif", "hereinafter",
    "pursuant", "jurisdiction", "plaintiff", "defendant", "arbitration",
    "invoice", "receipt", "vat", "gst", "tax", "balance sheet", "audit",
    "diagnosis", "prescription", "patient", "dosage", "medical", "clinical",
    "compliance", "regulation", "gdpr", "hipaa",
]


# =============================================================================
# App setup
# =============================================================================

app = FastAPI(
    title="DocAI",
    version="1.0.0",
    docs_url="/api/docs",   # Swagger UI at /api/docs (handy during development)
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten this in production if needed
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Step 1: Extract text from uploaded file
# =============================================================================

def extract_text(content: bytes, filename: str) -> str:
    """
    Pull plain text out of PDF, DOCX, or text files.
    Returns a string. Raises HTTPException if extraction fails.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n\n".join(p.strip() for p in pages if p.strip())
            if not text:
                raise HTTPException(400, "PDF appears to be a scanned image — no text found.")
            return text
        except ImportError:
            raise HTTPException(500, "pdfplumber not installed")

    if ext in (".docx", ".doc"):
        try:
            import docx, io
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(500, "python-docx not installed")

    # Plain text: .txt .md .csv or unknown
    return content.decode("utf-8", errors="ignore")


# =============================================================================
# Step 2: Decide which model to use
# =============================================================================

def pick_model(doc_text: str, hint: str = "") -> tuple[str, str]:
    """
    Returns (server_url, human_readable_name).

    Llama 3  = fast, good at general / personal documents
    Mistral  = slower, better at structured extraction from technical docs
    """
    sample = (doc_text[:2000] + " " + hint).lower()
    for word in TECHNICAL_WORDS:
        if word in sample:
            return LLAMA_B, "Mistral 7B"
    return LLAMA_A, "Llama 3 8B"


# =============================================================================
# Step 3: Call the AI model
# =============================================================================

async def call_llm(server_url: str, system: str, user: str) -> str:
    """
    POST to llama.cpp's /v1/chat/completions endpoint.

    This is identical to calling OpenAI's API — same JSON format.
    llama.cpp is just running locally instead of in a remote datacenter.
    """
    payload = {
        "model":       "local",     # llama.cpp ignores this field, uses loaded model
        "messages":    [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.05,        # near-zero = deterministic, good for extraction
        "max_tokens":  2048,
        "stream":      False,
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{server_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    except httpx.ConnectError:
        raise HTTPException(
            503,
            detail=(
                f"Cannot reach AI model at {server_url}. "
                "Run: docker compose up"
            )
        )
    except httpx.TimeoutException:
        raise HTTPException(504, "AI model timed out — try a shorter document")


# =============================================================================
# Step 4: Parse JSON from the model's response
# =============================================================================

def parse_json(raw: str) -> dict:
    """
    LLMs sometimes wrap JSON in markdown fences like:
        ```json
        { "name": "John" }
        ```
    This function strips those and returns a clean dict.
    """
    # Remove ```json ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: find a JSON object anywhere in the text
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}   # Return empty dict if all parsing fails


# =============================================================================
# Default fields (used when the user doesn't specify custom ones)
# =============================================================================

DEFAULT_FIELDS = [
    {"name": "full_name",     "type": "text",     "description": "Full name of the person"},
    {"name": "date",          "type": "date",     "description": "Primary date in YYYY-MM-DD format"},
    {"name": "organization",  "type": "text",     "description": "Company or organization name"},
    {"name": "email",         "type": "email",    "description": "Email address"},
    {"name": "phone",         "type": "tel",      "description": "Phone number"},
    {"name": "address",       "type": "text",     "description": "Full address"},
    {"name": "amount",        "type": "text",     "description": "Any monetary amount"},
    {"name": "reference_no",  "type": "text",     "description": "Reference number or document ID"},
    {"name": "summary",       "type": "textarea", "description": "2-sentence plain English summary"},
]


# =============================================================================
# API Routes
# =============================================================================

@app.get("/api/health")
async def health():
    """
    Called by the React frontend on startup to show the status dots.
    Checks if both AI servers are reachable.
    """
    status = {}
    for name, url in [("llama3", LLAMA_A), ("mistral", LLAMA_B)]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{url}/health")
                status[name] = "ready" if r.status_code == 200 else "error"
        except Exception:
            status[name] = "offline"

    return {
        "status":              "ok",
        "models":              status,
        "offline_capable":     True,
        "data_leaves_machine": False,
    }


@app.post("/api/analyze")
async def analyze(
    file:      UploadFile = File(...),
    form_type: str = Form(default="general"),
    fields:    str = Form(default="[]"),
):
    """
    Main endpoint.
    Upload a document → AI extracts field values → returns JSON.

    fields parameter is a JSON array like:
    [
      {"name": "client_name", "type": "text",  "description": "Client full name"},
      {"name": "sign_date",   "type": "date",  "description": "Date signed"},
      {"name": "total",       "type": "text",  "description": "Total amount due"}
    ]
    If fields is empty, uses the 9 default fields above.
    """

    # Read and size-check the file
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large — maximum 25 MB")

    # Extract text
    doc_text = extract_text(content, file.filename or "document")
    if not doc_text.strip():
        raise HTTPException(400, "No text could be extracted from this document")

    # Parse field list
    try:
        form_fields = json.loads(fields) if fields.strip() not in ("[]", "") else []
    except json.JSONDecodeError:
        form_fields = []

    if not form_fields:
        form_fields = DEFAULT_FIELDS

    # Pick model
    server, model_name = pick_model(doc_text, form_type)

    # Build extraction prompt
    field_descriptions = "\n".join(
        f'  - "{f["name"]}" ({f.get("type","text")}): {f.get("description","")}'
        for f in form_fields
    )
    field_names = [f["name"] for f in form_fields]
    example = "{" + ", ".join(f'"{n}": "..."' for n in field_names[:3]) + ", ...}"

    system_prompt = (
        "You are a precise document data extraction assistant. "
        "Extract ONLY information that is explicitly stated in the document. "
        "Return ONLY a valid JSON object — no explanation, no markdown fences. "
        "Use null for any field not found in the document."
    )

    user_prompt = f"""Extract the following fields from this document:

{field_descriptions}

Document content:
\"\"\"
{doc_text[:6000]}
\"\"\"

Return only a JSON object like: {example}
Rules:
- Only extract what is explicitly written — do not guess
- Use null for missing fields
- Dates must be YYYY-MM-DD format"""

    # Call AI
    raw_response = await call_llm(server, system_prompt, user_prompt)
    extracted    = parse_json(raw_response)

    found = sum(1 for v in extracted.values() if v is not None)

    return {
        "model_used": model_name,
        "fields":     extracted,
        "found":      found,
        "total":      len(form_fields),
        "preview":    doc_text[:400] + ("…" if len(doc_text) > 400 else ""),
    }


@app.post("/api/chat")
async def chat(
    question:    str = Form(...),
    doc_context: str = Form(default=""),
    model:       str = Form(default="auto"),
):
    """Ask a free-form question about an already-analyzed document."""
    server     = LLAMA_B if model == "mistral" else LLAMA_A
    model_name = "Mistral 7B" if model == "mistral" else "Llama 3 8B"

    system = (
        "You are a helpful assistant answering questions about a document. "
        "Be concise and accurate. Only use information from the document provided."
    )
    user = f'Document:\n"""\n{doc_context[:4000]}\n"""\n\nQuestion: {question}'

    answer = await call_llm(server, system, user)
    return {"answer": answer, "model": model_name}


# =============================================================================
# Serve React frontend
# Active only in production (inside Docker).
# During development, Vite runs separately on port 5173.
# =============================================================================

STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"

if STATIC_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(STATIC_DIR / "assets")),
        name="assets",
    )

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Send all non-API routes to React's index.html (SPA routing)."""
        return FileResponse(str(STATIC_DIR / "index.html"))


# =============================================================================
# Dev entry point (not used inside Docker)
# Run directly with: python backend/main.py
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)