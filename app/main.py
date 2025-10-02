from fastapi import FastAPI, HTTPException, Request, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, HTMLResponse
from pydantic import BaseModel, Field
from uuid import uuid4
import os, time

from .model import SentimentModel

# -------------------- Config --------------------
MAX_CHARS = int(os.getenv("MAX_CHARS", "50000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "custom")  # 'custom' | 'transformers'
MODEL_PATH = os.getenv("MODEL_PATH", str(os.path.join(os.path.dirname(__file__), "..", "artifacts", "model.pth")))
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "")
TRANSFORMERS_MODEL_NAME = os.getenv("TRANSFORMERS_MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
DEVICE = os.getenv("DEVICE", "cpu")

# -------------------- App --------------------
app = FastAPI(
    default_response_class=ORJSONResponse,
    title="Sentiment Service",
    version=os.getenv("MODEL_VERSION", "0.1.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Model --------------------
model = SentimentModel(
    backend=MODEL_BACKEND,
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    hf_model_name=TRANSFORMERS_MODEL_NAME,
    device=DEVICE
)

# -------------------- Schemas --------------------
class InText(BaseModel):
    # keep a conservative max_length to fail fast on huge JSON bodies
    text: str = Field(..., min_length=1, max_length=MAX_CHARS, description="UTF-8 text to analyze")

# -------------------- Helpers --------------------
def _predict_and_pack(text: str) -> dict:
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"Text too long (> {MAX_CHARS} chars).")

    t0 = time.time()
    try:
        pred = model.predict(text)
    except Exception as e:
        # NOTE: in production you'd log request_id + stacktrace
        raise HTTPException(status_code=500, detail=f"inference_error: {e}")

    latency_ms = round((time.time() - t0) * 1000.0, 2)
    return {
        "request_id": str(uuid4()),
        "model_backend": MODEL_BACKEND,
        "model_version": model.model_version,
        "latency_ms": latency_ms,
        "label": pred["label"],
        "score": float(pred["score"]),
        "probs": {k: float(v) for k, v in pred.get("probs", {}).items()},
        # If you later expose tokenization stats, add: "tokens_used": N, "truncated": bool
        "tokens": pred.get("tokens", []),
    }

# -------------------- Routes --------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_backend": MODEL_BACKEND, "model_version": model.model_version}

@app.get("/", response_class=HTMLResponse)
def index():
    # tiny browser form for quick manual testing
    return """
    <html><body>
      <h3>Sentiment quick test</h3>
      <form action="/predict/form" method="post">
        <textarea name="text" rows="10" cols="80" placeholder="Paste review text here..."></textarea><br/>
        <button type="submit">Analyze</button>
      </form>
      <p>Or use <code>/docs</code> for the interactive API.</p>
    </body></html>
    """

@app.post("/predict")
def predict(inp: InText):
    text = inp.text.strip()
    return _predict_and_pack(text)

@app.post("/predict/plain", response_class=ORJSONResponse)
async def predict_plain(body: str = Body(..., media_type="text/plain")):
    text = body.strip()
    return _predict_and_pack(text)

@app.post("/predict/form", response_class=ORJSONResponse)
def predict_form(text: str = Form(...)):
    return _predict_and_pack(text.strip())

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # silence browser favicon requests
