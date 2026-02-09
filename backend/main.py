print("🚀 main.py started")
# =====================================
# IMPORT
# =====================================
import os
import time
import joblib
import numpy as np
import torch
import requests
from fastapi import Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# =====================================
# PREPROCESS
# =====================================
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text

# =====================================
# LOAD MODELS (SMART FINAL FIX)
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)
# =====================================
# APP INIT
# =====================================
app = FastAPI(
    title="NahPuean Movie Review API",
    version="2.1"
)


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)





models = {}
model_registry = {} 

def is_transformer_dir(path: str) -> bool:
    return (
        os.path.isdir(os.path.join(path, "model")) or
        os.path.exists(os.path.join(path, "config.json"))
    )

for v in os.listdir(MODEL_DIR):
    vp = os.path.join(MODEL_DIR, v)
    if not os.path.isdir(vp):
        continue

    if is_transformer_dir(vp):
        model_registry[v] = {
            "type": "transformer",
            "path": vp
        }

    elif v == "v5_ensemble":
        model_registry[v] = {
            "type": "ensemble",
            "path": vp
        }

    else:
        model_registry[v] = {
            "type": "single",
            "path": vp
        }

print("📦 Available models:", list(model_registry.keys()))

def load_model(name: str):
    if name in models:
        return models[name]

    meta = model_registry.get(name)
    if not meta:
        return None

    print(f"🧠 Loading model {name}...")
    path = meta["path"]

    if meta["type"] == "transformer":
        if name == "v6_transformer":
            model = DistilBertForSequenceClassification.from_pretrained(path)
            tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True
            )
        model.eval()
        model = model.to("cpu")

        models[name] = {
            "type": "transformer",
            "model": model,
            "tokenizer": tokenizer
        }

    elif meta["type"] == "ensemble":
        models[name] = {
            "type": "ensemble",
            "vectorizer": joblib.load(os.path.join(path, "vectorizer.joblib")),
            "lr": joblib.load(os.path.join(path, "logistic.joblib")),
            "svm": joblib.load(os.path.join(path, "svm.joblib"))
        }

    else:
        models[name] = {
            "type": "single",
            "model": joblib.load(os.path.join(path, "model.joblib")),
            "vectorizer": joblib.load(os.path.join(path, "vectorizer.joblib"))
        }

    return models[name]
def unload_model(name: str):
    if name in models:
        print(f"🧹 Unloading {name}")
        del models[name]
        torch.cuda.empty_cache()

# =====================================
# SCHEMA
# =====================================
class CompareRequest(BaseModel):
    text: str
    model_a: str
    model_b: str | None = None
    tags: list[str] = []
    movie_name: str | None = None

# =====================================
# UTILS
# =====================================
def extract_keywords(text: str, limit=5):
    words = text.replace("\n", " ").split()
    uniq = list(dict.fromkeys(words))
    return uniq[:limit]

# =====================================
# PREDICT FUNCTIONS
# =====================================
def predict_transformer(m, text):
    start = time.time()
    text = preprocess_text(text)

    inputs = m["tokenizer"](
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = m["model"](**inputs).logits
        probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    return {
        "sentiment": "Positive" if pred == 1 else "Negative",
        "confidence": round(confidence, 3),
        "latency_ms": round((time.time() - start) * 1000, 2)
    }

def predict_single(m, text):
    start = time.time()
    text = preprocess_text(text)

    X = m["vectorizer"].transform([text])
    pred = m["model"].predict(X)[0]
    score = m["model"].decision_function(X)[0]
    confidence = 1 / (1 + np.exp(-abs(score)))

    return {
        "sentiment": "Positive" if pred == 1 else "Negative",
        "confidence": round(confidence, 3),
        "latency_ms": round((time.time() - start) * 1000, 2)
    }

def predict_ensemble(m, text):
    start = time.time()
    text = preprocess_text(text)

    X = m["vectorizer"].transform([text])

    score = (
        m["lr"].decision_function(X)[0] +
        m["svm"].decision_function(X)[0]
    ) / 2

    pred = 1 if score > 0 else 0
    confidence = 1 / (1 + np.exp(-abs(score)))

    return {
        "sentiment": "Positive" if pred == 1 else "Negative",
        "confidence": round(confidence, 3),
        "latency_ms": round((time.time() - start) * 1000, 2)
    }

# =====================================
# ROUTES
# =====================================
@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    return list(model_registry.keys())

@app.get("/model/info")
def model_info():
    return {
        k: {"type": v["type"]}
        for k, v in model_registry.items()
    }

@app.post("/predict")
def predict(req: CompareRequest):
    m = load_model(req.model_a)
    if not m:
        return {"error": "model not found"}

    try:
        if m["type"] == "transformer":
            return predict_transformer(m, req.text)
        elif m["type"] == "ensemble":
            return predict_ensemble(m, req.text)
        else:
            return predict_single(m, req.text)
    finally:
        unload_model(req.model_a)

@app.post("/compare")
def compare(req: CompareRequest):
    results = {}

    for name in [req.model_a, req.model_b]:
        if not name:
            continue

        m = load_model(name)
        if not m:
            results[name] = {"error": "model not found"}
            continue

        try:
            if m["type"] == "transformer":
                r = predict_transformer(m, req.text)
            elif m["type"] == "ensemble":
                r = predict_ensemble(m, req.text)
            else:
                r = predict_single(m, req.text)

            r["keywords"] = extract_keywords(req.text)
            results[name] = r

        finally:
            unload_model(name)

    return {
        "movie_name": req.movie_name,
        "tags": req.tags,
        "results": results
    }

TMDB_KEY = os.getenv("TMDB_KEY")


@app.get("/search_movie")
def search_movie(q: str = Query("", min_length=1)):
    if not q.strip():
        return []
    r = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={
            "api_key": TMDB_KEY,
            "query": q,
            "language": "en-US"
        }
    )

    results = r.json().get("results", [])[:5]

    return [
        {
            "title": m["title"],
            "year": m["release_date"][:4] if m.get("release_date") else "",
            "genre_ids": m.get("genre_ids", [])
        }
        for m in results
    ]


