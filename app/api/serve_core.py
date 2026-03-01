"""自研后端 API 主链路（纯 numpy）"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.models import TinyLM, TransformerLM
from app.core.tokenizer import CharTokenizer


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str


@dataclass
class State:
    model: TinyLM | None = None
    tokenizer: CharTokenizer | None = None


state = State()
app = FastAPI(title="LLM Core API", version="0.1.0")


def _load_or_init():
    ckpt = os.getenv("LLM_CHECKPOINT", "checkpoints/model_core.pkl")
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            payload = pickle.load(f)
        tok = CharTokenizer.from_dict(payload["tokenizer"])
        model_cfg = payload["model"]

        if all(k in model_cfg for k in ("n_layers", "n_heads", "max_seq_len")):
            model = TransformerLM(
                vocab_size=model_cfg["vocab_size"],
                n_embd=model_cfg["n_embd"],
                n_layers=model_cfg["n_layers"],
                n_heads=model_cfg["n_heads"],
                max_seq_len=model_cfg["max_seq_len"],
                dropout=model_cfg.get("dropout", 0.1),
            )
        else:
            model = TinyLM(vocab_size=model_cfg["vocab_size"], n_embd=model_cfg["n_embd"])

        state_dict = model_cfg["state_dict"]
        for i, p in enumerate(model.parameters()):
            key = f"param_{i}"
            if key not in state_dict:
                raise ValueError(f"checkpoint 缺少参数: {key}")
            src = state_dict[key]
            if p.data.shape != src.shape:
                raise ValueError(
                    f"checkpoint 参数形状不匹配: {key}, src={src.shape}, dst={p.data.shape}"
                )
            p.data[...] = src

        state.model = model
        state.tokenizer = tok
        return

    tok = CharTokenizer.from_texts(["你好，世界", "自研后端服务"]) 
    state.model = TinyLM(vocab_size=tok.vocab_size, n_embd=128)
    state.tokenizer = tok


@app.on_event("startup")
def startup_event():
    _load_or_init()


@app.get("/health")
def health():
    return {"status": "ok", "backend": "core"}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if state.model is None or state.tokenizer is None:
        raise HTTPException(status_code=503, detail="model not ready")

    ids = state.tokenizer.encode(req.prompt)
    if not ids:
        ids = [0]

    max_ctx = getattr(state.model, "max_seq_len", None)
    for _ in range(req.max_new_tokens):
        ctx = ids[-max_ctx:] if isinstance(max_ctx, int) and max_ctx > 0 else ids
        x = np.array([ctx], dtype=np.int64)
        logits, _ = state.model(x, None)
        next_logits = logits.data[0, -1]
        next_logits = next_logits / req.temperature
        next_logits = next_logits - np.max(next_logits)
        probs = np.exp(next_logits)
        probs = probs / (probs.sum() + 1e-12)
        next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)

    text = state.tokenizer.decode(ids)
    if text.startswith(req.prompt):
        text = text[len(req.prompt):]

    return GenerateResponse(text=text)
