"""工业化推理服务（FastAPI）"""
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import ModelConfig
from data import load_tokenizer
from model import GPT


def post_process_text(text: str) -> str:
    """轻量后处理：清理维基标记与多余符号"""
    text = text.replace(' @-@ ', '-')
    text = text.replace('@-@', '-')
    text = text.replace(' @,@ ', ',')
    text = text.replace('@,@', ',')
    text = text.replace(' @.@ ', '.')
    text = text.replace('@.@', '.')
    text = re.sub(r'(\s*=\s*){2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=120, ge=1, le=1024)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=1000)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    model_params_m: float
    device: str


class ServiceState:
    model: Optional[GPT] = None
    tokenizer = None
    device: Optional[torch.device] = None
    model_params_m: float = 0.0
    ready: bool = False
    start_time: float = time.time()


state = ServiceState()


def resolve_device() -> torch.device:
    force = os.getenv("LLM_DEVICE", "").strip().lower()
    if force in {"cpu", "cuda", "mps"}:
        if force == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if force == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def load_model_from_checkpoint() -> tuple[GPT, ModelConfig]:
    checkpoint_path = os.getenv("LLM_CHECKPOINT", "checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, _ = load_model_from_checkpoint()
        tokenizer = load_tokenizer()
        device = resolve_device()
        model.to(device)

        state.model = model
        state.tokenizer = tokenizer
        state.device = device
        state.model_params_m = model.get_num_params() / 1e6
        state.ready = True
    except Exception as e:
        state.ready = False
        print(f"[serve] 启动失败: {e}")

    yield


app = FastAPI(title="LLM Service", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - state.start_time, 2),
    }


@app.get("/readyz")
def readyz():
    if not state.ready:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "status": "ready",
        "device": str(state.device),
        "model_params_m": round(state.model_params_m, 2),
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not state.ready or state.model is None or state.tokenizer is None or state.device is None:
        raise HTTPException(status_code=503, detail="model not ready")

    t0 = time.time()

    with torch.no_grad():
        tokens = state.tokenizer.encode(req.prompt, return_tensors="pt").to(state.device)
        output = state.model.generate(
            tokens,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        text = state.tokenizer.decode(output[0].tolist())
        text = post_process_text(text)

    latency_ms = (time.time() - t0) * 1000.0

    return GenerateResponse(
        text=text,
        latency_ms=round(latency_ms, 2),
        model_params_m=round(state.model_params_m, 2),
        device=str(state.device),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
