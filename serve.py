"""工业化推理服务（FastAPI）"""
import base64
import io
import json
import logging
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Optional
from uuid import uuid4

import jwt
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from config import ModelConfig
from data import load_tokenizer
from model import GPT

try:
    from PIL import Image
except ImportError:
    Image = None


def load_image_from_bytes(image_data: bytes) -> torch.Tensor:
    """从字节数据加载图片并转换为张量"""
    if Image is None:
        raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")
    
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    # 标准化图片大小为 224x224
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # 转换为张量并归一化 [0, 1]
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
    img_tensor = img_tensor.view(224, 224, 3) / 255.0
    
    # 转换为 (C, H, W) 格式
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # 标准的 ImageNet 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0)  # 添加 batch 维度


def post_process_text(text: str) -> str:
    """增强后处理：清理维基标记、重复、多余符号"""
    # 清理维基标记
    text = text.replace(' @-@ ', '-')
    text = text.replace('@-@', '-')
    text = text.replace(' @,@ ', ',')
    text = text.replace('@,@', ',')
    text = text.replace(' @.@ ', '.')
    text = text.replace('@.@', '.')
    
    # 清理连续的等号
    text = re.sub(r'(\s*=\s*){2,}', ' ', text)
    
    # 清理过多的标点重复（如连续逗号、点等）
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r';{2,}', ';', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # 清理不合理的文本模式（太多特殊字符）
    text = re.sub(r'(\s+["\']?\s*){3,}', ' ', text)
    
    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 移除只包含标点的句子
    sentences = text.split('。')
    sentences = [s.strip() for s in sentences if s.strip() and not re.match(r'^[\s\,\.\:\;\!\?\'"\-_]+$', s.strip())]
    text = '。'.join(sentences) if sentences else text
    
    return text


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=120, ge=1, le=1024)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: Optional[int] = Field(default=40, ge=1, le=1000)
    top_p: Optional[float] = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    session_id: Optional[str] = Field(default=None, max_length=128)
    use_history: bool = Field(default=True)
    max_history_messages: int = Field(default=8, ge=0, le=50)
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image data")


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    model_params_m: float
    device: str
    session_id: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class SessionMessage(BaseModel):
    role: str
    content: str
    ts: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]


class ServiceState:
    model: Optional[GPT] = None
    tokenizer = None
    device: Optional[torch.device] = None
    model_params_m: float = 0.0
    ready: bool = False
    start_time: float = time.time()


state = ServiceState()


logger = logging.getLogger("llm.serve")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("LLM_LOG_LEVEL", "INFO").upper())


REQUEST_COUNT = Counter(
    "llm_http_requests_total",
    "HTTP requests count",
    ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "llm_http_request_latency_seconds",
    "HTTP request latency",
    ["method", "path"]
)
GENERATE_COUNT = Counter(
    "llm_generate_requests_total",
    "Text generation requests",
    ["status"]
)
GENERATE_LATENCY = Histogram(
    "llm_generate_latency_seconds",
    "Text generation latency in seconds"
)
MODEL_READY = Gauge("llm_model_ready", "Model ready state, 1=ready, 0=not ready")
SESSIONS_TOTAL = Gauge("llm_sessions_total", "Unique session count in store")


class SlidingWindowRateLimiter:
    """线程安全滑动窗口限流器"""
    def __init__(self):
        self._events = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        if limit <= 0:
            return True

        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                return False

            bucket.append(now)
            return True


rate_limiter = SlidingWindowRateLimiter()


class SessionStore:
    """SQLite 持久化会话存储"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()
        self._ensure_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_sid ON session_messages(session_id)")
            conn.commit()

    def append_message(self, session_id: str, role: str, content: str):
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO session_messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, utc_now_iso()),
                )
                conn.commit()

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, ts FROM session_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        rows = list(reversed(rows))
        return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]

    def delete_session(self, session_id: str):
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
                conn.commit()

    def unique_session_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(DISTINCT session_id) AS c FROM session_messages").fetchone()
        return int(row["c"]) if row else 0


session_store = SessionStore(os.getenv("LLM_SESSION_DB", "sessions.db"))


def get_api_keys() -> set[str]:
    raw = os.getenv("LLM_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/oauth/token", auto_error=False)


def get_users() -> dict[str, str]:
    # 形如: LLM_USERS="admin:admin123,ops:ops123"
    raw = os.getenv("LLM_USERS", "").strip()
    users = {}
    if not raw:
        return users
    for pair in raw.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            u, p = u.strip(), p.strip()
            if u and p:
                users[u] = p
    return users


def jwt_secret() -> str:
    return os.getenv("LLM_JWT_SECRET", "change-me-in-production")


def jwt_expire_minutes() -> int:
    try:
        return max(1, int(os.getenv("LLM_JWT_EXPIRE_MINUTES", "60")))
    except ValueError:
        return 60


def create_access_token(subject: str) -> tuple[str, int]:
    expire_minutes = jwt_expire_minutes()
    exp = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    payload = {
        "sub": subject,
        "exp": exp,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, jwt_secret(), algorithm="HS256")
    return token, expire_minutes * 60


def decode_access_token(token: str) -> str:
    payload = jwt.decode(token, jwt_secret(), algorithms=["HS256"])
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="invalid token")
    return str(subject)


def get_rate_limit_rpm() -> int:
    try:
        return max(0, int(os.getenv("LLM_RATE_LIMIT_RPM", "0")))
    except ValueError:
        return 0


def optional_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    bearer_token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """返回调用方identity: apikey:<key> 或 user:<username>。"""
    keys = get_api_keys()
    users = get_users()
    if not keys and not users:
        return request.client.host if request.client else "anonymous"

    if x_api_key and x_api_key in keys:
        return f"apikey:{x_api_key}"

    if bearer_token:
        try:
            subject = decode_access_token(bearer_token)
            return f"user:{subject}"
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"invalid bearer token: {e}") from e

    raise HTTPException(status_code=401, detail="authentication required")


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
    
    # Handle torch.compile checkpoint format with _orig_mod prefix
    state_dict = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
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
        MODEL_READY.set(1)
        SESSIONS_TOTAL.set(session_store.unique_session_count())
    except Exception as e:
        state.ready = False
        MODEL_READY.set(0)
        print(f"[serve] 启动失败: {e}")

    yield


app = FastAPI(title="LLM Service", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with safe error serialization"""
    try:
        # Try normal error response first
        from fastapi.exception_handlers import request_validation_error_handler
        return await request_validation_error_handler(request, exc)
    except (UnicodeDecodeError, UnicodeEncodeError):
        # If encoding fails (e.g., binary data in fields), return safe error message
        return {
            "detail": "Invalid request: request body contains invalid characters or binary data",
            "error": "validation_error"
        }


def get_cors_origins() -> list[str]:
    raw = os.getenv("LLM_CORS_ORIGINS", "http://localhost:3000").strip()
    if not raw:
        return []
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


cors_origins = get_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def request_log_and_metrics(request: Request, call_next):
    request_id = str(uuid4())
    start = time.perf_counter()
    path = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        logger.exception("http_request_failed", extra={
            "path": path,
            "method": method,
        })
        REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
        REQUEST_LATENCY.labels(method=method, path=path).observe(time.perf_counter() - start)
        raise

    duration = time.perf_counter() - start
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration)

    response.headers["X-Request-ID"] = request_id
    logger.info(json.dumps({
        "event": "http_request",
        "request_id": request_id,
        "method": method,
        "path": path,
        "status": status,
        "latency_ms": round(duration * 1000.0, 2)
    }, ensure_ascii=False))
    return response


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - state.start_time, 2),
    }


@app.get("/readyz")
def readyz():
    if not state.ready:
        MODEL_READY.set(0)
        raise HTTPException(status_code=503, detail="model not ready")
    MODEL_READY.set(1)
    return {
        "status": "ready",
        "device": str(state.device),
        "model_params_m": round(state.model_params_m, 2),
    }


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/oauth/token", response_model=TokenResponse)
def oauth_token(form_data: OAuth2PasswordRequestForm = Depends()):
    users = get_users()
    if not users:
        raise HTTPException(status_code=400, detail="oauth is not enabled (LLM_USERS is empty)")

    expected = users.get(form_data.username)
    if expected is None or expected != form_data.password:
        raise HTTPException(status_code=401, detail="invalid username or password")

    token, expires_in = create_access_token(form_data.username)
    return TokenResponse(access_token=token, expires_in=expires_in)


def build_prompt_with_history(prompt: str, session_id: Optional[str], use_history: bool, max_history_messages: int) -> str:
    if not session_id or not use_history or max_history_messages <= 0:
        return prompt

    history = session_store.get_messages(session_id, limit=max_history_messages)
    if not history:
        return prompt

    chunks = []
    for m in history:
        role = m["role"].strip().lower()
        prefix = "User" if role == "user" else "Assistant"
        chunks.append(f"{prefix}: {m['content']}")

    chunks.append(f"User: {prompt}")
    chunks.append("Assistant:")
    return "\n".join(chunks)


@app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str, identity: str = Depends(optional_auth)):
    _ = identity
    messages = session_store.get_messages(session_id, limit=200)
    return SessionResponse(
        session_id=session_id,
        messages=[SessionMessage(**m) for m in messages],
    )


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str, identity: str = Depends(optional_auth)):
    _ = identity
    session_store.delete_session(session_id)
    SESSIONS_TOTAL.set(session_store.unique_session_count())
    return {"status": "deleted", "session_id": session_id}


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request, identity: str = Depends(optional_auth)):
    if not state.ready or state.model is None or state.tokenizer is None or state.device is None:
        GENERATE_COUNT.labels(status="not_ready").inc()
        raise HTTPException(status_code=503, detail="model not ready")

    rpm_limit = get_rate_limit_rpm()
    if not rate_limiter.allow(identity, rpm_limit, window_seconds=60):
        GENERATE_COUNT.labels(status="rate_limited").inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    session_id = req.session_id or str(uuid4())
    model_prompt = build_prompt_with_history(
        prompt=req.prompt,
        session_id=session_id,
        use_history=req.use_history,
        max_history_messages=req.max_history_messages,
    )

    t0 = time.time()

    try:
        # Process image if provided
        image_tensor = None
        if req.image_base64:
            try:
                image_data = base64.b64decode(req.image_base64)
                image_tensor = load_image_from_bytes(image_data).to(state.device)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                # Continue without image
                image_tensor = None

        with torch.no_grad():
            tokens = state.tokenizer.encode(model_prompt, return_tensors="pt").to(state.device)
            output = state.model.generate(
                tokens,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                image=image_tensor,
            )
            raw_text = state.tokenizer.decode(output[0].tolist())
            text = raw_text
            text = post_process_text(text)

            if text.startswith(model_prompt):
                text = text[len(model_prompt):].strip()
            elif text.startswith(req.prompt):
                text = text[len(req.prompt):].strip()
    except Exception as e:
        logger.exception("generation_failed", extra={
            "session_id": session_id,
            "identity": identity,
        })
        GENERATE_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"generation failed: {e}") from e

    latency_ms = (time.time() - t0) * 1000.0
    GENERATE_COUNT.labels(status="ok").inc()
    GENERATE_LATENCY.observe(latency_ms / 1000.0)

    if req.use_history:
        session_store.append_message(session_id, "user", req.prompt)
        session_store.append_message(session_id, "assistant", text)
        SESSIONS_TOTAL.set(session_store.unique_session_count())

    return GenerateResponse(
        text=text,
        latency_ms=round(latency_ms, 2),
        model_params_m=round(state.model_params_m, 2),
        device=str(state.device),
        session_id=session_id,
    )


@app.post("/v1/generate-multipart", response_model=GenerateResponse)
async def generate_multipart(
    prompt: str = "",
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    session_id: Optional[str] = None,
    use_history: bool = True,
    max_history_messages: int = 8,
    image: Optional[UploadFile] = File(None),
    request_obj: Request = None,
    identity: str = Depends(optional_auth),
):
    """多部分表单生成端点 - 支持图片上传"""
    if not state.ready or state.model is None or state.tokenizer is None or state.device is None:
        GENERATE_COUNT.labels(status="not_ready").inc()
        raise HTTPException(status_code=503, detail="model not ready")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    rpm_limit = get_rate_limit_rpm()
    if not rate_limiter.allow(identity, rpm_limit, window_seconds=60):
        GENERATE_COUNT.labels(status="rate_limited").inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded")

    session_id = session_id or str(uuid4())
    model_prompt = build_prompt_with_history(
        prompt=prompt,
        session_id=session_id,
        use_history=use_history,
        max_history_messages=max_history_messages,
    )

    t0 = time.time()

    try:
        # Process image if provided
        image_tensor = None
        if image:
            try:
                image_data = await image.read()
                image_tensor = load_image_from_bytes(image_data).to(state.device)
            except Exception as e:
                logger.warning(f"Failed to process image: {e}")
                # Continue without image
                image_tensor = None

        with torch.no_grad():
            tokens = state.tokenizer.encode(model_prompt, return_tensors="pt").to(state.device)
            output = state.model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                image=image_tensor,
            )
            raw_text = state.tokenizer.decode(output[0].tolist())
            text = raw_text
            text = post_process_text(text)

            if text.startswith(model_prompt):
                text = text[len(model_prompt):].strip()
            elif text.startswith(prompt):
                text = text[len(prompt):].strip()
    except Exception as e:
        logger.exception("generation_failed", extra={
            "session_id": session_id,
            "identity": identity,
        })
        GENERATE_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"generation failed: {e}") from e

    latency_ms = (time.time() - t0) * 1000.0
    GENERATE_COUNT.labels(status="ok").inc()
    GENERATE_LATENCY.observe(latency_ms / 1000.0)

    if use_history:
        session_store.append_message(session_id, "user", prompt)
        session_store.append_message(session_id, "assistant", text)
        SESSIONS_TOTAL.set(session_store.unique_session_count())

    return GenerateResponse(
        text=text,
        latency_ms=round(latency_ms, 2),
        model_params_m=round(state.model_params_m, 2),
        device=str(state.device),
        session_id=session_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
