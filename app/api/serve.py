"""API 服务入口（默认使用 core 自研后端）"""

from app.api.serve_core import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.serve:app", host="0.0.0.0", port=8000, reload=False)
