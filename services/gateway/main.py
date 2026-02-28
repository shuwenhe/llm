from app.api.serve import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.gateway.main:app", host="0.0.0.0", port=8000, reload=False)
