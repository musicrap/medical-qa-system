import warnings
warnings.filterwarnings("ignore")
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
os.chdir(PROJECT_ROOT)

#
for key in ("OPENAI_API_KEY", "OPENAI_API_BASE", "LLM_MODEL"):
    os.environ.pop(key, None)


from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import chat_router, knowledge_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[Start] {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"[Config] LLM: {settings.LLM_MODEL}")
    print(f"[Config] Embedding: {settings.EMBEDDING_PROVIDER}")
    try:
        from app.rag.pipeline import get_rag
        rag = get_rag()
        st = rag.get_status()
        print(f"[KB] Status: {st.get('status')}, Docs: {st.get('total_documents')}")
    except Exception as e:
        print(f"[KB] Warning: {e}")
    yield
    print("[Shutdown]")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(knowledge_router)


@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": settings.APP_VERSION}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
