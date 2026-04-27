from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import logging

load_dotenv()  # Load .env file automatically

from backend.db.session import init_db
from backend.routers import chat, files, auth
from backend.rag.metrics import ensure_metrics_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Banking RAG BOT API...")
    await init_db()
    await ensure_metrics_table()
    # Pre-load agent graph
    from backend.agents.graph import get_graph
    get_graph()
    logger.info("✅ All systems ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Banking Multi-Agent Chatbot",
    description="Multi-agent RAG system with Hybrid Search, SQLite aggregations, and MCP integration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(files.router)


@app.get("/")
async def root():
    return {"message": "Banking Chatbot API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
