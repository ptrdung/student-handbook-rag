from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.routers import chat, ingestion, evaluation
from app.api.deps import get_rag_pipeline, get_ingestion_pipeline, get_evaluation_pipeline

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Pre-loads heavy models to ensure first-request speed.
    """
    print("🚀 System starting up... Pre-loading models...")
    # Trigger lazy-loading singletons in deps.py
    get_rag_pipeline()
    get_ingestion_pipeline()
    get_evaluation_pipeline()
    print("✅ All models loaded and ready!")
    yield
    print("🛑 System shutting down...")

app = FastAPI(
    title="Student Handbook RAG API",
    lifespan=lifespan
)

# Include routers
app.include_router(chat.router)
app.include_router(ingestion.router)
app.include_router(evaluation.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Student Handbook RAG API"}

