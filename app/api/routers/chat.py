import json
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from app.api.deps import get_rag_pipeline
from app.modules.pipeline.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])

@router.get("/stream")
async def chat_stream(
    query: str = Query(..., description="User query"),
    session_id: str = Query("default_session", description="Conversation session ID"),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    RAG Streaming endpoint. 
    Returns a stream of chunks as they are generated.
    """
    async def event_generator():
        async for chunk in rag_pipeline.answer(query, session_id):
            # Format as Server-Sent Events (SSE) or simple JSON lines
            # Here we use simple JSON lines for simplicity
            yield json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
