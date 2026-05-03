import os
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks
from app.api.deps import get_ingestion_pipeline
from app.modules.ingestion.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/ingestion", tags=["ingestion"])

# Directory for temporary file storage
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
):
    """
    Upload a PDF document and trigger the ingestion pipeline.
    Processing runs in the background.
    """
    file_path = UPLOAD_DIR / file.filename
    
    # Save file locally
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run ingestion in background to not block the response
    background_tasks.add_task(
        ingestion_pipeline.run, 
        file_path=str(file_path),
        output_dir="data/processed"
    )
    
    return {
        "message": f"File '{file.filename}' uploaded successfully. Ingestion started in background.",
        "file_path": str(file_path)
    }
