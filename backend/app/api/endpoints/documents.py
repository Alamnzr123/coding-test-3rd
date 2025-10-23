"""
Document API endpoints
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from datetime import datetime
from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import (
    Document as DocumentSchema,
    DocumentUploadResponse,
    DocumentStatus
)
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
from app.db.session import SessionLocal
import asyncio

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fund_id: int = None,
    db: Session = Depends(get_db)
):
    """Upload and process a PDF document"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE} bytes"
        )
    
    # Create upload directory if it doesn't exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Save file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create document record
    document = Document(
        fund_id=fund_id,
        file_name=file.filename,
        file_path=file_path,
        parsing_status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Prefer Celery if available, otherwise use BackgroundTasks
    try:
        # import locally to avoid mandatory celery dependency
        from backend.app.tasks.parse_tasks import parse_document  # type: ignore

        # enqueue Celery task
        parse_document.delay(str(file_path), document.id)
    except Exception:
        # fallback to in-process background task
        background_tasks.add_task(
            process_document_task,
            document.id,
            file_path,
            fund_id or 1  # Default fund_id if not provided
        )
    
    return DocumentUploadResponse(
        document_id=document.id,
        task_id=None,
        status="pending",
        message="Document uploaded successfully. Processing started."
    )


async def process_document_task(document_id: int, file_path: str, fund_id: int):
    """Background task to process document (used when Celery not available)"""
    db = SessionLocal()
    doc = None
    try:
        if document_id is not None:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                try:
                    doc.parsing_status = "processing"
                    db.commit()
                except Exception:
                    db.rollback()

        processor = DocumentProcessor()
        # run the async processor
        res = await processor.process_document(file_path, document_id or 0, fund_id)

        if document_id is not None and doc:
            try:
                doc.parsing_status = "done"
                parsed_path = res.get("parsed_json") or res.get("parsed_json_path")
                if parsed_path and hasattr(doc, "parsed_json"):
                    setattr(doc, "parsed_json", parsed_path)
                db.commit()
            except Exception:
                db.rollback()
    except Exception as e:
        logger.exception("Background processing failed: %s", e)
        if document_id is not None and doc:
            try:
                doc.parsing_status = "error"
                if hasattr(doc, "error_message"):
                    setattr(doc, "error_message", str(e))
                db.commit()
            except Exception:
                db.rollback()
    finally:
        try:
            db.close()
        except Exception:
            pass


@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(document_id: int, db: Session = Depends(get_db)):
    """Get document parsing status"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentStatus(
        document_id=document.id,
        status=document.parsing_status,
        error_message=document.error_message
    )


@router.get("/{document_id}", response_model=DocumentSchema)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document details"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.get("/", response_model=List[DocumentSchema])
async def list_documents(
    fund_id: int = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all documents"""
    query = db.query(Document)
    
    if fund_id:
        query = query.filter(Document.fund_id == fund_id)
    
    documents = query.offset(skip).limit(limit).all()
    return documents


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document"""
    document = db.query(Document).filter(Document.id == document_id).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    if document.file_path and os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete database record
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}
