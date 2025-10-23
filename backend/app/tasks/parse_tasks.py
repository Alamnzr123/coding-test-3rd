import json
import logging
import os
import asyncio
from pathlib import Path

from .celery_app import celery_app

from backend.app.services.document_processor import DocumentProcessor
from app.db.session import SessionLocal
from app.models.document import Document

logger = logging.getLogger(__name__)


@celery_app.task(name="backend.app.tasks.parse_tasks.parse_document")
def parse_document(file_path: str, document_id: int | None = None) -> dict:
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

        # run async processor synchronously in Celery
        res = asyncio.run(processor.process_document(file_path, document_id or 0, getattr(doc, "fund_id", None) if doc else None))

        # attempt to update DB record as done and attach parsed path if available
        if document_id is not None and doc:
            try:
                doc.parsing_status = "done"
                # if processor returned parsed_json path, try to set on model
                parsed_path = res.get("parsed_json") or res.get("parsed_json_path") or res.get("parsed_json")
                if parsed_path and hasattr(doc, "parsed_json"):
                    setattr(doc, "parsed_json", parsed_path)
                db.commit()
            except Exception:
                db.rollback()

        return {"status": "ok", "output": res}
    except Exception as e:
        logger.exception("Parsing task failed: %s", e)
        if document_id is not None and doc:
            try:
                doc.parsing_status = "error"
                if hasattr(doc, "error_message"):
                    setattr(doc, "error_message", str(e))
                db.commit()
            except Exception:
                db.rollback()
        return {"status": "error", "error": str(e)}
    finally:
        try:
            db.close()
        except Exception:
            pass