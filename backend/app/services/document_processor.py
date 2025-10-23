"""
Document processing service using pdfplumber

TODO: Implement the document processing pipeline
- Extract tables from PDF using pdfplumber
- Classify tables (capital calls, distributions, adjustments)
- Extract and chunk text for vector storage
- Handle errors and edge cases
"""
from typing import Dict, List, Any
import pdfplumber
from app.core.config import settings
from app.services.table_parser import TableParser
import pathlib
import pandas as pd
import json
import logging
from typing import Any, Tuple
import asyncio
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process PDF documents and extract structured data"""
    
    def __init__(self):
        self.table_parser = TableParser()
    
    async def process_document(self, file_path: str, document_id: int, fund_id: int) -> Dict[str, Any]:
        """
        Process a PDF document:
         - parse PDF (text + tables)
         - chunk text for vector storage
         - save parsed JSON as a fallback
         - try to ingest chunks into RAGEngine (if available)
        Returns processing summary dict.
        """
        pdf_path = Path(file_path)
        if not pdf_path.exists():
            msg = f"PDF not found: {file_path}"
            logger.error(msg)
            return {"status": "error", "error": msg}

        # run blocking parse in thread to avoid blocking event loop
        try:
            parsed = await asyncio.get_running_loop().run_in_executor(None, parse_pdf, str(pdf_path))
        except Exception as e:
            logger.exception("Failed to parse PDF %s: %s", pdf_path, e)
            return {"status": "error", "error": str(e)}

        # prepare text pieces for chunking (preserve page markers)
        text_chunks_input: List[Dict[str, Any]] = []
        # parse_pdf returns 'text' (whole text) and 'tables' list; add full text and table summaries
        full_text = parsed.get("text", "")
        if full_text:
            text_chunks_input.append({"id": str(uuid.uuid4()), "text": full_text, "meta": {"source": "full_text"}})

        for t in parsed.get("tables", []):
            # include table as small textual blob to be chunked/ingested
            tbl_text = " | ".join(t.get("headers", [])) + "\n"
            # sample up to first 5 rows as text
            rows = t.get("rows", [])[:5]
            for r in rows:
                # flatten row values
                vals = [str(v) for v in r.values()]
                tbl_text += " | ".join(vals) + "\n"
            text_chunks_input.append({"id": str(uuid.uuid4()), "text": tbl_text, "meta": {"source": "table", "classification": t.get("classification")}})

        # chunk texts
        chunks = self._chunk_text(text_chunks_input)

        # persist parsed JSON to uploads dir as fallback
        uploads_dir = Path(settings.UPLOAD_DIR or "/app/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        out_path = uploads_dir / f"parsed-{document_id or pdf_path.stem}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"parsed": parsed, "chunks": chunks}, fh, default=str, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Failed to write parsed JSON to %s", out_path)

        # try ingesting into RAG engine if available
        ingestion_count = 0
        try:
            from app.services.rag_engine import RAGEngine  # optional
            rag = RAGEngine()
            # ingest each chunk's text with metadata
            for i, c in enumerate(chunks):
                doc_id = f"{document_id or pdf_path.stem}"
                rag.ingest_document(doc_id, c["text"], metadata={"document_id": document_id, "fund_id": fund_id, **c.get("meta", {})})
                ingestion_count += 1
        except Exception:
            # non-fatal: RAG ingestion optional
            logger.debug("RAG ingestion skipped or failed (optional).")

        stats = {
            "status": "ok",
            "document_id": document_id,
            "file": str(pdf_path),
            "tables_found": len(parsed.get("tables", [])),
            "chunks_created": len(chunks),
            "chunks_ingested": ingestion_count,
            "parsed_json": str(out_path),
        }
        return stats
    
    def _chunk_text(self, text_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk text content for vector storage using simple whitespace token chunker.
        Preserves overlap and returns list of {id, text, meta}.
        """
        chunk_size = int(getattr(settings, "CHUNK_SIZE", 1000))
        overlap = int(getattr(settings, "CHUNK_OVERLAP", 200))
        result: List[Dict[str, Any]] = []

        for item in text_content:
            text = item.get("text", "") or ""
            meta = item.get("meta", {})
            if not text.strip():
                continue
            words = text.split()
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words).strip()
                if chunk_text:
                    result.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "meta": {**meta, "source_item_id": item.get("id")}
                    })
                if end >= len(words):
                    break
                start = max(0, end - overlap)
        return result
def _clean_amount(val: Any) -> Any:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    # remove currency symbols, commas, parentheses -> negative
    neg = False
    if "(" in s and ")" in s:
        neg = True
    s = s.replace("$", "").replace(",", "").replace("—", "").strip()
    s = s.replace("(", "").replace(")", "")
    try:
        f = float(s)
        return -f if neg else f
    except Exception:
        return s


def _detect_amount_date_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    amount_cols = []
    date_cols = []
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ("amount", "amt", "value", "$", "total")):
            amount_cols.append(col)
        if "date" in name or "period" in name or "as of" in name:
            date_cols.append(col)
    # heuristic: also scan values
    if not amount_cols:
        for col in df.columns:
            sample = df[col].astype(str).str.replace(r"[^\d\.\-\(\),$]", "", regex=True)
            # if most rows contain digits/currency-like pattern
            if sample.str.contains(r"\d").sum() > max(1, len(sample) // 3):
                amount_cols.append(col)
    if not date_cols:
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > max(1, len(df) // 3):
                date_cols.append(col)
    return amount_cols, date_cols


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip headers and cells
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda v: v.strip() if isinstance(v, str) else v)
    # detect amount/date and normalize
    amount_cols, date_cols = _detect_amount_date_columns(df)
    for c in amount_cols:
        df[c] = df[c].apply(_clean_amount)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def parse_pdf(path: str) -> Dict[str, Any]:
    pdf_path = pathlib.Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_parts: List[str] = []
    tables: List[Dict[str, Any]] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)

                raw_tables = page.extract_tables() or []
                for raw in raw_tables:
                    if not raw or len(raw) < 2:
                        continue
                    headers = [str(h).strip() for h in raw[0]]
                    try:
                        df = pd.DataFrame(raw[1:], columns=headers)
                    except Exception:
                        df = pd.DataFrame(raw[1:])
                    # drop empty columns
                    df = df.loc[:, ~(df.columns.astype(str).str.strip() == "")]
                    try:
                        norm = _normalize_table(df)
                    except Exception as e:
                        logger.exception("Failed to normalize table: %s", e)
                        norm = df.fillna("")

                    rows = norm.fillna("").to_dict(orient="records")
                    classification, confidence = TableParser.classify_with_confidence(norm)

                    tables.append(
                        {
                            "page": getattr(page, "page_number", None),
                            "headers": list(norm.columns.astype(str)),
                            "rows": rows,
                            "classification": classification,
                            "classification_confidence": confidence,
                        }
                    )
    except Exception as e:
        logger.exception("Error parsing PDF %s: %s", pdf_path, e)
        raise

    return {"text": "\n\n".join(text_parts), "tables": tables}
