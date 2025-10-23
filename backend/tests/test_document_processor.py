import os
from pathlib import Path

import pytest

from backend.app.services.document_processor import parse_pdf

ROOT = Path(__file__).resolve().parents[2]
FILES_DIR = ROOT / "files"

SAMPLE_PDFS = [
    FILES_DIR / "Sample_Fund_Performance_Report.pdf",
    FILES_DIR / "ILPA based Capital Accounting and Performance Metrics_ PIC, Net PIC, DPI, IRR  .pdf",
]


def locate_sample() -> Path | None:
    for p in SAMPLE_PDFS:
        if p.exists():
            return p
    return None


@pytest.mark.unit
def test_parse_pdf_extracts_text_and_tables():
    sample = locate_sample()
    if sample is None:
        pytest.skip("No sample PDF found in files/ — skipping parse test")

    result = parse_pdf(str(sample))
    assert isinstance(result, dict)
    assert "text" in result and isinstance(result["text"], str)
    assert "tables" in result and isinstance(result["tables"], list)

    # at least text should be non-empty
    assert result["text"].strip() != ""

    # if tables were found, validate structure of first table
    if result["tables"]:
        t0 = result["tables"][0]
        assert "headers" in t0 and isinstance(t0["headers"], list)
        assert "rows" in t0 and isinstance(t0["rows"], list)
        assert "classification" in t0