from pathlib import Path
import pytest
from extraction.pdf_extractor import extract_all

FILES=list(Path("tests/fixtures").glob("*.pdf"))

@pytest.mark.parametrize("pdf", FILES, ids=lambda p:p.name)
def test_invoice_pdf_extraction(pdf):
    result=extract_all(pdf)
    for engine in ("pymupdf","pdfplumber","pypdf2"):
        assert len(result[engine]) > 100, f"{engine} extracted too little text from {pdf.name}"
    assert result["tables"], f"No table detected in {pdf.name}"
