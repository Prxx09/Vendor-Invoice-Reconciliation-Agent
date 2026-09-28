import os
from pathlib import Path
import pytest
from extraction.pdf_extractor import extract_pymupdf
from extraction.llm_parser import parse_groq, parse_hf

FILES=list(Path("tests/fixtures").glob("*.pdf"))
REQUIRED={"document_type","vendor","invoice_number","invoice_date","due_date","billing_period","currency","line_items","subtotal","tax","total"}

@pytest.mark.parametrize("pdf", FILES, ids=lambda p:p.name)
def test_groq_json(pdf):
    if not os.getenv("GROQ_API_KEY"): pytest.skip("GROQ_API_KEY not configured")
    data=parse_groq(extract_pymupdf(pdf))
    assert REQUIRED.issubset(data)

@pytest.mark.parametrize("pdf", FILES, ids=lambda p:p.name)
def test_hf_json(pdf):
    if not os.getenv("HF_TOKEN"): pytest.skip("HF_TOKEN not configured")
    data=parse_hf(extract_pymupdf(pdf))
    assert REQUIRED.issubset(data)
