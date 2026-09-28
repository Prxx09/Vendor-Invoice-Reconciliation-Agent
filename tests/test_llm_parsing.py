import json
import os
from pathlib import Path

import pytest

from extraction.pdf_extractor import extract_pymupdf
from extraction.llm_parser import parse_groq, parse_hf

FILES = list(Path("tests/fixtures").glob("*.pdf"))
REQUIRED = {
    "document_type", "document_title", "invoice", "seller", "buyer",
    "line_items", "amounts", "payment_instructions", "notes", "source_evidence"
}
OUTPUT = Path("test-output")
OUTPUT.mkdir(exist_ok=True)


def save_result(pdf, provider, data):
    path = OUTPUT / f"{pdf.stem}_{provider}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n--- {provider.upper()} PARSED RESULT: {pdf.name} ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def validate_schema(data):
    assert REQUIRED.issubset(data)
    assert {
        "invoice_number", "invoice_date", "billing_period",
        "payment_due_date", "lease_id", "currency"
    }.issubset(data["invoice"])
    assert {"subtotal", "tax", "total_amount_due", "currency"}.issubset(data["amounts"])
    assert {"label", "rate_percent", "amount"}.issubset(data["amounts"]["tax"])
    assert {
        "bank", "account_name", "account_number", "ifsc_or_routing_code"
    }.issubset(data["payment_instructions"])


@pytest.mark.parametrize("pdf", FILES, ids=lambda p: p.name)
def test_groq_json(pdf):
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not configured")
    data = parse_groq(extract_pymupdf(pdf))
    save_result(pdf, "groq", data)
    validate_schema(data)


@pytest.mark.parametrize("pdf", FILES, ids=lambda p: p.name)
def test_hf_json(pdf):
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not configured")
    data = parse_hf(extract_pymupdf(pdf))
    save_result(pdf, "huggingface", data)
    validate_schema(data)
