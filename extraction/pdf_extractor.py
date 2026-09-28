from pathlib import Path
import fitz
import pdfplumber
from PyPDF2 import PdfReader
import camelot

def extract_pymupdf(path):
    doc=fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc).strip()

def extract_pdfplumber(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages).strip()

def extract_pypdf2(path):
    reader=PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

def extract_tables(path):
    try:
        tables=camelot.read_pdf(str(path),pages="all",flavor="lattice")
        if len(tables)==0:
            tables=camelot.read_pdf(str(path),pages="all",flavor="stream")
    except Exception:
        tables=camelot.read_pdf(str(path),pages="all",flavor="stream")
    return [t.df.values.tolist() for t in tables]

def extract_all(path):
    path=Path(path)
    return {
        "pymupdf": extract_pymupdf(path),
        "pdfplumber": extract_pdfplumber(path),
        "pypdf2": extract_pypdf2(path),
        "tables": extract_tables(path),
    }
