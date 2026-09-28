import base64, json, os, re, time
from pathlib import Path
import fitz
from groq import Groq
from extraction.pdf_extractor import extract_all

PROMPT="""Extract the invoice into JSON only. Never invent missing values.
Use exactly these keys:
invoice_number, invoice_date, due_date, billing_period, currency, subtotal,
tax_amount, total, vendor_name, line_items.
line_items must be an array of objects with description, quantity, unit_price, amount.
Use null when absent. Return monetary values without currency symbols or thousands separators."""

def normalize(v):
    if v is None: return None
    return re.sub(r"[^a-z0-9.]","",str(v).lower())

def score(data, truth):
    fields=[k for k in truth if k!="line_item_count"]
    ok=sum(normalize(data.get(k))==normalize(truth[k]) for k in fields)
    line_ok=len(data.get("line_items") or [])==truth["line_item_count"]
    return {"field_accuracy":round(ok/len(fields)*100,2),"line_items_correct":line_ok,
            "matched_fields":ok,"total_fields":len(fields)}

def groq_text(text):
    c=Groq(api_key=os.environ["GROQ_API_KEY"])
    t=time.perf_counter()
    r=c.chat.completions.create(model=os.getenv("GROQ_TEXT_MODEL","llama-3.3-70b-versatile"),
      temperature=0,response_format={"type":"json_object"},
      messages=[{"role":"system","content":PROMPT},{"role":"user","content":text}])
    return json.loads(r.choices[0].message.content), time.perf_counter()-t

def page_data_url(pdf):
    doc=fitz.open(pdf)
    page=doc[0]
    pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
    return "data:image/png;base64,"+base64.b64encode(pix.tobytes("png")).decode()

def groq_vision(pdf):
    c=Groq(api_key=os.environ["GROQ_API_KEY"])
    t=time.perf_counter()
    r=c.chat.completions.create(model=os.getenv("GROQ_VISION_MODEL","meta-llama/llama-4-scout-17b-16e-instruct"),
      temperature=0,response_format={"type":"json_object"},
      messages=[{"role":"user","content":[{"type":"text","text":PROMPT},
        {"type":"image_url","image_url":{"url":page_data_url(pdf)}}]}])
    return json.loads(r.choices[0].message.content), time.perf_counter()-t

def main():
    truth=json.loads(Path("tests/ground_truth.json").read_text())
    out={}
    for name,gt in truth.items():
        pdf=Path("tests/fixtures")/name
        start=time.perf_counter(); parsed=extract_all(pdf); local_s=time.perf_counter()-start
        text=max((parsed[k] for k in ("pymupdf","pdfplumber","pypdf2")),key=len)
        a,llm_s=groq_text(text)
        b,vision_s=groq_vision(pdf)
        out[name]={
          "local_parser_then_llm":{"result":a,"local_parse_seconds":round(local_s,3),"llm_seconds":round(llm_s,3),
             "total_seconds":round(local_s+llm_s,3),**score(a,gt)},
          "direct_vision_llm":{"result":b,"total_seconds":round(vision_s,3),**score(b,gt)}
        }
    Path("benchmark-results.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
