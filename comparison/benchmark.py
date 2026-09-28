import base64, json, os, re, time
from pathlib import Path
import fitz
from groq import Groq
from extraction.pdf_extractor import extract_all

SCHEMA="""Return JSON only using this schema. Never invent a value; use null when absent.
{
 "document_type":null,"document_title":null,
 "invoice":{"invoice_number":null,"invoice_date":null,"billing_period":null,"payment_due_date":null,"lease_id":null,"currency":null},
 "seller":{"name":null,"address":null,"tax_id":null,"email":null},
 "buyer":{"name":null,"department":null,"location_or_store":null,"store_id":null,"address":null},
 "line_items":[{"description":null,"period":null,"quantity":null,"rate":null,"amount":null}],
 "amounts":{"subtotal":null,"tax":{"label":null,"rate_percent":null,"amount":null},"total_amount_due":null,"currency":null},
 "payment_instructions":{"bank":null,"account_name":null,"account_number":null,"ifsc_or_routing_code":null},
 "notes":[]
}
Preserve document values faithfully. Monetary fields must be numeric."""

def norm(v):
    if v is None: return ""
    return re.sub(r"[^a-z0-9]","",str(v).lower())

def leaves(obj,path=""):
    out=[]
    if isinstance(obj,dict):
        for k,v in obj.items(): out += leaves(v,f"{path}.{k}" if path else k)
    elif isinstance(obj,list):
        for i,v in enumerate(obj): out += leaves(v,f"{path}[{i}]")
    elif obj is not None:
        out.append((path,obj))
    return out

def score(result,truth):
    actual=dict(leaves(result)); expected=leaves(truth); matches=[]
    for p,v in expected:
        ok=p in actual and norm(actual[p])==norm(v)
        matches.append({"field":p,"expected":v,"actual":actual.get(p),"correct":ok})
    n=sum(x["correct"] for x in matches)
    return {"field_accuracy":round(100*n/len(matches),2),"matched_fields":n,
            "total_fields":len(matches),"field_details":matches}

def local_recovery(parsed,truth):
    raw_text="\n".join(parsed.get(k,"") for k in ("pymupdf","pdfplumber","pypdf2"))
    table_text=json.dumps(parsed.get("tables",[]),ensure_ascii=False)
    corpus=norm(raw_text+"\n"+table_text)
    details=[]
    for p,v in leaves(truth):
        # Very short/common numeric values (1, 3, etc.) are not meaningful evidence checks.
        nv=norm(v)
        if len(nv)<3: continue
        found=nv in corpus
        details.append({"field":p,"expected":v,"found_locally":found})
    n=sum(x["found_locally"] for x in details)
    return {"recoverable_values":n,"checked_values":len(details),
            "recovery_percent":round(100*n/len(details),2) if details else 0,
            "value_details":details,"text_chars":len(raw_text),
            "tables_detected":len(parsed.get("tables",[]))}

def local_payload(parsed):
    return "LOCAL PDF TEXT:\n"+parsed["pymupdf"]+"\n\nLOCAL TABLES:\n"+json.dumps(parsed["tables"],ensure_ascii=False)

def call_text(payload):
    c=Groq(api_key=os.environ["GROQ_API_KEY"]); t=time.perf_counter()
    r=c.chat.completions.create(model=os.environ["GROQ_TEXT_MODEL"],temperature=0,
      response_format={"type":"json_object"},
      messages=[{"role":"system","content":SCHEMA},{"role":"user","content":payload}])
    return json.loads(r.choices[0].message.content),time.perf_counter()-t

def image_urls(pdf):
    doc=fitz.open(pdf); urls=[]
    for page in doc:
        pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
        urls.append("data:image/png;base64,"+base64.b64encode(pix.tobytes("png")).decode())
    return urls

def call_vision(pdf):
    c=Groq(api_key=os.environ["GROQ_API_KEY"]); t=time.perf_counter()
    parts=[{"type":"text","text":SCHEMA}]
    parts += [{"type":"image_url","image_url":{"url":u}} for u in image_urls(pdf)]
    r=c.chat.completions.create(model=os.environ["GROQ_VISION_MODEL"],temperature=0,
      response_format={"type":"json_object"},messages=[{"role":"user","content":parts}])
    return json.loads(r.choices[0].message.content),time.perf_counter()-t

def main():
    truth=json.loads(Path("tests/ground_truth.json").read_text())
    out={}
    for name,gt in truth.items():
        pdf=Path("tests/fixtures")/name
        t=time.perf_counter(); parsed=extract_all(pdf); parse_s=time.perf_counter()-t
        entry={"local_parser_only":{**local_recovery(parsed,gt),
          "parse_seconds":round(parse_s,3),
          "raw_text":{"pymupdf":parsed["pymupdf"],"pdfplumber":parsed["pdfplumber"],"pypdf2":parsed["pypdf2"]},
          "tables":parsed["tables"]}}
        try:
            a,s=call_text(local_payload(parsed))
            entry["local_parser_then_llm"]={"result":a,"local_parse_seconds":round(parse_s,3),
              "llm_seconds":round(s,3),"total_seconds":round(parse_s+s,3),**score(a,gt)}
        except Exception as e: entry["local_parser_then_llm"]={"error":repr(e)}
        try:
            b,s=call_vision(pdf)
            entry["direct_vision_llm"]={"result":b,"total_seconds":round(s,3),**score(b,gt)}
        except Exception as e: entry["direct_vision_llm"]={"error":repr(e)}
        out[name]=entry
    Path("benchmark-results.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
