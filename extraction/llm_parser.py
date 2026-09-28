import json, os, re
from groq import Groq
from huggingface_hub import InferenceClient

SCHEMA_PROMPT="""Extract this invoice into JSON. Do not invent values. Return JSON only.
Required keys: document_type, vendor, invoice_number, invoice_date, due_date,
billing_period, currency, line_items, subtotal, tax, total.
DOCUMENT:
{document}
"""

def _json(text):
    text=re.sub(r"^\x60\x60\x60(?:json)?|\x60\x60\x60$", "", text.strip(), flags=re.I).strip()
    return json.loads(text)

def parse_groq(document):
    client=Groq(api_key=os.environ["GROQ_API_KEY"])
    r=client.chat.completions.create(
        model=os.getenv("GROQ_MODEL","openai/gpt-oss-120b"),
        temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":SCHEMA_PROMPT.format(document=document)}])
    return _json(r.choices[0].message.content)

def parse_hf(document):
    client=InferenceClient(model=os.getenv("HF_MODEL","Qwen/Qwen2.5-72B-Instruct"),token=os.environ["HF_TOKEN"])
    r=client.chat_completion(messages=[{"role":"user","content":SCHEMA_PROMPT.format(document=document)}],temperature=0,max_tokens=1500)
    return _json(r.choices[0].message.content)
