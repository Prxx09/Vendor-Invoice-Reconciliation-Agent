import json, os, re
from groq import Groq
from huggingface_hub import InferenceClient

SCHEMA_PROMPT = """You are an invoice information extraction engine.
Extract the maximum information explicitly present in the document and return ONE valid JSON object only.

Rules:
- Never invent, infer, calculate, normalize away, or silently correct a value that is not explicitly supported by the document.
- Preserve identifiers, dates, addresses, emails, bank details, tax IDs, lease/store IDs, notes, currency and charge descriptions.
- Every monetary figure visible in the document must be represented in the JSON in a semantically appropriate field.
- For every charge preserve description, period, quantity, rate and amount when present.
- Tax must be an object with tax_type/label, rate_percent and amount, not only the tax amount.
- Keep subtotal and total_amount_due separately.
- Extract both seller/landlord and buyer/bill-to details.
- Extract payment instructions including bank, account name, account number and IFSC/routing code.
- Use null for a requested scalar that is absent and [] for absent collections.
- Do not expose chain-of-thought. The source_evidence field is only short document values/labels that make numeric fields auditable.

Return this structure:
{
  "document_type": null,
  "document_title": null,
  "invoice": {
    "invoice_number": null,
    "invoice_date": null,
    "billing_period": null,
    "payment_due_date": null,
    "lease_id": null,
    "currency": null
  },
  "seller": {
    "name": null,
    "address": null,
    "tax_id": null,
    "email": null
  },
  "buyer": {
    "name": null,
    "department": null,
    "location_or_store": null,
    "store_id": null,
    "address": null
  },
  "line_items": [
    {
      "description": null,
      "period": null,
      "quantity": null,
      "rate": null,
      "amount": null
    }
  ],
  "amounts": {
    "subtotal": null,
    "tax": {
      "label": null,
      "rate_percent": null,
      "amount": null
    },
    "total_amount_due": null,
    "currency": null
  },
  "payment_instructions": {
    "bank": null,
    "account_name": null,
    "account_number": null,
    "ifsc_or_routing_code": null
  },
  "notes": [],
  "source_evidence": {
    "line_item_amounts": [],
    "subtotal": null,
    "tax": null,
    "total": null
  }
}

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
    r=client.chat_completion(
        messages=[{"role":"user","content":SCHEMA_PROMPT.format(document=document)}],
        temperature=0,
        max_tokens=3000
    )
    return _json(r.choices[0].message.content)
