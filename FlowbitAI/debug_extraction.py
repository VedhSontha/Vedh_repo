from src.database import SessionLocal
from src.processor import InvoiceProcessor

# Mock invoice text exactly as in demo_invoice_2_test.txt
text = """Invoice #DEMO-002
Date: 2025-02-01
From: Gooogle Inc
To: My Company

Items:
1. Ad Services $1200.00

Total: $1200.00
"""

processor = InvoiceProcessor(None) # DB not needed for extraction
data = processor.extract_data(text)

print(f"Extracted Data: {data}")
if data.get("vendor_name") == "Gooogle Inc":
    print("SUCCESS: Vendor extracted correctly.")
else:
    print(f"FAILURE: Vendor not extracted. Got: {data.get('vendor_name')}")
