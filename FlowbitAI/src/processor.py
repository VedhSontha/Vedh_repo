from sqlalchemy.orm import Session
from .memory import get_suggested_correction, add_correction

from .models import Invoice, PurchaseOrder
import re

class InvoiceProcessor:
    def __init__(self, db: Session):
        self.db = db

    def extract_data(self, raw_text: str):
        """
        Input: "Invoice #123 from Acme Corp, Total: $500.00"
        """
        data = {
            "invoice_number": None,
            "vendor_name": None,
            "total_amount": 0.0,
            "po_number": None,
            "service_date": None,
            "currency": "USD",
            "notes": [],
            "items": []
        }
        
        lines = raw_text.split('\n')
        line_buffer = []
        
        for line in lines:
            line_lower = line.lower()
            line_buffer.append(line)
            
            # --- Invoice Number ---
            if "invoice #" in line_lower:
                # "Invoice #DEMO-001"
                try:
                    parts = re.split(r'invoice\s*#', line, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        data["invoice_number"] = parts[1].strip().split()[0]
                except: pass

            # --- Total Amount & Currency ---
            if "total:" in line_lower or "amount:" in line_lower or "$" in line or "€" in line:
                matches = re.findall(r'[\$\s]*([\d,]+\.\d{2})', line)
                if matches:
                    try:
                        vals = [float(m.replace(',','')) for m in matches]
                        candidate = max(vals)
                        if candidate > data["total_amount"]:
                            data["total_amount"] = candidate
                    except: pass
                
                # Currency
                if "€" in line or "EUR" in line:
                    data["currency"] = "EUR"
                    
            # --- Vendor Name ---
            if "from" in line_lower:
                parts = re.split(r'from[:\s]+', line, flags=re.IGNORECASE)
                if len(parts) > 1:
                    vendor_candidate = parts[1].split(",")[0].strip()
                    if vendor_candidate:
                        data["vendor_name"] = vendor_candidate

            # --- PO Matching ---
            # Look for PO- pattern e.g. PO-A-051
            po_match = re.search(r'(PO-[A-Z]-\d{3})', line)
            if po_match:
                data["po_number"] = po_match.group(1)

            # --- Service Date (Leistungsdatum) ---
            # "Leistungsdatum: 2025-03-01" or "Service Date: 2025-03-01"
            if "leistungsdatum" in line_lower or "service date" in line_lower:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                if date_match:
                    data["service_date"] = date_match.group(1)

            # --- Advanced Logic: VAT / Gross ---
            if "mwst. inkl" in line_lower or "incl. vat" in line_lower or "gross" in line_lower:
                if "Triggered Gross/Net Strategy" not in data["notes"]:
                    data["notes"].append("Triggered Gross/Net Strategy")

            # --- Advanced Logic: Skonto ---
            if "skonto" in line_lower:
                 if "Skonto Terms Detected" not in data["notes"]:
                    data["notes"].append("Skonto Terms Detected")

        return data

    def process(self, raw_text: str):
        extracted = self.extract_data(raw_text)
        changes = []
        status = "Pending"
        
        # 1. Duplicate Detection
        if extracted.get("invoice_number") and extracted.get("vendor_name"):
            # Simplified check: Check exact invoice number collision globally for demo
            dup = self.db.query(Invoice).filter(Invoice.invoice_number == extracted["invoice_number"]).first()
            if dup:
                 status = "Duplicate"
                 changes.append(f"Marked as Duplicate of Invoice #{dup.id}")
        
        # 2. PO Matching
        if extracted.get("po_number"):
            po = self.db.query(PurchaseOrder).filter(PurchaseOrder.po_number == extracted["po_number"]).first()
            if po:
                changes.append(f"Linked to PO {po.po_number}")
                extracted["po_linked"] = True
            else:
                changes.append(f"PO {extracted['po_number']} not found in DB")

        # 3. Memory Correction (Vendor)
        if extracted.get("vendor_name"):
            corrected, conf = get_suggested_correction(
                self.db, 
                context_key="global_vendor_map", 
                field_type="vendor_name", 
                original_value=extracted["vendor_name"]
            )
            if corrected and conf > 0.8:
                original = extracted["vendor_name"]
                extracted["vendor_name"] = corrected
                changes.append(f"Auto-corrected vendor from {original} to {corrected}")

        # Review Status Logic
        if status != "Duplicate":
            if changes and "Auto-corrected" in str(changes):
                status = "Auto-Corrected"
            elif extracted.get("po_linked"):
                status = "Auto-Corrected" # PO Match treats as success
        
        # Add Notes to changes
        if extracted.get("notes"):
            changes.extend(extracted["notes"])

        return extracted, status, changes

    def learn_correction(self, field: str, original: str, corrected: str):
        context = "global_vendor_map" if field == "vendor_name" else "specific_invoice_context"
        add_correction(self.db, context, field, original, corrected)
