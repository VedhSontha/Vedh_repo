import streamlit as st
import sys
import os

sys.path.append(os.getcwd())

from src.models import Invoice, CorrectionMemory
from src.processor import InvoiceProcessor

st.set_page_config(page_title="Review Invoices", layout="wide")

st.title("Review Pending Invoices")

if "db" not in st.session_state:
    st.error("Please run the main app first or refresh.")
    st.stop()

db = st.session_state.db
processor = InvoiceProcessor(db)

# Fetch pending invoices
pending_invoices = db.query(Invoice).filter(Invoice.status == "Pending").all()

if not pending_invoices:
    st.success("No pending invoices! 🎉")
else:
    for inv in pending_invoices:
        with st.expander(f"Invoice #{inv.invoice_number} (ID: {inv.id})", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Text")
                st.code(inv.raw_text)
            
            with col2:
                st.subheader("Extracted Data")
                # Re-extract to facilitate editing (in real app, we'd store extracted JSON in DB)
                extracted_data = processor.extract_data(inv.raw_text)
                
                # Form for correction
                with st.form(key=f"form_{inv.id}"):
                    vendor = st.text_input("Vendor Name", value=extracted_data.get("vendor_name", ""))
                    total = st.number_input("Total Amount", value=extracted_data.get("total_amount", 0.0))
                    
                    submitted = st.form_submit_button("Approve & Learn")
                    
                    if submitted:
                        # Check if correction needed
                        original_vendor = extracted_data.get("vendor_name", "")
                        if original_vendor != vendor:
                            st.write(f"Learning correction: {original_vendor} -> {vendor}")
                            processor.learn_correction("vendor_name", original_vendor, vendor)
                        
                        # Update Invoice
                        inv.status = "Approved"
                        inv.total_amount = total
                        db.commit()
                        st.success("Invoice Approved!")
                        st.rerun()
