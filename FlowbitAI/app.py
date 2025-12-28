import streamlit as st
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from src.database import SessionLocal, engine, Base
from src.processor import InvoiceProcessor
from src.models import Invoice, InvoiceItem

st.set_page_config(page_title="Flowbit Invoice AI", layout="wide")

st.title("Flowbit Invoice Processor AI 🤖")

# Validation: ensure DB tables exist
Base.metadata.create_all(bind=engine)

if "db" not in st.session_state:
    st.session_state.db = SessionLocal()

db = st.session_state.db
processor = InvoiceProcessor(db)

# Dashboard Stats
col1, col2, col3 = st.columns(3)
total_inv = db.query(Invoice).count()
pending_inv = db.query(Invoice).filter(Invoice.status == "Pending").count()
auto_inv = db.query(Invoice).filter(Invoice.status == "Auto-Corrected").count()

col1.metric("Total Invoices", total_inv)
col2.metric("Pending Review", pending_inv)
col3.metric("Auto-Corrected", auto_inv)

st.divider()

st.header("Upload Invoice")
uploaded_file = st.file_uploader("Choose a file (mock text/txt)", type=["txt"])

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    try:
        raw_text = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = bytes_data.decode("latin-1")
    st.text_area("File Content", raw_text, height=100)
    
    if st.button("Process Invoice"):
        extracted, status, changes = processor.process(raw_text)
        
        # Save to DB
        new_inv = Invoice(
            invoice_number=extracted.get("invoice_number"),
            total_amount=extracted.get("total_amount"),
            status=status,
            raw_text=raw_text,
            file_path=uploaded_file.name
        )
        if extracted.get("vendor_name"):
            # ideally we link to Vendor table, but skipping for speed
            pass
            
        db.add(new_inv)
        db.commit()
        
        if status == "Auto-Corrected":
            st.success(f"Processed! Auto-Corrected: {changes}")
        elif status == "Duplicate":
            st.warning(f"Warning: This seems to be a Duplicate Invoice! {changes}")
        else:
            st.info("Processed. Needs Review.")
