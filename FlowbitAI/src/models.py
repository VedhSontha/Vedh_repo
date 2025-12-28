from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from .database import Base

class Vendor(Base):
    __tablename__ = "vendors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    invoices = relationship("Invoice", back_populates="vendor")

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id = Column(Integer, primary_key=True, index=True)
    po_number = Column(String, unique=True, index=True) # e.g. PO-A-051
    total_amount = Column(Float)
    vendor_name = Column(String)

class Invoice(Base):
    __tablename__ = "invoices"
    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=True)
    po_id = Column(Integer, ForeignKey("purchase_orders.id"), nullable=True) # Link to PO
    invoice_number = Column(String, index=True)
    date = Column(DateTime)
    total_amount = Column(Float)
    currency = Column(String, default="USD")
    status = Column(String, default="Pending") # Pending, Reviewed, Approved, Duplicate
    raw_text = Column(Text, nullable=True) # The OCR text
    file_path = Column(String)
    
    vendor = relationship("Vendor", back_populates="invoices")
    po = relationship("PurchaseOrder") # Relationship
    items = relationship("InvoiceItem", back_populates="invoice")

class InvoiceItem(Base):
    __tablename__ = "invoice_items"
    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"))
    description = Column(String)
    quantity = Column(Float)
    unit_price = Column(Float)
    amount = Column(Float)
    category = Column(String, nullable=True) # e.g., "Office Supplies"
    
    invoice = relationship("Invoice", back_populates="items")

class CorrectionMemory(Base):
    __tablename__ = "correction_memory"
    id = Column(Integer, primary_key=True, index=True)
    # The context that triggered the mistake (e.g., a specific vendor name or line item description)
    context_key = Column(String, index=True) 
    # What the system originally thought or a specific pattern
    original_value = Column(String)
    # What the user corrected it to
    corrected_value = Column(String)
    # Granularity: 'vendor', 'category', 'total', etc.
    field_type = Column(String) 
    confidence_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CorrectionMemory(field={self.field_type}, original={self.original_value}, corrected={self.corrected_value})>"
