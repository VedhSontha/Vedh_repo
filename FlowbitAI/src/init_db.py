from src.database import engine, Base
from src.models import Vendor, Invoice, InvoiceItem, CorrectionMemory

def init_db():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Seed PO data
    from src.database import SessionLocal
    from src.models import PurchaseOrder
    db = SessionLocal()
    if not db.query(PurchaseOrder).filter_by(po_number="PO-A-051").first():
        db.add(PurchaseOrder(po_number="PO-A-051", total_amount=5000.0, vendor_name="Tech Corp"))
        db.commit()
        print("Seeded PO-A-051")
    
    print("Tables created.")

if __name__ == "__main__":
    init_db()
