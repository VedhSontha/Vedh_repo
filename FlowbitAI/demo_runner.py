import os
import time

def create_demo_files():
    print("creating demo files...")
    
    # Invoice 1: The "Teacher"
    # Contains a typo: "Gooogle Inc"
    # used to teach the system.
    inv1 = """Invoice #DEMO-001
Date: 2025-01-01
From: Gooogle Inc
To: My Company

Items:
1. Cloud Services $500.00

Total: $500.00
"""
    with open("demo_invoice_1_teach.txt", "w", encoding="utf-8") as f:
        f.write(inv1)
    
    # Invoice 2: The "Test"
    # Contains the SAME typo: "Gooogle Inc"
    # used to prove the system learned.
    inv2 = """Invoice #DEMO-002
Date: 2025-02-01
From: Gooogle Inc
To: My Company

Items:
1. Ad Services $1200.00

Total: $1200.00
"""
    with open("demo_invoice_2_test.txt", "w", encoding="utf-8") as f:
        f.write(inv2)

    # Invoice 3: The "Complex"
    # Testing PO Mathcing, VAT, Service Date, Skonto
    inv3 = """Invoice #DEMO-003
Date: 2025-03-01
From: Tech Corp
PO-Number: PO-A-051
Leistungsdatum: 2025-03-01

Items:
1. Hard Drive Included MwSt. inkl. $5000.00
2. Skonto 3% if paid in 10 days

Total: $5000.00
"""
    with open("demo_invoice_3_complex.txt", "w", encoding="utf-8") as f:
        f.write(inv3)

    print("✔ Created 'demo_invoice_3_complex.txt'")

    # Invoice 4: The "Duplicate"
    # Same Invoice Number as #3, should trigger Duplicate Status
    inv4 = inv3 
    with open("demo_invoice_4_duplicate.txt", "w", encoding="utf-8") as f:
        f.write(inv4)
    print("✔ Created 'demo_invoice_4_duplicate.txt'")

    # Invoice 5: The "EUR"
    # Testing Currency
    inv5 = """Invoice #DEMO-005
Date: 2025-04-01
From: EuroParts GmbH
Total: 500,00 €
"""
    with open("demo_invoice_5_currency.txt", "w", encoding="utf-8") as f:
        f.write(inv5)
    print("✔ Created 'demo_invoice_5_currency.txt'")

    print("✔ Created 'demo_invoice_2_test.txt'")

def reset_database():
    print("Resetting database for clean demo...")
    if os.path.exists("flowbit.db"):
        os.remove("flowbit.db")
        print("✔ Deleted old database")
    
    # helper to re-init
    from src.database import engine, Base
    from src.models import Vendor, Invoice, InvoiceItem, CorrectionMemory, PurchaseOrder
    Base.metadata.create_all(bind=engine)
    
    # Seed PO
    from src.database import SessionLocal
    db = SessionLocal()
    db.add(PurchaseOrder(po_number="PO-A-051", total_amount=5000.0, vendor_name="Tech Corp"))
    db.commit()
    
    print("✔ Created new empty database with PO-A-051")

def print_guide():
    print("\n" + "="*50)
    print("   VIDEO DEMO SCRIPT GUIDE   ")
    print("="*50)
    print("1. START RECORDING your screen.")
    print("2. SAY: 'This is the Flowbit AI Invoice Processor. It learns from corrections.'")
    print("\n--- STEP 1: The Problem ---")
    print("3. Run the app: `streamlit run app.py`")
    print("4. Upload 'demo_invoice_1_teach.txt'.")
    print("5. Click 'Process Invoice'.")
    print("6. Show that the Vendor is 'Pending' or extracted incorrectly as 'Gooogle Inc'.")
    
    print("\n--- STEP 2: The Teaching ---")
    print("7. Go to 'Review Invoices' page (sidebar).")
    print("8. Change Vendor Name from 'Gooogle Inc' to 'Google Inc'.")
    print("9. Click 'Approve & Learn'.")
    print("10. SAY: 'The system has now memorized this correction.'")

    print("\n--- STEP 3: The Magic ---")
    print("11. Go back to 'app' (main page).")
    print("12. Upload 'demo_invoice_2_test.txt'.")
    print("13. Click 'Process Invoice'.")
    print("14. SHOW the green message: 'Auto-Corrected... to Google Inc'.")
    
    print("\n--- STEP 4: Advanced Features ---")
    print("15. Upload 'demo_invoice_3_complex.txt'.")
    print("16. Show that it matched PO-A-051 and extracted dates/notes.")
    print("17. Upload 'demo_invoice_4_duplicate.txt'.")
    print("18. Show that it flagged it as 'Duplicate'.")
    print("\n" + "="*50)

if __name__ == "__main__":
    reset_database()
    create_demo_files()
    print_guide()
    
    print("\n🚀 Launching App in 3 seconds...")
    time.sleep(3)
    os.system("streamlit run app.py")
