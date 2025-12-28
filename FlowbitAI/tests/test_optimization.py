import sys
import os
sys.path.append(os.getcwd())

from src.database import SessionLocal, Base, engine
from src.processor import InvoiceProcessor

def test_optimization():
    # Reset DB
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    processor = InvoiceProcessor(db)
    
    # 1. Teach the "correct" vendor
    print("Optimization Test Start")
    processor.learn_correction("vendor_name", "Gooogle Inc", "Google Inc") 
    print("Learned: 'Gooogle Inc' -> 'Google Inc'")
    
    # 2. Test Fuzzy Match
    # "Gooogle lnc" (letter l instead of I, space diff) - should fuzzily match "Gooogle Inc"
    # Wait, the learned pair key is "Gooogle Inc". 
    # If the new invoice has "Gooogle lnc", we want to match it to the memory KEY "Gooogle Inc" ??
    # Actually, the logic in memory.py compares the NEW invoice value to the STORED "original_value".
    # Stored original: "Gooogle Inc"
    # New inv value: "Gooogle lnc"
    # They should match.
    
    raw_text = "Invoice #999 from Gooogle lnc, Total: $ 1,234.56" # typo in name, complex money format
    
    extracted, status, changes = processor.process(raw_text)
    
    print(f"Extracted Vendor: {extracted['vendor_name']}")
    print(f"Extracted Total: {extracted['total_amount']}")
    print(f"Status: {status}")
    print(f"Changes: {changes}")
    
    # Verify Fuzzy Logic
    # It should have found "Gooogle Inc" in memory (similar to Gooogle lnc) 
    # And applied the correction "Google Inc"
    if extracted['vendor_name'] == "Google Inc":
        print("SUCCESS: Fuzzy matching worked!")
    else:
        print("FAILURE: Fuzzy matching failed.")

    # Verify Regex Logic
    if extracted['total_amount'] == 1234.56:
         print("SUCCESS: Regex extraction worked!")
    else:
         print(f"FAILURE: Expected 1234.56, got {extracted['total_amount']}")

    db.close()

if __name__ == "__main__":
    test_optimization()
