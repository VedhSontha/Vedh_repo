import sys
import os
sys.path.append(os.getcwd())

from src.database import SessionLocal, Base, engine
from src.processor import InvoiceProcessor
from src.models import CorrectionMemory

def test_learning_loop():
    # Setup clean DB for test
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    processor = InvoiceProcessor(db)
    
    # Scene 1: First encounter
    # "Vndr_Bad" is a typo for "Vendor Good"
    raw_text_1 = "Invoice #001 from Vndr_Bad, Total: $100.00"
    extracted_1, status_1, changes_1 = processor.process(raw_text_1)
    
    print(f"Run 1: {extracted_1['vendor_name']} | Status: {status_1}")
    assert extracted_1['vendor_name'] == "Vndr_Bad"
    assert status_1 == "Pending"
    
    # Scene 2: User corrects it
    print("User corrects 'Vndr_Bad' -> 'Vendor Good'")
    processor.learn_correction("vendor_name", "Vndr_Bad", "Vendor Good")
    
    # Scene 3: Second encounter
    raw_text_2 = "Invoice #002 from Vndr_Bad, Total: $200.00"
    extracted_2, status_2, changes_2 = processor.process(raw_text_2)
    
    print(f"Run 2: {extracted_2['vendor_name']} | Status: {status_2}")
    
    if extracted_2['vendor_name'] == "Vendor Good":
        print("SUCCESS: System learned the correction!")
    else:
        print("FAILURE: System did not apply correction.")
        
    db.close()

if __name__ == "__main__":
    test_learning_loop()
