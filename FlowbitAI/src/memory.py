from sqlalchemy.orm import Session
from .models import CorrectionMemory

def add_correction(db: Session, context_key: str, field_type: str, original_value: str, corrected_value: str):
    """
    Learns a new correction.
    """
    # Check if we already have this specific correction to avoid duplicates or update confidence
    existing = db.query(CorrectionMemory).filter(
        CorrectionMemory.context_key == context_key,
        CorrectionMemory.field_type == field_type,
        CorrectionMemory.original_value == original_value
    ).first()

    if existing:
        existing.corrected_value = corrected_value
        existing.confidence_score += 0.1 # Boost confidence
    else:
        new_memory = CorrectionMemory(
            context_key=context_key,
            field_type=field_type,
            original_value=original_value,
            corrected_value=corrected_value
        )
        db.add(new_memory)
    db.commit()

def get_suggested_correction(db: Session, context_key: str, field_type: str, original_value: str):
    """
    Retrieves a correction if one exists with sufficient confidence.
    """
    # Retrieve all memories for this context/field
    memories = db.query(CorrectionMemory).filter(
        CorrectionMemory.context_key == context_key,
        CorrectionMemory.field_type == field_type
    ).all()

    best_match = None
    highest_ratio = 0.0

    import difflib

    for mem in memories:
        if not mem.original_value or not original_value:
            continue
        # Check similarity
        ratio = difflib.SequenceMatcher(None, mem.original_value.lower(), original_value.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = mem

    # Threshold for fuzzy match (e.g., 80% similar)
    if best_match and highest_ratio > 0.8:
        return best_match.corrected_value, best_match.confidence_score * highest_ratio # Scale confidence by similarity
    
    return None, 0.0
