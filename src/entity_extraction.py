import re
from typing import List, Tuple

def extract_triplets(text_chunk: str) -> List[Tuple[str, str, str]]:
    """
    Extracts biological causal triplets: (Subject, Relationship, Object).
    Example: 'GeneA inhibits ProteinB' -> ('GENEA', 'INHIBITS', 'PROTEINB')
    """
    triplets = []
    # Pattern to capture core biological relationships
    pattern = r"([A-Z0-9a-z_-]+)\s+(inhibits|activates|binds to|expresses|suppresses)\s+([A-Z0-9a-z_-]+)"
    matches = re.findall(pattern, text_chunk, re.IGNORECASE)
    
    for subj, rel, obj in matches:
        triplets.append((
            subj.strip().upper(), 
            rel.strip().upper().replace(" ", "_"), 
            obj.strip().upper()
        ))
        
    return triplets