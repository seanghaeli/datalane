from rapidfuzz import fuzz
from typing import List
from src.config import FUZZY_THRESHOLD
from src.models import CandidateRecord

def has_high_confidence_match(
    name: str,
    address: str,
    candidates: List[CandidateRecord],
    name_weight: float = 0.25,
    addr_weight: float = 0.75,
    threshold: float = FUZZY_THRESHOLD,
) -> bool:
    """
    Determine whether an input business record has a high-confidence fuzzy match
    among its candidate records.

    Args:
        name (str): Business name.
        address (str): Business address.
        candidates (List[CandidateRecord]): List of candidate records.
        name_weight (float): Weight for the name similarity score (default=0.25).
        addr_weight (float): Weight for the address similarity score (default=0.75).
        threshold (float): Minimum weighted score required to consider a match.

    Returns:
        bool: True if a high-confidence match was found.
    """
    for cand in candidates:
        cand_name = cand.name or ""
        cand_addr = cand.address or ""

        # Compute similarity scores
        name_score = fuzz.ratio(str(name).lower(), str(cand_name).lower()) if name else 0
        addr_score = fuzz.ratio(str(address).lower(), str(cand_addr).lower()) if address and cand_addr else 0
        total_score = (name_weight * name_score) + (addr_weight * addr_score)
        
        # Early exit once we exceed threshold
        if total_score >= threshold:
            return True

    return False
