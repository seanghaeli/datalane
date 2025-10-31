from rapidfuzz import fuzz
from typing import List, Dict, Any
from src.config import FUZZY_THRESHOLD

def has_high_confidence_match(
    names: List[str],
    addresses: List[str],
    candidates: Dict[int, List[Dict[str, Any]]],
    name_weight: float = 0.25,
    addr_weight: float = 0.75,
    threshold: float = FUZZY_THRESHOLD,
) -> Dict[int, bool]:
    """
    Determine whether each input business record has a high-confidence fuzzy match
    among its candidate records.

    Args:
        names (List[str]): List of business names for the batch.
        addresses (List[str]): Corresponding business addresses.
        candidates (Dict[int, List[Dict[str, Any]]]): Mapping of row index → list of candidate dicts.
        name_weight (float): Weight for the name similarity score (default=0.4).
        addr_weight (float): Weight for the address similarity score (default=0.6).
        threshold (float): Minimum weighted score required to consider a match.

    Returns:
        Dict[int, bool]: Mapping of row index → True if a high-confidence match was found.
    """
    results = {}

    for i, (name, addr) in enumerate(zip(names, addresses)):
        row_candidates = candidates.get(i, [])
        found = False
        for cand in row_candidates:
            cand_name = cand.get("name", "")
            cand_addr = cand.get("address", "")

            # Compute similarity scores
            name_score = fuzz.ratio(str(name).lower(), str(cand_name).lower()) if name else 0
            addr_score = fuzz.ratio(str(addr).lower(), str(cand_addr).lower()) if addr and cand_addr else 0
            total_score = (name_weight * name_score) + (addr_weight * addr_score)
            print(f"Name, addr and total score: {name} {addr} {cand_name} {cand_addr} {name_score} {addr_score} {total_score}")
            print(threshold)
            # Early exit once we exceed threshold
            if total_score >= threshold:
                found = True
                print(f"Match found for row {i}: {name} {addr} {cand_name} {cand_addr} {name_score} {addr_score}")
                break

        results[i] = found

    return results
