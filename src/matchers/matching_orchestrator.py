# src/matching_orchestrator.py

from typing import Dict, List
from src.models import BusinessRecord, CandidateRecord, MatchingResult
from src.matchers.classical_matcher import has_high_confidence_match
from src.matchers.llm_matcher import llm_check_batch
from src.matchers.google_matcher import activity_confidence_check

async def matching_orchestrator(
    batch_records: List[BusinessRecord], 
    candidates: Dict[int, List[CandidateRecord]]
) -> List[MatchingResult]:
    """
    Orchestrate all matching subsystems (fuzzy, LLM, heuristic)
    and produce a unified boolean decision per business record.
    
    Args:
        batch_records (List[BusinessRecord]): Batch of business records.
        candidates (Dict[int, List[CandidateRecord]]): Mapping from batch index
            to candidate registry matches retrieved from Zyte.

    Returns:
        List[MatchingResult]: Matching results for each business in the batch.
    """

    # Classical fuzzy match
    names = [r.name for r in batch_records]
    addresses = [r.street_1 or "" for r in batch_records]
    results = has_high_confidence_match(names, addresses, candidates)

    # LLM semantic match
    resultsLLM = await llm_check_batch(names, addresses, candidates)

    # Combine classical + LLM
    overallResults = {
        i: results.get(i, False) or resultsLLM.get(i, False)
        for i in set(results.keys()) | set(resultsLLM.keys())
    }
    
    # Google activity heuristic
    resultsGoogleCheck = await activity_confidence_check(batch_records)

    # Merge Google signal with force rules: -1 => force False
    for i, g_val in enumerate(resultsGoogleCheck):
        if g_val == -1:
            overallResults[i] = False

    # Return list of MatchingResult objects
    n = len(batch_records)
    return [
        MatchingResult(
            name=batch_records[i].name,
            results=bool(results.get(i, False)),
            results_llm=bool(resultsLLM.get(i, False)),
            results_google_check=int(resultsGoogleCheck[i]) if i < len(resultsGoogleCheck) else 0,
            overall_results=bool(overallResults.get(i, False)),
        )
        for i in range(n)
    ]
