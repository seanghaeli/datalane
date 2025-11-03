# src/matching_orchestrator.py

from typing import List
from src.models import BusinessRecord, CandidateRecord, MatchingResult
from src.matchers.classical_matcher import has_high_confidence_match
from src.matchers.llm_matcher import llm_check
from src.matchers.google_matcher import activity_confidence_check

async def matching_orchestrator(
    record: BusinessRecord,
    candidates: List[CandidateRecord]
) -> MatchingResult:
    """
    Orchestrate all matching subsystems (fuzzy, LLM, heuristic)
    and produce a unified boolean decision for a single business record.
    
    Args:
        record (BusinessRecord): Business record to match.
        candidates (List[CandidateRecord]): Candidate registry matches retrieved from Zyte.

    Returns:
        MatchingResult: Matching result for this business.
    """

    # Classical fuzzy match
    classical_match = has_high_confidence_match(
        record.name,
        record.street_1 or "",
        candidates
    )

    # LLM semantic match (only if we have candidates)
    llm_match = False
    if candidates:
        llm_match = await llm_check(record.name, record.street_1 or "", candidates)

    # Combine classical + LLM
    overall_result = classical_match or llm_match
    
    # Google activity heuristic
    google_check = await activity_confidence_check(record)

    # Merge Google signal with force rules: -1 => force False
    # if google_check == -1:
    #     overall_result = False

    return MatchingResult(
        name=record.name,
        results=classical_match,
        results_llm=llm_match,
        results_google_check=google_check,
        overall_results=overall_result,
    )
