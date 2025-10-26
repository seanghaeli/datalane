# src/matching_orchestrator.py

from src.matchers.classical_matcher import has_high_confidence_match
from src.matchers.llm_matcher import llm_check_batch
from src.matchers.google_matcher import activity_confidence_check

async def matching_orchestrator(batch_df, candidates):
    """
    Orchestrate all matching subsystems (fuzzy, LLM, heuristic)
    and produce a unified boolean decision per business record.
    
    Args:
        batch_df (pd.DataFrame): Batch of business records containing at least
            'Name' and 'Street 1' columns.
        candidates (Dict[int, List[Dict[str, Any]]]): Mapping from batch index
            to candidate registry matches retrieved from Zyte.

    Returns:
        List[bool]: A list of boolean decisions, one per input row,
        indicating whether the business should be kept.
    """

    # Classical fuzzy match
    results = has_high_confidence_match(
        batch_df["Name"].tolist(),
        batch_df["Street 1"].tolist(),
        candidates
    )

    # LLM semantic match
    resultsLLM = await llm_check_batch(
        batch_df["Name"].tolist(),
        batch_df["Street 1"].tolist(),
        candidates
    )

    # Combine classical + LLM
    overallResults = {
        i: results.get(i, False) or resultsLLM.get(i, False)
        for i in set(results.keys()) | set(resultsLLM.keys())
    }

    """
    Google activity heuristic (Not tested)
    """
    # resultsGoogleCheck = await activity_confidence_check(batch_df)
    # for i, g_val in enumerate(resultsGoogleCheck):
    #     if g_val == 1:
    #         overallResults[i] = True   # Force keep
    #     elif g_val == -1:
    #         overallResults[i] = False  # Force drop

    # Return as flat list in batch order
    return [overallResults.get(i, False) for i in range(len(batch_df))]
