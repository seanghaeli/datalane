# src/matching_orchestrator.py

import pandas as pd
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
    Google activity heuristic
    """
    resultsGoogleCheck = await activity_confidence_check(batch_df)

    # Merge Google signal with force rules: +1 => force True, -1 => force False
    for i, g_val in enumerate(resultsGoogleCheck):
        if g_val == 1:
            overallResults[i] = True
        elif g_val == -1:
            overallResults[i] = False

    # Return DataFrame with Name and decision columns
    n = len(batch_df)
    out_df = pd.DataFrame({
        "Name": batch_df["Name"].tolist(),
        "results": [bool(results.get(i, False)) for i in range(n)],
        "resultsLLM": [bool(resultsLLM.get(i, False)) for i in range(n)],
        "resultsGoogleCheck": [
            int(resultsGoogleCheck[i]) if isinstance(resultsGoogleCheck, list) and i < len(resultsGoogleCheck) else 0
            for i in range(n)
        ],
        "overallResults": [bool(overallResults.get(i, False)) for i in range(n)],
    })

    return out_df
