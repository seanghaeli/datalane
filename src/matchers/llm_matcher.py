import asyncio
from typing import Dict, List
from src.models import CandidateRecord
from src.clients import OpenAIClient
from loguru import logger

async def llm_check_one(name: str, address: str, candidates: List[CandidateRecord]) -> bool:
    """
    LLM checking to determine if any of the given candidates correspond to the same business.
    Returns True if the model thinks at least one is a match.

    Args:
        name (str): The target business name.
        address (str): The target business address.
        candidates (List[CandidateRecord]): Candidate records.

    Returns:
        bool: True if the LLM deems at least one candidate to be the same business.
    """
    if not candidates:
        return False

    # Build a simple list of candidate addresses only
    cand_text = "\n".join(
        [f"- {c.address or ''}" for c in candidates]
    )

    prompt = f"""
You are given a target address and a list of candidate addresses.

Goal: Determine if any candidate address likely corresponds to the SAME real-world location as the target address. An exact string match is NOT required. Consider common variations, abbreviations, suite/unit numbers, formatting differences, or nearby equivalences that strongly indicate the same place.

Target address:
{address}

Candidate addresses:
{cand_text}

If there is enough evidence that at least one candidate is likely referring to the target location, respond with ONLY "YES". Otherwise, respond with ONLY "NO".
    """
    # Cache here - Cache LLM match results by (address, sorted_candidate_addresses) tuple
    # Same target address + same candidate addresses should produce same result (temperature=0)
    # Note: Sort candidate addresses for consistent cache key
    try:
        openai_client = OpenAIClient()
        resp = await openai_client.chat_completions_create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        logger.debug(f"⚠️ LLM check failed for '{name}': {e}")
        return False


async def llm_check_batch(
    names: List[str],
    addresses: List[str],
    candidates: Dict[int, List[CandidateRecord]],
) -> Dict[int, bool]:
    """
    Perform parallel LLM checks across a batch of businesses.

    Args:
        names (List[str]): Target business names.
        addresses (List[str]): Corresponding business addresses.
        candidates (Dict[int, List[CandidateRecord]]): Mapping of row index → list of candidate records.

    Returns:
        Dict[int, bool]: Mapping of row index → boolean indicating if a match was found.
    """
    results: Dict[int, bool] = {}

    async def one_task(idx: int, name: str, addr: str, cands: List[CandidateRecord]):
        results[idx] = await llm_check_one(name, addr, cands)

    tasks = []
    for i, (n, a) in enumerate(zip(names, addresses)):
        cands = candidates.get(i, [])
        if cands:
            tasks.append(one_task(i, n, a, cands))

    await asyncio.gather(*tasks)
    return results
