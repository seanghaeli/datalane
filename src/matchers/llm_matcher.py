import os
import asyncio
from typing import Dict, List, Any
from openai import AsyncOpenAI
from src.config import CONCURRENCY
from loguru import logger

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def llm_check_one(name: str, address: str, candidates: List[Dict[str, Any]]) -> bool:
    """
    LLM checking to determine if any of the given candidates correspond to the same business.
    Returns True if the model thinks at least one is a match.

    Args:
        name (str): The target business name.
        address (str): The target business address.
        candidates (List[Dict[str, Any]]): Candidate records, each with 'name' and 'address'.

    Returns:
        bool: True if the LLM deems at least one candidate to be the same business.
    """
    if not candidates:
        return False

    cand_text = "\n".join(
        [f"- {c.get('name', '')}, {c.get('address', '')}" for c in candidates]
    )

    prompt = f"""
You are verifying whether the target business is the same as any of the candidate businesses.

Target business:
Name: {name}
Address: {address}

Candidate businesses:
{cand_text}

If at least one candidate represents the same business (even if the names are written differently),
respond with ONLY "YES". Otherwise respond with ONLY "NO".
    """

    try:
        resp = await client.chat.completions.create(
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
    candidates: Dict[int, List[Dict[str, Any]]],
) -> Dict[int, bool]:
    """
    Perform parallel LLM checks across a batch of businesses.

    Args:
        names (List[str]): Target business names.
        addresses (List[str]): Corresponding business addresses.
        candidates (Dict[int, List[Dict[str, Any]]]): Mapping of row index → list of candidate dicts.

    Returns:
        Dict[int, bool]: Mapping of row index → boolean indicating if a match was found.
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    results: Dict[int, bool] = {}

    async def one_task(idx: int, name: str, addr: str, cands: List[Dict[str, Any]]):
        async with sem:
            results[idx] = await llm_check_one(name, addr, cands)

    tasks = []
    for i, (n, a) in enumerate(zip(names, addresses)):
        cands = candidates.get(i, [])
        if cands:
            tasks.append(one_task(i, n, a, cands))

    await asyncio.gather(*tasks)
    return results
