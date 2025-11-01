import os
import asyncio
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.config import CONCURRENCY
from src.models import BusinessRecord
from loguru import logger

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

PROMPT_TEMPLATE = """You are given a business name. There is a business registry with a search interface which allows us to search for business names. Other than the provided name, come up with EXACTLY ONE alternative string to query that will be likely to reveal the correct business in the interface's search response. You should try to identify the root substring of the original name that is most likely to reveal the correct business in the interface's search response. For example, if the original name is "Condal Tapas Restaurant & Rooftop Lounge", then the suggested alternative should be "Condal".

Return JUST the text of the suggested alternative

Business name: {name}
"""

async def expand_one(name: str, sem: asyncio.Semaphore) -> List[str]:
    """
    Generate one alternative query string for a given business name using the LLM.

    Args:
        name (str): Original business name.
        sem (asyncio.Semaphore): Semaphore controlling concurrency of LLM requests.

    Returns:
        List[str]: A two-element list: [original_name, expanded_query]
    """
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You generate short, realistic business name alternatives."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(name=name)},
                ],
                temperature=0.2,
            )
            suggestion = resp.choices[0].message.content.strip()
            if not suggestion:
                suggestion = name
            return [name, suggestion]
        except Exception as e:
            logger.debug(f"LLM expansion failed for {name}: {e}")
            return [name, name]

async def expand_queries_for_batch(batch_records: List[BusinessRecord]) -> List[Tuple[str, str]]:
    """
    Expand search queries for a batch of businesses using the LLM.

    Each business name receives one generated variant string, which can then be
    used alongside the original in the registry search step to improve recall.

    Args:
        batch_records (List[BusinessRecord]): Batch of input business records.

    Returns:
        List[Tuple[str, str]]: A list of (original, alternative) query pairs.
                              Example: [("PETCO", "PET CO"), ("LOTE 23", "LOT 23")]
    """
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        expand_one(record.name, sem)
        for record in batch_records
    ]
    results = await asyncio.gather(*tasks)
    logger.debug("Expanded search set")
    logger.debug(results)
    # Convert List[str] results to List[Tuple[str, str]]
    return [(r[0], r[1]) for r in results]