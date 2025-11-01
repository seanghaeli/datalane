import asyncio
from typing import List, Tuple
from src.models import BusinessRecord
from src.clients import OpenAIClient
from loguru import logger

PROMPT_TEMPLATE = """You are given a business name. There is a business registry with a search interface which allows us to search for business names. Other than the provided name, come up with EXACTLY ONE alternative string to query that will be likely to reveal the correct business in the interface's search response. You should try to identify the root substring of the original name that is most likely to reveal the correct business in the interface's search response. For example, if the original name is "Condal Tapas Restaurant & Rooftop Lounge", then the suggested alternative should be "Condal".

Return JUST the text of the suggested alternative

Business name: {name}
"""

async def expand_one(name: str) -> List[str]:
    """
    Generate one alternative query string for a given business name using the LLM.

    Args:
        name (str): Original business name.

    Returns:
        List[str]: A two-element list: [original_name, expanded_query]
    """
    try:
        openai_client = OpenAIClient()
        resp = await openai_client.chat_completions_create(
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
    tasks = [
        expand_one(record.name)
        for record in batch_records
    ]
    results = await asyncio.gather(*tasks)
    logger.debug("Expanded search set")
    logger.debug(results)
    # Convert List[str] results to List[Tuple[str, str]]
    return [(r[0], r[1]) for r in results]