import asyncio
import json
import time
from datetime import datetime
from typing import List, Tuple
from loguru import logger
from src.config import GOV_URL
from src.models import BusinessRecord, CandidateRecord
from src.clients import ZyteClient

BASE_PAYLOAD = {
    "cancellationMode": False,
    "comparisonType": 1,
    "isWorkFlowSearch": False,
    "limit": 10,
    "matchType": 4,
    "method": None,
    "onlyActive": True,
    "registryNumber": None,
    "advanceSearch": None,
}

async def _post_one(zyte_client: ZyteClient, search_str: str) -> List[dict]:
    """
    Execute a single asynchronous Zyte API POST request to fetch registry records
    for a given search term (typically a business name).

    Args:
        zyte_client (ZyteClient): The Zyte client singleton instance.
        search_str (str): The business name to query.

    Returns:
        List[Dict[str, Any]]: List of registry record dictionaries returned from Zyte.
                              Returns an empty list on timeout or failure.
    """

    start = time.perf_counter()
    logger.debug(f"▶️ [{datetime.now().strftime('%H:%M:%S')}] START Zyte request for '{search_str}'")
    # Cache here - Cache search results by search_str (business name query)
    # Registry search results are relatively stable
    try:
        body = dict(BASE_PAYLOAD)
        body["corpName"] = search_str

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }

        logger.debug(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] Awaiting Zyte JSON for '{search_str}'")
        
        data = await zyte_client.post_request(
            url=GOV_URL,
            request_body=body,
            headers=headers
        )
        
        records = data.get("response", {}).get("records", [])
        duration = time.perf_counter() - start
        logger.debug(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Completed Zyte call for '{search_str}' in {duration:.2f}s")
        return records if isinstance(records, list) else []
    except asyncio.TimeoutError:
        logger.debug(f"⏱️ [{datetime.now().strftime('%H:%M:%S')}] TIMEOUT Zyte request for '{search_str}' after 30s")
        return []
    except Exception as e:
        logger.debug(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] ERROR for '{search_str}': {e}")
        """
        PSEUDOCODE: trigger monitoring alert for unexpected Zyte failure in the event Government registry API changes

        `alert_engineer` could for example post to a Slack webhook or send a PagerDuty event.

        This would ensure on-call engineers are notified immediately.
        """
        # alert_engineer(
        #     title="Zyte request failure",
        #     message=f"Unexpected error for '{search_str}': {e}",
        #     severity="high",
        #     context={
        #         "function": "_post_one",
        #         "search_term": search_str,
        #         "timestamp": datetime.now().isoformat(),
        #     }
        # )
        return []

async def _fetch_info_one(zyte_client: ZyteClient, reg_index: str, corp_name: str) -> dict:
    """
    Fetch detailed corporation info from the Puerto Rico registry for a given
    registration index, using Zyte as the proxy API.

    Args:
        zyte_client (ZyteClient): The Zyte client singleton instance.
        reg_index (str): Registry index identifier for the corporation.
        corp_name (str): Name of the corporation (for logging purposes).

    Returns:
        Dict[str, Any]: Dictionary containing at least the field {"address": str or None}.
                        Returns {"address": None} on failure or missing data.
    """
    start = time.perf_counter()
    logger.debug(f"📥 [{datetime.now().strftime('%H:%M:%S')}] Fetching info for registrationIndex={reg_index}")
    # Cache here - Cache corporation info by reg_index (registrationIndex)
    # Corporation addresses don't change frequently
    info_url = f"https://rceapi.estado.pr.gov/api/corporation/info/{reg_index}"
    
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        logger.debug(corp_name)
        # Use get_request which returns parsed JSON
        data = await zyte_client.get_request(url=info_url, headers=headers)
        addr = (
            data.get("response", {})
            .get("corpStreetAddress", {})
            .get("address1")
        )
        duration = time.perf_counter() - start
        logger.debug(f"🏁 [{datetime.now().strftime('%H:%M:%S')}] Done fetching info {reg_index} in {duration:.2f}s → {addr}")
        return {"address": addr}
    except asyncio.TimeoutError:
        logger.debug(f"⏱️ [{datetime.now().strftime('%H:%M:%S')}] TIMEOUT for info request {reg_index}")
        return {"address": None}
    except Exception as e:
        logger.debug(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] ERROR fetching info {reg_index}: {e}")
        return {"address": None}

async def fetch_registry_for_record(
    record: BusinessRecord,
    expansion: Tuple[str, str]
) -> List[CandidateRecord]:
    """
    Fetch and enrich government registry data for a single business name.

    Args:
        record (BusinessRecord): Input business record.
        expansion (Tuple[str, str]): (query1, query2) pair generated from expand_queries_for_record.

    Returns:
        List[CandidateRecord]: List of candidate records found for this business.
    """
    # Get singleton Zyte client instance
    zyte_client = ZyteClient()

    # Search stage - search both the original and alternative query
    query1, query2 = expansion
    
    # Execute both searches in parallel
    search_results = await asyncio.gather(
        _post_one(zyte_client, query1),
        _post_one(zyte_client, query2),
        return_exceptions=True
    )

    # Combine results from both queries
    combined_records = []
    for recs in search_results:
        if isinstance(recs, list):
            combined_records.extend(recs)
        elif isinstance(recs, Exception):
            logger.debug(f"Search failed: {recs}")

    # Address lookup stage - fetch addresses for all found records
    async def address_task(record_dict):
        reg_id = record_dict.get("registrationIndex")
        if not reg_id:
            return None
        corp_name = record_dict.get("corpName", "")
        info = await _fetch_info_one(zyte_client, reg_id, corp_name)
        return corp_name, info.get("address")

    addr_tasks = [address_task(rec) for rec in combined_records if rec.get("registrationIndex")]
    addr_results = await asyncio.gather(*addr_tasks, return_exceptions=True)

    # Build final candidate list
    candidates = []
    for result in addr_results:
        if isinstance(result, tuple):
            name, addr = result
            if name:  # Only add if we got a valid name
                candidates.append(CandidateRecord(name=name, address=addr))
        elif isinstance(result, Exception):
            logger.debug(f"Address lookup failed: {result}")

    return candidates
