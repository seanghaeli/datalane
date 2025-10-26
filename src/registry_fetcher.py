import asyncio
import json
import time
from base64 import b64decode
from datetime import datetime
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from typing import Dict, List
from loguru import logger
from src.config import ZYTE_API_KEY, ZYTE_URL, GOV_URL, CONCURRENCY

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

async def _post_one(session: ClientSession, search_str: str, sem: asyncio.Semaphore) -> List[dict]:
    """
    Execute a single asynchronous Zyte API POST request to fetch registry records
    for a given search term (typically a business name).

    Args:
        session (ClientSession): Shared aiohttp session for making HTTP requests.
        search_str (str): The business name to query.
        sem (asyncio.Semaphore): Semaphore to limit concurrent outbound requests.

    Returns:
        List[Dict[str, Any]]: List of registry record dictionaries returned from Zyte.
                              Returns an empty list on timeout or failure.
    """

    start = time.perf_counter()
    async with sem:
        logger.debug(f"▶️ [{datetime.now().strftime('%H:%M:%S')}] START Zyte request for '{search_str}'")
        try:
            body = dict(BASE_PAYLOAD)
            body["corpName"] = search_str

            zyte_payload = {
                "url": GOV_URL,
                "httpResponseBody": True,
                "httpRequestMethod": "POST",
                "httpRequestText": json.dumps(body),
                "customHttpRequestHeaders": [
                    {"name": "Content-Type", "value": "application/json"},
                    {"name": "User-Agent", "value": "Mozilla/5.0"},
                ],
            }

            async with session.post(
                ZYTE_URL,
                auth=BasicAuth(ZYTE_API_KEY, ""),
                json=zyte_payload,
                timeout=ClientTimeout(total=15)
            ) as resp:
                logger.debug(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] Awaiting Zyte JSON for '{search_str}'")

                data = await resp.json()
                raw = b64decode(data["httpResponseBody"])
                parsed = json.loads(raw)
                records = parsed.get("response", {}).get("records", [])
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

async def _fetch_info_one(session: ClientSession, reg_index: str, sem: asyncio.Semaphore, corpName) -> dict:
    """
    Fetch detailed corporation info from the Puerto Rico registry for a given
    registration index, using Zyte as the proxy API.

    Args:
        session (ClientSession): Shared aiohttp session for outbound requests.
        reg_index (str): Registry index identifier for the corporation.
        sem (asyncio.Semaphore): Semaphore to limit concurrent requests.
        corp_name (str): Name of the corporation (for logging purposes).

    Returns:
        Dict[str, Any]: Dictionary containing at least the field {"address": str or None}.
                        Returns {"address": None} on failure or missing data.
    """
    start = time.perf_counter()
    async with sem:
        logger.debug(f"📥 [{datetime.now().strftime('%H:%M:%S')}] Fetching info for registrationIndex={reg_index}")
        info_url = f"https://rceapi.estado.pr.gov/api/corporation/info/{reg_index}"
        zyte_payload = {
            "url": info_url,
            "httpResponseBody": True,
            "httpRequestMethod": "GET",
            "customHttpRequestHeaders": [{"name": "User-Agent", "value": "Mozilla/5.0"}],
        }

        try:
            logger.debug(corpName)
            async with session.post(
                ZYTE_URL,
                auth=BasicAuth(ZYTE_API_KEY, ""),
                json=zyte_payload,
                timeout=ClientTimeout(total=30)
            ) as resp:
                data = await resp.json()
                raw = b64decode(data["httpResponseBody"])
                parsed = json.loads(raw)
                addr = (
                    parsed.get("response", {})
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

async def fetch_registry_for_batch(batch_df, expansions) -> Dict[int, List[dict]]:
    """
    Fetch and enrich government registry data for a batch of business names.

    Args:
        batch_df (pd.DataFrame): Batch of input records containing at least a 'Name' column.
        expansions (List[tuple]): List of (query1, query2) pairs generated from expand_queries_for_batch.

    Returns:
        Dict[int, List[Dict[str, Any]]]: Mapping of batch indices to lists of candidate
        records in the form {"name": str, "address": str or None}.
    """

    logger.debug("batch coming in")
    logger.debug(batch_df["Name"])
    timeout = ClientTimeout(total=60)
    sem = asyncio.Semaphore(CONCURRENCY)
    logger.debug("Concurrency: " )

    logger.debug(f"\n🕒 [{datetime.now().strftime('%H:%M:%S')}] Starting batch of {len(batch_df)} rows")

    t_batch_start = time.perf_counter()

    async with ClientSession(timeout=timeout) as session:
        # Search stage
        t1 = time.perf_counter()
        logger.debug(f"🔍 [{datetime.now().strftime('%H:%M:%S')}] Starting name searches...")

        async def search_task(batch_idx, query):
            recs = await _post_one(session, query, sem)
            return batch_idx, recs

        search_tasks = [
            asyncio.create_task(search_task(i, q))
            for i, (q1, q2) in enumerate(expansions)
            for q in (q1, q2)
        ]

        search_results = await asyncio.gather(*search_tasks)

        combined_results = {i: [] for i in range(len(batch_df))}
        for idx, recs in search_results:
            if isinstance(recs, list):
                combined_results[idx].extend(recs)

        # Address lookup stage
        async def address_task(batch_idx, record):
            reg_id = record.get("registrationIndex")
            info = await _fetch_info_one(session, reg_id, sem, record.get("corpName"))
            return batch_idx, record.get("corpName"), info.get("address")

        addr_tasks = []
        for idx, recs in combined_results.items():
            for record in recs:
                if record.get("registrationIndex"):
                    addr_tasks.append(asyncio.create_task(address_task(idx, record)))

        addr_results = await asyncio.gather(*addr_tasks)

        final_results = {i: [] for i in range(len(batch_df))}
        for idx, name, addr in addr_results:
            final_results[idx].append({"name": name, "address": addr})

    return final_results
