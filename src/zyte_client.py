import json
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from base64 import b64decode
from src.config import ZYTE_URL

async def post_request(api_key: str, url: str, request_body: dict, headers: dict | None = None) -> dict:
    """
    Send a POST request via Zyte’s smart proxy and return the decoded JSON body.

    Args:
        api_key (str): Zyte API key for authentication.
        url (str): Target URL to request.
        request_body (Dict[str, Any]): JSON payload to include in the POST request.
        headers (Optional[Dict[str, str]]): Additional HTTP headers.

    Returns:
        Dict[str, Any]: Parsed JSON response body from the target site.
    """
    payload = {
        "url": url,
        "httpResponseBody": True,
        "httpRequestMethod": "POST",
        "httpRequestText": json.dumps(request_body),
    }

    if headers:
        payload["customHttpRequestHeaders"] = [
            {"name": k, "value": v} for k, v in headers.items()
        ]

    async with ClientSession(timeout=ClientTimeout(total=60)) as session:
        async with session.post(
            ZYTE_URL,
            auth=BasicAuth(api_key, ""),
            json=payload,
        ) as resp:
            data = await resp.json()
            body = json.loads(b64decode(data["httpResponseBody"]))
            return body


async def get_request(api_key: str, url: str, headers: dict | None = None) -> str:
    """
    Send a GET request via Zyte’s smart proxy and return the decoded response text.

    Args:
        api_key (str): Zyte API key for authentication.
        url (str): Target URL to request.
        headers (Optional[Dict[str, str]]): Additional HTTP headers.

    Returns:
        str: Decoded text content of the target response.
    """
    payload = {
        "url": url,
        "httpResponseBody": True,
        "httpRequestMethod": "GET",
    }

    if headers:
        payload["customHttpRequestHeaders"] = [
            {"name": k, "value": v} for k, v in headers.items()
        ]

    async with ClientSession(timeout=ClientTimeout(total=60)) as session:
        async with session.post(
            ZYTE_URL,
            auth=BasicAuth(api_key, ""),
            json=payload,
        ) as resp:
            data = await resp.json()
            decoded = b64decode(data["httpResponseBody"]).decode("utf-8", errors="ignore")
            return decoded