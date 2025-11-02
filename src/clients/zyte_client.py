"""
Singleton Zyte client with rate limiting using aiolimiter.
"""
import json
from base64 import b64decode
from aiohttp import ClientSession, ClientTimeout, BasicAuth
from aiolimiter import AsyncLimiter
from typing import Dict, Any, Optional
from loguru import logger

from src.config import ZYTE_API_KEY, ZYTE_URL


class ZyteClient:
    """
    Singleton Zyte client for making requests through Zyte's smart proxy API.
    Uses AsyncRateLimiter for rate limiting instead of semaphores.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not ZyteClient._initialized:
            self.api_key = ZYTE_API_KEY
            self.base_url = ZYTE_URL
            # Rate limiter: allow CONCURRENCY requests per second (adjust as needed)
            # Using a token bucket: max_rate and time_period
            # For CONCURRENCY=1000, allow 1000 requests per second
            from src.config import CONCURRENCY
            self.rate_limiter = AsyncLimiter(max_rate=CONCURRENCY, time_period=1.0)
            self._session: Optional[ClientSession] = None
            ZyteClient._initialized = True

    async def _get_session(self) -> ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=ClientTimeout(total=60))
        return self._session

    async def post_request(
        self, 
        url: str, 
        request_body: Dict[str, Any], 
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a POST request via Zyte's smart proxy and return the decoded JSON body.
        
        Args:
            url: Target URL to request.
            request_body: JSON payload to include in the POST request.
            headers: Optional HTTP headers.
            
        Returns:
            Parsed JSON response body from the target site.
        """
        async with self.rate_limiter:
            session = await self._get_session()
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

            try:
                # Extract business name from request body for debugging
                business_name = request_body.get("corpName", "unknown")
                
                async with session.post(
                    self.base_url,
                    auth=BasicAuth(self.api_key, ""),
                    json=payload,
                    timeout=ClientTimeout(total=60)
                ) as resp:
                    data = await resp.json()
                    print(f"Zyte POST response for '{business_name}': {data}")
                    
                    # Check if Zyte returned an error response (like 520 Website Ban)
                    if "status" in data and data.get("status") not in [200, None]:
                        # This is a Zyte API error, not the target website response
                        error_type = data.get("type", "unknown")
                        error_title = data.get("title", "unknown")
                        error_detail = data.get("detail", "no details")
                        error_status = data.get("status")
                        raise Exception(
                            f"Zyte API error ({error_title}): {error_detail}. "
                            f"Status: {error_status}, Type: {error_type}"
                        )
                    
                    # Check if httpResponseBody exists (it might not for errors)
                    if "httpResponseBody" not in data:
                        raise Exception(
                            f"Missing httpResponseBody in Zyte response. "
                            f"Response keys: {list(data.keys())}, "
                            f"Response: {data}"
                        )
                    
                    body = json.loads(b64decode(data["httpResponseBody"]))
                    return body
            except Exception as e:
                logger.debug(f"⚠️ Zyte POST request failed: {e}")
                raise

    async def get_request(
        self, 
        url: str, 
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a GET request via Zyte's smart proxy and return the parsed JSON response.
        
        Args:
            url: Target URL to request.
            headers: Optional HTTP headers.
            
        Returns:
            Parsed JSON response as a dictionary.
        """
        async with self.rate_limiter:
            session = await self._get_session()
            payload = {
                "url": url,
                "httpResponseBody": True,
                "httpRequestMethod": "GET",
            }

            if headers:
                payload["customHttpRequestHeaders"] = [
                    {"name": k, "value": v} for k, v in headers.items()
                ]

            try:
                async with session.post(
                    self.base_url,
                    auth=BasicAuth(self.api_key, ""),
                    json=payload,
                    timeout=ClientTimeout(total=60)
                ) as resp:
                    data = await resp.json()
                    
                    # Check if Zyte returned an error response
                    if "status" in data and data.get("status") not in [200, None]:
                        error_type = data.get("type", "unknown")
                        error_title = data.get("title", "unknown")
                        error_detail = data.get("detail", "no details")
                        error_status = data.get("status")
                        raise Exception(
                            f"Zyte API error ({error_title}): {error_detail}. "
                            f"Status: {error_status}, Type: {error_type}"
                        )
                    
                    # Check if httpResponseBody exists
                    if "httpResponseBody" not in data:
                        raise Exception(
                            f"Missing httpResponseBody in Zyte response. "
                            f"Response keys: {list(data.keys())}, "
                            f"Response: {data}"
                        )
                    
                    body = json.loads(b64decode(data["httpResponseBody"]))
                    return body
            except Exception as e:
                logger.debug(f"⚠️ Zyte GET request failed: {e}")
                raise

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
