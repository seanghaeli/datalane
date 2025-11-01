"""
Singleton OpenAI client with rate limiting using aiolimiter.
"""
import os
from typing import Optional
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from loguru import logger

from src.config import OPENAI_API_KEY, CONCURRENCY


class OpenAIClient:
    """
    Singleton OpenAI client for making API requests.
    Uses AsyncRateLimiter for rate limiting instead of semaphores.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not OpenAIClient._initialized:
            api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in environment or config")
            
            self.client = AsyncOpenAI(api_key=api_key)
            # Rate limiter: allow CONCURRENCY requests per second
            # Using token bucket: max_rate and time_period
            # For CONCURRENCY=1000, allow 1000 requests per second
            # Adjust based on OpenAI API rate limits (typically 500/min for free tier, higher for paid)
            self.rate_limiter = AsyncLimiter(max_rate=min(CONCURRENCY, 500), time_period=1.0)
            OpenAIClient._initialized = True

    async def chat_completions_create(self, **kwargs):
        """
        Create a chat completion with rate limiting.
        Accepts all arguments that AsyncOpenAI.chat.completions.create accepts.
        
        Returns:
            The response from OpenAI's chat completions API.
        """
        async with self.rate_limiter:
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.debug(f"⚠️ OpenAI API request failed: {e}")
                raise

    async def chat_completion_acreate(self, **kwargs):
        """
        Legacy async create method (for compatibility with older openai library usage).
        Use chat_completions_create instead.
        
        Returns:
            The response from OpenAI's chat completions API.
        """
        async with self.rate_limiter:
            try:
                # For older openai library compatibility
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                logger.debug(f"⚠️ OpenAI API request failed: {e}")
                raise
