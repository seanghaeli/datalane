"""Client singletons for external API interactions."""
from src.clients.zyte_client import ZyteClient
from src.clients.openai_client import OpenAIClient

__all__ = ["ZyteClient", "OpenAIClient"]
