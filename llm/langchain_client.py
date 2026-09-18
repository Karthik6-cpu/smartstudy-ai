"""
LangChain Client Factory for Ollama.
Wraps modern langchain-ollama ChatOllama integration with configurable parameters,
streaming support, and pre-flight connectivity checks.
"""

from typing import Optional, Tuple
from config.settings import DEFAULT_MODEL, DEFAULT_OLLAMA_HOST
from llm.ollama_client import OllamaService


class LangChainOllamaFactory:
    """Factory to instantiate and configure LangChain ChatOllama instances."""

    @staticmethod
    def create_chat_model(
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_HOST,
        temperature: float = 0.3,
    ):
        """
        Create a modern LangChain ChatOllama instance.

        Prefers modern `langchain_ollama.ChatOllama`, with fallback to
        `langchain_community.chat_models.ChatOllama` if needed.
        """
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                base_url=base_url,
                temperature=temperature,
            )
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                )
            except ImportError as e:
                raise ImportError(
                    "Neither 'langchain-ollama' nor 'langchain-community' is available. "
                    "Please install them via: pip install langchain-ollama"
                ) from e

    @staticmethod
    def validate_connection(base_url: str = DEFAULT_OLLAMA_HOST) -> Tuple[bool, str]:
        """Verify that local Ollama instance is reachable."""
        service = OllamaService(host=base_url)
        return service.check_connection()


def get_chat_ollama(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.3,
):
    """Convenience helper to get configured ChatOllama instance."""
    return LangChainOllamaFactory.create_chat_model(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
