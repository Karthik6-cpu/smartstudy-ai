"""
Tests for Ollama client, configuration, and graceful degradation.
"""

import sys
import os

# Add parent directory to path so tests can run directly with python tests/test_ollama.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    PAGES,
    PAGE_CHAT,
    PAGE_DASHBOARD,
)
from llm.ollama_client import OllamaService


def test_config_settings():
    """Verify default configurations."""
    print("Testing config settings...")
    assert DEFAULT_MODEL == "llama3.2", f"Expected default model llama3.2, got {DEFAULT_MODEL}"
    assert DEFAULT_OLLAMA_HOST == "http://localhost:11434"
    assert PAGE_CHAT in PAGES
    assert PAGE_DASHBOARD in PAGES
    assert len(PAGES) == 8
    print(" Config settings valid.")


def test_ollama_service_init():
    """Verify service instantiation and defaults."""
    print("Testing OllamaService initialization...")
    service = OllamaService()
    assert service.host == "http://localhost:11434"

    custom_service = OllamaService(host="http://localhost:9999")
    assert custom_service.host == "http://localhost:9999"
    print(" OllamaService initialization valid.")


def test_ollama_graceful_offline_handling():
    """Verify that offline / unreachable hosts return polite error tuples instead of crashing."""
    print("Testing offline host handling...")
    # Use a port that is almost certainly closed
    service = OllamaService(host="http://localhost:59999")
    is_connected, msg = service.check_connection()

    assert is_connected is False, "Expected connection to be False on unused port"
    assert "Unable to connect to Ollama" in msg
    assert "ollama serve" in msg

    models = service.get_installed_models()
    assert isinstance(models, list)
    assert len(models) == 0

    installed = service.is_model_installed("llama3.2")
    assert installed is False

    # Test streaming behavior when offline: should yield friendly instructions
    stream_output = list(service.stream_chat(
        messages=[{"role": "user", "content": "Explain binary search"}],
        model="llama3.2"
    ))
    assert len(stream_output) == 1
    assert "Connection Error" in stream_output[0]
    print(" Offline handling passed gracefully.")


if __name__ == "__main__":
    test_config_settings()
    test_ollama_service_init()
    test_ollama_graceful_offline_handling()
    print("\nAll unit & integration checks passed successfully! 🎉")
