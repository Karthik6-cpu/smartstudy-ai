"""
Ollama Client Wrapper for SmartStudy AI.
Handles communication with the local Ollama instance, model checking,
connection diagnostics, and streaming LLM responses.
"""

from typing import Generator, List, Dict, Tuple, Optional
import ollama
from ollama import Client

from config.settings import DEFAULT_OLLAMA_HOST, DEFAULT_MODEL, SYSTEM_PROMPT


class OllamaService:
    """Service to interact with local Ollama server."""

    def __init__(self, host: Optional[str] = None):
        self.host = host or DEFAULT_OLLAMA_HOST
        self._client = Client(host=self.host)

    def check_connection(self) -> Tuple[bool, str]:
        """
        Check if the Ollama service is reachable.
        Returns:
            Tuple of (is_connected: bool, status_message: str)
        """
        try:
            # Attempt to query model list as a health check
            self._client.list()
            return True, f"Connected to Ollama at {self.host}"
        except Exception as e:
            error_msg = (
                f"Unable to connect to Ollama at {self.host}.\n\n"
                "**How to fix:**\n"
                "1. Check if Ollama is installed. If not, download from [ollama.com](https://ollama.com).\n"
                "2. Start Ollama:\n"
                "   - **Windows/Mac:** Open the Ollama desktop app from your applications/system tray.\n"
                "   - **Linux/Terminal:** Run `ollama serve` in a terminal window.\n"
                f"3. Error details: `{str(e)}`"
            )
            return False, error_msg

    def get_installed_models(self) -> List[str]:
        """
        Retrieve a list of model names currently installed in local Ollama.
        """
        try:
            response = self._client.list()
            model_names = []

            # Handle both object attributes (newer ollama-python) and dict responses
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])

            if models:
                for item in models:
                    if hasattr(item, "model"):
                        name = item.model
                    elif hasattr(item, "name"):
                        name = item.name
                    elif isinstance(item, dict):
                        name = item.get("model") or item.get("name")
                    else:
                        name = str(item)

                    if name:
                        model_names.append(name)

            return sorted(model_names)
        except Exception:
            return []

    def is_model_installed(self, model_name: str) -> bool:
        """
        Check if a specific model is available in the local Ollama instance.
        Handles tags like 'llama3.2' matching 'llama3.2:latest'.
        """
        installed = self.get_installed_models()
        clean_target = model_name.strip().lower()

        for m in installed:
            m_lower = m.lower()
            if (
                m_lower == clean_target
                or m_lower == f"{clean_target}:latest"
                or m_lower.startswith(f"{clean_target}:")
                or clean_target == m_lower.split(":")[0]
            ):
                return True
        return False

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
    ) -> Generator[str, None, None]:
        """
        Stream chat responses from Ollama given a conversation history.

        Args:
            messages: List of chat messages with 'role' and 'content'.
            model: Model name to invoke.
            system_prompt: Optional system prompt to prepend.

        Yields:
            Token chunks as strings.
        """
        # Verify connection first
        connected, conn_error = self.check_connection()
        if not connected:
            yield f"⚠️ **Connection Error:**\n\n{conn_error}"
            return

        # Verify model presence
        if not self.is_model_installed(model):
            installed_list = self.get_installed_models()
            installed_str = ", ".join(f"`{m}`" for m in installed_list) if installed_list else "None"
            yield (
                f"⚠️ **Model `{model}` Not Found Locally!**\n\n"
                f"Your local Ollama instance currently has: {installed_str}.\n\n"
                f"**To download this model, run this command in your terminal:**\n"
                f"```bash\nollama pull {model}\n```\n"
                f"Or you can run it directly:\n"
                f"```bash\nollama run {model}\n```\n"
                f"Once downloaded, refresh this page or try asking again."
            )
            return

        # Prepare messages payload with system prompt
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        try:
            response_stream = self._client.chat(
                model=model,
                messages=formatted_messages,
                stream=True,
            )

            for chunk in response_stream:
                # Handle both dict and object response formats
                content = None
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                elif isinstance(chunk, dict):
                    msg_obj = chunk.get("message", {})
                    if isinstance(msg_obj, dict):
                        content = msg_obj.get("content")

                if content:
                    yield content

        except Exception as e:
            yield f"\n\n❌ **Error during generation:** `{str(e)}`"


# Module-level convenience instance
default_service = OllamaService()
