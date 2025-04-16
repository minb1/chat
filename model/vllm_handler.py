# chatRAG/model/vllm_handler.py
import json
import requests
from typing import Dict, Any, Optional
import logging
from .base_handler import BaseModelHandler
# Import requests exceptions for more specific handling if desired
from requests.exceptions import RequestException

# Define a custom exception for clarity (optional but good practice)
class VLLMCommunicationError(Exception):
    pass

class VLLMHandler(BaseModelHandler):
    """Handler for VLLM-served models."""

    def __init__(self, model_name: str , host: str, port: int):
        self.base_url = f"http://{host}:{port}/v1"
        self.model_name = model_name
        # Get the specific logger configured elsewhere if possible, otherwise use __name__
        # If your main app configures a JSON logger globally, this might pick it up.
        # If not, you might need to explicitly pass the configured logger instance.
        self.logger = logging.getLogger(__name__) # <<< Check if this logger outputs JSON
        # If not, consider using: self.logger = logging.getLogger('rag_metrics')
        # Or better: Pass the logger instance during initialization
        self.logger.info(f"Initialized VLLM handler for model: {model_name} at {self.base_url}")

    def generate_text(self, content: str) -> str:
        """
        Generate text response using the VLLM API.

        Args:
            content: The prompt text to send to the model

        Returns:
            The generated text response

        Raises:
            VLLMCommunicationError: If communication with the VLLM server fails.
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        url = f"{self.base_url}/chat/completions"
        self.logger.debug(f"Sending VLLM request to {url} with payload: {payload}") # Use debug level

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60 # Add a timeout
            )
            response.raise_for_status() # Raise HTTPError for bad status codes (4xx or 5xx)

            response_data = response.json()
            generated_text = response_data["choices"][0]["message"]["content"]
            return generated_text.strip()

        # Catch specific requests exceptions first
        except RequestException as e:
            error_message = f"VLLM API request failed: {str(e)}"
            self.logger.exception(error_message) # Log with stack trace
            # Re-raise as a custom or generic exception
            raise VLLMCommunicationError(error_message) from e
        # Catch other potential exceptions (e.g., JSON decoding)
        except Exception as e:
            error_message = f"Unexpected error during VLLM communication: {str(e)}"
            self.logger.exception(error_message)
            # Re-raise
            raise VLLMCommunicationError(error_message) from e

    # get_model_info remains the same, maybe add raise_for_status() and timeout there too
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        url = f"{self.base_url}/models"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            error_message = f"Failed to get model info from {url}: {str(e)}"
            self.logger.exception(error_message)
            # Decide whether to raise or return error dict
            # Raising might be better for consistency
            raise VLLMCommunicationError(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error getting model info from {url}: {str(e)}"
            self.logger.exception(error_message)
            raise VLLMCommunicationError(error_message) from e