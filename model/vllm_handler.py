# chatRAG/model/vllm_handler.py
import json
import requests
from typing import Dict, Any, Optional
import logging
from .base_handler import BaseModelHandler


class VLLMHandler(BaseModelHandler):
    """Handler for VLLM-served models."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", host: str = "vllm", port: int = 8080):
        """
        Initialize the VLLM handler.

        Args:
            model_name: The name of the model loaded in VLLM server
            host: Hostname of the VLLM server
            port: Port number of the VLLM server
        """
        self.base_url = f"http://{host}:{port}/v1"
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized VLLM handler for model: {model_name} at {self.base_url}")

    def generate_text(self, content: str) -> str:
        """
        Generate text response using the VLLM API.

        Args:
            content: The prompt text to send to the model

        Returns:
            The generated text response
        """
        try:
            # Prepare the API request following the OpenAI-compatible API format that vLLM uses
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            print("Sending payload..")
            print(self.base_url)
            # Send the request to the VLLM server
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            # Check for success and parse the response
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data["choices"][0]["message"]["content"]
                return generated_text.strip()
            else:
                self.logger.error(f"VLLM API request failed with status {response.status_code}: {response.text}")
                return f"Error: Failed to generate response (Status: {response.status_code})"

        except Exception as e:
            self.logger.exception(f"Exception when calling VLLM API: {str(e)}")
            return f"Error: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get model info: {response.status_code}")
                return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            self.logger.exception(f"Exception when getting model info: {str(e)}")
            return {"error": str(e)}