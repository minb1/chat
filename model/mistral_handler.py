import os
from mistralai import Mistral
from dotenv import load_dotenv
from .base_handler import BaseModelHandler

load_dotenv()

class MistralHandler(BaseModelHandler):
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = "mistral-small-latest"
        if not self.api_key:
            raise ValueError("No API key found for Mistral.")
        self.client = Mistral(api_key=self.api_key)

    def generate_text(self, content: str) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": content}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating response from Mistral: {str(e)}"