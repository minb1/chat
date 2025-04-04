# chatRAG/embeddings/gemini_handler.py
from google import genai
from dotenv import load_dotenv
import os
from typing import List
from .base_handler import BaseEmbeddingHandler

load_dotenv()


class GeminiEmbeddingHandler(BaseEmbeddingHandler):
    def __init__(self, model_name="text-embedding-004"):
        self.model_name = model_name
        self._dim = 768 if model_name == "text-embedding-004" else 1024  # Update based on model

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings using Google's Gemini API."""
        try:
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            if not GOOGLE_API_KEY:
                raise ValueError("No API key found for Gemini.")

            client = genai.Client(api_key=GOOGLE_API_KEY)
            result = client.models.embed_content(
                model=self.model_name,
                contents=[text]
            )
            return result.embeddings[0].values
        except Exception as e:
            raise Exception(f"Error getting Gemini embedding: {str(e)}")

    @property
    def dimension(self) -> int:
        return self._dim
