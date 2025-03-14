# chatRAG/model/gemini_handler.py
from google import genai
from dotenv import load_dotenv
import os
from .base_handler import BaseModelHandler

load_dotenv()

class GeminiHandler(BaseModelHandler):
    def generate_text(self, content: str) -> str:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            return "No API key found for Gemini."
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[content]
        )
        return response.text
