# chatRAG/model/openai_handler.py
import os
import openai
from dotenv import load_dotenv
from .base_handler import BaseModelHandler

load_dotenv()

class OpenAIHandler(BaseModelHandler):
    def generate_text(self, content: str) -> str:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            return "No API key found for OpenAI."
        openai.api_key = OPENAI_API_KEY
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=content,
            max_tokens=150
        )
        return response.choices[0].text.strip()
