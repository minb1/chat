# chatRAG/model/model_factory.py
import os
from .gemini_handler import GeminiHandler
from .openai_handler import OpenAIHandler
from .mistral_handler import MistralHandler
# from .mistral_handler import MistralHandler  # Uncomment when available

def get_available_models():
    models = {}
    if os.getenv("GOOGLE_API_KEY"):
        models["gemini"] = "Gemini 2.0"
    if os.getenv("OPENAI_API_KEY"):
        models["openai"] = "OpenAI (text-davinci-003)"
    if os.getenv("MISTRAL_API_KEY"):
        models["mistral"] = "Mistral Model"
    return models

def get_model_handler(model_name: str):
    if model_name == "gemini":
        return GeminiHandler()
    elif model_name == "openai":
        return OpenAIHandler()
    elif model_name == "mistral":
        return MistralHandler()
    else:
        # Fallback to Gemini if model not recognized
        return GeminiHandler()
