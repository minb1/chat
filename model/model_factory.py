# chatRAG/model/model_factory.py
import os
from .gemini_handler import GeminiHandler
# from .openai_handler import OpenAIHandler
from .mistral_handler import MistralHandler
from .vllm_handler import VLLMHandler


def get_available_models():
    models = {}
    if os.getenv("GOOGLE_API_KEY"):
        models["gemini"] = "Gemini 2.0"
    # if os.getenv("OPENAI_API_KEY"):
    #     models["openai"] = "OpenAI (text-davinci-003)"
    if os.getenv("MISTRAL_API_KEY"):
        models["mistral"] = "Mistral Model"

    # Add VLLM models - always available since it's running in our Docker setup
    # Extract model name from the environment or use default
    vllm_model = os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    models["vllm"] = f"Local VLLM ({vllm_model.split('/')[-1]})"

    return models


def get_model_handler(model_name: str):
    if model_name == "gemini":
        return GeminiHandler()
    # elif model_name == "openai":
    #     return OpenAIHandler()
    elif model_name == "mistral":
        return MistralHandler()
    elif model_name == "vllm":
        # Get configuration from environment variables with defaults
        vllm_host = os.getenv("VLLM_HOST", "vllm")
        vllm_port = int(os.getenv("VLLM_PORT", "8080"))
        vllm_model = os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
        return VLLMHandler(model_name=vllm_model, host=vllm_host, port=vllm_port)
    else:
        # Fallback to Gemini if model not recognized
        return GeminiHandler()