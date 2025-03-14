from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()


def get_embedding(query):

    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        return "No API key found"

    client = genai.Client(api_key=GOOGLE_API_KEY)
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=[query]
    )
    return result.embeddings[0].values
