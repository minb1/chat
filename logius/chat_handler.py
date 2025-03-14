#chatRAG/logius/chat_handler.py

# from model.gemini_handler import generate_text
from model.model_factory import get_model_handler
from embedding.gemini_embeder import get_embedding
from vectorstorage.pinecone_client import retrieve_vectors
from prompt.prompt_builder import create_prompt
from database.get_from_local import retrieve_chunks, retrieve_docs
from memory.redis_handler import log_message, format_chat_history_for_prompt
import os

def process_user_message(message, context=None, chat_id=None, model_name='gemini'):
    """
    Main logic for handling the user queries
    For now, Simple console log
    TODO::
    2. Keep track of what docs are retrieved most (keep with redis?)
    3. Insert in database: chatlogs, and retrieved doc amount
    5. Get qdrant running
    """
    # Embed user query
    user_embedding = get_embedding(message)

    # Retrieve file_paths based on embedding
    file_paths = retrieve_vectors(user_embedding)

    # This is the part where we would query our data warehouse for the chatlogs based on the filepaths. Will do locally for now
    context = retrieve_chunks(file_paths)

    docs = retrieve_docs(file_paths)
    # Compute unique markdown files from the chunk file paths

    # Retrieve chat log from redis?
    chat_history = ""
    if chat_id:
        chat_history = format_chat_history_for_prompt(chat_id)
        print("Retrieved chat history: ", chat_history)

    # Have some prompt creator file create the prompt with context
    prompt = create_prompt(message, context, chat_history)

    # Sent that prompt to the text completion API, keep above separate so we can easy send prompts to different apis
    handler = get_model_handler(model_name)
    print(model_name)
    print(handler)
    response = handler.generate_text(prompt)

    # This is where you would log the user message and user response into redis?
    if chat_id:
        log_message(chat_id, "user", message)
        log_message(chat_id, "assistant", response)
        print("Logged message and response to redis for chat_id: ", chat_id)
    # Return message value
    return {"response": response, "docs": docs}
