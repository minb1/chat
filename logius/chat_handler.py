# chatRAG/logius/chat_handler.py

import os

from database.db_retriever import retrieve_chunks_from_db
from model.model_factory import get_model_handler
from embedding.embedding_factory import get_embedding_handler
from vectorstorage.vector_factory import get_vector_client, get_embedding_model_for_kb, get_available_knowledge_bases
from prompt.prompt_builder import create_prompt
from memory.redis_handler import log_message, format_chat_history_for_prompt


def process_user_message(message, context=None, chat_id=None, model_name='gemini', kb_id='qdrant-logius'):
    """
    Main logic for handling user queries and generating responses.

    Args:
        message: User's query text
        context: Optional context information
        chat_id: Chat session identifier
        model_name: LLM model to use for response generation
        kb_id: Knowledge base identifier to use for retrieval

    Returns:
        Dictionary containing the response and retrieved documents


        1. Keep track of what docs are retrieved most (keep with redis?)
        2. Insert in database: chatlogs, and retrieved doc amount
    """
    # Step 1: Get embeddings for the query
    print(message)
    embedding_model = get_embedding_model_for_kb(kb_id)
    embedding_handler = get_embedding_handler(embedding_model)
    user_embedding = embedding_handler.get_embedding(message)
    # Step 2: Retrieve relevant documents
    print("Getting vclient")
    vector_client = get_vector_client(kb_id)
    print("Getting related vectors now")
    retrieved_paths = vector_client.retrieve_vectors(user_embedding)
    print("Retrieved file paths from vector storage")
    print(retrieved_paths)
    # Step 3: Extract content from retrieved documents
    # context = retrieve_chunks(file_paths)
    # docs = retrieve_docs(file_paths)

    # Step 3: Retrieve content for the identified paths from PostgreSQL
    print("Retrieving chunk content from PostgreSQL...")
    # Use the new function to get context from the database
    context_from_db = retrieve_chunks_from_db(retrieved_paths)
    print(context_from_db)

    # Step 4: Prepare chat history and create prompt
    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
    prompt = create_prompt(message, context_from_db, chat_history)
    print(prompt)

    # Step 5: Generate response from LLM
    model_handler = get_model_handler(model_name)
    response = model_handler.generate_text(prompt)

    # Step 6: Log conversation if chat_id provided
    if chat_id:
        log_message(chat_id, "user", message)
        log_message(chat_id, "assistant", response)
        print(f"Logged message and response to redis for chat_id: {chat_id}")

    return {"response": response, "docs": context_from_db}


def get_kb_options():
    """
    Get available knowledge base options for the frontend.

    Returns:
        Dictionary of KB options with IDs and display names
    """
    return get_available_knowledge_bases()