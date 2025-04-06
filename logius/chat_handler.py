import os
import json
import logging
import datetime
from typing import Dict, List

from database.db_retriever import retrieve_chunks_from_db
from model.model_factory import get_model_handler
from embedding.embedding_factory import get_embedding_handler
from vectorstorage.vector_factory import (
    get_vector_client,
    get_embedding_model_for_kb,
    get_available_knowledge_bases
)
from prompt.prompt_builder import create_prompt
from memory.redis_handler import log_message, format_chat_history_for_prompt

# Import the reranker
from reranking.reranker import SentenceTransformerReranker

# Configure logging
logger = logging.getLogger('rag_metrics')
logger.setLevel(logging.INFO)


def process_user_message(message, context=None, chat_id=None, model_name='gemini', kb_id='qdrant-logius'):
    """
    Main logic for handling user queries and generating responses.

    Args:
        message: User's query text
        context: Optional context information (less relevant now)
        chat_id: Chat session identifier
        model_name: LLM model to use for response generation
        kb_id: Knowledge base identifier to use for retrieval

    Returns:
        Dictionary containing the response and retrieved documents as a dictionary.
        If reranking is applied, returns the reranked docs; otherwise, returns the original context.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"

    # Step 1: Get embeddings for the query
    print(f"Processing message: {message}")
    embedding_model = get_embedding_model_for_kb(kb_id)
    embedding_handler = get_embedding_handler(embedding_model)
    user_embedding = embedding_handler.get_embedding(message)

    # Step 2: Retrieve relevant document paths
    print("Getting vclient")
    vector_client = get_vector_client(kb_id)
    print("Getting related vectors now")
    retrieved_paths = vector_client.retrieve_vectors(user_embedding)  # Assume this returns List[str]
    print("Retrieved file paths from vector storage")
    print(retrieved_paths)

    # Step 3: Retrieve content for the identified paths from PostgreSQL
    print("Retrieving chunk content from PostgreSQL...")
    context_dict_from_db: Dict[str, str] = retrieve_chunks_from_db(retrieved_paths)
    print("Retrieved Context Dictionary:")

    # **New Step 3.1: Rerank retrieved documents using the Sentence Transformer CrossEncoder**
    # Convert the dictionary to a list of document dictionaries.
    documents: List[Dict[str, str]] = [
        {"file_path": path, "content": content} for path, content in context_dict_from_db.items()
    ]

    reranked_docs = []
    if documents:
        # Initialize the reranker with the ms-marco-MiniLM-L-6-v2 model.
        reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Rerank documents based on the query (using "content" as the key).
        reranked_docs = reranker.rerank(query=message, documents=documents, content_key="content", top_k=10)

    # Step 4: Format context for the LLM prompt (create a single string)
    if reranked_docs:
        context_string_for_prompt = "\n\n".join(
            f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
            for doc in reranked_docs
        )
        docs_to_return = reranked_docs
    else:
        context_string_for_prompt = "\n\n".join(
            f"Document Path: {path}\nContent:\n{content}"
            for path, content in context_dict_from_db.items()
        )
        docs_to_return = context_dict_from_db

    # Prepare chat history and create prompt
    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
    prompt = create_prompt(message, context_string_for_prompt, chat_history)
    print("Generated Prompt for LLM:")

    # Step 5: Generate response from LLM
    model_handler = get_model_handler(model_name)
    response = model_handler.generate_text(prompt)
    print(f"LLM Response: {response}")

    # Step 6: Log conversation if chat_id provided
    if chat_id:
        log_message(chat_id, "user", message)
        log_message(chat_id, "assistant", response)
        print(f"Logged message and response to redis for chat_id: {chat_id}")

    # Log response time and length
    response_info = {
        "timestamp": timestamp,
        "query_id": query_id,
        "response_length": len(response),
        "kb_id": kb_id
    }
    logger.info(f"RESPONSE_GENERATED {json.dumps(response_info)}")
    print(docs_to_return)
    # IMPORTANT: Return ictionary of documents, using reranked docs if available.
    return {"response": response, "docs": docs_to_return}


def get_kb_options():
    """
    Get available knowledge base options for the frontend.

    Returns:
        Dictionary of KB options with IDs and display names
    """
    return get_available_knowledge_bases()
