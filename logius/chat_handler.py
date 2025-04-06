import os
import json
import logging
import datetime
from typing import Dict, List, Optional # Import Optional

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


def process_user_message(
    message: str,
    chat_id: Optional[str] = None,
    model_name: str = 'gemini',
    kb_id: str = 'qdrant-logius',
    use_reranker: bool = True # <<< Add parameter with default True
) -> Dict:
    """
    Main logic for handling user queries and generating responses.

    Args:
        message: User's query text.
        chat_id: Chat session identifier.
        model_name: LLM model to use for response generation.
        kb_id: Knowledge base identifier to use for retrieval.
        use_reranker: Boolean flag to indicate whether to use the reranker.

    Returns:
        Dictionary containing the response and a list of retrieved/reranked documents.
        Each document in the list is a dictionary with 'file_path' and 'content',
        and potentially 'rerank_score' if reranking was used.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"

    # Step 1: Get embeddings for the query
    print(f"Processing message: {message}, Use Reranker: {use_reranker}")
    embedding_model = get_embedding_model_for_kb(kb_id)
    embedding_handler = get_embedding_handler(embedding_model)
    user_embedding = embedding_handler.get_embedding(message)

    # Step 2: Retrieve relevant document paths
    print("Getting vclient")
    vector_client = get_vector_client(kb_id)
    print("Getting related vectors now")
    retrieved_paths = vector_client.retrieve_vectors(user_embedding)
    print("Retrieved file paths from vector storage:", retrieved_paths)

    # Step 3: Retrieve content for the identified paths from PostgreSQL
    print("Retrieving chunk content from PostgreSQL...")
    context_dict_from_db: Dict[str, str] = retrieve_chunks_from_db(retrieved_paths)
    print(f"Retrieved {len(context_dict_from_db)} chunks from DB.")

    # Convert the initial retrieval to the standard document list format
    # We do this regardless of reranking to have a consistent starting point
    documents: List[Dict[str, str]] = [
        {"file_path": path, "content": content}
        for path, content in context_dict_from_db.items()
    ]

    # Step 3.1: Conditionally Rerank retrieved documents
    if use_reranker and documents: # Only rerank if requested AND we have documents
        print("Reranking documents...")
        try:
            # Initialize the reranker
            reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            # Rerank documents based on the query
            reranked_docs_list = reranker.rerank(query=message, documents=documents, content_key="content", top_k=10)
            print(f"Reranked {len(reranked_docs_list)} documents.")
            # Use the reranked list for context building and returning
            docs_for_context = reranked_docs_list
            docs_to_return = reranked_docs_list # This list includes scores
        except Exception as e:
            print(f"Error during reranking: {e}. Falling back to original documents.")
            logger.error(f"Reranking failed for query_id {query_id}: {e}")
            # Fallback to using the original documents if reranking fails
            docs_for_context = documents
            # Add 'rerank_score': None for consistency in the structure expected by frontend
            docs_to_return = [{**doc, 'rerank_score': None} for doc in documents]
    else:
        # No reranking requested or no initial documents
        if not documents:
             print("No documents retrieved initially.")
        else:
             print("Skipping reranking as requested.")
        docs_for_context = documents
        # Add 'rerank_score': None for consistency in the structure expected by frontend
        docs_to_return = [{**doc, 'rerank_score': None} for doc in documents]


    # Step 4: Format context for the LLM prompt using the selected documents (reranked or original)
    context_string_for_prompt = "\n\n".join(
        f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
        for doc in docs_for_context # Use the potentially reranked list
    ) if docs_for_context else "No relevant documents found." # Handle empty case

    # Prepare chat history and create prompt
    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
    prompt = create_prompt(message, context_string_for_prompt, chat_history)
    # print("Generated Prompt for LLM:", prompt) # Optionally print full prompt for debugging

    # Step 5: Generate response from LLM
    print(f"Generating response using model: {model_name}")
    model_handler = get_model_handler(model_name)
    response = model_handler.generate_text(prompt)
    print(f"LLM Response generated.")

    # Step 6: Log conversation if chat_id provided
    if chat_id:
        log_message(chat_id, "user", message)
        log_message(chat_id, "assistant", response)
        print(f"Logged message and response to redis for chat_id: {chat_id}")

    # Log response info
    response_info = {
        "timestamp": timestamp,
        "query_id": query_id,
        "response_length": len(response),
        "kb_id": kb_id,
        "model_used": model_name,
        "reranker_used": use_reranker if documents else False, # Log if reranker was intended (if docs existed)
        "docs_retrieved_count": len(documents),
        "docs_returned_count": len(docs_to_return)
    }
    logger.info(f"RESPONSE_GENERATED {json.dumps(response_info)}")

    # IMPORTANT: Always return the list of document dictionaries.
    # This list will contain rerank_score if reranking was successful.
    print(f"Returning {len(docs_to_return)} documents to frontend.")
    return {"response": response, "docs": docs_to_return}


def get_kb_options():
    """
    Get available knowledge base options for the frontend.

    Returns:
        Dictionary of KB options with IDs and display names
    """
    return get_available_knowledge_bases()