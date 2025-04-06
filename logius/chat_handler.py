import os
import json
import logging
import datetime
from typing import Dict, List, Optional

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
from reranking.reranker import SentenceTransformerReranker # Keep reranker import

# Configure logging
logger = logging.getLogger('rag_metrics')
logger.setLevel(logging.INFO)

# Define defaults here as well, matching the view defaults is good practice
DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_RERANKER_TOP_K = 10


def process_user_message(
    message: str,
    chat_id: Optional[str] = None,
    model_name: str = 'gemini',
    kb_id: str = 'qdrant-logius',
    use_reranker: bool = True,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K, # <<< Add parameter with default
    reranker_top_k: int = DEFAULT_RERANKER_TOP_K    # <<< Add parameter with default
) -> Dict:
    """
    Main logic for handling user queries and generating responses.

    Args:
        message: User's query text.
        chat_id: Chat session identifier.
        model_name: LLM model to use for response generation.
        kb_id: Knowledge base identifier to use for retrieval.
        use_reranker: Boolean flag to indicate whether to use the reranker.
        retrieval_top_k: Number of documents to retrieve initially from vector DB.
        reranker_top_k: Number of documents to return after reranking.

    Returns:
        Dictionary containing the response and a list of retrieved/reranked documents.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"

    # Log received parameters
    print(
        f"Processing message: '{message}', Use Reranker: {use_reranker}, "
        f"Retrieval K: {retrieval_top_k}, Reranker K: {reranker_top_k}"
    )
    logger.info(
        f"Query ID {query_id}: Reranker={use_reranker}, "
        f"RetrievalK={retrieval_top_k}, RerankerK={reranker_top_k}"
     )


    # Step 1: Get embeddings for the query
    embedding_model = get_embedding_model_for_kb(kb_id)
    embedding_handler = get_embedding_handler(embedding_model)
    user_embedding = embedding_handler.get_embedding(message)

    # Step 2: Retrieve relevant document paths using retrieval_top_k
    print(f"Getting vclient for KB: {kb_id}")
    vector_client = get_vector_client(kb_id)
    print(f"Getting top {retrieval_top_k} related vectors now...")
    retrieved_paths = vector_client.retrieve_vectors(
        user_embedding,
        top_k=retrieval_top_k # <<< Use the parameter here
    )
    print(f"Retrieved {len(retrieved_paths)} file paths from vector storage:", retrieved_paths)

    # Step 3: Retrieve content for the identified paths from PostgreSQL
    print("Retrieving chunk content from PostgreSQL...")
    context_dict_from_db: Dict[str, str] = retrieve_chunks_from_db(retrieved_paths)
    print(f"Retrieved {len(context_dict_from_db)} chunks from DB.")

    # Convert initial retrieval to standard document list format
    documents: List[Dict[str, str]] = [
        {"file_path": path, "content": content}
        for path, content in context_dict_from_db.items()
        # Ensure we only process paths that were successfully retrieved from DB
        if path in context_dict_from_db
    ]

    # Step 3.1: Conditionally Rerank retrieved documents using reranker_top_k
    if use_reranker and documents:
        print(f"Reranking {len(documents)} documents, keeping top {reranker_top_k}...")
        try:
            reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            # Pass reranker_top_k to the rerank method
            reranked_docs_list = reranker.rerank(
                query=message,
                documents=documents,
                content_key="content",
                top_k=reranker_top_k # <<< Use the parameter here
            )
            print(f"Reranked to {len(reranked_docs_list)} documents.")
            docs_for_context = reranked_docs_list
            docs_to_return = reranked_docs_list # This list includes scores
        except Exception as e:
            print(f"Error during reranking: {e}. Falling back to original {len(documents)} documents.")
            logger.error(f"Reranking failed for query_id {query_id}: {e}")
            # Fallback: Use original docs, but still limit to reranker_top_k for consistency if desired?
            # Or just use all originally retrieved? Let's use the originally retrieved ones for fallback.
            docs_for_context = documents
            # Add 'rerank_score': None for consistency
            docs_to_return = [{**doc, 'rerank_score': None} for doc in documents]
    else:
        if not documents:
             print("No documents retrieved initially.")
        elif not use_reranker:
             print(f"Skipping reranking as requested. Using initially retrieved {len(documents)} documents.")
        docs_for_context = documents
        # Add 'rerank_score': None for consistency
        docs_to_return = [{**doc, 'rerank_score': None} for doc in documents]


    # Step 4: Format context for the LLM prompt
    context_string_for_prompt = "\n\n".join(
        f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
        for doc in docs_for_context # Use the potentially reranked list
    ) if docs_for_context else "No relevant documents found."

    # Prepare chat history and create prompt
    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
    prompt = create_prompt(message, context_string_for_prompt, chat_history)

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
        "reranker_used": use_reranker if documents else False,
        "retrieval_top_k": retrieval_top_k, # Log K values used
        "reranker_top_k": reranker_top_k if use_reranker else None,
        "docs_retrieved_count": len(documents), # Initial count before reranking
        "docs_returned_count": len(docs_to_return) # Final count after potential reranking
    }
    logger.info(f"RESPONSE_GENERATED {json.dumps(response_info)}")

    # Return response and the final list of documents (potentially reranked and limited)
    print(f"Returning {len(docs_to_return)} documents to frontend.")
    return {"response": response, "docs": docs_to_return}


def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()