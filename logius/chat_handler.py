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
# Import HyDE prompt function
from prompt.prompt_builder import create_prompt, HyDE
from memory.redis_handler import log_message, format_chat_history_for_prompt
from reranking.reranker import SentenceTransformerReranker

# Configure logging
logger = logging.getLogger('rag_metrics')
logger.setLevel(logging.INFO)

# Define defaults
DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_RERANKER_TOP_K = 10


def process_user_message(
    message: str,
    chat_id: Optional[str] = None,
    model_name: str = 'gemini',
    kb_id: str = 'qdrant-logius',
    use_reranker: bool = True,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    reranker_top_k: int = DEFAULT_RERANKER_TOP_K,
    use_hyde: bool = False  # <<< Add HyDE parameter with default
) -> Dict:
    """
    Main logic for handling user queries and generating responses.

    Args:
        message: User's query text.
        chat_id: Chat session identifier.
        model_name: LLM model to use for response generation (and HyDE if enabled).
        kb_id: Knowledge base identifier to use for retrieval.
        use_reranker: Boolean flag to indicate whether to use the reranker.
        retrieval_top_k: Number of documents to retrieve initially from vector DB.
        reranker_top_k: Number of documents to return after reranking.
        use_hyde: Boolean flag to indicate whether to use HyDE generation.

    Returns:
        Dictionary containing the response and a list of retrieved/reranked documents.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"

    # Log received parameters
    print(
        f"Processing message: '{message}', Use Reranker: {use_reranker}, Use HyDE: {use_hyde}, "
        f"Retrieval K: {retrieval_top_k}, Reranker K: {reranker_top_k}"
    )
    logger.info(
        f"Query ID {query_id}: Reranker={use_reranker}, HyDE={use_hyde}, "
        f"RetrievalK={retrieval_top_k}, RerankerK={reranker_top_k}"
     )

    # --- Step 1: Get Embedding Handler (needed for both HyDE and standard query) ---
    embedding_model = get_embedding_model_for_kb(kb_id)
    embedding_handler = get_embedding_handler(embedding_model)

    # --- Step 1.5: Generate Query Embedding (HyDE or Standard) ---
    query_embedding = None
    hypothetical_doc_generated = False # Flag for logging

    if use_hyde:
        print(f"HyDE enabled. Generating hypothetical document for query: '{message}' using model '{model_name}'...")
        try:
            hyde_prompt = HyDE(message)
            # Use the same model selected by the user for consistency,
            # or choose a specific (potentially faster/cheaper) model for HyDE generation
            hyde_llm_handler = get_model_handler(model_name)
            hypothetical_document = hyde_llm_handler.generate_text(hyde_prompt)
            print(f"Generated Hypothetical Document:\n---\n{hypothetical_document}\n---")
            # Embed the *hypothetical* document
            query_embedding = embedding_handler.get_embedding(hypothetical_document)
            print("Generated embedding for hypothetical document.")
            hypothetical_doc_generated = True
        except Exception as e:
            print(f"Error generating hypothetical document or embedding: {e}. Falling back to standard query embedding.")
            logger.error(f"HyDE generation/embedding failed for query_id {query_id}: {e}")
            # Fallback: embed the original message if HyDE fails
            query_embedding = embedding_handler.get_embedding(message)
            print("Generated embedding for original user query (HyDE fallback).")
    else:
        # Standard path: embed the original user message
        print("HyDE disabled. Generating embedding for original user query.")
        query_embedding = embedding_handler.get_embedding(message)
        print("Generated embedding for original user query.")

    if query_embedding is None:
         # Should not happen if fallback works, but as a safety measure
         error_msg = "Failed to generate any query embedding."
         print(error_msg)
         logger.error(f"Embedding generation failed completely for query_id {query_id}.")
         return {"error": error_msg, "docs": []}

    # --- Step 2: Retrieve relevant document paths using the generated embedding ---
    print(f"Getting vclient for KB: {kb_id}")
    vector_client = get_vector_client(kb_id)
    print(f"Getting top {retrieval_top_k} related vectors now (using {'HyDE' if hypothetical_doc_generated else 'standard'} query embedding)...")
    retrieved_paths = vector_client.retrieve_vectors(
        query_embedding, # Use the generated embedding (either HyDE or standard)
        top_k=retrieval_top_k
    )
    print(f"Retrieved {len(retrieved_paths)} file paths from vector storage:", retrieved_paths)

    # --- Step 3: Retrieve content for the identified paths from PostgreSQL ---
    print("Retrieving chunk content from PostgreSQL...")
    context_dict_from_db: Dict[str, str] = retrieve_chunks_from_db(retrieved_paths)
    print(f"Retrieved {len(context_dict_from_db)} chunks from DB.")

    # Convert initial retrieval to standard document list format
    documents: List[Dict[str, str]] = [
        {"file_path": path, "content": content}
        for path, content in context_dict_from_db.items()
        if path in context_dict_from_db
    ]

    # --- Step 3.1: Conditionally Rerank retrieved documents ---
    docs_for_context = []
    docs_to_return = []
    if use_reranker and documents:
        print(f"Reranking {len(documents)} documents, keeping top {reranker_top_k}...")
        try:
            reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranked_docs_list = reranker.rerank(
                query=message, # Rerank based on the original query
                documents=documents,
                content_key="content",
                top_k=reranker_top_k
            )
            print(f"Reranked to {len(reranked_docs_list)} documents.")
            docs_for_context = reranked_docs_list
            docs_to_return = reranked_docs_list # Includes scores
        except Exception as e:
            print(f"Error during reranking: {e}. Falling back to original {len(documents)} documents (limited to reranker_top_k).")
            logger.error(f"Reranking failed for query_id {query_id}: {e}")
            # Fallback: Use original docs, limited to reranker_top_k
            docs_for_context = documents[:reranker_top_k]
            docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
    else:
        # No reranking or no initial documents
        limit = reranker_top_k if use_reranker else retrieval_top_k # If reranker is off, don't limit further
        docs_for_context = documents[:limit]
        docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
        if not documents:
             print("No documents retrieved initially.")
        elif not use_reranker:
             print(f"Skipping reranking. Using initially retrieved {len(docs_to_return)} documents.")


    # --- Step 4: Format context for the LLM prompt ---
    context_string_for_prompt = "\n\n".join(
        f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
        for doc in docs_for_context
    ) if docs_for_context else "No relevant documents found."

    # Prepare chat history and create prompt
    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
    prompt = create_prompt(message, context_string_for_prompt, chat_history)

    # --- Step 5: Generate response from LLM ---
    print(f"Generating response using model: {model_name}")
    model_handler = get_model_handler(model_name)
    response = model_handler.generate_text(prompt)
    print(f"LLM Response generated.")

    # --- Step 6: Log conversation if chat_id provided ---
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
        "hyde_used": use_hyde, # Log HyDE usage
        "hyde_generated_doc": hypothetical_doc_generated, # Log if doc was actually generated
        "reranker_used": use_reranker if documents else False,
        "retrieval_top_k": retrieval_top_k,
        "reranker_top_k": reranker_top_k if use_reranker else None,
        "docs_retrieved_count": len(documents),
        "docs_returned_count": len(docs_to_return)
    }
    logger.info(f"RESPONSE_GENERATED {json.dumps(response_info)}")

    # Return response and the final list of documents
    print(f"Returning {len(docs_to_return)} documents to frontend.")
    return {"response": response, "docs": docs_to_return}

def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()