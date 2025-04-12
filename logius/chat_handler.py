# chat_handler.py
import os
import json
import logging
import datetime
from typing import Dict, List, Optional

from database.db_retriever import retrieve_chunks_from_db
from model.model_factory import get_model_handler # Assuming VLLM call is in here
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
logger.setLevel(logging.INFO) # Keep INFO level

# --- Assume your logging is configured elsewhere to output JSON ---

# Define defaults
DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_RERANKER_TOP_K = 10


def process_user_message(
    message: str,
    chat_id: Optional[str] = None,
    model_name: str = 'gemini', # <<< Ensure this is set to your VLLM model name for testing VLLM errors
    kb_id: str = 'qdrant-logius',
    use_reranker: bool = True,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    reranker_top_k: int = DEFAULT_RERANKER_TOP_K,
    use_hyde: bool = False
) -> Dict:
    """
    Main logic for handling user queries and generating responses.
    Includes improved error handling for LLM calls.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"
    # Default error response if LLM fails later
    error_response_message = "Sorry, an error occurred while generating the response."
    response = error_response_message # Set default
    docs_to_return = [] # Initialize here
    hypothetical_doc_generated = False # Flag for logging
    final_llm_error = False # Flag for specific LLM error

    try:
        # Log received parameters
        print( # Keep print for immediate console feedback if needed
            f"Processing message: '{message}', Use Reranker: {use_reranker}, Use HyDE: {use_hyde}, "
            f"Retrieval K: {retrieval_top_k}, Reranker K: {reranker_top_k}, Model: {model_name}"
        )
        logger.info(
            f"Query ID {query_id}: Reranker={use_reranker}, HyDE={use_hyde}, "
            f"RetrievalK={retrieval_top_k}, RerankerK={reranker_top_k}, Model={model_name}"
         )

        # --- Step 1: Get Embedding Handler ---
        embedding_model = get_embedding_model_for_kb(kb_id)
        embedding_handler = get_embedding_handler(embedding_model)

        # --- Step 1.5: Generate Query Embedding ---
        query_embedding = None
        if use_hyde:
            print(f"HyDE enabled. Generating hypothetical document for query: '{message}' using model '{model_name}'...")
            try:
                hyde_prompt = HyDE(message)
                hyde_llm_handler = get_model_handler(model_name) # This could fail
                hypothetical_document = hyde_llm_handler.generate_text(hyde_prompt)
                print(f"Generated Hypothetical Document:\n---\n{hypothetical_document}\n---")
                query_embedding = embedding_handler.get_embedding(hypothetical_document)
                print("Generated embedding for hypothetical document.")
                hypothetical_doc_generated = True
            except Exception as e:
                # Log HyDE specific error using logger.exception
                print(f"Error during HyDE generation/embedding: {e}. Falling back.")
                logger.exception(f"HyDE generation/embedding failed for query_id {query_id}") # Use exception logger
                # Fallback: embed the original message
                query_embedding = embedding_handler.get_embedding(message)
                print("Generated embedding for original user query (HyDE fallback).")
        else:
            print("HyDE disabled. Generating embedding for original user query.")
            query_embedding = embedding_handler.get_embedding(message)
            print("Generated embedding for original user query.")

        if query_embedding is None:
             error_msg = "Failed to generate any query embedding."
             print(error_msg)
             logger.error(f"Embedding generation failed completely for query_id {query_id}.")
             return {"error": error_msg, "docs": []} # Return error immediately

        # --- Step 2: Retrieve relevant document paths ---
        print(f"Getting vclient for KB: {kb_id}")
        vector_client = get_vector_client(kb_id)
        print(f"Getting top {retrieval_top_k} related vectors now...")
        retrieved_paths = vector_client.retrieve_vectors(query_embedding, top_k=retrieval_top_k)
        print(f"Retrieved {len(retrieved_paths)} file paths from vector storage:", retrieved_paths)

        # --- Step 3: Retrieve content from PostgreSQL ---
        print("Retrieving chunk content from PostgreSQL...")
        context_dict_from_db: Dict[str, str] = retrieve_chunks_from_db(retrieved_paths)
        print(f"Retrieved {len(context_dict_from_db)} chunks from DB.")

        documents: List[Dict[str, str]] = [
            {"file_path": path, "content": content}
            for path, content in context_dict_from_db.items()
            if path in context_dict_from_db
        ]

        # --- Step 3.1: Conditionally Rerank ---
        docs_for_context = []
        if use_reranker and documents:
            print(f"Reranking {len(documents)} documents, keeping top {reranker_top_k}...")
            try:
                # Consider making reranker model configurable if needed
                reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                reranked_docs_list = reranker.rerank(
                    query=message, documents=documents, content_key="content", top_k=reranker_top_k
                )
                print(f"Reranked to {len(reranked_docs_list)} documents.")
                docs_for_context = reranked_docs_list
                docs_to_return = reranked_docs_list # Includes scores
            except Exception as e:
                print(f"Error during reranking: {e}. Falling back.")
                logger.exception(f"Reranking failed for query_id {query_id}") # Log exception
                docs_for_context = documents[:reranker_top_k]
                docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
        else:
            limit = reranker_top_k if use_reranker else retrieval_top_k
            docs_for_context = documents[:limit]
            docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
            if not documents: print("No documents retrieved initially.")
            elif not use_reranker: print(f"Skipping reranking. Using {len(docs_to_return)} documents.")

        # --- Step 4: Format context ---
        context_string_for_prompt = "\n\n".join(
            f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
            for doc in docs_for_context
        ) if docs_for_context else "No relevant documents found."

        chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
        prompt = create_prompt(message, context_string_for_prompt, chat_history)

        # --- Step 5: Generate response from LLM ---
        # <<< WRAP THIS SECTION IN TRY/EXCEPT >>>
        try:
            print(f"Generating response using model: {model_name}")
            model_handler = get_model_handler(model_name) # This factory might raise error
            llm_response = model_handler.generate_text(prompt) # This calls VLLMHandler.generate_text
            print(f"LLM Response generated.")
            response = llm_response # Update response only on success
        except Exception as e:
            # Log the specific error during final LLM generation
            final_llm_error = True # Set flag
            print(f"Error generating final response from LLM ({model_name}): {e}")
            # Use logger.exception to include stack trace!
            logger.exception(f"LLM generation failed for query_id {query_id} using model {model_name}")
            # `response` variable already holds the default error message set earlier

        # --- Step 6: Log conversation ---
        if chat_id:
            log_message(chat_id, "user", message)
            # Log the actual response OR the error message shown to the user
            log_message(chat_id, "assistant", response)
            print(f"Logged message and response to redis for chat_id: {chat_id}")

        # Log response info - This will log even if LLM failed, showing the error state
        response_info = {
            "timestamp": timestamp,
            "query_id": query_id,
            "response_length": len(response),
            "kb_id": kb_id,
            "model_used": model_name,
            "hyde_used": use_hyde,
            "hyde_generated_doc": hypothetical_doc_generated,
            "reranker_used": use_reranker if documents else False,
            "retrieval_top_k": retrieval_top_k,
            "reranker_top_k": reranker_top_k if use_reranker else None,
            "docs_retrieved_count": len(documents),
            "docs_returned_count": len(docs_to_return),
            "llm_generation_error": final_llm_error # Add the error flag
        }
        # Use INFO level for metric, but include error flag if applicable
        logger.info(f"RESPONSE_GENERATED {json.dumps(response_info)}")

        # Return response (either generated or the error message) and docs
        print(f"Returning {len(docs_to_return)} documents to frontend.")
        # If the final LLM call failed, return the error message set earlier
        if final_llm_error:
             return {"error": response, "docs": docs_to_return} # Return default error message
        else:
             return {"response": response, "docs": docs_to_return} # Return successful response

    except Exception as e:
        # Catch any other unexpected errors in the whole process
        print(f"Unexpected error processing message for query_id {query_id}: {e}")
        logger.exception(f"Unexpected error for query_id {query_id}") # Log trace
        # Return a generic error response
        return {"error": "An unexpected server error occurred.", "docs": []}

def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()