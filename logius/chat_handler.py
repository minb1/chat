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
# Import HyDE and AugmentQuery prompt functions
from prompt.prompt_builder import create_prompt, HyDE, AugmentQuery # <<< Import AugmentQuery
from memory.redis_handler import log_message, format_chat_history_for_prompt
from reranking.reranker import SentenceTransformerReranker

# Configure logging
logger = logging.getLogger('rag_metrics')
logger.setLevel(logging.INFO) # Keep INFO level for general operation

# --- Assume your logging is configured elsewhere to output JSON ---

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
    use_hyde: bool = False,
    use_augmentation: bool = False # <<< Add new parameter
) -> Dict:
    """
    Main logic for handling user queries and generating responses.
    Includes improved error handling for LLM calls and prompt augmentation.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"
    error_response_message = "Sorry, an error occurred while generating the response."
    response = error_response_message
    docs_to_return = []
    hypothetical_doc_generated = False # Flag for HyDE logging
    augmented_query_generated = False # <<< Flag for Augmentation logging
    final_llm_error = False

    # --- Enforce Mutual Exclusivity ---
    # If both are somehow passed as True, prioritize one (e.g., HyDE) or disable both.
    # Here, we prioritize HyDE if both are True.
    if use_hyde and use_augmentation:
        logger.warning(f"Query ID {query_id}: Both HyDE and Augmentation were enabled. Prioritizing HyDE.")
        use_augmentation = False

    try:
        # ... (rest of the initial setup and logging remains the same) ...
        print( # Keep print for immediate console feedback if needed
            f"Processing message: '{message}', Use Reranker: {use_reranker}, "
            f"Use HyDE: {use_hyde}, Use Augmentation: {use_augmentation}, " # <<< Add Augmentation to print
            f"Retrieval K: {retrieval_top_k}, Reranker K: {reranker_top_k}, Model: {model_name}"
        )
        logger.info(
            f"Query ID {query_id}: Reranker={use_reranker}, HyDE={use_hyde}, "
            f"Augmentation={use_augmentation}, RetrievalK={retrieval_top_k}, " # <<< Add Augmentation to log
            f"RerankerK={reranker_top_k}, Model={model_name}"
         )

        # ... (Steps 1-4: Embedding, Retrieval, Reranking, Context Formatting) ...
        # --- Step 1: Get Embedding Handler ---
        embedding_model = get_embedding_model_for_kb(kb_id)
        embedding_handler = get_embedding_handler(embedding_model)

        # --- Step 1.5: Generate Query Embedding (with HyDE or Augmentation) ---
        query_embedding = None
        query_for_retrieval = message # Default to original message

        if use_hyde:
            print(f"HyDE enabled. Generating hypothetical document for query: '{message}' using model '{model_name}'...")
            try:
                hyde_prompt = HyDE(message)
                hyde_llm_handler = get_model_handler(model_name)
                hypothetical_document = hyde_llm_handler.generate_text(hyde_prompt)
                print(f"Generated Hypothetical Document:\n---\n{hypothetical_document}\n---")
                query_for_retrieval = hypothetical_document
                query_embedding = embedding_handler.get_embedding(query_for_retrieval)
                print("Generated embedding for hypothetical document.")
                hypothetical_doc_generated = True
            except Exception as e:
                print(f"Error during HyDE generation/embedding: {e}. Falling back to original query.")
                # Use logger.error or logger.warning, exception logs the stacktrace too
                logger.error(f"HyDE generation/embedding failed for query_id {query_id}: {e}")
                query_embedding = embedding_handler.get_embedding(message) # Fallback
                print("Generated embedding for original user query (HyDE fallback).")

        elif use_augmentation:
            print(f"Augmentation enabled. Generating augmented query for: '{message}' using model '{model_name}'...")
            try:
                augment_prompt = AugmentQuery(message)
                augment_llm_handler = get_model_handler(model_name)
                augmented_query = augment_llm_handler.generate_text(augment_prompt).strip()
                if not augmented_query or len(augmented_query) < 5:
                    print("Augmented query generation resulted in short/empty response. Falling back.")
                    logger.warning(f"Augmentation resulted in unusable query for query_id {query_id}. Original: '{message}', Augmented: '{augmented_query}'")
                    augmented_query = message # Fallback to original
                else:
                    print(f"Generated Augmented Query:\n---\n{augmented_query}\n---")
                    augmented_query_generated = True

                query_for_retrieval = augmented_query
                query_embedding = embedding_handler.get_embedding(query_for_retrieval)
                print("Generated embedding for augmented query.")

            except Exception as e:
                print(f"Error during Query Augmentation generation/embedding: {e}. Falling back to original query.")
                # Use logger.error or logger.warning
                logger.error(f"Query Augmentation generation/embedding failed for query_id {query_id}: {e}")
                query_embedding = embedding_handler.get_embedding(message) # Fallback
                print("Generated embedding for original user query (Augmentation fallback).")
        else:
            print("HyDE and Augmentation disabled. Generating embedding for original user query.")
            query_embedding = embedding_handler.get_embedding(message)
            print("Generated embedding for original user query.")

        if query_embedding is None:
             error_msg = "Failed to generate any query embedding."
             print(error_msg)
             logger.error(f"Embedding generation failed completely for query_id {query_id}.")
             # Ensure final_llm_error reflects this specific failure if needed downstream
             # final_llm_error = True # Or a more specific error flag
             return {"error": error_msg, "docs": []}

        # --- Step 2: Retrieve relevant document paths ---
        print(f"Getting vclient for KB: {kb_id}")
        vector_client = get_vector_client(kb_id)
        print(f"Getting top {retrieval_top_k} related vectors using {'HyDE doc' if use_hyde else ('augmented query' if use_augmentation else 'original query')}...")
        retrieved_paths = vector_client.retrieve_vectors(query_embedding, top_k=retrieval_top_k)
        print(f"Retrieved {len(retrieved_paths)} file paths from vector storage:", retrieved_paths)

        # --- Step 3: Retrieve content from PostgreSQL ---
        print("Retrieving chunk content from PostgreSQL...")
        context_dict_from_db: Dict[str, str] = retrieve_chunks_from_db(retrieved_paths)
        print(f"Retrieved {len(context_dict_from_db)} chunks from DB.")

        documents: List[Dict[str, str]] = [
            {"file_path": path, "content": content}
            for path, content in context_dict_from_db.items()
            if path in context_dict_from_db # Ensure path exists in retrieved content
        ]

        # --- Step 3.1: Conditionally Rerank ---
        query_for_reranker = message
        docs_for_context = []
        if use_reranker and documents:
            print(f"Reranking {len(documents)} documents against original query, keeping top {reranker_top_k}...")
            try:
                reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
                reranked_docs_list = reranker.rerank(
                    query=query_for_reranker,
                    documents=documents,
                    content_key="content",
                    top_k=reranker_top_k
                )
                print(f"Reranked to {len(reranked_docs_list)} documents.")
                docs_for_context = reranked_docs_list
                docs_to_return = reranked_docs_list # Includes scores
            except Exception as e:
                print(f"Error during reranking: {e}. Falling back.")
                # Use logger.error or logger.warning
                logger.error(f"Reranking failed for query_id {query_id}: {e}")
                docs_for_context = documents[:reranker_top_k]
                docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
        else:
            limit = reranker_top_k if use_reranker else retrieval_top_k
            docs_for_context = documents[:limit]
            docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
            if not documents: print("No documents retrieved initially.")
            elif not use_reranker: print(f"Skipping reranking. Using top {len(docs_to_return)} retrieved documents.")
            else: print(f"Using top {len(docs_to_return)} retrieved documents (reranker limit applied before reranking).")


        # --- Step 4: Format context ---
        context_string_for_prompt = "\n\n".join(
            f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
            for doc in docs_for_context
        ) if docs_for_context else "No relevant documents found."

        chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""
        prompt = create_prompt(message, context_string_for_prompt, chat_history)

        # --- Step 5: Generate response from LLM ---
        try:
            print(f"Generating final response using model: {model_name}")
            model_handler = get_model_handler(model_name)
            llm_response = model_handler.generate_text(prompt)
            print(f"LLM Response generated.")
            response = llm_response
        except Exception as e:
            final_llm_error = True # <<< Set the flag here
            response = error_response_message # Ensure error message is set
            print(f"Error generating final response from LLM ({model_name}): {e}")
            # Log the exception details immediately
            # Using logger.exception automatically adds stack trace and logs at ERROR level
            logger.exception(f"LLM generation failed for query_id {query_id} using model {model_name}")
            # No need to log separately here, logger.exception handles it.

        # --- Step 6: Log conversation ---
        if chat_id:
            log_message(chat_id, "user", message)
            log_message(chat_id, "assistant", response) # Log final response or error message
            print(f"Logged message and response to redis for chat_id: {chat_id}")

        # Prepare response info dictionary
        response_info = {
            "timestamp": timestamp,
            "query_id": query_id,
            "response_length": len(response),
            "kb_id": kb_id,
            "model_used": model_name,
            "hyde_used": use_hyde,
            "hyde_generated_doc": hypothetical_doc_generated,
            "augmentation_used": use_augmentation,
            "augmentation_generated_query": augmented_query_generated,
            "reranker_used": use_reranker if documents else False,
            "retrieval_top_k": retrieval_top_k,
            "reranker_top_k": reranker_top_k if use_reranker else None,
            "docs_retrieved_count": len(documents),
            "docs_returned_count": len(docs_to_return),
            "docs_returned_paths": [doc.get('file_path', 'unknown_path') for doc in docs_to_return],
            "llm_generation_error": final_llm_error # This flag is key
        }

        # --- *** MODIFICATION START *** ---
        # Log RESPONSE_GENERATED summary, using ERROR level if the LLM failed
        log_message_content = f"RESPONSE_GENERATED {json.dumps(response_info)}"
        if final_llm_error:
            logger.error(log_message_content) # Log as ERROR on LLM failure
        else:
            logger.info(log_message_content)  # Log as INFO on success
        # --- *** MODIFICATION END *** ---

        # Return response and docs
        print(f"Returning {len(docs_to_return)} documents to frontend.")
        # Return structure indicates error slightly differently now
        if final_llm_error:
             # The 'response' variable already holds the error message here
             return {"error": response, "docs": docs_to_return}
        else:
             # The 'response' variable holds the successful LLM response
             return {"response": response, "docs": docs_to_return}

    except Exception as e:
        # Catch any other unexpected errors during the whole process
        print(f"Unexpected error processing message for query_id {query_id}: {e}")
        # Log unexpected errors with stack trace
        logger.exception(f"Unexpected error for query_id {query_id}")
        # Ensure final return indicates an error
        return {"error": "An unexpected server error occurred.", "docs": []}

def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()