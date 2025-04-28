# chat_handler.py
import os
import json
import logging
import datetime
import sys
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4
from django.db import transaction
import numpy as np # For cosine similarity
from sklearn.metrics.pairwise import cosine_similarity # Alternative for cosine similarity

try:
    from logius.models import Document, ChatQuery
except ImportError:
    # Define dummy classes if models cannot be imported (e.g., during setup)
    class Document: pass
    class ChatQuery: pass

from database.db_retriever import retrieve_chunks_from_db
from model.model_factory import get_model_handler
from embedding.embedding_factory import get_embedding_handler
from vectorstorage.vector_factory import (
    get_vector_client,
    get_embedding_model_for_kb,
    get_available_knowledge_bases
)
# Assuming redis_handler has format_chat_history_for_prompt(chat_id, max_turns=3)
from memory.redis_handler import log_message, format_chat_history_for_prompt
from reranking.reranker import SentenceTransformerReranker
# Import the new CQR prompt
from prompt.prompt_builder import create_prompt, HyDE, AugmentQuery, create_cqr_prompt


# Configure logging
logger = logging.getLogger('rag_metrics')
logger.setLevel(logging.INFO) # Or DEBUG for more verbose logs

# --- Constants for Conversational RAG ---
DEFAULT_FRESH_K = 6  # K: Number of fresh documents to retrieve
DEFAULT_STICKY_S = 4  # S: Number of sticky documents to keep/retrieve
MAX_TOTAL_DOCS = 10 # Target total docs after merging and reranking (K+S)
SIMILARITY_THRESHOLD = 0.30 # Threshold to flush sticky memory
HISTORY_TURNS_FOR_CQR = 1 # Number of previous Q-A pairs for CQR context
HISTORY_TURNS_FOR_PROMPT = 3 # Number of turns for final answer prompt
SUMMARY_MAX_TOKENS = 100 # Placeholder if using summaries instead of raw history
# --- End Constants ---

DEFAULT_RETRIEVAL_TOP_K = DEFAULT_FRESH_K # Base retrieval K for non-conversational or first turn
DEFAULT_RERANKER_TOP_K = MAX_TOTAL_DOCS # Rerank to the target total
ERROR_RESPONSE_MESSAGE = "Sorry, an error occurred while generating the response."

# --- Helper Functions ---

def _get_previous_turn_data(chat_id: str) -> Optional[ChatQuery]:
    """Fetches the most recent ChatQuery object for a given chat_id."""
    if not chat_id:
        return None
    try:
        # Fetch the latest query for this chat session
        return ChatQuery.objects.filter(chat_id=chat_id).latest('created_at')
    except ChatQuery.DoesNotExist:
        logger.info(f"No previous turns found for chat_id {chat_id}.")
        return None
    except Exception as e:
        logger.error(f"Error retrieving previous turn data for chat_id {chat_id}: {e}", exc_info=True)
        return None

def _rewrite_query_cqr(
    user_query: str,
    chat_history_cqr: str,
    model_name: str,
    query_id: str
) -> str:
    """Generates a standalone query using CQR."""
    logger.info(f"Query ID {query_id}: Rewriting query using CQR with model {model_name}.")
    try:
        cqr_prompt = create_cqr_prompt(user_query, chat_history_cqr)
        cqr_llm_handler = get_model_handler(model_name) # Use configured model
        rewritten_query = cqr_llm_handler.generate_text(cqr_prompt).strip()

        # Basic validation
        if not rewritten_query or len(rewritten_query) < 5:
            logger.warning(f"Query ID {query_id}: CQR resulted in short/empty query '{rewritten_query}'. Falling back to original query.")
            return user_query
        if rewritten_query == user_query:
             logger.info(f"Query ID {query_id}: CQR returned the original query.")
        else:
             logger.info(f"Query ID {query_id}: Original: '{user_query}', Rewritten: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        logger.error(f"Query ID {query_id}: CQR failed: {e}. Falling back to original query.", exc_info=True)
        return user_query

def _calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        logger.warning("Cannot calculate cosine similarity due to invalid vectors.")
        return 0.0 # Return neutral similarity if vectors are invalid
    try:
        # Using numpy
        # sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Using sklearn (handles reshaping for single vectors)
        sim = cosine_similarity(np.array(vec1).reshape(1, -1), np.array(vec2).reshape(1, -1))[0][0]
        return float(sim)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        return 0.0

def _retrieve_sticky_results(
    sticky_ids: List[Union[str, int, uuid.UUID]],
    kb_id: str,
    query_id: str
) -> List[Dict]:
    """Retrieves documents directly by their IDs (sticky memory)."""
    if not sticky_ids:
        return []
    logger.info(f"Query ID {query_id}: Retrieving {len(sticky_ids)} sticky documents by ID.")
    try:
        vector_client = get_vector_client(kb_id)
        # Use the new method to fetch by ID
        sticky_results_qdrant = vector_client.retrieve_by_ids(sticky_ids)

        # We only have payload from Qdrant, need full Document object from DB
        sticky_paths = [
            item['payload']['file_path']
            for item in sticky_results_qdrant
            if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']
        ]

        if not sticky_paths:
            logger.warning(f"Query ID {query_id}: No valid file_paths found in sticky results from Qdrant.")
            return []

        sticky_document_objects = retrieve_chunks_from_db(sticky_paths)
        doc_map = {doc.file_path: doc for doc in sticky_document_objects}

        # Combine Qdrant info (ID) with DB info (content, object)
        final_sticky_docs = []
        for item in sticky_results_qdrant:
            file_path = item.get('payload', {}).get('file_path')
            if file_path and file_path in doc_map:
                doc_obj = doc_map[file_path]
                final_sticky_docs.append({
                    "id": item['id'], # Qdrant ID
                    "file_path": doc_obj.file_path,
                    "content": doc_obj.content,
                    "document": doc_obj,
                    "retrieval_score": None, # No relevance score for sticky retrieval
                    "rerank_score": None,
                    "is_sticky": True # Mark as sticky
                })
            else:
                logger.warning(f"Query ID {query_id}: Could not find DB document for sticky path: {file_path} (Qdrant ID: {item.get('id')})")

        logger.info(f"Query ID {query_id}: Successfully retrieved {len(final_sticky_docs)} full sticky documents.")
        return final_sticky_docs

    except Exception as e:
        logger.error(f"Query ID {query_id}: Error retrieving sticky documents: {e}", exc_info=True)
        return []

def _merge_and_deduplicate_results(
    fresh_results: List[Dict],
    sticky_results: List[Dict],
    max_total: int
) -> List[Dict]:
    """Merges fresh and sticky results, removing duplicates based on 'file_path'."""
    # Prioritize fresh results if duplicates exist, but keep sticky flag if applicable
    merged_dict = {}

    # Add sticky first, marking them
    for doc in sticky_results:
        doc['is_sticky'] = True
        merged_dict[doc['file_path']] = doc

    # Add fresh, potentially overwriting/merging with sticky
    for doc in fresh_results:
        file_path = doc['file_path']
        if file_path in merged_dict:
            # If already present from sticky, update scores but keep sticky flag
            merged_dict[file_path]['retrieval_score'] = doc.get('retrieval_score')
            # Keep other fields from sticky like 'id' if needed, assuming fresh retrieval doesn't provide Qdrant ID directly in this structure
        else:
            # New document from fresh search
             doc['is_sticky'] = False
             merged_dict[file_path] = doc

    # Convert back to list
    merged_list = list(merged_dict.values())

    # Optional: Sort primarily by retrieval_score (desc), then maybe sticky flag?
    # This simple merge doesn't enforce strict K/S counts *before* reranking
    # Reranking will determine the final top documents anyway.
    # merged_list.sort(key=lambda x: (x.get('retrieval_score') or -1.0), reverse=True)

    logger.info(f"Merged {len(fresh_results)} fresh and {len(sticky_results)} sticky results into {len(merged_list)} unique documents.")

    # We don't strictly limit to max_total here, reranker will do that
    return merged_list


# --- Main Function ---

@transaction.atomic # Wrap in transaction for database operations
def process_user_message(
    message: str,
    chat_id: Optional[str] = None,
    model_name: str = 'gemini', # Model for CQR and Final Answer
    kb_id: str = 'qdrant-logius',
    use_reranker: bool = True,
    # Retrieval K/S now handled by conversational logic
    # retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K, # Remove this, use fresh_k/sticky_s
    reranker_top_k: int = DEFAULT_RERANKER_TOP_K, # Keep this for final selection
    # Deprecate HyDE/Augment in favor of CQR for conversational context
    # use_hyde: bool = False,
    # use_augmentation: bool = False
) -> Dict:
    """
    Process a user message in a conversational RAG pipeline with CQR and Sticky Memory.

    Args:
        message: The user's input message.
        chat_id: Chat identifier for conversation history and state. If None, treated as first turn.
        model_name: The LLM to use for CQR and response generation.
        kb_id: Knowledge base identifier for vector retrieval.
        use_reranker: Whether to rerank retrieved documents.
        reranker_top_k: Number of documents to keep after reranking (<= MAX_TOTAL_DOCS).

    Returns:
        Dictionary containing response, query_id, chat_id, or error and relevant documents.
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = str(uuid4())
    # If no chat_id provided, generate one for this turn (acts like a single-turn chat)
    is_follow_up = bool(chat_id)
    if not chat_id:
        chat_id = str(uuid4()) # Assign a new chat ID for this interaction
        logger.info(f"No chat_id provided. Starting new session: {chat_id}")

    response = ERROR_RESPONSE_MESSAGE
    final_docs_to_return_serialized = []
    rewritten_query = message # Default to original query
    rewritten_query_embedding = None
    final_doc_ids_for_next_turn = [] # Qdrant IDs or unique file_paths

    # Track features used in this turn
    query_features = {
        "cqr_used": False,
        "sticky_memory_used": False,
        "sticky_memory_flushed": False,
        "cosine_similarity": None,
        "fresh_k_used": DEFAULT_FRESH_K,
        "sticky_s_used": DEFAULT_STICKY_S,
        "final_llm_error": False,
        # "hypothetical_doc_generated": False, # Deprecated
        # "augmented_query_generated": False, # Deprecated
    }

    try:
        logger.info(f"--- Turn Start --- Query ID: {query_id}, Chat ID: {chat_id}, Message: '{message}' ---")
        log_request_parameters(query_id, message, use_reranker, False, False, # No hyde/augment
                              DEFAULT_FRESH_K, DEFAULT_STICKY_S, reranker_top_k, model_name) # Log K/S

        # 1. Get Previous Turn State (if applicable)
        previous_turn: Optional[ChatQuery] = _get_previous_turn_data(chat_id)
        previous_sticky_ids = []
        previous_query_embedding = None

        if is_follow_up and previous_turn:
            logger.info(f"Query ID {query_id}: Follow-up turn detected. Previous query ID: {previous_turn.id}")
            previous_sticky_ids = previous_turn.final_doc_ids or []
            previous_query_embedding = previous_turn.rewritten_query_embedding
            # Ensure embedding is loaded correctly (handle JSON vs ArrayField)
            if isinstance(previous_query_embedding, str): # Simple check for JSON string
                 try: previous_query_embedding = json.loads(previous_query_embedding)
                 except: previous_query_embedding = None
            logger.info(f"Query ID {query_id}: Loaded {len(previous_sticky_ids)} sticky IDs from previous turn.")
        else:
            logger.info(f"Query ID {query_id}: First turn or no previous state found for chat_id {chat_id}.")
            is_follow_up = False # Treat as first turn if no previous state


        # 2. Query Rewriting (CQR) - Only if it's a follow-up
        embedding_handler = get_embedding_handler(get_embedding_model_for_kb(kb_id)) # Needed regardless

        if is_follow_up:
            # Get history snippet for CQR (e.g., last Q+A)
            # Option 1: Use Redis history formatter
            chat_history_for_cqr = format_chat_history_for_prompt(chat_id, max_turns=HISTORY_TURNS_FOR_CQR)
            # Option 2: Use previous ChatQuery object directly (simpler if only 1 turn needed)
            # chat_history_for_cqr = f"Vorige Vraag: {previous_turn.user_query}\nVorige Antwoord: {previous_turn.llm_response[:200]}..." # Example
            
            rewritten_query = _rewrite_query_cqr(message, chat_history_for_cqr, model_name, query_id)
            query_features["cqr_used"] = (rewritten_query != message)
        else:
            # First turn, use original message as the query for retrieval
            rewritten_query = message

        # 3. Embed the (potentially rewritten) query
        try:
            query_embedding = embedding_handler.get_embedding(rewritten_query)
            if query_embedding is None: raise ValueError("Embedding generation returned None")
            logger.info(f"Query ID {query_id}: Generated embedding for query: '{rewritten_query[:100]}...'")
        except Exception as embed_e:
            logger.error(f"Embedding generation failed for query_id {query_id}: {embed_e}", exc_info=True)
            # Critical failure, cannot proceed with retrieval
            return {"error": "Failed to generate query embedding", "docs": [], "query_id": query_id, "chat_id": chat_id}


        # 4. Adaptive Sticky Memory Logic
        current_fresh_k = DEFAULT_FRESH_K
        current_sticky_s = DEFAULT_STICKY_S
        use_sticky = False # Default to not using sticky memory

        if is_follow_up and previous_sticky_ids and previous_query_embedding:
            # Calculate similarity between current rewritten query and previous rewritten query
            similarity = _calculate_cosine_similarity(query_embedding, previous_query_embedding)
            query_features["cosine_similarity"] = round(similarity, 4)
            logger.info(f"Query ID {query_id}: Cosine similarity with previous query: {similarity:.4f}")

            if similarity < SIMILARITY_THRESHOLD:
                # Flush sticky memory - Topic changed significantly
                logger.info(f"Query ID {query_id}: Similarity < {SIMILARITY_THRESHOLD}. Flushing sticky memory.")
                previous_sticky_ids = [] # Clear the IDs
                query_features["sticky_memory_flushed"] = True
                # Optional: Increase K when flushing?
                current_fresh_k = MAX_TOTAL_DOCS # Retrieve more fresh docs
                current_sticky_s = 0
            else:
                # Keep sticky memory - Topic is related
                logger.info(f"Query ID {query_id}: Similarity >= {SIMILARITY_THRESHOLD}. Keeping sticky memory active.")
                use_sticky = True
                query_features["sticky_memory_used"] = True
                # Optional: Adaptive S/K based on similarity (Example)
                # if similarity > 0.7: # Very similar, prioritize sticky
                #     current_sticky_s = 6
                #     current_fresh_k = 4
                # else: # Moderately similar
                #     current_sticky_s = 4
                #     current_fresh_k = 6
                # For simplicity, we stick to defaults if not flushing
                current_sticky_s = DEFAULT_STICKY_S
                current_fresh_k = DEFAULT_FRESH_K

        else:
             # First turn or no previous state with embedding/IDs
             logger.info(f"Query ID {query_id}: Not using sticky memory (first turn or missing previous state).")
             current_fresh_k = MAX_TOTAL_DOCS # Retrieve more fresh docs for the first query
             current_sticky_s = 0

        query_features["fresh_k_used"] = current_fresh_k
        query_features["sticky_s_used"] = current_sticky_s if use_sticky else 0

        # 5. Hybrid Retrieval
        # 5a. Retrieve Fresh Documents
        logger.info(f"Query ID {query_id}: Retrieving {current_fresh_k} fresh documents.")
        fresh_results_qdrant, fresh_retrieved_paths = retrieve_vector_results(
            kb_id, query_embedding, current_fresh_k, query_id
        )
        fresh_document_objects = retrieve_chunks_from_db(fresh_retrieved_paths)
        fresh_score_map = {item['payload']['file_path']: item['score'] for item in fresh_results_qdrant
                           if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']}
        
        fresh_intermediate_docs = [
            {
                "id": item['id'], # Keep Qdrant ID if available
                "file_path": doc.file_path,
                "content": doc.content,
                "document": doc,
                "retrieval_score": fresh_score_map.get(doc.file_path),
                "rerank_score": None,
                 "is_sticky": False
            }
            for doc in fresh_document_objects
            # Match based on DB object path existing in Qdrant results' paths
            if doc.file_path in fresh_score_map
            # Find corresponding Qdrant result to get ID
            for item in fresh_results_qdrant if item.get('payload', {}).get('file_path') == doc.file_path
        ]
        logger.info(f"Query ID {query_id}: Retrieved {len(fresh_intermediate_docs)} full fresh documents.")


        # 5b. Retrieve Sticky Documents
        sticky_intermediate_docs = []
        if use_sticky and previous_sticky_ids:
            sticky_intermediate_docs = _retrieve_sticky_results(previous_sticky_ids, kb_id, query_id)

        # 5c. Merge and Deduplicate
        merged_intermediate_docs = _merge_and_deduplicate_results(
            fresh_intermediate_docs, sticky_intermediate_docs, MAX_TOTAL_DOCS
        )
        logger.info(f"Query ID {query_id}: Total unique documents for reranking: {len(merged_intermediate_docs)}")

        # 6. Reranking
        # Rerank the *merged* set using the *rewritten* query
        docs_for_context, docs_for_ranking = apply_reranking(
            merged_intermediate_docs,
            rewritten_query, # Use the rewritten query for reranking relevance
            use_reranker,
            reranker_top_k, # Use the specified limit after reranking
            len(merged_intermediate_docs), # Pass the total number before reranking
            query_id
        )

        # Ensure docs_for_context has the 'document' object needed later
        # apply_reranking should already preserve the structure based on your previous code

        # 7. Answer Generation
        # Get chat history buffer for the final prompt
        chat_history_for_prompt = format_chat_history_for_prompt(chat_id, max_turns=HISTORY_TURNS_FOR_PROMPT)

        response = generate_response(
            message, # Pass the ORIGINAL user message to the LLM
            docs_for_context,
            chat_history_for_prompt, # Pass formatted history
            model_name,
            query_id,
            query_features, # Pass features dict to update final_llm_error
            rewritten_query # Pass rewritten query for potential inclusion in prompt (optional)
        )

        # 8. Prepare data for persistence and response
        # Get the IDs of the final top documents for the *next* turn's sticky memory
        # Use Qdrant 'id' if available, otherwise fallback to 'file_path' (ensure file_path is unique enough)
        final_doc_ids_for_next_turn = [
            d.get('id', d.get('file_path')) # Prioritize Qdrant ID
            for d in docs_for_ranking # Use the reranked and trimmed list
            if d.get('id') or d.get('file_path') # Ensure there is some identifier
        ][:MAX_TOTAL_DOCS] # Limit to the max sticky size

        logger.info(f"Query ID {query_id}: Storing {len(final_doc_ids_for_next_turn)} doc IDs for next turn's sticky memory.")

        # Serialize documents for the API response
        final_docs_to_return_serialized = serialize_documents_for_response(docs_for_ranking) # Use reranked list

        # 9. Persist State (including data for next turn)
        # @transaction.atomic handles commit/rollback
        doc_tags = [doc['document'].doc_tag for doc in docs_for_context if
                    doc.get('document') and getattr(doc['document'], 'doc_tag', None)]

        # Convert embedding to list for JSONField or keep as list for ArrayField
        embedding_to_save = query_embedding if isinstance(query_embedding, list) else query_embedding.tolist() if query_embedding is not None else None

        ChatQuery.objects.create(
            id=query_id,
            chat_id=chat_id,
            user_query=message,
            rewritten_query=rewritten_query,
            rewritten_query_embedding=embedding_to_save, # Store the embedding
            llm_response=response if not query_features["final_llm_error"] else None,
            final_doc_ids=final_doc_ids_for_next_turn, # Save IDs for next turn
            doc_tag=doc_tags, # Tags of docs used in *this* response
            # file_paths field is replaced by final_doc_ids
            model_used=model_name,
        )
        logger.info(f"Query ID {query_id}: ChatQuery object created/saved successfully.")

        # 10. Log conversation turn to Redis (if needed for simple history buffer)
        log_message(chat_id, "user", message)
        log_message(chat_id, "assistant", response)

        # 11. Log Metrics
        log_doc_details = prepare_doc_details_for_logging(final_docs_to_return_serialized)
        log_response_metrics(
            timestamp=timestamp,
            query_id=query_id,
            chat_id=chat_id, # Add chat_id to logs
            response=response,
            kb_id=kb_id,
            model_name=model_name,
            use_reranker=use_reranker,
            reranker_top_k=reranker_top_k,
            # Log conversational features
            query_features=query_features,
            # Counts reflect the final state
            docs_retrieved_count=len(merged_intermediate_docs), # Total unique docs before rerank
            docs_found_in_db_count=len(merged_intermediate_docs), # Assuming merge handles DB lookup
            docs_returned_count=len(final_docs_to_return_serialized),
            log_doc_details=log_doc_details
        )

        # 12. Return response
        response_payload = {
            "response": response,
            "docs": final_docs_to_return_serialized,
            "query_id": query_id,
            "chat_id": chat_id # Return chat_id so client can maintain session
        }
        if query_features["final_llm_error"]:
             response_payload["error"] = response # Overwrite response field if LLM failed
             response_payload["response"] = ERROR_RESPONSE_MESSAGE # Set standard error message

        return response_payload

    except Exception as e:
        logger.exception(f"Unexpected error processing message for query_id {query_id}, chat_id {chat_id}")
        # Attempt to save minimal error state if possible (outside transaction might be tricky)
        # Log error metrics
        log_response_metrics(
            timestamp=timestamp, query_id=query_id, chat_id=chat_id, response=ERROR_RESPONSE_MESSAGE,
            kb_id=kb_id, model_name=model_name, use_reranker=use_reranker, reranker_top_k=reranker_top_k,
            query_features={**query_features, "final_llm_error": True, "unexpected_error": str(e)},
            docs_retrieved_count=0, docs_found_in_db_count=0, docs_returned_count=0, log_doc_details=[]
        )
        return {"error": "An unexpected server error occurred.", "docs": [], "query_id": query_id, "chat_id": chat_id}


# --- Updated Helper Functions ---

def apply_reranking(
    intermediate_docs: List[Dict],
    query: str, # Should be the rewritten query for relevance
    use_reranker: bool,
    reranker_top_k: int,
    retrieval_top_k: int, # This might be less relevant now, use len(intermediate_docs)
    query_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply reranking to merged documents using the potentially rewritten query.

    Args:
        intermediate_docs: List of merged dicts including 'content', 'file_path', 'document', 'retrieval_score', 'is_sticky'.
        query: The query to rerank against (should be the rewritten query).
        use_reranker: Whether to use reranking.
        reranker_top_k: Number of documents to keep after reranking.
        retrieval_top_k: Number of documents *before* reranking (informational).
        query_id: Unique query identifier.

    Returns:
        Tuple of:
        - docs_for_context: List[Dict] containing necessary fields for LLM prompt.
        - docs_for_ranking: List[Dict] containing full info including scores.
    """
    docs_for_context = []
    docs_for_ranking = [] # This will be the final list after reranking/selection

    if not intermediate_docs:
        logger.warning(f"Query ID {query_id}: No documents provided for reranking.")
        return [], []

    # If not using reranker, select based on retrieval score (if available) or just take top N
    if not use_reranker:
        logger.info(f"Query ID {query_id}: Skipping reranking. Selecting top {reranker_top_k} from {len(intermediate_docs)} merged documents.")
        # Sort by retrieval score (descending), putting None scores last
        intermediate_docs.sort(key=lambda x: x.get('retrieval_score') if x.get('retrieval_score') is not None else -float('inf'), reverse=True)
        docs_for_ranking = intermediate_docs[:reranker_top_k]
        # Ensure 'rerank_score' is None if not calculated
        for doc in docs_for_ranking:
            doc['rerank_score'] = None

    # If using reranker
    else:
        logger.info(f"Query ID {query_id}: Reranking {len(intermediate_docs)} documents against query: '{query[:100]}...', keeping top {reranker_top_k}.")
        try:
            # Ensure necessary keys exist
            docs_to_rerank = [
                doc for doc in intermediate_docs
                if doc.get('file_path') and doc.get('content') # Ensure basic fields are present
            ]
            if not docs_to_rerank:
                 logger.warning(f"Query ID {query_id}: No valid documents found to pass to the reranker after filtering.")
                 raise ValueError("No valid documents to rerank.")

            # Adapt input format for your reranker if needed
            # Assuming reranker takes list of dicts with 'id' and 'content' keys
            reranker_input = [
                {"id": doc.get('id', doc['file_path']), "content": doc['content']} # Use Qdrant ID if present, else file_path
                for doc in docs_to_rerank
            ]

            reranker = SentenceTransformerReranker() # Assuming default model or configured elsewhere
            reranked_output = reranker.rerank(
                query=query,
                documents=reranker_input,
                content_key="content",
                top_k=reranker_top_k # Ask reranker to return only top K
            )
            logger.info(f"Query ID {query_id}: Reranker returned {len(reranked_output)} documents.")

            # Map reranked scores back to the original intermediate_docs structure
            reranked_scores_map = {
                item.get('id'): item.get('rerank_score') # Assuming 'rerank_score' is the key added by your reranker
                for item in reranked_output if item.get('id') is not None and item.get('rerank_score') is not None
            }

            temp_ranked_docs = []
            for doc_dict in intermediate_docs:
                # Use the same ID logic (Qdrant ID or file_path) to match
                doc_id = doc_dict.get('id', doc_dict.get('file_path'))
                if doc_id in reranked_scores_map:
                    doc_dict['rerank_score'] = float(reranked_scores_map[doc_id])
                    temp_ranked_docs.append(doc_dict)
                # else: # Optionally keep docs that weren't reranked? Typically discard.
                #    doc_dict['rerank_score'] = None
                #    temp_ranked_docs.append(doc_dict)

            # Sort the documents that received a rerank score
            docs_for_ranking = sorted(
                temp_ranked_docs,
                key=lambda x: x.get('rerank_score', -float('inf')), # Handle potential None during sorting
                reverse=True
            )
            # Ensure we don't exceed reranker_top_k (though reranker.rerank might already handle this)
            docs_for_ranking = docs_for_ranking[:reranker_top_k]

        except Exception as e:
            logger.error(f"Reranking failed for query_id {query_id}: {e}", exc_info=True)
            logger.warning(f"Query ID {query_id}: Falling back to top {reranker_top_k} documents based on initial retrieval score.")
            # Fallback: Sort by original retrieval score and take top K
            intermediate_docs.sort(key=lambda x: x.get('retrieval_score') if x.get('retrieval_score') is not None else -float('inf'), reverse=True)
            docs_for_ranking = intermediate_docs[:reranker_top_k]
            for doc in docs_for_ranking:
                doc['rerank_score'] = None # Indicate reranking failed/skipped

    # Prepare the list for the LLM context (content and path needed for prompt)
    docs_for_context = [
        # Ensure the 'document' object is included if needed by generate_response or serialization
        {"file_path": d['file_path'], "content": d['content'], "document": d.get('document')}
        for d in docs_for_ranking
    ]

    logger.info(f"Query ID {query_id}: Final number of documents selected for context: {len(docs_for_context)}")
    # Log details of top selected docs
    for i, d in enumerate(docs_for_ranking[:5]):
         logger.debug(f"  Top Doc {i+1}: Path='{d.get('file_path')}', RetrScore={d.get('retrieval_score'):.4f}, RerankScore={d.get('rerank_score'):.4f if d.get('rerank_score') is not None else 'N/A'}, Sticky={d.get('is_sticky')}")

    return docs_for_context, docs_for_ranking


def generate_response(
    message: str, # Original user message
    docs_for_context: List[Dict],
    chat_history: str, # Formatted history string
    model_name: str,
    query_id: str,
    query_features: Dict[str, bool], # Mutable dict to update error status
    rewritten_query: Optional[str] = None # Optional rewritten query
) -> str:
    """
    Generate response from LLM using retrieved context and chat history.
    Uses the *original* user message as the primary question for the LLM.
    """
    if docs_for_context:
        # Format context for the prompt (ensure sensitive info like IDs isn't leaked if necessary)
        context_string_for_prompt = "\n\n".join(
            # Use a helper to format path if needed, otherwise use file_path directly
            f"Document Path: {doc['file_path']}\nContent:\n{doc['content'][:1000]}..." # Limit content length per doc if needed
            for doc in docs_for_context
        )
    else:
        context_string_for_prompt = "Geen relevante documenten gevonden."

    # Use the updated create_prompt function
    prompt = create_prompt(
        user_query=message, # Use the original query here
        context=context_string_for_prompt,
        chat_history=chat_history,
        rewritten_query=rewritten_query # Pass rewritten query (optional, handled by create_prompt)
        )

    logger.debug(f"Query ID {query_id}: Final prompt for LLM (length {len(prompt)}):\n{prompt[:500]}...\n...\n...{prompt[-500:]}")

    try:
        logger.info(f"Query ID {query_id}: Generating final response using model: {model_name}")
        model_handler = get_model_handler(model_name)
        response = model_handler.generate_text(prompt)
        logger.info(f"Query ID {query_id}: LLM response generated successfully (length {len(response)}).")
        query_features["final_llm_error"] = False
        return response
    except Exception as e:
        query_features["final_llm_error"] = True # Update the dict
        logger.exception(f"LLM generation failed for query_id {query_id} using model {model_name}")
        return ERROR_RESPONSE_MESSAGE # Return error message, status is tracked in query_features

def serialize_documents_for_response(docs_for_ranking: List[Dict]) -> List[Dict]:
    """
    Converts final documents (after reranking) to JSON-serializable format for API response.
    Includes Qdrant ID, scores, and stickiness.
    """
    serialized_docs = []
    for result in docs_for_ranking:
        doc_obj = result.get('document')
        # Fallback if 'document' object is missing for some reason
        file_path = result.get('file_path', 'N/A')
        content = result.get('content', '')

        serialized_doc = {
            'qdrant_id': result.get('id'), # Include Qdrant ID if available
            'file_path': file_path,
            'content': content, # Consider truncating for API response if large
            'retrieval_score': result.get('retrieval_score'),
            'rerank_score': result.get('rerank_score'),
            'is_sticky': result.get('is_sticky', False) # Include sticky status
        }

        # Add details from the Document object if present
        if isinstance(doc_obj, Document):
            serialized_doc.update({
                'db_id': doc_obj.id,
                'doc_tag': doc_obj.doc_tag,
                'original_url': doc_obj.original_url,
                'chunk_url': doc_obj.chunk_url,
                'inserted_at': doc_obj.inserted_at.isoformat() if doc_obj.inserted_at else None,
                'updated_at': doc_obj.updated_at.isoformat() if doc_obj.updated_at else None,
            })
        else:
             logger.warning(f"Document object missing for file_path '{file_path}' during serialization.")

        serialized_docs.append(serialized_doc)

    # logger.debug(f"Serialized documents for API response: {json.dumps(serialized_docs, indent=2)}") # Can be very verbose
    return serialized_docs

def prepare_doc_details_for_logging(serialized_docs: List[Dict]) -> List[Dict]:
    """
    Prepare document details specifically for logging (less verbose).
    """
    log_details = []
    for doc in serialized_docs:
        log_details.append({
            "path": doc.get('file_path', 'N/A'),
            "q_id": doc.get('qdrant_id', 'N/A'),
            "tag": doc.get('doc_tag', 'N/A'),
            "ret_sco": f"{doc.get('retrieval_score'):.4f}" if doc.get('retrieval_score') is not None else None,
            "rr_sco": f"{doc.get('rerank_score'):.4f}" if doc.get('rerank_score') is not None else None,
            "sticky": doc.get('is_sticky', False)
        })
    return log_details

def log_request_parameters(query_id, message, use_reranker, use_hyde, use_augmentation,
                          fresh_k, sticky_s, reranker_top_k, model_name): # Added K/S
    """Log initial request parameters."""
    # Deprecated hyde/augment
    # logger.info(
    #     f"Query ID {query_id}: Reranker={use_reranker}, HyDE={use_hyde}, "
    #     f"Augmentation={use_augmentation}, RetrievalK={retrieval_top_k}, "
    #     f"RerankerK={reranker_top_k}, Model={model_name}"
    # )
    logger.info(
        f"Query ID {query_id}: Params: Reranker={use_reranker}, FreshK={fresh_k}, "
        f"StickyS={sticky_s}, RerankerK={reranker_top_k}, Model={model_name}"
    )

# Deprecated: process_query_enhancements - CQR handled separately
# def process_query_enhancements(...) -> ...:

def retrieve_vector_results(kb_id: str, query_embedding: List[float], top_k: int, query_id: str) -> Tuple[
    List[Dict], List[str]]:
    """
    Retrieve FRESH results from vector store based on similarity search.
    (Renamed slightly for clarity vs direct ID retrieval)
    """
    logger.info(f"Query ID {query_id}: Getting vector client for KB: {kb_id}")
    vector_client = get_vector_client(kb_id)
    logger.info(f"Query ID {query_id}: Retrieving top {top_k} FRESH vectors via similarity search...")

    retrieved_results = [] # List of dicts {'id', 'score', 'payload'}
    retrieved_paths = [] # List of file_paths from payload

    try:
        # This now calls the method specifically for similarity search
        retrieved_results = vector_client.retrieve_vectors(query_embedding, top_k=top_k)
        logger.info(f"Query ID {query_id}: Retrieved {len(retrieved_results)} raw results from vector storage search.")

        # Log top few results retrieved
        if retrieved_results:
            logger.debug(f"Query ID {query_id}: Top {min(5, len(retrieved_results))} FRESH results from Vector Store:")
            for item in retrieved_results[:5]:
                payload = item.get('payload', {})
                path = payload.get('file_path', 'N/A')
                tag = payload.get('doc_tag', 'N/A')
                score = item.get('score', float('nan'))
                qdrant_id = item.get('id', 'N/A')
                logger.debug(f"  - ID: {qdrant_id}, Path: {path}, Tag: {tag}, Score: {score:.4f}")

        # Extract file paths for DB lookup
        retrieved_paths = [
            item['payload']['file_path']
            for item in retrieved_results
            if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']
        ]

        if len(retrieved_paths) != len(retrieved_results):
            logger.warning(
                f"Query ID {query_id}: Mismatch between total retrieved results ({len(retrieved_results)}) "
                f"and results with valid file_paths ({len(retrieved_paths)})."
            )
    except Exception as vector_e:
        logger.error(f"Vector retrieval via search failed for query_id {query_id}: {vector_e}", exc_info=True)
        # Return empty lists on failure
        retrieved_results = []
        retrieved_paths = []


    return retrieved_results, retrieved_paths


def log_response_metrics(
    timestamp: str,
    query_id: str,
    chat_id: str, # Add chat_id
    response: str,
    kb_id: str,
    model_name: str,
    use_reranker: bool,
    reranker_top_k: int,
    query_features: Dict[str, Any], # Pass the whole dict
    # retrieval_top_k removed, use query_features['fresh_k_used'] etc.
    docs_retrieved_count: int, # Total unique docs before rerank
    docs_found_in_db_count: int, # Should match docs_retrieved if logic is sound
    docs_returned_count: int, # Final count after rerank
    log_doc_details: List[Dict]
) -> None:
    """Log detailed metrics about the conversational response generation process."""

    # Flatten query_features for easier logging
    flat_features = {k: v for k, v in query_features.items()}

    response_info = {
        "timestamp": timestamp,
        "query_id": query_id,
        "chat_id": chat_id,
        "response_length": len(response),
        "kb_id": kb_id,
        "model_used": model_name,
        "reranker_used": use_reranker,
        "reranker_top_k": reranker_top_k if use_reranker else None,
        # Conversational features
        **flat_features, # Include all features like cqr_used, sticky_*, similarity, K/S used
        # Document counts
        "docs_retrieved_merged_count": docs_retrieved_count,
        "docs_found_in_db_count": docs_found_in_db_count, # Keep for sanity check
        "docs_returned_final_count": docs_returned_count,
        "docs_returned_details": log_doc_details,
        # Error flags already in query_features
        # "llm_generation_error": query_features["final_llm_error"], # Redundant
        # "unexpected_error": query_features.get("unexpected_error") # Include if present
    }
    # Remove None values for cleaner logs
    response_info = {k: v for k, v in response_info.items() if v is not None}


    log_message_content = f"CONV_RESPONSE_METRICS {json.dumps(response_info)}"
    if query_features.get("final_llm_error") or query_features.get("unexpected_error"):
        logger.error(log_message_content)
    else:
        logger.info(log_message_content)

def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()
