# chat_handler.py
import os
import json
import logging
import datetime
import sys
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid # <--- Fix from previous step (Import the module)
from uuid import uuid4
from django.db import transaction
import numpy as np # For cosine similarity calculation
from sklearn.metrics.pairwise import cosine_similarity # Alternative/robust cosine similarity

# --- Django Model Imports ---
try:
    # Assumes models are in an app named 'logius' or adjust as needed
    from logius.models import Document, ChatQuery
except ImportError:
    # Define dummy classes if Django apps aren't loaded or models unavailable
    # This helps with basic script parsing but won't work at runtime without models
    logger = logging.getLogger(__name__)
    logger.warning("Could not import Django models. Using dummy classes.")
    class Document: pass
    class ChatQuery: pass

# --- Local Project Imports ---
# Adjust paths based on your project structure
from database.db_retriever import retrieve_chunks_from_db
from model.model_factory import get_model_handler # For LLM calls (CQR, Answer)
from embedding.embedding_factory import get_embedding_handler # For generating embeddings
from vectorstorage.vector_factory import (
    get_vector_client, # For interacting with Qdrant/vector store
    get_embedding_model_for_kb, # To know which embedding model to use
    get_available_knowledge_bases # Potentially for listing options
)
# Assumes redis_handler provides history formatting
# Adjust import path as needed
from memory.redis_handler import log_message, format_chat_history_for_prompt
from reranking.reranker import SentenceTransformerReranker # For reranking results
# Import prompt generation functions
from prompt.prompt_builder import create_prompt, HyDE, AugmentQuery, create_cqr_prompt


# --- Logging Configuration ---
# Configure root logger or specific logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chat_handler') # Specific logger for this module
# Example: Set higher level for noisy libraries
# logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


# --- Constants for Conversational RAG ---
DEFAULT_FRESH_K = 6       # K: Default number of fresh documents to retrieve per turn
DEFAULT_STICKY_S = 4      # S: Default number of sticky documents to keep/retrieve per turn
MAX_TOTAL_DOCS = 10     # Target total documents after merging and reranking (approx K+S)
SIMILARITY_THRESHOLD = 0.30 # Cosine similarity threshold to flush sticky memory (lower means less related)
HISTORY_TURNS_FOR_CQR = 1 # Number of previous Q-A pairs for CQR context (adjust based on CQR model needs)
HISTORY_TURNS_FOR_PROMPT = 3 # Max number of turns for final answer generation prompt context
# SUMMARY_MAX_TOKENS = 100 # Placeholder if using summary instead of raw history turns

# --- Other Constants ---
DEFAULT_RERANKER_TOP_K = MAX_TOTAL_DOCS # Rerank and keep up to this many documents
ERROR_RESPONSE_MESSAGE = "Sorry, er is een fout opgetreden bij het verwerken van uw verzoek. Probeer het later opnieuw."


# --- Internal Helper Functions ---

def _get_previous_turn_data(chat_id: str) -> Optional[ChatQuery]:
    """
    Fetches the most recent ChatQuery object (previous turn state) for a given chat_id.
    Returns None if no previous turn exists or an error occurs.
    """
    if not chat_id:
        return None
    try:
        # Fetch the latest completed query for this chat session using Django ORM
        previous_turn = ChatQuery.objects.filter(chat_id=chat_id).order_by('-created_at').first()
        if previous_turn:
            logger.debug(f"Found previous turn data for chat_id {chat_id} (Query ID: {previous_turn.id})")
            return previous_turn
        else:
            logger.debug(f"No previous turns found in DB for chat_id {chat_id}.")
            return None
    except Exception as e:
        # Log error but allow flow to continue (treat as first turn)
        logger.error(f"Error retrieving previous turn data for chat_id {chat_id}: {e}", exc_info=True)
        return None

def _rewrite_query_cqr(
    user_query: str,
    chat_history_cqr: str,
    model_name: str,
    query_id: str # For logging context
) -> str:
    """
    Generates a standalone query using Contextual Query Rewriting (CQR).
    Uses the specified LLM model. Falls back to the original query on failure.

    Args:
        user_query: The latest user message.
        chat_history_cqr: Formatted string of recent chat history for context.
        model_name: Identifier for the LLM to use for CQR.
        query_id: Unique ID of the current request for logging.

    Returns:
        The rewritten, standalone query, or the original query if CQR fails or produces invalid output.
    """
    logger.info(f"Query ID {query_id}: Attempting CQR for query: '{user_query[:100]}...'")
    try:
        # 1. Create the CQR prompt
        cqr_prompt = create_cqr_prompt(user_query, chat_history_cqr)
        logger.debug(f"Query ID {query_id}: CQR prompt created (length {len(cqr_prompt)}).")

        # 2. Get the LLM handler
        # Using the same model factory as for answer generation, but could be different
        cqr_llm_handler = get_model_handler(model_name)

        # 3. Generate the rewritten query
        rewritten_query = cqr_llm_handler.generate_text(cqr_prompt).strip()
        print("REWRITTEN QUERY ::::", rewritten_query)
        logger.debug(f"Query ID {query_id}: Raw CQR output: '{rewritten_query}'")

        # 4. Basic validation of the output
        if not rewritten_query or len(rewritten_query) < 5: # Check if empty or very short
            logger.warning(f"Query ID {query_id}: CQR produced short/empty output. Falling back to original query.")
            return user_query
        elif rewritten_query.lower() == user_query.lower():
             logger.info(f"Query ID {query_id}: CQR returned the original query (or functionally identical).")
             # Return original to avoid storing redundant data if needed elsewhere
             return user_query
        else:
             logger.info(f"Query ID {query_id}: Query successfully rewritten via CQR.")
             logger.info(f"  Original:  '{user_query}'")
             logger.info(f"  Rewritten: '{rewritten_query}'")
             return rewritten_query

    except Exception as e:
        logger.error(f"Query ID {query_id}: CQR generation failed: {e}. Falling back to original query.", exc_info=True)
        return user_query # Fallback on any error

def _calculate_cosine_similarity(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
    """
    Calculates the cosine similarity between two vectors.
    Handles potential None inputs or dimension mismatches gracefully.

    Returns:
        Cosine similarity as a float between -1.0 and 1.0 (or 0.0 if calculation is not possible).
    """
    if vec1 is None or vec2 is None:
        logger.debug("Cannot calculate cosine similarity: one or both vectors are None.")
        return 0.0
    if len(vec1) != len(vec2):
        logger.warning(f"Cannot calculate cosine similarity: vector dimensions mismatch ({len(vec1)} vs {len(vec2)}).")
        return 0.0
    if not vec1 or not vec2: # Handle empty lists
         logger.debug("Cannot calculate cosine similarity: one or both vectors are empty.")
         return 0.0

    try:
        # Using sklearn's implementation is generally robust
        vec1_np = np.array(vec1).reshape(1, -1)
        vec2_np = np.array(vec2).reshape(1, -1)
        similarity = cosine_similarity(vec1_np, vec2_np)[0][0]
        # Clamp result just in case of floating point issues
        return float(max(-1.0, min(1.0, similarity)))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}", exc_info=True)
        return 0.0 # Return neutral similarity on calculation error

def _retrieve_sticky_results(
    sticky_ids: List[Union[str, int, uuid.UUID]], # Use uuid.UUID here
    kb_id: str, # Knowledge base ID to target the correct vector store/collection
    query_id: str # For logging context
) -> List[Dict]:
    """
    Retrieves full document data for sticky documents using their stored IDs.
    Fetches payload from vector store (Qdrant) and full content from database (PostgreSQL).

    Args:
        sticky_ids: List of Qdrant point IDs (or file_paths if used as IDs) from the previous turn.
        kb_id: Identifier for the knowledge base / Qdrant collection.
        query_id: Unique ID of the current request for logging.

    Returns:
        A list of dictionaries, each representing a retrieved sticky document,
        formatted similarly to fresh results but marked as sticky.
        Example dict: {'id': ..., 'file_path': ..., 'content': ..., 'document': ..., 'is_sticky': True, ...}
    """
    if not sticky_ids:
        return []

    logger.info(f"Query ID {query_id}: Retrieving {len(sticky_ids)} sticky documents by ID: {sticky_ids}")
    final_sticky_docs = []

    try:
        # 1. Retrieve payloads from Vector Store (Qdrant) using IDs
        vector_client = get_vector_client(kb_id)
        # Assuming vector_client has retrieve_by_ids implemented as shown previously
        sticky_results_qdrant = vector_client.retrieve_by_ids(sticky_ids)
        # Result format: [{'id': point_id, 'payload': {...}, 'score': None}, ...]

        if not sticky_results_qdrant:
             logger.warning(f"Query ID {query_id}: Vector store returned no results for sticky IDs: {sticky_ids}")
             return []

        # Create a map of Qdrant ID -> Qdrant result for easier lookup
        qdrant_results_map = {item['id']: item for item in sticky_results_qdrant}

        # 2. Extract file_paths needed for DB lookup
        sticky_paths_to_fetch = [
            item['payload']['file_path']
            for item in sticky_results_qdrant
            if item.get('payload') and item['payload'].get('file_path')
        ]

        if not sticky_paths_to_fetch:
            logger.warning(f"Query ID {query_id}: No valid 'file_path' found in payloads of retrieved sticky documents from Qdrant.")
            return []

        # Remove duplicates before hitting the DB
        unique_sticky_paths = sorted(list(set(sticky_paths_to_fetch)))
        logger.debug(f"Query ID {query_id}: Unique sticky file paths to fetch from DB: {unique_sticky_paths}")


        # 3. Retrieve full Document objects from Database (PostgreSQL)
        sticky_document_objects = retrieve_chunks_from_db(unique_sticky_paths)
        # Create a map of file_path -> Document object for efficient merging
        db_doc_map = {doc.file_path: doc for doc in sticky_document_objects}
        logger.debug(f"Query ID {query_id}: Retrieved {len(db_doc_map)} documents from DB for sticky paths.")

        # 4. Combine Qdrant info (ID) with DB info (content, object)
        processed_qdrant_ids = set()
        for qdrant_id, qdrant_item in qdrant_results_map.items():
            # Avoid processing the same Qdrant ID multiple times if `sticky_ids` had duplicates
            if qdrant_id in processed_qdrant_ids: continue

            file_path = qdrant_item.get('payload', {}).get('file_path')
            if file_path and file_path in db_doc_map:
                doc_obj = db_doc_map[file_path]
                final_sticky_docs.append({
                    "id": qdrant_id, # The ID from Qdrant (used for next turn's sticky list)
                    "file_path": doc_obj.file_path,
                    "content": doc_obj.content,
                    "document": doc_obj, # The full Django model instance
                    "retrieval_score": None, # No similarity score for ID-based retrieval
                    "rerank_score": None, # Will be populated by reranker if used
                    "is_sticky": True # Mark this document as originating from sticky memory
                })
                processed_qdrant_ids.add(qdrant_id)
            else:
                logger.warning(f"Query ID {query_id}: Could not find matching DB document for sticky Qdrant result (ID: {qdrant_id}, Path: {file_path}). Skipping.")

        logger.info(f"Query ID {query_id}: Successfully processed {len(final_sticky_docs)} sticky documents by combining vector store and DB data.")
        return final_sticky_docs

    except Exception as e:
        logger.error(f"Query ID {query_id}: Failed during sticky document retrieval or processing: {e}", exc_info=True)
        return [] # Return empty list on error

def _merge_and_deduplicate_results(
    fresh_results: List[Dict],
    sticky_results: List[Dict]
    # max_total: int # Max total is handled by reranker top_k later
) -> List[Dict]:
    """
    Merges fresh and sticky document results, removing duplicates based on 'file_path'.
    If a document is both fresh and sticky, it retains its fresh retrieval score
    but is marked as sticky.

    Args:
        fresh_results: List of dictionaries for freshly retrieved documents.
        sticky_results: List of dictionaries for sticky documents retrieved by ID.

    Returns:
        A single list of unique document dictionaries.
    """
    # Use file_path as the key for deduplication
    merged_dict: Dict[str, Dict] = {}

    # Add sticky results first, marking them
    for doc in sticky_results:
        if doc.get('file_path'):
            doc['is_sticky'] = True # Ensure sticky flag is set
            merged_dict[doc['file_path']] = doc
        else:
             logger.warning(f"Sticky document missing file_path, cannot merge: {doc.get('id')}")


    # Add fresh results, overwriting non-score fields if duplicate, merging scores/flags
    for doc in fresh_results:
        file_path = doc.get('file_path')
        if not file_path:
            logger.warning(f"Fresh document missing file_path, cannot merge: {doc.get('id')}")
            continue

        if file_path in merged_dict:
            # Document was also sticky. Keep the sticky flag, update with fresh score.
            merged_dict[file_path]['retrieval_score'] = doc.get('retrieval_score')
            # Keep the Qdrant ID from the sticky result if the fresh one didn't have it, or reconcile if needed
            if 'id' not in merged_dict[file_path] and 'id' in doc:
                 merged_dict[file_path]['id'] = doc['id']
            # Ensure content and document object are consistent (usually fresh is fine)
            merged_dict[file_path]['content'] = doc.get('content')
            merged_dict[file_path]['document'] = doc.get('document')
            # Mark explicitly it was found in both
            merged_dict[file_path]['found_in_fresh'] = True

        else:
            # New document only found in fresh search
            doc['is_sticky'] = False # Mark as not sticky
            doc['found_in_fresh'] = True
            merged_dict[file_path] = doc

    # Convert back to list
    merged_list = list(merged_dict.values())

    logger.info(f"Merged {len(fresh_results)} fresh and {len(sticky_results)} sticky results into {len(merged_list)} unique documents.")

    # Optional: Sort before reranking? Reranker handles final ordering.
    # merged_list.sort(key=lambda x: x.get('retrieval_score', -float('inf')), reverse=True)

    return merged_list


# --- Main Processing Function ---

# Wrap in transaction.atomic if database operations within need to be rolled back together on error
@transaction.atomic
def process_user_message(
    message: str,
    chat_id: Optional[str] = None,
    model_name: str = 'gemini',
    kb_id: str = 'qdrant-logius',
    use_reranker: bool = True,
    reranker_top_k: int = DEFAULT_RERANKER_TOP_K
    # Removed deprecated args: retrieval_top_k, use_hyde, use_augmentation
) -> Dict[str, Any]:
    """
    Processes a user message within a conversational RAG pipeline.

    Handles query rewriting (CQR), adaptive sticky memory, hybrid retrieval (fresh+sticky),
    reranking, answer generation, state persistence, and logging.

    Args:
        message: The user's input message for the current turn.
        chat_id: Optional identifier for the conversation session. If None, a new session is started.
        model_name: Identifier for the LLM to be used (e.g., 'gemini', 'openai').
        kb_id: Identifier for the knowledge base (vector store collection) to query.
        use_reranker: Boolean flag to enable the reranking step.
        reranker_top_k: The number of documents to select after the reranking step.

    Returns:
        A dictionary containing the response and metadata:
        {
            "response": str, # The generated answer
            "docs": List[Dict], # List of source documents used, with scores and metadata
            "query_id": str, # Unique ID for this specific request/turn
            "chat_id": str, # Identifier for the ongoing conversation session
            "error": Optional[str] # Error message if processing failed
        }
    """
    start_time = datetime.datetime.now()
    timestamp = start_time.isoformat()
    query_id = str(uuid4()) # Unique ID for this specific turn/request

    # Ensure a chat_id exists; generate if not provided (starts a new session)
    is_new_session = False
    if not chat_id:
        chat_id = str(uuid4())
        is_new_session = True
        logger.info(f"Query ID {query_id}: No chat_id provided. Started new session: {chat_id}")
    else:
        logger.info(f"Query ID {query_id}: Processing message for existing session: {chat_id}")

    # Initialize response variables and state trackers
    response = ERROR_RESPONSE_MESSAGE # Default error message
    final_docs_to_return_serialized: List[Dict] = []
    rewritten_query: str = message # Default: use original query if no CQR happens
    rewritten_query_embedding: Optional[List[float]] = None
    final_doc_ids_for_next_turn: List[Union[str, int, uuid.UUID]] = [] # Store Qdrant IDs or file_paths

    # Dictionary to track features/metrics for this turn
    query_features = {
        "query_id": query_id,
        "chat_id": chat_id,
        "is_follow_up": not is_new_session,
        "cqr_used": False,
        "sticky_memory_used": False,
        "sticky_memory_flushed": False,
        "sticky_ids_count_in": 0,
        "cosine_similarity": None,
        "fresh_k_retrieved": 0,
        "sticky_s_retrieved": 0,
        "merged_docs_count": 0,
        "reranker_used": use_reranker,
        "final_docs_count": 0,
        "final_llm_error": False,
        "processing_error": None, # To store unexpected error messages
        "processing_time_ms": 0
    }

    try:
        logger.info(f"--- Turn Start --- Query ID: {query_id}, Chat ID: {chat_id}, Original Message: '{message[:100]}...' ---")
        # Log parameters actually used by this function
        log_request_parameters(query_id, message, use_reranker, False, False, # Explicitly False for deprecated Hyde/Augment
                              DEFAULT_FRESH_K, DEFAULT_STICKY_S, reranker_top_k, model_name)

        # --- Step 1: Load Previous Turn State ---
        previous_turn: Optional[ChatQuery] = None
        previous_sticky_ids: List[Union[str, int, uuid.UUID]] = []
        previous_query_embedding: Optional[List[float]] = None

        if not is_new_session:
            previous_turn = _get_previous_turn_data(chat_id)
            if previous_turn:
                query_features["is_follow_up"] = True # Confirm it's a follow-up with state
                previous_sticky_ids = previous_turn.final_doc_ids or [] # Load IDs saved from last turn
                # Load embedding, handle potential JSON storage
                raw_embedding = previous_turn.rewritten_query_embedding
                if isinstance(raw_embedding, str): # Simple check for JSON string
                    try: previous_query_embedding = json.loads(raw_embedding)
                    except json.JSONDecodeError: previous_query_embedding = None
                elif isinstance(raw_embedding, list):
                    previous_query_embedding = raw_embedding
                else: previous_query_embedding = None

                query_features["sticky_ids_count_in"] = len(previous_sticky_ids)
                logger.info(f"Query ID {query_id}: Follow-up turn. Loaded {len(previous_sticky_ids)} sticky IDs and previous embedding (exists: {previous_query_embedding is not None}) from Query ID {previous_turn.id}.")
            else:
                # No previous turn found in DB, treat as first turn of this session
                logger.info(f"Query ID {query_id}: Follow-up chat_id provided, but no previous state found in DB. Treating as first turn.")
                query_features["is_follow_up"] = False # Override based on state found

        # --- Step 2: Query Rewriting (CQR) ---
        # CQR only makes sense if there's history/previous context
        embedding_handler = get_embedding_handler(get_embedding_model_for_kb(kb_id)) # Needed for embeddings later

        if query_features["is_follow_up"]:
            # Get history formatted for CQR prompt (e.g., last Q+A pair)
            # Using Redis handler: format_chat_history_for_prompt(chat_id, max_turns=HISTORY_TURNS_FOR_CQR)
            # Or directly from previous_turn object if only last turn needed:
            if previous_turn and previous_turn.user_query and previous_turn.llm_response:
                 chat_history_for_cqr = f"Vorige Vraag: {previous_turn.user_query}\nVorige Antwoord: {previous_turn.llm_response[:300]}..." # Example format
            else:
                 chat_history_for_cqr = format_chat_history_for_prompt(chat_id, max_turns=HISTORY_TURNS_FOR_CQR) # Fallback to Redis

            rewritten_query = _rewrite_query_cqr(message, chat_history_for_cqr, model_name, query_id)
            query_features["cqr_used"] = (rewritten_query != message) # Mark if CQR changed the query
        else:
            # First turn, the original message *is* the query for retrieval
            rewritten_query = message
            logger.info(f"Query ID {query_id}: First turn, using original message as retrieval query.")

        # --- Step 3: Embed the Query for Retrieval ---
        # Embed the potentially rewritten query
        try:
            # Use the handler associated with the knowledge base
            rewritten_query_embedding = embedding_handler.get_embedding(rewritten_query)
            if rewritten_query_embedding is None: raise ValueError("Embedding generation returned None")
            logger.info(f"Query ID {query_id}: Generated embedding (dim: {len(rewritten_query_embedding)}) for query: '{rewritten_query[:100]}...'")
        except Exception as embed_e:
            logger.error(f"Query ID {query_id}: CRITICAL - Failed to generate embedding for the query: {embed_e}", exc_info=True)
            # This is a fatal error for RAG, cannot proceed with retrieval.
            query_features["processing_error"] = "Embedding generation failed"
            # Skip to final steps, returning error
            raise embed_e # Re-raise to be caught by the main try-except block

        # --- Step 4: Adaptive Sticky Memory Decision ---
        current_fresh_k = DEFAULT_FRESH_K
        current_sticky_s_target = DEFAULT_STICKY_S # How many sticky docs we AIM to retrieve
        use_sticky = False # Default to not using sticky memory

        if query_features["is_follow_up"] and previous_sticky_ids and previous_query_embedding:
            # Calculate similarity between current rewritten query and previous rewritten query
            similarity = _calculate_cosine_similarity(rewritten_query_embedding, previous_query_embedding)
            query_features["cosine_similarity"] = round(similarity, 4)
            logger.info(f"Query ID {query_id}: Cosine similarity with previous turn's query embedding: {similarity:.4f}")

            if similarity < SIMILARITY_THRESHOLD:
                # Topic changed significantly - Flush sticky memory
                logger.info(f"Query ID {query_id}: Similarity ({similarity:.4f}) < Threshold ({SIMILARITY_THRESHOLD}). Flushing sticky memory.")
                previous_sticky_ids = [] # Clear the IDs for retrieval step
                query_features["sticky_memory_flushed"] = True
                # Increase fresh K when topic changes drastically
                current_fresh_k = MAX_TOTAL_DOCS # Retrieve more fresh docs
                current_sticky_s_target = 0
            else:
                # Topic is related - Keep sticky memory active
                logger.info(f"Query ID {query_id}: Similarity ({similarity:.4f}) >= Threshold ({SIMILARITY_THRESHOLD}). Activating sticky memory.")
                use_sticky = True
                query_features["sticky_memory_used"] = True
                # Keep default K/S for simplicity, could add adaptive logic here based on similarity level
                current_fresh_k = DEFAULT_FRESH_K
                current_sticky_s_target = DEFAULT_STICKY_S # Aim to retrieve up to S sticky docs
        else:
             # First turn or no usable previous state (no IDs or embedding)
             logger.info(f"Query ID {query_id}: Not using sticky memory (first turn or missing previous state).")
             current_fresh_k = MAX_TOTAL_DOCS # Retrieve more fresh docs for the first query
             current_sticky_s_target = 0

        # --- Step 5: Hybrid Retrieval ---
        # 5a. Retrieve Fresh Documents (Similarity Search)
        logger.info(f"Query ID {query_id}: Retrieving up to {current_fresh_k} fresh documents using similarity search.")
        fresh_results_qdrant, fresh_retrieved_paths = retrieve_vector_results(
            kb_id, rewritten_query_embedding, current_fresh_k, query_id
        )
        # `fresh_results_qdrant` structure: [{'id', 'score', 'payload'}, ...]

        # Fetch full documents from DB for the retrieved paths
        fresh_document_objects = retrieve_chunks_from_db(fresh_retrieved_paths)
        fresh_db_map = {doc.file_path: doc for doc in fresh_document_objects}

        # Combine Qdrant score/ID with DB content/object for fresh results
        fresh_intermediate_docs = []
        processed_fresh_qdrant_ids = set()
        for item in fresh_results_qdrant:
             qdrant_id = item.get('id')
             if qdrant_id in processed_fresh_qdrant_ids: continue # Avoid duplicates from Qdrant results if any

             file_path = item.get('payload', {}).get('file_path')
             if file_path and file_path in fresh_db_map:
                  doc_obj = fresh_db_map[file_path]
                  fresh_intermediate_docs.append({
                      "id": qdrant_id, # Qdrant ID
                      "file_path": doc_obj.file_path,
                      "content": doc_obj.content,
                      "document": doc_obj, # Django object
                      "retrieval_score": item.get('score'),
                      "rerank_score": None,
                      "is_sticky": False # Mark as fresh
                  })
                  processed_fresh_qdrant_ids.add(qdrant_id)
             else:
                 logger.warning(f"Query ID {query_id}: Could not find DB document for fresh Qdrant result (ID: {qdrant_id}, Path: {file_path}). Skipping.")

        query_features["fresh_k_retrieved"] = len(fresh_intermediate_docs)
        logger.info(f"Query ID {query_id}: Processed {len(fresh_intermediate_docs)} fresh documents after DB lookup.")

        # 5b. Retrieve Sticky Documents (ID Lookup)
        sticky_intermediate_docs = []
        if use_sticky and previous_sticky_ids:
            # Retrieve only up to the target number of sticky docs needed
            ids_to_fetch = previous_sticky_ids[:current_sticky_s_target]
            sticky_intermediate_docs = _retrieve_sticky_results(ids_to_fetch, kb_id, query_id)
            query_features["sticky_s_retrieved"] = len(sticky_intermediate_docs)
        else:
            query_features["sticky_s_retrieved"] = 0


        # 5c. Merge and Deduplicate Fresh and Sticky Results
        merged_intermediate_docs = _merge_and_deduplicate_results(
            fresh_intermediate_docs, sticky_intermediate_docs
        )
        query_features["merged_docs_count"] = len(merged_intermediate_docs)


        # --- Step 6: Reranking ---
        # Rerank the merged set using the *rewritten* query for relevance.
        # NOTE: The 'retrieval_top_k' variable used inside apply_reranking call is
        # set to len(merged_intermediate_docs), NOT an input parameter.
        if use_reranker and merged_intermediate_docs:
             logger.info(f"Query ID {query_id}: Reranking {len(merged_intermediate_docs)} merged documents...")
             docs_for_context, docs_for_ranking = apply_reranking(
                 intermediate_docs=merged_intermediate_docs,
                 query=rewritten_query, # Use rewritten query for reranking relevance
                 use_reranker=True, # Explicitly pass True here
                 reranker_top_k=reranker_top_k, # Target number after reranking
                 retrieval_top_k=len(merged_intermediate_docs), # Pass the count *before* reranking
                 query_id=query_id
             )
        elif merged_intermediate_docs:
            # No reranking, just select top N based on initial retrieval score (if available)
             logger.info(f"Query ID {query_id}: Skipping reranking. Selecting top {reranker_top_k} from merged documents based on initial score.")
             # Sort by retrieval score (desc), Nones last, then take top K
             merged_intermediate_docs.sort(key=lambda x: x.get('retrieval_score', -float('inf')), reverse=True)
             docs_for_ranking = merged_intermediate_docs[:reranker_top_k]
             # Prepare context list (content/path/doc obj) from the selected docs
             docs_for_context = [
                {"file_path": d['file_path'], "content": d['content'], "document": d.get('document')}
                for d in docs_for_ranking
             ]
             # Ensure rerank_score is None if reranker wasn't used
             for d in docs_for_ranking: d['rerank_score'] = None
        else:
            # No documents retrieved/merged at all
            logger.warning(f"Query ID {query_id}: No documents available for context generation.")
            docs_for_context = []
            docs_for_ranking = []

        query_features["final_docs_count"] = len(docs_for_ranking)


        # --- Step 7: Answer Generation ---
        # Get formatted chat history buffer for the final prompt context
        chat_history_for_prompt = format_chat_history_for_prompt(chat_id, max_turns=HISTORY_TURNS_FOR_PROMPT)

        response = generate_response(
            message=message, # Use the ORIGINAL user message for the LLM to answer
            docs_for_context=docs_for_context, # The final, reranked context
            chat_history=chat_history_for_prompt, # Formatted history snippet
            model_name=model_name, # LLM for generation
            query_id=query_id,
            query_features=query_features, # Pass mutable dict to update llm_error status
            rewritten_query=rewritten_query # Pass rewritten query (optional, for prompt builder)
        )

        # Check if LLM generation failed (status updated in generate_response)
        if query_features["final_llm_error"]:
             logger.error(f"Query ID {query_id}: LLM generation failed. Response set to error message.")
             # Keep the error message in 'response' variable for now


        # --- Step 8: Prepare Data for Persistence and API Response ---
        # Get the identifiers (Qdrant ID preferred, fallback to file_path) of the final
        # documents (after reranking) to store for the *next* turn's sticky memory.
        final_doc_ids_for_next_turn = [
            d.get('id') or d.get('file_path') # Prioritize Qdrant ID if available
            for d in docs_for_ranking # Use the final list selected for context
            if d.get('id') or d.get('file_path') # Ensure there is *some* identifier
        ][:MAX_TOTAL_DOCS] # Store up to the max target size
        logger.info(f"Query ID {query_id}: Prepared {len(final_doc_ids_for_next_turn)} doc IDs for next turn's sticky memory: {final_doc_ids_for_next_turn}")

        # Serialize the final documents (docs_for_ranking) for the API response payload
        final_docs_to_return_serialized = serialize_documents_for_response(docs_for_ranking)

        # --- Step 9: Persist State to Database ---
        # Store the results of this turn, including state needed for the next turn.
        # @transaction.atomic ensures this save is rolled back if an error occurred earlier in the block.
        doc_tags_used = list(set( # Get unique tags from the final context docs
             doc['document'].doc_tag
             for doc in docs_for_context
             if doc.get('document') and getattr(doc['document'], 'doc_tag', None)
        ))

        # Convert embedding to list for JSONField storage if needed
        embedding_to_save = None
        if rewritten_query_embedding is not None:
             if isinstance(rewritten_query_embedding, np.ndarray):
                 embedding_to_save = rewritten_query_embedding.tolist()
             elif isinstance(rewritten_query_embedding, list):
                  embedding_to_save = rewritten_query_embedding
             # else: logger.warning("Cannot determine type of embedding for saving.")


        new_query_record = ChatQuery(
            id=query_id,
            chat_id=chat_id,
            user_query=message, # Original query
            rewritten_query=rewritten_query if query_features["cqr_used"] else None, # Store only if rewritten
            rewritten_query_embedding=embedding_to_save, # Store the list/None
            llm_response=response if not query_features["final_llm_error"] else None, # Store None if LLM failed
            final_doc_ids=final_doc_ids_for_next_turn, # Save IDs for next turn's sticky memory
            doc_tag=doc_tags_used, # Tags of docs used in *this* response's context
            model_used=model_name,
            # created_at is auto_now_add
        )
        new_query_record.save()
        logger.info(f"Query ID {query_id}: Successfully saved ChatQuery record to database.")


        # --- Step 10: Log Conversation Turn to External Store (Optional) ---
        # Example: Log user message and assistant response to Redis for simple history buffer
        if not query_features["final_llm_error"]:
            log_message(chat_id, "user", message)
            log_message(chat_id, "assistant", response)


        # --- Step 11: Final Processing & Return ---
        end_time = datetime.datetime.now()
        query_features["processing_time_ms"] = round((end_time - start_time).total_seconds() * 1000)

        # Log comprehensive metrics for this turn
        log_doc_details = prepare_doc_details_for_logging(final_docs_to_return_serialized)
        log_response_metrics(
            timestamp=timestamp,
            query_features=query_features, # Pass the whole features dict
            response=response, # Pass the actual response or error message
            kb_id=kb_id,
            model_name=model_name,
            # Docs counts already in query_features
            log_doc_details=log_doc_details
        )

        # Prepare the final API response payload
        response_payload = {
            "response": response if not query_features["final_llm_error"] else ERROR_RESPONSE_MESSAGE, # Return standard error if LLM failed
            "docs": final_docs_to_return_serialized,
            "query_id": query_id,
            "chat_id": chat_id, # Return chat_id so client can continue the conversation
            "error": ERROR_RESPONSE_MESSAGE if query_features["final_llm_error"] else None # Signal error clearly
        }
        logger.info(f"--- Turn End --- Query ID: {query_id}, Duration: {query_features['processing_time_ms']}ms ---")
        return response_payload

    except Exception as e:
        # Catch any unexpected error during the process
        end_time = datetime.datetime.now()
        processing_time = round((end_time - start_time).total_seconds() * 1000)
        error_message = f"An unexpected error occurred: {type(e).__name__} - {e}"
        logger.exception(f"Query ID {query_id}, Chat ID {chat_id}: CRITICAL - {error_message}") # Log full traceback

        # Update features for error logging
        query_features["processing_error"] = error_message
        query_features["final_llm_error"] = True # Mark as failed
        query_features["processing_time_ms"] = processing_time

        # Log error metrics (outside the transaction)
        # We might not have all data points if error happened early
        log_response_metrics(
            timestamp=timestamp,
            query_features=query_features,
            response=ERROR_RESPONSE_MESSAGE, kb_id=kb_id, model_name=model_name,
            log_doc_details=[] # No docs to log in case of early failure
        )

        # Return a standardized error response to the client
        return {
            "response": None,
            "docs": [],
            "query_id": query_id,
            "chat_id": chat_id,
            "error": ERROR_RESPONSE_MESSAGE # Standard user-facing error
            }


# --- Helper Function Updates (Refined versions) ---

def apply_reranking(
    intermediate_docs: List[Dict],
    query: str, # Should be the rewritten query for relevance calculation
    use_reranker: bool, # Should be True if this function is called in the rerank path
    reranker_top_k: int,
    retrieval_top_k: int, # Number of docs *before* reranking (for logging/context)
    query_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Applies reranking to the merged list of documents using a Cross-Encoder model.

    Args:
        intermediate_docs: List of merged document dicts (fresh & sticky).
                           Must contain 'content', 'file_path', and optionally 'id'.
        query: The query (ideally rewritten) to use for calculating relevance scores.
        use_reranker: Boolean indicating if reranking should be performed (should be True here).
        reranker_top_k: The target number of documents to return after reranking.
        retrieval_top_k: The number of documents *input* to reranking (for logging).
        query_id: Unique query identifier for logging.

    Returns:
        A tuple containing:
        - docs_for_context: List[Dict] prepared for the LLM prompt (content, path, doc object).
        - docs_for_ranking: List[Dict] containing the final reranked and selected documents
                            with full metadata including 'rerank_score'.
    """
    if not intermediate_docs:
        logger.warning(f"Query ID {query_id}: No documents provided to apply_reranking function.")
        return [], []

    if not use_reranker:
        # This function assumes reranking is intended. If called without use_reranker=True, log error.
        logger.error(f"Query ID {query_id}: apply_reranking called with use_reranker=False. This indicates a logic error.")
        # Fallback: return top K based on retrieval score as done in the main function's non-rerank path
        intermediate_docs.sort(key=lambda x: x.get('retrieval_score', -float('inf')), reverse=True)
        docs_for_ranking = intermediate_docs[:reranker_top_k]
        for doc in docs_for_ranking: doc['rerank_score'] = None # Mark as not reranked
    else:
        # Proceed with reranking
        logger.info(f"Query ID {query_id}: Applying reranking to {len(intermediate_docs)} documents against query: '{query[:100]}...', keeping top {reranker_top_k}.")
        try:
            # 1. Prepare documents in the format expected by the reranker
            # Ensure necessary keys are present and valid
            docs_to_rerank_input = []
            original_doc_map = {} # Map reranker input ID back to original dict
            for doc in intermediate_docs:
                content = doc.get('content')
                # Use Qdrant ID if available, otherwise file_path as unique ID for reranker item
                doc_rerank_id = doc.get('id') or doc.get('file_path')
                if content and doc_rerank_id:
                    reranker_item = {"id": doc_rerank_id, "content": content}
                    docs_to_rerank_input.append(reranker_item)
                    original_doc_map[doc_rerank_id] = doc # Store original dict
                else:
                    logger.warning(f"Query ID {query_id}: Skipping doc for reranking due to missing content or identifier: {doc.get('id')}/{doc.get('file_path')}")

            if not docs_to_rerank_input:
                 logger.warning(f"Query ID {query_id}: No valid documents found to pass to the reranker after filtering.")
                 raise ValueError("No valid documents to rerank.")

            # 2. Initialize and run the reranker
            # Assumes SentenceTransformerReranker or similar interface
            reranker = SentenceTransformerReranker() # Consider passing model name if needed
            reranked_output = reranker.rerank(
                query=query,
                documents=docs_to_rerank_input, # Pass the prepared list
                content_key="content", # Key reranker should use for content
                top_k=reranker_top_k # Ask reranker to return only the top K results
            )
            # Expected output format: [{'id': ..., 'rerank_score': ...}, ...] sorted by score

            logger.info(f"Query ID {query_id}: Reranker returned {len(reranked_output)} documents.")

            # 3. Map scores back and reconstruct the final list
            docs_for_ranking = []
            if reranked_output:
                # Create a score map from the reranked output
                reranked_scores_map = {
                    item['id']: item['rerank_score']
                    for item in reranked_output if 'id' in item and 'rerank_score' in item
                }

                # Iterate through the *reranked order* to build the final list
                for item in reranked_output:
                    rerank_id = item['id']
                    if rerank_id in original_doc_map:
                        original_doc_dict = original_doc_map[rerank_id]
                        # Update the original dict with the rerank score
                        original_doc_dict['rerank_score'] = float(reranked_scores_map[rerank_id])
                        docs_for_ranking.append(original_doc_dict)
                    else:
                        # This shouldn't happen if maps are built correctly
                         logger.error(f"Query ID {query_id}: Mismatch - Reranked ID '{rerank_id}' not found in original document map.")

                # Ensure the list doesn't exceed top_k (should be handled by reranker's top_k)
                docs_for_ranking = docs_for_ranking[:reranker_top_k]
            else:
                 logger.warning(f"Query ID {query_id}: Reranker returned an empty list.")
                 docs_for_ranking = []


        except Exception as e:
            logger.error(f"Query ID {query_id}: Reranking process failed: {e}", exc_info=True)
            logger.warning(f"Query ID {query_id}: Falling back to selecting top {reranker_top_k} documents based on initial retrieval score due to reranking error.")
            # Fallback: Sort by original retrieval score and take top K
            intermediate_docs.sort(key=lambda x: x.get('retrieval_score', -float('inf')), reverse=True)
            docs_for_ranking = intermediate_docs[:reranker_top_k]
            # Mark rerank_score as None to indicate failure/fallback
            for doc in docs_for_ranking:
                doc['rerank_score'] = None

    # --- Prepare final outputs ---
    # 1. List for LLM context generation (needs specific fields for the prompt)
    docs_for_context = [
        {
            "file_path": d.get('file_path', 'N/A'),
            "content": d.get('content', ''),
            "document": d.get('document') # Pass the DB object if available and needed downstream
         }
        for d in docs_for_ranking # Use the final selected & ordered list
    ]

    # 2. The ranked list itself (docs_for_ranking) is also returned for persistence/API response

    logger.info(f"Query ID {query_id}: Final number of documents selected after reranking/selection: {len(docs_for_ranking)}")
    # Log details of top N selected docs for debugging
    for i, d in enumerate(docs_for_ranking[:5]): # Log top 5
         rr_score_str = f"{d.get('rerank_score'):.4f}" if d.get('rerank_score') is not None else 'N/A'
         ret_score_str = f"{d.get('retrieval_score'):.4f}" if d.get('retrieval_score') is not None else 'N/A'
         logger.debug(f"  Top Doc {i+1}: Path='{d.get('file_path')}', RerankScore={rr_score_str}, RetrScore={ret_score_str}, Sticky={d.get('is_sticky', False)}")

    return docs_for_context, docs_for_ranking


def generate_response(
    message: str, # Original user message/query for the current turn
    docs_for_context: List[Dict], # Final selected documents for context
    chat_history: str, # Formatted string of recent conversation history
    model_name: str, # Identifier of the LLM to use for generation
    query_id: str, # Unique ID for logging
    query_features: Dict[str, Any], # Mutable dict to update 'final_llm_error' status
    rewritten_query: Optional[str] = None # Optional rewritten query (usually not for LLM)
) -> str:
    """
    Generates the final response using the LLM, based on the original user query,
    provided context documents, and chat history. Updates query_features on error.

    Args:
        message: The original user query for this turn.
        docs_for_context: List of dictionaries containing 'file_path' and 'content' of context docs.
        chat_history: Formatted string of recent chat history.
        model_name: Identifier of the LLM to use.
        query_id: Unique identifier for logging.
        query_features: Mutable dictionary to track execution status, specifically 'final_llm_error'.
        rewritten_query: The rewritten query (optional, mainly for prompt template logic if needed).

    Returns:
        The generated text response from the LLM, or an error message string if generation fails.
    """
    logger.info(f"Query ID {query_id}: Preparing context and prompt for final answer generation using model {model_name}.")

    # 1. Format the context string for the prompt
    if docs_for_context:
        # Join document contents, potentially adding metadata like file path
        # Ensure sensitive info (like internal IDs) isn't leaked if context is logged elsewhere
        context_items = []
        for i, doc in enumerate(docs_for_context):
             # Truncate content per document to avoid excessive prompt length
             truncated_content = doc.get('content', '')[:1500] # Adjust limit as needed
             # Use file_path for citation hint in the prompt
             context_items.append(f"--- Document {i+1} (Path: {doc.get('file_path', 'N/A')}) ---\n{truncated_content}")
        context_string_for_prompt = "\n\n".join(context_items)
    else:
        context_string_for_prompt = "Geen relevante context gevonden in de documentatie."
        logger.warning(f"Query ID {query_id}: No context documents available for final prompt generation.")

    # 2. Create the final prompt using the dedicated prompt builder function
    prompt = create_prompt(
        user_query=message, # IMPORTANT: Use the original query for the LLM to answer
        context=context_string_for_prompt,
        chat_history=chat_history,
        rewritten_query=rewritten_query # Pass along if needed by the prompt template logic
    )

    # Log prompt details carefully (avoid logging full sensitive context if necessary)
    logger.debug(f"Query ID {query_id}: Final prompt length: {len(prompt)} chars.")
    # logger.debug(f"Query ID {query_id}: Prompt Snippet:\n{prompt[:300]}...\n...\n...{prompt[-300:]}")

    # 3. Call the LLM to generate the response
    try:
        logger.info(f"Query ID {query_id}: Sending request to LLM ({model_name})...")
        model_handler = get_model_handler(model_name)
        response_text = model_handler.generate_text(prompt)

        if not response_text or response_text.strip() == "":
             logger.warning(f"Query ID {query_id}: LLM ({model_name}) returned an empty response.")
             # Handle empty response - maybe return a specific message or treat as error
             query_features["final_llm_error"] = True
             return "Sorry, ik kon geen antwoord genereren op basis van de beschikbare informatie."

        logger.info(f"Query ID {query_id}: LLM response generated successfully (length {len(response_text)}).")
        query_features["final_llm_error"] = False # Explicitly mark success
        return response_text.strip() # Return the cleaned response

    except Exception as e:
        # Mark the error in the shared features dict
        query_features["final_llm_error"] = True
        logger.exception(f"Query ID {query_id}: LLM generation failed using model {model_name}: {e}")
        # Return the generic error message to be displayed to the user
        return ERROR_RESPONSE_MESSAGE

def serialize_documents_for_response(docs_for_ranking: List[Dict]) -> List[Dict]:
    """
    Converts the final list of documents (after reranking/selection) into a
    JSON-serializable format suitable for the API response payload.
    Includes key metadata like scores and origin (sticky/fresh).

    Args:
        docs_for_ranking: The final list of selected document dictionaries.

    Returns:
        A list of JSON-serializable dictionaries representing the documents.
    """
    serialized_docs = []
    if not docs_for_ranking:
        return []

    for rank, result in enumerate(docs_for_ranking):
        doc_obj = result.get('document') # Get the Django Document object if present
        file_path = result.get('file_path', 'N/A')
        content_snippet = result.get('content', '')[:200] + '...' if result.get('content') else '' # Example snippet

        serialized_doc = {
            'rank': rank + 1,
            'qdrant_id': result.get('id'), # Qdrant point ID (if available)
            'file_path': file_path,
            'retrieval_score': result.get('retrieval_score'), # Score from initial vector search
            'rerank_score': result.get('rerank_score'), # Score from reranker (if used)
            'is_sticky': result.get('is_sticky', False), # Was this from sticky memory?
            # Include details from the Document object if available
            'db_id': doc_obj.id if isinstance(doc_obj, Document) else None,
            'doc_tag': doc_obj.doc_tag if isinstance(doc_obj, Document) else None,
            'original_url': doc_obj.original_url if isinstance(doc_obj, Document) else None,
            'chunk_url': doc_obj.chunk_url if isinstance(doc_obj, Document) else None,
            'content_snippet': content_snippet, # Add snippet for preview
            # Add timestamps if needed and available on doc_obj
            # 'inserted_at': doc_obj.inserted_at.isoformat() if isinstance(doc_obj, Document) and doc_obj.inserted_at else None,
            # 'updated_at': doc_obj.updated_at.isoformat() if isinstance(doc_obj, Document) and doc_obj.updated_at else None,
        }
        # Clean None values if desired for cleaner API response
        # serialized_doc = {k: v for k, v in serialized_doc.items() if v is not None}
        serialized_docs.append(serialized_doc)

    logger.debug(f"Serialized {len(serialized_docs)} documents for API response.")
    return serialized_docs

def prepare_doc_details_for_logging(serialized_docs: List[Dict]) -> List[Dict]:
    """
    Prepares a compact list of document details suitable for structured logging.

    Args:
        serialized_docs: The list of documents already serialized for the API response.

    Returns:
        A list of dictionaries with concise document information for logging.
    """
    log_details = []
    for doc in serialized_docs:
         # Format scores nicely for logs
         ret_score_str = f"{doc.get('retrieval_score'):.4f}" if doc.get('retrieval_score') is not None else None
         rr_score_str = f"{doc.get('rerank_score'):.4f}" if doc.get('rerank_score') is not None else None
         log_details.append({
             "rank": doc.get('rank'),
             "path": doc.get('file_path', 'N/A')[-60:], # Log truncated path
             "qid": str(doc.get('qdrant_id', 'N/A'))[:8], # Log truncated Qdrant ID
             # "tag": doc.get('doc_tag', 'N/A'), # Uncomment if tag is important for logs
             "ret_sco": ret_score_str,
             "rr_sco": rr_score_str,
             "sticky": doc.get('is_sticky', False)
         })
    return log_details

def log_request_parameters(query_id, message, use_reranker, use_hyde, use_augmentation,
                          fresh_k, sticky_s, reranker_top_k, model_name):
    """Logs the key parameters received for the request."""
    # Log deprecated flags as false for consistency if needed by downstream parsing
    use_hyde = False
    use_augmentation = False
    logger.info(
        f"Query ID {query_id}: Request Params - Reranker={use_reranker}, "
        f"DefaultFreshK={fresh_k}, DefaultStickyS={sticky_s}, RerankerK={reranker_top_k}, "
        f"Model={model_name}"
        # f"HyDE={use_hyde}, Augmentation={use_augmentation}" # Log if needed, but should be false
    )

def retrieve_vector_results(kb_id: str, query_embedding: List[float], top_k: int, query_id: str) -> Tuple[
    List[Dict], List[str]]:
    """
    Retrieves FRESH document results from the vector store via similarity search.

    Args:
        kb_id: Knowledge base/collection identifier.
        query_embedding: The embedding vector of the query.
        top_k: The maximum number of results to retrieve.
        query_id: Unique identifier for logging.

    Returns:
        A tuple containing:
        - List of raw results from vector store client (e.g., [{'id', 'score', 'payload'}, ...]).
        - List of file_path strings extracted from the payloads.
    """
    if top_k <= 0:
        logger.info(f"Query ID {query_id}: Skipping fresh vector retrieval as top_k is {top_k}.")
        return [], []

    logger.info(f"Query ID {query_id}: Accessing vector client for KB: {kb_id}")
    vector_client = get_vector_client(kb_id) # Get the configured client (e.g., QdrantClient)
    logger.info(f"Query ID {query_id}: Performing similarity search for top {top_k} fresh vectors...")

    retrieved_results_raw: List[Dict] = [] # Raw results from client.retrieve_vectors
    retrieved_paths: List[str] = [] # Corresponding file paths

    try:
        # Assuming the client has a method like retrieve_vectors for similarity search
        # This method should return dicts with 'id', 'score', 'payload'
        retrieved_results_raw = vector_client.retrieve_vectors(query_embedding, top_k=top_k)
        count = len(retrieved_results_raw)
        logger.info(f"Query ID {query_id}: Vector store search returned {count} raw results.")

        # Log top few results for debugging
        if retrieved_results_raw:
            logger.debug(f"Query ID {query_id}: Top {min(5, count)} FRESH results from vector search:")
            for item in retrieved_results_raw[:5]:
                payload = item.get('payload', {})
                path = payload.get('file_path', 'N/A')
                # tag = payload.get('doc_tag', 'N/A')
                score = item.get('score', float('nan'))
                qdrant_id = item.get('id', 'N/A')
                logger.debug(f"  - ID: {qdrant_id}, Path: {path}, Score: {score:.4f}")

        # Extract file paths for subsequent DB lookup
        # Ensure payload exists and contains the 'file_path' key
        retrieved_paths = [
            item['payload']['file_path']
            for item in retrieved_results_raw
            if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']
        ]

        # Sanity check: log if counts mismatch significantly
        if len(retrieved_paths) != count:
            logger.warning(
                f"Query ID {query_id}: Mismatch after extracting file paths. "
                f"Vector search returned {count} results, but only {len(retrieved_paths)} had valid file_paths in payload."
            )

    except Exception as vector_e:
        logger.error(f"Query ID {query_id}: Vector retrieval via similarity search failed: {vector_e}", exc_info=True)
        # Ensure empty lists are returned on failure
        retrieved_results_raw = []
        retrieved_paths = []

    return retrieved_results_raw, retrieved_paths


def log_response_metrics(
    timestamp: str,
    query_features: Dict[str, Any], # Contains query_id, chat_id, flags, counts, etc.
    response: str, # The final response string (or error message)
    kb_id: str,
    model_name: str,
    log_doc_details: List[Dict] # Compact list of final doc details
) -> None:
    """Logs detailed metrics about the conversational response generation turn."""

    # Prepare the final log record by combining fixed args and dynamic features
    log_record = {
        "timestamp": timestamp,
        "event_type": "CONV_RESPONSE_METRICS",
        **query_features, # Unpack all collected features/metrics for this turn
        "response_length": len(response) if response else 0,
        "kb_id": kb_id,
        "model_used": model_name,
        "final_docs_details": log_doc_details,
        # Ensure error flags are present and correctly reflect state
        "success": not (query_features.get("final_llm_error") or query_features.get("processing_error"))
    }

    # Remove None values for cleaner logs (optional)
    # log_record = {k: v for k, v in log_record.items() if v is not None}

    # Convert the dictionary to a JSON string for logging
    log_message_content = json.dumps(log_record, default=str) # Use default=str for non-serializable types like UUID

    # Log as ERROR if any failure occurred, INFO otherwise
    if log_record["success"]:
        logger.info(log_message_content)
    else:
        logger.error(log_message_content)


def get_kb_options():
    """Placeholder function to get available knowledge base options."""
    logger.debug("Fetching available knowledge base options...")
    # Replace with actual implementation using vector_factory or config
    try:
        return get_available_knowledge_bases()
    except Exception as e:
        logger.error(f"Failed to get available knowledge bases: {e}")
        return [{"id": "default", "name": "Default KB (Error)"}]