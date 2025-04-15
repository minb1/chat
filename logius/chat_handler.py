# chat_handler.py
import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple

from database.db_retriever import retrieve_chunks_from_db
from model.model_factory import get_model_handler
from embedding.embedding_factory import get_embedding_handler
from vectorstorage.vector_factory import (
    get_vector_client,
    get_embedding_model_for_kb,
    get_available_knowledge_bases
)
from prompt.prompt_builder import create_prompt, HyDE, AugmentQuery
from memory.redis_handler import log_message, format_chat_history_for_prompt
from reranking.reranker import SentenceTransformerReranker

# Configure logging
logger = logging.getLogger('rag_metrics')
logger.setLevel(logging.INFO)

# Constants
DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_RERANKER_TOP_K = 10
ERROR_RESPONSE_MESSAGE = "Sorry, an error occurred while generating the response."


def process_user_message(
        message: str,
        chat_id: Optional[str] = None,
        model_name: str = 'gemini',
        kb_id: str = 'qdrant-logius',
        use_reranker: bool = True,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        reranker_top_k: int = DEFAULT_RERANKER_TOP_K,
        use_hyde: bool = False,
        use_augmentation: bool = False
) -> Dict:
    """
    Process a user message to generate a contextually relevant response using RAG.

    Args:
        message: The user's input message
        chat_id: Optional chat identifier for conversation history
        model_name: The LLM to use for response generation
        kb_id: Knowledge base identifier for vector retrieval
        use_reranker: Whether to rerank retrieved documents
        retrieval_top_k: Number of documents to retrieve from vector store
        reranker_top_k: Number of documents to keep after reranking
        use_hyde: Whether to use Hypothetical Document Embedding
        use_augmentation: Whether to use query augmentation

    Returns:
        Dictionary containing response or error and relevant documents
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = chat_id or f"anonymous_{timestamp}"
    response = ERROR_RESPONSE_MESSAGE
    docs_to_return = []

    # Track feature usage for logging
    query_features = {
        "hypothetical_doc_generated": False,
        "augmented_query_generated": False,
        "final_llm_error": False
    }

    # Enforce mutual exclusivity of query enhancement features
    if use_hyde and use_augmentation:
        logger.warning(f"Query ID {query_id}: Both HyDE and Augmentation enabled. Prioritizing HyDE.")
        use_augmentation = False

    try:
        # Log initial request parameters
        log_request_parameters(query_id, message, use_reranker, use_hyde, use_augmentation,
                               retrieval_top_k, reranker_top_k, model_name)

        # Get embedding handler for the knowledge base
        embedding_handler = get_embedding_handler(get_embedding_model_for_kb(kb_id))

        # Process query with potential enhancements (HyDE or query augmentation)
        query_for_retrieval, query_embedding, query_features = process_query_enhancements(
            message, model_name, embedding_handler, use_hyde, use_augmentation, query_id, query_features
        )

        if query_embedding is None:
            logger.error(f"Embedding generation failed completely for query_id {query_id}.")
            return {"error": "Failed to generate query embedding", "docs": []}

        # Retrieve relevant documents from vector store
        retrieved_results, retrieved_paths = retrieve_vector_results(
            kb_id, query_embedding, retrieval_top_k, query_id
        )

        # Retrieve document content from database
        documents = retrieve_document_content(retrieved_paths, retrieved_results, query_id)

        # Apply reranking if enabled
        docs_for_context, docs_to_return = apply_reranking(
            documents, message, use_reranker, reranker_top_k, retrieval_top_k, query_id
        )

        # Generate response using retrieved context
        response = generate_response(
            message, docs_for_context, chat_id, model_name, query_id, query_features
        )

        # Log conversation in Redis if chat_id is provided
        if chat_id:
            log_message(chat_id, "user", message)
            log_message(chat_id, "assistant", response)

        # Prepare document details for logging
        doc_details = prepare_doc_details(docs_to_return, retrieved_results)

        # Log response metrics
        log_response_metrics(
            timestamp, query_id, response, kb_id, model_name,
            use_hyde, use_augmentation, use_reranker,
            retrieval_top_k, reranker_top_k,
            len(retrieved_results), len(documents), len(docs_to_return),
            doc_details, query_features
        )

        # Return response and documents
        if query_features["final_llm_error"]:
            return {"error": response, "docs": docs_to_return}
        else:
            return {"response": response, "docs": docs_to_return}

    except Exception as e:
        logger.exception(f"Unexpected error for query_id {query_id}")
        return {"error": "An unexpected server error occurred.", "docs": []}


def log_request_parameters(query_id, message, use_reranker, use_hyde, use_augmentation,
                           retrieval_top_k, reranker_top_k, model_name):
    """Log initial request parameters."""
    print(
        f"Processing message: '{message}', Use Reranker: {use_reranker}, "
        f"Use HyDE: {use_hyde}, Use Augmentation: {use_augmentation}, "
        f"Retrieval K: {retrieval_top_k}, Reranker K: {reranker_top_k}, Model: {model_name}"
    )
    logger.info(
        f"Query ID {query_id}: Reranker={use_reranker}, HyDE={use_hyde}, "
        f"Augmentation={use_augmentation}, RetrievalK={retrieval_top_k}, "
        f"RerankerK={reranker_top_k}, Model={model_name}"
    )


def process_query_enhancements(
        message: str,
        model_name: str,
        embedding_handler: Any,
        use_hyde: bool,
        use_augmentation: bool,
        query_id: str,
        query_features: Dict[str, bool]
) -> Tuple[str, List[float], Dict[str, bool]]:
    """
    Process query enhancements (HyDE or augmentation) and return the embedding.

    Args:
        message: Original user message
        model_name: LLM model to use
        embedding_handler: Embedding model handler
        use_hyde: Whether to use HyDE
        use_augmentation: Whether to use query augmentation
        query_id: Unique query identifier
        query_features: Dict to track feature usage

    Returns:
        Tuple of (query_for_retrieval, query_embedding, updated_query_features)
    """
    query_for_retrieval = message  # Default to original message
    query_embedding = None

    if use_hyde:
        print(f"HyDE enabled. Generating hypothetical document for query: '{message}'...")
        try:
            hyde_prompt = HyDE(message)
            hyde_llm_handler = get_model_handler(model_name)
            hypothetical_document = hyde_llm_handler.generate_text(hyde_prompt)
            print(f"Generated Hypothetical Document:\n---\n{hypothetical_document}\n---")
            query_for_retrieval = hypothetical_document
            query_embedding = embedding_handler.get_embedding(query_for_retrieval)
            query_features["hypothetical_doc_generated"] = True
        except Exception as e:
            logger.error(f"HyDE generation/embedding failed for query_id {query_id}: {e}")
            query_embedding = embedding_handler.get_embedding(message)  # Fallback
            print("Generated embedding for original user query (HyDE fallback).")

    elif use_augmentation:
        print(f"Augmentation enabled. Generating augmented query...")
        try:
            augment_prompt = AugmentQuery(message)
            augment_llm_handler = get_model_handler(model_name)
            augmented_query = augment_llm_handler.generate_text(augment_prompt).strip()

            if not augmented_query or len(augmented_query) < 5:
                logger.warning(
                    f"Augmentation resulted in unusable query for query_id {query_id}. "
                    f"Original: '{message}', Augmented: '{augmented_query}'"
                )
                augmented_query = message  # Fallback to original
            else:
                print(f"Generated Augmented Query:\n---\n{augmented_query}\n---")
                query_features["augmented_query_generated"] = True

            query_for_retrieval = augmented_query
            query_embedding = embedding_handler.get_embedding(query_for_retrieval)
        except Exception as e:
            logger.error(f"Query Augmentation failed for query_id {query_id}: {e}")
            query_embedding = embedding_handler.get_embedding(message)  # Fallback
            print("Generated embedding for original user query (Augmentation fallback).")
    else:
        print("Using original query embedding.")
        query_embedding = embedding_handler.get_embedding(message)

    return query_for_retrieval, query_embedding, query_features


def retrieve_vector_results(kb_id: str, query_embedding: List[float], top_k: int, query_id: str) -> Tuple[
    List[Dict], List[str]]:
    """
    Retrieve results from vector store.

    Args:
        kb_id: Knowledge base identifier
        query_embedding: Embedding vector for query
        top_k: Number of results to retrieve
        query_id: Unique query identifier

    Returns:
        Tuple of (retrieved_results, retrieved_paths)
    """
    print(f"Getting vector client for KB: {kb_id}")
    vector_client = get_vector_client(kb_id)
    print(f"Getting top {top_k} related vectors...")

    retrieved_results = vector_client.retrieve_vectors(query_embedding, top_k=top_k)
    print(f"Retrieved {len(retrieved_results)} results from vector storage.")

    # Log top results for debugging
    if retrieved_results:
        print("Top results from Vector Store (Path, Tag, Score):")
        for item in retrieved_results[:5]:  # Print top 5 for brevity
            payload = item.get('payload', {})
            path = payload.get('file_path', 'N/A')
            tag = payload.get('doc_tag', 'N/A')
            score = item.get('score', float('nan'))
            print(f"  - Path: {path}, Tag: {tag}, Score: {score:.4f}")

    # Extract file paths for database retrieval
    retrieved_paths = [
        item['payload']['file_path']
        for item in retrieved_results
        if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']
    ]

    if len(retrieved_paths) != len(retrieved_results):
        logger.warning(
            f"Mismatch between total retrieved results ({len(retrieved_results)}) "
            f"and results with valid file_paths ({len(retrieved_paths)}) for query_id {query_id}."
        )

    return retrieved_results, retrieved_paths


def retrieve_document_content(retrieved_paths: List[str], retrieved_results: List[Dict], query_id: str) -> List[Dict]:
    """
    Retrieve document content from database.

    Args:
        retrieved_paths: List of file paths to retrieve
        retrieved_results: Original results from vector store
        query_id: Unique query identifier

    Returns:
        List of documents with content
    """
    print(f"Retrieving chunk content from PostgreSQL for {len(retrieved_paths)} paths...")
    context_dict_from_db = retrieve_chunks_from_db(retrieved_paths)
    print(f"Retrieved {len(context_dict_from_db)} chunks from DB.")

    # Create documents list with path and content
    documents = [
        {"file_path": path, "content": content}
        for path, content in context_dict_from_db.items()
        if path in retrieved_paths
    ]

    # Log if some paths from DB weren't in the initial vector retrieval list
    if len(documents) != len(context_dict_from_db):
        original_db_count = len(context_dict_from_db)
        final_doc_count = len(documents)
        logger.warning(
            f"DB returned {original_db_count} docs, but only {final_doc_count} "
            f"matched vector results for query_id {query_id}. Using the matched set."
        )

    return documents


def apply_reranking(
        documents: List[Dict],
        query: str,
        use_reranker: bool,
        reranker_top_k: int,
        retrieval_top_k: int,
        query_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply reranking to documents if enabled.

    Args:
        documents: List of documents with content
        query: User query for reranking
        use_reranker: Whether to use reranking
        reranker_top_k: Number of documents to keep after reranking
        retrieval_top_k: Number of documents retrieved initially
        query_id: Unique query identifier

    Returns:
        Tuple of (docs_for_context, docs_to_return)
    """
    docs_for_context = []
    docs_to_return = []

    if use_reranker and documents:
        print(f"Reranking {len(documents)} documents against original query, keeping top {reranker_top_k}...")
        try:
            reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranked_docs_list = reranker.rerank(
                query=query,
                documents=documents,
                content_key="content",
                top_k=reranker_top_k
            )
            print(f"Reranked to {len(reranked_docs_list)} documents.")
            docs_for_context = reranked_docs_list
            docs_to_return = reranked_docs_list
        except Exception as e:
            logger.error(f"Reranking failed for query_id {query_id}: {e}")
            docs_for_context = documents[:reranker_top_k]
            docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]
    else:
        limit = reranker_top_k if use_reranker else retrieval_top_k
        docs_for_context = documents[:limit]
        docs_to_return = [{**doc, 'rerank_score': None} for doc in docs_for_context]

        if not documents:
            print("No documents retrieved initially.")
        elif not use_reranker:
            print(f"Skipping reranking. Using top {len(docs_to_return)} retrieved documents.")
        else:
            print(f"Using top {len(docs_to_return)} retrieved documents (reranker limit applied before reranking).")

    return docs_for_context, docs_to_return


def generate_response(
        message: str,
        docs_for_context: List[Dict],
        chat_id: Optional[str],
        model_name: str,
        query_id: str,
        query_features: Dict[str, bool]
) -> str:
    """
    Generate response from LLM using retrieved context.

    Args:
        message: User message
        docs_for_context: Documents to use as context
        chat_id: Optional chat identifier
        model_name: LLM model to use
        query_id: Unique query identifier
        query_features: Dict to track feature usage

    Returns:
        Generated response text
    """
    # Format context for prompt
    context_string_for_prompt = "\n\n".join(
        f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
        for doc in docs_for_context
    ) if docs_for_context else "No relevant documents found."

    # Get chat history if available
    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""

    # Create prompt and generate response
    prompt = create_prompt(message, context_string_for_prompt, chat_history)

    try:
        print(f"Generating final response using model: {model_name}")
        model_handler = get_model_handler(model_name)
        response = model_handler.generate_text(prompt)
        print("LLM response generated successfully.")
        return response
    except Exception as e:
        query_features["final_llm_error"] = True
        logger.exception(f"LLM generation failed for query_id {query_id} using model {model_name}")
        return ERROR_RESPONSE_MESSAGE


def prepare_doc_details(docs_to_return: List[Dict], retrieved_results: List[Dict]) -> List[Dict]:
    """
    Prepare detailed document information for logging.

    Args:
        docs_to_return: Documents in final response
        retrieved_results: Original results from vector store

    Returns:
        List of document details for logging
    """
    # Create mapping from file_path to original vector store result
    original_results_map = {
        item['payload']['file_path']: item
        for item in retrieved_results
        if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']
    }

    # Prepare detailed info including tags and scores
    logged_doc_details = []
    for doc in docs_to_return:
        file_path = doc.get('file_path')
        original_result = original_results_map.get(file_path)
        doc_tag = "N/A"
        original_score = float('nan')

        if original_result and isinstance(original_result.get('payload'), dict):
            doc_tag = original_result['payload'].get('doc_tag', 'N/A')
            original_score = original_result.get('score', float('nan'))

        logged_doc_details.append({
            "path": file_path or "N/A",
            "tag": doc_tag,
            "retrieval_score": original_score,
            "rerank_score": doc.get('rerank_score')
        })

    return logged_doc_details


def log_response_metrics(
        timestamp: str,
        query_id: str,
        response: str,
        kb_id: str,
        model_name: str,
        use_hyde: bool,
        use_augmentation: bool,
        use_reranker: bool,
        retrieval_top_k: int,
        reranker_top_k: int,
        docs_retrieved_count: int,
        docs_found_in_db_count: int,
        docs_returned_count: int,
        doc_details: List[Dict],
        query_features: Dict[str, bool]
) -> None:
    """Log detailed metrics about the response generation process."""
    response_info = {
        "timestamp": timestamp,
        "query_id": query_id,
        "response_length": len(response),
        "kb_id": kb_id,
        "model_used": model_name,
        "hyde_used": use_hyde,
        "hyde_generated_doc": query_features["hypothetical_doc_generated"],
        "augmentation_used": use_augmentation,
        "augmentation_generated_query": query_features["augmented_query_generated"],
        "reranker_used": use_reranker,
        "retrieval_top_k": retrieval_top_k,
        "reranker_top_k": reranker_top_k if use_reranker else None,
        "docs_retrieved_count": docs_retrieved_count,
        "docs_found_in_db_count": docs_found_in_db_count,
        "docs_returned_count": docs_returned_count,
        "docs_returned_details": doc_details,
        "llm_generation_error": query_features["final_llm_error"]
    }

    log_message_content = f"RESPONSE_GENERATED {json.dumps(response_info)}"
    if query_features["final_llm_error"]:
        logger.error(log_message_content)
    else:
        logger.info(log_message_content)


def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()