import os
import json
import logging
import datetime
import sys
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from django.db import transaction

try:
    from logius.models import Document, ChatQuery
except ImportError:
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
        Dictionary containing response, query_id, or error and relevant documents
    """
    timestamp = datetime.datetime.now().isoformat()
    query_id = str(uuid4())
    response = ERROR_RESPONSE_MESSAGE
    final_docs_to_return_serialized = []

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

        # Process query with potential enhancements
        query_for_retrieval, query_embedding, query_features = process_query_enhancements(
            message, model_name, embedding_handler, use_hyde, use_augmentation, query_id, query_features
        )

        if query_embedding is None:
            logger.error(f"Embedding generation failed completely for query_id {query_id}.")
            return {"error": "Failed to generate query embedding", "docs": [], "query_id": query_id}

        # Retrieve relevant document paths/scores from vector store
        retrieved_results, retrieved_paths = retrieve_vector_results(
            kb_id, query_embedding, retrieval_top_k, query_id
        )

        # Retrieve full Document objects from database
        document_objects = retrieve_chunks_from_db(retrieved_paths)

        # Create score map from vector store results
        score_map = {item['payload']['file_path']: item['score'] for item in retrieved_results
                     if isinstance(item.get('payload'), dict) and 'file_path' in item['payload']}

        # Create intermediate structure
        intermediate_docs = [
            {
                "file_path": doc.file_path,
                "content": doc.content,
                "document": doc,
                "retrieval_score": score_map.get(doc.file_path),
                "rerank_score": None
            }
            for doc in document_objects if doc.file_path in score_map
        ]

        # Apply reranking
        docs_for_context, docs_for_ranking = apply_reranking(
            intermediate_docs, message, use_reranker, reranker_top_k, retrieval_top_k, query_id
        )

        # Generate response
        response = generate_response(
            message, docs_for_context, chat_id, model_name, query_id, query_features
        )

        # Store in ChatQuery
        with transaction.atomic():
            file_paths = [doc['file_path'] for doc in docs_for_context]
            doc_tags = [doc['document'].doc_tag for doc in docs_for_context if
                        doc.get('document') and getattr(doc['document'], 'doc_tag', None)]
            ChatQuery.objects.create(
                id=query_id,
                chat_id=chat_id or query_id,
                user_query=message,
                llm_response=response if not query_features["final_llm_error"] else None,
                doc_tag=doc_tags,
                file_paths=file_paths,
                model_used=model_name
            )

        # Log conversation in Redis
        if chat_id:
            log_message(chat_id, "user", message)
            log_message(chat_id, "assistant", response)

        # Prepare final API response
        final_docs_to_return_serialized = serialize_documents_for_response(docs_for_ranking)

        # Log metrics
        log_doc_details = prepare_doc_details_for_logging(final_docs_to_return_serialized)
        log_response_metrics(
            timestamp, query_id, response, kb_id, model_name,
            use_hyde, use_augmentation, use_reranker,
            retrieval_top_k, reranker_top_k,
            len(retrieved_results), len(document_objects), len(final_docs_to_return_serialized),
            log_doc_details, query_features
        )

        # Return response
        if query_features["final_llm_error"]:
            return {"error": response, "docs": final_docs_to_return_serialized, "query_id": query_id}
        return {"response": response, "docs": final_docs_to_return_serialized, "query_id": query_id}

    except Exception as e:
        logger.exception(f"Unexpected error processing message for query_id {query_id}")
        return {"error": "An unexpected server error occurred.", "docs": final_docs_to_return_serialized, "query_id": query_id}

def apply_reranking(
    intermediate_docs: List[Dict],
    query: str,
    use_reranker: bool,
    reranker_top_k: int,
    retrieval_top_k: int,
    query_id: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply reranking to documents if enabled.

    Args:
        intermediate_docs: List of dicts containing 'document', 'content', 'file_path', 'retrieval_score'.
        query: User query for reranking.
        use_reranker: Whether to use reranking.
        reranker_top_k: Number of documents to keep after reranking.
        retrieval_top_k: Number of documents retrieved initially.
        query_id: Unique query identifier.

    Returns:
        Tuple of:
        - docs_for_context: List[Dict] containing only 'file_path' and 'content' for LLM.
        - docs_for_ranking: List[Dict] containing original dict structure plus 'rerank_score'.
    """
    docs_for_context = []
    docs_for_ranking = []

    if not intermediate_docs:
        print("No documents received for potential reranking.")
        return [], []

    # Sort by initial retrieval score
    intermediate_docs.sort(key=lambda x: x.get('retrieval_score') if x.get('retrieval_score') is not None else -1.0, reverse=True)

    if use_reranker:
        print(f"Reranking {len(intermediate_docs)} documents against original query, keeping top {reranker_top_k}...")
        reranking_successful = False
        try:
            reranker = SentenceTransformerReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            docs_to_rerank = [
                {"id": doc['file_path'], "content": doc['content']}
                for doc in intermediate_docs if doc.get('file_path') and doc.get('content')
            ]

            if not docs_to_rerank:
                print("No valid documents left to rerank after filtering.")
                raise ValueError("No valid documents to rerank.")

            reranked_output = reranker.rerank(
                query=query,
                documents=docs_to_rerank,
                content_key="content",
                top_k=reranker_top_k
            )
            print(f"Reranked to {len(reranked_output)} documents.")
            logger.debug(f"Query ID {query_id}: Raw reranker output: {reranked_output}")

            reranked_scores_map = {}
            if reranked_output and isinstance(reranked_output[0], dict):
                possible_score_keys = ['score', 'relevance_score', 'rerank_score']
                actual_score_key = None
                for key in possible_score_keys:
                    if key in reranked_output[0]:
                        actual_score_key = key
                        print(f"Detected score key from reranker: '{actual_score_key}'")
                        break

                if actual_score_key:
                    reranked_scores_map = {
                        item.get('id'): item.get(actual_score_key)
                        for item in reranked_output if item.get('id') is not None
                    }
                    reranking_successful = True
                else:
                    logger.warning(f"Query ID {query_id}: Could not find a known score key in reranker output.")
            else:
                logger.warning(f"Query ID {query_id}: Reranker output was not a list of dictionaries.")

            if reranking_successful:
                temp_ranked_docs = []
                for doc_dict in intermediate_docs:
                    file_path = doc_dict['file_path']
                    if file_path in reranked_scores_map:
                        doc_dict['rerank_score'] = float(reranked_scores_map.get(file_path)) if reranked_scores_map.get(file_path) is not None else None
                        temp_ranked_docs.append(doc_dict)

                docs_for_ranking = sorted(
                    temp_ranked_docs,
                    key=lambda x: x.get('rerank_score') if x.get('rerank_score') is not None else -1.0,
                    reverse=True
                )
            else:
                raise RuntimeError("Reranking produced no usable scores.")

        except Exception as e:
            logger.error(f"Reranking failed for query_id {query_id}: {e}", exc_info=True)
            limit = min(reranker_top_k, len(intermediate_docs))
            docs_for_ranking = intermediate_docs[:limit]
            for doc in docs_for_ranking:
                doc['rerank_score'] = None
    else:
        limit = min(retrieval_top_k, len(intermediate_docs))
        docs_for_ranking = intermediate_docs[:limit]
        for doc in docs_for_ranking:
            doc['rerank_score'] = None
        print(f"Skipping reranking. Using top {len(docs_for_ranking)} retrieved documents.")

    docs_for_context = [
        {"file_path": d['file_path'], "content": d['content'], "document": d['document']}
        for d in docs_for_ranking
    ]
    print(docs_for_context[0])
    return docs_for_context, docs_for_ranking

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
    """
    if docs_for_context:
        context_string_for_prompt = "\n\n".join(
            f"Document Path: {doc['file_path']}\nContent:\n{doc['content']}"
            for doc in docs_for_context
        )
    else:
        context_string_for_prompt = "No relevant documents found."

    chat_history = format_chat_history_for_prompt(chat_id) if chat_id else ""

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

def serialize_documents_for_response(docs_for_ranking: List[Dict]) -> List[Dict]:
    """
    Converts documents to JSON-serializable format.
    """
    serialized_docs = []
    for result in docs_for_ranking:
        doc_obj = result.get('document')
        if not isinstance(doc_obj, Document):
            logger.warning(f"Item in docs_for_ranking is missing Document object: {result.get('file_path')}")
            continue

        serialized_doc = {
            'id': doc_obj.id,
            'file_path': doc_obj.file_path,
            'doc_tag': doc_obj.doc_tag,
            'content': doc_obj.content,
            'original_url': doc_obj.original_url,
            'chunk_url': doc_obj.chunk_url,
            'inserted_at': doc_obj.inserted_at.isoformat() if doc_obj.inserted_at else None,
            'updated_at': doc_obj.updated_at.isoformat() if doc_obj.updated_at else None,
            'retrieval_score': result.get('retrieval_score'),
            'rerank_score': result.get('rerank_score')
        }
        serialized_docs.append(serialized_doc)
    print(serialized_docs)
    return serialized_docs

def prepare_doc_details_for_logging(serialized_docs: List[Dict]) -> List[Dict]:
    """
    Prepare document details for logging.
    """
    log_details = []
    for doc in serialized_docs:
        log_details.append({
            "path": doc.get('file_path', 'N/A'),
            "tag": doc.get('doc_tag', 'N/A'),
            "retrieval_score": doc.get('retrieval_score'),
            "rerank_score": doc.get('rerank_score')
        })
    return log_details

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
) -> Tuple[str, Optional[List[float]], Dict[str, bool]]:
    """
    Process query enhancements (HyDE or augmentation).
    """
    query_for_retrieval = message
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
            logger.error(f"HyDE generation/embedding failed for query_id {query_id}: {e}", exc_info=True)
            try:
                query_embedding = embedding_handler.get_embedding(message)
                print("Generated embedding for original user query (HyDE fallback).")
            except Exception as embed_e:
                logger.error(f"Fallback embedding also failed for query_id {query_id}: {embed_e}", exc_info=True)
                query_embedding = None

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
                augmented_query = message
            else:
                print(f"Generated Augmented Query:\n---\n{augmented_query}\n---")
                query_features["augmented_query_generated"] = True

            query_for_retrieval = augmented_query
            query_embedding = embedding_handler.get_embedding(query_for_retrieval)
        except Exception as e:
            logger.error(f"Query Augmentation failed for query_id {query_id}: {e}", exc_info=True)
            try:
                query_embedding = embedding_handler.get_embedding(message)
                print("Generated embedding for original user query (Augmentation fallback).")
            except Exception as embed_e:
                logger.error(f"Fallback embedding also failed for query_id {query_id}: {embed_e}", exc_info=True)
                query_embedding = None
    else:
        print("Using original query embedding.")
        try:
            query_embedding = embedding_handler.get_embedding(message)
        except Exception as embed_e:
            logger.error(f"Original query embedding failed for query_id {query_id}: {embed_e}", exc_info=True)
            query_embedding = None

    return query_for_retrieval, query_embedding, query_features

def retrieve_vector_results(kb_id: str, query_embedding: List[float], top_k: int, query_id: str) -> Tuple[
    List[Dict], List[str]]:
    """
    Retrieve results from vector store.
    """
    print(f"Getting vector client for KB: {kb_id}")
    vector_client = get_vector_client(kb_id)
    print(f"Getting top {top_k} related vectors...")

    retrieved_results = []
    retrieved_paths = []
    try:
        retrieved_results = vector_client.retrieve_vectors(query_embedding, top_k=top_k)
        print(f"Retrieved {len(retrieved_results)} results from vector storage.")

        if retrieved_results:
            print("Top results from Vector Store (Path, Tag, Score):")
            for item in retrieved_results[:5]:
                payload = item.get('payload', {})
                path = payload.get('file_path', 'N/A')
                tag = payload.get('doc_tag', 'N/A')
                score = item.get('score', float('nan'))
                print(f"  - Path: {path}, Tag: {tag}, Score: {score:.4f}")

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
    except Exception as vector_e:
        logger.error(f"Vector retrieval failed for query_id {query_id}: {vector_e}", exc_info=True)

    return retrieved_results, retrieved_paths

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
    log_doc_details: List[Dict],
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
        "docs_returned_details": log_doc_details,
        "llm_generation_eTESTrror": query_features["final_llm_error"]
    }

    log_message_content = f"RESPONSE_GENERATED {json.dumps(response_info)}"
    if query_features["final_llm_error"]:
        logger.error(log_message_content)
    else:
        logger.info(log_message_content)

def get_kb_options():
    """Get available knowledge base options."""
    return get_available_knowledge_bases()