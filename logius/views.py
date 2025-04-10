# logius/views.py

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from logius.chat_handler import process_user_message
from memory.redis_handler import create_chat_session, get_chat_history
from model.model_factory import get_available_models
from utils.git_fetch_and_chunk import fetch_git_and_chunk
from utils.embedder import embed_chunks
from utils.postgresql_insert import insert_chunks_to_postgres
from utils.qdrant_upsert import insert_into_qdrant

import logging
logger = logging.getLogger(__name__)

# Default values for Top K parameters
DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_RERANKER_TOP_K = 10
MAX_TOP_K = 50 # Define a reasonable maximum


def chat_view(request):
    return render(request, 'index.html')

class ChatSessionView(APIView):
    """API view for creating a new chat session."""
    def post(self, request):
        try:
            chat_id = create_chat_session()
            logger.info(f"Created new chat session: {chat_id}")
            return Response({"chat_id": chat_id}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.exception("Error creating chat session")
            return Response({"error": "Failed to create chat session"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatHistoryView(APIView):
    """API view for retrieving chat history."""
    def get(self, request, chat_id):
        try:
            history = get_chat_history(chat_id)
            if history is None:
                logger.warning(f"Chat history not found for chat_id: {chat_id}")
                return Response({"error": "Chat session not found"}, status=status.HTTP_404_NOT_FOUND)
            logger.debug(f"Retrieved history for chat_id: {chat_id}")
            return Response({"history": history}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception(f"Error retrieving chat history for chat_id: {chat_id}")
            return Response({"error": "Failed to retrieve chat history"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatView(APIView):
    """API view for handling chat requests."""
    def post(self, request):
        """
        Process a chat request.
        Request body (JSON):
            {"query": "user message", "chat_id": "uuid-string", "model_name": "string",
             "use_reranker": boolean, "retrieval_top_k": int, "reranker_top_k": int,
             "use_hyde": boolean} # <<< Added use_hyde
        Returns:
            {"response": "assistant response", "docs": [...]} or {"error": "message"}
        """
        data = {} # Initialize data to avoid errors in exception logging if request.data fails
        try:
            data = request.data
            user_message = data.get('query')
            chat_id = data.get('chat_id')
            model_name = data.get('model_name', 'gemini') # Default model
            use_reranker = data.get('use_reranker', True)
            use_hyde = data.get('use_hyde', False) # <<< Get HyDE parameter, default False

            # --- Extract and Validate Top K parameters ---
            try:
                retrieval_top_k = int(data.get('retrieval_top_k', DEFAULT_RETRIEVAL_TOP_K))
                if not (1 <= retrieval_top_k <= MAX_TOP_K):
                    logger.warning(f"Invalid retrieval_top_k value ({retrieval_top_k}), using default {DEFAULT_RETRIEVAL_TOP_K}.")
                    retrieval_top_k = DEFAULT_RETRIEVAL_TOP_K
            except (ValueError, TypeError):
                logger.warning(f"Non-integer retrieval_top_k received, using default {DEFAULT_RETRIEVAL_TOP_K}.")
                retrieval_top_k = DEFAULT_RETRIEVAL_TOP_K

            try:
                reranker_top_k = int(data.get('reranker_top_k', DEFAULT_RERANKER_TOP_K))
                 # Reranker K should not be greater than Retrieval K
                if not (1 <= reranker_top_k <= retrieval_top_k):
                    logger.warning(f"Invalid reranker_top_k value ({reranker_top_k}) or > retrieval_top_k. Adjusting.")
                    reranker_top_k = min(retrieval_top_k, DEFAULT_RERANKER_TOP_K)
                    if reranker_top_k < 1:
                        reranker_top_k = 1
            except (ValueError, TypeError):
                logger.warning(f"Non-integer reranker_top_k received, using default {DEFAULT_RERANKER_TOP_K}.")
                # Ensure reranker_top_k doesn't exceed the (potentially defaulted) retrieval_top_k
                reranker_top_k = min(retrieval_top_k, DEFAULT_RERANKER_TOP_K)
                if reranker_top_k < 1: reranker_top_k = 1


            # --- Input Validation ---
            if not user_message:
                logger.warning("Chat request received without query.")
                return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
            if not chat_id:
                 logger.warning("Chat request received without chat_id.")
                 return Response({'error': 'No chat_id provided'}, status=status.HTTP_400_BAD_REQUEST)
            if not isinstance(use_reranker, bool):
                 logger.warning(f"Invalid 'use_reranker' value received: {use_reranker}. Defaulting to True.")
                 use_reranker = True
            if not isinstance(use_hyde, bool): # <<< Validate HyDE parameter
                 logger.warning(f"Invalid 'use_hyde' value received: {use_hyde}. Defaulting to False.")
                 use_hyde = False

            logger.info(
                f"Processing chat request - ChatID: {chat_id}, Model: {model_name}, "
                f"Reranker: {use_reranker}, RetrK: {retrieval_top_k}, RerankK: {reranker_top_k}, "
                f"HyDE: {use_hyde}" # <<< Log HyDE status
            )

            # --- Call the processing function ---
            result = process_user_message(
                message=user_message,
                chat_id=chat_id,
                model_name=model_name,
                kb_id='qdrant-logius', # Keep KB ID or make dynamic if needed
                use_reranker=use_reranker,
                retrieval_top_k=retrieval_top_k,
                reranker_top_k=reranker_top_k,
                use_hyde=use_hyde # <<< Pass validated HyDE value
            )

            logger.info(f"Chat request processed successfully for ChatID: {chat_id}")
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            # Use the 'data' dict captured at the start for logging context
            chat_id_for_log = data.get('chat_id', 'N/A')
            logger.exception(f"Error processing chat request for chat_id {chat_id_for_log}: {e}")
            return Response({'error': f'An internal server error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelsView(APIView):
    """API view for getting available models."""
    def get(self, request):
        try:
            models = get_available_models()
            return Response(models, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("Error retrieving available models")
            return Response({"error": "Failed to retrieve models"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class KnowledgeBase(APIView):
    """API view for triggering knowledge base update."""
    def get(self, request):
        logger.info("Knowledge base update triggered via API.")
        try:
            # Consider async task here for production
            logger.info("Fetching Git pages & Chunking...")
            fetch_git_and_chunk()
            logger.info("Fetched chunks, creating embeddings...")
            embed_chunks()
            logger.info("Chunks embedded, inserting into Qdrant...")
            insert_into_qdrant()
            logger.info("Qdrant operation complete, moving to PostgreSQL...")
            insert_chunks_to_postgres()
            logger.info("Knowledge base update process completed.")
            return Response({'message': 'Knowledge base update process finished successfully.'}, status=status.HTTP_200_OK)
        except Exception as e:
             logger.exception("Error during knowledge base update process triggered via API.")
             return Response({'error': f'Knowledge base update failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)