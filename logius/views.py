# logius/views.py

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status # Import status codes for clearer responses

# Remove json import if only used here, DRF handles it
# from django.http import JsonResponse # Use DRF Response instead
# Remove decorators if not strictly needed for DRF views
# from django.views.decorators.http import require_POST
# from django.views.decorators.csrf import csrf_exempt

# Your existing imports
from logius.chat_handler import process_user_message
from memory.redis_handler import create_chat_session, get_chat_history
from model.model_factory import get_available_models
from utils.git_fetch_and_chunk import fetch_git_and_chunk
from utils.embedder import embed_chunks
from utils.postgresql_insert import insert_chunks_to_postgres
from utils.qdrant_upsert import insert_into_qdrant

# It's good practice to have logging
import logging
logger = logging.getLogger(__name__)

# Create your views here.
def chat_view(request):
    return render(request, 'index.html')

class ChatSessionView(APIView):
    """
    API view for creating a new chat session.
    """
    def post(self, request):
        """
        Create a new chat session.
        Returns:
            {"chat_id": "uuid-string"}
        """
        try:
            chat_id = create_chat_session()
            logger.info(f"Created new chat session: {chat_id}")
            return Response({"chat_id": chat_id}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.exception("Error creating chat session")
            return Response({"error": "Failed to create chat session"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatHistoryView(APIView):
    """
    API view for retrieving chat history.
    """
    def get(self, request, chat_id):
        """
        Get the chat history for a specific chat session.
        Returns:
            {"history": [{"role": "user/assistant", "content": "message", "timestamp": "..."}]}
        """
        try:
            history = get_chat_history(chat_id)
            if history is None: # Handle case where chat_id might not exist in Redis
                logger.warning(f"Chat history not found for chat_id: {chat_id}")
                return Response({"error": "Chat session not found"}, status=status.HTTP_404_NOT_FOUND)
            logger.debug(f"Retrieved history for chat_id: {chat_id}")
            return Response({"history": history}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception(f"Error retrieving chat history for chat_id: {chat_id}")
            return Response({"error": "Failed to retrieve chat history"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatView(APIView):
    """
    API view for handling chat requests.
    """
    def post(self, request):
        """
        Process a chat request.
        Request body (JSON):
            {"query": "user message", "chat_id": "uuid-string", "model_name": "string", "use_reranker": boolean}
        Returns:
            {"response": "assistant response", "docs": [...]} or {"error": "message"}
        """
        try:
            # Use DRF's request.data which handles JSON parsing automatically
            data = request.data
            user_message = data.get('query')
            chat_id = data.get('chat_id')
            # Use a default model if none is provided by the frontend
            model_name = data.get('model_name', 'gemini') # Or get default from settings
            # Extract the use_reranker flag, defaulting to True if not sent
            use_reranker = data.get('use_reranker', True)

            # --- Input Validation ---
            if not user_message:
                logger.warning("Chat request received without query.")
                return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
            if not chat_id:
                 logger.warning("Chat request received without chat_id.")
                 # Depending on your logic, you might allow chats without IDs
                 # For now, assume it's required based on previous code
                 return Response({'error': 'No chat_id provided'}, status=status.HTTP_400_BAD_REQUEST)
            if not isinstance(use_reranker, bool):
                 logger.warning(f"Invalid 'use_reranker' value received: {use_reranker}. Defaulting to True.")
                 use_reranker = True # Default to True if invalid type received

            logger.info(f"Processing chat request - ChatID: {chat_id}, Model: {model_name}, Reranker: {use_reranker}")

            # --- Call the processing function ---
            result = process_user_message(
                message=user_message, # Pass message correctly
                chat_id=chat_id,
                model_name=model_name,
                use_reranker=use_reranker # <<< Pass the extracted flag here
            )

            logger.info(f"Chat request processed successfully for ChatID: {chat_id}")
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            # Log the full exception traceback for debugging
            logger.exception(f"Error processing chat request for chat_id {chat_id}: {e}")
            # Return a generic error to the client
            return Response({'error': f'An internal server error occurred: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelsView(APIView):
    def get(self, request):
        """
        Get available LLM models.
        """
        try:
            models = get_available_models()
            return Response(models, status=status.HTTP_200_OK)
        except Exception as e:
            logger.exception("Error retrieving available models")
            return Response({"error": "Failed to retrieve models"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class KnowledgeBase(APIView):
    def get(self, request):
        """
        Triggers the knowledge base update process.
        NOTE: This should ideally be an async task (Celery, RQ) for long-running processes.
              Returning immediately might be misleading if the process takes time.
        """
        # Consider adding authentication/authorization here
        logger.info("Knowledge base update triggered via API.")
        try:
            # Ideally, you would trigger an asynchronous task here
            # For simplicity now, calling directly:
            logger.info("Fetching Git pages & Chunking...")
            fetch_git_and_chunk()
            logger.info("Fetched chunks, creating embeddings...")
            embed_chunks()
            logger.info("Chunks embedded, inserting into Qdrant...")
            insert_into_qdrant()
            logger.info("Qdrant operation complete, moving to PostgreSQL...")
            insert_chunks_to_postgres()
            logger.info("Knowledge base update process completed.")
            # Use DRF Response
            return Response({'message': 'Knowledge base update process finished successfully.'}, status=status.HTTP_200_OK)
        except Exception as e:
             logger.exception("Error during knowledge base update process triggered via API.")
             return Response({'error': f'Knowledge base update failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)