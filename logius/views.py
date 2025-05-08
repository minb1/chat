# logius/views.py
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
import logging

# Import the updated function
from logius.chat_handler import process_user_message
from memory.redis_handler import create_chat_session, get_chat_history
from model.model_factory import get_available_models
from utils.git_fetch_and_chunk import fetch_git_and_chunk
from utils.embedder import embed_chunks
from utils.postgresql_insert import insert_chunks_to_postgres
from utils.qdrant_upsert import insert_into_qdrant
from logius.models import ChatQuery, Feedback

# Loggers
logger = logging.getLogger(__name__)
feedback_logger = logging.getLogger('user_feedback')

# Default values for Top K parameters
# DEFAULT_RETRIEVAL_TOP_K is no longer used directly by process_user_message
DEFAULT_RERANKER_TOP_K = 10
# MAX_TOP_K might still be useful for validating reranker_top_k if desired
# MAX_TOP_K = 50 # Example limit for reranker

def chat_view(request):
    return render(request, 'index.html')

def chat_view_temp(request):
    return render(request, 'temp_adi.html')


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
        data = {}
        try:
            data = request.data
            user_message = data.get('query')
            chat_id = data.get('chat_id')
            model_name = data.get('model_name', 'gemini') # Keep model selection
            use_reranker = data.get('use_reranker', True) # Keep reranker flag

            # --- REMOVED unused retrieval_top_k logic ---
            # retrieval_top_k is now handled internally by process_user_message (DEFAULT_FRESH_K, MAX_TOTAL_DOCS etc.)

            # Extract and validate Reranker Top K parameter
            try:
                # Use the default from chat_handler if not provided or invalid
                # You can still allow overriding it via the API if needed
                reranker_top_k = int(data.get('reranker_top_k', DEFAULT_RERANKER_TOP_K))
                # Add validation if desired, e.g., against a MAX_TOP_K
                # if not (1 <= reranker_top_k <= MAX_TOP_K):
                #     logger.warning(f"Invalid reranker_top_k value ({reranker_top_k}). Adjusting.")
                #     reranker_top_k = DEFAULT_RERANKER_TOP_K
                if reranker_top_k < 1:
                     logger.warning(f"reranker_top_k ({reranker_top_k}) cannot be less than 1. Using 1.")
                     reranker_top_k = 1

            except (ValueError, TypeError):
                logger.warning(f"Non-integer reranker_top_k received, using default {DEFAULT_RERANKER_TOP_K}.")
                reranker_top_k = DEFAULT_RERANKER_TOP_K

            # --- REMOVED unused use_hyde and use_augmentation logic ---
            # These flags are no longer passed to process_user_message

            # Input validation (keep relevant ones)
            if not user_message:
                logger.warning("Chat request received without query.")
                return Response({'error': 'No query provided'}, status=status.HTTP_400_BAD_REQUEST)
            if not chat_id:
                logger.warning("Chat request received without chat_id.")
                return Response({'error': 'No chat_id provided'}, status=status.HTTP_400_BAD_REQUEST)
            if not isinstance(use_reranker, bool):
                logger.warning(f"Invalid 'use_reranker' value received: {use_reranker}. Defaulting to True.")
                use_reranker = True

            # Log the relevant parameters being used
            logger.info(
                f"Processing chat request - ChatID: {chat_id}, Model: {model_name}, "
                f"Reranker: {use_reranker}, RerankK: {reranker_top_k}" # Removed RetrK, HyDE, Augment
            )

            # Call the processing function with the correct arguments
            result = process_user_message(
                message=user_message,
                chat_id=chat_id,
                model_name=model_name,
                kb_id='qdrant-logius', # Keep kb_id if it's configurable or needed
                use_reranker=use_reranker,
                reranker_top_k=reranker_top_k
                # --- REMOVED retrieval_top_k ---
                # --- REMOVED use_hyde ---
                # --- REMOVED use_augmentation ---
            )

            logger.info(f"Chat request processed successfully for ChatID: {chat_id}")
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            chat_id_for_log = data.get('chat_id', 'N/A')
            logger.exception(f"Error processing chat request for chat_id {chat_id_for_log}: {e}")
            # Check if the error is the specific TypeError we were solving
            if isinstance(e, TypeError) and 'unexpected keyword argument' in str(e):
                 error_msg = f"Internal configuration error: Mismatch in function call arguments. Please check server logs. ({str(e)})"
                 logger.error("Argument mismatch detected between view and chat_handler. Verify removed arguments.")
                 # Return 500 but with a more specific hint for the developer if possible
                 return Response({'error': error_msg}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            else:
                 # General error handling
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

class FeedbackView(APIView):
    """API view for receiving user feedback on LLM responses."""
    def post(self, request):
        feedback_data = {}
        try:
            feedback_data = request.data
            query_id = feedback_data.get('query_id')
            feedback_type = feedback_data.get('feedback_type')
            feedback_text = feedback_data.get('feedback_text', '')

            if not query_id or not feedback_type:
                missing = [k for k in ['query_id', 'feedback_type'] if not feedback_data.get(k)]
                logger.warning(f"Received incomplete feedback data. Missing: {missing}")
                return Response({'error': f'Missing required fields: {missing}'}, status=status.HTTP_400_BAD_REQUEST)

            valid_feedback_types = [choice[0] for choice in Feedback.FEEDBACK_CHOICES]
            if feedback_type not in valid_feedback_types:
                logger.warning(f"Received invalid feedback type: {feedback_type}")
                return Response({'error': f'Invalid feedback_type. Must be one of {valid_feedback_types}'}, status=status.HTTP_400_BAD_REQUEST)

            chat_query = ChatQuery.objects.get(id=query_id)
            Feedback.objects.create(
                query=chat_query,
                feedback_type=feedback_type,
                feedback_text=feedback_text if feedback_text else None
            )

            # Log feedback
            log_payload = {
                "query_id": query_id,
                "feedback_type": feedback_type,
                "feedback_text": feedback_text,
                "user_query": chat_query.user_query,
                "llm_response": chat_query.llm_response
            }
            feedback_logger.info(json.dumps(log_payload))

            return Response({'status': 'Feedback received successfully'}, status=status.HTTP_200_OK)

        except ChatQuery.DoesNotExist:
            logger.warning(f"ChatQuery not found for query_id: {feedback_data.get('query_id', 'N/A')}")
            return Response({'error': 'Query not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            query_id_for_log = feedback_data.get('query_id', 'N/A')
            logger.exception(f"Error processing feedback for query_id {query_id_for_log}: {e}")
            return Response({'error': 'An internal server error occurred'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)