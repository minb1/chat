from django.shortcuts import render
from rest_framework.response import Response

from rest_framework.views import APIView
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from logius.chat_handler import process_user_message
from memory.redis_handler import create_chat_session, get_chat_history
from model.model_factory import get_available_models

from utils.git_fetch_and_chunk import fetch_git_and_chunk
from utils.embedder import embed_chunks
from utils.postgresql_insert import insert_chunks_to_postgres
from utils.qdrant_upsert import insert_into_qdrant


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
        chat_id = create_chat_session()
        return Response({"chat_id": chat_id})

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
        history = get_chat_history(chat_id)
        return Response({"history": history})


class ChatView(APIView):
    """
    API view for handling chat requests.
    """

    def post(self, request):
        """
        Process a chat request.
        Request body:
            {"query": "user message", "chat_id": "uuid-string" (optional)}
        Returns:
            {"response": "assistant response"}
        """
        try:
            data = json.loads(request.body)
            user_message = data.get('query')
            chat_id = data.get('chat_id')
            model_name = data.get('model_name')
            if not user_message:
                return JsonResponse({'error': 'No message provided'}, status=400)

            result = process_user_message(user_message, chat_id=chat_id, model_name=model_name)
            return Response(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

class ModelsView(APIView):
    def get(self, request):
        models = get_available_models()
        return Response(models)

class KnowledgeBase(APIView):
    def get(self, request):

        print("Calling knowledge base update scripts")
        print("Fetching Git pages & Chunking")
        fetch_git_and_chunk()
        print("Fetched chunsk, creating embeddings...")
        embed_chunks()
        print("Chunks embedded, inserting into qdrant..")
        insert_into_qdrant()
        print("Qdrant operation complete, moving to PostGreSQL")
        insert_chunks_to_postgres()
        return JsonResponse({'status': 'update triggered'})