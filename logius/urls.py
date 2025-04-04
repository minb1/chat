from django.urls import path
from .views import chat_view, ChatView, ChatSessionView, ChatHistoryView, ModelsView, KnowledgeBase

urlpatterns = [
    path('', chat_view, name='chat'),
    path('api/chat/', ChatView.as_view(), name='api_chat'),
    path('api/chat/session/', ChatSessionView.as_view(), name='chat_session'),
    path('api/chat/history/<str:chat_id>/', ChatHistoryView.as_view(), name='chat_history'),
    path('api/chat/models/', ModelsView.as_view(), name='chat_models'),

    path('api/knowledge_base/build/', KnowledgeBase.as_view(), name='knowledge_base')
]
