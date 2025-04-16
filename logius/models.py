import uuid
from django.db import models

# ------------------------
# Document Model
# ------------------------
class Document(models.Model):
    file_path = models.TextField(unique=True)
    doc_tag = models.TextField(blank=True, null=True)
    content = models.TextField()
    original_url = models.URLField(max_length=500, blank=True, null=True, help_text="Main source URL for the document")
    chunk_url = models.URLField(max_length=500, blank=True, null=True, help_text="Permalink to the specific document chunk")
    inserted_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['doc_tag'], name='idx_documents_doc_tag'),
        ]
        ordering = ['-inserted_at']

    def __str__(self):
        return f"{self.doc_tag or 'No Tag'}: {self.file_path}"

# ------------------------
# ChatQuery Model
# ------------------------
class ChatQuery(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chat_id = models.CharField(max_length=36, db_index=True, help_text="Session ID for grouping queries")
    user_query = models.TextField(help_text="Original user query")
    llm_response = models.TextField(blank=True, null=True, help_text="Response from the LLM")
    doc_tag = models.JSONField(
        default=list,
        blank=True,
        help_text="List of document tags for the query context"
    )
    file_paths = models.JSONField(
        default=list,
        help_text="List of file_path strings for retrieved document chunks"
    )
    model_used = models.CharField(max_length=100, blank=True, help_text="The LLM model used")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['chat_id'], name='idx_chatquery_chat_id'),
            models.Index(fields=['doc_tag'], name='idx_chatquery_doc_tag'),
            models.Index(fields=['created_at'], name='idx_chatquery_created_at'),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"Query {self.id} in session {self.chat_id}"

# ------------------------
# Feedback Model
# ------------------------
class Feedback(models.Model):
    FEEDBACK_CHOICES = [
        ('helpful', 'Helpful'),
        ('unhelpful', 'Unhelpful'),
        ('false', 'False'),
    ]
    query = models.ForeignKey(
        'ChatQuery',
        related_name='feedbacks',
        on_delete=models.CASCADE,
        help_text="The query this feedback is for"
    )
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_CHOICES)
    feedback_text = models.TextField(
        blank=True,
        null=True,
        help_text="Optional additional comments on the LLM response"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Feedback for Query {self.query.id}: {self.feedback_type}"