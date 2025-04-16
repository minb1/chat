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
# Query Model
# ------------------------
class Query(models.Model):
    # Using a UUID for enhanced uniqueness (alternatively, you could use the chat_id supplied by your client)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_query = models.TextField(help_text="Original user query")
    enhanced_query = models.TextField(
        blank=True,
        null=True,
        help_text="Query enhanced using HyDE or query augmentation (if applicable)"
    )
    model_used = models.CharField(max_length=100, help_text="The LLM model used for processing")
    additional_parameters = models.JSONField(
        blank=True,
        null=True,
        help_text="JSON object storing parameters such as use_hyde, use_augmentation, etc."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query {self.id} by Model: {self.model_used}"

# ------------------------
# QueryDocument Junction Model
# ------------------------
class QueryDocument(models.Model):
    query = models.ForeignKey(Query, related_name='query_documents', on_delete=models.CASCADE)
    document = models.ForeignKey(Document, related_name='document_queries', on_delete=models.CASCADE)
    retrieval_score = models.FloatField(null=True, blank=True, help_text="Score from the vector retrieval process")
    rank_position = models.PositiveIntegerField(null=True, blank=True, help_text="Ranking position in the retrieval set")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('query', 'document')
        ordering = ['rank_position']

    def __str__(self):
        return f"Query {self.query.id} -> Document {self.document.id}"

# ------------------------
# Feedback Model
# ------------------------
class Feedback(models.Model):
    FEEDBACK_CHOICES = [
        ('helpful', 'Helpful'),
        ('unhelpful', 'Unhelpful'),
        ('false', 'False'),
    ]
    query = models.ForeignKey(Query, related_name='feedbacks', on_delete=models.CASCADE)
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_CHOICES)
    feedback_text = models.TextField(
        blank=True,
        null=True,
        help_text="Optional additional comments on the LLM response"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback for Query {self.query.id}: {self.feedback_type}"
