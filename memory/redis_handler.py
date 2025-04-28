import redis
import json
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True,
    username=os.getenv('REDIS_USERNAME'),
    password=os.getenv('REDIS_PASSWORD'),
)


def create_chat_session():
    """
    Create a new chat session with a unique ID and 30 min TTL

    Returns:
        str: The unique chat session ID
    """
    chat_id = str(uuid.uuid4())
    # Store creation timestamp
    redis_client.hset(f"chat:{chat_id}", "created_at", datetime.now().isoformat())

    # Set TTL for both the chat metadata and messages
    redis_client.expire(f"chat:{chat_id}", 3600)
    redis_client.expire(f"chat:{chat_id}:messages", 3600)

    return chat_id


def log_message(chat_id, role, content):
    """
    Log a message to the chat history.

    Args:
        chat_id (str): The chat session ID
        role (str): The role of the message sender ('user' or 'assistant')
        content (str): The message content

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a message object
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # Add message to the chat history list
        redis_client.rpush(f"chat:{chat_id}:messages", json.dumps(message))
        return True
    except Exception as e:
        print(f"Error logging message to Redis: {e}")
        return False


def get_chat_history(chat_id):
    """
    Retrieve the chat history for a given chat session.

    Args:
        chat_id (str): The chat session ID

    Returns:
        list: List of message objects with role and content
    """
    try:
        # Get all messages from the chat history list
        messages_json = redis_client.lrange(f"chat:{chat_id}:messages", 0, -1)

        # Parse JSON messages
        messages = [json.loads(msg) for msg in messages_json]
        return messages
    except Exception as e:
        print(f"Error retrieving chat history from Redis: {e}")
        return []


def format_chat_history_for_prompt(chat_id: str, max_turns: int = 3) -> str:
    """
    Format the MOST RECENT chat history for inclusion in the prompt,
    limited by the number of turns (Q+A pairs).

    Args:
        chat_id (str): The chat session ID
        max_turns (int): The maximum number of Q+A turns to include.

    Returns:
        str: Formatted chat history for the prompt, or empty string.
    """
    try:
        # Retrieve the last N messages (N = max_turns * 2)
        # lrange with negative indices: -1 is last, -2 is second last, etc.
        num_messages_to_fetch = max_turns * 2
        messages_json = redis_client.lrange(f"chat:{chat_id}:messages", -num_messages_to_fetch, -1)

        if not messages_json:
            return "" # No history

        messages = [json.loads(msg) for msg in messages_json]

        # Format the selected messages
        formatted_history = [] # Build parts in a list
        for msg in messages: # Already in chronological order (oldest of the N to newest)
            role = "Gebruiker" if msg.get("role") == "user" else "Assistent"
            content = msg.get("content", "").strip()
            if content: # Avoid adding empty lines
                 formatted_history.append(f"{role}: {content}")

        # Join with double newlines for readability in the prompt
        return "\n\n".join(formatted_history)

    except Exception as e:
        print(f"Error formatting chat history from Redis: {e}")
        return "" # Return empty string on error
