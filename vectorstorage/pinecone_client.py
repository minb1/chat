from pinecone import Pinecone


from dotenv import load_dotenv
import os

load_dotenv()

def retrieve_vectors(vector):

    try:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    except:
        return "No API key found"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("google-embed-004-768d")

    response = index.query(
        namespace="ns1",
        vector=vector,
        top_k=50,
        include_values=False,
        include_metadata=True
    )

    # Retrieves all relatiev file paths
    chunk_paths = [match["metadata"].get("file_path") for match in response["matches"]]

    return chunk_paths