from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

import config

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(query):
    """  
    Embed a query string or list of strings into vector representations  
    """
    try:
        if isinstance(query, str):
            # Ensure single string is processed correctly
            encoded = embedding_model.encode(query, convert_to_tensor=False)
            return encoded
        elif isinstance(query, list):
            # Handle list of strings
            return embedding_model.encode(query, convert_to_tensor=False)
        else:
            raise ValueError("Query must be a string or a list of strings.")
    except Exception as e:
        print(f"Error in embed_query: {str(e)}")
        raise

# Load FAISS vector store


def get_retriever():
    retriever = FAISS.load_local(
        config.FAISS_INDEX_PATH,
        embeddings=embed_query,
        allow_dangerous_deserialization=True
    ).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever
