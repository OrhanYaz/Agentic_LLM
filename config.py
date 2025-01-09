import os

import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()


# Azure OpenAI Config

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_LLM"),
    openai_api_version="2023-05-15"
)

# FAISS Vector Store Setup
embedding_model = OpenAIEmbeddings(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                   openai_api_key=os.getenv("AZURE_OPENAI_KEY"), deployment="dsl-embedding-3")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS Vector Store
db = FAISS.load_local("faiss_index", embedding_model,
                      allow_dangerous_deserialization=True)

# Example Query
query = "Summarize the HR policy on remote work."

# Ensure proper formatting of the query before passing it to the model
query_embedding = embedding_model.encode(query, convert_to_tensor=False)


# Perform similarity search using the query embedding
try:
    results = db.similarity_search_by_vector(query_embedding, k=3)
    print("Search results:", results)
except Exception as e:
    print(f"Error during similarity search: {e}")
