import os

import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15"
)
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")

# FAISS Vector Store Setup
embedding_model = OpenAIEmbeddings(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                   openai_api_key=os.getenv("AZURE_OPENAI_KEY"), model="text-embedding-3-large")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class AzureOpenAIEmbeddings(OpenAIEmbeddings):
    def _init_(self, client, model="text-embedding-3-large", deployment="dsl-embedding-3"):
        self.client = client
        self.model = model
        self.deployment = deployment

    def embed(self, texts):
        response = self.client.embeddings.create(
            input=texts, model=self.model, deployment_name=self.deployment)
        return [embedding['embedding'] for embedding in response['data']]


texts = [
    "This is a test sentence.",
    "This is another test sentence.",
    "LangChain and FAISS are useful for text embeddings."
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


texts = [
    "Remote work is allowed for three days per week.",
    "Quarterly financial reports are due by the 10th of each quarter.",
    "Employees are eligible for promotion after one year of service."
]

# Initialize Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, convert_to_tensor=False)

text_embeddings = list(zip(texts, embeddings))

# Create FAISS Index from Texts
db = FAISS.from_embeddings(text_embeddings, embedding_model)

# Save to Local Disk
db.save_local("faiss_index")
