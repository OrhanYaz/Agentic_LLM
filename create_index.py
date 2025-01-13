"""
This script creates a FAISS vector store index from text documents using SentenceTransformer embeddings.
The script performs the following operations:
1. Loads a sentence transformer model (all-MiniLM-L6-v2)
2. Encodes sample texts into embeddings
3. Creates a FAISS index from the text-embedding pairs
4. Saves the index to local disk
Dependencies:
    - sentence_transformers
    - langchain
    - faiss
    - numpy
    - os
Example texts used are company policies, but can be replaced with any text documents.
The resulting index is saved in a 'faiss_index' directory in the local filesystem,
which can be later loaded for similarity search operations.
"""


import os

import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer

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
