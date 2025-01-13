import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = "gpt-4o"
OPENAI_API_VERSION = "2023-05-15"

# FAISS Index Path
FAISS_INDEX_PATH = "faiss_index"
