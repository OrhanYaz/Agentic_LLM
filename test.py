import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import (
    AzureChatOpenAI,  # Assuming you're using AzureOpenAI or another LLM model
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o",
    openai_api_version="2023-05-15"
)

# Embeddings & Vector Store Setup
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

# Initialize FAISS vector store with embeddings


retriever = FAISS.load_local(
    "faiss_index", embeddings=embed_query, allow_dangerous_deserialization=True).as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Define Prompts
hr_prompt = PromptTemplate.from_template(
    "Answer HR-related questions based on the following context: {context} \nQuestion: {question}")
finance_prompt = PromptTemplate.from_template(
    "Answer finance-related queries based on the following context: {context} \nQuestion: {question}")
general_prompt = PromptTemplate.from_template(
    "Provide general information based on the following context: {context} \nQuestion: {question}")
summarize_prompt = PromptTemplate.from_template(
    "Summarize the following text: {context}")
qa_prompt = PromptTemplate.from_template(
    "Answer questions based on this document: {context} \nQuestion: {question}")


# RetrievalQA Chains for Different Agents
hr_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": hr_prompt}
)

finance_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": finance_prompt}
)

general_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": general_prompt}
)

summarize_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": summarize_prompt}
)

qa_agent = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt}
)

# Register Tools
tools = [
    Tool(
        name="HR Agent",
        func=hr_agent.run,
        description="Handles HR queries. Answer questions related to HR policies, benefits, etc."
    ),
    Tool(
        name="Finance Agent",
        func=finance_agent.run,
        description="Handles finance questions. Answer queries about accounting, budgeting, etc."
    ),
    Tool(
        name="General Agent",
        func=general_agent.run,
        description="Answer general information queries."
    ),
    Tool(
        name="Summarization Agent",
        func=summarize_agent.run,
        description="Summarizes text or documents."
    ),
    Tool(
        name="Document QA Agent",
        func=qa_agent.run,
        description="Answers questions from documents based on context."
    )
]

# Initialize the agent with the tools and set the agent type
agent = initialize_agent(
    tools=tools,
    # Use this or adjust depending on your needs
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

# Example query to run through the agent
query = "What is the policy on remote work?"

query_embedding = embedding_model.encode(query, convert_to_tensor=False)
# Get the answer from the agent
result = agent.run(query)
print(result)
