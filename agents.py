from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# Embeddings & Vector Store
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever()

# Agent Prompts
hr_prompt = PromptTemplate.from_template("Answer HR-related questions. {question}")
finance_prompt = PromptTemplate.from_template("Answer finance-related queries. {question}")
general_prompt = PromptTemplate.from_template("Provide general information. {question}")
summarize_prompt = PromptTemplate.from_template("Summarize the following text: {text}")
qa_prompt = PromptTemplate.from_template("Answer questions based on this document: {question}")

# Tools
hr_agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                       chain_type_kwargs={"prompt": hr_prompt})
finance_agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                            chain_type_kwargs={"prompt": finance_prompt})
general_agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                            chain_type_kwargs={"prompt": general_prompt})
summarize_agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                              chain_type_kwargs={"prompt": summarize_prompt})
qa_agent = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff",
                                       chain_type_kwargs={"prompt": qa_prompt})

# Register Tools
tools = [
    Tool(name="HR Agent", func=hr_agent.run, description="Handles HR queries"),
    Tool(name="Finance Agent", func=finance_agent.run, description="Handles finance questions"),
    Tool(name="General Agent", func=general_agent.run, description="General queries"),
    Tool(name="Summarization Agent", func=summarize_agent.run, description="Summarizes text or docs"),
    Tool(name="Document QA Agent", func=qa_agent.run, description="Answers questions from documents")
]
