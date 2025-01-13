from langchain.chains import RetrievalQA

import embeddings
import llm
import prompts

retriever = embeddings.get_retriever()

# HR Agent
hr_agent = RetrievalQA.from_chain_type(
    llm=llm.llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompts.hr_prompt}
)

# Finance Agent
finance_agent = RetrievalQA.from_chain_type(
    llm=llm.llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompts.finance_prompt}
)

# General Agent
general_agent = RetrievalQA.from_chain_type(
    llm=llm.llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompts.general_prompt}
)

# Summarize Agent
summarize_agent = RetrievalQA.from_chain_type(
    llm=llm.llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompts.summarize_prompt}
)

# QA Agent
qa_agent = RetrievalQA.from_chain_type(
    llm=llm.llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompts.qa_prompt}
)
