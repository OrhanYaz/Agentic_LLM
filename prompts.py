from langchain.prompts import PromptTemplate

# Define Prompts
hr_prompt = PromptTemplate.from_template(
    "Answer HR-related questions based on the following context: {context} \nQuestion: {question} using user's same language"
)
finance_prompt = PromptTemplate.from_template(
    "Answer finance-related queries based on the following context: {context} \nQuestion: {question} using user's same language "
)
general_prompt = PromptTemplate.from_template(
    "Provide general information based on the following context: {context} \nQuestion: {question} using user's same language "
)
summarize_prompt = PromptTemplate.from_template(
    "Summarize the following text: {context} using user's same language"
)
qa_prompt = PromptTemplate.from_template(
    "Answer questions based on this document: {context} \nQuestion: {question} using user's same language "
)
