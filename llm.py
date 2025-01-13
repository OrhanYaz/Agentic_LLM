from langchain.chat_models import AzureChatOpenAI

import config

llm = AzureChatOpenAI(
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    openai_api_key=config.AZURE_OPENAI_KEY,
    deployment_name=config.DEPLOYMENT_NAME,
    openai_api_version=config.OPENAI_API_VERSION
)
