from langchain.agents import Tool

import agents

tools = [
    Tool(
        name="HR Agent",
        func=agents.hr_agent.run,
        description="Handles HR queries. Answer questions related to HR policies, benefits, etc."
    ),
    Tool(
        name="Finance Agent",
        func=agents.finance_agent.run,
        description="Handles finance questions. Answer queries about accounting, budgeting, etc."
    ),
    Tool(
        name="General Agent",
        func=agents.general_agent.run,
        description="Answer general information queries."
    ),
    Tool(
        name="Summarization Agent",
        func=agents.summarize_agent.run,
        description="Summarizes text or documents."
    ),
    Tool(
        name="Document QA Agent",
        func=agents.qa_agent.run,
        description="Answers questions from documents based on context."
    )
]
