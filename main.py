from langchain.agents import AgentExecutor
from langchain.agents.agent import Agent

class RouterAgent(Agent):
    def route(self, query):
        if "HR" in query:
            return "HR Agent"
        elif "finance" in query:
            return "Finance Agent"
        elif "summarize" in query:
            return "Summarization Agent"
        elif "document" in query:
            return "Document QA Agent"
        else:
            return "General Agent"

router = RouterAgent(tools=tools)
agent_executor = AgentExecutor(agent=router, tools=tools)

def handle_query(query):
    return agent_executor.run(query)