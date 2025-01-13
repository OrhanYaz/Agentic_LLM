from langchain.agents import AgentType, initialize_agent

import llm
import tools


def run_agent(query):
    # Initialize the agent with the tools and set the agent type
    agent = initialize_agent(
        tools=tools.tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm.llm,
        verbose=True
    )

    # Get the answer from the agent
    result = agent.run(query)
    return result


if __name__ == "__main__":
    # Example query to run through the agent
    query = "What is the policy on remote work?"
    answer = run_agent(query)
    print(answer)
