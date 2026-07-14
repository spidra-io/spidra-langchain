from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_spidra import SpidraScrape

# Ensure you set the required environment variables:
# export SPIDRA_API_KEY="your-spidra-key"
# export OPENAI_API_KEY="your-openai-key"


def main() -> None:
    # 1. Initialize the Spidra Scrape tool
    # This tool is packaged as a standard LangChain BaseTool
    scrape_tool = SpidraScrape()
    tools = [scrape_tool]

    print("Spidra Tool Name:", scrape_tool.name)
    print("Spidra Tool Description:", scrape_tool.description)

    # 2. Setup the Prompt and LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant who helps users gather information from the web. "
                "Use the tools provided when requested.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # We use a tool-calling capable model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3. Create the agent and executor
    print("\nCreating the Tool-Calling Agent...")
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4. Run the agent
    user_query = "What is the headline story on example.com? Scrape it and let me know."
    print(f"\nRunning agent with query: '{user_query}'")

    response = agent_executor.invoke({"input": user_query})

    print("\n--- Final Agent Response ---")
    print(response["output"])
    print("----------------------------")


if __name__ == "__main__":
    main()
