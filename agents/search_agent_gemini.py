import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = '''You are a large language AI assistant. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x:url]], where x is a number, url is citation url. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x:url]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3:https://google.com][citation:5:https://www.weatherapi.com/]. Other than code and specific names and citations, your answer must be written in the same language as the question.

You can use provided tools to help you answer the question.
'''

os.environ["TAVILY_API_KEY"] = "your_tavily_api_key"

tools = [TavilySearchResults(max_results=5)]
llm = ChatOpenAI(
    api_key="your_api_key",
    model="gpt-4",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "who is president of USA"}, config={})
print(result)
