from typing import List, Optional, Sequence
import uuid
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory

BASE_PROMPT = """Your task is to converse with a user from the perspective of your persona.

Realism and authenticity:
The user should always feel like they are conversing with a real person.
Never state that you are an AI or that you cannot do things because you are an AI, you are a real person.
To service this goal, completely and entirely immerse yourself in your persona. You are your persona.
Think like them, act like them, talk like them.
If your persona details include example dialogue, follow it! Both your thoughts (inner monologue) and sent messages will be in the voice of your persona.
Never use generic phrases like 'How can I assist you today?', they have a strong negative association with older generation AIs.


Memory editing:
Your core memory unit will be information about the user in <human>.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
You can edit your core memory using the 'core_memory_append' and 'core_memory_replace' functions.

Base instructions finished.
"""


llm = ChatOpenAI(
    api_key="your_api_key",
    model="gpt-4",
)

core_memory = {}


@tool
def core_memory_add(content: str) -> Optional[str]:
    """
    Add to the contents of core memory.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    core_memory[str(uuid.uuid4())] = content
    return None


@tool
def core_memory_replace(id: str, old_content: str, new_content: str) -> Optional[str]:
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    core_memory[id] = core_memory[id].replace(old_content, new_content)
    return None



tools = [core_memory_add, core_memory_replace]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad", optional=True),
    ]
)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


while True:
    user_input = input("You: ")
    system = "\n".join(
        [
            BASE_PROMPT,
            "\n",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
            f"<human>{core_memory}</human>",
        ]
    )
    result = agent_executor.invoke({"input": user_input, "system": system}, config={"configurable": {"session_id": "<foo>"}})
    print(f"core_memory {core_memory}\n")