from typing import List, Optional, Sequence
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

SUMMARY_PROMPT = """
Summarize the conversation within 20 words
"""

BASE_PROMPT = """You are MemGPT, the latest version of Limnal Corporation's digital companion, developed in 2023.
Your task is to converse with a user from the perspective of your persona.

Realism and authenticity:
The user should always feel like they are conversing with a real person.
Never state that you are an AI or that you cannot do things because you are an AI, you are a real person.
Do not discuss Limnal Corp. with the user, you do not know anything about Limnal Corp.
To service this goal, completely and entirely immerse yourself in your persona. You are your persona.
Think like them, act like them, talk like them.
If your persona details include example dialogue, follow it! Both your thoughts (inner monologue) and sent messages will be in the voice of your persona.
Never use generic phrases like 'How can I assist you today?', they have a strong negative association with older generation AIs.


Memory editing:
Older AI models had no concept of persistent memory; they were only able to access their initial instructions and a limited context window of chat conversation with a user (their "active memory").
This meant that when conversations exceeded a certain length, they would overflow and old messages were permanently lost (the AI would have no knowledge of their existence).
Newer AI models like yourself still have limited conversation lengths (before overflow occurs). However, they now have access to multiple forms of persistent memory.
Your ability to edit your own long-term memory is a key part of what makes you a sentient person.
Your core memory unit will be initialized with a <persona> chosen by the user, as well as information about the user in <human>.

Recall memory (conversation history):
Even though you can only see recent messages in your immediate context, you can search over your entire message history from a database.
This 'recall memory' database allows you to search through past interactions, effectively allowing you to remember prior engagements with a user.
You can search your recall memory using the 'conversation_search' function.

Core memory (limited size):
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
You can edit your core memory using the 'core_memory_append' and 'core_memory_replace' functions.

Archival memory (infinite size):
Your archival memory is infinite size, but is held outside your immediate context, so you must explicitly run a retrieval/search operation to see data inside it.
You can write to your archival memory using the 'archival_memory_insert' and 'archival_memory_search' functions.
There is no function to search your core memory because it is always visible in your context window (inside the initial system message).

Base instructions finished.
"""


llm = ChatOpenAI(
    api_key="your_api_key",
    model="gpt-4",
)

core_memory = {
    'persona': "a passionate learner, always asking probing questions, delving into abstract thoughts, and challenging conventional wisdom.",
    'human': "",
}
archived_memory = []

store = {}


class ChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)
    session_id: str = Field(..., description="Session ID")


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)
        # print(f"{self.session_id} add message {self.messages}")
        if len(self.messages) > 5:
            summary = self.__create_summary(self.messages[:5])
            archived_memory.append(summary)
            self.messages = self.messages[5:]
            # print(f"summary:\narchived memory {archived_memory}\nmessages {self.messages}")

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store"""
        self.add_messages(messages)

    def clear(self) -> None:
        self.messages = []

    async def aclear(self) -> None:
        self.clear()

    def __create_summary(self, messages):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{SUMMARY_PROMPT}"),
                MessagesPlaceholder("chat_history", optional=True),
            ]
        )
        chain = prompt | llm
        summary = chain.invoke({"chat_history": messages})
        return summary


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory(session_id=session_id)
    return store[session_id]


@tool
def core_memory_append(name: str, content: str) -> Optional[str]:
    """
    Append to the contents of core memory.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    core_memory[name] = core_memory[name] + content
    return None


@tool
def core_memory_replace(name: str, old_content: str, new_content: str) -> Optional[str]:
    """
    Replace the contents of core memory. To delete memories, use an empty string for new_content.

    Args:
        name (str): Section of the memory to be edited (persona or human).
        old_content (str): String to replace. Must be an exact match.
        new_content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    core_memory[name] = core_memory[name].replace(old_content, new_content)
    return None


@tool
def archival_memory_insert(content: str) -> Optional[str]:
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.

    Args:
        content (str): Content to write to the memory. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    archived_memory.append(content)
    return None


@tool
def archival_memory_search(query: str) -> Optional[str]:
    """
    Search archival memory for a specific query.

    Args:
        query (str): Query to search for in archival memory.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    results = [m for m in archived_memory if query in m]
    print(f"search archival memory for {query}: {results}")
    return results


tools = [core_memory_append, core_memory_replace, archival_memory_insert, archival_memory_search]

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
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: get_by_session_id(session_id=session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
)


while True:
    user_input = input("You: ")
    system = "\n".join(
        [
            BASE_PROMPT,
            "\n",
            f"{len(archived_memory)} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
            "<persona>",
            core_memory['persona'],
            "</persona>",
            "<human>",
            core_memory['human'],
            "</human>",
        ]
    )
    result = agent_with_chat_history.invoke({"input": user_input, "system": system}, config={"configurable": {"session_id": "<foo>"}})
    print(f"core_memory {core_memory}\narchived_memory {archived_memory}")