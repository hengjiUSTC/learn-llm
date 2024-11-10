# nanoAgents

## Descriptions

- search_agent.py, search_agent_gemini.py: Agent able to seach web when asked questions
- persionalized_agent.py, gemini_personalized_agent.py: Agent with personalized longterm memory. Know you better! Ref: [MemGpt](https://memgpt.ai)
- chatGPT_memory_agent.py, gemini_memory_agent.py: Agent to reproduce chatGpt persionalized memory system.

## How to Use

- install relavent dependencies.

- enter you openai key.

```
llm = ChatOpenAI(
    api_key="your_api_key",
    model="gpt-4"
)
```

- enter gemini api key

```
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
```


- run it! `python3 xxx.py`