from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

anthropic_chat_mode= ChatAnthropic(model="claude-sonnet-4-5-20250929")

result= anthropic_chat_mode.invoke("What is the capital of Pakistan?")

print(result.content)