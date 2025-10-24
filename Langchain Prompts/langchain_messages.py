from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model= ChatOpenAI(model="gpt-4")

messages=[
    SystemMessage(content="You are a helpful Assistant"),
    HumanMessage(content="Tell me about LangChain")
]

result= model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)