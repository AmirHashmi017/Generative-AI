from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="chat-completion"
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of Pakistan?")

print(result.content)