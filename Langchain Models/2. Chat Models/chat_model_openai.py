from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# temperature tells the creativity amount in answer that how ditinguish anser should be , where
# max_completion_tokens define the maximum number of words there should be in response
chat_model=ChatOpenAI(model='gpt-4',temperature=1.5,max_completion_tokens=10)

result= chat_model.invoke('Write 5 lines on cricket')
print(result)

# For printing exact output
print(result.content)