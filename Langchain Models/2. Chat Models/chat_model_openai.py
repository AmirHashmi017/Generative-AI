from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# If temprature is 0 means we get same response for same prompt everytime but if 
# we chnage that to 0.5 or 1 or 1.5 we will get diffferent response for same input and greater
# the temprature scre more distinguished response there will be

chat_model=ChatOpenAI(model='gpt-4',temperature=1.5,max_completion_tokens=10)

result= chat_model.invoke('Write 5 lines on cricket')
print(result)

# For printing exact output
print(result.content)