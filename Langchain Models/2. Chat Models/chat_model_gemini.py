from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

gemini_model= ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result=gemini_model.invoke("What is the capital of Pakistan?")

print(result.content)