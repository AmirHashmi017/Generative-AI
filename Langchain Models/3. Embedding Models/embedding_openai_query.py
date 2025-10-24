# For Embedding Single Sentence
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

result= embedding.embed_query('Islamabad is the capital of Pakistan')

print(str(result))