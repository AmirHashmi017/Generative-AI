from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

loader= TextLoader('cricket.txt',encoding='utf-8')

docs= loader.load()
print(docs)
print(type(docs))
print(len(docs))
print(docs[0])

model=ChatOpenAI()
parser=StrOutputParser()
prompt= PromptTemplate(
    template='Write summary of this text {text}',
    input_variables=['text'])

chain= prompt | model | parser

result= chain.invoke({'text':docs[0].page_content})
print(f"Summarized Result: {result}")
