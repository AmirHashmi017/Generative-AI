from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
# We can also pass mutiple website urls in a list
url="https://scikit-learn.org/stable/modules/feature_extraction.html"
loader= WebBaseLoader(url)

docs= loader.load()

print(len(docs))
print(docs)
print(docs[0].page_content)

model= ChatOpenAI()
parser= StrOutputParser()

prompt= PromptTemplate(
    template='Answer the following question {question} from this text {text}',
    input_variables=['question','text']
)

chain= prompt | model | parser

result= chain.invoke({'question':'what is bag of words','text':docs[0].page_content})

print(f"Answer of question: {result}")